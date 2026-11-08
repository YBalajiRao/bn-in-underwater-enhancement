"""WWE-UIE limited GPU memory / speed pilot (TRAIN split only). NOT approved to run yet - see the approval history in the chat.

What it does : builds WWEUIE + the WWE composite loss (v2_tools/wwe_loss.py), draws a bounded subset (64-300) of the 3,463 TRAIN pairs,
               preloads ONLY that subset, then runs `warmup` untimed + `measured` timed training iterations (single pass, no epochs) and
               writes memory / timing / finite-value records to a NEW runs_v2/wwe_pilot_* directory.
Fixed        : batch 4, FP32 (no autocast, no GradScaler), seed 0, AdamW lr 2e-4 (constant) wd 1e-5, grad-clip 1.0, paired horizontal flip only.
Never        : touches any split except `train` (all other keys are deleted from train_v2.MANIFESTS before any file access; verify_manifests()
               is NOT called - only the train manifest is hashed), validates / evaluates / runs inference on anything, saves a checkpoint or
               loads one, runs the gate, uses AMP, loops without a bound, or writes outside its own new directory.
Checks       : EVERY iteration, outside the timed region: all loss terms, the network output, every parameter gradient (none may be None),
               the pre-clip gradient norm, and (after the step) all parameters are finite. First failure -> status FAILED, loop stops.
Timing       : step = forward + loss + backward + clip + optimizer step (CUDA-synchronised at both ends); the finite-value audit and the data
               fetch (+ host-to-device copy) are timed separately and excluded from `step_s`.
Memory       : torch.cuda.max_memory_allocated / max_memory_reserved (reset at the start, again at the start of the measured phase).
Usage (GPU; only after approval): python v2_tools/wwe_pilot.py --run_name wwe_pilot_lr2e-4_s0_v1 [--n_images 200 --warmup_iters 10 --measured_iters 40]"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.wwe_uie import EXPECTED_PARAMS, WWE_COMMIT, WWEUIE  # noqa: E402
from v2_tools import wwe_config as cfg  # noqa: E402
from v2_tools.wwe_loss import WWEUIELoss  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

BS, SEED, LR, WD, CLIP = 4, 0, 2e-4, 1e-5, 1.0
N_MIN, N_MAX, MAX_ITERS = 64, 300, 100
NAME_RE = re.compile(r"^wwe_pilot_[A-Za-z0-9_.-]{1,60}$")
LIMITATIONS = [
    "Estimates extrapolate from a few dozen iterations of one seed on a bounded subset; they are not measurements of a full epoch.",
    "Excluded from the estimate: the per-epoch Val-A evaluation (not run here), checkpoint writing, process start-up and disk/OS effects.",
    "The real run uses 2 DataLoader workers; here data fetch is single-process and reported separately (step_s excludes it).",
    "Peak memory covers only the measured subset and bs 4 at 256x256; fragmentation over 86,600 iterations may raise reserved memory.",
    "Shared desktop GPU: other processes can perturb timing; clocks/thermals are not controlled.",
]


class PilotFailure(RuntimeError):
    pass


# ----------------------------------------------------------------------------- validation helpers (pure, CPU-testable)
def validate_config(n_images, warmup, measured, run_name):
    if not isinstance(run_name, str) or not NAME_RE.match(run_name):
        raise ValueError(f"run_name must match {NAME_RE.pattern}")
    if not (N_MIN <= n_images <= N_MAX):
        raise ValueError(f"n_images must be in [{N_MIN}, {N_MAX}], got {n_images}")
    if n_images % BS:
        raise ValueError(f"n_images must be a multiple of the batch size {BS} (no partial batches), got {n_images}")
    if warmup < 1 or measured < 5:
        raise ValueError("need warmup_iters >= 1 and measured_iters >= 5")
    avail = n_images // BS
    if warmup + measured > avail:
        raise ValueError(f"warmup+measured = {warmup + measured} exceeds the single-pass limit {avail} (= n_images/{BS}); the pilot never repeats data")
    if warmup + measured > MAX_ITERS:
        raise ValueError(f"warmup+measured must be <= {MAX_ITERS}")
    return avail


def make_out_dir(root, run_name):
    d = Path(root) / "runs_v2" / run_name
    d.mkdir(parents=True, exist_ok=False)  # FileExistsError if it exists (even if empty): never overwrite, never reuse
    return d


def enforce_train_only(mod):
    """Delete every manifest key except `train` BEFORE any file is hashed or opened; returns the single remaining (path, key, prefix)."""
    for k in [k for k in mod.MANIFESTS if k != "train"]:
        del mod.MANIFESTS[k]
    if list(mod.MANIFESTS) != ["train"]:
        raise AssertionError(f"manifest allowlist failed: {list(mod.MANIFESTS)}")
    p = mod.MANIFESTS["train"][0]
    if "uieb" in str(p).lower() or "val" in Path(p).name.lower():
        raise AssertionError(f"train manifest path looks restricted: {p}")
    return mod.MANIFESTS["train"]


def finite_report(out, ld, model, gnorm):
    """-> list of problems (empty == all finite)."""
    bad = []
    if not torch.isfinite(out).all():
        bad.append("output")
    for k, v in ld.items():
        if not torch.isfinite(v).all():
            bad.append(f"loss:{k}")
    if not math.isfinite(float(gnorm)):
        bad.append("grad_norm")
    for n, p in model.named_parameters():
        if p.grad is None:
            bad.append(f"grad_none:{n}")
        elif not torch.isfinite(p.grad).all():
            bad.append(f"grad:{n}")
    return bad


def params_finite(model):
    return all(torch.isfinite(p).all() for p in model.parameters())


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _mb(x):
    return None if x is None else x / 2**20


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}


# ----------------------------------------------------------------------------- the pilot loop (data injected -> CPU-testable)
def run_pilot(X, Y, device, out_dir, warmup, measured, train_n, meta=None, vgg_features=None):
    """X, Y: uint8 [N,3,256,256] subset tensors (already validated). Writes pilot_iterations.jsonl and pilot_result.json into out_dir."""
    out_dir = Path(out_dir)
    n = len(X)
    validate_config(n, warmup, measured, "wwe_pilot_inline")
    tv.seed_everything(SEED)
    cuda = device.type == "cuda"
    model = WWEUIE().to(device).train()
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS
    crit = WWEUIELoss(device, vgg_features=vgg_features)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    ds = tv.PairedAugDataset(X, Y, SEED, jitter=False)
    g = torch.Generator().manual_seed(SEED)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=BS, sampler=torch.utils.data.RandomSampler(ds, generator=g), num_workers=0, drop_last=True))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    mem = {"setup_allocated_mb": _mb(torch.cuda.memory_allocated()) if cuda else None}
    recs, steps, fetches, audits = [], [], [], []
    status, fail = "OK", None
    total = warmup + measured
    log = open(out_dir / "pilot_iterations.jsonl", "w")
    try:
        for it in range(total):
            phase = "warmup" if it < warmup else "measured"
            if it == warmup and cuda:  # steady-state peak: separate warm-up (cudnn autotune, allocator growth) from the measured phase
                mem.update(warmup_peak_allocated_mb=_mb(torch.cuda.max_memory_allocated()), warmup_peak_reserved_mb=_mb(torch.cuda.max_memory_reserved()))
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            x, y = next(loader)
            x, y = x.to(device), y.to(device)
            _sync(device)
            t_fetch = time.perf_counter() - t0

            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            ld = tv.compute_loss(crit, out, y)
            ld["total"].backward()
            _sync(device)
            t_a = time.perf_counter() - t0

            t0 = time.perf_counter()  # audit BEFORE clip/step: a bad gradient is never applied
            gn_pre = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # norm only (no clipping), for the audit
            bad = finite_report(out["enhanced"], ld, model, gn_pre)
            _sync(device)
            t_audit = time.perf_counter() - t0
            if bad:
                raise PilotFailure(f"non-finite at iteration {it} ({phase}): {bad[:6]}")

            t0 = time.perf_counter()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            _sync(device)
            t_b = time.perf_counter() - t0
            t0 = time.perf_counter()
            if not params_finite(model):
                raise PilotFailure(f"non-finite parameters after the step of iteration {it} ({phase})")
            t_audit += time.perf_counter() - t0
            step_s = t_a + t_b
            rec = {"iter": it, "phase": phase, "step_s": step_s, "fetch_s": t_fetch, "audit_s": t_audit, "grad_norm_preclip": float(gn_pre),
                   "loss": {k: float(v) for k, v in ld.items()}, "all_finite": True}
            log.write(json.dumps(rec) + "\n"); log.flush()
            recs.append(rec)
            if phase == "measured":
                steps.append(step_s); fetches.append(t_fetch); audits.append(t_audit)
    except PilotFailure as e:
        status, fail = "FAILED", str(e)
    finally:
        log.close()
    if cuda:
        mem.update(measured_peak_allocated_mb=_mb(torch.cuda.max_memory_allocated()), measured_peak_reserved_mb=_mb(torch.cuda.max_memory_reserved()))
        mem.update(overall_peak_allocated_mb=max(mem.get("warmup_peak_allocated_mb") or 0, mem["measured_peak_allocated_mb"]),
                   overall_peak_reserved_mb=max(mem.get("warmup_peak_reserved_mb") or 0, mem["measured_peak_reserved_mb"]),
                   total_device_mb=_mb(torch.cuda.get_device_properties(device).total_memory))
    result = {"status": status, "failure": fail, "iterations_completed": len(recs), "iterations_planned": total,
              "config": {"batch_size": BS, "precision": "fp32 (no autocast, no GradScaler)", "seed": SEED, "optimizer": f"AdamW lr {LR} (constant) wd {WD}", "grad_clip": CLIP,
                         "augmentation": "paired horizontal flip only", "n_images": n, "warmup_iters": warmup, "measured_iters": measured,
                         "loss_weights": cfg.LOSS_WEIGHTS, "density_k": cfg.DENSITY_K, "single_pass_no_repeat": True},
              "wwe": {"commit": WWE_COMMIT, "params": EXPECTED_PARAMS}, "train_n_verified": train_n, "memory_mb": mem,
              "all_iterations_finite": status == "OK", "env": tv.env_info(), "meta": meta or {}}
    if status == "OK":
        ipe = math.ceil(train_n / BS)
        sm, fm = float(np.mean(steps)), float(np.mean(fetches))
        result["timing_s"] = {"step": _stats(steps), "data_fetch_incl_h2d": _stats(fetches), "audit_excluded": _stats(audits)}
        result["ESTIMATE_not_measured"] = {
            "iterations_per_epoch": ipe, "basis": "mean measured step_s x ceil(train_n/4); train_n verified from the manifest",
            "epoch_s_step_only": ipe * sm, "epoch_s_step_plus_single_process_fetch": ipe * (sm + fm),
            "epoch_s_if_p90_step": ipe * float(np.percentile(steps, 90)),
            "hours_100_epochs_step_only": 100 * ipe * sm / 3600, "hours_100_epochs_with_fetch": 100 * ipe * (sm + fm) / 3600,
            "hours_100_epochs_if_p90_step": 100 * ipe * float(np.percentile(steps, 90)) / 3600,
            "limitations": LIMITATIONS}
    json.dump(result, open(out_dir / "pilot_result.json", "w"), indent=1)
    return result


# ----------------------------------------------------------------------------- CLI (GPU only)
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run_name", required=True, help="new directory runs_v2/<run_name>; must start with wwe_pilot_ and must not exist")
    ap.add_argument("--n_images", type=int, default=200, help=f"train subset size, {N_MIN}-{N_MAX}, multiple of {BS}")
    ap.add_argument("--warmup_iters", type=int, default=10)
    ap.add_argument("--measured_iters", type=int, default=40)
    a = ap.parse_args(argv)
    try:
        validate_config(a.n_images, a.warmup_iters, a.measured_iters, a.run_name)
    except ValueError as e:
        sys.exit(f"STOP: {e}")
    if (ROOT / "runs_v2" / a.run_name).exists():
        sys.exit(f"STOP: runs_v2/{a.run_name} already exists; refusing to reuse or overwrite")
    if not torch.cuda.is_available():
        sys.exit("STOP: CUDA not available (the pilot is GPU-only; CPU use is limited to the unit tests)")
    device = torch.device("cuda")

    path, key, prefix = enforce_train_only(tv)  # before ANY manifest hash or open
    h = tv.sha256(path)
    if not h.startswith(prefix):
        sys.exit(f"STOP: train manifest hash {h[:8]} != expected {prefix}")
    pair_loader = checked_loader(tv.load_pairs, cfg.EXPECTED_N)  # validates the FULL train list (count, basenames, duplicates, restricted paths)
    full = pair_loader("train")
    idx = np.sort(np.random.RandomState(SEED).choice(len(full), a.n_images, replace=False))
    sub = [full[i] for i in idx]
    out_dir = make_out_dir(ROOT, a.run_name)
    X, Y = tv.preload(sub)  # only the subset's images are read
    meta = {"train_manifest_sha256": h, "subset_indices": idx.tolist(), "command": " ".join(sys.argv), "splits_used": ["train"], "pair_validation": "passed (full train list)"}
    r = run_pilot(X, Y, device, out_dir, a.warmup_iters, a.measured_iters, len(full), meta=meta)
    print(json.dumps({k: r[k] for k in ("status", "failure", "iterations_completed", "memory_mb", "timing_s", "ESTIMATE_not_measured") if k in r}, indent=1))
    sys.exit(0 if r["status"] == "OK" else 1)


if __name__ == "__main__":
    main()
