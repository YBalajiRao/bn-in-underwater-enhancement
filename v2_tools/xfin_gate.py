"""XFIN validation gate for the ONE finished run xfin_attrunk_in_r3_fp32_s0 (adapted from v2_tools/xf_gate.py, which is NOT modified; it re-scores on the GPU and reads Val-A).
Usage (NOT approved to run): python v2_tools/xfin_gate.py --run xfin_attrunk_in_r3_fp32_s0
Twelve checks (differences from the XF gate marked *):
  1 files exist (incl. xf_clip_log.jsonl)   2 model == at_trunk_in, params == 1,952,768*   3 recorded args == the expected XFIN argument set exactly (the XF recipe values, recipe name / model / run name of THIS run),
  run identity, fp32, no resume*   4 100 contiguous epochs   5 no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler (scaler_scale null)
  6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected prefixes, AND strict pair validation of the train and val_A lists   7 best epoch == argmax history
  8 checkpoint SHA-256 == result.json   9 independent Val-A re-score (ATMarineTrunkIN, loaded STRICTLY) == logged (1e-4), and the checkpoint's stored epoch == the best epoch*
  10 xfin_loss_config.json equals the record re-derived from the code (runtime flags well typed), its commit matches result.json, and original_repo/marine_net_model.py equals the PINNED SHA-256 and the hash recorded at launch
  11 xf_clip_log.jsonl: 100 lines, 86,600 steps, per-epoch steps / mean / max equal history.jsonl   12* normalisation structure: the built model has exactly 40 InstanceNorm2d and 0 BatchNorm2d layers, and best_valA.pth /
  last.pth hold NO running_mean / running_var / num_batches_tracked keys
The gate prints the run's metrics and the pre-registered reading against XF seed 0 (documentation numbers only; XF's result.json is never opened). It does NOT touch runs_v2/stage4a_at_trunk_s0.
Checks 2-5, 7, 10, 11 and 12 are pure functions, CPU-testable with synthetic dicts. val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import xfin_config as xi  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = 1e-4
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xi.LOSS_CONFIG_FILE, xi.CLIP_LOG_FILE]
STAT_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
CHECK_NAMES = ["files exist", "model/params", "recorded args == expected XFIN args (XF recipe values), fp32, run identity", "100 contiguous epochs",
               "no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler", "manifest hashes as expected (exactly train + val_A), pairs valid", "best epoch == argmax history",
               "checkpoint SHA-256 == result.json", "val-A re-score == logged, checkpoint epoch == best epoch", "xfin_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded",
               "xf_clip_log.jsonl: 100 epochs, 86,600 steps, consistent with history.jsonl", "normalisation structure: 40 InstanceNorm2d / 0 BatchNorm2d, no running statistics in the checkpoints"]


def result_path(d):
    """xfin_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / "xfin_gate.json"
    return p if not p.exists() else Path(d) / f"xfin_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def _finite(*vals):
    return all(v is not None and not isinstance(v, bool) and np.isfinite(v) for v in vals)


def static_checks(r, hist, recipe, xf_recipe):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access. recipe = xfin.recipe_args(...), xf_recipe = xf_config.recipe_args(...) (must be identical)."""
    out = []
    out.append((CHECK_NAMES[1], r.get("model") == xi.MODEL and r.get("params") == xi.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    want = xi.expected_args(recipe)
    bad = {k: (args.get(k), want.get(k)) for k in sorted(set(args) | set(want)) if args.get(k) != want.get(k)}
    meta = xi.check_result_metadata(r)
    ok = not bad and not meta and recipe == xf_recipe
    info = xi.RECIPE_NAME if ok else (bad or meta or "the XFIN recipe values differ from the XF recipe")
    out.append((CHECK_NAMES[2], ok, info))
    out.append((CHECK_NAMES[3], [h.get("epoch") for h in hist] == list(range(xi.EPOCHS)), len(hist)))
    try:
        fin = all(h["train"]["nonfinite"] == 0 and h["train"]["inf_grad_steps"] == 0 and h.get("scaler_scale") is None
                  and _finite(h["train"]["total"], h["train"]["gnorm_mean"], h["train"]["gnorm_max"], h["val_A"]["psnr_mean"], h["val_A"]["ssim_mean"]) for h in hist)
        info = f"non-finite loss steps {sum(h['train']['nonfinite'] for h in hist)}, non-finite-gradient steps {sum(h['train']['inf_grad_steps'] for h in hist)}"
    except (KeyError, TypeError) as e:
        fin, info = False, f"history record incomplete ({type(e).__name__}: {e})"
    out.append((CHECK_NAMES[4], fin, info))
    try:
        best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
        ok = best["epoch"] == r.get("best_valA_epoch") and abs(best["val_A"]["psnr_mean"] - r["best_valA"]["psnr_mean"]) < 1e-9
        info = f"epoch {best['epoch']}"
    except (KeyError, TypeError, ValueError) as e:
        ok, info = False, f"cannot determine the best epoch ({type(e).__name__})"
    out.append((CHECK_NAMES[6], ok, info))
    return out


def check_loss_config(lc, r, recipe, current_original_sha256):
    """Check 10 -> (passed, info). lc = parsed xfin_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, f"git_commit_full missing from {xi.LOSS_CONFIG_FILE}"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    problems = xi.check_runtime(lc.get("runtime"))
    if problems:
        return False, f"runtime record invalid: {problems}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = xi.loss_config(full, recipe, sha, lc["runtime"])
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"{xi.LOSS_CONFIG_FILE} differs from the expected record in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != xi.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    rt = lc["runtime"]
    return True, f"commit {full[:8]}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}, cudnn_allow_tf32 {rt['cudnn_allow_tf32']}"


def check_clip(records, hist):
    """Check 11 -> (passed, info). Pure wrapper around the XF clip-log check."""
    problems = xi.check_clip_log(records, hist)
    if problems:
        return False, "; ".join(problems[:4]) + (f" (+{len(problems) - 4} more)" if len(problems) > 4 else "")
    total = sum(r["steps"] for r in records)
    return True, f"{len(records)} lines, {total} steps, mean clipped fraction {float(np.mean([r['frac_clipped'] for r in records])):.4f}"


def state_problems(sd):
    """Problems (list) in a state dict of the IN model: no running-statistics key may exist (BatchNorm buffers would mean the wrong model class). Pure."""
    bad = sorted(k for k in sd if k.endswith(STAT_SUFFIXES))
    return [f"{len(bad)} running-statistics keys, e.g. {bad[:2]}"] if bad else []


def structure_check(model, best_ck, last_ck):
    """Check 12 -> (passed, info). model = the built ATMarineTrunkIN, best_ck / last_ck = loaded checkpoint dicts. Pure given those objects."""
    import torch.nn as nn
    mods = list(model.modules())
    n_in, n_bn = sum(isinstance(m, nn.InstanceNorm2d) for m in mods), sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in mods)
    probs = []
    if (n_in, n_bn) != (xi.N_NORM_LAYERS, 0):
        probs.append(f"model has {n_in} InstanceNorm / {n_bn} BatchNorm layers")
    if sum(p.numel() for p in model.parameters()) != xi.EXPECTED_PARAMS:
        probs.append("parameter count != 1,952,768")
    probs += [f"best_valA.pth: {p}" for p in state_problems(best_ck["model_state_dict"])]
    if last_ck.get("epoch") != xi.EPOCHS - 1:
        probs.append(f"last.pth epoch {last_ck.get('epoch')} != {xi.EPOCHS - 1}")
    probs += [f"last.pth: {p}" for p in state_problems(last_ck["model"])]
    return (not probs), (f"{n_in} InstanceNorm / {n_bn} BatchNorm layers; best epoch {best_ck.get('epoch')}; last epoch {last_ck.get('epoch')}" if not probs else "; ".join(probs)[:400])


def main():
    import torch
    import train_v2 as tv
    from models.at_trunk_in import ATMarineTrunkIN
    from v2_tools import xf_config as xf
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    xi.validate_run_name(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xi.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xi.ALLOWED_MANIFESTS)
    d = ROOT / "runs_v2" / a.run
    checks, ok = {}, True

    def rec(name, cond, info=""):
        nonlocal ok
        checks[name] = {"pass": bool(cond), "info": str(info)}
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)

    present = all((d / f).exists() for f in FILES)
    rec(CHECK_NAMES[0], present)
    if not present:
        if d.is_dir():
            with open(result_path(d), "x") as f:
                json.dump({"pass": False, "checks": checks}, f, indent=1)
        sys.exit(1)
    r = json.load(open(d / "result.json"))
    hist = [json.loads(line) for line in open(d / "history.jsonl")]
    recipe = xi.recipe_args(tv.RECIPES)
    st = static_checks(r, hist, recipe, xf.recipe_args(tv.RECIPES))
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), xi.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec(CHECK_NAMES[5], set(mh) == set(xi.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in xi.EXPECTED_HASH_PREFIXES.items()) and pairs_ok, f"{sorted(mh)}; {pinfo}")
    nm, c, i = st[4]
    rec(nm, c, i)
    rec(CHECK_NAMES[7], all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    m = ATMarineTrunkIN().to(dev)
    ck = torch.load(d / "best_valA.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)  # strict: a BatchNorm checkpoint (extra buffers) would be refused
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec(CHECK_NAMES[8], s["n"] == 166 and dp < TOL and ds < TOL and ck.get("epoch") == r["best_valA_epoch"],
        f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e}) checkpoint epoch {ck.get('epoch')} vs {r['best_valA_epoch']}")
    lc = json.load(open(d / xi.LOSS_CONFIG_FILE))
    try:
        cur = xi.sha256_file(ROOT / xi.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, cur)
    rec(CHECK_NAMES[9], c10, i10)
    try:
        records = [json.loads(line) for line in open(d / xi.CLIP_LOG_FILE)]
        c11, i11 = check_clip(records, hist)
    except ValueError as e:
        c11, i11 = False, f"{xi.CLIP_LOG_FILE} is not valid JSON lines ({e})"
    rec(CHECK_NAMES[10], c11, i11)
    last = torch.load(d / "last.pth", map_location="cpu", weights_only=False)
    c12, i12 = structure_check(ATMarineTrunkIN(), ck, last)
    rec(CHECK_NAMES[11], c12, i12)
    best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
    nonfinite = sum(h["train"]["nonfinite"] + h["train"]["inf_grad_steps"] for h in hist)
    last10 = float(np.mean([h["val_A"]["psnr_mean"] for h in hist[-10:]]))
    reading = xi.decision(best["val_A"]["psnr_mean"], last10, best["val_A"]["ssim_mean"], nonfinite)
    print(f"RUN METRICS (exploratory Val-A): best {best['val_A']['psnr_mean']:.4f} dB at epoch {best['epoch']}, last-10 mean {last10:.4f}, SSIM at best {best['val_A']['ssim_mean']:.4f}; "
          f"pre-registered reading vs XF seed 0 (practical thresholds, one seed, not a significance test): {reading}", flush=True)
    with open(result_path(d), "x") as f:
        json.dump({"run": a.run, "pass": ok, "checks": checks, "rescore": s, "run_metrics": {"best_psnr": best["val_A"]["psnr_mean"], "best_epoch": best["epoch"], "last10_mean_psnr": last10,
                                                                                         "ssim_at_best": best["val_A"]["ssim_mean"], "nonfinite_steps": nonfinite}, "reading_vs_xf_seed0": reading}, f, indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
