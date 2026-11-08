"""XFMS validation gate for ONE finished run (xf_attrunk_r3_fp32_s1 or _s2). Sibling of v2_tools/xf_gate.py (NOT modified; its clip-log check and finite helper are imported); it re-scores on the GPU and reads Val-A.
Usage (NOT approved to run): python v2_tools/xfms_gate.py --run xf_attrunk_r3_fp32_s1
Twelve checks (the eleven XF checks with the seed-dependent values, plus the pairing check 12):
  1 files exist   2 model == at_trunk, params == 1,952,768   3 recorded args == the expected XF argument set of THIS seed exactly (recipe = train_v2.RECIPES["R3"] with ONLY amp -> False and eta_min -> 2e-8), run
  identity (seed, run name, recipe, fp32, no resume)   4 100 contiguous epochs   5 no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler
  6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected prefixes, AND strict pair validation of the train and val_A lists   7 best epoch == argmax history   8 checkpoint SHA-256 == result.json
  9 independent Val-A re-score (ATMarineTrunk, strict load) == logged (1e-4), and the checkpoint's stored epoch == the best epoch
  10 xf_loss_config.json equals the record re-derived from the code for this seed, its commit matches result.json, and original_repo/marine_net_model.py equals the PINNED SHA-256 and the hash recorded at launch
  11 xf_clip_log.jsonl: 100 lines, 86,600 steps, per-epoch steps / mean / max equal history.jsonl, statistics internally consistent
  12 pairing: the recorded args differ from the existing XF seed-0 run's (runs_v2/xf_attrunk_r3_fp32_s0/result.json, which holds the train and val_A manifests only) ONLY in {seed, run_name}, and that XF seed-0
     record itself describes XF seed 0; the XF seed-0 result.json must also have the PINNED SHA-256 (xfms_config.XF_S0_RESULT_SHA256), read once so the hashed bytes are the parsed bytes
The gate prints the run metrics (best, last-10 mean, SSIM) for the paired comparison; the pre-registered rule (xfms_config.paired_summary) needs all six runs and is NOT evaluated here.
Checks 2-5, 7, 10, 11 and 12 are pure functions (`static_checks`, `check_loss_config`, `check_clip`, `check_pairing`), CPU-testable with synthetic dicts. The AMP stage4a_at_trunk_s* runs are never opened.
val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools import xf_gate as xfg  # noqa: E402
from v2_tools import xfms_config as xm  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = 1e-4
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xm.LOSS_CONFIG_FILE, xm.CLIP_LOG_FILE]
CHECK_NAMES = ["files exist", "model/params", "recorded args == expected XF args of this seed (R3 recipe with ONLY amp False and eta_min 2e-8), fp32, run identity", "100 contiguous epochs",
               "no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler", "manifest hashes as expected (exactly train + val_A), pairs valid", "best epoch == argmax history",
               "checkpoint SHA-256 == result.json", "val-A re-score == logged, checkpoint epoch == best epoch", "xf_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded",
               "xf_clip_log.jsonl: 100 epochs, 86,600 steps, consistent with history.jsonl", "args pair with the existing XF seed-0 run (result.json SHA-256 pinned): differ only in seed / run_name"]
assert len(CHECK_NAMES) == 12


def result_path(d):
    """Where the gate writes its JSON verdict: xfms_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / xm.GATE_JSON_FILE
    return p if not p.exists() else Path(d) / f"xfms_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def _finite(*vals):
    return all(v is not None and not isinstance(v, bool) and np.isfinite(v) for v in vals)


def static_checks(r, hist, recipe, r3_recipe, seed):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access. recipe = xm.recipe_args(...), r3_recipe = train_v2.RECIPES['R3']."""
    xm.validate_seed(seed)
    out = []
    out.append((CHECK_NAMES[1], r.get("model") == xm.MODEL and r.get("params") == xm.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    diff_r3 = set(xm.recipe_diff(r3_recipe, recipe))
    want = xm.expected_args(recipe, seed)
    bad = {k: (args.get(k), want.get(k)) for k in sorted(set(args) | set(want)) if args.get(k) != want.get(k)}
    meta = xm.check_result_metadata(r, seed)
    ok = not bad and not meta and diff_r3 == set(xf.R3_DEVIATIONS)
    info = xm.RECIPE_NAME if ok else (bad or meta or f"recipe differs from R3 in {sorted(diff_r3)}, expected {sorted(xf.R3_DEVIATIONS)}")
    out.append((CHECK_NAMES[2], ok, info))
    out.append((CHECK_NAMES[3], [h.get("epoch") for h in hist] == list(range(xm.EPOCHS)), len(hist)))
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


def check_loss_config(lc, r, recipe, current_original_sha256, seed):
    """Check 10 -> (passed, info). lc = parsed xf_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    xm.validate_seed(seed)
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, "git_commit_full missing from xf_loss_config.json"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    problems = xm.check_runtime(lc.get("runtime"))
    if problems:
        return False, f"runtime record invalid: {problems}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = xm.loss_config(seed, full, recipe, sha, lc["runtime"])
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"xf_loss_config.json differs from the expected record for seed {seed} in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != xm.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    rt = lc["runtime"]
    return True, f"commit {full[:8]}, seed {seed}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}, cudnn_allow_tf32 {rt['cudnn_allow_tf32']}"


def check_clip(records, hist):
    """Check 11 -> (passed, info). The XF check, unchanged."""
    return xfg.check_clip(records, hist)


def check_pairing(args, xf0, xf0_sha256, expected_sha256=xm.XF_S0_RESULT_SHA256):
    """Check 12 -> (passed, info). args = result.json['args'] of the new run, xf0 = the parsed result.json of the existing XF seed-0 run (or None). The XF seed-0 record must itself describe XF seed 0 (identity,
    fp32, eta_min, manifest scope), the file must have the pinned SHA-256 (xf0_sha256 = the hash of the exact bytes that were parsed; None = unreadable -> fail), and the two argument sets must differ in
    EXACTLY {seed, run_name}. Pure."""
    if not isinstance(xf0, dict) or not isinstance(args, dict):
        return False, "the XF seed-0 result.json (or the new run's args) is missing or not a JSON object"
    if xf0_sha256 != expected_sha256:
        return False, f"the XF seed-0 result.json SHA-256 {str(xf0_sha256)[:12]}... != the pinned {expected_sha256[:12]}...: the reference changed or could not be verified"
    problems = xf.check_result_metadata(xf0)
    if problems:
        return False, f"the XF seed-0 record does not describe XF seed 0: {problems}"
    xa = xf0.get("args") or {}
    diff = {k for k in set(args) | set(xa) if args.get(k) != xa.get(k)}
    if diff != xm.EXPECTED_DIFF_VS_XF_S0:
        return False, f"args differ from XF seed 0 in {sorted(diff)}, expected exactly {sorted(xm.EXPECTED_DIFF_VS_XF_S0)}"
    return True, f"differs from {xm.EXISTING_XF_RUN} only in {sorted(diff)}"


def main():
    import torch
    import train_v2 as tv
    from models.at_marine import ATMarineTrunk
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    seed = xm.seed_of_run(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xm.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xm.ALLOWED_MANIFESTS)
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
                json.dump({"run": a.run, "seed": seed, "pass": False, "checks": checks}, f, indent=1)
        sys.exit(1)
    r = json.load(open(d / "result.json"))
    hist = [json.loads(line) for line in open(d / "history.jsonl")]
    recipe = xm.recipe_args(tv.RECIPES)
    st = static_checks(r, hist, recipe, tv.RECIPES[xm.BASELINE_RECIPE], seed)
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), xm.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec(CHECK_NAMES[5], set(mh) == set(xm.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in xm.EXPECTED_HASH_PREFIXES.items()) and pairs_ok, f"{sorted(mh)}; {pinfo}")
    nm, c, i = st[4]
    rec(nm, c, i)
    rec(CHECK_NAMES[7], all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    m = ATMarineTrunk().to(dev)
    ck = torch.load(d / "best_valA.pth", map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec(CHECK_NAMES[8], s["n"] == 166 and dp < TOL and ds < TOL and ck.get("epoch") == r["best_valA_epoch"],
        f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e}) checkpoint epoch {ck.get('epoch')} vs {r['best_valA_epoch']}")
    lc = json.load(open(d / xm.LOSS_CONFIG_FILE))
    try:
        cur = xm.sha256_file(ROOT / xm.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, cur, seed)
    rec(CHECK_NAMES[9], c10, i10)
    try:
        records = [json.loads(line) for line in open(d / xm.CLIP_LOG_FILE)]
        c11, i11 = check_clip(records, hist)
    except ValueError as e:
        c11, i11 = False, f"xf_clip_log.jsonl is not valid JSON lines ({e})"
    rec(CHECK_NAMES[10], c11, i11)
    try:
        raw = (ROOT / "runs_v2" / xm.EXISTING_XF_RUN / "result.json").read_bytes()  # read ONCE: the hashed bytes are the parsed bytes
        xf0_sha, xf0 = hashlib.sha256(raw).hexdigest(), json.loads(raw)
    except (OSError, ValueError):
        xf0_sha, xf0 = None, None
    c12, i12 = check_pairing(r.get("args"), xf0, xf0_sha)
    rec(CHECK_NAMES[11], c12, i12)
    try:
        mt = xm.metrics_from_history(hist)
        print(f"RUN METRICS (exploratory Val-A): seed {seed} best {mt['best_psnr']:.4f} dB at epoch {mt['best_epoch']}, last-10 mean {mt['last10_psnr']:.4f}, SSIM at best {mt['best_ssim']:.4f}, "
              f"last-10 mean SSIM {mt['last10_ssim']:.5f}; the paired rule needs all six runs and is not evaluated here", flush=True)
    except ValueError as e:
        print(f"RUN METRICS unavailable: {e}", flush=True)
    with open(result_path(d), "x") as f:
        json.dump({"run": a.run, "seed": seed, "pass": ok, "checks": checks, "rescore": s}, f, indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
