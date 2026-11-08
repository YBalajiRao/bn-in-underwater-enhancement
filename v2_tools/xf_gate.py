"""XF validation gate for the ONE finished run xf_attrunk_r3_fp32_s0 (adapted from v2_tools/xr_gate.py; it re-scores on the GPU and reads Val-A).
Usage (NOT approved to run): python v2_tools/xf_gate.py --run xf_attrunk_r3_fp32_s0
Eleven checks (differences from the XR gate marked *):
  1 files exist (incl. xf_clip_log.jsonl*)   2 model == at_trunk, params == 1,952,768*   3 recorded args == the expected XF argument set exactly (recipe = train_v2.RECIPES["R3"] with ONLY amp -> False and
  eta_min -> 2e-8), run identity, fp32, no resume*   4 100 contiguous epochs   5 no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler (scaler_scale null)*
  6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected prefixes (no val, val_clean or val_B entry), AND strict pair validation of the train and val_A lists
  7 best epoch == argmax history   8 checkpoint SHA-256 == result.json   9 independent Val-A re-score (ATMarineTrunk) == logged (1e-4), and the checkpoint's stored epoch == the best epoch*
  10 xf_loss_config.json equals the record re-derived from the code (runtime flags well typed), its commit matches result.json, and original_repo/marine_net_model.py is readable, equals the PINNED
  SHA-256 and the hash recorded at launch (fail closed if it cannot be verified)*   11 xf_clip_log.jsonl: 100 lines, 86,600 steps, per-epoch steps / mean / max equal history.jsonl, statistics internally consistent*
This gate NEVER opens runs_v2/stage4a_at_trunk_s0/result.json (legacy manifest-hash entry of a restricted split): the recipe is compared with train_v2.RECIPES["R3"].
Checks 2-5, 7, 10 and 11 are pure functions (`static_checks`, `check_loss_config`, `check_clip`), CPU-testable with synthetic dicts.
val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = 1e-4
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xf.LOSS_CONFIG_FILE, xf.CLIP_LOG_FILE]


def result_path(d):
    """Where the gate writes its JSON verdict: xf_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / "xf_gate.json"
    return p if not p.exists() else Path(d) / f"xf_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def _finite(*vals):
    return all(v is not None and not isinstance(v, bool) and np.isfinite(v) for v in vals)


def static_checks(r, hist, recipe, r3_recipe):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access. recipe = xf.recipe_args(...), r3_recipe = train_v2.RECIPES['R3']."""
    out = []
    out.append(("model/params", r.get("model") == xf.MODEL and r.get("params") == xf.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    diff_r3 = set(xf.recipe_diff(r3_recipe, recipe))
    want = xf.expected_args(recipe)
    bad = {k: (args.get(k), want.get(k)) for k in sorted(set(args) | set(want)) if args.get(k) != want.get(k)}
    meta = xf.check_result_metadata(r)
    ok = not bad and not meta and diff_r3 == set(xf.R3_DEVIATIONS)
    info = xf.RECIPE_NAME if ok else (bad or meta or f"recipe differs from R3 in {sorted(diff_r3)}, expected {sorted(xf.R3_DEVIATIONS)}")
    out.append(("recorded args == expected XF args (R3 recipe with ONLY amp False and eta_min 2e-8), fp32, run identity", ok, info))
    out.append(("100 contiguous epochs", [h.get("epoch") for h in hist] == list(range(xf.EPOCHS)), len(hist)))
    try:
        fin = all(h["train"]["nonfinite"] == 0 and h["train"]["inf_grad_steps"] == 0 and h.get("scaler_scale") is None
                  and _finite(h["train"]["total"], h["train"]["gnorm_mean"], h["train"]["gnorm_max"], h["val_A"]["psnr_mean"], h["val_A"]["ssim_mean"]) for h in hist)
        info = f"non-finite loss steps {sum(h['train']['nonfinite'] for h in hist)}, non-finite-gradient steps {sum(h['train']['inf_grad_steps'] for h in hist)}"
    except (KeyError, TypeError) as e:
        fin, info = False, f"history record incomplete ({type(e).__name__}: {e})"
    out.append(("no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler", fin, info))
    try:
        best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
        ok = best["epoch"] == r.get("best_valA_epoch") and abs(best["val_A"]["psnr_mean"] - r["best_valA"]["psnr_mean"]) < 1e-9
        info = f"epoch {best['epoch']}"
    except (KeyError, TypeError, ValueError) as e:
        ok, info = False, f"cannot determine the best epoch ({type(e).__name__})"
    out.append(("best epoch == argmax history", ok, info))
    return out


def check_loss_config(lc, r, recipe, current_original_sha256):
    """Check 10 -> (passed, info). lc = parsed xf_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, "git_commit_full missing from xf_loss_config.json"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    problems = xf.check_runtime(lc.get("runtime"))
    if problems:
        return False, f"runtime record invalid: {problems}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = xf.loss_config(full, recipe, sha, lc["runtime"])
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"xf_loss_config.json differs from the expected record in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != xf.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    rt = lc["runtime"]
    return True, f"commit {full[:8]}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}, cudnn_allow_tf32 {rt['cudnn_allow_tf32']}"


def check_clip(records, hist):
    """Check 11 -> (passed, info). Pure wrapper around xf.check_clip_log."""
    problems = xf.check_clip_log(records, hist)
    if problems:
        return False, "; ".join(problems[:4]) + (f" (+{len(problems) - 4} more)" if len(problems) > 4 else "")
    total = sum(r["steps"] for r in records)
    mean_frac = float(np.mean([r["frac_clipped"] for r in records]))
    return True, f"{len(records)} lines, {total} steps, mean clipped fraction {mean_frac:.4f}"


def main():
    import torch
    import train_v2 as tv
    from models.at_marine import ATMarineTrunk
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    xf.validate_run_name(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xf.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xf.ALLOWED_MANIFESTS)
    d = ROOT / "runs_v2" / a.run
    checks, ok = {}, True

    def rec(name, cond, info=""):
        nonlocal ok
        checks[name] = {"pass": bool(cond), "info": str(info)}
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)

    present = all((d / f).exists() for f in FILES)
    rec("files exist", present)
    if not present:
        if d.is_dir():
            with open(result_path(d), "x") as f:
                json.dump({"pass": False, "checks": checks}, f, indent=1)
        sys.exit(1)
    r = json.load(open(d / "result.json"))
    hist = [json.loads(line) for line in open(d / "history.jsonl")]
    recipe = xf.recipe_args(tv.RECIPES)
    st = static_checks(r, hist, recipe, tv.RECIPES[xf.BASELINE_RECIPE])
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), xf.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec("manifest hashes as expected (exactly train + val_A), pairs valid", set(mh) == set(xf.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in xf.EXPECTED_HASH_PREFIXES.items()) and pairs_ok,
        f"{sorted(mh)}; {pinfo}")
    nm, c, i = st[4]
    rec(nm, c, i)
    rec("checkpoint SHA-256 == result.json", all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    m = ATMarineTrunk().to(dev)
    ck = torch.load(d / "best_valA.pth", map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec("val-A re-score == logged, checkpoint epoch == best epoch", s["n"] == 166 and dp < TOL and ds < TOL and ck.get("epoch") == r["best_valA_epoch"],
        f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e}) checkpoint epoch {ck.get('epoch')} vs {r['best_valA_epoch']}")
    lc = json.load(open(d / xf.LOSS_CONFIG_FILE))
    try:
        cur = xf.sha256_file(ROOT / xf.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, cur)
    rec("xf_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded", c10, i10)
    try:
        records = [json.loads(line) for line in open(d / xf.CLIP_LOG_FILE)]
        c11, i11 = check_clip(records, hist)
    except ValueError as e:
        c11, i11 = False, f"xf_clip_log.jsonl is not valid JSON lines ({e})"
    rec("xf_clip_log.jsonl: 100 epochs, 86,600 steps, consistent with history.jsonl", c11, i11)
    with open(result_path(d), "x") as f:
        json.dump({"run": a.run, "pass": ok, "checks": checks, "rescore": s}, f, indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
