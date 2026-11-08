"""Reverse cross-experiment (XR) validation gate for the ONE finished run xr_wwe_r3loss_s0 (adapted from v2_tools/xa_gate.py and wwe_gate.py; it re-scores on the GPU and reads Val-A).
Usage (NOT approved to run): python v2_tools/xr_gate.py --run xr_wwe_r3loss_s0
Ten checks (differences from the WWE-UIE gate marked *):
  1 files exist   2 model == wwe_uie, params == 734,059   3 recipe args == the XR recipe (= WWE_lr2e-4), run identity / metadata (fp32, no resume), AND the recorded args differ from the
  WWE-UIE seed-0 reference (runs_v2/wwe_lr2e-4_s0/result.json) ONLY in {recipe, run_name}*   4 100 contiguous epochs
  5 no non-finite loss steps, ZERO non-finite-gradient steps, finite train loss / gradient norms / Val-A*   6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected
  prefixes (no val, val_clean or val_B entry), AND strict pair validation of the train and val_A lists   7 best epoch == argmax history   8 checkpoint SHA-256 == result.json
  9 independent Val-A re-score == logged (1e-4), and the checkpoint's stored epoch == the best epoch*   10 xr_loss_config.json equals the record re-derived from the code, its commit matches
  result.json, and original_repo/marine_net_model.py is readable, equals the PINNED SHA-256 and the hash recorded at launch (fail closed if it cannot be verified)*
Checks 2-5, 7 and 10 are pure functions (`static_checks`, `check_loss_config`), CPU-testable with synthetic dicts.
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
from v2_tools import xr_config as xr  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = 1e-4
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", "xr_loss_config.json"]


def result_path(d):
    """Where the gate writes its JSON verdict: xr_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / "xr_gate.json"
    return p if not p.exists() else Path(d) / f"xr_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def _finite(*vals):
    return all(v is not None and np.isfinite(v) for v in vals)


def static_checks(r, hist, baseline_args, recipe):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access."""
    out = []
    out.append(("model/params", r.get("model") == xr.MODEL and r.get("params") == xr.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    bad = {k: args.get(k) for k, v in recipe.items() if args.get(k) != v}
    meta = xr.check_result_metadata(r)
    diff = xr.recipe_diff(baseline_args, args)
    out.append(("recipe args == XR recipe (= WWE_lr2e-4), fp32, run identity, differs from the WWE-UIE s0 reference only in recipe / run_name",
                not bad and not meta and set(diff) == {"recipe", "run_name"}, bad or meta or ({k: v for k, v in diff.items()} if set(diff) != {"recipe", "run_name"} else xr.RECIPE_NAME)))
    out.append(("100 contiguous epochs", [h.get("epoch") for h in hist] == list(range(100)), len(hist)))
    try:
        fin = all(h["train"]["nonfinite"] == 0 and h["train"]["inf_grad_steps"] == 0 and _finite(h["train"]["total"], h["train"]["gnorm_mean"], h["train"]["gnorm_max"], h["val_A"]["psnr_mean"], h["val_A"]["ssim_mean"])
                  for h in hist)
        info = f"non-finite loss steps {sum(h['train']['nonfinite'] for h in hist)}, non-finite-gradient steps {sum(h['train']['inf_grad_steps'] for h in hist)}"
    except (KeyError, TypeError) as e:
        fin, info = False, f"history record incomplete ({type(e).__name__}: {e})"
    out.append(("no non-finite loss / gradient steps, finite losses, gradient norms and Val-A", fin, info))
    try:
        best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
        ok = best["epoch"] == r.get("best_valA_epoch") and abs(best["val_A"]["psnr_mean"] - r["best_valA"]["psnr_mean"]) < 1e-9
        info = f"epoch {best['epoch']}"
    except (KeyError, TypeError, ValueError) as e:
        ok, info = False, f"cannot determine the best epoch ({type(e).__name__})"
    out.append(("best epoch == argmax history", ok, info))
    return out


def check_loss_config(lc, r, recipe, wwe_commit, current_original_sha256):
    """Check 10 -> (passed, info). lc = parsed xr_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, "git_commit_full missing from xr_loss_config.json"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = xr.loss_config(full, recipe, wwe_commit, sha)
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"xr_loss_config.json differs from the expected record in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != xr.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    return True, f"commit {full[:8]}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}"


def main():
    import torch
    import train_v2 as tv
    from models.wwe_uie import WWE_COMMIT, WWEUIE
    from v2_tools import wwe_config as wcfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    xr.validate_run_name(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xr.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xr.ALLOWED_MANIFESTS)
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
    baseline_args = json.load(open(ROOT / "runs_v2" / xr.BASELINE_RUN / "result.json"))["args"]
    recipe = xr.recipe_args(wcfg.RECIPES)
    st = static_checks(r, hist, baseline_args, recipe)
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), xr.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec("manifest hashes as expected (exactly train + val_A), pairs valid", set(mh) == set(xr.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in xr.EXPECTED_HASH_PREFIXES.items()) and pairs_ok,
        f"{sorted(mh)}; {pinfo}")
    nm, c, i = st[4]
    rec(nm, c, i)
    rec("checkpoint SHA-256 == result.json", all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    m = WWEUIE().to(dev)
    ck = torch.load(d / "best_valA.pth", map_location=dev, weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec("val-A re-score == logged, checkpoint epoch == best epoch", s["n"] == 166 and dp < TOL and ds < TOL and ck.get("epoch") == r["best_valA_epoch"],
        f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e}) checkpoint epoch {ck.get('epoch')} vs {r['best_valA_epoch']}")
    lc = json.load(open(d / "xr_loss_config.json"))
    try:
        cur = xr.sha256_file(ROOT / xr.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, WWE_COMMIT, cur)
    rec("xr_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded", c10, i10)
    with open(result_path(d), "x") as f:
        json.dump({"run": a.run, "pass": ok, "checks": checks, "rescore": s}, f, indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
