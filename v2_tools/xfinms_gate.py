"""XFINMS validation gate for ONE finished run xfin_attrunk_in_r3_fp32_s1 or xfin_attrunk_in_r3_fp32_s2 (adapted from v2_tools/xfin_gate.py and xfms_gate.py, which are NOT modified; it re-scores on the GPU and
reads Val-A). Usage (NOT approved to run): python v2_tools/xfinms_gate.py --run xfin_attrunk_in_r3_fp32_s1
Thirteen checks: the twelve XFIN checks with the seed parameterised, plus
  13 (new) args pairing with the existing XFIN seed-0 run: that run's result.json must have the PINNED SHA-256 (the gate hashes and parses the same bytes), describe XFIN seed 0, and its recorded argument set must
     differ from the new run's in EXACTLY {seed, run_name}.
  1 files exist (incl. xf_clip_log.jsonl)   2 model == at_trunk_in, params == 1,952,768   3 recorded args == the expected XFIN argument set of THIS seed exactly (XF recipe values), run identity, fp32, no resume
  4 100 contiguous epochs   5 no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler   6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected
  prefixes, AND strict pair validation of the train and val_A lists   7 best epoch == argmax history   8 checkpoint SHA-256 == result.json   9 independent Val-A re-score (ATMarineTrunkIN, loaded STRICTLY) ==
  logged (1e-4), and the checkpoint's stored epoch == the best epoch   10 xfin_loss_config.json equals the record re-derived from the code for this seed, its commit matches result.json, and
  original_repo/marine_net_model.py equals the PINNED SHA-256 and the hash recorded at launch   11 xf_clip_log.jsonl: 100 lines, 86,600 steps, per-epoch values equal history.jsonl
  12 normalisation structure: exactly 40 InstanceNorm2d and 0 BatchNorm2d layers, and no running statistics in best_valA.pth / last.pth.
The gate prints this run's metrics; the pre-registered PAIRED rule (xfms_config, unchanged) needs all six runs and is NOT evaluated here. It does not open any other run's result.json except the pinned XFIN
seed-0 reference. val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import xfin_config as xi  # noqa: E402
from v2_tools import xfin_gate as xig  # noqa: E402
from v2_tools import xfinms_config as xim  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = xig.TOL
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xim.LOSS_CONFIG_FILE, xim.CLIP_LOG_FILE]
CHECK_NAMES = list(xig.CHECK_NAMES) + ["args pair with the existing XFIN seed-0 run (result.json SHA-256 pinned): differ only in seed / run_name"]
assert len(CHECK_NAMES) == 13
check_clip = xig.check_clip  # the XFIN check, unchanged
structure_check = xig.structure_check
state_problems = xig.state_problems
_finite = xig._finite


def result_path(d):
    """xfinms_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / xim.GATE_JSON_FILE
    return p if not p.exists() else Path(d) / f"xfinms_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def static_checks(r, hist, recipe, xf_recipe, seed):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access. recipe = xim.recipe_args(...), xf_recipe = xf_config.recipe_args(...) (must be identical)."""
    xim.validate_seed(seed)
    out = []
    out.append((CHECK_NAMES[1], r.get("model") == xim.MODEL and r.get("params") == xim.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    want = xim.expected_args(recipe, seed)
    bad = {k: (args.get(k), want.get(k)) for k in sorted(set(args) | set(want)) if args.get(k) != want.get(k)}
    meta = xim.check_result_metadata(r, seed)
    ok = not bad and not meta and recipe == xf_recipe
    info = xim.RECIPE_NAME if ok else (bad or meta or "the XFINMS recipe values differ from the XF recipe")
    out.append((CHECK_NAMES[2], ok, info))
    out.append((CHECK_NAMES[3], [h.get("epoch") for h in hist] == list(range(xim.EPOCHS)), len(hist)))
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
    """Check 10 -> (passed, info). lc = parsed xfin_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    xim.validate_seed(seed)
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, f"git_commit_full missing from {xim.LOSS_CONFIG_FILE}"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    problems = xim.check_runtime(lc.get("runtime"))
    if problems:
        return False, f"runtime record invalid: {problems}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = xim.loss_config(seed, full, recipe, sha, lc["runtime"])
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"{xim.LOSS_CONFIG_FILE} differs from the expected record for seed {seed} in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != xim.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    rt = lc["runtime"]
    return True, f"commit {full[:8]}, seed {seed}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}, cudnn_allow_tf32 {rt['cudnn_allow_tf32']}"


def check_pairing(args, xin0, xin0_sha256, expected_sha256=xim.XFIN_S0_RESULT_SHA256):
    """Check 13 -> (passed, info). args = result.json['args'] of the new run, xin0 = the parsed result.json of the existing XFIN seed-0 run (or None). The XFIN seed-0 record must itself describe XFIN seed 0
    (identity, model, fp32, eta_min, manifest scope), the file must have the pinned SHA-256 (xin0_sha256 = the hash of the exact bytes that were parsed; None = unreadable -> fail), and the two argument sets
    must differ in EXACTLY {seed, run_name}. Pure."""
    if not isinstance(xin0, dict) or not isinstance(args, dict):
        return False, "the XFIN seed-0 result.json (or the new run's args) is missing or not a JSON object"
    if xin0_sha256 != expected_sha256:
        return False, f"the XFIN seed-0 result.json SHA-256 {str(xin0_sha256)[:12]}... != the pinned {expected_sha256[:12]}...: the reference changed or could not be verified"
    problems = xi.check_result_metadata(xin0)
    if problems:
        return False, f"the XFIN seed-0 record does not describe XFIN seed 0: {problems}"
    xa = xin0.get("args") or {}
    diff = {k for k in set(args) | set(xa) if args.get(k) != xa.get(k)}
    if diff != xim.EXPECTED_DIFF_VS_XFIN_S0:
        return False, f"args differ from XFIN seed 0 in {sorted(diff)}, expected exactly {sorted(xim.EXPECTED_DIFF_VS_XFIN_S0)}"
    return True, f"differs from {xim.EXISTING_XFIN_RUN} only in {sorted(diff)}"


def main():
    import torch
    import train_v2 as tv
    from models.at_trunk_in import ATMarineTrunkIN
    from v2_tools import xf_config as xf
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    seed = xim.seed_of_run(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xim.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xim.ALLOWED_MANIFESTS)
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
    recipe = xim.recipe_args(tv.RECIPES)
    st = static_checks(r, hist, recipe, xf.recipe_args(tv.RECIPES), seed)
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), xim.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec(CHECK_NAMES[5], set(mh) == set(xim.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in xim.EXPECTED_HASH_PREFIXES.items()) and pairs_ok, f"{sorted(mh)}; {pinfo}")
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
    lc = json.load(open(d / xim.LOSS_CONFIG_FILE))
    try:
        cur = xim.sha256_file(ROOT / xim.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, cur, seed)
    rec(CHECK_NAMES[9], c10, i10)
    try:
        records = [json.loads(line) for line in open(d / xim.CLIP_LOG_FILE)]
        c11, i11 = check_clip(records, hist)
    except ValueError as e:
        c11, i11 = False, f"{xim.CLIP_LOG_FILE} is not valid JSON lines ({e})"
    rec(CHECK_NAMES[10], c11, i11)
    last = torch.load(d / "last.pth", map_location="cpu", weights_only=False)
    c12, i12 = structure_check(ATMarineTrunkIN(), ck, last)
    rec(CHECK_NAMES[11], c12, i12)
    try:
        raw = (ROOT / "runs_v2" / xim.EXISTING_XFIN_RUN / "result.json").read_bytes()  # read ONCE: the hashed bytes are the parsed bytes
        x0_sha, x0 = hashlib.sha256(raw).hexdigest(), json.loads(raw)
    except (OSError, ValueError):
        x0_sha, x0 = None, None
    c13, i13 = check_pairing(r.get("args"), x0, x0_sha)
    rec(CHECK_NAMES[12], c13, i13)
    try:
        mt = xim.metrics_from_history(hist)
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
