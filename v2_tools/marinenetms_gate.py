"""MARINENETMS validation gate for ONE finished run (marinenet_bn_r3_fp32_s{0,1,2} or marinenet_in_r3_fp32_s{0,1,2}), adapted from v2_tools/armcinms_gate.py (NOT modified); it re-scores on the GPU and reads Val-A.
Usage (NOT approved to run): python v2_tools/marinenetms_gate.py --run marinenet_bn_r3_fp32_s0
Fourteen checks (the ARMCINMS gate structure with the model, structure and pairing reference adapted; no check is removed or weakened):
  1 files exist (incl. xf_clip_log.jsonl)   2 model == marine_default / marine_default_in of the variant, params == 2,609,401 (trainable)   3 recorded args == the expected argument set of this variant and seed
  exactly (XF recipe values), run identity, fp32, no resume   4 100 contiguous epochs   5 no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler
  6 result.json["manifest_sha256"] holds EXACTLY {train, val_A} with the expected prefixes, AND strict pair validation of the train and val_A lists   7 best epoch == argmax history
  8 checkpoint SHA-256 == result.json   9 independent Val-A re-score (MARINENet / MARINENetIN, loaded STRICTLY) == logged (1e-4), and the checkpoint's stored epoch == the best epoch
  10 marinenet_loss_config.json equals the record re-derived for this variant and seed, its commit matches result.json, and original_repo/marine_net_model.py equals the PINNED SHA-256 and the hash recorded at launch
  11 xf_clip_log.jsonl: 100 lines, 86,600 steps, per-epoch values equal history.jsonl
  12 structure: the exact MARINE-Net class of the variant, 2,609,401 trainable parameters, 40 BatchNorm2d / 0 InstanceNorm2d ('bn') or 40 InstanceNorm2d / 0 BatchNorm2d ('in'), state-dict element counts
     (2,622,241 / 2,609,401), no final_output_only, and the COMPLETE state-dict key set and tensor shapes of both checkpoints equal to the instantiated model's ('in': no running statistics in the checkpoints)
  13 args pairing: the AT-Trunk run of the SAME seed (XF for 'bn', XFIN for 'in'; result.json with its PINNED SHA-256, describing that run) differs from this run's recorded args in EXACTLY {model, run_name}
  14 provenance (git, not self-reported): the commit recorded in marinenet_loss_config.json is a real 40-hex commit and an ancestor of HEAD; every provenance file is tracked and unmodified and equals the file
     at the RECORDED launch commit; and the XF / XFIN / XFINMS files equal their pinned commits. Fail closed: git unavailable, timing out or erroring = FAIL.
     LIMITATION (documented, not bypassed): this proves that the code on disk equals the code committed at the recorded launch commit, NOT that exactly that code was executed by the training process.
The gate prints this run's metrics. It applies NO decision rule (Option A); the descriptive paired report is a separate read-only script. It does not open any other run's result.json except the pinned
reference. val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.marine_net_in import MARINENet, MARINENetIN, model_problems  # noqa: E402
from v2_tools import marinenetms_config as mc  # noqa: E402
from v2_tools import xfin_gate as xig  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

TOL = xig.TOL
FILES = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", mc.LOSS_CONFIG_FILE, mc.CLIP_LOG_FILE]
CHECK_NAMES = ["files exist", "model/params (marine_default / marine_default_in of the variant, 2,609,401 trainable)", "recorded args == expected MARINENETMS args of this variant and seed (XF recipe values), fp32, run identity",
               "100 contiguous epochs", "no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler", "manifest hashes as expected (exactly train + val_A), pairs valid",
               "best epoch == argmax history", "checkpoint SHA-256 == result.json", "val-A re-score == logged, checkpoint epoch == best epoch",
               "marinenet_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded", "xf_clip_log.jsonl: 100 epochs, 86,600 steps, consistent with history.jsonl",
               "structure: exact MARINE-Net class of the variant, 2,609,401 params, 40 BatchNorm2d or 40 InstanceNorm2d, full checkpoint key sets",
               "args pair with the AT-Trunk run of the same seed (XF for bn, XFIN for in; result.json SHA-256 pinned): differ only in model / run_name",
               "provenance: recorded commit is an ancestor of HEAD, tracked provenance files unmodified and equal to the launch commit and to the pinned commits (git)"]
assert len(CHECK_NAMES) == 14 and len(set(CHECK_NAMES)) == 14
check_clip = xig.check_clip  # the XFIN check, unchanged
_finite = xig._finite
GIT_TIMEOUT_S = 60
PINNED_COMMITS = mc.PINNED_COMMITS


def result_path(d):
    """marinenet_gate.json, or (if that already exists, e.g. a manual re-run) a unique timestamped name. Never overwrites an existing file."""
    p = Path(d) / mc.GATE_JSON_FILE
    return p if not p.exists() else Path(d) / f"marinenet_gate_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"


def run_metrics(hist):
    """-> dict of this run's exploratory Val-A metrics (best / last-10 PSNR and SSIM), or raises ValueError. Pure."""
    if not hist:
        raise ValueError("empty history")
    best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
    return {"best_psnr": best["val_A"]["psnr_mean"], "best_epoch": best["epoch"], "best_ssim": best["val_A"]["ssim_mean"],
            "last10_psnr": float(np.mean([h["val_A"]["psnr_mean"] for h in hist[-10:]])), "last10_ssim": float(np.mean([h["val_A"]["ssim_mean"] for h in hist[-10:]]))}


def static_checks(r, hist, recipe, xf_recipe, variant, seed):
    """-> list of (name, passed, info) for checks 2, 3, 4, 5, 7 (in that order); no file or GPU access. recipe = mc.recipe_args(...), xf_recipe = xf_config.recipe_args(...) (must be identical)."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    out = []
    out.append((CHECK_NAMES[1], r.get("model") == mc.MODEL[variant] and r.get("params") == mc.EXPECTED_PARAMS, f"{r.get('model')} {r.get('params')}"))
    args = r.get("args") or {}
    want = mc.expected_args(variant, recipe, seed)
    bad = {k: (args.get(k), want.get(k)) for k in sorted(set(args) | set(want)) if args.get(k) != want.get(k)}
    meta = mc.check_result_metadata(r, variant, seed)
    ok = not bad and not meta and recipe == xf_recipe
    info = mc.RECIPE_NAME[variant] if ok else (bad or meta or "the MARINENETMS recipe values differ from the XF recipe")
    out.append((CHECK_NAMES[2], ok, info))
    out.append((CHECK_NAMES[3], [h.get("epoch") for h in hist] == list(range(mc.EPOCHS)), len(hist)))
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


def check_loss_config(lc, r, recipe, current_original_sha256, variant, seed):
    """Check 10 -> (passed, info). lc = parsed marinenet_loss_config.json, r = result.json, current_original_sha256 = the hash of the file NOW (None = could not be read -> fail). Pure."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, f"git_commit_full missing from {mc.LOSS_CONFIG_FILE}"
    rec = (r.get("env") or {}).get("git_commit", "")
    if not rec or not isinstance(full, str) or len(full) != 40 or not full.startswith(rec):
        return False, f"loss-config commit {full!r} does not match result.json commit {rec!r}"
    problems = mc.check_runtime(lc.get("runtime"))
    if problems:
        return False, f"runtime record invalid: {problems}"
    sha = lc["code"].get("untracked_dependency_sha256")
    exp = mc.loss_config(variant, seed, full, recipe, sha, lc["runtime"])
    if lc != exp:
        diff = sorted(k for k in set(lc) | set(exp) if lc.get(k) != exp.get(k))
        return False, f"{mc.LOSS_CONFIG_FILE} differs from the expected record in {diff}"
    if current_original_sha256 is None:
        return False, "original_repo/marine_net_model.py cannot be read now: R3 provenance cannot be verified"
    if sha != mc.PINNED_ORIGINAL_SHA256:
        return False, "the hash recorded at launch is not the pinned R3 hash"
    if current_original_sha256 != sha:
        return False, "original_repo/marine_net_model.py hash differs from the one recorded at launch (changed during or after the run)"
    rt = lc["runtime"]
    return True, f"commit {full[:8]}, R3 file hash == pinned == recorded, weights {exp['loss']['weights']}, cudnn_allow_tf32 {rt['cudnn_allow_tf32']}"


def key_set_problems(label, sd, model):
    """Problems (list of str) unless the state dict `sd` has EXACTLY the keys (and, where present, the tensor shapes) of the instantiated `model`'s state dict. Missing and unexpected keys are listed explicitly. Pure."""
    if not isinstance(sd, dict):
        return [f"{label}: state dict is not a dict"]
    want = model.state_dict()
    missing, extra = sorted(set(want) - set(sd)), sorted(set(sd) - set(want))
    out = []
    if missing:
        out.append(f"{label}: {len(missing)} missing key(s), e.g. {missing[:3]}")
    if extra:
        out.append(f"{label}: {len(extra)} unexpected key(s), e.g. {extra[:3]}")
    bad = sorted(k for k in set(want) & set(sd) if tuple(getattr(sd[k], "shape", ())) != tuple(want[k].shape))
    if bad:
        out.append(f"{label}: {len(bad)} key(s) with a wrong shape, e.g. {bad[:3]}")
    return out


def structure_check(model, best_ck, last_ck, variant):
    """Check 12 -> (passed, info). model = the built MARINENet / MARINENetIN, best_ck / last_ck = loaded checkpoint dicts. Pure given those objects."""
    from models.at_trunk_in import norm_layers
    probs = list(model_problems(model, variant))
    try:
        for label, sd in (("best_valA.pth", best_ck["model_state_dict"]), ("last.pth", last_ck["model"])):
            if isinstance(sd, dict) and variant == "in":
                probs += [f"{label}: {p}" for p in xig.state_problems(sd)]
            probs += key_set_problems(label, sd, model)
    except (KeyError, TypeError) as e:
        probs.append(f"checkpoint dict incomplete ({type(e).__name__}: {e})")
    if last_ck.get("epoch") != mc.EPOCHS - 1:
        probs.append(f"last.pth epoch {last_ck.get('epoch')} != {mc.EPOCHS - 1}")
    n_in, n_bn = norm_layers(model)
    return (not probs), (f"{variant}: {n_in} InstanceNorm / {n_bn} BatchNorm layers; {mc.EXPECTED_PARAMS:,} trainable parameters; full key sets of both checkpoints == the model's; best epoch {best_ck.get('epoch')}; "
                         f"last epoch {last_ck.get('epoch')}" if not probs else "; ".join(probs)[:600])


def check_pairing(args, variant, seed, ref_rec, ref_sha256, expected_sha256=None):
    """Check 13 -> (passed, info). args = result.json['args'] of the new run. ref_rec = the parsed result.json of the AT-Trunk reference of the SAME seed (XF for bn, XFIN for in; None = unreadable),
    ref_sha256 = the hash of the exact bytes that were parsed. The reference must have its pinned SHA-256 and describe its own run, and the argument sets must differ in EXACTLY {model, run_name}. Pure."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    expected_sha256 = mc.reference_pin(variant, seed) if expected_sha256 is None else expected_sha256
    name = mc.reference_run_name(variant, seed)
    if not isinstance(ref_rec, dict) or not isinstance(args, dict):
        return False, f"the reference result.json ({name}) or the new run's args is missing or not a JSON object"
    if ref_sha256 != expected_sha256:
        return False, f"the {name} result.json SHA-256 {str(ref_sha256)[:12]}... != the pinned {expected_sha256[:12]}...: the reference changed or could not be verified"
    problems = mc.reference_metadata_problems(variant, seed, ref_rec)
    if problems:
        return False, f"the {name} record does not describe {name}: {problems}"
    ra = ref_rec.get("args") or {}
    diff = {k for k in set(args) | set(ra) if args.get(k) != ra.get(k)}
    if diff != mc.EXPECTED_DIFF_VS_REFERENCE:
        return False, f"args differ from {name} in {sorted(diff)}, expected exactly {sorted(mc.EXPECTED_DIFF_VS_REFERENCE)}"
    return True, f"differs from {name} only in {sorted(diff)}"


def _git(args, run=subprocess.run):
    return run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=GIT_TIMEOUT_S)


def provenance_check(lc, run=subprocess.run):
    """Check 14 -> (passed, info). Enforceable provenance from git itself (not from the run's own records): the commit that marinenet_loss_config.json records must be a 40-hex commit that is an ancestor of HEAD;
    every provenance file (the new tooling included) must be tracked and unmodified and equal to the file at the RECORDED launch commit; and the XF / XFIN / XFINMS files must be ancestors of HEAD and equal to
    their pinned commits. Fail closed: a git failure, timeout or missing commit is a FAIL. `run` is injectable for tests."""
    try:
        full = lc["code"]["git_commit_full"]
    except (KeyError, TypeError):
        return False, f"git_commit_full missing from {mc.LOSS_CONFIG_FILE}"
    if not isinstance(full, str) or not re.fullmatch(r"[0-9a-f]{40}", full):
        return False, f"recorded launch commit {full!r} is not a 40-hex commit id"
    files = list(mc.PROVENANCE_FILES)
    probs = []
    try:
        if _git(["merge-base", "--is-ancestor", full, "HEAD"], run).returncode != 0:
            probs.append(f"recorded launch commit {full[:8]} is not an ancestor of HEAD (or does not exist)")
        r = _git(["status", "--porcelain", "--", *files], run)
        if r.returncode != 0 or r.stdout.strip():
            probs.append(f"provenance files are modified or untracked: {r.stdout.strip()[:160]!r}")
        if _git(["ls-files", "--error-unmatch", "--", *files], run).returncode != 0:
            probs.append("a provenance file is not tracked by git")
        if _git(["diff", "--quiet", full, "--", *files], run).returncode != 0:
            probs.append(f"a provenance file on disk differs from the recorded launch commit {full[:8]}")
        for label, commit, cfiles in PINNED_COMMITS:
            if _git(["merge-base", "--is-ancestor", commit, "HEAD"], run).returncode != 0:
                probs.append(f"the {label} commit {commit[:8]} is not an ancestor of HEAD")
            if _git(["diff", "--quiet", commit, "--", *cfiles], run).returncode != 0:
                probs.append(f"a {label} file differs from its pinned commit {commit[:8]}")
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"git could not be run ({type(e).__name__}): provenance cannot be verified"
    if probs:
        return False, "; ".join(probs)[:600]
    return True, f"launch commit {full[:8]} is an ancestor of HEAD; {len(files)} provenance files tracked, unmodified and equal to it; XF / XFIN / XFINMS files equal their pinned commits"


def read_pinned(path):
    """-> (sha256 hex, parsed JSON) of the exact bytes read once, or (None, None) if unreadable. Reads only that one file."""
    try:
        raw = Path(path).read_bytes()
        return hashlib.sha256(raw).hexdigest(), json.loads(raw)
    except (OSError, ValueError):
        return None, None


def main():
    import torch
    import train_v2 as tv
    from v2_tools import xf_config as xfc
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    variant, seed = mc.parse_run_name(a.run)
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in mc.ALLOWED_MANIFESTS]:  # only train and val_A may be processed
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(mc.ALLOWED_MANIFESTS)
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
                json.dump({"run": a.run, "pass": False, "checks": checks}, f, indent=1)
        sys.exit(1)
    r = json.load(open(d / "result.json"))
    hist = [json.loads(line) for line in open(d / "history.jsonl")]
    recipe = mc.recipe_args(tv.RECIPES)
    st = static_checks(r, hist, recipe, xfc.recipe_args(tv.RECIPES), variant, seed)
    for name, cond, info in st[:4]:
        rec(name, cond, info)
    mh = r.get("manifest_sha256") or {}
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), mc.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec(CHECK_NAMES[5], set(mh) == set(mc.ALLOWED_MANIFESTS) and all(mh.get(k, "").startswith(v) for k, v in mc.EXPECTED_HASH_PREFIXES.items()) and pairs_ok, f"{sorted(mh)}; {pinfo}")
    nm, c, i = st[4]
    rec(nm, c, i)
    rec(CHECK_NAMES[7], all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    cls = MARINENet if variant == "bn" else MARINENetIN
    m = cls().to(dev)
    ck = torch.load(d / "best_valA.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"], strict=True)  # strict: a checkpoint of the other normalisation type (extra / missing buffers) would be refused
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec(CHECK_NAMES[8], s["n"] == 166 and dp < TOL and ds < TOL and ck.get("epoch") == r["best_valA_epoch"],
        f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e}) checkpoint epoch {ck.get('epoch')} vs {r['best_valA_epoch']}")
    lc = json.load(open(d / mc.LOSS_CONFIG_FILE))
    try:
        cur = mc.sha256_file(ROOT / mc.UNTRACKED_DEPENDENCY)
    except OSError:
        cur = None
    c10, i10 = check_loss_config(lc, r, recipe, cur, variant, seed)
    rec(CHECK_NAMES[9], c10, i10)
    try:
        records = [json.loads(line) for line in open(d / mc.CLIP_LOG_FILE)]
        c11, i11 = check_clip(records, hist)
    except ValueError as e:
        c11, i11 = False, f"{mc.CLIP_LOG_FILE} is not valid JSON lines ({e})"
    rec(CHECK_NAMES[10], c11, i11)
    last = torch.load(d / "last.pth", map_location="cpu", weights_only=False)
    c12, i12 = structure_check(cls(), ck, last, variant)
    rec(CHECK_NAMES[11], c12, i12)
    x_sha, x_rec = read_pinned(ROOT / "runs_v2" / mc.reference_run_name(variant, seed) / "result.json")  # read ONCE: the hashed bytes are the parsed bytes
    c13, i13 = check_pairing(r.get("args"), variant, seed, x_rec, x_sha)
    rec(CHECK_NAMES[12], c13, i13)
    c14, i14 = provenance_check(lc)
    rec(CHECK_NAMES[13], c14, i14)
    try:
        mt = run_metrics(hist)
        print(f"RUN METRICS (exploratory Val-A; descriptive only, no decision rule): best {mt['best_psnr']:.4f} dB at epoch {mt['best_epoch']}, last-10 mean {mt['last10_psnr']:.4f}, "
              f"SSIM at best {mt['best_ssim']:.4f}, last-10 mean SSIM {mt['last10_ssim']:.5f}; the paired report is produced separately by v2_tools/marinenetms_paired_summary.py", flush=True)
    except (ValueError, KeyError, TypeError) as e:
        mt = None
        print(f"RUN METRICS unavailable: {e}", flush=True)
    with open(result_path(d), "x") as f:
        json.dump({"run": a.run, "variant": variant, "seed": seed, "pass": ok, "checks": checks, "rescore": s, "metrics": mt}, f, indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
