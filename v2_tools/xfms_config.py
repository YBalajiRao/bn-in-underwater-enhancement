"""XFMS configuration: the XF experiment (AT-Trunk, R3 loss, fp32, cosine floor 2e-8) repeated for seeds 1 and 2, the matched baselines for the XFIN (BatchNorm -> InstanceNorm) experiment.
Sibling of v2_tools/xf_config.py, which is NOT modified and is imported for every constant, the recipe and the loss record, so nothing can drift from XF seed 0. Only the seed (and the seed-dependent run
name) changes: seeds 1 and 2, run names xf_attrunk_r3_fp32_s1 / xf_attrunk_r3_fp32_s2. Seed 0 is the existing XF run xf_attrunk_r3_fp32_s0 and can never be launched from this tooling.
Purpose: measure the seed-to-seed variance of the XF baseline and give every XFIN seed a matched XF partner, so XFIN-vs-XF deltas can be paired per seed. Every Val-A number is exploratory (Val-A selects
the checkpoint). Running these seeds is NOT approved by this file; XFIN seeds 1 and 2 would need their own reviewed tooling and approval.
Import-light and side-effect free (no manifest, checkpoint or image access). Val-B and UIEB are never referenced.

PRE-REGISTERED PAIRED COMPARISON RULE (fixed by committing this file BEFORE any XFMS or XFIN seed-1/2 run; practical thresholds, NOT significance tests; n = 3 seeds, so the smallest possible one-sided sign-test
p value is 0.125 and no p value is claimed). Inputs: for each seed s in {0, 1, 2} the metrics of the XF run and of the XFIN run (see metrics_from_history). Deltas are XFIN minus XF of the SAME seed.
  primary metric   : last-10-epoch mean Val-A PSNR (delta_s = XFIN_s - XF_s)
  secondary metrics: best-epoch Val-A PSNR, last-10-epoch mean SSIM, SSIM at the best epoch (reported; the rule uses the first three as stated below)
  invalid          : any of the six runs missing, a seed set other than {0, 1, 2}, a non-finite or missing metric, or any non-finite loss / gradient step  ->  'inconclusive'
  supports_in_helps: EVERY primary delta >= +0.5 dB AND the mean primary delta >= +1.0 dB AND the mean primary delta > the range (max - min) of the three XF last-10 means (the effect must exceed the baseline
                     seed spread) AND every best-PSNR delta > 0 AND every last-10 SSIM delta >= -0.005
  weakens_in_hypothesis: the mean primary delta <= +0.3 dB
  otherwise        : 'inconclusive'
  Comparisons marked >= / <= are inclusive and use DECISION_TOLERANCE (absolute, 1e-9) so that a float delta such as 0.4999999999999964 is read as the boundary 0.5 it represents; metrics are never rounded.
  INTERIM (after XF seeds 1 and 2 only, nothing about IN is claimed): report the range of the three XF last-10 means; BASELINE_RANGE_FLAG (1.0 dB) marks a large baseline spread.
"""
import hashlib
from pathlib import Path

import numpy as np

from v2_tools import xf_config as xf

ROOT = xf.ROOT
RECIPE_NAME = xf.RECIPE_NAME  # the SAME recipe name and values as XF seed 0 (XF_R3_fp32_floor), so the recorded args differ from XF seed 0's only in seed / run_name
MODEL = xf.MODEL
ALLOWED_SEEDS = (1, 2)  # seed 0 is the existing XF run and can never be launched from this tooling
PAIRED_SEEDS = (0, 1, 2)  # seeds that enter the paired comparison (0 = the existing XF / XFIN seed-0 runs)
EXISTING_XF_RUN = xf.RUN_NAME  # xf_attrunk_r3_fp32_s0: its recorded args must differ from a new run's only in {seed, run_name}
EXISTING_XF_SEED = xf.SEED
EXPECTED_DIFF_VS_XF_S0 = {"seed", "run_name"}
# SHA-256 of the existing XF seed-0 result.json (train + val_A manifests only), the reference of the args pairing. Pinned in the source like the R3 loss file: the preflight and the gate both compare the file with
# this value, so a change at any time after this constant was committed (e.g. between the preflight and the gate) is detected. The file itself is never modified by this tooling.
XF_S0_RESULT_SHA256 = "ecdaf4dcc1a1f6f6f906a8dc0bbe71c6e0e6e5baebdd5a37bccd384bc22cdc18"
XF_COMMIT = "c943032d454825d3bceb383ed64abc5381732a3c"  # commit holding the reviewed XF tooling this tooling imports (ancestor of HEAD; the XF files must equal it at launch)
LOSS_CONFIG_FILE = xf.LOSS_CONFIG_FILE  # same record / log names as XF seed 0, inside the new run directory
CLIP_LOG_FILE = xf.CLIP_LOG_FILE
GATE_LOG_FILE = "xfms_gate.log"
GATE_JSON_FILE = "xfms_gate.json"
EXPECTED_PARAMS = xf.EXPECTED_PARAMS
EPOCHS = xf.EPOCHS
STEPS_PER_EPOCH = xf.STEPS_PER_EPOCH
EXPECTED_TOTAL_STEPS = xf.EXPECTED_TOTAL_STEPS
BASELINE_RECIPE = xf.BASELINE_RECIPE  # R3
EXPECTED_N = dict(xf.EXPECTED_N)
ALLOWED_MANIFESTS = tuple(xf.ALLOWED_MANIFESTS)
EXPECTED_HASH_PREFIXES = dict(xf.EXPECTED_HASH_PREFIXES)
UNTRACKED_DEPENDENCY = xf.UNTRACKED_DEPENDENCY
PINNED_ORIGINAL_SHA256 = xf.PINNED_ORIGINAL_SHA256
R3_WEIGHTS = dict(xf.R3_WEIGHTS)
sha256_file = xf.sha256_file
recipe_diff = xf.recipe_diff
verify_original_dependency = xf.verify_original_dependency
check_runtime = xf.check_runtime
check_clip_log = xf.check_clip_log
recipe_args = xf.recipe_args  # R3 from train_v2.RECIPES with ONLY amp False and eta_min 2e-8: never retyped

NEW_TOOLING_FILES = ["v2_tools/xfms_config.py", "v2_tools/train_v2_no_valb_xfms.py", "v2_tools/xfms_gate.py", "v2_tools/run_xfms_single.py", "tests_v2/test_xfms_pipeline.py"]
XF_IMPORTED_FILES = list(xf.NEW_TOOLING_FILES)  # imported (recorder, constants, helpers); must equal the XF commit
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + xf.PROVENANCE_FILES))  # tracked + unmodified at launch (wrapper and driver both check)

# plain documentation numbers from the completed seed-0 runs (NOT read from any file by the tooling): Val-A, exploratory
REFERENCE_NUMBERS = {"xf_s0": {"best_psnr": 28.5012, "last10_psnr": 28.3836, "best_ssim": 0.9338, "last10_ssim": 0.93256, "run_seconds": 18431.8},
                     "xfin_s0": {"best_psnr": 30.4617, "last10_psnr": 30.4513, "best_ssim": 0.9391, "last10_ssim": 0.93914, "run_seconds": 20442.6},
                     "at_trunk_r3_amp_seed_range_best_psnr": 0.408}
ESTIMATED_RUN_HOURS = 5.1  # ESTIMATE ONLY: extrapolated from the recorded XF seed-0 run_seconds (18,431.8 s); not a measurement for these seeds

# ----------------------------------------------------------------------------- pre-registered paired rule
PAIRED_THRESHOLDS = {"supports_min_delta_each": 0.5, "supports_min_delta_mean": 1.0, "weakens_max_delta_mean": 0.3, "ssim_max_drop": 0.005}
DECISION_TOLERANCE = 1e-9
BASELINE_RANGE_FLAG = 1.0
METRIC_KEYS = ("best_psnr", "last10_psnr", "best_ssim", "last10_ssim", "nonfinite")

REMAINING_CONFOUNDERS = [
    "one seed per run; Val-A selects the checkpoint (best-epoch selection inflates noisy curves); all Val-A numbers are exploratory",
    "cuDNN is not bitwise deterministic and the fp32 convolutions may use TF32 (PyTorch default, recorded in xf_loss_config.json): the same seed does not reproduce a run exactly",
    "these runs only measure the XF baseline spread; no normalisation change is tested here, so nothing about InstanceNorm can be concluded from them alone",
    "the AMP AT-Trunk R3 seeds 1 and 2 (stage4a_at_trunk_s1 / _s2) are NOT matched baselines (AMP, cosine floor 1e-6) and are not opened by this tooling",
]


def verify_xf_reference_hash(path=None, expected=XF_S0_RESULT_SHA256):
    """Problems (list of str) unless the existing XF seed-0 result.json is readable and has the pinned SHA-256 (fail closed: missing or unreadable = refused). Reads only that one file."""
    p = Path(path) if path is not None else ROOT / "runs_v2" / EXISTING_XF_RUN / "result.json"
    try:
        h = sha256_file(p)
    except OSError as e:
        return [f"{p.parent.name}/result.json cannot be read ({type(e).__name__}): the XF seed-0 reference cannot be verified"]
    return [] if h == expected else [f"{p.parent.name}/result.json SHA-256 {h[:12]}... != the pinned {expected[:12]}...: the XF seed-0 reference changed"]


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: XFMS is defined for seeds {list(ALLOWED_SEEDS)} only (seed 0 is the existing XF run)")
    return seed


def run_name(seed):
    validate_seed(seed)
    return f"xf_attrunk_r3_fp32_s{seed}"


def seed_of_run(name):
    """-> seed for one of the two dedicated run names; anything else (the existing XF seed-0 run, an XFIN / XAMS / baseline run name or a typo) raises ValueError."""
    for s in ALLOWED_SEEDS:
        if name == run_name(s):
            return s
    raise ValueError(f"run name {name!r} is not one of {[run_name(s) for s in ALLOWED_SEEDS]}")


def expected_argv(seed):
    validate_seed(seed)
    return {"--recipe": RECIPE_NAME, "--model": MODEL, "--seed": str(seed), "--run_name": run_name(seed)}


def expected_args(recipe, seed):
    """The full argparse namespace train_v2.main records in result.json['args']: XF's record with ONLY the seed and the run name replaced. Pure."""
    validate_seed(seed)
    a = xf.expected_args(recipe)
    a.update(seed=seed, run_name=run_name(seed))
    return a


def loss_config(seed, git_commit_full, recipe, original_sha256, runtime):
    """XF's record (xf_config.loss_config, unchanged) with ONLY the seed-dependent / descriptive fields replaced; the gate re-derives it with this function for comparison."""
    validate_seed(seed)
    rec = xf.loss_config(git_commit_full, recipe, original_sha256, runtime)
    rec["experiment"] = ("XFMS: the XF experiment (AT-Trunk, R3 loss, fp32, cosine floor 2e-8) repeated for seed %d as the matched baseline for XFIN; model, loss, recipe and precision are XF seed 0's" % seed)
    rec["run_name"] = run_name(seed)
    rec["seed"] = seed
    rec["baseline_reference"] = {"primary": EXISTING_XF_RUN, "primary_recipe": RECIPE_NAME, "result_json_read_by_tooling": True,
                                 "note": "the existing XF seed-0 result.json (train + val_A manifests only) is read by the preflight and the gate for the args pairing and must have the pinned SHA-256; the AMP stage4a_at_trunk_s* runs are not opened",
                                 "result_json_sha256": XF_S0_RESULT_SHA256,
                                 "reference_numbers": {k: dict(v) if isinstance(v, dict) else v for k, v in REFERENCE_NUMBERS.items()},
                                 "comparison_rule": "paired per seed with XFIN of the same seed; see PAIRED_THRESHOLDS; all Val-A numbers are exploratory"}
    rec["decision_rule"] = {"primary": "last-10-epoch mean Val-A PSNR, XFIN minus XF of the same seed", "secondary": "best PSNR; last-10 mean SSIM; SSIM at the best epoch",
                            "thresholds": dict(PAIRED_THRESHOLDS), "tolerance": DECISION_TOLERANCE,
                            "classes": "supports_in_helps: all deltas >= +0.5, mean >= +1.0, mean > XF seed range, best-PSNR deltas > 0, SSIM deltas >= -0.005; weakens_in_hypothesis: mean <= +0.3; else inconclusive"}
    rec["remaining_confounders"] = list(REMAINING_CONFOUNDERS)
    rec["code"] = dict(rec["code"], xf_commit=XF_COMMIT)
    return rec


def check_result_metadata(r, seed):
    """Problems if result.json does not describe the XF run of this seed (identity, fp32, eta_min, no resume, manifest scope); [] if it does."""
    validate_seed(seed)
    a = r.get("args") or {}
    out = []
    if r.get("seed") != seed or a.get("seed") != seed:
        out.append(f"seed {r.get('seed')!r}/{a.get('seed')!r} != {seed}")
    if a.get("run_name") != run_name(seed):
        out.append(f"run_name {a.get('run_name')!r} != {run_name(seed)!r}")
    if a.get("recipe") != RECIPE_NAME:
        out.append(f"recipe {a.get('recipe')!r} != {RECIPE_NAME!r}")
    if r.get("model") != MODEL or a.get("model") != MODEL:
        out.append(f"model {r.get('model')!r}/{a.get('model')!r} != {MODEL!r}")
    if r.get("params") != EXPECTED_PARAMS:
        out.append(f"params {r.get('params')!r} != {EXPECTED_PARAMS:,}")
    if a.get("amp") is not False:
        out.append(f"amp is {a.get('amp')!r}, expected False (fp32)")
    if not _num(a.get("eta_min")) or abs(a["eta_min"] - 2e-8) > 1e-15:
        out.append(f"eta_min {a.get('eta_min')!r} != 2e-08")
    if a.get("resume") is not False:
        out.append(f"resume flag is {a.get('resume')!r}, expected False")
    m = r.get("manifest_sha256")
    if not isinstance(m, dict) or set(m) != set(ALLOWED_MANIFESTS):
        out.append(f"manifest_sha256 keys {sorted(m) if isinstance(m, dict) else m!r} != {sorted(ALLOWED_MANIFESTS)} (only train and val_A may be hashed)")
    else:
        for k, prefix in EXPECTED_HASH_PREFIXES.items():
            if not str(m.get(k, "")).startswith(prefix):
                out.append(f"manifest {k} hash does not start with {prefix}")
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def metrics_from_history(hist):
    """-> the metrics the paired rule uses, from parsed history.jsonl records (>= 10 epochs): best Val-A PSNR, last-10 mean PSNR / SSIM, SSIM at the best epoch, and the number of non-finite loss + gradient steps.
    Raises ValueError on fewer than 10 records or a missing / non-finite value. Pure."""
    if not isinstance(hist, list) or len(hist) < 10:
        raise ValueError("history needs at least 10 epoch records")
    try:
        p = [float(h["val_A"]["psnr_mean"]) for h in hist]
        s = [float(h["val_A"]["ssim_mean"]) for h in hist]
        nonfinite = sum(int(h["train"]["nonfinite"]) + int(h["train"]["inf_grad_steps"]) for h in hist)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"history record incomplete ({type(e).__name__}: {e})") from e
    if not (np.isfinite(p).all() and np.isfinite(s).all()):
        raise ValueError("a non-finite Val-A value in the history")
    b = max(range(len(p)), key=lambda i: p[i])
    return {"best_psnr": p[b], "best_epoch": hist[b].get("epoch", b), "best_ssim": s[b], "last10_psnr": float(np.mean(p[-10:])), "last10_ssim": float(np.mean(s[-10:])), "nonfinite": nonfinite}


def _valid_runs(runs):
    if not isinstance(runs, dict) or set(runs) != set(PAIRED_SEEDS):
        return False
    for s in PAIRED_SEEDS:
        r = runs[s]
        if not isinstance(r, dict) or any(k not in r for k in METRIC_KEYS) or not all(_num(r[k]) for k in METRIC_KEYS) or r["nonfinite"] != 0:
            return False
    return True


def baseline_range(xf_runs):
    """Range (max - min) of the three XF last-10 means, or None if the input is invalid. The INTERIM statistic after XF seeds 1 and 2."""
    if not _valid_runs(xf_runs):
        return None
    v = [xf_runs[s]["last10_psnr"] for s in PAIRED_SEEDS]
    return max(v) - min(v)


def paired_summary(xf_runs, xfin_runs):
    """Pre-registered reading (see the module docstring). xf_runs / xfin_runs: {seed: {best_psnr, last10_psnr, best_ssim, last10_ssim, nonfinite}} for seeds 0, 1, 2. Pure; used by nothing at run time.
    -> {"class": 'supports_in_helps' | 'weakens_in_hypothesis' | 'inconclusive', "valid": bool, "deltas", "mean_primary_delta", "xf_range", ...}. Invalid input is always 'inconclusive'."""
    if not (_valid_runs(xf_runs) and _valid_runs(xfin_runs)):
        return {"class": "inconclusive", "valid": False}
    d = {m: [xfin_runs[s][m] - xf_runs[s][m] for s in PAIRED_SEEDS] for m in ("last10_psnr", "best_psnr", "last10_ssim", "best_ssim")}
    mean = sum(d["last10_psnr"]) / len(PAIRED_SEEDS)
    rng = baseline_range(xf_runs)
    t, tol = PAIRED_THRESHOLDS, DECISION_TOLERANCE
    supports = (all(x >= t["supports_min_delta_each"] - tol for x in d["last10_psnr"]) and mean >= t["supports_min_delta_mean"] - tol and mean > rng + tol
                and all(x > tol for x in d["best_psnr"]) and all(x >= -t["ssim_max_drop"] - tol for x in d["last10_ssim"]))
    weakens = mean <= t["weakens_max_delta_mean"] + tol
    cls = "supports_in_helps" if supports else ("weakens_in_hypothesis" if weakens else "inconclusive")
    return {"class": cls, "valid": True, "deltas": d, "mean_primary_delta": mean, "min_primary_delta": min(d["last10_psnr"]), "xf_range": rng, "baseline_range_large": rng >= BASELINE_RANGE_FLAG - tol}


def paired_decision(xf_runs, xfin_runs):
    return paired_summary(xf_runs, xfin_runs)["class"]
