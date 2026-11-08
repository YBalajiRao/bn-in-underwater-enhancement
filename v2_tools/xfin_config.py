"""XFIN experiment configuration: AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d (models/at_trunk_in.py), trained under the EXACT recipe of XF (R3 loss, fp32, cosine floor 2e-8,
seed 0). Run name xfin_attrunk_in_r3_fp32_s0. The ONLY intended difference from XF (xf_attrunk_r3_fp32_s0) is the normalisation class.
Hypothesis (exploratory): replacing BatchNorm by InstanceNorm in AT-Trunk improves Val-A PSNR, e.g. by removing batch coupling / the train-eval statistics mismatch. The experiment does NOT separate that mechanism from the
other effect of the swap (per-image channel normalisation). Compare ONLY with XF seed 0 (best 28.501, last-10 mean 28.384, SSIM at best 0.9338): same seed, loss, precision and schedule; every Val-A number is exploratory
(Val-A also selects the checkpoint) and a single seed cannot exclude a seed effect (the AT-Trunk R3 seeds span 0.41 dB).
Import-light and side-effect free (no manifest, checkpoint or image access). Reuses xf_config for every constant, the recipe and the loss record so nothing can drift from XF. The XF result.json is never read by this
tooling: the reference numbers below are plain documentation constants. Val-B and UIEB are never referenced."""
from v2_tools import xf_config as xf

ROOT = xf.ROOT
RUN_NAME = "xfin_attrunk_in_r3_fp32_s0"
RECIPE_NAME = "XFIN_R3_fp32_floor_IN"  # the XF recipe VALUES under a new name (recorded args differ from XF's only in recipe / model / run_name)
MODEL = "at_trunk_in"
SEED = 0
ALLOWED_SEEDS = (SEED,)
EXPECTED_PARAMS = xf.EXPECTED_PARAMS  # 1,952,768: unchanged by the swap
EPOCHS = xf.EPOCHS
STEPS_PER_EPOCH = xf.STEPS_PER_EPOCH
EXPECTED_TOTAL_STEPS = xf.EXPECTED_TOTAL_STEPS
N_NORM_LAYERS = 40
BASELINE_RUN = xf.RUN_NAME  # xf_attrunk_r3_fp32_s0: documentation only, its result.json is NEVER opened by this tooling
BASELINE_RECIPE = xf.RECIPE_NAME
XF_COMMIT = "c943032d454825d3bceb383ed64abc5381732a3c"  # commit that holds the reviewed XF tooling this tooling imports (ancestor of HEAD; XF files must equal it at launch)
LOSS_CONFIG_FILE = "xfin_loss_config.json"
CLIP_LOG_FILE = xf.CLIP_LOG_FILE  # the XF recorder writes this file name; it lives in the new run directory
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

# plain documentation numbers from the completed XF run (NOT read from any file by the tooling): Val-A, exploratory
REFERENCE_NUMBERS = {"xf_best_psnr": 28.501, "xf_last10_mean_psnr": 28.384, "xf_ssim_at_best": 0.9338, "at_trunk_r3_seed_range_best_psnr": 0.408,
                     "xn_wwe_uie_best_psnr": 29.694, "xn_wwe_uie_last10_mean_psnr": 29.649, "matched_gap_xn_minus_xf_best": 1.193}
# pre-registered practical thresholds (NOT significance tests; one seed)
DECISION_THRESHOLDS = {"supports_min_delta": 0.5, "weakens_max_delta": 0.3, "ssim_max_drop": 0.005}
# The thresholds above are INCLUSIVE and compared with this absolute tolerance, so that a delta computed in floating point (e.g. 29.001 - 28.501 = 0.4999999999999964 or 0.5000000000000036) is read as exactly the
# boundary it represents. 1e-9 is ~1e-6 of the smallest reported precision (1e-4 dB / 1e-6 SSIM) and far below any real metric difference; the reported metrics themselves are never rounded.
DECISION_TOLERANCE = 1e-9

NEW_TOOLING_FILES = ["models/at_trunk_in.py", "v2_tools/xfin_config.py", "v2_tools/train_v2_no_valb_xfin.py", "v2_tools/xfin_gate.py", "v2_tools/run_xfin_single.py"]
XF_IMPORTED_FILES = list(xf.NEW_TOOLING_FILES)  # imported (recorder, constants, helpers); must equal the XF commit
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + xf.PROVENANCE_FILES))  # tracked + unmodified at launch (wrapper and driver both check)

REMAINING_CONFOUNDERS = [
    "mechanism: the swap removes batch coupling and the train / eval statistics mismatch AND adds per-image channel normalisation; one run cannot tell which one acts",
    "full-channel InstanceNorm2d on all 40 sites, unlike WWE-UIE's half-channel InstanceNorm2d on 10 layers: the result does not transfer to WWE-UIE's design, and WWE-UIE's remaining differences stay bundled",
    "global colour head: it reads the mean-pooled bottleneck (at_marine.py:171) and the grid head the pooled bottleneck (:174); instance normalisation zeroes the per-image channel mean of every block BRANCH, so colour-cast "
    "information must reach the heads through the unnormalised stem and the residual skips; the experiment cannot exclude that this costs accuracy (the result would show it only as a lower PSNR)",
    "numerics: InstanceNorm2d in the fp32 path (this is why XF, not the AMP baseline, is the reference); the fp32 convolutions may use TF32 (PyTorch default, recorded in xfin_loss_config.json)",
    "one seed: the AT-Trunk R3 seeds span 0.41 dB; cuDNN is not bitwise deterministic, so the same seed does not reproduce XF exactly; Val-A selects the checkpoint (best-epoch selection inflates noisy curves)",
    "the reference is XF seed 0 only (28.501 / 28.384); the three-seed AMP baseline is NOT the primary comparison",
]


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: XFIN is defined for seed {list(ALLOWED_SEEDS)} only")
    return seed


def validate_run_name(name):
    if name != RUN_NAME:
        raise ValueError(f"run name {name!r} is not the dedicated XFIN run name {RUN_NAME!r}")
    return name


def recipe_args(tv_recipes):
    """The XF recipe (R3 from train_v2.RECIPES with amp False and eta_min 2e-8), never retyped."""
    return xf.recipe_args(tv_recipes)


def expected_args(recipe):
    """The full argparse namespace train_v2.main records in result.json['args'] for this run: XF's expected args with recipe / model / run_name replaced. Pure."""
    a = xf.expected_args(recipe)
    a.update(model=MODEL, run_name=RUN_NAME, recipe=RECIPE_NAME)
    return a


def expected_diff_keys_vs_xf():
    return {"recipe", "model", "run_name"}


def decision(best_psnr, last10_mean_psnr, ssim_at_best, nonfinite_steps=0):
    """Pre-registered reading of the finished run against XF seed 0 (documentation numbers above). Pure; used by nothing at run time.
    delta_best = best - 28.501, delta_tail = last-10 mean - 28.384, delta_ssim = ssim - 0.9338. Exclusive classes (thresholds inclusive, compared with DECISION_TOLERANCE = 1e-9):
      'supports_in_helps'      : both PSNR deltas >= +0.5, SSIM drop <= 0.005 and no non-finite step
      'weakens_in_hypothesis'  : both PSNR deltas <= +0.3 (this includes any decrease)
      'inconclusive'           : everything else (needs more seeds).  A single seed excludes only a large effect.
    Fail-safe: if any metric is not a finite real number (NaN, +-inf, None, bool, non-numeric) or nonfinite_steps is not a non-negative integer, the result is 'inconclusive' - never a positive decision."""
    def real(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))

    if not (real(best_psnr) and real(last10_mean_psnr) and real(ssim_at_best)) or isinstance(nonfinite_steps, bool) or not isinstance(nonfinite_steps, int) or nonfinite_steps < 0:
        return "inconclusive"
    t, eps = DECISION_THRESHOLDS, DECISION_TOLERANCE
    d1 = best_psnr - REFERENCE_NUMBERS["xf_best_psnr"]
    d2 = last10_mean_psnr - REFERENCE_NUMBERS["xf_last10_mean_psnr"]
    ds = ssim_at_best - REFERENCE_NUMBERS["xf_ssim_at_best"]
    if nonfinite_steps == 0 and d1 >= t["supports_min_delta"] - eps and d2 >= t["supports_min_delta"] - eps and ds >= -t["ssim_max_drop"] - eps:
        return "supports_in_helps"
    if d1 <= t["weakens_max_delta"] + eps and d2 <= t["weakens_max_delta"] + eps:
        return "weakens_in_hypothesis"
    return "inconclusive"


def loss_config(git_commit_full, recipe, original_sha256, runtime):
    """The exact record written to runs_v2/<run>/xfin_loss_config.json (and re-derived by the gate): XF's record (xf_config.loss_config, unchanged) with ONLY the experiment-specific fields replaced."""
    rec = xf.loss_config(git_commit_full, recipe, original_sha256, runtime)
    rec["experiment"] = ("XFIN: AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d (same channel counts, affine, no running statistics); the R3 loss, the fp32 training path, "
                         "the cosine floor 2e-8, seed 0 and every other setting are exactly those of XF")
    rec.update(run_name=RUN_NAME, recipe_name=RECIPE_NAME, model=MODEL, params=EXPECTED_PARAMS, seed=SEED)
    rec["baseline_reference"] = {"primary": BASELINE_RUN, "primary_recipe": BASELINE_RECIPE, "result_json_read_by_tooling": False, "reference_numbers": dict(REFERENCE_NUMBERS),
                                 "comparison_rule": "compare with XF seed 0 only (same seed, loss, precision, floor); the three-seed AMP AT-Trunk baseline is NOT the primary comparison; all Val-A numbers are exploratory"}
    rec["differs_from_xf"] = {"normalisation": ["40 x BatchNorm2d(c, affine, running statistics)", "40 x InstanceNorm2d(c, affine, no running statistics)"],
                              "recorded_args_differ_in": sorted(expected_diff_keys_vs_xf()), "parameters": "unchanged (1,952,768)", "buffers": "-120 (BatchNorm running statistics)"}
    rec["normalisation"] = {"module": "models/at_trunk_in.py :: ATMarineTrunkIN", "sites": N_NORM_LAYERS, "channels_in_module_order": "(80,)*8 + (160,)*8 + (320,)*8 + (160,)*8 + (80,)*8",
                            "initialisation": "construction and random-number use identical to ATMarineTrunk(); BatchNorm2d and InstanceNorm2d both start at weight 1 / bias 0 without drawing random numbers, so all convolution / linear parameters are bit-identical for the same seed"}
    rec["decision_rule"] = {"primary": "best-epoch Val-A PSNR minus 28.501", "secondary": "last-10-epoch mean Val-A PSNR minus 28.384", "ssim": "SSIM at the best epoch minus 0.9338", "thresholds": dict(DECISION_THRESHOLDS), "comparison_tolerance": DECISION_TOLERANCE,
                            "classes": "supports_in_helps: both PSNR deltas >= +0.5, SSIM drop <= 0.005, no non-finite step; weakens_in_hypothesis: both deltas <= +0.3; otherwise inconclusive. "
                                       "Practical thresholds, not significance tests; one seed excludes only a large effect."}
    rec["remaining_confounders"] = list(REMAINING_CONFOUNDERS)
    rec["code"] = dict(rec["code"], xf_commit=XF_COMMIT)
    rec["code"].pop("xr_commit", None)
    return rec


def check_result_metadata(r):
    """Problems if result.json does not describe THIS experiment (seed, run name, recipe, model, params, fp32, eta_min, no resume, manifest scope); [] if it does."""
    a = r.get("args") or {}
    out = []
    if r.get("seed") != SEED or a.get("seed") != SEED:
        out.append(f"seed {r.get('seed')!r}/{a.get('seed')!r} != {SEED}")
    if a.get("run_name") != RUN_NAME:
        out.append(f"run_name {a.get('run_name')!r} != {RUN_NAME!r}")
    if a.get("recipe") != RECIPE_NAME:
        out.append(f"recipe {a.get('recipe')!r} != {RECIPE_NAME!r}")
    if r.get("model") != MODEL or a.get("model") != MODEL:
        out.append(f"model {r.get('model')!r}/{a.get('model')!r} != {MODEL!r}")
    if r.get("params") != EXPECTED_PARAMS:
        out.append(f"params {r.get('params')!r} != {EXPECTED_PARAMS:,}")
    if a.get("amp") is not False:
        out.append(f"amp is {a.get('amp')!r}, expected False (fp32)")
    if not isinstance(a.get("eta_min"), (int, float)) or isinstance(a.get("eta_min"), bool) or abs(a["eta_min"] - 2e-8) > 1e-15:
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
