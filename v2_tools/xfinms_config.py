"""XFINMS configuration: the XFIN experiment (AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d, R3 loss, fp32, cosine floor 2e-8) repeated for seeds 1 and 2, the matched
XFIN partners of the XFMS baselines xf_attrunk_r3_fp32_s1 / _s2. Sibling of v2_tools/xfin_config.py, which is NOT modified and is imported for every constant, the recipe and the loss record, so nothing can
drift from XFIN seed 0. Only the seed (and the seed-dependent run name) changes: seeds 1 and 2, run names xfin_attrunk_in_r3_fp32_s1 / xfin_attrunk_in_r3_fp32_s2. Seed 0 is the existing XFIN run
xfin_attrunk_in_r3_fp32_s0 and can never be launched from this tooling.
The pre-registered paired comparison rule (XFIN minus XF of the same seed, last-10 mean PSNR primary; xfms_config.PAIRED_THRESHOLDS / paired_decision) is defined in v2_tools/xfms_config.py, committed BEFORE any
XFIN seed-1/2 run, and is imported here UNCHANGED: this file defines no threshold of its own. Every Val-A number is exploratory (Val-A selects the checkpoint). Running these seeds is NOT approved by this file.
Import-light and side-effect free (no manifest, checkpoint or image access). Val-B and UIEB are never referenced."""
from pathlib import Path

from v2_tools import xf_config as xf
from v2_tools import xfin_config as xi
from v2_tools import xfms_config as xm

ROOT = xi.ROOT
RECIPE_NAME = xi.RECIPE_NAME  # SAME recipe name and values as XFIN seed 0, so the recorded args differ from XFIN seed 0's only in seed / run_name
MODEL = xi.MODEL
ALLOWED_SEEDS = (1, 2)  # seed 0 is the existing XFIN run and can never be launched from this tooling
PAIRED_SEEDS = xm.PAIRED_SEEDS
EXISTING_XFIN_RUN = xi.RUN_NAME  # xfin_attrunk_in_r3_fp32_s0: its recorded args must differ from a new run's only in {seed, run_name}
EXISTING_XFIN_SEED = xi.SEED
EXPECTED_DIFF_VS_XFIN_S0 = {"seed", "run_name"}
# SHA-256 of the existing XFIN seed-0 result.json (train + val_A manifests only), the reference of the args pairing. Pinned in the source like the R3 loss file: the preflight and the gate both compare the
# file with this value, so a change at any time after this constant was committed is detected. The file itself is never modified by this tooling.
XFIN_S0_RESULT_SHA256 = "36d6641cae178ebcf5d91bc9607df32a16d5088a29c49b5918c76dc2a0b48ce1"
XFIN_COMMIT = "5f04019654996076886ca4e8124da67117362131"  # commit holding the reviewed XFIN tooling this tooling imports (ancestor of HEAD; the XFIN files must equal it at launch)
XF_COMMIT = xi.XF_COMMIT
LOSS_CONFIG_FILE = xi.LOSS_CONFIG_FILE  # xfin_loss_config.json, inside the new run directory
CLIP_LOG_FILE = xi.CLIP_LOG_FILE
GATE_LOG_FILE = "xfinms_gate.log"
GATE_JSON_FILE = "xfinms_gate.json"
EXPECTED_PARAMS = xi.EXPECTED_PARAMS
N_NORM_LAYERS = xi.N_NORM_LAYERS
EPOCHS = xi.EPOCHS
STEPS_PER_EPOCH = xi.STEPS_PER_EPOCH
EXPECTED_TOTAL_STEPS = xi.EXPECTED_TOTAL_STEPS
EXPECTED_N = dict(xi.EXPECTED_N)
ALLOWED_MANIFESTS = tuple(xi.ALLOWED_MANIFESTS)
EXPECTED_HASH_PREFIXES = dict(xi.EXPECTED_HASH_PREFIXES)
UNTRACKED_DEPENDENCY = xi.UNTRACKED_DEPENDENCY
PINNED_ORIGINAL_SHA256 = xi.PINNED_ORIGINAL_SHA256
R3_WEIGHTS = dict(xi.R3_WEIGHTS)
sha256_file = xi.sha256_file
recipe_diff = xi.recipe_diff
verify_original_dependency = xi.verify_original_dependency
check_runtime = xi.check_runtime
check_clip_log = xi.check_clip_log
recipe_args = xi.recipe_args  # the XF recipe (R3 with ONLY amp False and eta_min 2e-8): never retyped

NEW_TOOLING_FILES = ["v2_tools/xfinms_config.py", "v2_tools/train_v2_no_valb_xfinms.py", "v2_tools/xfinms_gate.py", "v2_tools/run_xfinms_single.py", "tests_v2/test_xfinms_pipeline.py"]
XFIN_IMPORTED_FILES = list(xi.NEW_TOOLING_FILES)  # imported (config, gate helpers, model); must equal the XFIN commit
XF_IMPORTED_FILES = list(xi.XF_IMPORTED_FILES)  # imported (recorder, constants, helpers); must equal the XF commit
# tracked + unmodified at launch (wrapper and driver both check); the XFMS files are listed because the paired rule is imported from them
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + xm.NEW_TOOLING_FILES + xi.PROVENANCE_FILES))

# plain documentation numbers from the completed seed-0 runs (NOT read from any file by the tooling): Val-A, exploratory; the XF seeds 1 / 2 numbers are the finished XFMS runs
REFERENCE_NUMBERS = {"xf_s0": dict(xm.REFERENCE_NUMBERS["xf_s0"]), "xfin_s0": dict(xm.REFERENCE_NUMBERS["xfin_s0"]),
                     "xf_s1": {"best_psnr": 28.2217, "last10_psnr": 28.0272, "best_ssim": 0.9312, "last10_ssim": 0.92958, "run_seconds": 18449.1},
                     "xf_s2": {"best_psnr": 28.1408, "last10_psnr": 28.0082, "best_ssim": 0.9305, "last10_ssim": 0.92950, "run_seconds": 18452.4},
                     "at_trunk_r3_amp_seed_range_best_psnr": xm.REFERENCE_NUMBERS["at_trunk_r3_amp_seed_range_best_psnr"]}
ESTIMATED_RUN_HOURS = 5.7  # ESTIMATE ONLY: extrapolated from the recorded XFIN seed-0 run_seconds (20,442.6 s = 5.68 h); not a measurement for these seeds

# The paired rule is NOT redefined here: these are the very objects of xfms_config.
PAIRED_THRESHOLDS = xm.PAIRED_THRESHOLDS
DECISION_TOLERANCE = xm.DECISION_TOLERANCE
metrics_from_history = xm.metrics_from_history
paired_summary = xm.paired_summary
paired_decision = xm.paired_decision

REMAINING_CONFOUNDERS = list(xi.REMAINING_CONFOUNDERS[:-1]) + [
    "paired design: each XFIN seed is compared with the XF run of the SAME seed (xf_attrunk_r3_fp32_s0 / _s1 / _s2); n = 3, so no significance claim is possible (the smallest one-sided sign-test p value is 0.125)",
    "the AMP AT-Trunk R3 seeds are NOT matched baselines (AMP, cosine floor 1e-6) and are not opened by this tooling",
]


def verify_xfin_reference_hash(path=None, expected=XFIN_S0_RESULT_SHA256):
    """Problems (list of str) unless the existing XFIN seed-0 result.json is readable and has the pinned SHA-256 (fail closed: missing or unreadable = refused). Reads only that one file."""
    p = Path(path) if path is not None else ROOT / "runs_v2" / EXISTING_XFIN_RUN / "result.json"
    try:
        h = sha256_file(p)
    except OSError as e:
        return [f"{p.parent.name}/result.json cannot be read ({type(e).__name__}): the XFIN seed-0 reference cannot be verified"]
    return [] if h == expected else [f"{p.parent.name}/result.json SHA-256 {h[:12]}... != the pinned {expected[:12]}...: the XFIN seed-0 reference changed"]


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: XFINMS is defined for seeds {list(ALLOWED_SEEDS)} only (seed 0 is the existing XFIN run)")
    return seed


def run_name(seed):
    validate_seed(seed)
    return f"xfin_attrunk_in_r3_fp32_s{seed}"


def seed_of_run(name):
    """-> seed for one of the two dedicated run names; anything else (the existing XFIN seed-0 run, an XF / XFMS / XAMS / baseline run name or a typo) raises ValueError."""
    for s in ALLOWED_SEEDS:
        if name == run_name(s):
            return s
    raise ValueError(f"run name {name!r} is not one of {[run_name(s) for s in ALLOWED_SEEDS]}")


def expected_argv(seed):
    validate_seed(seed)
    return {"--recipe": RECIPE_NAME, "--model": MODEL, "--seed": str(seed), "--run_name": run_name(seed)}


def expected_args(recipe, seed):
    """The full argparse namespace train_v2.main records in result.json['args']: XFIN's record with ONLY the seed and the run name replaced. Pure."""
    validate_seed(seed)
    a = xi.expected_args(recipe)
    a.update(seed=seed, run_name=run_name(seed))
    return a


def loss_config(seed, git_commit_full, recipe, original_sha256, runtime):
    """XFIN's record (xfin_config.loss_config, unchanged) with ONLY the seed-dependent / descriptive fields replaced; the gate re-derives it with this function for comparison."""
    validate_seed(seed)
    rec = xi.loss_config(git_commit_full, recipe, original_sha256, runtime)
    rec["experiment"] = ("XFINMS: the XFIN experiment (AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d, R3 loss, fp32, cosine floor 2e-8) repeated for seed %d, paired with the XF "
                         "run xf_attrunk_r3_fp32_s%d; model, loss, recipe and precision are XFIN seed 0's" % (seed, seed))
    rec["run_name"] = run_name(seed)
    rec["seed"] = seed
    rec["baseline_reference"] = {"primary": xm.run_name(seed), "primary_recipe": xm.RECIPE_NAME, "xfin_seed0_run": EXISTING_XFIN_RUN, "result_json_read_by_tooling": True,
                                 "note": "the existing XFIN seed-0 result.json (train + val_A manifests only) is read by the preflight and the gate for the args pairing and must have the pinned SHA-256; "
                                         "the XF result.json files and the AMP stage4a_at_trunk_s* runs are not opened",
                                 "xfin_s0_result_json_sha256": XFIN_S0_RESULT_SHA256,
                                 "reference_numbers": {k: dict(v) if isinstance(v, dict) else v for k, v in REFERENCE_NUMBERS.items()},
                                 "comparison_rule": "paired per seed with XF of the same seed; thresholds imported unchanged from xfms_config.PAIRED_THRESHOLDS; all Val-A numbers are exploratory"}
    rec["decision_rule"] = {"primary": "last-10-epoch mean Val-A PSNR, XFIN minus XF of the same seed", "secondary": "best PSNR; last-10 mean SSIM; SSIM at the best epoch",
                            "thresholds": dict(PAIRED_THRESHOLDS), "tolerance": DECISION_TOLERANCE, "defined_in": "v2_tools/xfms_config.py (paired_summary / paired_decision), unchanged",
                            "classes": "supports_in_helps: all deltas >= +0.5, mean >= +1.0, mean > XF seed range, best-PSNR deltas > 0, SSIM deltas >= -0.005; weakens_in_hypothesis: mean <= +0.3; else inconclusive"}
    rec["remaining_confounders"] = list(REMAINING_CONFOUNDERS)
    rec["code"] = dict(rec["code"], xfin_commit=XFIN_COMMIT)
    return rec


def check_result_metadata(r, seed):
    """Problems if result.json does not describe the XFIN run of this seed (identity, model, fp32, eta_min, no resume, manifest scope); [] if it does."""
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
