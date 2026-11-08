"""XF experiment configuration: AT-Trunk (architecture, in-model output clamp and identity initialisation UNCHANGED) trained with the R3 loss (original_repo MARINELoss, UNCHANGED) under
Recipe R3 EXCEPT two training-side settings: fp32 (amp False, no GradScaler) and cosine floor eta_min 2e-8 (R3: AMP, 1e-6). Seed 0 only. Run name xf_attrunk_r3_fp32_s0.
Purpose: test whether the training-side differences that are aliased with the model in every earlier cross-comparison (AMP, cosine floor) explain part of the 1.155 dB gap between WWE-UIE (XR, R3 loss)
and AT-Trunk (R3, seed 0). Compare ONLY with AT-Trunk R3 seed 0 (28.411 dB, last-10 mean 28.113) as stated in the decision rule below; every Val-A number is exploratory (Val-A selects the checkpoint).
Import-light and side-effect free (no manifest, checkpoint or image access): constants, the recipe derivation, the loss-config record, the gradient-clip log statistics and pure checks shared by the
wrapper (train_v2_no_valb_xf.py), the gate (xf_gate.py) and the driver (run_xf_single.py). Val-B and UIEB are never referenced. The AT-Trunk R3 seed-0 result.json is NEVER read by this tooling
(its record holds a legacy manifest-hash entry for a restricted split): the reference numbers below are plain documentation constants and the recipe is compared against train_v2.RECIPES["R3"]."""
import numpy as np

from v2_tools import xr_config as xr

ROOT = xr.ROOT
RUN_NAME = "xf_attrunk_r3_fp32_s0"
RECIPE_NAME = "XF_R3_fp32_floor"
MODEL = "at_trunk"
SEED = 0
ALLOWED_SEEDS = (SEED,)
EXPECTED_PARAMS = 1_952_768
EPOCHS = 100
STEPS_PER_EPOCH = 866  # ceil(3463 / 4)
EXPECTED_TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH  # 86,600
BASELINE_RECIPE = "R3"
BASELINE_RUN = "stage4a_at_trunk_s0"  # documentation only: NEVER opened by the XF tooling
CROSS_REFERENCE_RUNS = ("xr_wwe_r3loss_s0", "xa_attrunk_wweloss_s0", "wwe_lr2e-4_s0", "stage4a_at_trunk_s1", "stage4a_at_trunk_s2")
LOSS_CONFIG_FILE = "xf_loss_config.json"
CLIP_LOG_FILE = "xf_clip_log.jsonl"
XR_COMMIT = "a1261b53387b9d5b0d982d2c64ec1973e4b94764"  # the commit that holds the reviewed XR tooling this tooling imports (must be an ancestor of HEAD at launch)
EXPECTED_N = dict(xr.EXPECTED_N)  # {"train": 3463, "val_A": 166}: the only splits this experiment may load
ALLOWED_MANIFESTS = tuple(xr.ALLOWED_MANIFESTS)  # ("train", "val_A")
EXPECTED_HASH_PREFIXES = dict(xr.EXPECTED_HASH_PREFIXES)
UNTRACKED_DEPENDENCY = xr.UNTRACKED_DEPENDENCY
PINNED_ORIGINAL_SHA256 = xr.PINNED_ORIGINAL_SHA256  # the tests compare this with an independent literal
R3_WEIGHTS = dict(xr.R3_WEIGHTS)
sha256_file = xr.sha256_file
recipe_diff = xr.recipe_diff

# recipe deviations from R3 (the ONLY two); expected recorded values
R3_DEVIATIONS = {"amp": [True, False], "eta_min": [1e-6, 2e-8]}
# plain documentation numbers taken from the completed runs (NOT read from any file by the tooling): Val-A, exploratory
REFERENCE_NUMBERS = {"at_trunk_r3_s0_best_psnr": 28.411, "at_trunk_r3_s0_last10_mean_psnr": 28.113, "xr_wwe_uie_r3loss_best_psnr": 29.566,
                     "gap_under_r3_loss_seed0": 1.155, "at_trunk_r3_seed_range": 0.408}
DECISION_THRESHOLDS = {"not_main_max_delta": 0.4, "explains_most_min_delta": 0.8, "hurts_max_delta": -0.4}

NEW_TOOLING_FILES = ["v2_tools/xf_config.py", "v2_tools/train_v2_no_valb_xf.py", "v2_tools/xf_gate.py", "v2_tools/run_xf_single.py",
                     "tests_v2/test_xf_pipeline.py", "tests_v2/test_xf_driver.py"]
XR_IMPORTED_FILES = ["v2_tools/xr_config.py", "v2_tools/train_v2_no_valb_xr.py", "v2_tools/run_xr_single.py"]  # imported (guarded_step, constants, fp32 history scan, bounded probes)
WWE_IMPORTED_FILES = ["models/wwe_uie.py", "v2_tools/wwe_config.py", "v2_tools/wwe_pairing.py", "v2_tools/run_wwe_single.py"]  # imported by the XR modules / helpers
EXISTING_DEPENDENCIES = ["models/at_marine.py", "models_v2/color_nets.py", "train_v2.py", "tools/eval_reproduce_baseline.py"]  # train_v2 imports the first two
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + XR_IMPORTED_FILES + WWE_IMPORTED_FILES + EXISTING_DEPENDENCIES))  # tracked + unmodified at launch (wrapper and driver both check)

CLIP_KEYS = ("steps", "n_clipped", "frac_clipped", "mean_scale", "gnorm_mean", "gnorm_p50", "gnorm_p90", "gnorm_p99", "gnorm_max", "clip")
RUNTIME_KEYS = ("cudnn_allow_tf32", "cuda_matmul_allow_tf32", "float32_matmul_precision", "cudnn_benchmark", "torch")

REMAINING_CONFOUNDERS = [
    "model: AT-Trunk (1,952,768 parameters, in-model output clamp, identity-initialised heads) vs WWE-UIE (734,059, unbounded output, gray-world white-balance front end) stays bundled",
    "R3 clamps prediction and target inside the loss; AT-Trunk's output is already clamped in the model, so this is identical to the AT-Trunk R3 runs",
    "gradient-clip regime: the pre-clip norm depends on model and loss; the clipped fraction is logged for this run only (no earlier run logged it)",
    "the fp32 training path may use TF32 convolutions (PyTorch default, left unchanged and recorded in xf_loss_config.json runtime); the WWE-UIE fp32 runs share this default but did not record it",
    "one seed; Val-A selects the checkpoint (best-epoch selection inflates noisy curves); cuDNN is not bitwise deterministic, so the same seed does not reproduce AT-Trunk R3 seed 0 exactly",
]


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: XF is defined for seed {list(ALLOWED_SEEDS)} only")
    return seed


def validate_run_name(name):
    if name != RUN_NAME:
        raise ValueError(f"run name {name!r} is not the dedicated XF run name {RUN_NAME!r}")
    return name


def recipe_args(tv_recipes):
    """Recipe R3 (copied from train_v2.RECIPES, never retyped) with exactly two changes: amp False, eta_min 2e-8."""
    r = dict(tv_recipes[BASELINE_RECIPE])
    r["amp"] = False
    r["eta_min"] = 2e-8
    return r


def expected_args(recipe):
    """The full argparse namespace train_v2.main records in result.json['args'] for this run (recipe values + the non-recipe options). Pure."""
    a = dict(recipe)
    a.update(model=MODEL, run_name=RUN_NAME, seed=SEED, workers=2, time_iters=0, smoke=0, train_limit=None, resume=False, underfit_n=300, recipe=RECIPE_NAME)
    return a


def verify_original_dependency(path=None, expected=PINNED_ORIGINAL_SHA256):
    """Problems (list of str) unless the git-ignored R3 loss file exists, is readable and has the pinned SHA-256 (fail closed)."""
    return xr.verify_original_dependency(path, expected)


def decision(best_psnr, last10_mean_psnr):
    """Pre-registered reading of the finished run against AT-Trunk R3 seed 0 (documentation numbers above). Pure; used by nothing at run time.
    Delta_best = best - 28.411, delta_tail = last-10 mean - 28.113. Exclusive classes, both deltas must agree:
      both >= +0.8 -> 'nuisances_explain_most'; else both <= +0.4 -> 'fp32_hurts' if either <= -0.4 else 'nuisances_not_main'; otherwise 'ambiguous'."""
    d1 = best_psnr - REFERENCE_NUMBERS["at_trunk_r3_s0_best_psnr"]
    d2 = last10_mean_psnr - REFERENCE_NUMBERS["at_trunk_r3_s0_last10_mean_psnr"]
    t = DECISION_THRESHOLDS
    if d1 >= t["explains_most_min_delta"] and d2 >= t["explains_most_min_delta"]:
        return "nuisances_explain_most"
    if d1 <= t["not_main_max_delta"] and d2 <= t["not_main_max_delta"]:
        return "fp32_hurts" if (d1 <= t["hurts_max_delta"] or d2 <= t["hurts_max_delta"]) else "nuisances_not_main"
    return "ambiguous"


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def check_runtime(rt):
    """Problems (list) unless the recorded runtime flags are present and well typed (the VALUES are recorded, not constrained: the default is left unchanged)."""
    if not isinstance(rt, dict):
        return ["runtime record missing"]
    out = []
    if set(rt) != set(RUNTIME_KEYS):
        out.append(f"runtime keys {sorted(rt)} != {sorted(RUNTIME_KEYS)}")
        return out
    for k in ("cudnn_allow_tf32", "cuda_matmul_allow_tf32", "cudnn_benchmark"):
        if not isinstance(rt[k], bool):
            out.append(f"runtime.{k} is not a bool")
    if rt["float32_matmul_precision"] not in ("highest", "high", "medium"):
        out.append("runtime.float32_matmul_precision is not highest / high / medium")
    if not isinstance(rt["torch"], str) or not rt["torch"]:
        out.append("runtime.torch is not a non-empty string")
    return out


def loss_config(git_commit_full, recipe, original_sha256, runtime):
    """The exact record written to runs_v2/<run>/xf_loss_config.json (and re-derived by the gate for comparison; `runtime` is validated separately by check_runtime)."""
    return {
        "experiment": "XF: AT-Trunk architecture and R3 loss UNCHANGED; training-side change only: AMP -> fp32 (no GradScaler) and cosine floor eta_min 1e-6 -> 2e-8",
        "run_name": RUN_NAME, "recipe_name": RECIPE_NAME, "model": MODEL, "params": EXPECTED_PARAMS, "seed": SEED,
        "baseline_reference": {"primary": BASELINE_RUN, "primary_recipe": BASELINE_RECIPE, "result_json_read_by_tooling": False, "other_references": list(CROSS_REFERENCE_RUNS),
                               "reference_numbers": dict(REFERENCE_NUMBERS),
                               "comparison_rule": "compare with AT-Trunk R3 seed 0 only (same seed, data order, model and loss); all Val-A numbers are exploratory (Val-A selects the checkpoint)"},
        "recipe_args": dict(recipe),
        "differs_from_r3": {k: list(v) for k, v in R3_DEVIATIONS.items()},
        "data": {"manifest_scope": "ONLY the train and val_A manifests are hashed / loaded; val_B, val and val_clean are removed from train_v2.MANIFESTS before train_v2.main() runs; UIEB is never referenced",
                 "allowed_manifests": list(ALLOWED_MANIFESTS), "expected_counts": dict(EXPECTED_N), "hash_prefixes": dict(EXPECTED_HASH_PREFIXES)},
        "loss": {
            "implementation": "original_repo/marine_net_model.py :: MARINELoss (unchanged, SHA-256 pinned) via train_v2.build exactly as for the AT-Trunk R3 runs: train_v2.FinalOutputLoss (auxiliary L1 terms fed the target, exactly 0) and train_v2.ZeroLoss for the weight-0.0 perceptual term",
            "weights": dict(R3_WEIGHTS),
            "equation": "1.0*L1(clamp(pred),clamp(y)) + 0.5*(1-SSIM11) + 0.4*GradientL1 + 0.2*ColourStats(mean MSE + std MSE) [+ 0.3*(aux1+aux2)=0] [+ 0.0*perceptual]",
            "clamp": "pred and target are clamped to [0,1] inside the loss; AT-Trunk's output is already clamped inside the model, so this is identical to the AT-Trunk R3 runs",
            "precision": "fp32 forward and fp32 loss (amp False, no GradScaler); convolution TF32 is left at the PyTorch default and recorded under 'runtime'",
            "non_finite_guard": "train_v2.step is replaced by the reviewed XR guarded_step wrapped by a recorder: clip_grad_norm_ as before, a non-finite total norm raises FloatingPointError BEFORE optimizer.step (train_v2.step would step with NaN gradients under fp32)",
            "r3_internal_nan_guards": "MARINELoss replaces a non-finite term by a constant; that masking is part of the R3 definition and is NOT altered; the driver and gate additionally stop on non-finite gradient norms",
        },
        "clip_logging": {"file": CLIP_LOG_FILE, "one_line_per_epoch": list(CLIP_KEYS), "clipped_means": "clip_grad_norm_ rescaled the gradient (scale = min(1, clip / (norm + 1e-6)) < 1)",
                         "note": "observation only: the recorder reads the pre-clip norm that the step already returns; no training arithmetic is changed; no earlier run logged this, so there is no baseline fraction"},
        "unchanged": ["AT-Trunk architecture, output clamp, initialisation and parameter count", "MARINELoss and its weights", "train_v2 training loop, optimizer AdamW, warm-up 3, clip 1.0, batch 4, lr 2e-4, 100 epochs",
                      "paired horizontal flips only, same data and data order for seed 0", "frozen evaluator train_v2.evaluate and Val-A best-PSNR selection"],
        "remaining_confounders": list(REMAINING_CONFOUNDERS),
        "decision_rule": {"primary": "best-epoch Val-A PSNR minus 28.411", "secondary": "last-10-epoch mean Val-A PSNR minus 28.113", "thresholds": dict(DECISION_THRESHOLDS),
                          "classes": "both >= +0.8: nuisances explain most; both <= +0.4: nuisances not the main factor (fp32 hurts if either <= -0.4); otherwise ambiguous (needs a second seed). A single seed excludes only a large effect."},
        "runtime": runtime,
        "code": {"git_commit_full": git_commit_full, "xr_commit": XR_COMMIT,
                 "untracked_dependency": UNTRACKED_DEPENDENCY, "untracked_dependency_sha256": original_sha256, "pinned_sha256": PINNED_ORIGINAL_SHA256,
                 "untracked_dependency_note": "original_repo/ is git-ignored; its SHA-256 is pinned in xr_config and re-verified before launch, while training, after training and in the gate"},
    }


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


# ----------------------------------------------------------------------------- gradient-clip statistics (shared by the wrapper's recorder and the gate)
def summarize_norms(norms, clip):
    """Per-epoch clip statistics from the list of pre-clip total gradient norms. Raises ValueError on an empty list or a non-finite norm. Pure.
    scale = min(1, clip / (norm + 1e-6)) is exactly the factor torch.nn.utils.clip_grad_norm_ applies; a step is 'clipped' when that factor is < 1."""
    g = np.asarray(norms, dtype=np.float64)
    if g.size == 0 or not bool(np.isfinite(g).all()):
        raise ValueError("gradient norm list is empty or contains a non-finite value")
    scale = np.minimum(1.0, clip / (g + 1e-6))
    clipped = scale < 1.0
    return {"steps": int(g.size), "n_clipped": int(clipped.sum()), "frac_clipped": float(clipped.mean()), "mean_scale": float(scale.mean()), "gnorm_mean": float(g.mean()),
            "gnorm_p50": float(np.percentile(g, 50)), "gnorm_p90": float(np.percentile(g, 90)), "gnorm_p99": float(np.percentile(g, 99)), "gnorm_max": float(g.max()), "clip": float(clip)}


def check_clip_log(records, hist, clip=1.0, tol=1e-6, epochs=EPOCHS, total_steps=EXPECTED_TOTAL_STEPS):
    """Problems (list of str) in the parsed xf_clip_log.jsonl records against the parsed history.jsonl records; [] if consistent. Pure."""
    out = []
    if len(records) != epochs:
        out.append(f"{len(records)} clip-log lines, expected {epochs}")
    if len(records) != len(hist):
        out.append(f"{len(records)} clip-log lines but {len(hist)} history records")
    if [r.get("epoch") if isinstance(r, dict) else None for r in records] != list(range(len(records))):
        out.append("clip-log epochs are not contiguous from 0")
    total = 0
    want = set(CLIP_KEYS) | {"epoch"}
    for i, r in enumerate(records):
        if not isinstance(r, dict) or set(r) != want:
            out.append(f"line {i}: keys {sorted(r) if isinstance(r, dict) else r!r} != {sorted(want)}")
            continue
        if not all(_num(r[k]) for k in want):
            out.append(f"line {i}: a value is missing or not a finite number")
            continue
        s, n = r["steps"], r["n_clipped"]
        total += s
        if r["clip"] != clip:
            out.append(f"line {i}: clip {r['clip']} != {clip}")
        if not (s > 0 and 0 <= n <= s):
            out.append(f"line {i}: n_clipped {n} outside 0..steps {s}")
            continue
        if abs(r["frac_clipped"] - n / s) > 1e-9:
            out.append(f"line {i}: frac_clipped {r['frac_clipped']} != n_clipped/steps {n / s}")
        if not (0.0 < r["mean_scale"] <= 1.0 + 1e-12):
            out.append(f"line {i}: mean_scale {r['mean_scale']} outside (0, 1]")
        if not (r["gnorm_p50"] <= r["gnorm_p90"] + 1e-12 and r["gnorm_p90"] <= r["gnorm_p99"] + 1e-12 and r["gnorm_p99"] <= r["gnorm_max"] + 1e-12 and r["gnorm_mean"] <= r["gnorm_max"] + 1e-12):
            out.append(f"line {i}: gradient-norm statistics are not ordered (p50 <= p90 <= p99 <= max, mean <= max)")
        if n == 0 and (r["gnorm_max"] > clip or abs(r["mean_scale"] - 1.0) > 1e-12):
            out.append(f"line {i}: no clipped steps but max norm {r['gnorm_max']} > clip or mean_scale != 1")
        if n > 0 and r["gnorm_max"] <= clip - 1e-5:
            out.append(f"line {i}: {n} clipped steps but max norm {r['gnorm_max']} <= clip")
        if i < len(hist):
            t = (hist[i] or {}).get("train") or {}
            if t.get("steps") != s:
                out.append(f"line {i}: steps {s} != history steps {t.get('steps')}")
            for a_, b_ in (("gnorm_mean", "gnorm_mean"), ("gnorm_max", "gnorm_max")):
                hv = t.get(b_)
                if not _num(hv) or abs(r[a_] - hv) > tol:
                    out.append(f"line {i}: {a_} {r[a_]} != history {hv}")
    if total != total_steps:
        out.append(f"total steps {total} != {total_steps}")
    return out
