"""Reverse cross-experiment ("XR") configuration: the WWE-UIE architecture (unchanged, UNBOUNDED output) trained with the R3 loss (original_repo MARINELoss, unchanged),
everything else = the WWE-UIE seed-0 reference recipe WWE_lr2e-4 (fp32, lr 2e-4, eta_min 2e-8, batch 4, 3-epoch warm-up, clip 1.0, 100 epochs, paired flips, Val-A selection).
Import-light and side-effect free (no manifest, checkpoint or image access): constants, the recipe derivation, the loss-config record and pure checks shared by the wrapper
(train_v2_no_valb_xr.py), the gate (xr_gate.py) and the driver (run_xr_single.py). Seed 0 only. Val-B and UIEB are never referenced.
Compare ONLY with WWE-UIE seed 0 (wwe_lr2e-4_s0), Experiment A (xa_attrunk_wweloss_s0) and the AT-Trunk R3 runs; all Val-A numbers are exploratory (Val-A selects the checkpoint)."""
import hashlib
from pathlib import Path

from v2_tools import wwe_config as wcfg

ROOT = Path(__file__).resolve().parent.parent
RUN_NAME = "xr_wwe_r3loss_s0"
RECIPE_NAME = "XR_WWE_R3loss"
MODEL = "wwe_uie"
SEED = 0
ALLOWED_SEEDS = (SEED,)
EXPECTED_PARAMS = 734_059  # == models.wwe_uie.EXPECTED_PARAMS (asserted by the tests; kept literal so this module stays import-light)
BASELINE_RUN = "wwe_lr2e-4_s0"        # the WWE-UIE seed-0 reference: same model, same recipe, WWE composite loss
BASELINE_RECIPE = "WWE_lr2e-4"
CROSS_REFERENCE_RUNS = ("xa_attrunk_wweloss_s0", "stage4a_at_trunk_s0", "stage4a_at_trunk_s1", "stage4a_at_trunk_s2")
EXPECTED_N = dict(wcfg.EXPECTED_N)  # {"train": 3463, "val_A": 166}: the only splits this experiment may load
# ONLY the training and Val-A manifests may be hashed or loaded (val_B first, then val and val_clean, which train_v2.verify_manifests would otherwise hash, are removed first)
ALLOWED_MANIFESTS = ("train", "val_A")
EXPECTED_HASH_PREFIXES = {"train": "b4871a98", "val_A": "2d8daae6"}
# The R3 loss lives in a git-IGNORED file, so git cannot verify it. Its SHA-256 is PINNED here (observed 2026-10-19 on the file with mtime 2026-10-09, which predates every recorded AT-Trunk run and
# equals the hash recorded by the Experiment A preflight/loss config). Missing file, unreadable file or any other hash = fail closed, before launch, while training, after training and in the gate.
UNTRACKED_DEPENDENCY = "original_repo/marine_net_model.py"
PINNED_ORIGINAL_SHA256 = "8c3a265408c2d71dfdc61d299503d6174549ac5ef58d03231369c94f3dad0c91"
# files whose bytes must be tracked and unmodified at launch (the run's recorded commit then contains the whole tooling)
NEW_TOOLING_FILES = ["v2_tools/xr_config.py", "v2_tools/train_v2_no_valb_xr.py", "v2_tools/xr_gate.py", "v2_tools/run_xr_single.py",
                     "tests_v2/test_xr_pipeline.py", "tests_v2/test_xr_driver.py"]
WWE_COMMITTED_FILES = ["models/wwe_uie.py", "v2_tools/wwe_config.py", "v2_tools/wwe_loss.py", "v2_tools/wwe_pairing.py", "v2_tools/train_v2_no_valb_wwe.py",
                       "v2_tools/wwe_gate.py", "v2_tools/wwe_pilot.py", "tests_v2/test_wwe_uie.py", "tests_v2/test_wwe_pipeline.py", "tests_v2/test_wwe_pilot.py"]  # == run_wwe_single.COMMITTED_FILES (tested)
EXISTING_DEPENDENCIES = ["models/at_marine.py", "models_v2/color_nets.py", "train_v2.py", "tools/eval_reproduce_baseline.py",  # train_v2 imports the first two
                         "models/wwe_uie.py", "v2_tools/wwe_config.py", "v2_tools/wwe_pairing.py"]
IMPORTED_DRIVER_HELPERS = ["v2_tools/run_wwe_single.py"]  # imported by run_xr_single.py for kill_tree / run_bounded / make_kill_on_close_job / popen_kwargs / check_manifests / check_disk_and_cache
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + WWE_COMMITTED_FILES + EXISTING_DEPENDENCIES + IMPORTED_DRIVER_HELPERS))  # tracked + unmodified at launch (wrapper and driver both check)

R3_WEIGHTS = {"l1": 1.0, "aux_l1_each": 0.3, "ssim11": 0.5, "perceptual": 0.0, "gradient": 0.4, "color_stats": 0.2}  # documentation of MARINELoss.forward (pinned by the file hash)
UNAVOIDABLE_DIFFERENCES = {
    "vs_wwe_uie_seed0_reference": [
        "loss: WWE composite loss -> R3 (the intended change)",
        "R3 clamps prediction and target to [0,1] INSIDE the loss, so pixels the unbounded WWE-UIE output pushes outside [0,1] receive exactly zero gradient (dead zone); the WWE loss' Charbonnier term acted on the raw output",
        "history.jsonl keys l1/ssim/gradient/color now hold the R3 terms (L1 / 1-SSIM11 / gradient L1 / colour statistics), not the WWE terms",
        "train_v2.step is replaced by an identical step that raises BEFORE optimizer.step on a non-finite gradient norm (no effect when gradients are finite)"],
    "vs_at_trunk_r3_and_experiment_a": [
        "model: WWE-UIE (734,059 parameters, residual output, unbounded) vs AT-Trunk (1,952,768, in-model output clamp)",
        "precision: fp32 here vs AMP forward + GradScaler",
        "cosine floor eta_min 2e-8 here vs 1e-6",
        "gradient-clip regime differs (pre-clip norm depends on loss and model)"],
}


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: the reverse cross-experiment is defined for seed {list(ALLOWED_SEEDS)} only")
    return seed


def validate_run_name(name):
    if name != RUN_NAME:
        raise ValueError(f"run name {name!r} is not the dedicated reverse-cross run name {RUN_NAME!r}")
    return name


def recipe_args(wwe_recipes):
    """The WWE seed-0 reference recipe, copied from wwe_config.RECIPES (never retyped, so it cannot drift from the reference run)."""
    return dict(wwe_recipes[BASELINE_RECIPE])


def recipe_diff(baseline_args, run_args):
    """{key: (baseline, run)} for every argument that differs (keys missing on one side count as differences)."""
    return {k: (baseline_args.get(k), run_args.get(k)) for k in sorted(set(baseline_args) | set(run_args)) if baseline_args.get(k) != run_args.get(k)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_original_dependency(path=None, expected=PINNED_ORIGINAL_SHA256):
    """Problems (list of str) unless the git-ignored R3 loss file exists, is readable and has the pinned SHA-256. FAIL CLOSED: any error is a problem. Cheap (19 KB), safe to poll."""
    p = Path(path) if path is not None else ROOT / UNTRACKED_DEPENDENCY
    try:
        got = sha256_file(p)
    except OSError as e:
        return [f"{p.name}: cannot be read ({type(e).__name__}); R3 provenance cannot be verified"]
    if got != expected:
        return [f"{p.name}: SHA-256 {got[:12]}... != pinned {expected[:12]}...; the R3 loss definition is not the verified one"]
    return []


def loss_config(git_commit_full, recipe, wwe_commit, original_sha256):
    """The exact record written to runs_v2/<run>/xr_loss_config.json (and re-derived by the gate for comparison)."""
    return {
        "experiment": "XR (reverse cross): WWE-UIE architecture unchanged (unbounded residual output); ONLY the loss is replaced (WWE composite loss -> R3 loss)",
        "run_name": RUN_NAME, "recipe_name": RECIPE_NAME, "model": MODEL, "params": EXPECTED_PARAMS, "seed": SEED,
        "baseline_reference": {"primary": BASELINE_RUN, "primary_recipe": BASELINE_RECIPE, "other_references": list(CROSS_REFERENCE_RUNS),
                               "comparison_rule": "compare with WWE-UIE seed 0 (same model and recipe) and Experiment A / AT-Trunk R3 only as stated in unavoidable_differences; all Val-A numbers are exploratory (Val-A selects the checkpoint)"},
        "recipe_args": dict(recipe),
        "data": {"manifest_scope": "ONLY the train and val_A manifests are hashed / loaded; val_B, val and val_clean are removed from train_v2.MANIFESTS before train_v2.main() runs; UIEB is never referenced",
                 "allowed_manifests": list(ALLOWED_MANIFESTS), "expected_counts": dict(EXPECTED_N), "hash_prefixes": dict(EXPECTED_HASH_PREFIXES)},
        "loss": {
            "implementation": "original_repo/marine_net_model.py :: MARINELoss (unchanged, SHA-256 pinned) via train_v2.FinalOutputLoss (auxiliary L1 terms are fed the target, so exactly 0) with train_v2.ZeroLoss replacing the weight-0.0 perceptual term, exactly as train_v2.build does for AT-Trunk",
            "weights": dict(R3_WEIGHTS),
            "equation": "1.0*L1(clamp(pred),clamp(y)) + 0.5*(1-SSIM11) + 0.4*GradientL1 + 0.2*ColourStats(mean MSE + std MSE) [+ 0.3*(aux1+aux2)=0] [+ 0.0*perceptual]",
            "clamp": "pred and target are clamped to [0,1] INSIDE the loss (marine_net_model.py lines 420-423); no clamp is added to the model",
            "dead_zone": "DEAD ZONE: torch.clamp passes zero gradient for pixels outside [0,1]: a pixel of the unbounded WWE-UIE output below 0 or above 1 gets NO gradient from any R3 term and can only return through shared weights; evaluation clamps to [0,1] as well",
            "precision": "fp32 forward and fp32 loss (amp False, no GradScaler)",
            "non_finite_guard": "train_v2.step is replaced by guarded_step: clip_grad_norm_ as before, but a non-finite total norm raises FloatingPointError BEFORE optimizer.step (train_v2.step would step with NaN gradients under fp32)",
            "r3_internal_nan_guards": "MARINELoss replaces a non-finite term by a constant (marine_net_model.py 438-459); that masking is part of the R3 definition and is NOT altered; the driver and gate additionally stop on non-finite gradient norms",
            "log_key_map": {"l1": "L1", "ssim": "1-SSIM11", "gradient": "gradient L1", "color": "colour statistics (mean/std MSE)"},
        },
        "unchanged": ["WWE-UIE architecture and parameter count", "train_v2 training loop, optimizer AdamW, scheduler, warm-up, clip", "paired horizontal flips only, same data and data order for seed 0",
                      "frozen evaluator train_v2.evaluate and Val-A best-PSNR selection"],
        "unavoidable_differences": UNAVOIDABLE_DIFFERENCES,
        "code": {"git_commit_full": git_commit_full, "wwe_commit": wwe_commit,
                 "untracked_dependency": UNTRACKED_DEPENDENCY, "untracked_dependency_sha256": original_sha256, "pinned_sha256": PINNED_ORIGINAL_SHA256,
                 "untracked_dependency_note": "original_repo/ is git-ignored; its SHA-256 is pinned in xr_config and re-verified before launch, while training, after training and in the gate"},
    }


def check_result_metadata(r):
    """Problems if result.json does not describe THIS experiment (seed, run name, recipe, model, params, fp32, no resume, manifest scope); [] if it does."""
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
