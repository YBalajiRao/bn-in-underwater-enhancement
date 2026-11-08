"""Reverse cross-experiment (XR) training wrapper around train_v2.main(): the WWE-UIE architecture (unchanged, unbounded output, 734,059 parameters) trained with the R3 loss
(original_repo MARINELoss, unchanged, via train_v2.FinalOutputLoss + ZeroLoss exactly as train_v2.build does for AT-Trunk) under the WWE-UIE seed-0 reference recipe
(fp32, lr 2e-4, eta_min 2e-8, batch 4, 3-epoch warm-up, clip 1.0, 100 epochs, paired flips, Val-A best-PSNR selection).
Safeguards:
  * val_B is removed from train_v2.MANIFESTS BEFORE anything is hashed or loaded, then val and val_clean too; the table is asserted to hold exactly {train, val_A};
  * the command line is a strict whitelist: exactly --recipe XR_WWE_R3loss --model wwe_uie --seed 0 --run_name xr_wwe_r3loss_s0 (no --resume, --smoke, --time_iters, --train_limit, --stop_after);
  * train_v2.load_pairs is wrapped with the strict pair validator (only train and val_A, the FULL list validated);
  * the output directory must be new or contain only the driver's train_attempt0.log (train_v2.main uses mkdir(exist_ok=True) and APPENDS to history.jsonl); xr_loss_config.json is created exclusively;
  * the tooling and code dependencies must be tracked and unmodified, and the index empty, before anything is processed;
  * original_repo/marine_net_model.py (git-ignored) must have the PINNED SHA-256 before the run, and again after tv.main() returns (otherwise exit non-zero);
  * the R3 loss is NOT changed and NO clamp is added to the model; R3's own clamp inside the loss creates a zero-gradient dead zone for out-of-range pixels (documented in xr_loss_config.json);
  * train_v2.step is replaced by guarded_step: identical to train_v2.step for fp32, except that a non-finite total gradient norm raises BEFORE optimizer.step (train_v2.step would step with NaN gradients).
Importing this module has no side effects; everything happens in main(). train_v2.py, the evaluator and all models / losses are not modified.
Usage (NOT approved to run): python v2_tools/train_v2_no_valb_xr.py --recipe XR_WWE_R3loss --model wwe_uie --seed 0 --run_name xr_wwe_r3loss_s0"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.wwe_uie import WWE_COMMIT, WWEUIE  # noqa: E402
from v2_tools import wwe_config as wcfg  # noqa: E402
from v2_tools import xr_config as xr  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

LOSS_CONFIG_FILE = "xr_loss_config.json"
ALLOWED_PREEXISTING = {"train_attempt0.log"}  # created by the single-run driver before it starts this wrapper
EXPECTED_ARGV = {"--recipe": xr.RECIPE_NAME, "--model": xr.MODEL, "--seed": str(xr.SEED), "--run_name": xr.RUN_NAME}
_TRAIN_V2_STEP = tv.step  # the original, kept for the tests


def validate_argv(argv):
    """Exactly the four expected option/value pairs (any order); anything else raises ValueError. Pure."""
    argv = list(argv)
    if len(argv) != 2 * len(EXPECTED_ARGV) or dict(zip(argv[0::2], argv[1::2])) != EXPECTED_ARGV or len(set(argv[0::2])) != len(EXPECTED_ARGV):
        raise ValueError(f"argv must be exactly {' '.join(a for kv in EXPECTED_ARGV.items() for a in kv)} (any order); got {argv}")
    return argv


def prepare_output_dir(root, run_name):
    """Dedicated, collision-safe output directory: new, or existing with ONLY the driver's train_attempt0.log. Anything else (a previous run) is refused."""
    xr.validate_run_name(run_name)
    d = Path(root) / "runs_v2" / run_name
    if d.exists():
        extra = sorted(p.name for p in d.iterdir() if p.name not in ALLOWED_PREEXISTING)
        if extra:
            raise FileExistsError(f"{d} already contains {extra}; refusing to reuse an existing run directory")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git(args, run=subprocess.run):
    try:
        return run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"git {' '.join(args)} timed out; refusing to run without provenance") from e


def git_full_commit(run=subprocess.run):
    r = _git(["rev-parse", "HEAD"], run)
    h = r.stdout.strip()
    if r.returncode != 0 or len(h) != 40:
        raise RuntimeError("cannot determine the git commit; refusing to run without provenance")
    return h


def ensure_clean_provenance(run=subprocess.run):
    """RuntimeError unless the index is empty and every tooling / code-dependency file is tracked and unmodified."""
    if _git(["diff", "--cached", "--quiet"], run).returncode != 0:
        raise RuntimeError("the git index is not empty; refusing to run")
    st = _git(["status", "--porcelain", "--", *xr.PROVENANCE_FILES], run)
    if st.returncode != 0 or st.stdout.strip():
        raise RuntimeError(f"provenance files are modified or untracked: {st.stdout.strip()[:200]!r}; refusing to run")
    for f in xr.PROVENANCE_FILES:
        if _git(["ls-files", "--error-unmatch", f], run).returncode != 0:
            raise RuntimeError(f"{f} is not tracked by git; refusing to run")


def ensure_original_dependency():
    """RuntimeError unless original_repo/marine_net_model.py has the pinned SHA-256 (fail closed: unreadable = refused)."""
    problems = xr.verify_original_dependency()
    if problems:
        raise RuntimeError("; ".join(problems))


def write_loss_config(d, git_commit_full):
    """Exclusive creation ('x'): a second call, or a pre-existing file, raises FileExistsError."""
    rec = xr.loss_config(git_commit_full, xr.recipe_args(wcfg.RECIPES), WWE_COMMIT, xr.sha256_file(ROOT / xr.UNTRACKED_DEPENDENCY))
    with open(Path(d) / LOSS_CONFIG_FILE, "x") as f:
        json.dump(rec, f, indent=1)
    return rec


def guarded_step(model, opt, scaler, a):
    """train_v2.step for fp32 (amp False) with ONE addition: a non-finite total gradient norm raises FloatingPointError BEFORE opt.step(), so the weights are never updated with NaN / Inf
    gradients (train_v2.step clips and steps unconditionally; under fp32 there is no GradScaler to skip the step). With finite gradients it is identical to train_v2.step."""
    if a.amp:
        raise RuntimeError("guarded_step is defined for fp32 only (amp must be False)")
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
    if not bool(torch.isfinite(gn)):
        raise FloatingPointError(f"non-finite gradient norm ({float(gn)}) before optimizer.step; training stopped, weights not updated by this step")
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(gn)


def build(a, device):
    """train_v2.build replacement: WWE-UIE unchanged + the R3 loss exactly as train_v2.build builds it for a single-output model (MARINELoss, ZeroLoss perceptual, FinalOutputLoss)."""
    assert a.recipe == xr.RECIPE_NAME and a.model == xr.MODEL and a.seed == xr.SEED and a.run_name == xr.RUN_NAME and not a.resume, "unexpected arguments"
    assert a.amp is False and a.loss_in_autocast is False, "the reverse cross-experiment is fp32 (amp and loss_in_autocast must be False)"
    model = WWEUIE().to(device)
    assert sum(p.numel() for p in model.parameters()) == xr.EXPECTED_PARAMS, f"parameter count != {xr.EXPECTED_PARAMS:,}"
    crit = tv.MARINELoss().to(device)
    if not getattr(crit, "ssim_loss", None):
        raise RuntimeError("SSIM loss disabled (pytorch-msssim missing): the R3 loss would not be the R3 loss")
    crit.perceptual = tv.ZeroLoss()  # weight 0.0 in MARINELoss; same replacement as train_v2.build (skip_zero_weight_perceptual)
    return model, tv.FinalOutputLoss(crit)


def setup(argv):
    """All process-wide changes. Idempotence is NOT supported (a second call raises)."""
    validate_argv(argv)
    assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
    tv.MANIFESTS.pop("val_B")  # BEFORE any hashing or loading
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xr.ALLOWED_MANIFESTS]:  # val, val_clean: not needed with select_on=val_A, and verify_manifests() would hash them
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xr.ALLOWED_MANIFESTS), f"manifest table must hold exactly {xr.ALLOWED_MANIFESTS}, holds {sorted(tv.MANIFESTS)}"
    assert not any("val_b" in str(p).lower() for p, _, _ in tv.MANIFESTS.values())
    assert all(tv.MANIFESTS[k][2] == xr.EXPECTED_HASH_PREFIXES[k] for k in xr.ALLOWED_MANIFESTS), "expected hash prefixes differ from train_v2"
    assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
    assert xr.RECIPE_NAME not in tv.RECIPES, "recipe name already registered"
    tv.RECIPES[xr.RECIPE_NAME] = xr.recipe_args(wcfg.RECIPES)
    r = tv.RECIPES[xr.RECIPE_NAME]
    assert r == wcfg.RECIPES[xr.BASELINE_RECIPE] and r["amp"] is False and r["loss_in_autocast"] is False and r["lr"] == 2e-4 and r["select_on"] == "val_A", "recipe is not the WWE seed-0 reference recipe"
    assert xr.MODEL not in tv.MODEL_REGISTRY, "model name already registered"
    tv.MODEL_REGISTRY[xr.MODEL] = lambda: WWEUIE()
    m = tv.MODEL_REGISTRY[xr.MODEL]()  # CPU build, no data access
    assert sum(p.numel() for p in m.parameters()) == xr.EXPECTED_PARAMS, f"parameter count != {xr.EXPECTED_PARAMS:,}"
    del m
    tv.load_pairs = checked_loader(tv.load_pairs, xr.EXPECTED_N)  # refuses every split except train / val_A BEFORE touching the original loader
    tv.build = build
    tv.step = guarded_step  # train_one_epoch looks `step` up in the train_v2 module at call time


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        validate_argv(argv)
        ensure_clean_provenance()
        ensure_original_dependency()
        setup(argv)
        d = prepare_output_dir(ROOT, xr.RUN_NAME)
        write_loss_config(d, git_full_commit())
    except (ValueError, FileExistsError, RuntimeError) as e:
        sys.exit(f"STOP: {e}")
    tv.main()  # parses sys.argv itself (identical to the validated argv)
    try:
        ensure_original_dependency()  # the R3 definition must still be the pinned one after the run
    except RuntimeError as e:
        sys.exit(f"STOP: after training: {e}")


if __name__ == "__main__":
    main()
