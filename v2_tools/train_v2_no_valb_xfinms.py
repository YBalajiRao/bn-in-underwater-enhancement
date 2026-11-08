"""XFINMS training wrapper around train_v2.main(): the XFIN experiment (AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d, R3 loss, fp32, cosine floor 2e-8) for seeds 1 and 2
only. Sibling of v2_tools/train_v2_no_valb_xfin.py (NOT modified): identical safeguards, identical model / loss / recipe; the XF gradient-clip recorder, step guard and helpers are imported from the XF wrapper
unchanged. Only the seed and the seed-dependent run name differ from XFIN seed 0.
Safeguards (as XFIN, with the seed parameterised):
  * val_B is removed from train_v2.MANIFESTS BEFORE anything is hashed or loaded, then val and val_clean too; the table is asserted to hold exactly {train, val_A};
  * the command line is a strict whitelist: exactly one of
        --recipe XFIN_R3_fp32_floor_IN --model at_trunk_in --seed 1 --run_name xfin_attrunk_in_r3_fp32_s1
        --recipe XFIN_R3_fp32_floor_IN --model at_trunk_in --seed 2 --run_name xfin_attrunk_in_r3_fp32_s2
    (any order; no seed 0, no mixed seed / run name, no --resume, --smoke, --time_iters, --train_limit, --stop_after);
  * train_v2.load_pairs is wrapped with the strict pair validator (only train and val_A, the FULL list validated) BEFORE the original loader can run;
  * the output directory must be new or contain only the driver's train_attempt0.log (never the existing XFIN seed-0 directory: its name is refused); the loss config and the clip log are created exclusively;
  * the tooling and code dependencies must be tracked and unmodified, the index empty, the XFIN files equal to the XFIN commit and the XF files equal to the XF commit, before anything is processed;
  * original_repo/marine_net_model.py (git-ignored) must have the PINNED SHA-256 before the run and again after tv.main() returns;
  * the model name at_trunk_in is registered in train_v2.MODEL_REGISTRY in this process only (train_v2.py is not edited); train_v2.build is called as is (XF's _ORIG_BUILD) and its result is asserted:
    ATMarineTrunkIN, 1,952,768 parameters, exactly 40 InstanceNorm2d and 0 BatchNorm2d layers, FinalOutputLoss around MARINELoss with ZeroLoss as perceptual term;
  * train_v2.step -> the XF recording_step, train_v2.train_one_epoch -> the XF recorded_train_one_epoch; no training arithmetic changes.
Importing this module has no side effects; everything happens in main(). train_v2.py, the evaluator and all models / losses are not modified.
Usage (NOT approved to run): python v2_tools/train_v2_no_valb_xfinms.py --recipe XFIN_R3_fp32_floor_IN --model at_trunk_in --seed 1 --run_name xfin_attrunk_in_r3_fp32_s1"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.at_trunk_in import ATMarineTrunkIN, N_NORM_LAYERS, norm_layers  # noqa: E402
from v2_tools import train_v2_no_valb_xf as xfw  # noqa: E402  (imported unchanged: RECORDER, recording_step, recorded_train_one_epoch, runtime_info, ensure_original_dependency, git_full_commit, _git)
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools import xfinms_config as xim  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

LOSS_CONFIG_FILE = xim.LOSS_CONFIG_FILE
ALLOWED_PREEXISTING = {"train_attempt0.log"}  # created by the single-run driver before it starts this wrapper
OPTION_KEYS = ("--recipe", "--model", "--seed", "--run_name")
RECORDER = xfw.RECORDER


def validate_argv(argv):
    """-> the seed. Exactly the four option/value pairs (any order) of ONE allowed seed (1 or 2) with ITS run name; anything else raises ValueError. Pure."""
    argv = list(argv)
    if len(argv) != 2 * len(OPTION_KEYS) or set(argv[0::2]) != set(OPTION_KEYS) or len(set(argv[0::2])) != len(OPTION_KEYS):
        raise ValueError(f"argv must be exactly the four option/value pairs {' '.join(OPTION_KEYS)} (any order); got {argv}")
    d = dict(zip(argv[0::2], argv[1::2]))
    try:
        seed = int(d["--seed"])
    except ValueError as e:
        raise ValueError(f"--seed {d['--seed']!r} is not an integer") from e
    xim.validate_seed(seed)
    if d != xim.expected_argv(seed):
        raise ValueError(f"argv must be exactly {' '.join(a for kv in xim.expected_argv(seed).items() for a in kv)} (any order); got {argv}")
    return seed


def prepare_output_dir(root, run_name):
    """Dedicated, collision-safe output directory: new, or existing with ONLY the driver's train_attempt0.log. Anything else (a previous run) is refused; the XFIN seed-0 run name is refused outright."""
    xim.seed_of_run(run_name)
    d = Path(root) / "runs_v2" / run_name
    if d.exists():
        extra = sorted(p.name for p in d.iterdir() if p.name not in ALLOWED_PREEXISTING)
        if extra:
            raise FileExistsError(f"{d} already contains {extra}; refusing to reuse an existing run directory")
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_clean_provenance(run=subprocess.run):
    """RuntimeError unless the index is empty, every tooling / code-dependency file is tracked and unmodified, and the XFIN and XF tooling files equal their commits."""
    git = xfw._git
    if git(["diff", "--cached", "--quiet"], run).returncode != 0:
        raise RuntimeError("the git index is not empty; refusing to run")
    st = git(["status", "--porcelain", "--", *xim.PROVENANCE_FILES], run)
    if st.returncode != 0 or st.stdout.strip():
        raise RuntimeError(f"provenance files are modified or untracked: {st.stdout.strip()[:200]!r}; refusing to run")
    for f in xim.PROVENANCE_FILES:
        if git(["ls-files", "--error-unmatch", f], run).returncode != 0:
            raise RuntimeError(f"{f} is not tracked by git; refusing to run")
    if git(["diff", "--quiet", xim.XF_COMMIT, "--", *xim.XF_IMPORTED_FILES], run).returncode != 0:
        raise RuntimeError(f"an XF file differs from commit {xim.XF_COMMIT[:8]}; refusing to run")
    if git(["diff", "--quiet", xim.XFIN_COMMIT, "--", *xim.XFIN_IMPORTED_FILES], run).returncode != 0:
        raise RuntimeError(f"an XFIN file differs from commit {xim.XFIN_COMMIT[:8]}; refusing to run")


def write_loss_config(d, seed, git_commit_full, runtime=None):
    """Exclusive creation ('x'): a second call, or a pre-existing file, raises FileExistsError."""
    xim.validate_seed(seed)
    rt = xfw.runtime_info() if runtime is None else runtime
    problems = xim.check_runtime(rt)
    if problems:
        raise RuntimeError(f"runtime record invalid: {problems}")
    rec = xim.loss_config(seed, git_commit_full, xim.recipe_args(tv.RECIPES), xim.sha256_file(ROOT / xim.UNTRACKED_DEPENDENCY), rt)
    with open(Path(d) / LOSS_CONFIG_FILE, "x") as f:
        json.dump(rec, f, indent=1)
    return rec


def build(a, device):
    """train_v2.build, UNCHANGED (xfw._ORIG_BUILD), then asserted: ATMarineTrunkIN with 1,952,768 parameters, 40 InstanceNorm2d / 0 BatchNorm2d, and the R3 loss exactly as for the AT-Trunk R3 runs."""
    assert a.recipe == xim.RECIPE_NAME and a.model == xim.MODEL and a.seed in xim.ALLOWED_SEEDS and a.run_name == xim.run_name(a.seed) and not a.resume, "unexpected arguments"
    assert a.amp is False and a.loss_in_autocast is False, "XFINMS is fp32 (amp and loss_in_autocast must be False)"
    assert abs(a.eta_min - 2e-8) < 1e-15, "eta_min must be 2e-8"
    model, crit = xfw._ORIG_BUILD(a, device)
    assert type(model) is ATMarineTrunkIN, "model is not ATMarineTrunkIN"
    assert sum(p.numel() for p in model.parameters()) == xim.EXPECTED_PARAMS, f"parameter count != {xim.EXPECTED_PARAMS:,}"
    assert norm_layers(model) == (N_NORM_LAYERS, 0), f"normalisation layers (InstanceNorm, BatchNorm) = {norm_layers(model)}, expected ({N_NORM_LAYERS}, 0)"
    assert type(crit) is tv.FinalOutputLoss and type(crit.crit) is tv.MARINELoss and type(crit.crit.perceptual) is tv.ZeroLoss, "the loss is not FinalOutputLoss(MARINELoss) with ZeroLoss as perceptual term"
    assert getattr(crit.crit, "ssim_loss", None), "SSIM loss disabled: the R3 loss would not be the R3 loss"
    return model, crit


def setup(argv):
    """All process-wide changes -> the seed. Idempotence is NOT supported (a second call raises)."""
    seed = validate_argv(argv)
    assert tv.build is xfw._ORIG_BUILD and tv.train_one_epoch is xfw._ORIG_TRAIN_ONE_EPOCH, "train_v2 was already patched"
    assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
    tv.MANIFESTS.pop("val_B")  # BEFORE any hashing or loading
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xim.ALLOWED_MANIFESTS]:  # val, val_clean: not needed with select_on=val_A, and verify_manifests() would hash them
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xim.ALLOWED_MANIFESTS), f"manifest table must hold exactly {xim.ALLOWED_MANIFESTS}, holds {sorted(tv.MANIFESTS)}"
    assert not any("val_b" in str(p).lower() for p, _, _ in tv.MANIFESTS.values())
    assert all(tv.MANIFESTS[k][2] == xim.EXPECTED_HASH_PREFIXES[k] for k in xim.ALLOWED_MANIFESTS), "expected hash prefixes differ from train_v2"
    assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
    assert xim.RECIPE_NAME not in tv.RECIPES, "recipe name already registered"
    r3_before = dict(tv.RECIPES["R3"])
    tv.RECIPES[xim.RECIPE_NAME] = xim.recipe_args(tv.RECIPES)
    r = tv.RECIPES[xim.RECIPE_NAME]
    assert tv.RECIPES["R3"] == r3_before, "recipe R3 was modified"
    assert r == xf.recipe_args(tv.RECIPES), "the XFINMS recipe must equal the XF recipe exactly"
    assert set(xim.recipe_diff(r3_before, r)) == set(xf.R3_DEVIATIONS) and r["amp"] is False and abs(r["eta_min"] - 2e-8) < 1e-15 and r["loss_in_autocast"] is False \
        and r["select_on"] == "val_A", "the recipe must differ from R3 only in amp and eta_min"
    assert xim.MODEL not in tv.MODEL_REGISTRY, "model name already registered"
    assert tv.MODEL_REGISTRY["at_trunk"]().__class__ is tv.ATMarineTrunk, "the existing at_trunk registry entry changed"
    tv.MODEL_REGISTRY[xim.MODEL] = lambda: ATMarineTrunkIN()  # registered BEFORE tv.main() builds its argparse choices from the registry
    m = tv.MODEL_REGISTRY[xim.MODEL]()  # CPU build, no data access
    assert type(m) is ATMarineTrunkIN and sum(p.numel() for p in m.parameters()) == xim.EXPECTED_PARAMS and norm_layers(m) == (N_NORM_LAYERS, 0), "unexpected model structure"
    del m
    tv.load_pairs = checked_loader(tv.load_pairs, xim.EXPECTED_N)  # refuses every split except train / val_A BEFORE touching the original loader
    assert not RECORDER.norms and RECORDER.path is None and RECORDER.epochs_written == 0, "recorder already used"
    tv.build = build
    tv.step = xfw.recording_step  # train_one_epoch looks `step` up in the train_v2 module at call time
    tv.train_one_epoch = xfw.recorded_train_one_epoch  # train_v2.main looks `train_one_epoch` up in the train_v2 module at call time
    return seed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        seed = validate_argv(argv)
        ensure_clean_provenance()
        xfw.ensure_original_dependency()
        setup(argv)
        d = prepare_output_dir(ROOT, xim.run_name(seed))
        write_loss_config(d, seed, xfw.git_full_commit())
        RECORDER.create_log(d)
    except (ValueError, FileExistsError, RuntimeError) as e:
        sys.exit(f"STOP: {e}")
    tv.main()  # parses sys.argv itself (identical to the validated argv)
    try:
        xfw.ensure_original_dependency()  # the R3 definition must still be the pinned one after the run
    except RuntimeError as e:
        sys.exit(f"STOP: after training: {e}")


if __name__ == "__main__":
    main()
