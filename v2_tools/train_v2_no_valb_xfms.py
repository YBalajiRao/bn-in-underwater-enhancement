"""XFMS training wrapper around train_v2.main(): the XF experiment (AT-Trunk, R3 loss, fp32, cosine floor 2e-8) for seeds 1 and 2 only. Sibling of v2_tools/train_v2_no_valb_xf.py (NOT modified): identical
safeguards, identical model / loss / recipe; the XF gradient-clip recorder, step guard and helpers are imported from it unchanged. Only the seed and the seed-dependent run name differ.
Safeguards (as XF, with the seed parameterised):
  * val_B is removed from train_v2.MANIFESTS BEFORE anything is hashed or loaded, then val and val_clean too; the table is asserted to hold exactly {train, val_A};
  * the command line is a strict whitelist: exactly one of
        --recipe XF_R3_fp32_floor --model at_trunk --seed 1 --run_name xf_attrunk_r3_fp32_s1
        --recipe XF_R3_fp32_floor --model at_trunk --seed 2 --run_name xf_attrunk_r3_fp32_s2
    (any order; no seed 0, no mixed seed / run name, no --resume, --smoke, --time_iters, --train_limit, --stop_after);
  * train_v2.load_pairs is wrapped with the strict pair validator (only train and val_A, the FULL list validated) BEFORE the original loader can run;
  * the output directory must be new or contain only the driver's train_attempt0.log (never the existing XF seed-0 directory: its name is refused); the loss config and the clip log are created exclusively;
  * the tooling and code dependencies must be tracked and unmodified, the index empty, and the XF files equal to the XF commit, before anything is processed;
  * original_repo/marine_net_model.py (git-ignored) must have the PINNED SHA-256 before the run and again after tv.main() returns;
  * train_v2.build is called as is (XF's _ORIG_BUILD) and its result is asserted: ATMarineTrunk, 1,952,768 parameters, FinalOutputLoss around MARINELoss with ZeroLoss as perceptual term;
  * train_v2.step -> the XF recording_step, train_v2.train_one_epoch -> the XF recorded_train_one_epoch; no training arithmetic changes.
Importing this module has no side effects; everything happens in main(). train_v2.py, the evaluator and all models / losses are not modified.
Usage (NOT approved to run): python v2_tools/train_v2_no_valb_xfms.py --recipe XF_R3_fp32_floor --model at_trunk --seed 1 --run_name xf_attrunk_r3_fp32_s1"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from v2_tools import train_v2_no_valb_xf as xfw  # noqa: E402  (imported unchanged: RECORDER, recording_step, recorded_train_one_epoch, runtime_info, ensure_original_dependency, git_full_commit, _git)
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools import xfms_config as xm  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

LOSS_CONFIG_FILE = xm.LOSS_CONFIG_FILE
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
    xm.validate_seed(seed)
    if d != xm.expected_argv(seed):
        raise ValueError(f"argv must be exactly {' '.join(a for kv in xm.expected_argv(seed).items() for a in kv)} (any order); got {argv}")
    return seed


def prepare_output_dir(root, run_name):
    """Dedicated, collision-safe output directory: new, or existing with ONLY the driver's train_attempt0.log. Anything else (a previous run) is refused; the XF seed-0 run name is refused outright."""
    xm.seed_of_run(run_name)
    d = Path(root) / "runs_v2" / run_name
    if d.exists():
        extra = sorted(p.name for p in d.iterdir() if p.name not in ALLOWED_PREEXISTING)
        if extra:
            raise FileExistsError(f"{d} already contains {extra}; refusing to reuse an existing run directory")
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_clean_provenance(run=subprocess.run):
    """RuntimeError unless the index is empty, every tooling / code-dependency file is tracked and unmodified, and the XF tooling files equal their commit."""
    git = xfw._git
    if git(["diff", "--cached", "--quiet"], run).returncode != 0:
        raise RuntimeError("the git index is not empty; refusing to run")
    st = git(["status", "--porcelain", "--", *xm.PROVENANCE_FILES], run)
    if st.returncode != 0 or st.stdout.strip():
        raise RuntimeError(f"provenance files are modified or untracked: {st.stdout.strip()[:200]!r}; refusing to run")
    for f in xm.PROVENANCE_FILES:
        if git(["ls-files", "--error-unmatch", f], run).returncode != 0:
            raise RuntimeError(f"{f} is not tracked by git; refusing to run")
    if git(["diff", "--quiet", xm.XF_COMMIT, "--", *xm.XF_IMPORTED_FILES], run).returncode != 0:
        raise RuntimeError(f"an XF file differs from commit {xm.XF_COMMIT[:8]}; refusing to run")


def write_loss_config(d, seed, git_commit_full, runtime=None):
    """Exclusive creation ('x'): a second call, or a pre-existing file, raises FileExistsError."""
    xm.validate_seed(seed)
    rt = xfw.runtime_info() if runtime is None else runtime
    problems = xm.check_runtime(rt)
    if problems:
        raise RuntimeError(f"runtime record invalid: {problems}")
    rec = xm.loss_config(seed, git_commit_full, xm.recipe_args(tv.RECIPES), xm.sha256_file(ROOT / xm.UNTRACKED_DEPENDENCY), rt)
    with open(Path(d) / LOSS_CONFIG_FILE, "x") as f:
        json.dump(rec, f, indent=1)
    return rec


def build(a, device):
    """train_v2.build, UNCHANGED (xfw._ORIG_BUILD), then asserted: AT-Trunk with 1,952,768 parameters and the R3 loss built exactly as for the AT-Trunk R3 runs. Identical to the XF build apart from the seed / run-name assertion."""
    assert a.recipe == xm.RECIPE_NAME and a.model == xm.MODEL and a.seed in xm.ALLOWED_SEEDS and a.run_name == xm.run_name(a.seed) and not a.resume, "unexpected arguments"
    assert a.amp is False and a.loss_in_autocast is False, "XFMS is fp32 (amp and loss_in_autocast must be False)"
    assert abs(a.eta_min - 2e-8) < 1e-15, "eta_min must be 2e-8"
    model, crit = xfw._ORIG_BUILD(a, device)
    assert type(model) is tv.ATMarineTrunk, "model is not ATMarineTrunk"
    assert sum(p.numel() for p in model.parameters()) == xm.EXPECTED_PARAMS, f"parameter count != {xm.EXPECTED_PARAMS:,}"
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
    for k in [k for k in tv.MANIFESTS if k not in xm.ALLOWED_MANIFESTS]:  # val, val_clean: not needed with select_on=val_A, and verify_manifests() would hash them
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xm.ALLOWED_MANIFESTS), f"manifest table must hold exactly {xm.ALLOWED_MANIFESTS}, holds {sorted(tv.MANIFESTS)}"
    assert not any("val_b" in str(p).lower() for p, _, _ in tv.MANIFESTS.values())
    assert all(tv.MANIFESTS[k][2] == xm.EXPECTED_HASH_PREFIXES[k] for k in xm.ALLOWED_MANIFESTS), "expected hash prefixes differ from train_v2"
    assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
    assert xm.RECIPE_NAME not in tv.RECIPES, "recipe name already registered"
    r3_before = dict(tv.RECIPES[xm.BASELINE_RECIPE])
    tv.RECIPES[xm.RECIPE_NAME] = xm.recipe_args(tv.RECIPES)
    r = tv.RECIPES[xm.RECIPE_NAME]
    assert tv.RECIPES[xm.BASELINE_RECIPE] == r3_before, "recipe R3 was modified"
    assert r == xf.recipe_args(tv.RECIPES), "the XFMS recipe must equal the XF recipe exactly"
    assert set(xm.recipe_diff(r3_before, r)) == set(xf.R3_DEVIATIONS) and r["amp"] is False and abs(r["eta_min"] - 2e-8) < 1e-15 and r["loss_in_autocast"] is False \
        and r["select_on"] == "val_A", "the recipe must differ from R3 only in amp and eta_min"
    m = tv.MODEL_REGISTRY[xm.MODEL]()  # CPU build, no data access
    assert type(m) is tv.ATMarineTrunk and sum(p.numel() for p in m.parameters()) == xm.EXPECTED_PARAMS, f"parameter count != {xm.EXPECTED_PARAMS:,}"
    del m
    tv.load_pairs = checked_loader(tv.load_pairs, xm.EXPECTED_N)  # refuses every split except train / val_A BEFORE touching the original loader
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
        d = prepare_output_dir(ROOT, xm.run_name(seed))
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
