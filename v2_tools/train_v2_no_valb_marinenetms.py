"""MARINENETMS training wrapper around train_v2.main(): MARINE-Net (original architecture, 'bn', or with its 40 BatchNorm2d replaced by full-channel InstanceNorm2d, 'in') under the XF / XFIN fp32 recipe
(R3 loss, fp32, cosine floor 2e-8) for ONE seed in {0, 1, 2}. Sibling of v2_tools/train_v2_no_valb_armcinms.py and train_v2_no_valb_xf.py (NOT modified; the XF recorder, step and epoch wrappers, runtime and
dependency helpers are imported unchanged). NOT approved to run. Safeguards:
  * val_B is removed from train_v2.MANIFESTS BEFORE anything is hashed or loaded, then val and val_clean too; the table is asserted to hold exactly {train, val_A};
  * the command line is a strict whitelist: exactly one of the six configurations
        --recipe XF_R3_fp32_floor        --model marine_default    --seed {0,1,2} --run_name marinenet_bn_r3_fp32_s{0,1,2}
        --recipe XFIN_R3_fp32_floor_IN   --model marine_default_in --seed {0,1,2} --run_name marinenet_in_r3_fp32_s{0,1,2}
    (any order; a mixed variant / seed / run name, --resume, --smoke, --time_iters, --train_limit and --stop_after are refused);
  * train_v2.load_pairs is wrapped with the strict pair validator (only train and val_A, the FULL list validated) BEFORE the original loader can run;
  * the output directory must be new or contain only the driver's train_attempt0.log (any other existing content, including every existing run's directory, is refused); the loss config and the clip log are
    created exclusively; no existing run directory is ever reused;
  * the tooling and code dependencies must be tracked and unmodified, the index empty, the XF / XFIN / XFINMS files equal to their commits, before anything is processed;
  * original_repo/marine_net_model.py (git-ignored) must have the PINNED SHA-256 before the run and again after tv.main() returns;
  * the model name marine_default_in is registered in train_v2.MODEL_REGISTRY in this process only (train_v2.py is not edited); the existing marine_default entry is asserted unchanged and is used as is for 'bn';
    train_v2.build is called as is (XF's _ORIG_BUILD) and its result is asserted: the exact MARINE-Net class of the variant, 2,609,401 trainable parameters, 40 BatchNorm2d / 0 InstanceNorm2d ('bn') or
    40 InstanceNorm2d / 0 BatchNorm2d ('in'), and the FULL MARINELoss (auxiliary CNN / INR terms ACTIVE, NOT wrapped in FinalOutputLoss) with ZeroLoss as perceptual term;
  * train_v2.step -> the XF recording_step, train_v2.train_one_epoch -> the XF recorded_train_one_epoch; no training arithmetic changes.
Importing this module has no side effects; everything happens in main(). train_v2.py, the evaluator and all models / losses are not modified.
Usage (NOT approved to run): python v2_tools/train_v2_no_valb_marinenetms.py --recipe XF_R3_fp32_floor --model marine_default --seed 0 --run_name marinenet_bn_r3_fp32_s0"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.marine_net_in import MARINENetIN, model_problems  # noqa: E402
from v2_tools import marinenetms_config as mc  # noqa: E402
from v2_tools import train_v2_no_valb_xf as xfw  # noqa: E402  (imported unchanged: RECORDER, recording_step, recorded_train_one_epoch, runtime_info, ensure_original_dependency, git_full_commit, _git)
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

LOSS_CONFIG_FILE = mc.LOSS_CONFIG_FILE
ALLOWED_PREEXISTING = {"train_attempt0.log"}  # created by the single-run driver before it starts this wrapper
OPTION_KEYS = ("--recipe", "--model", "--seed", "--run_name")
RECORDER = xfw.RECORDER


def validate_argv(argv):
    """-> (variant, seed). Exactly the four option/value pairs (any order) of one of the six allowed configurations; anything else raises ValueError. Pure."""
    argv = list(argv)
    if len(argv) != 2 * len(OPTION_KEYS) or set(argv[0::2]) != set(OPTION_KEYS) or len(set(argv[0::2])) != len(OPTION_KEYS):
        raise ValueError(f"argv must be exactly the four option/value pairs {' '.join(OPTION_KEYS)} (any order); got {argv}")
    d = dict(zip(argv[0::2], argv[1::2]))
    variant = mc.variant_of_model(d["--model"])
    try:
        seed = int(d["--seed"])
    except ValueError as e:
        raise ValueError(f"--seed {d['--seed']!r} is not an integer") from e
    mc.validate_seed(seed)
    if d != mc.expected_argv(variant, seed):
        raise ValueError(f"argv must be exactly {' '.join(a for kv in mc.expected_argv(variant, seed).items() for a in kv)} (any order); got {argv}")
    return variant, seed


def prepare_output_dir(root, run_name):
    """Dedicated, collision-safe output directory: new, or existing with ONLY the driver's train_attempt0.log. Anything else (a previous run) is refused; every run name except the six dedicated ones is refused outright."""
    mc.parse_run_name(run_name)
    d = Path(root) / "runs_v2" / run_name
    if d.exists():
        extra = sorted(p.name for p in d.iterdir() if p.name not in ALLOWED_PREEXISTING)
        if extra:
            raise FileExistsError(f"{d} already contains {extra}; refusing to reuse an existing run directory")
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_clean_provenance(run=subprocess.run):
    """RuntimeError unless the index is empty, every tooling / code-dependency file is tracked and unmodified, and the XF, XFIN and XFINMS files equal their commits."""
    git = xfw._git
    if git(["diff", "--cached", "--quiet"], run).returncode != 0:
        raise RuntimeError("the git index is not empty; refusing to run")
    st = git(["status", "--porcelain", "--", *mc.PROVENANCE_FILES], run)
    if st.returncode != 0 or st.stdout.strip():
        raise RuntimeError(f"provenance files are modified or untracked: {st.stdout.strip()[:200]!r}; refusing to run")
    for f in mc.PROVENANCE_FILES:
        if git(["ls-files", "--error-unmatch", f], run).returncode != 0:
            raise RuntimeError(f"{f} is not tracked by git; refusing to run")
    for label, commit, files in mc.PINNED_COMMITS:
        if git(["diff", "--quiet", commit, "--", *files], run).returncode != 0:
            raise RuntimeError(f"an {label} file differs from commit {commit[:8]}; refusing to run")


def write_loss_config(d, variant, seed, git_commit_full, runtime=None):
    """Exclusive creation ('x'): a second call, or a pre-existing file, raises FileExistsError."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    rt = xfw.runtime_info() if runtime is None else runtime
    problems = mc.check_runtime(rt)
    if problems:
        raise RuntimeError(f"runtime record invalid: {problems}")
    rec = mc.loss_config(variant, seed, git_commit_full, mc.recipe_args(tv.RECIPES), mc.sha256_file(ROOT / mc.UNTRACKED_DEPENDENCY), rt)
    with open(Path(d) / LOSS_CONFIG_FILE, "x") as f:
        json.dump(rec, f, indent=1)
    return rec


def build(a, device):
    """train_v2.build, UNCHANGED (xfw._ORIG_BUILD), then asserted: the exact MARINE-Net class of the variant (structure, parameter and buffer counts) and the FULL R3 loss (auxiliary CNN / INR terms active)."""
    variant = mc.variant_of_model(a.model)
    assert a.recipe == mc.RECIPE_NAME[variant] and a.seed in mc.ALLOWED_SEEDS and a.run_name == mc.run_name(variant, a.seed) and not a.resume, "unexpected arguments"
    assert a.amp is False and a.loss_in_autocast is False, "MARINENETMS is fp32 (amp and loss_in_autocast must be False)"
    assert abs(a.eta_min - 2e-8) < 1e-15, "eta_min must be 2e-8"
    model, crit = xfw._ORIG_BUILD(a, device)
    probs = model_problems(model, variant)
    assert not probs, f"unexpected model structure: {probs}"
    assert type(crit) is tv.MARINELoss and type(crit.perceptual) is tv.ZeroLoss, "the loss is not the full MARINELoss with ZeroLoss as perceptual term (auxiliary CNN / INR terms must stay active)"
    assert getattr(crit, "ssim_loss", None), "SSIM loss disabled: the R3 loss would not be the R3 loss"
    return model, crit


def setup(argv):
    """All process-wide changes -> (variant, seed). Idempotence is NOT supported (a second call raises)."""
    variant, seed = validate_argv(argv)
    assert tv.build is xfw._ORIG_BUILD and tv.train_one_epoch is xfw._ORIG_TRAIN_ONE_EPOCH, "train_v2 was already patched"
    assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
    tv.MANIFESTS.pop("val_B")  # BEFORE any hashing or loading
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in mc.ALLOWED_MANIFESTS]:  # val, val_clean: not needed with select_on=val_A, and verify_manifests() would hash them
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(mc.ALLOWED_MANIFESTS), f"manifest table must hold exactly {mc.ALLOWED_MANIFESTS}, holds {sorted(tv.MANIFESTS)}"
    assert not any("val_b" in str(p).lower() for p, _, _ in tv.MANIFESTS.values())
    assert all(tv.MANIFESTS[k][2] == mc.EXPECTED_HASH_PREFIXES[k] for k in mc.ALLOWED_MANIFESTS), "expected hash prefixes differ from train_v2"
    assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
    rname = mc.RECIPE_NAME[variant]
    assert rname not in tv.RECIPES, "recipe name already registered"
    r3_before = dict(tv.RECIPES["R3"])
    tv.RECIPES[rname] = mc.recipe_args(tv.RECIPES)
    r = tv.RECIPES[rname]
    assert tv.RECIPES["R3"] == r3_before, "recipe R3 was modified"
    assert r == xf.recipe_args(tv.RECIPES), "the MARINENETMS recipe must equal the XF recipe exactly"
    assert set(mc.recipe_diff(r3_before, r)) == set(xf.R3_DEVIATIONS) and r["amp"] is False and abs(r["eta_min"] - 2e-8) < 1e-15 and r["loss_in_autocast"] is False \
        and r["select_on"] == "val_A", "the recipe must differ from R3 only in amp and eta_min"
    assert tv.MODEL_REGISTRY["marine_default"]().__class__ is tv.MARINENet, "the existing marine_default registry entry changed"
    if variant == "in":
        assert mc.MODEL["in"] not in tv.MODEL_REGISTRY, "model name already registered"
        tv.MODEL_REGISTRY[mc.MODEL["in"]] = lambda: MARINENetIN()  # registered BEFORE tv.main() builds its argparse choices from the registry
    m = tv.MODEL_REGISTRY[mc.MODEL[variant]]()  # CPU build, no data access
    assert not model_problems(m, variant), f"unexpected model structure: {model_problems(m, variant)}"
    del m
    tv.load_pairs = checked_loader(tv.load_pairs, mc.EXPECTED_N)  # refuses every split except train / val_A BEFORE touching the original loader
    assert not RECORDER.norms and RECORDER.path is None and RECORDER.epochs_written == 0, "recorder already used"
    tv.build = build
    tv.step = xfw.recording_step  # train_one_epoch looks `step` up in the train_v2 module at call time
    tv.train_one_epoch = xfw.recorded_train_one_epoch  # train_v2.main looks `train_one_epoch` up in the train_v2 module at call time
    return variant, seed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        variant, seed = validate_argv(argv)
        ensure_clean_provenance()
        xfw.ensure_original_dependency()
        setup(argv)
        d = prepare_output_dir(ROOT, mc.run_name(variant, seed))
        write_loss_config(d, variant, seed, xfw.git_full_commit())
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
