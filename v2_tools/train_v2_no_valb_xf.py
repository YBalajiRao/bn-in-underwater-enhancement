"""XF training wrapper around train_v2.main(): AT-Trunk (unchanged: in-model output clamp, identity initialisation, 1,952,768 parameters) trained with the R3 loss (original_repo MARINELoss, unchanged,
built by the UNCHANGED train_v2.build exactly as for the AT-Trunk R3 runs) under Recipe R3 EXCEPT amp False (fp32, no GradScaler) and cosine floor eta_min 2e-8 (R3: AMP, 1e-6). Seed 0 only.
Safeguards:
  * val_B is removed from train_v2.MANIFESTS BEFORE anything is hashed or loaded, then val and val_clean too; the table is asserted to hold exactly {train, val_A};
  * the command line is a strict whitelist: exactly --recipe XF_R3_fp32_floor --model at_trunk --seed 0 --run_name xf_attrunk_r3_fp32_s0 (no --resume, --smoke, --time_iters, --train_limit, --stop_after);
  * train_v2.load_pairs is wrapped with the strict pair validator (only train and val_A, the FULL list validated);
  * the output directory must be new or contain only the driver's train_attempt0.log (train_v2.main uses mkdir(exist_ok=True) and APPENDS to history.jsonl); xf_loss_config.json and xf_clip_log.jsonl are created exclusively;
  * the tooling and code dependencies must be tracked and unmodified, and the index empty, before anything is processed;
  * original_repo/marine_net_model.py (git-ignored) must have the PINNED SHA-256 before the run, and again after tv.main() returns (otherwise exit non-zero);
  * the model and the loss are NOT changed: train_v2.build is called as is and its result is asserted (ATMarineTrunk, 1,952,768 parameters, FinalOutputLoss around MARINELoss with ZeroLoss as perceptual term);
  * train_v2.step is replaced by recording_step = the reviewed XR guarded_step (raises BEFORE optimizer.step on a non-finite gradient norm; otherwise identical to train_v2.step for fp32) plus a recorder that
    appends the pre-clip norm it already returns to a list. train_v2.train_one_epoch is wrapped only to flush one line per epoch to xf_clip_log.jsonl and to cross-check the recorder against the epoch result;
    no training arithmetic changes. The convolution / matmul tf32 flags are NOT touched; their values are recorded in xf_loss_config.json under 'runtime'.
Importing this module has no side effects; everything happens in main(). train_v2.py, the evaluator and all models / losses are not modified.
Usage (NOT approved to run): python v2_tools/train_v2_no_valb_xf.py --recipe XF_R3_fp32_floor --model at_trunk --seed 0 --run_name xf_attrunk_r3_fp32_s0"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools.train_v2_no_valb_xr import _TRAIN_V2_STEP, guarded_step  # noqa: E402  (reviewed XR step guard; _TRAIN_V2_STEP = the original train_v2.step, kept for the tests)
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

LOSS_CONFIG_FILE = xf.LOSS_CONFIG_FILE
CLIP_LOG_FILE = xf.CLIP_LOG_FILE
ALLOWED_PREEXISTING = {"train_attempt0.log"}  # created by the single-run driver before it starts this wrapper
EXPECTED_ARGV = {"--recipe": xf.RECIPE_NAME, "--model": xf.MODEL, "--seed": str(xf.SEED), "--run_name": xf.RUN_NAME}
_ORIG_BUILD = tv.build  # train_v2.build, called unchanged by build() below
_ORIG_TRAIN_ONE_EPOCH = tv.train_one_epoch


def validate_argv(argv):
    """Exactly the four expected option/value pairs (any order); anything else raises ValueError. Pure."""
    argv = list(argv)
    if len(argv) != 2 * len(EXPECTED_ARGV) or dict(zip(argv[0::2], argv[1::2])) != EXPECTED_ARGV or len(set(argv[0::2])) != len(EXPECTED_ARGV):
        raise ValueError(f"argv must be exactly {' '.join(a for kv in EXPECTED_ARGV.items() for a in kv)} (any order); got {argv}")
    return argv


def prepare_output_dir(root, run_name):
    """Dedicated, collision-safe output directory: new, or existing with ONLY the driver's train_attempt0.log. Anything else (a previous run) is refused."""
    xf.validate_run_name(run_name)
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
    st = _git(["status", "--porcelain", "--", *xf.PROVENANCE_FILES], run)
    if st.returncode != 0 or st.stdout.strip():
        raise RuntimeError(f"provenance files are modified or untracked: {st.stdout.strip()[:200]!r}; refusing to run")
    for f in xf.PROVENANCE_FILES:
        if _git(["ls-files", "--error-unmatch", f], run).returncode != 0:
            raise RuntimeError(f"{f} is not tracked by git; refusing to run")


def ensure_original_dependency():
    """RuntimeError unless original_repo/marine_net_model.py has the pinned SHA-256 (fail closed: unreadable = refused)."""
    problems = xf.verify_original_dependency()
    if problems:
        raise RuntimeError("; ".join(problems))


def runtime_info():
    """The flags that decide whether the 'fp32' training convolutions / matmuls use tf32. READ ONLY: this wrapper never sets them."""
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "torch": torch.__version__}


def write_loss_config(d, git_commit_full, runtime=None):
    """Exclusive creation ('x'): a second call, or a pre-existing file, raises FileExistsError."""
    rt = runtime_info() if runtime is None else runtime
    problems = xf.check_runtime(rt)
    if problems:
        raise RuntimeError(f"runtime record invalid: {problems}")
    rec = xf.loss_config(git_commit_full, xf.recipe_args(tv.RECIPES), xf.sha256_file(ROOT / xf.UNTRACKED_DEPENDENCY), rt)
    with open(Path(d) / LOSS_CONFIG_FILE, "x") as f:
        json.dump(rec, f, indent=1)
    return rec


class ClipRecorder:
    """Observation only. add() stores the pre-clip norm that the step already computed; end_epoch() writes one line to the exclusively created clip log and cross-checks the epoch result."""

    def __init__(self):
        self.norms, self.clip, self.path, self.epochs_written = [], None, None, 0

    def create_log(self, d):
        self.path = Path(d) / CLIP_LOG_FILE
        with open(self.path, "x"):
            pass
        return self.path

    def add(self, gn, clip):
        gn = float(gn)
        if gn != gn or gn in (float("inf"), float("-inf")):
            raise FloatingPointError(f"non-finite gradient norm {gn} reached the recorder")
        if self.clip is not None and clip != self.clip:
            raise RuntimeError(f"clip value changed during an epoch ({self.clip} -> {clip})")
        self.clip = clip
        self.norms.append(gn)

    def begin_epoch(self):
        if self.norms:
            raise RuntimeError("the previous epoch's gradient norms were not flushed")
        self.clip = None

    def end_epoch(self, res):
        if self.path is None:
            raise RuntimeError("the clip log was not created (create_log must run before training)")
        st = xf.summarize_norms(self.norms, self.clip)
        if st["steps"] != res["steps"] or abs(st["gnorm_mean"] - res["gnorm_mean"]) > 1e-9 or abs(st["gnorm_max"] - res["gnorm_max"]) > 1e-9:
            raise RuntimeError(f"recorder disagrees with the epoch result (steps {st['steps']} vs {res['steps']}, mean {st['gnorm_mean']} vs {res['gnorm_mean']})")
        line = {"epoch": self.epochs_written, **st}
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")
        self.epochs_written += 1
        self.norms = []
        return line


RECORDER = ClipRecorder()


def recording_step(model, opt, scaler, a):
    """The reviewed XR guarded_step (identical to train_v2.step for fp32, but a non-finite norm raises BEFORE opt.step) plus the recorder. Returns the same pre-clip norm."""
    gn = guarded_step(model, opt, scaler, a)
    RECORDER.add(gn, a.clip)
    return gn


def recorded_train_one_epoch(*args, **kwargs):
    """train_v2.train_one_epoch unchanged, bracketed by the recorder's epoch begin / end."""
    RECORDER.begin_epoch()
    res = _ORIG_TRAIN_ONE_EPOCH(*args, **kwargs)
    RECORDER.end_epoch(res)
    return res


def build(a, device):
    """train_v2.build, UNCHANGED, then asserted: AT-Trunk with 1,952,768 parameters and the R3 loss built exactly as for the AT-Trunk R3 runs."""
    assert a.recipe == xf.RECIPE_NAME and a.model == xf.MODEL and a.seed == xf.SEED and a.run_name == xf.RUN_NAME and not a.resume, "unexpected arguments"
    assert a.amp is False and a.loss_in_autocast is False, "XF is fp32 (amp and loss_in_autocast must be False)"
    assert abs(a.eta_min - 2e-8) < 1e-15, "eta_min must be 2e-8"
    model, crit = _ORIG_BUILD(a, device)
    assert type(model) is tv.ATMarineTrunk, "model is not ATMarineTrunk"
    assert sum(p.numel() for p in model.parameters()) == xf.EXPECTED_PARAMS, f"parameter count != {xf.EXPECTED_PARAMS:,}"
    assert type(crit) is tv.FinalOutputLoss and type(crit.crit) is tv.MARINELoss and type(crit.crit.perceptual) is tv.ZeroLoss, "the loss is not FinalOutputLoss(MARINELoss) with ZeroLoss as perceptual term"
    assert getattr(crit.crit, "ssim_loss", None), "SSIM loss disabled: the R3 loss would not be the R3 loss"
    return model, crit


def setup(argv):
    """All process-wide changes. Idempotence is NOT supported (a second call raises)."""
    validate_argv(argv)
    assert tv.build is _ORIG_BUILD and tv.train_one_epoch is _ORIG_TRAIN_ONE_EPOCH, "train_v2 was already patched"
    assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
    tv.MANIFESTS.pop("val_B")  # BEFORE any hashing or loading
    assert "val_B" not in tv.MANIFESTS
    for k in [k for k in tv.MANIFESTS if k not in xf.ALLOWED_MANIFESTS]:  # val, val_clean: not needed with select_on=val_A, and verify_manifests() would hash them
        del tv.MANIFESTS[k]
    assert sorted(tv.MANIFESTS) == sorted(xf.ALLOWED_MANIFESTS), f"manifest table must hold exactly {xf.ALLOWED_MANIFESTS}, holds {sorted(tv.MANIFESTS)}"
    assert not any("val_b" in str(p).lower() for p, _, _ in tv.MANIFESTS.values())
    assert all(tv.MANIFESTS[k][2] == xf.EXPECTED_HASH_PREFIXES[k] for k in xf.ALLOWED_MANIFESTS), "expected hash prefixes differ from train_v2"
    assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
    assert xf.RECIPE_NAME not in tv.RECIPES, "recipe name already registered"
    r3_before = dict(tv.RECIPES[xf.BASELINE_RECIPE])
    tv.RECIPES[xf.RECIPE_NAME] = xf.recipe_args(tv.RECIPES)
    r = tv.RECIPES[xf.RECIPE_NAME]
    assert tv.RECIPES[xf.BASELINE_RECIPE] == r3_before, "recipe R3 was modified"
    assert set(xf.recipe_diff(r3_before, r)) == set(xf.R3_DEVIATIONS) and r["amp"] is False and abs(r["eta_min"] - 2e-8) < 1e-15 and r["loss_in_autocast"] is False \
        and r["select_on"] == "val_A", "XF recipe must differ from R3 only in amp and eta_min"
    m = tv.MODEL_REGISTRY[xf.MODEL]()  # CPU build, no data access
    assert type(m) is tv.ATMarineTrunk and sum(p.numel() for p in m.parameters()) == xf.EXPECTED_PARAMS, f"parameter count != {xf.EXPECTED_PARAMS:,}"
    del m
    tv.load_pairs = checked_loader(tv.load_pairs, xf.EXPECTED_N)  # refuses every split except train / val_A BEFORE touching the original loader
    assert not RECORDER.norms and RECORDER.path is None and RECORDER.epochs_written == 0, "recorder already used"
    tv.build = build
    tv.step = recording_step  # train_one_epoch looks `step` up in the train_v2 module at call time
    tv.train_one_epoch = recorded_train_one_epoch  # train_v2.main looks `train_one_epoch` up in the train_v2 module at call time


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        validate_argv(argv)
        ensure_clean_provenance()
        ensure_original_dependency()
        setup(argv)
        d = prepare_output_dir(ROOT, xf.RUN_NAME)
        write_loss_config(d, git_full_commit())
        RECORDER.create_log(d)
    except (ValueError, FileExistsError, RuntimeError) as e:
        sys.exit(f"STOP: {e}")
    tv.main()  # parses sys.argv itself (identical to the validated argv)
    try:
        ensure_original_dependency()  # the R3 definition must still be the pinned one after the run
    except RuntimeError as e:
        sys.exit(f"STOP: after training: {e}")


if __name__ == "__main__":
    main()
