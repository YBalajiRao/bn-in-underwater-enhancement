"""Single-run gated driver for experiment XFINMS: XFIN (AT-Trunk with its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d + R3 loss + fp32 + cosine floor eta_min 2e-8) for seed 1 or seed 2,
the matched partners of the XFMS baselines. Sibling of v2_tools/run_xfin_single.py and run_xfms_single.py (NOT modified; the XF driver's helpers are imported). NOT approved to run.
One invocation = ONE seed = ONE run: `--seed {1,2}` (seed 0 is the existing XFIN run and is refused). Run names are fixed (xfin_attrunk_in_r3_fp32_s1 / _s2), so the existing XFIN seed-0 directory, log and
gate files can never be touched. Sequence: preflight -> ONE training process -> artifact + metadata checks -> training_complete -> gate (bounded by a timeout) -> gate_passed -> all_done.
Exactly one launch per invocation: no retry, no resume (--resume is never passed), no second run; ANY failure ends the driver with exit 1 and a STOP record; nothing is cleaned up. Seed 2 must be a separate,
separately approved invocation, started only after seed 1 has finished (two drivers must never run at once).
Safeguards (as the XFIN driver, with the seed parameterised):
  * exclusive creation of the driver log and of the preflight log (timestamp + process id); the output directory must not exist;
  * provenance preflight: the index is empty; the five new tooling files, the XFMS files (the paired rule is imported from them), the XFIN files, the imported XF / XR / WWE helper files and the other code
    dependencies are TRACKED and unmodified; the XF commit c943032 and the XFIN commit 5f04019 are ancestors of HEAD; the XF files equal the XF commit and the XFIN files equal the XFIN commit;
    external/WWE-UIE is clean at its pinned commit;
  * R3 provenance (fail closed): original_repo/marine_net_model.py is git-ignored, so its SHA-256 is PINNED and verified in the preflight, on EVERY poll while training runs (a change kills the process
    tree), once more after training, and by the gate;
  * recipe preflight: the XFIN recipe equals the XF recipe (R3 except amp False and eta_min 2e-8); ATMarineTrunkIN builds on the CPU with 1,952,768 parameters, 40 InstanceNorm2d and 0 BatchNorm2d layers;
    the original AT-Trunk has 40 BatchNorm layers (no data access);
  * the existing XFIN seed-0 result.json (train + val_A manifests only) must exist, have the PINNED SHA-256 (xfinms_config.XFIN_S0_RESULT_SHA256; the gate re-checks it) and describe XFIN seed 0 (it is the
    args-pairing reference); the XF and AMP result.json files are never opened;
  * train and Val-A manifests only (hash + strict pair validation, every other manifest key dropped first); result.json must hold exactly those two hashes;
  * fp32 history scan: ANY non-finite loss step, ANY non-finite-gradient step, a non-finite loss / gradient norm / Val-A value stops the run;
  * every external command is bounded (git calls, the GPU probe, the gate 30 min, training by the wall limit); a preflight check that raises is a STOP;
  * Windows fail-closed process cleanup: the training process runs inside a kill-on-close Job Object; if the job cannot be created or the process cannot be assigned, nothing is launched / the just-started
    tree is killed; on any failure, non-finite value, dependency change or timeout the whole tree is killed (taskkill /T /F only while alive + TerminateJobObject);
  * wall-time limit 10 h (a safeguard, not a duration target; ESTIMATE of about 5.7 h extrapolated from the recorded XFIN seed-0 run_seconds 20,442.6, NOT a measurement for these seeds);
    the gate runs under a 30 min timeout (tree killed on timeout; the run is then NOT validated);
  * result.json must describe this run (seed, run name, recipe, model, params, fp32, eta_min, no resume, exact manifest scope) and the clip log must exist before training_complete is logged.
Events: plan, preflight_ok*, preflight_done, start, tree_guard, training_complete (NOT a success marker), gate_passed, all_done (the only success event). STOP on any failure.
Usage (NOT approved to run): python v2_tools/run_xfinms_single.py --seed {1,2} [--preflight_only]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import run_wwe_single as rw  # noqa: E402
from v2_tools import run_xf_single as rf  # noqa: E402
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools import xfin_config as xi  # noqa: E402
from v2_tools import xfinms_config as xim  # noqa: E402
from v2_tools import xfinms_gate as xg  # noqa: E402
from v2_tools.run_wwe_single import kill_tree, make_kill_on_close_job, popen_kwargs, run_bounded  # noqa: E402

WRAPPER, GATE = "v2_tools/train_v2_no_valb_xfinms.py", "v2_tools/xfinms_gate.py"
ARTIFACTS = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xim.LOSS_CONFIG_FILE, xim.CLIP_LOG_FILE, "train_attempt0.log"]
GATES = list(xg.CHECK_NAMES)
PROVENANCE_FILES = list(xim.PROVENANCE_FILES)
MAX_WALL_S = rf.MAX_WALL_S  # 10 h safeguard, not a target
POLL_S = rf.POLL_S
GATE_TIMEOUT_S = rf.GATE_TIMEOUT_S
REQUIRE_JOB = os.name == "nt"  # fail closed on Windows if the Job Object cannot be created / assigned (tests may force it on other platforms)

scan_history_fp32 = rf.scan_history_fp32
git = rf.git
check_gpu_free = rf.check_gpu_free
check_disk_and_cache, check_manifests = rf.check_disk_and_cache, rf.check_manifests


def commands(seed):
    """-> (training command, gate command) for this seed; both name the seed's dedicated run."""
    run = xim.run_name(seed)
    return ([sys.executable, WRAPPER, "--recipe", xim.RECIPE_NAME, "--model", xim.MODEL, "--seed", str(seed), "--run_name", run], [sys.executable, GATE, "--run", run])


# ----------------------------------------------------------------------------- preflight checks
def check_git():
    out = []
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("diff", "--cached", "--quiet").returncode != 0:
        out.append("the index is not empty")
    for f in PROVENANCE_FILES:
        if git("ls-files", "--error-unmatch", f).returncode != 0:
            out.append(f"{f} is not tracked by git")
    if git("status", "--porcelain", "--", *PROVENANCE_FILES).stdout.strip():
        out.append("provenance files have uncommitted changes (modified or untracked)")
    for label, commit, files in (("XF", xim.XF_COMMIT, xim.XF_IMPORTED_FILES), ("XFIN", xim.XFIN_COMMIT, xim.XFIN_IMPORTED_FILES)):
        if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
            out.append(f"{commit[:8]} (the {label} tooling commit) is not an ancestor of HEAD {head[:8]}")
        if git("diff", "--quiet", commit, "--", *files).returncode != 0:
            out.append(f"an {label} tooling file differs from commit {commit[:8]} (working tree or HEAD)")
    w = ROOT / "external" / "WWE-UIE"
    if git("rev-parse", "HEAD", cwd=w).stdout.strip() != rw.WWE_COMMIT:
        out.append("external/WWE-UIE is not at the pinned commit")
    if git("status", "--porcelain", cwd=w).stdout.strip():
        out.append("external/WWE-UIE is not clean")
    return out


def check_original_dependency():
    """Fail closed: the git-ignored R3 loss file must be readable and have the pinned SHA-256."""
    return xim.verify_original_dependency()


def check_config_and_model():
    """No file is read: the recipe is compared with train_v2.RECIPES['R3'] and the XF recipe, and the IN model is built on the CPU (structure and parameter count only)."""
    out = []
    import train_v2 as tv
    from models.at_trunk_in import ATMarineTrunkIN, N_NORM_LAYERS, norm_layers
    recipe = xim.recipe_args(tv.RECIPES)
    want = dict(epochs=100, warmup=3, bs=4, accum=1, lr=2e-4, wd=1e-5, clip=1.0, amp=False, loss_in_autocast=False, no_jitter=True, select_on="val_A", skip_zero_weight_perceptual=True)
    bad = {k: recipe.get(k) for k, v in want.items() if recipe.get(k) != v}
    if bad:
        out.append(f"the XFINMS recipe values differ from the protocol: {bad}")
    if abs(recipe.get("eta_min", 0) - 2e-8) > 1e-15:
        out.append(f"eta_min {recipe.get('eta_min')!r} != 2e-8")
    if recipe != xf.recipe_args(tv.RECIPES):
        out.append("the XFINMS recipe differs from the XF recipe")
    if set(xim.recipe_diff(tv.RECIPES["R3"], recipe)) != set(xf.R3_DEVIATIONS):
        out.append(f"the recipe differs from R3 in {sorted(xim.recipe_diff(tv.RECIPES['R3'], recipe))}, expected {sorted(xf.R3_DEVIATIONS)}")
    m = ATMarineTrunkIN()
    if sum(p.numel() for p in m.parameters()) != xim.EXPECTED_PARAMS:
        out.append(f"ATMarineTrunkIN parameter count != {xim.EXPECTED_PARAMS:,}")
    if norm_layers(m) != (N_NORM_LAYERS, 0):
        out.append(f"normalisation layers (InstanceNorm, BatchNorm) = {norm_layers(m)}, expected ({N_NORM_LAYERS}, 0)")
    ref = tv.MODEL_REGISTRY["at_trunk"]()  # the original AT-Trunk, for the pairing check
    if sum(p.numel() for p in ref.parameters()) != xim.EXPECTED_PARAMS or norm_layers(ref) != (0, N_NORM_LAYERS):
        out.append("the original AT-Trunk does not have 1,952,768 parameters and 40 BatchNorm layers")
    return out


def check_xfin_reference(root=ROOT, expected_sha256=xim.XFIN_S0_RESULT_SHA256):
    """The existing XFIN seed-0 result.json (train + val_A manifests only) must exist, have the PINNED SHA-256 and describe XFIN seed 0: it is the reference of the gate's args pairing. Only this one file is read."""
    p = Path(root) / "runs_v2" / xim.EXISTING_XFIN_RUN / "result.json"
    if not p.is_file():
        return [f"{p.parent.name}/result.json is missing: the existing XFIN seed-0 reference run must be present"]
    problems = xim.verify_xfin_reference_hash(p, expected_sha256)
    if problems:
        return problems
    try:
        r = json.load(open(p))
    except (OSError, ValueError) as e:
        return [f"{p.parent.name}/result.json is unreadable ({type(e).__name__})"]
    bad = xi.check_result_metadata(r) if isinstance(r, dict) else ["not a JSON object"]
    return [f"{p.parent.name}/result.json does not describe XFIN seed 0: {bad}"] if bad else []


def default_checks(seed, root=ROOT):
    xim.validate_seed(seed)
    return {"git provenance": check_git, "R3 loss file (pinned SHA-256)": check_original_dependency, "config + models (IN / BN) + XF recipe": check_config_and_model,
            "XFIN seed-0 reference run": lambda: check_xfin_reference(root), "disk + vgg cache": check_disk_and_cache, "gpu free memory": check_gpu_free,
            "train/val_A manifests + pairing": check_manifests}


# ----------------------------------------------------------------------------- driver
def run_single(seed, root=ROOT, checks=None, train_cmd=None, gate_cmd=None, max_wall_s=MAX_WALL_S, poll_s=POLL_S, preflight_only=False, head=None, gate_timeout_s=GATE_TIMEOUT_S, verify_dep=None):
    xim.validate_seed(seed)  # BEFORE any file is created: seed 0 and every other value raise here
    RUN = xim.run_name(seed)
    root = Path(root)
    verify_dep = verify_dep or xim.verify_original_dependency
    # a preflight-only invocation logs to its own file (timestamp + process id), so it can never block (or be mistaken for) the real run's log
    log_path = root / "runs_v2" / (f"{RUN}_preflight_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl" if preflight_only else f"{RUN}_chain_driver.jsonl")
    out_dir = root / "runs_v2" / RUN
    assert out_dir.name != xim.EXISTING_XFIN_RUN and log_path.name != f"{xim.EXISTING_XFIN_RUN}_chain_driver.jsonl", "refusing to touch the existing XFIN seed-0 run"

    def log(**kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)

    def stop(reason, **kw):
        log(event="STOP", reason=reason, **kw)
        return 1

    try:  # exclusive creation: never reuse or append to an existing log, even if two invocations race
        open(log_path, "x").close()
    except FileExistsError:
        print(f"STOP: {log_path} already exists", file=sys.stderr)
        return 1
    if out_dir.exists():
        return stop(f"{out_dir.name}: output directory already exists (even if empty); nothing launched")
    import torch
    import train_v2 as tv
    d_train, d_gate = commands(seed)
    train_cmd = train_cmd or d_train
    gate_cmd = gate_cmd or d_gate
    assert "--resume" not in train_cmd
    try:
        dep_sha = xim.sha256_file(xim.ROOT / xim.UNTRACKED_DEPENDENCY)
    except OSError:
        dep_sha = None
    try:
        x0_sha = xim.sha256_file(root / "runs_v2" / xim.EXISTING_XFIN_RUN / "result.json")
    except OSError:
        x0_sha = None
    log(event="plan", run=RUN, seed=seed, recipe=xim.RECIPE_NAME, recipe_args=xim.recipe_args(tv.RECIPES), differs_from_r3=xf.R3_DEVIATIONS, model=xim.MODEL, params=xim.EXPECTED_PARAMS,
        change="the 40 BatchNorm2d layers of AT-Trunk -> full-channel InstanceNorm2d (affine, no running statistics); nothing else differs from XF; the only change from XFIN seed 0 is the seed / run name",
        precision="fp32 forward and loss (no AMP, no GradScaler); the tf32 convolution / matmul flags are left at the PyTorch default and recorded in xfin_loss_config.json",
        batch_size=4, epochs=100,
        loss="R3 loss (original_repo MARINELoss via FinalOutputLoss + ZeroLoss, unchanged, built by the unchanged train_v2.build); weights " + json.dumps(xim.R3_WEIGHTS),
        purpose="XFIN replicate for the paired comparison with the XF run of the same seed; all Val-A numbers are exploratory",
        existing_xfin_run=xim.EXISTING_XFIN_RUN, xfin_seed0_result_sha256={"pinned": xim.XFIN_S0_RESULT_SHA256, "observed": x0_sha, "verified": x0_sha == xim.XFIN_S0_RESULT_SHA256},
        paired_xf_run=xim.xm.run_name(seed), reference_numbers=xim.REFERENCE_NUMBERS, paired_thresholds=xim.PAIRED_THRESHOLDS,
        remaining_confounders=xim.REMAINING_CONFOUNDERS, clip_log=xim.CLIP_LOG_FILE, train_command=" ".join(train_cmd[1:]), gate_command=" ".join(gate_cmd[1:]),
        xf_commit=xim.XF_COMMIT, xfin_commit=xim.XFIN_COMMIT, wwe_commit=rw.WWE_COMMIT, provenance_files=PROVENANCE_FILES,
        untracked_dependency={"path": xim.UNTRACKED_DEPENDENCY, "sha256": dep_sha, "pinned_sha256": xim.PINNED_ORIGINAL_SHA256, "verified": dep_sha == xim.PINNED_ORIGINAL_SHA256,
                              "note": "git-ignored; pinned hash verified in the preflight, on every poll, after training and by the gate"},
        env={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda},
        data="train + val_A only (selection per frozen protocol); val_B / UIEB never loaded; the XF and AMP result.json files are not opened", gates=GATES, max_wall_s=max_wall_s,
        gate_timeout_s=gate_timeout_s, estimate_hours={"value": xim.ESTIMATED_RUN_HOURS, "label": "ESTIMATE extrapolated from the recorded XFIN seed-0 run_seconds; not a measurement"},
        process_cleanup="job object (kill-on-close, fail closed) + taskkill /T /F tree kill on failure/timeout",
        rule="one launch only; any failure stops; no retry, no resume, no second run; all Val-A numbers are exploratory")
    for name, fn in (checks if checks is not None else default_checks(seed, root)).items():
        try:
            problems = fn()
        except Exception as e:  # noqa: BLE001  a check that cannot be evaluated is a failed check (fail closed)
            problems = [f"check raised {type(e).__name__}: {e}"]
        if problems:
            return stop(f"preflight '{name}' failed: {problems}")
        log(event="preflight_ok", check=name)
    head = head or git("rev-parse", "HEAD", cwd=root).stdout.strip()
    log(event="preflight_done", head=head)
    if preflight_only:
        return 0
    job = make_kill_on_close_job()
    if job is None and REQUIRE_JOB:
        return stop("Job Object creation failed: refusing to launch training without descendant cleanup; nothing launched")
    out_dir.mkdir(parents=True, exist_ok=False)
    log(event="start", run=RUN, output_dir=f"runs_v2/{RUN}", disk_free_gb=round(shutil.disk_usage(root).free / 2**30, 1))
    t0 = time.time()
    reason = None
    proc = None
    try:
        with open(out_dir / "train_attempt0.log", "w") as lf:
            proc = subprocess.Popen(train_cmd, cwd=root, stdout=lf, stderr=subprocess.STDOUT, **popen_kwargs())
            assigned = bool(job and job.assign(proc.pid))
            log(event="tree_guard", job_object_created=job is not None, training_pid_assigned=assigned, pid=proc.pid,
                note="descendants spawned after assignment die with the driver; fallback on failure: taskkill /T /F" if assigned else "NO job object: only taskkill /T /F cleanup")
            if REQUIRE_JOB and not assigned:
                kill_tree(proc, job)
                return stop(f"{RUN}: Job Object assignment failed; training process tree killed by the driver, no resume", wall_s=round(time.time() - t0))
            while proc.poll() is None:
                reason = scan_history_fp32(out_dir / "history.jsonl")
                if reason is None:
                    dep = verify_dep()  # fail closed: the R3 definition must stay the pinned one for the whole run
                    if dep:
                        reason = f"R3 loss file check failed while training: {dep}"
                if reason is None and time.time() - t0 > max_wall_s:
                    reason = f"wall-time limit {max_wall_s}s exceeded"
                if reason:
                    kill_tree(proc, job)
                    return stop(f"{RUN}: {reason}; training process tree terminated by the driver, no resume", wall_s=round(time.time() - t0))
                time.sleep(poll_s)
    finally:
        if proc is not None and proc.poll() is None:  # unexpected exception in the driver: do not leave the tree running
            kill_tree(proc, job)
        if job is not None:
            job.close()  # also kills any worker the finished training left behind
    rc = proc.returncode
    if rc != 0:
        return stop(f"{RUN}: training failed (rc={rc}); no retry", wall_s=round(time.time() - t0))
    reason = scan_history_fp32(out_dir / "history.jsonl")
    if reason:
        return stop(f"{RUN}: {reason}")
    dep = verify_dep()
    if dep:
        return stop(f"{RUN}: R3 loss file check failed after training: {dep}")
    missing = [a for a in ARTIFACTS if not (out_dir / a).exists()]
    if missing:
        return stop(f"{RUN}: expected artifacts missing: {missing}")
    try:
        r = json.load(open(out_dir / "result.json"))
    except (OSError, ValueError) as e:
        return stop(f"{RUN}: result.json is unreadable ({type(e).__name__})")
    bad = xim.check_result_metadata(r, seed)
    if bad:
        return stop(f"{RUN}: result.json metadata does not match experiment XFINMS seed {seed}: {bad}")
    rec_commit = r.get("env", {}).get("git_commit", "")
    if not rec_commit or not head.startswith(rec_commit):
        return stop(f"{RUN}: recorded git_commit {rec_commit!r} != HEAD at preflight {head[:8]}")
    try:
        lc = json.load(open(out_dir / xim.LOSS_CONFIG_FILE))
        lc_commit, lc_sha = lc["code"]["git_commit_full"], lc["code"]["untracked_dependency_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return stop(f"{RUN}: {xim.LOSS_CONFIG_FILE} is unreadable or incomplete ({type(e).__name__})")
    if lc_commit != head:
        return stop(f"{RUN}: {xim.LOSS_CONFIG_FILE} commit {str(lc_commit)[:8]} != HEAD at preflight {head[:8]}")
    if lc_sha != xim.PINNED_ORIGINAL_SHA256:
        return stop(f"{RUN}: {xim.LOSS_CONFIG_FILE} records an R3 file hash that is not the pinned one")
    log(event="training_complete", run=RUN, seed=seed, wall_s=round(time.time() - t0), best_valA_epoch=r.get("best_valA_epoch"), best_valA=r.get("best_valA"), params=r.get("params"),
        recorded_git_commit=rec_commit, run_seconds=r.get("run_seconds"), checkpoint_sha256=r.get("checkpoint_sha256"))
    try:
        g_rc, g_out, g_err, timed_out = run_bounded(gate_cmd, root, gate_timeout_s)
    except Exception as e:  # noqa: BLE001  the gate could not even be started: the run is NOT validated
        return stop(f"{RUN}: validation gate could not be executed ({type(e).__name__}: {e}); the run is NOT validated")
    try:
        with open(out_dir / xim.GATE_LOG_FILE, "x") as gf:
            gf.write(g_out + "\n--- stderr ---\n" + g_err + (f"\n--- GATE TIMED OUT after {gate_timeout_s}s; process tree killed ---\n" if timed_out else ""))
    except FileExistsError:
        return stop(f"{RUN}: {xim.GATE_LOG_FILE} already exists; refusing to overwrite; the run is NOT validated")
    if timed_out:
        return stop(f"{RUN}: validation gate TIMED OUT after {gate_timeout_s}s (process tree killed; see runs_v2/{RUN}/{xim.GATE_LOG_FILE}); the run is NOT validated")
    if g_rc != 0:
        return stop(f"{RUN}: validation gate FAILED (rc={g_rc}; see runs_v2/{RUN}/{xim.GATE_LOG_FILE})")
    log(event="gate_passed", run=RUN)
    log(event="all_done", runs=[RUN], note="single run complete; nothing further launched")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, choices=list(xim.ALLOWED_SEEDS), help="1 or 2 (seed 0 is the existing XFIN run)")
    ap.add_argument("--preflight_only", action="store_true", help="run the preflight (reads train/val_A manifest JSON only, no images) and exit; launches nothing")
    a = ap.parse_args()
    sys.exit(run_single(a.seed, preflight_only=a.preflight_only))


if __name__ == "__main__":
    main()
