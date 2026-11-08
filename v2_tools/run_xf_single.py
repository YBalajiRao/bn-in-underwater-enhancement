"""Single-run gated driver for experiment XF: AT-Trunk (unchanged) + R3 loss (unchanged) + fp32 (amp False) + cosine floor eta_min 2e-8, seed 0 only; run name xf_attrunk_r3_fp32_s0. NOT approved to run.
Sequence: preflight -> ONE training process -> artifact + metadata checks -> training_complete -> gate (bounded by a timeout) -> gate_passed -> all_done.
Exactly one launch ever: no retry, no resume (--resume is never passed), no second run; ANY failure ends the driver with exit 1 and a STOP record; nothing is cleaned up.
The only option is --preflight_only (there is no seed or run-name option: the run is fixed, and its output paths are dedicated, so no existing run can be touched).
Safeguards (helpers imported unchanged from the reviewed v2_tools/run_xr_single.py and the hardened v2_tools/run_wwe_single.py; the orchestration below is new because the XR one is WWE-UIE specific):
  * exclusive creation of the driver log and of the preflight log (timestamp + process id); the output directory must not exist;
  * provenance preflight: the index is empty; the six new tooling files, the imported XR / WWE helper files and the other code dependencies are TRACKED and unmodified; the XR tooling commit
    a1261b5 is an ancestor of HEAD; external/WWE-UIE (imported through models/wwe_uie.py) is clean at its pinned commit;
  * R3 provenance (fail closed): original_repo/marine_net_model.py is git-ignored, so its SHA-256 is PINNED (xr_config) and verified in the preflight, on EVERY poll while training runs
    (a change kills the process tree), once more after training, and by the gate;
  * recipe preflight: the XF recipe equals train_v2.RECIPES["R3"] except amp False and eta_min 2e-8; AT-Trunk builds on the CPU with 1,952,768 parameters (no data access). The AT-Trunk R3 seed-0
    result.json is NOT read;
  * train and Val-A manifests only (hash + strict pair validation, every other manifest key dropped first); result.json must hold exactly those two hashes;
  * fp32 history scan: ANY non-finite loss step, ANY non-finite-gradient step (fp32 has no GradScaler; the wrapper also raises before optimizer.step), a non-finite loss / gradient norm / Val-A value stops the run;
  * every external command is bounded (git calls 120 s, the GPU probe 120 s, the gate 30 min, training by the wall limit); a preflight check that raises is a STOP;
  * Windows fail-closed process cleanup: the training process runs inside a kill-on-close Job Object; if the job cannot be created or the process cannot be assigned, nothing is launched / the
    just-started tree is killed; on any failure, non-finite value, dependency change or timeout the whole tree is killed (taskkill /T /F only while alive + TerminateJobObject);
  * wall-time limit 10 h (a safeguard, not a duration target; the estimate is about 4 to 6 h and is NOT a measurement); the gate runs under a 30 min timeout (tree killed on timeout; the run is then NOT validated);
  * result.json must describe this run (seed 0, run name, recipe, model, params, fp32, eta_min, no resume, exact manifest scope) and the clip log must exist before training_complete is logged.
Events: plan, preflight_ok*, preflight_done, start, tree_guard, training_complete (NOT a success marker), gate_passed, all_done (the only success event). STOP on any failure.
Usage (NOT approved to run): python v2_tools/run_xf_single.py [--preflight_only]"""
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
from v2_tools import run_xr_single as rx  # noqa: E402
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools.run_wwe_single import kill_tree, make_kill_on_close_job, popen_kwargs, run_bounded  # noqa: E402

GIT_TIMEOUT_S = rx.GIT_TIMEOUT_S
GPU_PROBE_TIMEOUT_S = rx.GPU_PROBE_TIMEOUT_S
RUN = xf.RUN_NAME
WRAPPER, GATE = "v2_tools/train_v2_no_valb_xf.py", "v2_tools/xf_gate.py"
ARTIFACTS = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", xf.LOSS_CONFIG_FILE, xf.CLIP_LOG_FILE, "train_attempt0.log"]
GATES = ["files exist", "model/params", "recorded args == expected XF args (R3 recipe with ONLY amp False and eta_min 2e-8), fp32, run identity", "100 contiguous epochs",
         "no non-finite loss / gradient steps, finite losses, gradient norms and Val-A, no GradScaler", "manifest hashes as expected (exactly train + val_A), pairs valid", "best epoch == argmax history",
         "checkpoint SHA-256 == result.json", "val-A re-score == logged, checkpoint epoch == best epoch", "xf_loss_config.json == expected loss record, commit matches, R3 file hash == pinned == recorded",
         "xf_clip_log.jsonl: 100 epochs, 86,600 steps, consistent with history.jsonl"]
PROVENANCE_FILES = list(xf.PROVENANCE_FILES)
MAX_WALL_S = 10 * 3600  # estimate ~4-6 h (an extrapolation, not a measurement); a hard safeguard, not a target
POLL_S = 30
GATE_TIMEOUT_S = 30 * 60
REQUIRE_JOB = os.name == "nt"  # fail closed on Windows if the Job Object cannot be created / assigned (tests may force it on other platforms)

scan_history_fp32 = rx.scan_history_fp32  # reviewed fp32 history scan (any non-finite loss / gradient step, non-finite loss / norm / Val-A)
git = rx.git  # bounded git call (timeout raises)
check_gpu_free = rx.check_gpu_free  # bounded GPU probe in a child process
check_disk_and_cache, check_manifests = rw.check_disk_and_cache, rw.check_manifests


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
    if git("merge-base", "--is-ancestor", xf.XR_COMMIT, "HEAD").returncode != 0:
        out.append(f"{xf.XR_COMMIT[:8]} (the XR tooling commit) is not an ancestor of HEAD {head[:8]}")
    w = ROOT / "external" / "WWE-UIE"
    if git("rev-parse", "HEAD", cwd=w).stdout.strip() != rw.WWE_COMMIT:
        out.append("external/WWE-UIE is not at the pinned commit")
    if git("status", "--porcelain", cwd=w).stdout.strip():
        out.append("external/WWE-UIE is not clean")
    return out


def check_original_dependency():
    """Fail closed: the git-ignored R3 loss file must be readable and have the pinned SHA-256."""
    return xf.verify_original_dependency()


def check_config_and_model():
    """No file is read: the recipe is compared with train_v2.RECIPES['R3'] and AT-Trunk is built on the CPU (parameter count only)."""
    out = []
    import train_v2 as tv
    recipe = xf.recipe_args(tv.RECIPES)
    want = dict(epochs=100, warmup=3, bs=4, accum=1, lr=2e-4, wd=1e-5, clip=1.0, amp=False, loss_in_autocast=False, no_jitter=True, select_on="val_A", skip_zero_weight_perceptual=True)
    bad = {k: recipe.get(k) for k, v in want.items() if recipe.get(k) != v}
    if bad:
        out.append(f"the XF recipe values differ from the protocol: {bad}")
    if abs(recipe.get("eta_min", 0) - 2e-8) > 1e-15:
        out.append(f"eta_min {recipe.get('eta_min')!r} != 2e-8")
    if set(xf.recipe_diff(tv.RECIPES[xf.BASELINE_RECIPE], recipe)) != set(xf.R3_DEVIATIONS):
        out.append(f"the XF recipe differs from R3 in {sorted(xf.recipe_diff(tv.RECIPES[xf.BASELINE_RECIPE], recipe))}, expected {sorted(xf.R3_DEVIATIONS)}")
    m = tv.MODEL_REGISTRY[xf.MODEL]()
    if type(m) is not tv.ATMarineTrunk or sum(p.numel() for p in m.parameters()) != xf.EXPECTED_PARAMS:
        out.append(f"AT-Trunk parameter count != {xf.EXPECTED_PARAMS:,}")
    return out


def default_checks():
    return {"git provenance": check_git, "R3 loss file (pinned SHA-256)": check_original_dependency, "config + model + R3-derived recipe": check_config_and_model,
            "disk + vgg cache": check_disk_and_cache, "gpu free memory": check_gpu_free, "train/val_A manifests + pairing": check_manifests}


# ----------------------------------------------------------------------------- driver
def run_single(root=ROOT, checks=None, train_cmd=None, gate_cmd=None, max_wall_s=MAX_WALL_S, poll_s=POLL_S, preflight_only=False, head=None, gate_timeout_s=GATE_TIMEOUT_S, verify_dep=None):
    root = Path(root)
    verify_dep = verify_dep or xf.verify_original_dependency
    # a preflight-only invocation logs to its own file (timestamp + process id), so it can never block (or be mistaken for) the real run's log
    log_path = root / "runs_v2" / (f"{RUN}_preflight_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl" if preflight_only else f"{RUN}_chain_driver.jsonl")
    out_dir = root / "runs_v2" / RUN

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
    train_cmd = train_cmd or [sys.executable, WRAPPER, "--recipe", xf.RECIPE_NAME, "--model", xf.MODEL, "--seed", str(xf.SEED), "--run_name", RUN]
    gate_cmd = gate_cmd or [sys.executable, GATE, "--run", RUN]
    assert "--resume" not in train_cmd
    try:
        dep_sha = xf.sha256_file(xf.ROOT / xf.UNTRACKED_DEPENDENCY)
    except OSError:
        dep_sha = None
    log(event="plan", run=RUN, recipe=xf.RECIPE_NAME, recipe_args=xf.recipe_args(tv.RECIPES), differs_from_r3=xf.R3_DEVIATIONS, seed=xf.SEED, model=xf.MODEL, params=xf.EXPECTED_PARAMS,
        precision="fp32 forward and loss (no AMP, no GradScaler); the tf32 convolution / matmul flags are left at the PyTorch default and recorded in xf_loss_config.json",
        batch_size=4, epochs=100,
        loss="R3 loss (original_repo MARINELoss via FinalOutputLoss + ZeroLoss, unchanged, built by the unchanged train_v2.build); weights " + json.dumps(xf.R3_WEIGHTS),
        baseline_reference=xf.BASELINE_RUN, reference_numbers=xf.REFERENCE_NUMBERS, decision_thresholds=xf.DECISION_THRESHOLDS, remaining_confounders=xf.REMAINING_CONFOUNDERS,
        clip_log=xf.CLIP_LOG_FILE, train_command=" ".join(train_cmd[1:]), gate_command=" ".join(gate_cmd[1:]), xr_commit=xf.XR_COMMIT, wwe_commit=rw.WWE_COMMIT,
        provenance_files=PROVENANCE_FILES, untracked_dependency={"path": xf.UNTRACKED_DEPENDENCY, "sha256": dep_sha, "pinned_sha256": xf.PINNED_ORIGINAL_SHA256, "verified": dep_sha == xf.PINNED_ORIGINAL_SHA256,
                                                                   "note": "git-ignored; pinned hash verified in the preflight, on every poll, after training and by the gate"},
        env={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda},
        data="train + val_A only (selection per frozen protocol); val_B / UIEB never loaded; the AT-Trunk R3 seed-0 result.json is not read", gates=GATES, max_wall_s=max_wall_s, gate_timeout_s=gate_timeout_s,
        process_cleanup="job object (kill-on-close, fail closed) + taskkill /T /F tree kill on failure/timeout",
        rule="one launch only; any failure stops; no retry, no resume, no second run; all Val-A numbers are exploratory")
    for name, fn in (checks if checks is not None else default_checks()).items():
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
    bad = xf.check_result_metadata(r)
    if bad:
        return stop(f"{RUN}: result.json metadata does not match experiment XF: {bad}")
    rec_commit = r.get("env", {}).get("git_commit", "")
    if not rec_commit or not head.startswith(rec_commit):
        return stop(f"{RUN}: recorded git_commit {rec_commit!r} != HEAD at preflight {head[:8]}")
    try:
        lc = json.load(open(out_dir / xf.LOSS_CONFIG_FILE))
        lc_commit, lc_sha = lc["code"]["git_commit_full"], lc["code"]["untracked_dependency_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return stop(f"{RUN}: {xf.LOSS_CONFIG_FILE} is unreadable or incomplete ({type(e).__name__})")
    if lc_commit != head:
        return stop(f"{RUN}: {xf.LOSS_CONFIG_FILE} commit {str(lc_commit)[:8]} != HEAD at preflight {head[:8]}")
    if lc_sha != xf.PINNED_ORIGINAL_SHA256:
        return stop(f"{RUN}: {xf.LOSS_CONFIG_FILE} records an R3 file hash that is not the pinned one")
    log(event="training_complete", run=RUN, wall_s=round(time.time() - t0), best_valA_epoch=r.get("best_valA_epoch"), best_valA=r.get("best_valA"), params=r.get("params"),
        recorded_git_commit=rec_commit, run_seconds=r.get("run_seconds"), checkpoint_sha256=r.get("checkpoint_sha256"))
    try:
        g_rc, g_out, g_err, timed_out = run_bounded(gate_cmd, root, gate_timeout_s)
    except Exception as e:  # noqa: BLE001  the gate could not even be started: the run is NOT validated
        return stop(f"{RUN}: validation gate could not be executed ({type(e).__name__}: {e}); the run is NOT validated")
    try:
        with open(out_dir / "xf_gate.log", "x") as gf:
            gf.write(g_out + "\n--- stderr ---\n" + g_err + (f"\n--- GATE TIMED OUT after {gate_timeout_s}s; process tree killed ---\n" if timed_out else ""))
    except FileExistsError:
        return stop(f"{RUN}: xf_gate.log already exists; refusing to overwrite; the run is NOT validated")
    if timed_out:
        return stop(f"{RUN}: validation gate TIMED OUT after {gate_timeout_s}s (process tree killed; see runs_v2/{RUN}/xf_gate.log); the run is NOT validated")
    if g_rc != 0:
        return stop(f"{RUN}: validation gate FAILED (rc={g_rc}; see runs_v2/{RUN}/xf_gate.log)")
    log(event="gate_passed", run=RUN)
    log(event="all_done", runs=[RUN], note="single run complete; nothing further launched")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preflight_only", action="store_true", help="run the preflight (reads train/val_A manifest JSON only, no images) and exit; launches nothing")
    sys.exit(run_single(preflight_only=ap.parse_args().preflight_only))


if __name__ == "__main__":
    main()
