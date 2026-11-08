"""Single-run gated driver: WWE-UIE, WWE_lr2e-4, one seed per invocation (batch 4, FP32, 100 epochs, train-only data, Val-A checkpoint selection per the frozen protocol).
Seed 0 was run by the first version of this driver (runs_v2/wwe_lr2e-4_s0, complete) and is refused here; this version accepts --seed 1 or --seed 2 only.
Sequence: preflight -> ONE training process -> artifact + metadata checks -> training_complete -> gate (bounded by a timeout) -> gate_passed -> all_done. Exactly one launch per invocation:
no retry, no resume (--resume is never passed), no second run. ANY failure ends the driver with exit 1 and a STOP record; nothing is cleaned up.
STOP conditions: failed preflight; output directory or driver log already exists (even empty); training exit code != 0; non-finite value seen in history.jsonl while
training (non-finite train loss, skipped/non-finite-gradient steps, non-finite Val-A) -> the training process TREE is killed; wall-time limit exceeded -> tree killed;
missing expected artifact; recorded git commit != the HEAD seen at preflight; gate exit code != 0; gate timeout (default 30 min) -> gate tree killed.
Process cleanup (Windows): the training process is placed in a Job Object with KILL_ON_JOB_CLOSE, so it and its descendants die if this driver exits for any reason;
failures/timeouts additionally run `taskkill /T /F` on the training process (only while it is still alive) and TerminateJobObject. On Windows the driver FAILS CLOSED:
if the job object cannot be created or the training process cannot be assigned to it, nothing is launched / the just-started tree is killed and a STOP is logged.
Events: plan, preflight_ok*, preflight_done, start, tree_guard, training_complete (training + artifact + metadata checks passed; NOT a success marker), gate_passed, all_done
(the only success event, emitted after the gate passed). Legacy: the seed-0 log (written by the first driver version) names the pre-gate event `done` instead of `training_complete`.
Code provenance: the 10 committed WWE files must be byte-identical to commit 32dec7754bb8ad72063ebc04ff47fca99f948ebd (which must be an ancestor of HEAD);
the index must be empty; external/WWE-UIE must be clean at its pinned commit. The pilot (runs_v2/wwe_pilot_lr2e-4_s0_v1) ran earlier from HEAD 09b4fad with
then-uncommitted code and is NOT claimed to have run from 32dec77.
Log: runs_v2/wwe_lr2e-4_s<seed>_chain_driver.jsonl (refuses to start if it exists).
Usage: python v2_tools/run_wwe_single.py --seed {1,2} [--preflight_only]"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED_COMMIT = "32dec7754bb8ad72063ebc04ff47fca99f948ebd"
WWE_COMMIT = "fd0697558203f9f849800cc13b804116bb3f921c"
RECIPE = "WWE_lr2e-4"
ALLOWED_SEEDS = (1, 2)  # seed 0 is the completed first run (old driver version); it must never be re-run or overwritten
COMMITTED_FILES = ["models/wwe_uie.py", "v2_tools/wwe_config.py", "v2_tools/wwe_loss.py", "v2_tools/wwe_pairing.py", "v2_tools/train_v2_no_valb_wwe.py",
                   "v2_tools/wwe_gate.py", "v2_tools/wwe_pilot.py", "tests_v2/test_wwe_uie.py", "tests_v2/test_wwe_pipeline.py", "tests_v2/test_wwe_pilot.py"]
WRAPPER, GATE = "v2_tools/train_v2_no_valb_wwe.py", "v2_tools/wwe_gate.py"
ARTIFACTS = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", "wwe_loss_config.json", "train_attempt0.log"]
GATES = ["files exist", "model/params", "recipe args == WWE recipe", "100 contiguous epochs", "no non-finite loss steps / finite val-A",
         "manifest hashes as expected, val_B absent, pairs valid", "best epoch == argmax history", "checkpoint SHA-256 == result.json", "val-A re-score == logged"]
MAX_WALL_S = 8 * 3600  # estimate is ~3.7 h; abort (terminate) beyond 8 h
POLL_S = 30
REQUIRE_JOB = os.name == "nt"  # fail closed on Windows if the Job Object cannot be created / assigned (tests may force it on other platforms)
GATE_TIMEOUT_S = 30 * 60
MIN_FREE_GB, MIN_GPU_FREE_MB = 5.0, 6000


def run_name(seed):
    """wwe_lr2e-4_s<seed>; only the new-workflow seeds are accepted (clear ValueError otherwise)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: only seeds {list(ALLOWED_SEEDS)} are supported by this workflow (seed 0 is the completed run and is never re-run)")
    return f"wwe_lr2e-4_s{seed}"


# ----------------------------------------------------------------------------- pure helpers
def check_result_metadata(r, seed, run):
    """Problems if result.json does not describe THIS requested run (seed, run name, recipe, no resume); [] if it does."""
    a = r.get("args") or {}
    out = []
    if r.get("seed") != seed or a.get("seed") != seed:
        out.append(f"seed {r.get('seed')!r}/{a.get('seed')!r} != requested {seed}")
    if a.get("run_name") != run:
        out.append(f"run_name {a.get('run_name')!r} != {run!r}")
    if a.get("recipe") != RECIPE:
        out.append(f"recipe {a.get('recipe')!r} != {RECIPE!r}")
    if a.get("resume") is not False:
        out.append(f"resume flag is {a.get('resume')!r}, expected False")
    return out


def scan_history(path):
    """Return a reason string if history.jsonl shows a non-finite value, else None. A partially written last line is ignored."""
    p = Path(path)
    if not p.exists():
        return None
    lines = p.read_text().splitlines()
    for i, l in enumerate(lines):
        try:
            h = json.loads(l)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            return f"history.jsonl line {i} is not valid JSON"
        t, v = h.get("train", {}), h.get("val_A", {})
        if t.get("nonfinite", 0) or t.get("inf_grad_steps", 0):
            return f"epoch {h.get('epoch')}: nonfinite loss steps {t.get('nonfinite')} / non-finite gradient steps {t.get('inf_grad_steps')}"
        for k, val in (("train.total", t.get("total")), ("val_A.psnr_mean", v.get("psnr_mean")), ("val_A.ssim_mean", v.get("ssim_mean"))):
            if val is None or val != val or val in (float("inf"), float("-inf")):
                return f"epoch {h.get('epoch')}: {k} = {val}"
    return None


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def check_git():
    out = []
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD").returncode != 0:
        out.append(f"{EXPECTED_COMMIT[:8]} is not an ancestor of HEAD {head[:8]}")
    if git("diff", "--cached", "--quiet").returncode != 0:
        out.append("the index is not empty")
    if git("diff", "--quiet", EXPECTED_COMMIT, "--", *COMMITTED_FILES).returncode != 0:
        out.append("a committed WWE file differs from commit 32dec77 (working tree or HEAD)")
    for f in COMMITTED_FILES:
        if git("cat-file", "-e", f"{EXPECTED_COMMIT}:{f}").returncode != 0:
            out.append(f"{f} is not part of commit 32dec77")
    if git("status", "--porcelain", "--", *COMMITTED_FILES).stdout.strip():
        out.append("committed WWE files have uncommitted changes")
    w = ROOT / "external" / "WWE-UIE"
    if git("rev-parse", "HEAD", cwd=w).stdout.strip() != WWE_COMMIT:
        out.append("external/WWE-UIE is not at the pinned commit")
    if git("status", "--porcelain", cwd=w).stdout.strip():
        out.append("external/WWE-UIE is not clean")
    return out


def check_config_and_model():
    out = []
    from models.wwe_uie import EXPECTED_PARAMS, WWEUIE
    from v2_tools import wwe_config as cfg
    r = cfg.RECIPES.get(RECIPE)
    if r is None or r["lr"] != 2e-4 or r["bs"] != 4 or r["amp"] is not False or r["epochs"] != 100 or r["select_on"] != "val_A":
        out.append(f"recipe {RECIPE} is not lr 2e-4 / bs 4 / fp32 / 100 epochs / val_A selection: {r}")
    if sum(p.numel() for p in WWEUIE().parameters()) != EXPECTED_PARAMS:
        out.append("parameter count mismatch")
    return out


def check_disk_and_cache():
    out = []
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < MIN_FREE_GB:
        out.append(f"only {free:.1f} GB free")
    import torch
    if not (Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth").exists():
        out.append("VGG16 weights missing from the torch hub cache (no download allowed)")
    return out


def check_gpu_free():
    """Free-memory probe in a short child process (keeps a CUDA context out of the driver)."""
    code = "import torch;print(torch.cuda.is_available(), torch.cuda.mem_get_info()[0]//2**20 if torch.cuda.is_available() else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        ok, free = r.stdout.split()
    except ValueError:
        return [f"GPU probe failed: {r.stderr[-200:]}"]
    return [] if ok == "True" and int(free) >= MIN_GPU_FREE_MB else [f"CUDA unavailable or only {free} MB free (need {MIN_GPU_FREE_MB})"]


def check_manifests():
    """Train and Val-A manifests ONLY: hash + strict pair validation (reads manifest JSON, no image). Every other manifest key is dropped first."""
    import train_v2 as tv
    from v2_tools import wwe_config as cfg
    from v2_tools.wwe_pairing import validate_pairs
    for k in [k for k in tv.MANIFESTS if k not in ("train", "val_A")]:
        del tv.MANIFESTS[k]
    out = []
    for k in ("train", "val_A"):
        path, key, prefix = tv.MANIFESTS[k]
        if not tv.sha256(path).startswith(prefix):
            out.append(f"manifest {k} hash mismatch")
            continue
        try:
            validate_pairs(json.load(open(path))[key], cfg.EXPECTED_N[k], k)
        except AssertionError as e:
            out.append(str(e))
    return out


def default_checks():
    return {"git provenance": check_git, "config + model": check_config_and_model, "disk + vgg cache": check_disk_and_cache,
            "gpu free memory": check_gpu_free, "train/val_A manifests + pairing": check_manifests}


# ----------------------------------------------------------------------------- driver
class JobGuard:
    """Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: every process assigned to it (and every descendant it spawns afterwards) is killed when the
    last handle to the job closes, i.e. when close() runs OR this driver process dies for any reason. Limitation: a descendant spawned before assign() is not covered."""
    KILL_ON_JOB_CLOSE, EXTENDED_LIMIT_INFO, SET_QUOTA, TERMINATE = 0x2000, 9, 0x0100, 0x0001

    def __init__(self, k32, handle):
        self.k32, self.handle = k32, handle

    @classmethod
    def create(cls):
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.argtypes, k.CreateJobObjectW.restype = [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE
        k.SetInformationJobObject.argtypes, k.SetInformationJobObject.restype = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD], wintypes.BOOL
        k.OpenProcess.argtypes, k.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
        k.AssignProcessToJobObject.argtypes, k.AssignProcessToJobObject.restype = [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL
        k.TerminateJobObject.argtypes, k.TerminateJobObject.restype = [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL
        k.CloseHandle.argtypes, k.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        h = k.CreateJobObjectW(None, None)
        if not h:
            return None
        info = Extended()
        info.Basic.LimitFlags = cls.KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(h, cls.EXTENDED_LIMIT_INFO, ctypes.byref(info), ctypes.sizeof(info)):
            k.CloseHandle(h)
            return None
        return cls(k, h)

    def assign(self, pid):
        ph = self.k32.OpenProcess(self.SET_QUOTA | self.TERMINATE, False, pid)
        if not ph:
            return False
        try:
            return bool(self.k32.AssignProcessToJobObject(self.handle, ph))
        finally:
            self.k32.CloseHandle(ph)

    def terminate(self):
        if self.handle:
            self.k32.TerminateJobObject(self.handle, 1)

    def close(self):
        """Closing the last handle kills any process still in the job (KILL_ON_JOB_CLOSE)."""
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None


def make_kill_on_close_job():
    """JobGuard on Windows, None elsewhere or if the job could not be created (callers must then rely on kill_tree only)."""
    if os.name != "nt":
        return None
    try:
        return JobGuard.create()
    except Exception:
        return None


def popen_kwargs():
    """POSIX: own session so the whole group can be killed (the real target is Windows, where taskkill /T walks the parent links)."""
    return {} if os.name == "nt" else {"start_new_session": True}


def kill_tree(proc, job=None, timeout=30):
    """Kill proc AND its descendants (and anything in the job object), then reap proc. Safe to call on an already-exited process:
    taskkill / killpg are issued ONLY while proc is still alive; TerminateJobObject always runs (it also catches orphans whose parent already exited).
    Windows: `taskkill /T /F` walks parent links while the parent is alive."""
    if proc.poll() is None:  # never signal a PID whose process already exited (it could have been recycled by an unrelated process)
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=timeout)
            except (subprocess.TimeoutExpired, OSError):
                pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    if job is not None:
        job.terminate()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_bounded(cmd, cwd, timeout):
    """Run cmd capturing output with a hard time limit; on timeout the whole process tree is killed. -> (returncode or None, stdout, stderr, timed_out)."""
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs())
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err, False
    except subprocess.TimeoutExpired:
        kill_tree(p)
        try:
            out, err = p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return None, out or "", err or "", True


def run_single(seed, root=ROOT, checks=None, train_cmd=None, gate_cmd=None, max_wall_s=MAX_WALL_S, poll_s=POLL_S, preflight_only=False, head=None,
               gate_timeout_s=GATE_TIMEOUT_S):
    RUN = run_name(seed)  # raises ValueError for seed 0 / unknown seeds before anything is created
    root = Path(root)
    # a preflight-only invocation logs to its own timestamped file, so it can never block (or be mistaken for) the real run's log
    log_path = root / "runs_v2" / (f"{RUN}_preflight_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl" if preflight_only else f"{RUN}_chain_driver.jsonl")
    out_dir = root / "runs_v2" / RUN
    state = {"launched": 0}

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
    train_cmd = train_cmd or [sys.executable, WRAPPER, "--recipe", RECIPE, "--model", "wwe_uie", "--seed", str(seed), "--run_name", RUN]
    gate_cmd = gate_cmd or [sys.executable, GATE, "--run", RUN]
    assert "--resume" not in train_cmd
    from v2_tools import wwe_config as cfg
    log(event="plan", run=RUN, recipe=RECIPE, recipe_args=cfg.RECIPES[RECIPE], seed=seed, precision="fp32", batch_size=4, epochs=100,
        train_command=" ".join(train_cmd[1:]), gate_command=" ".join(gate_cmd[1:]), expected_code_commit=EXPECTED_COMMIT, wwe_commit=WWE_COMMIT,
        env={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda},
        data="train + val_A only (selection per frozen protocol); val_B / UIEB never loaded", gates=GATES, max_wall_s=max_wall_s,
        pilot_note="pilot runs_v2/wwe_pilot_lr2e-4_s0_v1 ran from HEAD 09b4fad with then-uncommitted code; it is not part of this run's provenance",
        gate_timeout_s=gate_timeout_s, process_cleanup="job object (kill-on-close) + taskkill /T /F tree kill on failure/timeout",
        rule="one launch only; any failure stops; no retry, no resume, no second run")
    for name, fn in (checks if checks is not None else default_checks()).items():
        problems = fn()
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
    state["launched"] += 1
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
                reason = scan_history(out_dir / "history.jsonl")
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
    reason = scan_history(out_dir / "history.jsonl")
    if reason:
        return stop(f"{RUN}: {reason}")
    missing = [a for a in ARTIFACTS if not (out_dir / a).exists()]
    if missing:
        return stop(f"{RUN}: expected artifacts missing: {missing}")
    r = json.load(open(out_dir / "result.json"))
    bad = check_result_metadata(r, seed, RUN)
    if bad:
        return stop(f"{RUN}: result.json metadata does not match the requested run: {bad}")
    rec_commit = r.get("env", {}).get("git_commit", "")
    if not rec_commit or not head.startswith(rec_commit):
        return stop(f"{RUN}: recorded git_commit {rec_commit!r} != HEAD at preflight {head[:8]}")
    log(event="training_complete", run=RUN, wall_s=round(time.time() - t0), best_valA_epoch=r.get("best_valA_epoch"), best_valA=r.get("best_valA"), params=r.get("params"),
        recorded_git_commit=rec_commit, run_seconds=r.get("run_seconds"), checkpoint_sha256=r.get("checkpoint_sha256"))
    g_rc, g_out, g_err, timed_out = run_bounded(gate_cmd, root, gate_timeout_s)
    with open(out_dir / "wwe_gate.log", "w") as gf:
        gf.write(g_out + "\n--- stderr ---\n" + g_err + (f"\n--- GATE TIMED OUT after {gate_timeout_s}s; process tree killed ---\n" if timed_out else ""))
    if timed_out:
        return stop(f"{RUN}: validation gate TIMED OUT after {gate_timeout_s}s (process tree killed; see runs_v2/{RUN}/wwe_gate.log); the run is NOT validated")
    if g_rc != 0:
        return stop(f"{RUN}: validation gate FAILED (rc={g_rc}; see runs_v2/{RUN}/wwe_gate.log)")
    log(event="gate_passed", run=RUN)
    log(event="all_done", runs=[RUN], note="single run complete; nothing further launched")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, help=f"seed to train; only {list(ALLOWED_SEEDS)} are accepted (seed 0 is the completed first run)")
    ap.add_argument("--preflight_only", action="store_true", help="run the preflight (reads train/val_A manifest JSON only, no images) and exit; launches nothing")
    a = ap.parse_args()
    try:
        run_name(a.seed)
    except ValueError as e:
        ap.error(str(e))  # exit code 2, nothing created
    sys.exit(run_single(a.seed, preflight_only=a.preflight_only))


if __name__ == "__main__":
    main()
