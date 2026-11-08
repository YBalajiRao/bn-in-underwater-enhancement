"""Stage 3 driver: 9 training runs, strictly sequential, Recipe R3, selection on val_A.
Order: M1 s0 -> M2 s0 -> M3 s0 -> M1 s1 -> M2 s1 -> M3 s1 -> M1 s2 -> M2 s2 -> M3 s2.
A run whose result.json exists is skipped (driver can be restarted). If a run crashes, it is resumed
from its last.pth (--resume), at most 2 times; every attempt is logged. Before each run: free RAM /
commit / disk are logged. Output: runs_v2/stage3_driver.jsonl (one line per event)."""
import json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MODELS = ["m1_global", "m2_globallocal", "marine_default"]
RUNS = [(m, s) for s in (0, 1, 2) for m in MODELS]
LOG = ROOT / "runs_v2/stage3_driver.jsonl"


def mem():
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
            "$o=Get-CimInstance Win32_OperatingSystem; '{0} {1}' -f $o.FreePhysicalMemory,$o.FreeVirtualMemory"], text=True).split()
        return {"free_ram_gb": round(int(out[0]) / 2**20, 2), "free_commit_gb": round(int(out[1]) / 2**20, 2)}
    except Exception as e:  # noqa: BLE001
        return {"mem_error": repr(e)}


def log(**kw):
    kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def main():
    for model, seed in RUNS:
        name = f"stage3_{model}_s{seed}"
        rd = ROOT / "runs_v2" / name
        if (rd / "result.json").exists():
            log(event="skip_done", run=name); continue
        rd.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "train_v2.py", "--recipe", "R3", "--model", model, "--seed", str(seed), "--run_name", name]
        t0 = time.time()
        for attempt in range(3):
            c = cmd + (["--resume"] if attempt or (rd / "last.pth").exists() else [])
            log(event="start", run=name, attempt=attempt, command=" ".join(c[1:]),
                disk_free_gb=round(shutil.disk_usage(ROOT).free / 2**30, 1), **mem())
            logname = f"train_attempt{attempt}.log"
            k = 1
            while (rd / logname).exists():  # never overwrite a log from an earlier driver invocation
                logname = f"train_attempt{attempt}_{k}.log"; k += 1
            with open(rd / logname, "w") as lf:
                rc = subprocess.call(c, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            if rc == 0 and (rd / "result.json").exists():
                r = json.load(open(rd / "result.json"))
                log(event="done", run=name, attempt=attempt, wall_s=round(time.time() - t0), best_valA_epoch=r["best_valA_epoch"],
                    best_valA=r["best_valA"], train_300=r["train_300_random"], run_seconds=r["run_seconds"],
                    checkpoint=str((rd / "best_valA.pth").relative_to(ROOT)), checkpoint_sha256=r["checkpoint_sha256"]["best_valA.pth"])
                break
            log(event="failed", run=name, attempt=attempt, returncode=rc, has_last=(rd / "last.pth").exists(), **mem())
            if not (rd / "last.pth").exists() and attempt > 0:
                break
            time.sleep(30)
        else:
            log(event="giving_up", run=name)
        if not (rd / "result.json").exists():
            log(event="STOP", reason=f"{name} did not complete after retries; remaining runs not started")
            sys.exit(1)
    log(event="all_done")


if __name__ == "__main__":
    main()
