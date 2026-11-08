"""Single-run gated driver for experiment MARINENETMS: MARINE-Net under the XF / XFIN fp32 recipe (R3 loss, fp32, cosine floor eta_min 2e-8) in two variants - 'bn' (original architecture, 40 BatchNorm2d) and 'in'
(its 40 BatchNorm2d replaced by full-channel InstanceNorm2d) - for one seed in {0, 1, 2}. Sibling of v2_tools/run_armcinms_single.py (NOT modified); the XF driver helpers are reused unchanged. NOT approved to run.
One invocation = ONE run: `--variant {bn,in} --seed {0,1,2}`. The run name is fixed per variant and seed (marinenet_bn_r3_fp32_s{n} / marinenet_in_r3_fp32_s{n}), so no existing run directory, log or gate file can
be touched. Sequence: preflight -> ONE training process -> artifact + metadata checks -> training_complete -> provenance re-check -> gate (bounded by a timeout) -> gate_passed -> all_done.
Exactly one launch per invocation: no retry, no resume (--resume is never passed), no second run; ANY failure ends the driver with exit 1 and a STOP record; nothing is cleaned up.
Safeguards (as the ARMCINMS driver, with the model, references and checks changed):
  * exclusive creation of the driver log and of the preflight log (timestamp + process id); the output directory must not exist;
  * provenance preflight: the index is empty; the new tooling files, the XFINMS config files, the XFIN files and the other code dependencies are TRACKED and unmodified; the XF, XFIN and XFINMS commits are
    ancestors of HEAD and the files they hold equal those commits; external/WWE-UIE is clean at its pinned commit;
  * R3 provenance (fail closed): original_repo/marine_net_model.py is git-ignored, so its SHA-256 is PINNED and verified in the preflight, on EVERY poll while training runs (a change kills the process tree),
    once more after training, and by the gate;
  * recipe + model preflight: the recipe equals the XF recipe (R3 except amp False and eta_min 2e-8); MARINENet and MARINENetIN build on the CPU with 2,609,401 trainable parameters, 40 BatchNorm2d / 40
    InstanceNorm2d respectively, the expected state-dict element counts, and bit-identical convolution / linear / INR initialisation for the same seed (no data access);
  * the AT-Trunk reference of the same seed (XF for bn, XFIN for in) is read ONLY through its PINNED SHA-256; no other result.json is opened;
  * train and Val-A manifests only (hash + strict pair validation, every other manifest key dropped first); result.json must hold exactly those two hashes;
  * fp32 history scan: ANY non-finite loss step, ANY non-finite-gradient step, a non-finite loss / gradient norm / Val-A value stops the run;
  * every external command is bounded (git calls, the GPU probe, the gate 30 min, training by the wall limit); a preflight check that raises is a STOP;
  * Windows fail-closed process cleanup: the training process runs inside a kill-on-close Job Object; if the job cannot be created or the process cannot be assigned, nothing is launched / the just-started tree is killed;
  * provenance RE-CHECK immediately before the gate (the launch preflight's git checks plus HEAD unchanged): any failure is a STOP, the gate is not launched and the run is NOT validated;
  * wall-time limit 10 h (a safeguard, not a duration target; the estimate below is an extrapolation); the gate runs under a 30 min timeout;
  * result.json must describe this run (seed, run name, recipe, model, params, fp32, eta_min, no resume, exact manifest scope) and the clip log must exist before training_complete is logged.
No decision rule is applied or logged (Option A: descriptive reporting only).
Events: plan, preflight_ok*, preflight_done, start, tree_guard, training_complete (NOT a success marker), provenance_recheck_ok, gate_passed, all_done (the only success event). STOP on any failure.
Usage (NOT approved to run): python v2_tools/run_marinenetms_single.py --variant {bn,in} --seed {0,1,2} [--preflight_only]"""
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
from v2_tools import marinenetms_config as mc  # noqa: E402
from v2_tools import marinenetms_gate as mg  # noqa: E402
from v2_tools import run_wwe_single as rw  # noqa: E402
from v2_tools import run_xf_single as rf  # noqa: E402
from v2_tools import xf_config as xf  # noqa: E402
from v2_tools.run_wwe_single import kill_tree, make_kill_on_close_job, popen_kwargs, run_bounded  # noqa: E402

WRAPPER, GATE = "v2_tools/train_v2_no_valb_marinenetms.py", "v2_tools/marinenetms_gate.py"
ARTIFACTS = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", mc.LOSS_CONFIG_FILE, mc.CLIP_LOG_FILE, "train_attempt0.log"]
GATES = list(mg.CHECK_NAMES)
PROVENANCE_FILES = list(mc.PROVENANCE_FILES)
MAX_WALL_S = rf.MAX_WALL_S  # 10 h safeguard, not a target
POLL_S = rf.POLL_S
GATE_TIMEOUT_S = rf.GATE_TIMEOUT_S
REQUIRE_JOB = os.name == "nt"  # fail closed on Windows if the Job Object cannot be created / assigned (tests may force it on other platforms)

scan_history_fp32 = rf.scan_history_fp32
git = rf.git
check_gpu_free = rf.check_gpu_free
check_disk_and_cache, check_manifests = rf.check_disk_and_cache, rf.check_manifests


def commands(variant, seed):
    """-> (training command, gate command) for this variant and seed; both name the dedicated run."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    return ([sys.executable, WRAPPER, "--recipe", mc.RECIPE_NAME[variant], "--model", mc.MODEL[variant], "--seed", str(seed), "--run_name", mc.run_name(variant, seed)],
            [sys.executable, GATE, "--run", mc.run_name(variant, seed)])


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
    for label, commit, files in mc.PINNED_COMMITS:
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


def provenance_recheck(head):
    """Problems (list of str) for the re-check run IMMEDIATELY BEFORE the gate: every check of the launch preflight (check_git) AND HEAD unchanged since the preflight. A git call that raises is a problem (fail closed)."""
    try:
        out = list(check_git())
        now = git("rev-parse", "HEAD").stdout.strip()
    except Exception as e:  # noqa: BLE001  a check that cannot be evaluated is a failed check
        return [f"provenance re-check raised {type(e).__name__}: {e}"]
    if now != head:
        out.append(f"HEAD changed since the preflight ({str(head)[:8]} -> {now[:8]})")
    return out


def check_original_dependency():
    """Fail closed: the git-ignored R3 loss file must be readable and have the pinned SHA-256."""
    return mc.verify_original_dependency()


def check_config_and_model(variant):
    """No file is read: the recipe is compared with train_v2.RECIPES['R3'] and the XF recipe, and both MARINE-Net models are built on the CPU (structure, parameter / buffer counts, shared initialisation)."""
    import torch
    import train_v2 as tv
    from models.marine_net_in import MARINENet, MARINENetIN, model_problems
    mc.validate_variant(variant)
    out = []
    recipe = mc.recipe_args(tv.RECIPES)
    want = dict(epochs=100, warmup=3, bs=4, accum=1, lr=2e-4, wd=1e-5, clip=1.0, amp=False, loss_in_autocast=False, no_jitter=True, select_on="val_A", skip_zero_weight_perceptual=True)
    bad = {k: recipe.get(k) for k, v in want.items() if recipe.get(k) != v}
    if bad:
        out.append(f"the MARINENETMS recipe values differ from the protocol: {bad}")
    if abs(recipe.get("eta_min", 0) - 2e-8) > 1e-15:
        out.append(f"eta_min {recipe.get('eta_min')!r} != 2e-8")
    if recipe != xf.recipe_args(tv.RECIPES):
        out.append("the MARINENETMS recipe differs from the XF recipe")
    if set(mc.recipe_diff(tv.RECIPES["R3"], recipe)) != set(xf.R3_DEVIATIONS):
        out.append(f"the recipe differs from R3 in {sorted(mc.recipe_diff(tv.RECIPES['R3'], recipe))}, expected {sorted(xf.R3_DEVIATIONS)}")
    if tv.MODEL_REGISTRY["marine_default"]().__class__ is not tv.MARINENet:
        out.append("the train_v2 marine_default registry entry is not MARINENet")
    torch.manual_seed(0)
    bn = MARINENet()
    torch.manual_seed(0)
    inn = MARINENetIN()
    out += [f"MARINENet: {p}" for p in model_problems(bn, "bn")]
    out += [f"MARINENetIN: {p}" for p in model_problems(inn, "in")]
    sb, si = bn.state_dict(), inn.state_dict()
    shared = [k for k in si if k in sb]
    diff = [k for k in shared if not torch.equal(sb[k], si[k])]
    if diff:
        out.append(f"{len(diff)} parameter(s) differ between the BatchNorm and InstanceNorm model for the same seed, e.g. {diff[:3]}")
    if set(sb) - set(si) and any(not k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in set(sb) - set(si)):
        out.append("the BatchNorm model has keys other than running statistics that the InstanceNorm model lacks")
    return out


def _load_reference(p, expected_sha256, describe):
    """Problems for one pinned reference result.json: it must exist, have the PINNED SHA-256 and describe its own run. Only this one file is read."""
    if not p.is_file():
        return [f"{p.parent.name}/result.json is missing: the reference run must be present"]
    problems = mc.check_reference_hash(p, expected_sha256)
    if problems:
        return problems
    try:
        r = json.load(open(p))
    except (OSError, ValueError) as e:
        return [f"{p.parent.name}/result.json is unreadable ({type(e).__name__})"]
    bad = describe(r) if isinstance(r, dict) else ["not a JSON object"]
    return [f"{p.parent.name}/result.json does not describe the expected run: {bad}"] if bad else []


def check_reference(variant, seed, root=ROOT, expected_sha256=None):
    """The result.json of the AT-Trunk run of the SAME seed (XF for bn, XFIN for in; train + val_A manifests only) must exist, have the PINNED SHA-256 and describe that run: it is the reference of the gate's args pairing."""
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    expected_sha256 = mc.reference_pin(variant, seed) if expected_sha256 is None else expected_sha256
    return _load_reference(Path(root) / "runs_v2" / mc.reference_run_name(variant, seed) / "result.json", expected_sha256, lambda r: mc.reference_metadata_problems(variant, seed, r))


def default_checks(variant, seed, root=ROOT):
    mc.validate_variant(variant)
    mc.validate_seed(seed)
    return {"git provenance": check_git, "R3 loss file (pinned SHA-256)": check_original_dependency, "config + models (MARINE-Net BN / IN) + XF recipe": lambda: check_config_and_model(variant),
            "AT-Trunk same-seed reference run": lambda: check_reference(variant, seed, root), "disk + vgg cache": check_disk_and_cache, "gpu free memory": check_gpu_free,
            "train/val_A manifests + pairing": check_manifests}


# ----------------------------------------------------------------------------- driver
def run_single(variant, seed, root=ROOT, checks=None, train_cmd=None, gate_cmd=None, max_wall_s=MAX_WALL_S, poll_s=POLL_S, preflight_only=False, head=None, gate_timeout_s=GATE_TIMEOUT_S, verify_dep=None,
               recheck=None):
    mc.validate_variant(variant)  # BEFORE any file is created
    mc.validate_seed(seed)
    RUN = mc.run_name(variant, seed)
    root = Path(root)
    verify_dep = verify_dep or mc.verify_original_dependency
    recheck = recheck or provenance_recheck  # tests inject a fake; production always uses the real git re-check
    # a preflight-only invocation logs to its own file (timestamp + process id), so it can never block (or be mistaken for) the real run's log
    log_path = root / "runs_v2" / (f"{RUN}_preflight_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl" if preflight_only else f"{RUN}_chain_driver.jsonl")
    out_dir = root / "runs_v2" / RUN
    existing = {f"xf_attrunk_r3_fp32_s{s}" for s in mc.ALLOWED_SEEDS} | {f"xfin_attrunk_in_r3_fp32_s{s}" for s in mc.ALLOWED_SEEDS} | {f"armcin_idres_r3_fp32_s{s}" for s in mc.ALLOWED_SEEDS}
    assert out_dir.name not in existing and log_path.name not in {f"{n}_chain_driver.jsonl" for n in existing}, "refusing to touch an existing XF, XFIN or ArmC-IN run"

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
    d_train, d_gate = commands(variant, seed)
    train_cmd = train_cmd or d_train
    gate_cmd = gate_cmd or d_gate
    assert "--resume" not in train_cmd
    try:
        dep_sha = mc.sha256_file(mc.ROOT / mc.UNTRACKED_DEPENDENCY)
    except OSError:
        dep_sha = None
    ref_path = root / "runs_v2" / mc.reference_run_name(variant, seed) / "result.json"
    try:
        obs = mc.sha256_file(ref_path)
    except OSError:
        obs = None
    refs = {"attrunk_same_seed": {"run": ref_path.parent.name, "pinned": mc.reference_pin(variant, seed), "observed": obs, "verified": obs == mc.reference_pin(variant, seed)}}
    norm_text = "the original 40 BatchNorm2d layers" if variant == "bn" else "its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d (affine, no running statistics)"
    log(event="plan", run=RUN, variant=variant, seed=seed, recipe=mc.RECIPE_NAME[variant], recipe_args=mc.recipe_args(tv.RECIPES), differs_from_r3=xf.R3_DEVIATIONS, model=mc.MODEL[variant],
        params=mc.EXPECTED_PARAMS, state_dict_elements=mc.EXPECTED_STATE_ELEMENTS[variant],
        change=f"MARINE-Net (original_repo MARINENet: FAD, PGA, coordinate INR, scalar fusion) with {norm_text}; relative to the AT-Trunk run {mc.reference_run_name(variant, seed)} the architecture family differs "
               "(several components at once) and only the model / run name differ in the recorded args",
        precision="fp32 forward and loss (no AMP, no GradScaler); the tf32 convolution / matmul flags are left at the PyTorch default and recorded in marinenet_loss_config.json",
        batch_size=4, epochs=100,
        loss="R3 loss: the FULL original_repo MARINELoss (auxiliary CNN / INR L1 terms ACTIVE, not wrapped in FinalOutputLoss; ZeroLoss as perceptual term), unchanged, built by the unchanged train_v2.build; weights "
             + json.dumps(mc.R3_WEIGHTS),
        purpose=f"matched MARINE-Net {variant.upper()} baseline at seed {seed} for the 2 x 2 design (architecture x normalisation); descriptive reporting only (Option A): no decision rule, class, threshold or margin; "
                "all Val-A numbers are exploratory",
        decision_rule={"status": "NONE (Option A: descriptive paired reporting only)"}, delta_convention=mc.DELTA_CONVENTION, references=refs, remaining_confounders=mc.REMAINING_CONFOUNDERS,
        clip_log=mc.CLIP_LOG_FILE, train_command=" ".join(train_cmd[1:]), gate_command=" ".join(gate_cmd[1:]), xf_commit=mc.XF_COMMIT, xfin_commit=mc.XFIN_COMMIT, xfinms_commit=mc.XFINMS_COMMIT,
        wwe_commit=rw.WWE_COMMIT, provenance_files=PROVENANCE_FILES,
        untracked_dependency={"path": mc.UNTRACKED_DEPENDENCY, "sha256": dep_sha, "pinned_sha256": mc.PINNED_ORIGINAL_SHA256, "verified": dep_sha == mc.PINNED_ORIGINAL_SHA256,
                              "note": "git-ignored; pinned hash verified in the preflight, on every poll, after training and by the gate"},
        env={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda},
        data="train + val_A only (selection per frozen protocol); val_B / UIEB never loaded; only the pinned AT-Trunk reference result.json is opened", gates=GATES, max_wall_s=max_wall_s,
        gate_timeout_s=gate_timeout_s, estimate_hours={"value": mc.ESTIMATED_RUN_HOURS[variant], "label": "ESTIMATE (extrapolation, not a measurement): " + mc.ESTIMATE_BASIS},
        process_cleanup="job object (kill-on-close, fail closed) + taskkill /T /F tree kill on failure/timeout",
        rule="one launch only; any failure stops; no retry, no resume, no second run; all Val-A numbers are exploratory")
    for name, fn in (checks if checks is not None else default_checks(variant, seed, root)).items():
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
    bad = mc.check_result_metadata(r, variant, seed)
    if bad:
        return stop(f"{RUN}: result.json metadata does not match experiment MARINENETMS: {bad}")
    rec_commit = r.get("env", {}).get("git_commit", "")
    if not rec_commit or not head.startswith(rec_commit):
        return stop(f"{RUN}: recorded git_commit {rec_commit!r} != HEAD at preflight {head[:8]}")
    try:
        lc = json.load(open(out_dir / mc.LOSS_CONFIG_FILE))
        lc_commit, lc_sha = lc["code"]["git_commit_full"], lc["code"]["untracked_dependency_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return stop(f"{RUN}: {mc.LOSS_CONFIG_FILE} is unreadable or incomplete ({type(e).__name__})")
    if lc_commit != head:
        return stop(f"{RUN}: {mc.LOSS_CONFIG_FILE} commit {str(lc_commit)[:8]} != HEAD at preflight {head[:8]}")
    if lc_sha != mc.PINNED_ORIGINAL_SHA256:
        return stop(f"{RUN}: {mc.LOSS_CONFIG_FILE} records an R3 file hash that is not the pinned one")
    log(event="training_complete", run=RUN, variant=variant, seed=seed, wall_s=round(time.time() - t0), best_valA_epoch=r.get("best_valA_epoch"), best_valA=r.get("best_valA"), params=r.get("params"),
        recorded_git_commit=rec_commit, run_seconds=r.get("run_seconds"), checkpoint_sha256=r.get("checkpoint_sha256"))
    # the tooling (the gate included) is read from disk: re-check provenance IMMEDIATELY before the gate; a failure means the run is NOT validated and the gate is never launched
    try:
        prob = list(recheck(head))
    except Exception as e:  # noqa: BLE001  fail closed
        prob = [f"provenance re-check raised {type(e).__name__}: {e}"]
    if prob:
        return stop(f"{RUN}: provenance re-check before the gate failed: {prob}; the gate was NOT launched; the run is NOT validated")
    log(event="provenance_recheck_ok", run=RUN, head=head)
    try:
        g_rc, g_out, g_err, timed_out = run_bounded(gate_cmd, root, gate_timeout_s)
    except Exception as e:  # noqa: BLE001  the gate could not even be started: the run is NOT validated
        return stop(f"{RUN}: validation gate could not be executed ({type(e).__name__}: {e}); the run is NOT validated")
    try:
        with open(out_dir / mc.GATE_LOG_FILE, "x") as gf:
            gf.write(g_out + "\n--- stderr ---\n" + g_err + (f"\n--- GATE TIMED OUT after {gate_timeout_s}s; process tree killed ---\n" if timed_out else ""))
    except FileExistsError:
        return stop(f"{RUN}: {mc.GATE_LOG_FILE} already exists; refusing to overwrite; the run is NOT validated")
    if timed_out:
        return stop(f"{RUN}: validation gate TIMED OUT after {gate_timeout_s}s (process tree killed; see runs_v2/{RUN}/{mc.GATE_LOG_FILE}); the run is NOT validated")
    if g_rc != 0:
        return stop(f"{RUN}: validation gate FAILED (rc={g_rc}; see runs_v2/{RUN}/{mc.GATE_LOG_FILE})")
    log(event="gate_passed", run=RUN)
    log(event="all_done", runs=[RUN], note="single run complete; nothing further launched")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True, choices=list(mc.VARIANTS), help="bn = original MARINE-Net (BatchNorm); in = MARINE-Net with InstanceNorm")
    ap.add_argument("--seed", type=int, required=True, choices=list(mc.ALLOWED_SEEDS))
    ap.add_argument("--preflight_only", action="store_true", help="run the preflight (reads train/val_A manifest JSON only, no images) and exit; launches nothing")
    a = ap.parse_args()
    sys.exit(run_single(a.variant, a.seed, preflight_only=a.preflight_only))


if __name__ == "__main__":
    main()
