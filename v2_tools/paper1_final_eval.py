"""Paper 1 final-evaluation script (PREPARED, NOT RUN). Frozen evaluator: Resize((256,256)) -> [0,1], clamp, TF32 off, PSNR/SSIM from tools.eval_reproduce_baseline (via train_v2).
Fail-closed data policy: the only dataset this version can load is Val-A (manifest hash prefix 2d8daae6). train_v2.MANIFESTS is reduced to {val_A} BEFORE any manifest is hashed or read, so Val-B,
train, val_368, val_clean and UIEB are never opened. Any loaded path containing 'uieb' or any manifest key other than val_A aborts. `--dataset uieb` always aborts here: a UIEB loader will only be
added, and shown for review, after the user supplies the exact phrase `UIEB approved`.
Per model: best_valA.pth only. All SHA-256 are checked against runs_v2/<run>/result.json BEFORE any checkpoint is deserialised; a mismatch aborts (no substitution).
Metrics: PSNR, SSIM (frozen) and, as a pre-declared SECONDARY metric, LPIPS(net='alex', version='0.1', fp32, eval, spatial=False, inputs scaled to [-1,1], lower is better; lpips package version and
SHA-256 of the AlexNet backbone and linear-head weights are recorded). LPIPS never replaces or alters the primary metrics.
Statistics (exploratory on Val-A): paired per-image differences, mean/median, win/loss/tie (tie: |d| <= 1e-9), bootstrap 95% CI over images (10,000 resamples, seed 20261028), per-seed comparisons;
no significance claim across training seeds. Supplementary: per-seed Wilcoxon signed-rank (image level, exploratory, uncorrected).
Output directory is created with exist_ok=False and every file is opened with mode 'x'.
Usage (dry run, after review):  python v2_tools/paper1_final_eval.py --dataset val_A --out reports/data/paper1_final_eval_dryrun_valA [--models all|<comma list>]"""
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VAL_A_PREFIX = "2d8daae6"
UIEB_PHRASE = "UIEB approved"
BOOT_N, BOOT_SEED, TIE_EPS = 10_000, 20261028, 1e-9
LSUI_ROOT = ROOT / "datasets" / "source_copies" / "LSUI"  # the LSUI image root recorded in the Val-A manifest; no UIEB path may live under it
LEDGER = ROOT / "reports" / "data" / "paper1_uieb_run_ledger.jsonl"
STATE = {"out": None, "started": False, "res": None}  # filled by _run so that guarded() can record diagnostics on failure
UIEB_PROTOCOL_NOTE = ("CANDIDATE protocol, not final. Both images of every pair are converted to RGB and squashed to 256x256 with the same bilinear Resize (aspect ratio NOT preserved): "
                      "a distorted fixed-resolution evaluation, not native-resolution UIEB evaluation. Scores are not comparable with published UIEB numbers computed at native or other resolutions.")
UIEB_STATUS = ("Cross-dataset evaluation of models trained on LSUI only (WWE-UIE and WaterNet are our LSUI retrainings, not released checkpoints), one trained model per seed, under the candidate "
               "fixed 256x256 protocol. The numbers describe these trained models on this image set under this protocol; they do not by themselves show generalisation across datasets or across training "
               "seeds. Wilcoxon p-values are image-level, uncorrected and supplementary. No state-of-the-art, causal-mechanism or efficiency claim is made.")
SEEDS = (0, 1, 2)

# key -> (run directory name, class path). Class import is lazy.
MODELS = {}
for s in SEEDS:
    MODELS[f"marinenet_bn_s{s}"] = (f"marinenet_bn_r3_fp32_s{s}", "models.marine_net_in:MARINENet")
    MODELS[f"marinenet_in_s{s}"] = (f"marinenet_in_r3_fp32_s{s}", "models.marine_net_in:MARINENetIN")
    MODELS[f"attrunk_bn_s{s}"] = (f"xf_attrunk_r3_fp32_s{s}", "models.at_marine:ATMarineTrunk")
    MODELS[f"attrunk_in_s{s}"] = (f"xfin_attrunk_in_r3_fp32_s{s}", "models.at_trunk_in:ATMarineTrunkIN")
    MODELS[f"armcin_s{s}"] = (f"armcin_idres_r3_fp32_s{s}", "models.armc_idres_in:ATMarineIdResIN")
MODELS["wwe_uie_s0"] = ("wwe_lr2e-4_s0", "models.wwe_uie:WWEUIE")
MODELS["waternet_s0"] = ("waternet_lsui_s0", "models.waternet_wrap:WaterNetWrap")

# (name, tier, a, b): difference = a - b, per image, paired by file. Tier labels follow the pre-registered protocol.
COMPARISONS = [(f"primary: MARINE-Net IN - BN, seed {s}", "primary", f"marinenet_in_s{s}", f"marinenet_bn_s{s}") for s in SEEDS] + \
              [(f"secondary: AT-Trunk IN - BN, seed {s}", "secondary", f"attrunk_in_s{s}", f"attrunk_bn_s{s}") for s in SEEDS] + \
              [(f"optional ablation: Arm C-IN - AT-Trunk IN, seed {s}", "ablation", f"armcin_s{s}", f"attrunk_in_s{s}") for s in SEEDS]
SEED_MEAN_GROUPS = [("primary: MARINE-Net IN - BN, per-image mean over seeds 0-2", "primary", "marinenet_in", "marinenet_bn"),
                    ("secondary: AT-Trunk IN - BN, per-image mean over seeds 0-2", "secondary", "attrunk_in", "attrunk_bn")]


# ----------------------------------------------------------------------------- pure helpers (unit-tested on CPU)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_dataset_policy(dataset, uieb_phrase=None, uieb_args=None):
    """Fail closed. 'val_A' is always allowed. 'uieb' needs the exact phrase AND all user-supplied: manifest path, expected manifest sha256 prefix (>= 16 hex), expected pair count (> 0),
    and a path marker substring that every UIEB image path must contain. UIEB arguments on any other dataset are refused."""
    given = [x for x in (uieb_args or ()) if x not in (None, "")]
    if dataset != "uieb" and (uieb_phrase is not None or given):
        raise SystemExit("STOP: UIEB arguments were given for a non-UIEB run")
    if dataset == "uieb":
        if uieb_phrase != UIEB_PHRASE:
            raise SystemExit("STOP: UIEB needs the exact authorization phrase from the user (--uieb-phrase)")
        m, h, n, mk = (tuple(uieb_args) + (None,) * 4)[:4] if uieb_args else (None,) * 4
        if not m or not h or len(h) < 16 or any(c not in "0123456789abcdef" for c in h.lower()) or not isinstance(n, int) or n <= 0 or not mk:
            raise SystemExit("STOP: UIEB needs --uieb-manifest, --uieb-manifest-sha256 (>=16 hex chars), --uieb-expected-pairs (>0) and --uieb-path-marker, all supplied by the user")
        return
    if dataset != "val_A":
        raise SystemExit(f"STOP: dataset {dataset!r} is not allowed (only val_A, or uieb with authorization)")


def restrict_manifests(MANIFESTS):
    """Delete every manifest key except val_A in place, then assert."""
    for k in [k for k in MANIFESTS if k != "val_A"]:
        del MANIFESTS[k]
    if sorted(MANIFESTS) != ["val_A"]:
        raise SystemExit(f"STOP: manifest keys {sorted(MANIFESTS)} != ['val_A']")
    if "uieb" in str(MANIFESTS["val_A"][0]).lower() or not MANIFESTS["val_A"][2] == VAL_A_PREFIX:
        raise SystemExit("STOP: val_A manifest entry unexpected")


def assert_no_reserved_paths(pairs):
    for p in pairs:
        for k in ("input_path", "gt_path"):
            if "uieb" in str(p[k]).lower():
                raise SystemExit(f"STOP: reserved-data path in pairs: {p[k]}")


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def validate_pairs(pairs, expected_n, check_files=True):
    """Strict pair validation; returns a list of error strings (empty == OK). Touches the filesystem only via os.path.isfile when check_files.
    Rules: exact pair count; every item has non-empty str input_path and gt_path; input and GT basenames identical (case-sensitive); no duplicate basenames,
    input paths or GT paths (after normalisation); input path != GT path; (optional) both files exist."""
    errs = []
    if len(pairs) != expected_n:
        errs.append(f"pair count {len(pairs)} != expected {expected_n}")
    names, ins, gts = {}, {}, {}
    for i, p in enumerate(pairs):
        ip = p.get("input_path") if isinstance(p, dict) else None
        gp = p.get("gt_path") if isinstance(p, dict) else None
        if not (isinstance(ip, str) and ip and isinstance(gp, str) and gp):
            errs.append(f"item {i}: missing/empty input_path or gt_path")
            continue
        bi, bg = os.path.basename(ip), os.path.basename(gp)
        if bi != bg:
            errs.append(f"item {i}: filename mismatch {bi!r} vs {bg!r}")
        if _norm(ip) == _norm(gp):
            errs.append(f"item {i}: input_path == gt_path")
        for d, key, what in ((names, bi, "basename"), (ins, _norm(ip), "input_path"), (gts, _norm(gp), "gt_path")):
            if key in d:
                errs.append(f"item {i}: duplicate {what} {key!r} (also item {d[key]})")
            else:
                d[key] = i
        if check_files:
            for what, path in (("input", ip), ("gt", gp)):
                if not os.path.isfile(path):
                    errs.append(f"item {i}: {what} file missing {path!r}")
    return errs


def assert_uieb_paths(pairs, marker, lsui_root=None):
    """UIEB run: every path must contain the user-supplied marker (case-insensitive) and must not lie under the LSUI image root (guards against a wrong manifest)."""
    root = _norm(lsui_root if lsui_root is not None else LSUI_ROOT)
    for p in pairs:
        for k in ("input_path", "gt_path"):
            if marker.lower() not in str(p[k]).lower():
                raise SystemExit(f"STOP: path does not contain the UIEB marker {marker!r}: {p[k]}")
            if _norm(p[k]).startswith(root + os.sep):
                raise SystemExit(f"STOP: UIEB path lies under the LSUI image root: {p[k]}")


def load_uieb_manifest(path, sha_prefix, expected_n, marker, lsui_root=None):
    """Called ONLY after authorization and checkpoint verification. Hash first; then parse; key 'test_pairs' (fallback 'pairs') as in tools/eval_reproduce_baseline.py."""
    h = sha256_file(path)
    if not h.startswith(sha_prefix.lower()):
        raise SystemExit(f"STOP: UIEB manifest sha256 {h} does not start with the expected {sha_prefix}")
    data = json.load(open(path, encoding="utf-8"))
    pairs = data.get("test_pairs") or data.get("pairs")
    if not isinstance(pairs, list):
        raise SystemExit("STOP: UIEB manifest has neither 'test_pairs' nor 'pairs' list")
    errs = validate_pairs(pairs, expected_n, check_files=True)
    if errs:
        raise SystemExit("STOP: UIEB pair validation failed (%d errors); first: %s" % (len(errs), "; ".join(errs[:10])))
    assert_uieb_paths(pairs, marker, lsui_root)
    return pairs, h


def header_report(pairs):
    """Image headers only (PIL lazy open, no pixel decode): sizes and modes of input and GT, to disclose resizing / aspect / colour-mode effects."""
    from PIL import Image
    rows = []
    for p in pairs:
        with Image.open(p["input_path"]) as a, Image.open(p["gt_path"]) as b:
            rows.append({"filename": os.path.basename(p["input_path"]), "in_w": a.size[0], "in_h": a.size[1], "in_mode": a.mode, "gt_w": b.size[0], "gt_h": b.size[1], "gt_mode": b.mode})
    return rows


def header_summary(rows):
    return {"n": len(rows), "size_mismatch_input_vs_gt": sum(1 for r in rows if (r["in_w"], r["in_h"]) != (r["gt_w"], r["gt_h"])),
            "non_rgb_input": sum(1 for r in rows if r["in_mode"] != "RGB"), "non_rgb_gt": sum(1 for r in rows if r["gt_mode"] != "RGB"),
            "already_256x256": sum(1 for r in rows if (r["in_w"], r["in_h"]) == (256, 256) and (r["gt_w"], r["gt_h"]) == (256, 256)),
            "non_square_input": sum(1 for r in rows if r["in_w"] != r["in_h"])}


def ledger_read(ledger):
    if not Path(ledger).exists():
        return []
    return [json.loads(x) for x in open(ledger, encoding="utf-8") if x.strip()]


def ledger_attempts(ledger, dataset):
    """[(out, status)] for every started attempt; status is completed / failed / interrupted (a start with no end event, e.g. a killed process)."""
    recs = [r for r in ledger_read(ledger) if r.get("dataset") == dataset]
    res = []
    for r in recs:
        if r["event"] == "started":
            end = [e["event"] for e in recs if e["event"] in ("completed", "failed") and e.get("out") == r.get("out")]
            res.append((r.get("out"), end[-1] if end else "interrupted"))
    return res


def ledger_precheck(ledger, dataset, allow_rerun, reason):
    """No file is written. An ordinary run is refused if any attempt (completed, failed or interrupted) exists; a repeat needs --disclosed-rerun AND a non-empty --rerun-reason,
    and always a NEW output directory (exclusive create). Returns the number of earlier attempts."""
    att = ledger_attempts(ledger, dataset)
    if att and not allow_rerun:
        raise SystemExit(f"STOP: earlier {dataset} attempt(s) exist {att}; a repeat needs --disclosed-rerun --rerun-reason '...' and a new --out directory")
    if allow_rerun and not (reason and reason.strip()):
        raise SystemExit("STOP: --disclosed-rerun needs a non-empty --rerun-reason")
    return len(att)


def ledger_append(ledger, rec):
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def guarded(fn, state, ledger, dataset):
    """Run fn(). On ANY exception (including SystemExit / KeyboardInterrupt) write failure.json into the run's output directory (exclusive create; no summary.json is ever written for a
    failed run) and append a 'failed' ledger line, then re-raise. On success append 'completed'. A start line with neither is an interrupted attempt; nothing is ever erased."""
    now = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    try:
        r = fn()
    except BaseException as e:  # noqa: BLE001
        info = {"event": "failed", "dataset": dataset, "out": str(state.get("out")), "error_type": type(e).__name__, "error": str(e)[:2000], "time": now()}
        if state.get("out") is not None:
            try:
                write_json_x(Path(state["out"]) / "failure.json", dict(info, models_scored_so_far=state.get("res"),
                                                                        note="FAILED RUN: diagnostics only. No summary.json exists for this attempt; do not use these numbers as results."))
            except Exception as e2:  # noqa: BLE001
                info["failure_json_error"] = repr(e2)
        if state.get("started"):
            ledger_append(ledger, info)
        raise
    if state.get("started"):
        ledger_append(ledger, {"event": "completed", "dataset": dataset, "out": str(state.get("out")), "time": now()})
    return r


def verify_checkpoint(run_dir, root=None):
    """Return (path, sha) after comparing the file hash with result.json's recorded best_valA.pth hash; abort on mismatch."""
    root = root or (ROOT / "runs_v2")
    d = Path(root) / run_dir
    rec = json.load(open(d / "result.json"))["checkpoint_sha256"]["best_valA.pth"]
    got = sha256_file(d / "best_valA.pth")
    if got != rec:
        raise SystemExit(f"STOP: {run_dir} best_valA.pth sha256 {got} != recorded {rec}")
    return d / "best_valA.pth", got


def exclusive_dir(path):
    Path(path).mkdir(parents=True, exist_ok=False)
    return Path(path)


def write_json_x(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def paired_stats(a, b, n_boot=BOOT_N, seed=BOOT_SEED):
    """a, b: per-image metric arrays (same order). Difference d = a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "boot_ci95_mean": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "wins": int((d > TIE_EPS).sum()), "losses": int((d < -TIE_EPS).sum()), "ties": int((np.abs(d) <= TIE_EPS).sum())}


def wilcoxon_stats(a, b):
    """SUPPLEMENTARY, EXPLORATORY. Wilcoxon signed-rank test (two-sided, scipy default zero_method='wilcox': zero differences dropped) on per-image paired differences a - b.
    The unit is the IMAGE for ONE trained model pair. It does not establish generalisation across training seeds (3 seeds) or to other data; the p-value is not corrected for multiplicity."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = int((d != 0).sum())
    if nz == 0:
        return {"statistic": None, "p_value": None, "n_nonzero": 0, "note": "all differences zero"}
    r = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue), "n_nonzero": nz, "scope": "image-level, this seed pair only; exploratory"}


def count_nonfinite(t):
    import torch
    return int((~torch.isfinite(t)).sum().item())


def lpips_settings():
    return {"net": "alex", "version": "0.1", "spatial": False, "eval": True, "dtype": "float32", "input_range": "[-1,1] (x*2-1) on the 256x256 resized images", "direction": "lower is better",
            "role": "pre-declared secondary metric; does not replace or change PSNR/SSIM"}


# ----------------------------------------------------------------------------- GPU / data parts (not exercised by the CPU tests)
def load_model(key, ckpt, device):
    import importlib, torch
    mod, cls = MODELS[key][1].split(":")
    m = getattr(importlib.import_module(mod), cls)()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    return m.to(device).eval()


def build_lpips(device):
    import lpips, torch
    net = lpips.LPIPS(net="alex", version="0.1", pretrained=True, pnet_rand=False, verbose=False).to(device).eval()
    head = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
    back = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    info = dict(lpips_settings(), lpips_version=importlib.metadata.version("lpips"), linear_head_file=str(head), linear_head_sha256=sha256_file(head),
                backbone_file=str(back), backbone_sha256=sha256_file(back))
    return net, info


def score_model(model, X, Y, device, lp, bs=16):
    import torch
    import train_v2 as tv
    torch.backends.cudnn.allow_tf32 = False
    ps, ss, ls, nonfinite = [], [], [], 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device).float() / 255.0
            y = Y[i:i + bs].to(device).float() / 255.0
            raw = model(x)["enhanced"]
            nonfinite += count_nonfinite(raw)
            out = torch.clamp(raw, 0.0, 1.0)
            for j in range(len(x)):
                ps.append(tv.compute_psnr(out[j:j + 1], y[j:j + 1])); ss.append(tv.compute_ssim(out[j:j + 1], y[j:j + 1]))
            if lp is not None:
                ls += lp(out * 2 - 1, y * 2 - 1).flatten().float().cpu().tolist()
    return np.array(ps), np.array(ss), (np.array(ls) if lp is not None else None), nonfinite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="all"); ap.add_argument("--no-lpips", action="store_true"); ap.add_argument("--uieb-phrase", default=None)
    ap.add_argument("--uieb-manifest", default=None); ap.add_argument("--uieb-manifest-sha256", default=None); ap.add_argument("--uieb-expected-pairs", type=int, default=None)
    ap.add_argument("--disclosed-rerun", action="store_true"); ap.add_argument("--rerun-reason", default=None); ap.add_argument("--uieb-path-marker", default=None)
    a = ap.parse_args()
    check_dataset_policy(a.dataset, a.uieb_phrase, (a.uieb_manifest, a.uieb_manifest_sha256, a.uieb_expected_pairs, a.uieb_path_marker))
    if a.dataset == "uieb":
        guarded(lambda: _run(a), STATE, LEDGER, "uieb")
    else:
        _run(a)


def _run(a):
    is_uieb = a.dataset == "uieb"
    import torch
    import train_v2 as tv
    restrict_manifests(tv.MANIFESTS)  # BEFORE any manifest is hashed or read
    keys = list(MODELS) if a.models == "all" else a.models.split(",")
    assert all(k in MODELS for k in keys), keys
    if not is_uieb:
        mp = tv.MANIFESTS["val_A"][0]
        mh = sha256_file(mp); assert mh.startswith(VAL_A_PREFIX), mh
    ckpts = {k: verify_checkpoint(MODELS[k][0]) for k in keys}  # all hashes verified before any torch.load
    if is_uieb:  # first UIEB access in the whole script: after authorization AND after all checkpoint hashes are verified
        pairs, mh = load_uieb_manifest(a.uieb_manifest, a.uieb_manifest_sha256, a.uieb_expected_pairs, a.uieb_path_marker)
        mp = Path(a.uieb_manifest)
        out_path = ROOT / a.out
        n_prior = ledger_precheck(LEDGER, "uieb", a.disclosed_rerun, a.rerun_reason)  # writes nothing
        out = exclusive_dir(out_path); STATE["out"] = str(out)  # a pre-existing directory raises here, before any ledger line is written
        ledger_append(LEDGER, {"event": "started", "dataset": "uieb", "out": str(out), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "earlier_attempts": n_prior,
                               "rerun_reason": a.rerun_reason, "script_sha256": sha256_file(__file__), "manifest_sha256": mh})
        STATE["started"] = True
        hdr = header_report(pairs)
        with open(out / "pairs_meta.csv", "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hdr[0])); w.writeheader(); w.writerows(hdr)
        hdr_sum = header_summary(hdr)
    else:
        out = exclusive_dir(ROOT / a.out)  # created only after every checkpoint hash is verified
        pairs = tv.load_pairs("val_A"); assert len(pairs) == 166; assert_no_reserved_paths(pairs)
        hdr_sum = None
    X, Y = tv.preload(pairs)  # identical Resize((256,256)) + convert("RGB") for input and reference (same loop, same transform object)
    assert X.shape == Y.shape == (len(pairs), 3, 256, 256) and X.dtype == Y.dtype == torch.uint8, (X.shape, Y.shape)
    dev = torch.device("cuda")
    lp, lp_info = (None, None) if a.no_lpips else build_lpips(dev)
    res, per = {}, {}
    STATE["res"] = res  # diagnostics for failure.json
    for k in keys:
        m = load_model(k, ckpts[k][0], dev)
        ps, ss, ls, nf = score_model(m, X, Y, dev, lp)
        res[k] = {"run": MODELS[k][0], "checkpoint_sha256": ckpts[k][1], "psnr": float(ps.mean()), "ssim": float(ss.mean()), "lpips": (float(ls.mean()) if ls is not None else None),
                  "nonfinite_before_clamp": nf}
        if is_uieb:  # no logged UIEB scores exist; the acceptance check is finiteness only
            res[k]["matches_logged"] = None
            res[k]["finite_ok"] = bool(nf == 0 and np.isfinite(ps).all() and np.isfinite(ss).all() and (ls is None or np.isfinite(ls).all()))
            if not res[k]["finite_ok"]:
                raise SystemExit(f"STOP: non-finite outputs/scores for {k} (nonfinite_outputs={nf}, nonfinite_psnr={int((~np.isfinite(ps)).sum())}, nonfinite_ssim={int((~np.isfinite(ss)).sum())}); failure.json written, no summary.json")
        else:
            logged = json.load(open(ROOT / "runs_v2" / MODELS[k][0] / "result.json"))["best_valA"]
            res[k].update({"logged_psnr": logged["psnr_mean"], "logged_ssim": logged["ssim_mean"],
                           "matches_logged": bool(abs(ps.mean() - logged["psnr_mean"]) < 1e-3 and abs(ss.mean() - logged["ssim_mean"]) < 1e-4 and nf == 0)})
        per[k] = (ps, ss, ls)
        print(k, {x: (round(y, 5) if isinstance(y, float) else y) for x, y in res[k].items() if x not in ("run", "checkpoint_sha256")}, flush=True)
        del m; torch.cuda.empty_cache()
    with open(out / "per_image.csv", "x", newline="") as f:
        w = csv.writer(f); cols = [f"{k}_{n}" for k in keys for n in (("psnr", "ssim") + (() if lp is None else ("lpips",)))]
        w.writerow(["filename"] + cols)
        for i, p in enumerate(pairs):
            row = []
            for k in keys:
                row += [f"{per[k][0][i]:.6f}", f"{per[k][1][i]:.6f}"] + ([] if lp is None else [f"{per[k][2][i]:.6f}"])
            w.writerow([p["filename"]] + row)
    comps = []
    for name, tier, ka, kb in COMPARISONS:
        if ka in per and kb in per:
            comps.append({"name": name, "tier": tier, **{met: dict(paired_stats(per[ka][j], per[kb][j]), wilcoxon=wilcoxon_stats(per[ka][j], per[kb][j])) for j, met in enumerate(("psnr", "ssim"))},
                          **({} if lp is None else {"lpips": dict(paired_stats(per[ka][2], per[kb][2]), wilcoxon=wilcoxon_stats(per[ka][2], per[kb][2]))})})
    for name, tier, ga, gb in SEED_MEAN_GROUPS:
        ka, kb = [f"{ga}_s{s}" for s in SEEDS], [f"{gb}_s{s}" for s in SEEDS]
        if all(k in per for k in ka + kb):
            mean = lambda ks, j: np.mean([per[k][j] for k in ks], axis=0)  # noqa: E731
            comps.append({"name": name, "tier": tier, "note": "descriptive; per-image average over 3 training seeds, not a seed-level significance test",
                          **{met: paired_stats(mean(ka, j), mean(kb, j)) for j, met in enumerate(("psnr", "ssim"))}})
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": a.dataset, "n_images": len(pairs), **{("uieb_" if is_uieb else "val_A_") + "manifest_sha256": mh, ("uieb_" if is_uieb else "val_A_") + "manifest_path": str(mp)}, **({"uieb_header_summary": hdr_sum} if is_uieb else {}),
            "script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha256_file(__file__), "command": " ".join(sys.argv),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "git_dirty_tracked": subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines(),
            "torch": torch.__version__, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "preprocessing": "Resize((256,256)) of input and ground truth, ToTensor, clamp [0,1]; frozen PSNR/SSIM", "lpips": lp_info,
            "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "unit": "image", "ci": "percentile 2.5/97.5 of the mean paired difference"}, "tie_eps": TIE_EPS,
            "status": "Val-A is the checkpoint-selection set: all statistics are exploratory. Image-level Wilcoxon p-values (per seed, supplementary, uncorrected) do not establish generalisation across training seeds; no significance claim is made."}
    if is_uieb:
        meta['status'] = UIEB_STATUS
        meta['protocol_note'] = UIEB_PROTOCOL_NOTE
        meta['candidate_protocol'] = True
    write_json_x(out / "summary.json", {"meta": meta, "models": res, "comparisons": comps})
    print("all_match_logged", None if is_uieb else all(r["matches_logged"] for r in res.values()), "written", out)


if __name__ == "__main__":
    main()
