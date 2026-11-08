"""UIEB <-> LSUI leakage audit (PREPARED, NOT RUN). Implements section 5 of reports/paper1_uieb_evaluation_protocol_proposal.md.

Fail-closed: nothing under the UIEB directories is listed, opened or hashed unless ALL of the following are given by the user:
  --uieb-phrase "UIEB approved" (exact), --uieb-input-dir, --uieb-ref-dir, --uieb-path-marker (substring every UIEB path must contain).
UIEB directories may not lie under the LSUI image root. The output directory must not exist (created with exist_ok=False; files opened with 'x').
No model is loaded and no metric is computed: this script only compares image files.

Two separate methods, separate columns:
  1. Exact duplicates: SHA-256 of file bytes of every UIEB input and reference against every LSUI train-3463 and Val-A input and GT.
     A miss does not exclude re-encoded, resized or cropped copies.
  2. Perceptual screening (PROVISIONAL): dHash 8x9 grayscale LANCZOS (identical to v2_tools/make_val_ab.py), Hamming distance.
     Protocol distance = min(UIEB input vs LSUI inputs, UIEB reference vs LSUI GTs); cross distances (input vs GT, reference vs input) are recorded too.
     Flag if distance <= --thr (default 10, not validated for UIEB). A flag marks a candidate for human review; it does not prove leakage.
No pair is removed: every pair appears in the output with its flags; any 'not flagged' subset is a separate, labelled sensitivity analysis.

Two input modes (exactly one):
  directory mode: --uieb-input-dir, --uieb-ref-dir (strict identical basenames);
  manifest mode:  --uieb-manifest, --uieb-manifest-sha256 (>= 16 hex), --uieb-expected-pairs, --uieb-pairing strict|stem. Loaded by
                  paper1_final_eval_v2.load_uieb_manifest2 (the evaluator's own loader: manifest hash first, relative paths resolved against the project root,
                  pair validation, marker / LSUI-root checks, recorded per-image hashes verified), so audit and evaluation see the identical cohort.
Usage (only after the user writes the phrase):
  python v2_tools/uieb_overlap_audit.py --uieb-phrase "UIEB approved" --uieb-input-dir <dir> --uieb-ref-dir <dir> --uieb-path-marker <s> --out <new dir>
  python v2_tools/uieb_overlap_audit.py --uieb-phrase "UIEB approved" --uieb-manifest <json> --uieb-manifest-sha256 <hex> --uieb-expected-pairs <n> --uieb-pairing stem --uieb-path-marker <s> --out <new dir>"""
import argparse, csv, hashlib, json, platform, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
UIEB_PHRASE = "UIEB approved"
LSUI_ROOT = ROOT / "datasets" / "source_copies" / "LSUI"
LSUI_MANIFESTS = {"train": (ROOT / "datasets/splits/candidate_v1/lsui_train_3463.json", "train_pairs", "b4871a98"),
                  "val_A": (ROOT / "datasets/splits/derived_v3/lsui_val_A.json", "val_pairs", "2d8daae6")}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_THR = 10


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dhash(p):
    """8x9 grayscale LANCZOS dHash, identical to make_val_ab.dh."""
    g = Image.open(p).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.array(g, dtype=np.int32)
    return (a[:, 1:] > a[:, :-1]).flatten()


def check_manifest_policy(phrase, manifest, sha_prefix, expected_n, marker):
    """Manifest mode. Raise SystemExit unless the exact phrase and all user-supplied arguments are present. Touches no UIEB file."""
    if phrase != UIEB_PHRASE:
        raise SystemExit("STOP: the UIEB audit needs the exact authorization phrase from the user (--uieb-phrase)")
    if not manifest or not marker or not sha_prefix or len(sha_prefix) < 16 or any(c not in "0123456789abcdef" for c in sha_prefix.lower())             or not isinstance(expected_n, int) or expected_n <= 0:
        raise SystemExit("STOP: manifest mode needs --uieb-manifest, --uieb-manifest-sha256 (>=16 hex), --uieb-expected-pairs (>0) and --uieb-path-marker")


def check_policy(phrase, input_dir, ref_dir, marker, lsui_root=LSUI_ROOT):
    """Raise SystemExit unless the exact phrase and all user-supplied UIEB arguments are present and safe. Touches no UIEB file."""
    if phrase != UIEB_PHRASE:
        raise SystemExit("STOP: the UIEB audit needs the exact authorization phrase from the user (--uieb-phrase)")
    if not input_dir or not ref_dir or not marker:
        raise SystemExit("STOP: --uieb-input-dir, --uieb-ref-dir and --uieb-path-marker must all be supplied by the user")
    lr = Path(lsui_root).resolve()
    for d in (input_dir, ref_dir):
        r = Path(d).resolve()
        if marker.lower() not in str(r).lower():
            raise SystemExit(f"STOP: UIEB path {r} does not contain the path marker {marker!r}")
        if r == lr or lr in r.parents:
            raise SystemExit(f"STOP: UIEB path {r} lies under the LSUI root {lr}")
    if Path(input_dir).resolve() == Path(ref_dir).resolve():
        raise SystemExit("STOP: UIEB input and reference directories are identical")


def pair_dirs(input_dir, ref_dir):
    """Strict pairing: identical, case-sensitive basenames; every file in either directory must have a partner. Returns sorted [(name, input_path, ref_path)]."""
    def listing(d):
        fs = [p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
        names = [p.name for p in fs]
        if len(set(names)) != len(names):
            raise SystemExit(f"STOP: duplicate basenames in {d}")
        return {p.name: p for p in fs}
    a, b = listing(input_dir), listing(ref_dir)
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise SystemExit(f"STOP: unpaired files (strict basename rule): only inputs {only_a[:5]} ({len(only_a)}), only references {only_b[:5]} ({len(only_b)})")
    if not a:
        raise SystemExit("STOP: no image files found")
    return [(n, a[n], b[n]) for n in sorted(a)]


def rederive_frozen_subset(project_root=ROOT, n=90):
    """Re-apply the documented generator rule of tests/test_pre_training_verification.py::verify_and_build_uieb_manifest (commit a2c0b5b) WITHOUT writing
    anything: first matching input dir (**/raw-890, **/input, **/degraded) and reference dir (**/reference-890, **/gt, **/target) under
    datasets/source_copies/UIEB; inputs = sorted(png + jpg + bmp); reference = same name, else stem.png, else stem.jpg; first n pairs.
    Parity with the original function is tested on synthetic trees (tests_v2/test_uieb_overlap_audit.py)."""
    project_root = Path(project_root)
    uieb = project_root / "datasets/source_copies/UIEB"
    if not uieb.exists():
        raise SystemExit(f"STOP: {uieb} not found")
    input_dirs = list(uieb.glob("**/raw-890")) + list(uieb.glob("**/input")) + list(uieb.glob("**/degraded"))
    gt_dirs = list(uieb.glob("**/reference-890")) + list(uieb.glob("**/gt")) + list(uieb.glob("**/target"))
    pairs = []
    if input_dirs and gt_dirs:
        inp_d, gt_d = input_dirs[0], gt_dirs[0]
        for f in sorted(list(inp_d.glob("*.png")) + list(inp_d.glob("*.jpg")) + list(inp_d.glob("*.bmp"))):
            for g in (gt_d / f.name, gt_d / (f.stem + ".png"), gt_d / (f.stem + ".jpg")):
                if g.exists():
                    pairs.append({"filename": f.name, "input_path": str(f.relative_to(project_root)).replace("\\", "/"),
                                  "gt_path": str(g.relative_to(project_root)).replace("\\", "/"), "input_sha256": sha256_file(f), "gt_sha256": sha256_file(g)})
                    break
    return {"input_dirs_found": [str(d.relative_to(project_root)) for d in input_dirs], "gt_dirs_found": [str(d.relative_to(project_root)) for d in gt_dirs],
            "total_available": len(pairs), "pairs": pairs[:n] if len(pairs) >= n else pairs}


def compare_rederived(rederived, manifest_data):
    """Field-by-field, in order. The manifest is the raw JSON (relative paths, as written by the generator)."""
    mp = manifest_data.get("pairs") or manifest_data.get("test_pairs") or []
    keys = ("filename", "input_path", "gt_path", "input_sha256", "gt_sha256")
    diffs = []
    if len(mp) != len(rederived["pairs"]):
        diffs.append(f"pair count manifest {len(mp)} != rederived {len(rederived['pairs'])}")
    for i, (a, b) in enumerate(zip(mp, rederived["pairs"])):
        for k in keys:
            if a.get(k) != b.get(k):
                diffs.append(f"item {i} {k}: manifest {a.get(k)!r} != rederived {b.get(k)!r}")
    for k in ("total_available", "evaluation_subset_size"):
        want = rederived["total_available"] if k == "total_available" else len(rederived["pairs"])
        if k in manifest_data and manifest_data[k] != want:
            diffs.append(f"{k}: manifest {manifest_data[k]} != rederived {want}")
    return {"matches_documented_rule": not diffs, "n_differences": len(diffs), "first_differences": diffs[:20],
            "rederived_total_available": rederived["total_available"], "input_dirs_found": rederived["input_dirs_found"], "gt_dirs_found": rederived["gt_dirs_found"]}


def load_lsui(manifests=LSUI_MANIFESTS):
    """Return list of LSUI records {split, filename, input_path, gt_path} after verifying each manifest hash prefix."""
    recs = []
    for split, (path, key, prefix) in manifests.items():
        got = sha256_file(path)
        if not got.startswith(prefix):
            raise SystemExit(f"STOP: {path} sha256 {got} does not start with {prefix}")
        for p in json.load(open(path))[key]:
            recs.append({"split": split, "filename": p["filename"], "input_path": p["input_path"], "gt_path": p["gt_path"]})
    return recs


def audit(uieb_pairs, lsui, thr=DEFAULT_THR):
    """Pure audit. Returns (rows, summary). No pair is dropped."""
    exact = {}
    for r in lsui:
        for role in ("input", "gt"):
            exact.setdefault(sha256_file(r[f"{role}_path"]), []).append(f"{r['split']}:{role}:{r['filename']}")
    Li = np.stack([dhash(r["input_path"]) for r in lsui]); Lg = np.stack([dhash(r["gt_path"]) for r in lsui])
    rows = []
    for name, ip, rp in uieb_pairs:
        hi, hr = sha256_file(ip), sha256_file(rp)
        di, dr = dhash(ip), dhash(rp)
        d_ii = (Li != di).sum(1); d_rg = (Lg != dr).sum(1); d_ig = (Lg != di).sum(1); d_ri = (Li != dr).sum(1)
        proto = np.minimum(d_ii, d_rg)
        k = int(proto.argmin()); kc = int(np.minimum(d_ig, d_ri).argmin())
        ex = sorted(set(exact.get(hi, []) + exact.get(hr, [])))
        rows.append({"filename": name, "input_sha256": hi, "ref_sha256": hr,
                     "exact_match": int(bool(ex)), "exact_match_with": ";".join(ex),
                     "dhash_min_protocol": int(proto[k]), "closest_lsui_protocol": f"{lsui[k]['split']}:{lsui[k]['filename']}",
                     "dhash_input_vs_lsui_input": int(d_ii.min()), "dhash_ref_vs_lsui_gt": int(d_rg.min()),
                     "dhash_min_cross": int(min(d_ig.min(), d_ri.min())), "closest_lsui_cross": f"{lsui[kc]['split']}:{lsui[kc]['filename']}",
                     "flag_protocol": int(proto[k] <= thr), "flag_cross": int(min(d_ig.min(), d_ri.min()) <= thr)})
    hist = {str(d): sum(1 for r in rows if r["dhash_min_protocol"] == d) for d in range(0, thr + 1)}
    summary = {"n_uieb_pairs": len(rows), "n_lsui_records": len(lsui), "lsui_splits": sorted({r["split"] for r in lsui}),
               "threshold": thr, "threshold_status": "PROVISIONAL, not validated for UIEB",
               "n_exact_match": sum(r["exact_match"] for r in rows), "n_flag_protocol": sum(r["flag_protocol"] for r in rows),
               "n_flag_cross": sum(r["flag_cross"] for r in rows),
               "n_flag_any": sum(1 for r in rows if r["exact_match"] or r["flag_protocol"] or r["flag_cross"]),
               "protocol_distance_hist_le_thr": hist,
               "flagged_filenames": [r["filename"] for r in rows if r["exact_match"] or r["flag_protocol"] or r["flag_cross"]],
               "rule": "no pair removed; primary evaluation uses all pairs; the unflagged subset is a separate labelled sensitivity analysis"}
    return rows, summary


def write_outputs(out, rows, summary):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "uieb_overlap_per_pair.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary["per_pair_csv_sha256"] = sha256_file(out / "uieb_overlap_per_pair.csv")
    with open(out / "uieb_overlap_summary.json", "x") as f:
        json.dump(summary, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--uieb-phrase"); ap.add_argument("--uieb-input-dir"); ap.add_argument("--uieb-ref-dir"); ap.add_argument("--uieb-path-marker")
    ap.add_argument("--uieb-manifest"); ap.add_argument("--uieb-manifest-sha256"); ap.add_argument("--uieb-expected-pairs", type=int)
    ap.add_argument("--uieb-pairing", default="strict", choices=["strict", "stem"])
    ap.add_argument("--no-provenance-check", action="store_true", help="manifest mode: skip re-deriving the subset with the documented generator rule")
    ap.add_argument("--thr", type=int, default=DEFAULT_THR); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    manifest_mode = a.uieb_manifest is not None
    if manifest_mode and (a.uieb_input_dir or a.uieb_ref_dir):
        raise SystemExit("STOP: give either --uieb-manifest or --uieb-input-dir/--uieb-ref-dir, not both")
    if manifest_mode:  # before any UIEB access
        check_manifest_policy(a.uieb_phrase, a.uieb_manifest, a.uieb_manifest_sha256, a.uieb_expected_pairs, a.uieb_path_marker)
    else:
        check_policy(a.uieb_phrase, a.uieb_input_dir, a.uieb_ref_dir, a.uieb_path_marker)
    if Path(a.out).exists():
        raise SystemExit(f"STOP: {a.out} exists (never overwrite)")
    t0 = time.time()
    lsui = load_lsui()
    manifest_info = None
    if manifest_mode:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import paper1_final_eval_v2 as V
        mp, mh, mi = V.load_uieb_manifest2(a.uieb_manifest, a.uieb_manifest_sha256, a.uieb_expected_pairs, a.uieb_path_marker, a.uieb_pairing)
        pairs = [(Path(p["input_path"]).name, Path(p["input_path"]), Path(p["gt_path"])) for p in mp]
        manifest_info = dict(mi, manifest_path=str(Path(a.uieb_manifest).resolve()), manifest_sha256=mh)
        if not a.no_provenance_check:
            manifest_info["provenance_check"] = compare_rederived(rederive_frozen_subset(ROOT, a.uieb_expected_pairs), json.load(open(a.uieb_manifest, encoding="utf-8")))
    else:
        pairs = pair_dirs(a.uieb_input_dir, a.uieb_ref_dir)
    rows, summary = audit(pairs, lsui, a.thr)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    summary.update(script_sha256=sha256_file(Path(__file__)), git_head=head, python=platform.python_version(), command=sys.argv,
                   mode=("manifest" if manifest_mode else "directories"), manifest=manifest_info,
                   uieb_input_dir=(None if manifest_mode else str(Path(a.uieb_input_dir).resolve())), uieb_ref_dir=(None if manifest_mode else str(Path(a.uieb_ref_dir).resolve())),
                   lsui_manifest_sha256={k: sha256_file(v[0]) for k, v in LSUI_MANIFESTS.items()}, seconds=round(time.time() - t0, 1))
    write_outputs(a.out, rows, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged_filenames"}, indent=1))


if __name__ == "__main__":
    main()
