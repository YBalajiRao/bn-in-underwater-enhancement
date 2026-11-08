"""SOTA-preparation leakage screen (CPU only; read-only on datasets; no inference): UIEB-787 (Stage 3 primary cohort) against
(a) the FULL LSUI release copy (datasets/source_copies/LSUI: 4,279 input + 4,279 GT) and (b) every EUVP image that could be training data
(Paired trainA/trainB/validation x 3 subsets, Unpaired trainA/trainB/validation, test_samples Inp/GTr; eval_data = other methods' outputs is excluded).
Checks, reusing the frozen functions unchanged:
  1. exact: byte-identical files (SHA-256, uieb_overlap_audit.sha256_file);
  2. perceptual: 8x9 LANCZOS dHash (uieb_overlap_audit.dhash), Hamming <= 10 (the registered threshold of every earlier overlap audit), all UIEB-787
     input and reference images against all pool images (input/reference roles both ways);
  3. triage of every flagged UIEB image's nearest pool image: 256-bit dHash (stage3_uieb_remaining_audit.dhash16) and grey correlation at 256x256
     (stage3_uieb_remaining_audit.grey256). Calibration from Stage 3: confirmed duplicates dHash-256 <= 27 and r >= 0.86; unrelated collisions
     dHash-256 >= 49 and r <= 0.77. Visual-review rule (as Stage 3): dHash-256 < 50 OR r >= 0.70.
  4. side-by-side panels for every visual-review candidate.
Usage: python v2_tools/sota_overlap_screen.py --out paper1_uieb_runs/sota_overlap_screen"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import uieb_overlap_audit as A  # noqa: E402  (frozen)
import stage3_uieb_remaining_audit as S3A  # noqa: E402  (frozen)

UIEB_MANIFEST = "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json"
UIEB_MANIFEST_SHA = "dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08"
EXCL = "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"
EXCL_SHA = "d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c"
LSUI = ROOT / "datasets/source_copies/LSUI"
EUVP = ROOT / "datasets/source_copies/EUVP"
EUVP_DIRS = ["Paired/underwater_dark/trainA", "Paired/underwater_dark/trainB", "Paired/underwater_dark/validation",
             "Paired/underwater_imagenet/trainA", "Paired/underwater_imagenet/trainB", "Paired/underwater_imagenet/validation",
             "Paired/underwater_scenes/trainA", "Paired/underwater_scenes/trainB", "Paired/underwater_scenes/validation",
             "Unpaired/trainA", "Unpaired/trainB", "Unpaired/validation", "test_samples/Inp", "test_samples/GTr"]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
THR = A.DEFAULT_THR
REVIEW_DH256, REVIEW_R = 50, 0.70


def uieb787():
    import paper1_final_eval_v2 as V
    if A.sha256_file(ROOT / EXCL) != EXCL_SHA:
        raise SystemExit("STOP: exclusion file hash differs")
    pairs, _, _ = V.load_uieb_manifest2(str(ROOT / UIEB_MANIFEST), UIEB_MANIFEST_SHA, 800, "UIEB", "stem")  # verifies all recorded image hashes
    excl = set(json.load(open(ROOT / EXCL))["primary_exclusions"])
    sel = [p for p in pairs if Path(p["input_path"]).name not in excl]
    assert len(sel) == 787
    return sel


def pool():
    recs = []
    for role in ("input", "GT"):
        recs += [{"pool": "LSUI", "group": role, "path": str(p)} for p in sorted((LSUI / role).iterdir()) if p.suffix.lower() in EXT]
    for d in EUVP_DIRS:
        recs += [{"pool": "EUVP", "group": d, "path": str(p)} for p in sorted((EUVP / d).iterdir()) if p.suffix.lower() in EXT]
    return recs


def packbits(H):
    return np.packbits(H.astype(np.uint8), axis=1)  # 64 bits -> 8 bytes


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_min(Q, P, chunk=256):
    """Q: [q,8] uint8, P: [p,8] uint8 -> (min distance, argmin) per query."""
    mins = np.empty(len(Q), dtype=np.int32); arg = np.empty(len(Q), dtype=np.int64)
    for i in range(0, len(Q), chunk):
        d = POP[np.bitwise_xor(Q[i:i + chunk, None, :], P[None, :, :])].sum(-1, dtype=np.int32)
        mins[i:i + chunk] = d.min(1); arg[i:i + chunk] = d.argmin(1)
    return mins, arg


def panel(a, b, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for x, p, t in ((ax[0], a, "UIEB-787: " + Path(a).name), (ax[1], b, Path(b).parent.name + "/" + Path(b).name)):
        x.imshow(Image.open(p).convert("RGB")); x.set_title(t, fontsize=7); x.axis("off")
    fig.suptitle(title, fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    t0 = time.time()
    U = uieb787()
    uimg = [(Path(p["input_path"]).name, "input", p["input_path"]) for p in U] + [(Path(p["input_path"]).name, "reference", p["gt_path"]) for p in U]
    P = pool()
    print("UIEB images", len(uimg), "pool images", len(P), {k: sum(1 for r in P if r["pool"] == k) for k in ("LSUI", "EUVP")}, flush=True)
    # 1. exact
    ush = {A.sha256_file(p): (n, r) for n, r, p in uimg}
    psh = [A.sha256_file(r["path"]) for r in P]
    exact = [{"uieb": ush[h][0], "uieb_role": ush[h][1], "pool": P[i]["pool"], "pool_path": P[i]["path"]} for i, h in enumerate(psh) if h in ush]
    print("exact matches", len(exact), flush=True)
    # 2. perceptual
    QU = packbits(np.stack([A.dhash(p) for _, _, p in uimg]))
    QP = packbits(np.stack([A.dhash(r["path"]) for r in P]))
    print("hashed", round(time.time() - t0), "s", flush=True)
    res = {}
    for pname in ("LSUI", "EUVP"):
        idx = np.array([i for i, r in enumerate(P) if r["pool"] == pname])
        m, ar = hamming_min(QU, QP[idx])
        res[pname] = (m, idx[ar])
    rows, flagged = [], []
    for j, (n, role, p) in enumerate(uimg):
        for pname in ("LSUI", "EUVP"):
            m, k = res[pname][0][j], res[pname][1][j]
            if m <= THR:
                tri_dh = int((S3A.dhash16(p) != S3A.dhash16(P[k]["path"])).sum())
                tri_r = float(np.mean(S3A.grey256(p) * S3A.grey256(P[k]["path"])))
                rec = {"uieb": n, "uieb_role": role, "pool": pname, "nearest": P[k]["path"].replace(str(ROOT) + "\\", "").replace("\\", "/"),
                       "dhash64": int(m), "dhash256": tri_dh, "corr256": round(tri_r, 4), "visual_review": bool(tri_dh < REVIEW_DH256 or tri_r >= REVIEW_R)}
                flagged.append(rec)
    out.mkdir(parents=True)
    (out / "panels").mkdir()
    for i, r in enumerate(sorted([f for f in flagged if f["visual_review"]], key=lambda f: (f["dhash256"], -f["corr256"]))):
        up = next(p for n, role, p in uimg if n == r["uieb"] and role == r["uieb_role"])
        panel(up, str(ROOT / r["nearest"]), f"{r['pool']} dHash64={r['dhash64']} dHash256={r['dhash256']} r={r['corr256']}",
              out / "panels" / f"{i:03d}_{r['pool']}_{Path(r['uieb']).stem}_{r['uieb_role']}.png")
    with open(out / "flagged.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uieb", "uieb_role", "pool", "nearest", "dhash64", "dhash256", "corr256", "visual_review"]); w.writeheader(); w.writerows(flagged)
    summ = {"label": "SOTA-prep leakage screen; CPU; no inference", "n_uieb_pairs": len(U), "n_uieb_images": len(uimg),
            "pool_counts": {k: sum(1 for r in P if r["pool"] == k) for k in ("LSUI", "EUVP")}, "euvp_dirs": EUVP_DIRS, "exact_matches": exact,
            "flagged_dhash64_le_10": {k: sum(1 for f in flagged if f["pool"] == k) for k in ("LSUI", "EUVP")},
            "flagged_unique_uieb_pairs": {k: len({f["uieb"] for f in flagged if f["pool"] == k}) for k in ("LSUI", "EUVP")},
            "visual_review_candidates": sum(1 for f in flagged if f["visual_review"]), "thresholds": {"dhash64": THR, "review_dhash256_lt": REVIEW_DH256, "review_corr_ge": REVIEW_R},
            "script_sha256": A.sha256_file(__file__), "seconds": round(time.time() - t0, 1)}
    json.dump(summ, open(out / "summary.json", "x"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("euvp_dirs",)}, indent=1))


if __name__ == "__main__":
    main()
