"""Stage 4 preparation: geometric same-scene detection for HICRD (overlapping views of the same reef), for cluster definition. No model outputs.
Method: grey image downscaled to width 640, CLAHE (clip 2, 8x8) for the low-contrast water, ORB (2000 features), Hamming brute-force kNN with Lowe ratio 0.75,
RANSAC homography (reprojection 5 px); score = number of RANSAC inliers.
calibrate: labelled same-scene pairs (visual review, stage 4 panels) vs guaranteed different-scene pairs (different HICRD sites) -> inlier distributions.
Usage: python v2_tools/stage4_scene_overlap.py calibrate <out json>"""
import io, json, os, re, sys, zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "datasets/source_copies/HICRD/HICRD_paired_official_test/testA"
ZIP = Path(os.environ.get("HICRD_ZIP", "datasets/HICRD_paired.zip"))
_orb = cv2.ORB_create(nfeatures=2000)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def features(im):
    g = np.asarray(im.convert("L")); h = int(round(g.shape[0] * 640 / g.shape[1]))
    g = _clahe.apply(cv2.resize(g, (640, h), interpolation=cv2.INTER_AREA))
    kp, des = _orb.detectAndCompute(g, None)
    return (np.float32([k.pt for k in kp]), des)


def inliers(fa, fb):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    good = [m for m, n in (x for x in _bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    H, mask = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


SAME = [("test298", "test297"), ("test189", "test200"), ("test296", "test297"), ("test36", "test37"), ("test149", "test150"), ("test134", "test145"),
        ("test234", "test236"), ("test117", "test119"), ("test105", "test106"), ("test267", "test269"), ("test77", "test65"), ("test222", "test211"),
        ("test233", "test222"), ("test225", "test224"), ("test29", "test102")]
DIFF_VISUAL = [("test227", "test41")]


def calibrate(out):
    F = {}
    def ft(name):
        if name not in F:
            F[name] = features(Image.open(TEST / f"{name}.png"))
        return F[name]
    same = {f"{a}~{b}": inliers(ft(a), ft(b)) for a, b in SAME}
    dv = {f"{a}~{b}": inliers(ft(a), ft(b)) for a, b in DIFF_VISUAL}
    zf = zipfile.ZipFile(ZIP); rng = np.random.default_rng(4)
    by = {}
    for n in zf.namelist():
        m = re.search(r"trainA_paired/(site\d+)_good_\d+\.png$", n)
        if m:
            by.setdefault(m.group(1), []).append(n)
    sites = sorted(by); cache = {}
    def fz(n):
        if n not in cache:
            cache[n] = features(Image.open(io.BytesIO(zf.read(n))))
        return cache[n]
    neg = []
    for _ in range(200):
        s1, s2 = rng.choice(sites, 2, replace=False)
        a = by[s1][rng.integers(len(by[s1]))]; b = by[s2][rng.integers(len(by[s2]))]
        neg.append(inliers(fz(a), fz(b)))
    res = {"same_scene_labelled": same, "different_visual": dv, "cross_site_negatives": {"n": len(neg), "max": int(max(neg)), "q99": float(np.quantile(neg, .99)), "median": float(np.median(neg))},
           "same_min": int(min(same.values())), "same_values_sorted": sorted(same.values())}
    json.dump(res, open(out, "x"), indent=1)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        print(json.dumps(calibrate(sys.argv[2]), indent=1))


def cluster(out_dir, nfeat=500, thresholds=(15, 30, 60)):
    """All HICRD paired inputs (official test + paired train). Pairwise ORB-RANSAC inliers (500 features); edges stored if >= min(thresholds);
    scene clusters = connected components (union-find) at each threshold. Primary threshold 30 (calibration: same-scene >= 122, cross-site <= 5)."""
    import csv, time
    global _orb
    _orb = cv2.ORB_create(nfeatures=nfeat)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False); t0 = time.time()
    tr = ROOT / "datasets/source_copies/HICRD/HICRD_paired_official_train/trainA_paired"
    items = [(f"test:{p.name}", p) for p in sorted(TEST.iterdir(), key=lambda q: int(q.stem[4:]))]
    items += [(f"train:{p.name}", p) for p in sorted(tr.iterdir(), key=lambda q: (int(re.search(r"site(\d+)", q.name).group(1)), int(re.search(r"_good_(\d+)", q.name).group(1))))]
    feats = [features(Image.open(p)) for _, p in items]
    n = len(items); edges = []
    for i in range(n):
        for j in range(i + 1, n):
            k = inliers(feats[i], feats[j])
            if k >= min(thresholds):
                edges.append((i, j, k))
    with open(out / "edges.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["a", "b", "inliers"]); [w.writerow([items[i][0], items[j][0], k]) for i, j, k in edges]
    res = {"n_images": n, "n_edges_ge_min": len(edges), "seconds": None, "thresholds": {}}
    assign = {}
    for T in thresholds:
        par = list(range(n))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for i, j, k in edges:
            if k >= T:
                par[find(i)] = find(j)
        roots = [find(i) for i in range(n)]; ids = {r: c for c, r in enumerate(sorted(set(roots), key=roots.index))}
        assign[T] = [ids[r] for r in roots]
        sizes = np.bincount(assign[T])
        res["thresholds"][str(T)] = {"n_clusters": int(len(sizes)), "largest": int(sizes.max()), "singletons": int((sizes == 1).sum()),
                                     "clusters_spanning_test_and_train": int(sum(1 for c in range(len(sizes)) if len({items[i][0].split(':')[0] for i in range(n) if assign[T][i] == c}) == 2))}
    with open(out / "clusters.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["image"] + [f"cluster_T{T}" for T in thresholds]); [w.writerow([items[i][0]] + [assign[T][i] for T in thresholds]) for i in range(n)]
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out / "summary.json", "x"), indent=1)
    return res


if __name__ == "__main__" and sys.argv[1] == "cluster":
    print(json.dumps(cluster(sys.argv[2]), indent=1))
