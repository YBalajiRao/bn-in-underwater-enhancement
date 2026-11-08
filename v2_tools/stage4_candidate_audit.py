"""Stage 4 preparation: integrity, input-descriptor distribution and overlap audit of the cross-dataset candidates. No model, no inference.
Candidates: EUVP test_samples (515 pairs) and HICRD official paired test (300 pairs, extracted from the CSIRO archive).
Per candidate: pairing check, decode check, sizes, SHA-256 of every file, native input descriptors (channel-mean imbalance, luma mean, luma RMS contrast).
Overlap (exact SHA-256 and 8x9 dHash <= 10, any input/reference role) against: LSUI train + Val-A; UIEB-890 (inputs and references); the candidate's own
training/validation portions (EUVP Paired/*/trainA,trainB,validation + Unpaired/*; HICRD paired train); the other candidate; and within the candidate.
Triage features for every flagged pair: 16x17 dHash and grey correlation at 256. Writes one new folder.
Usage: python v2_tools/stage4_candidate_audit.py <out dir (new)>"""
import csv, hashlib, json, os, sys, time, zipfile, io
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))
import uieb_overlap_audit as A  # noqa: E402

THR = 10
EUVP = ROOT / "datasets/source_copies/EUVP"
HICRD = ROOT / "datasets/source_copies/HICRD/HICRD_paired_official_test"
HICRD_ZIP = Path(os.environ.get("HICRD_ZIP", "datasets/HICRD_paired.zip"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def dh8_img(im):
    g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16); return (g[:, 1:] > g[:, :-1]).flatten()


def dh16_img(im):
    g = np.asarray(im.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16); return (g[:, 1:] > g[:, :-1]).flatten()


def grey(im):
    a = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float64).ravel(); return (a - a.mean()) / (a.std() + 1e-12)


def native_desc(im):
    a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    mu = a.reshape(-1, 3).mean(0); y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return {"imbalance": float(mu.max() - mu.min()), "luma_mean": float(y.mean()), "luma_rms_contrast": float(y.std())}


def candidates():
    eu = []
    for p in sorted((EUVP / "test_samples/Inp").iterdir()):
        g = EUVP / "test_samples/GTr" / p.name
        eu.append({"filename": p.name, "input_path": str(p), "gt_path": str(g) if g.exists() else None})
    hi = []
    for p in sorted((HICRD / "testA").iterdir(), key=lambda q: int(q.stem.replace("test", ""))):
        g = HICRD / "testB" / f"test_ref{p.stem.replace('test', '')}.png"
        hi.append({"filename": p.name, "input_path": str(p), "gt_path": str(g) if g.exists() else None})
    return {"EUVP_test_515": eu, "HICRD_test_300": hi}


def reference_sets():
    """name -> list of (label, path or (zip, member))."""
    sets = {}
    lsui = A.load_lsui()
    sets["LSUI_train+ValA"] = [(f"{r['split']}:{r['filename']}:{role}", r[f"{role}_path"]) for r in lsui for role in ("input", "gt")]
    red = A.rederive_frozen_subset(ROOT, n=10**6)
    sets["UIEB_890"] = [(f"uieb:{p['filename']}:{role}", ROOT / p[f"{role}_path"]) for p in red["pairs"] for role in ("input", "gt")]
    eu = []
    for sub in ("underwater_dark", "underwater_imagenet", "underwater_scenes"):
        for part in ("trainA", "trainB", "validation"):
            eu += [(f"euvp:{sub}/{part}/{q.name}", q) for q in sorted((EUVP / "Paired" / sub / part).iterdir()) if q.is_file()]
    for part in ("trainA", "trainB", "validation"):
        eu += [(f"euvp:Unpaired/{part}/{q.name}", q) for q in sorted((EUVP / "Unpaired" / part).iterdir()) if q.is_file()]
    sets["EUVP_train+val+unpaired"] = eu
    zf = zipfile.ZipFile(HICRD_ZIP)
    sets["HICRD_paired_train"] = [(f"hicrd:{n}", (HICRD_ZIP, n)) for n in zf.namelist() if n.startswith(("HICRD_paired/trainA_paired/", "HICRD_paired/trainB_paired/")) and n.endswith(".png")]
    return sets


_ZIPS = {}


def open_img(src):
    if isinstance(src, tuple):
        z, m = src
        zf = _ZIPS.setdefault(str(z), zipfile.ZipFile(z)); data = zf.read(m)
        return Image.open(io.BytesIO(data)), hashlib.sha256(data).hexdigest()
    return Image.open(src), sha(src)


def index(items):
    labs, shas, d8 = [], [], []
    for lab, src in items:
        im, h = open_img(src); labs.append(lab); shas.append(h); d8.append(dh8_img(im))
    return labs, shas, np.stack(d8)


def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=False); t0 = time.time()
    cands = candidates(); report = {"candidates": {}, "reference_sets": {}}
    cidx = {}
    for cname, pairs in cands.items():
        rows, bad = [], []
        for p in pairs:
            r = {"filename": p["filename"], "input_path": p["input_path"], "gt_path": p["gt_path"]}
            if p["gt_path"] is None:
                bad.append(p["filename"]); continue
            ii, gi = Image.open(p["input_path"]), Image.open(p["gt_path"])
            ii.load(); gi.load()
            r.update({"input_sha256": sha(p["input_path"]), "gt_sha256": sha(p["gt_path"]), "input_size": ii.size, "gt_size": gi.size, "input_mode": ii.mode, "gt_mode": gi.mode,
                      "same_geometry": ii.size == gi.size}, **native_desc(ii))
            r["_d8"] = (dh8_img(ii), dh8_img(gi)); r["_d16"] = dh16_img(ii); r["_g"] = grey(ii)
            rows.append(r)
        cidx[cname] = rows
        sizes = {}
        for r in rows:
            sizes[str(tuple(r["input_size"]))] = sizes.get(str(tuple(r["input_size"])), 0) + 1
        x = np.array([r["imbalance"] for r in rows])
        report["candidates"][cname] = {"n_inputs_listed": len(pairs), "n_pairs_valid": len(rows), "missing_reference": bad,
                                       "geometry_mismatch_pairs": [r["filename"] for r in rows if not r["same_geometry"]],
                                       "input_sizes": sizes, "within_exact_duplicate_inputs": len(rows) - len({r["input_sha256"] for r in rows}),
                                       "imbalance_quantiles_0_10_25_50_75_90_100": np.quantile(x, [0, .1, .25, .5, .75, .9, 1]).round(4).tolist(),
                                       "luma_mean_median": float(np.median([r["luma_mean"] for r in rows])),
                                       "contrast_median": float(np.median([r["luma_rms_contrast"] for r in rows]))}
    refs = reference_sets()
    ridx = {}
    for name, items in refs.items():
        ridx[name] = index(items); report["reference_sets"][name] = len(items)
    flags = []
    for cname, rows in cidx.items():
        for r in rows:
            fi, fg = r["_d8"]
            for name, (labs, shas, D) in ridx.items():
                for role, h, d in (("input", r["input_sha256"], fi), ("gt", r["gt_sha256"], fg)):
                    ex = [labs[k] for k, s in enumerate(shas) if s == h]
                    dist = (D != d).sum(1); k = int(dist.argmin())
                    if ex or dist[k] <= THR:
                        flags.append({"candidate": cname, "filename": r["filename"], "role": role, "against": name, "exact": ";".join(ex[:5]), "dhash8": int(dist[k]), "closest": labs[k]})
            # other candidate
            for oname, orows in cidx.items():
                if oname == cname:
                    continue
                for role, h, d in (("input", r["input_sha256"], fi), ("gt", r["gt_sha256"], fg)):
                    best = min(((int(min((o["_d8"][0] != d).sum(), (o["_d8"][1] != d).sum())), o["filename"]) for o in orows))
                    ex = [o["filename"] for o in orows if h in (o["input_sha256"], o["gt_sha256"])]
                    if ex or best[0] <= THR:
                        flags.append({"candidate": cname, "filename": r["filename"], "role": role, "against": oname, "exact": ";".join(ex), "dhash8": best[0], "closest": best[1]})
        # within candidate (inputs and references)
        Di = np.stack([r["_d8"][0] for r in rows]); Dg = np.stack([r["_d8"][1] for r in rows])
        for k, r in enumerate(rows):
            dist = np.minimum((Di != r["_d8"][0]).sum(1), (Dg != r["_d8"][1]).sum(1)); dist[k] = 999; j = int(dist.argmin())
            if dist[j] <= THR:
                o = rows[j]
                flags.append({"candidate": cname, "filename": r["filename"], "role": "input/gt", "against": "WITHIN", "exact": "", "dhash8": int(dist[j]), "closest": o["filename"],
                              "dhash16": int((r["_d16"] != o["_d16"]).sum()), "corr": float(np.mean(r["_g"] * o["_g"]))})
    # triage for external flags (input vs closest external image)
    lab2src = {lab: src for items in refs.values() for lab, src in items}
    for f in flags:
        if f["against"] in ridx and "dhash16" not in f:
            r = next(x for x in cidx[f["candidate"]] if x["filename"] == f["filename"])
            im, _ = open_img(lab2src[f["closest"]])
            mine = Image.open(r["input_path"] if f["role"] == "input" else r["gt_path"])
            f["dhash16"] = int((dh16_img(mine) != dh16_img(im)).sum()); f["corr"] = float(np.mean(grey(mine) * grey(im)))
    with open(out / "flags.csv", "x", newline="", encoding="utf-8") as fh:
        cols = ["candidate", "filename", "role", "against", "exact", "dhash8", "closest", "dhash16", "corr"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({c: f.get(c, "") for c in cols}) for f in flags]
    for cname, rows in cidx.items():
        with open(out / f"{cname}_pairs.csv", "x", newline="", encoding="utf-8") as fh:
            cols = ["filename", "input_path", "gt_path", "input_sha256", "gt_sha256", "input_size", "gt_size", "input_mode", "gt_mode", "same_geometry", "imbalance", "luma_mean", "luma_rms_contrast"]
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in rows]
        fl = [f for f in flags if f["candidate"] == cname]
        report["candidates"][cname]["flags"] = {a: {"pairs": len({f["filename"] for f in fl if f["against"] == a}), "exact": sum(1 for f in fl if f["against"] == a and f["exact"])}
                                                for a in sorted({f["against"] for f in fl})}
        report["candidates"][cname]["n_pairs_any_flag"] = len({f["filename"] for f in fl})
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(out / "summary.json", "x", encoding="utf-8"), indent=1)
    return report


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
