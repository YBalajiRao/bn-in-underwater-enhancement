"""TUB (vapaau/TUB, HF revision 6214f00807df5dbbfd389cddb4cecc41e9d62110) read-only dataset audit. No model, no inference. Writes one new folder.
Parts: originals verified against the download manifest (before and after); inventory (scene x camera x stage, presence, dimensions, EXIF; absent EXIF = UNKNOWN);
camera settings analysed separately from turbidity; NTU from av_turbidity.csv cross-checked with labels.json; missing combinations; explicit clear-reference
pairing (Stage01 of the same scene AND camera) with the downstream impact of every missing reference; HF split vs paper split; alignment (phase correlation
Stage01 vs Stage02/03, same camera); per-image global statistics vs reference; exact + perceptual leakage vs LSUI train+Val-A, UIEB-890, HICRD-2000, EUVP;
cross-scene Stage01 similarity (CANDIDATES for visual verification only, never automatic exclusion). Contact sheets go to a scratch folder outside git.
Usage: python v2_tools/tub_dataset_audit.py --out DIR --scratch DIR"""
import argparse, collections, csv, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))
TUB = ROOT / "datasets/source_copies/TUB/hf_vapaau_TUB_rev6214f00807df"
PAT = re.compile(r"Exp(\d+)_Cam(\d+)_Stage(\d+)\.jpg$")
HF_TRAIN = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 20, 22, 24, 26, 27, 28, 29, 31, 32]
HF_TEST = [9, 14, 15, 18, 19, 20, 22, 24, 29]
PAPER_VAL_EXAMPLES = [10, 15, 20]  # scenes shown in figures captioned as validation (arXiv 2606.26295 HTML); full list not given
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
EXIF_FIELDS = ("ExposureTime", "ISO", "WhiteBalance", "ExposureMode", "Model")


def exif(im):
    """EXIF fields; any field absent (or unreadable) is reported as the string UNKNOWN, never imputed."""
    out = {}
    try:
        e = im.getexif(); sub = e.get_ifd(0x8769)
    except Exception:  # noqa: BLE001
        e, sub = {}, {}
    for t in ("ExposureTime", "ISOSpeedRatings", "WhiteBalance", "ExposureMode", "Model"):
        v = sub.get(TAGS[t]) if hasattr(sub, "get") else None
        if v is None and hasattr(e, "get"):
            v = e.get(TAGS[t])
        out["ISO" if t == "ISOSpeedRatings" else t] = "UNKNOWN" if v is None else str(v)
    return out


def missing_combinations(keys, scenes, cams, stages):
    present = set(keys)
    return [(s, c, t) for s in scenes for c in cams for t in stages if (s, c, t) not in present]


def build_pairs(rows):
    """Explicit scene-camera-stage pairing: each Stage>=2 image is paired with Stage01 of the same scene AND camera.
    Returns pairs, the unpaired degraded files, and the downstream impact of every missing Stage01 reference."""
    byk = {(r["scene"], r["cam"], r["stage"]): r for r in rows}
    pairs, unpaired = [], []
    for r in sorted(rows, key=lambda r: (r["scene"], r["cam"], r["stage"])):
        if r["stage"] == 1:
            continue
        ref = byk.get((r["scene"], r["cam"], 1))
        if ref is None:
            unpaired.append(r["file"]); continue
        pairs.append({"scene": r["scene"], "cam": r["cam"], "stage": r["stage"], "input": f"images/{r['file']}", "reference": f"images/{ref['file']}",
                      "input_sha256": r["sha256"], "reference_sha256": ref["sha256"], "ntu_input": r.get("ntu"), "ntu_reference": ref.get("ntu"),
                      "same_size": (r["w"], r["h"]) == (ref["w"], ref["h"])})
    impact = {}
    for (s, c) in sorted({(r["scene"], r["cam"]) for r in rows}):
        if (s, c, 1) not in byk:
            impact[f"Exp{s}_Cam{c}"] = sorted(r["file"] for r in rows if r["scene"] == s and r["cam"] == c)
    return pairs, unpaired, impact


def verify_originals(dm):
    bad = [f["path"] for f in dm["files"] if hashlib.sha256((TUB / f["path"]).read_bytes()).hexdigest() != f["sha256"]]
    return {"n_files": len(dm["files"]), "mismatches": bad}


def gray(im, w=640):
    g = np.asarray(im.convert("L"), dtype=np.uint8); h = int(round(g.shape[0] * w / g.shape[1]))
    return np.asarray(Image.fromarray(g).resize((w, h), Image.BILINEAR), dtype=np.float64)


def phase_shift(a, b):
    """integer translation (dy, dx) such that a ~ roll(b, (dy, dx)), and the correlation peak height."""
    A = np.fft.fft2(a - a.mean()); Bf = np.fft.fft2(b - b.mean()); Rr = A * np.conj(Bf); Rr /= np.abs(Rr) + 1e-12
    r = np.real(np.fft.ifft2(Rr)); i, j = np.unravel_index(np.argmax(r), r.shape)
    dy = i if i <= a.shape[0] // 2 else i - a.shape[0]; dx = j if j <= a.shape[1] // 2 else j - a.shape[1]
    return int(dy), int(dx), float(r.max())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=False); scr = Path(a.scratch); scr.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); R = {}
    dm = json.load(open(ROOT / "datasets/source_copies/TUB/DOWNLOAD_MANIFEST.json"))
    R["download"] = {k: dm[k] for k in ("source", "revision", "n_files", "total_bytes", "all_verified_against_official", "licence")}
    R["originals_before"] = verify_originals(dm)
    # ---------------- inventory
    rows = []
    for p in sorted((TUB / "images").glob("*.jpg")):
        m = PAT.search(p.name); im = Image.open(p)
        rows.append({"file": p.name, "scene": int(m.group(1)), "cam": int(m.group(2)), "stage": int(m.group(3)), "w": im.size[0], "h": im.size[1],
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), **exif(im)})
    scenes = sorted({r["scene"] for r in rows}); cams = sorted({r["cam"] for r in rows}); stages = sorted({r["stage"] for r in rows})
    missing = missing_combinations([(r["scene"], r["cam"], r["stage"]) for r in rows], scenes, cams, stages)
    R["inventory"] = {"n_images": len(rows), "scenes": scenes, "n_scenes": len(scenes), "cameras": cams, "stages": stages,
                      "expected_full_grid": len(scenes) * len(cams) * len(stages), "missing_combinations": [{"scene": s, "cam": c, "stage": t} for s, c, t in missing],
                      "dimensions": dict(collections.Counter(f"{r['w']}x{r['h']}" for r in rows)),
                      "duplicate_sha256": [k for k, v in collections.Counter(r["sha256"] for r in rows).items() if v > 1]}
    # ---------------- camera settings (analysed separately from turbidity)
    cs = {}
    for k in EXIF_FIELDS:
        series = collections.Counter()
        for (sc, c) in sorted({(r["scene"], r["cam"]) for r in rows}):
            vals = [r[k] for r in sorted(rows, key=lambda r: r["stage"]) if r["scene"] == sc and r["cam"] == c]
            series["UNKNOWN_in_series" if "UNKNOWN" in vals else ("constant_within_series" if len(set(vals)) == 1 else "varies_within_series")] += 1
        cs[k] = {"value_counts": dict(collections.Counter(r[k] for r in rows)), "scene_camera_series": dict(series),
                 "by_stage": {t: dict(collections.Counter(r[k] for r in rows if r["stage"] == t)) for t in stages}}
    R["camera_settings_separate_from_turbidity"] = cs
    # ---------------- NTU
    ntu = {}
    for r in csv.DictReader(open(TUB / "av_turbidity.csv")):
        ntu[(int(r["Exp_id"].replace("Exp", "")), int(r["stage"]))] = (float(r["avg"]), float(r["std"]))
    lj = json.load(open(TUB / "labels.json")); lj_ntu = {(im["experiment"], im["stage"]): im["turbidity"] for im in lj["images"]}
    csv_scenes = sorted({k[0] for k in ntu})
    R["ntu"] = {"csv_scenes": csv_scenes, "csv_rows": len(ntu), "csv_scenes_without_images": [s for s in csv_scenes if s not in scenes],
                "image_scenes_without_csv": [s for s in scenes if s not in csv_scenes],
                "labels_json_vs_csv_max_abs_diff": max(abs(lj_ntu[k] - ntu[k][0]) for k in lj_ntu if k in ntu), "labels_json_images": len(lj["images"]),
                "stage1_ntu_per_scene": {s: ntu[(s, 1)][0] for s in scenes if (s, 1) in ntu},
                "non_monotone_scenes": [s for s in scenes if not all(ntu[(s, t + 1)][0] >= ntu[(s, t)][0] for t in range(1, 10) if (s, t) in ntu and (s, t + 1) in ntu)],
                "ntu_range_by_stage": {t: [min(ntu[(s, t)][0] for s in scenes if (s, t) in ntu), max(ntu[(s, t)][0] for s in scenes if (s, t) in ntu)] for t in stages}}
    for r in rows:
        r["ntu"], r["ntu_sd"] = ntu.get((r["scene"], r["stage"]), (None, None))
    with open(out / "tub_inventory.csv", "x", newline="") as f:
        cols = ["file", "scene", "cam", "stage", "ntu", "ntu_sd", "w", "h", "sha256", *EXIF_FIELDS]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    # ---------------- pairing
    byk = {(r["scene"], r["cam"], r["stage"]): r for r in rows}
    pairs, unpaired, impact = build_pairs(rows)
    R["pairing"] = {"n_valid_pairs": len(pairs), "degraded_without_reference": unpaired, "missing_reference_impact": impact, "all_same_size": all(p["same_size"] for p in pairs),
                    "pairs_per_scene": dict(collections.Counter(p["scene"] for p in pairs)), "n_scene_camera_series_with_reference": len({(p["scene"], p["cam"]) for p in pairs})}
    # ---------------- alignment and global statistics
    align, gstat = [], []
    for (s, c) in sorted({(p["scene"], p["cam"]) for p in pairs}):
        ref = Image.open(TUB / "images" / byk[(s, c, 1)]["file"]); g1 = gray(ref)
        mu1 = np.asarray(ref.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(0) / 255
        for t in stages:
            if (s, c, t) not in byk:
                continue
            im = Image.open(TUB / "images" / byk[(s, c, t)]["file"])
            mu = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(0) / 255
            gstat.append({"scene": s, "cam": c, "stage": t, "ntu": byk[(s, c, t)]["ntu"], "mean_R": mu[0], "mean_G": mu[1], "mean_B": mu[2],
                          "luma": float(0.299 * mu[0] + 0.587 * mu[1] + 0.114 * mu[2]), "imbalance": float(mu.max() - mu.min()),
                          "g_ach_vs_ref": float(abs(mu1.mean() - mu.mean())), "g_tot_vs_ref": float(np.linalg.norm(mu1 - mu) / np.sqrt(3))})
            if t in (2, 3):
                dy, dx, pk = phase_shift(gray(im), g1); align.append({"scene": s, "cam": c, "stage": t, "dy_px_at_640w": dy, "dx_px_at_640w": dx, "peak": pk})
    with open(out / "tub_global_stats_per_image.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gstat[0])); w.writeheader(); w.writerows(gstat)
    with open(out / "tub_alignment_stage01_vs_02_03.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(align[0])); w.writeheader(); w.writerows(align)
    sh = np.array([max(abs(x["dy_px_at_640w"]), abs(x["dx_px_at_640w"])) for x in align])
    R["alignment"] = {"n_checked": len(align), "max_abs_shift_px_640w": int(sh.max()), "n_shift_gt_1px": int((sh > 1).sum()), "n_shift_gt_3px": int((sh > 3).sum()),
                      "worst": sorted(align, key=lambda x: -max(abs(x["dy_px_at_640w"]), abs(x["dx_px_at_640w"])))[:8]}
    gs = collections.defaultdict(list)
    for g in gstat:
        gs[g["stage"]].append(g)
    R["global_stats_by_stage_median"] = {t: {k: float(np.median([g[k] for g in v])) for k in ("ntu", "luma", "imbalance", "g_ach_vs_ref", "g_tot_vs_ref")} for t, v in sorted(gs.items())}
    # ---------------- splits
    allS = set(scenes)
    R["splits"] = {"hf_train": HF_TRAIN, "hf_test": HF_TEST, "hf_overlap_train_and_test": sorted(set(HF_TRAIN) & set(HF_TEST)),
                   "scenes_in_neither_hf_split": sorted(allS - set(HF_TRAIN) - set(HF_TEST)), "hf_pattern_scenes_without_images": sorted((set(HF_TRAIN) | set(HF_TEST)) - allS),
                   "hf_union_size": len(set(HF_TRAIN) | set(HF_TEST)),
                   "paper": "scene-level: 8 validation / 25 training; scene IDs not listed; figures captioned as validation show scenes 10, 15, 20",
                   "paper_val_examples_in_hf_train_only": [s for s in PAPER_VAL_EXAMPLES if s in HF_TRAIN and s not in HF_TEST]}
    # ---------------- leakage (exact + perceptual)
    import stage4_candidate_audit as S4
    refs = S4.reference_sets(); refs.pop("HICRD_paired_train", None)
    refs["HICRD_2000"] = [(f"hicrd:{p['id']}:{k}", ROOT / p[f"{k}_path"]) for p in json.load(open(ROOT / "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json"))["pairs"] for k in ("input", "gt")]
    idx = {k: S4.index(v) for k, v in refs.items()}
    leak = []
    for r in rows:
        mine = Image.open(TUB / "images" / r["file"]); d8 = S4.dh8_img(mine)
        for name, (labs, shas, D) in idx.items():
            ex = [labs[k] for k, s_ in enumerate(shas) if s_ == r["sha256"]]
            dist = (D != d8).sum(1); k = int(dist.argmin())
            if ex or dist[k] <= 10:
                im2, _ = S4.open_img(dict(refs[name])[labs[k]])
                leak.append({"tub": r["file"], "against": name, "exact": ";".join(ex), "dhash8": int(dist[k]), "closest": labs[k],
                             "dhash16": int((S4.dh16_img(mine) != S4.dh16_img(im2)).sum()), "corr": float(np.mean(S4.grey(mine) * S4.grey(im2)))})
    with open(out / "tub_leakage_flags.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tub", "against", "exact", "dhash8", "closest", "dhash16", "corr"]); w.writeheader(); w.writerows(leak)
    R["leakage"] = {"reference_set_sizes": {k: len(v) for k, v in refs.items()}, "n_flags": len(leak), "n_exact": sum(1 for x in leak if x["exact"]),
                    "flags_by_set": dict(collections.Counter(x["against"] for x in leak)), "strongest": sorted(leak, key=lambda x: (x["dhash16"], -x["corr"]))[:10]}
    # ---------------- cross-scene Stage01 similarity (candidates only)
    s1 = [r for r in rows if r["stage"] == 1]
    H16 = {r["file"]: S4.dh16_img(Image.open(TUB / "images" / r["file"])) for r in s1}
    cross = sorted((int((H16[ra["file"]] != H16[rb["file"]]).sum()), ra["file"], rb["file"]) for i, ra in enumerate(s1) for rb in s1[i + 1:]
                   if ra["scene"] != rb["scene"] and ra["cam"] == rb["cam"])
    R["cross_scene_stage01_same_camera"] = {"status": "CANDIDATES for visual verification only; no automatic exclusion", "n_pairs": len(cross),
                                            "closest_10": cross[:10], "dhash16_median": float(np.median([c[0] for c in cross])) if cross else None}
    # ---------------- contact sheets (scratch, not committed)
    from PIL import ImageDraw
    for s in [scenes[0], scenes[len(scenes) // 2], scenes[-1]]:
        sheet = Image.new("RGB", (5 * 240, len(cams) * 150), "white")
        for ci, c in enumerate(cams):
            for j, t in enumerate((1, 3, 5, 7, 10)):
                if (s, c, t) in byk:
                    th = Image.open(TUB / "images" / byk[(s, c, t)]["file"]).convert("RGB").resize((240, 135))
                    sheet.paste(th, (j * 240, ci * 150)); ImageDraw.Draw(sheet).text((j * 240 + 2, ci * 150 + 136), f"S{s} C{c} St{t} {byk[(s, c, t)]['ntu']:.1f}NTU", fill="black")
        sheet.save(scr / f"tub_contact_scene{s}.png")
    for c in cross[:6]:
        a_, b_ = Image.open(TUB / "images" / c[1]).convert("RGB").resize((480, 270)), Image.open(TUB / "images" / c[2]).convert("RGB").resize((480, 270))
        t_ = Image.new("RGB", (966, 290), "white"); t_.paste(a_, (0, 20)); t_.paste(b_, (486, 20)); ImageDraw.Draw(t_).text((2, 2), f"{c[1]} vs {c[2]} d16={c[0]}", fill="black")
        t_.save(scr / f"tub_crossscene_{c[1][:-4]}_vs_{c[2][:-4]}.png")
    json.dump(pairs, open(out / "PROPOSED_tub_pairs_manifest.json", "x"), indent=1)
    R["originals_after"] = verify_originals(dm)
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(out / "tub_audit_summary.json", "x"), indent=1, default=str)
    print(json.dumps({k: R[k] for k in ("inventory", "pairing", "splits")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
