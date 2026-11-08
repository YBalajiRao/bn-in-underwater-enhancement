"""Stage 3 preparation: provenance and overlap audit of the UIEB pairs NOT in the frozen-90 cohort. No model, no inference. Writes one new folder.
1. Re-derive all UIEB pairs with the documented generator rule (uieb_overlap_audit.rederive_frozen_subset, n = all); check that its first 90 equal the
   frozen-90 manifest field by field; the remaining pairs are the candidate cohort.
2. Overlap vs LSUI train + Val-A: exact SHA-256 and 8x9 dHash (thr 10), using uieb_overlap_audit.audit unchanged.
3. Within UIEB: exact SHA-256 duplicates across all pairs; dHash (thr 10) of each remaining input/reference vs every frozen-90 input/reference and vs
   every other remaining pair (independence from the H3 data and within the cohort).
4. Triage features for every flagged pair (not a decision): 16x17 dHash distance (256 bits) and grey correlation at 256x256 to the nearest match.
Usage: python v2_tools/stage3_uieb_remaining_audit.py <out dir (new)>"""
import csv, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))
import uieb_overlap_audit as A  # noqa: E402

FROZEN = ROOT / "datasets/splits/candidate_v1/uieb_test_frozen_90.json"
FROZEN_SHA = "0a807f28a33a891d63c93b1bcec2ea945952d5a9b5ad00fdbc3570814d364daf"
THR = A.DEFAULT_THR


def dhash16(p):
    g = np.asarray(Image.open(p).convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def grey256(p):
    a = np.asarray(Image.open(p).convert("L").resize((256, 256)), dtype=np.float64).ravel()
    return (a - a.mean()) / (a.std() + 1e-12)


def main(outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    assert A.sha256_file(FROZEN) == FROZEN_SHA, "frozen-90 manifest hash differs"
    red = A.rederive_frozen_subset(ROOT, n=10**6)
    allp = red["pairs"]
    frozen_data = json.load(open(FROZEN, encoding="utf-8"))
    first90 = dict(red, pairs=allp[:90])
    prov = A.compare_rederived(first90, frozen_data)
    inp_dir = ROOT / red["input_dirs_found"][0]; ref_dir = ROOT / red["gt_dirs_found"][0]
    n_inputs = len(list(inp_dir.glob("*.png")) + list(inp_dir.glob("*.jpg")) + list(inp_dir.glob("*.bmp")))
    n_refs = len(list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.bmp")))
    rem = allp[90:]
    frozen_names = {p["filename"] for p in frozen_data["pairs"]}
    assert not ({p["filename"] for p in rem} & frozen_names), "remaining pairs overlap the frozen-90 by filename"
    P = lambda rel: ROOT / rel  # noqa: E731
    # 2. vs LSUI (unchanged audit function)
    lsui = A.load_lsui()
    rows, summ = A.audit([(p["filename"], P(p["input_path"]), P(p["gt_path"])) for p in rem], lsui, THR)
    # 3. within UIEB
    sha_seen = {}
    for p in allp:
        for role in ("input", "gt"):
            sha_seen.setdefault(p[f"{role}_sha256"], []).append(f"{p['filename']}:{role}")
    exact_within = {h: v for h, v in sha_seen.items() if len(v) > 1}
    H = {p["filename"]: (A.dhash(P(p["input_path"])), A.dhash(P(p["gt_path"]))) for p in allp}
    fz = [p["filename"] for p in allp[:90]]; rn = [p["filename"] for p in rem]
    Fi = np.stack([H[n][0] for n in fz]); Fr = np.stack([H[n][1] for n in fz])
    Ri = np.stack([H[n][0] for n in rn]); Rr = np.stack([H[n][1] for n in rn])
    within = {}
    for k, n in enumerate(rn):
        dfz = np.minimum((Fi != H[n][0]).sum(1), (Fr != H[n][1]).sum(1))
        drm = np.minimum((Ri != H[n][0]).sum(1), (Rr != H[n][1]).sum(1)); drm[k] = 999
        within[n] = {"dhash_min_vs_frozen90": int(dfz.min()), "closest_frozen90": fz[int(dfz.argmin())],
                     "dhash_min_vs_remaining": int(drm.min()), "closest_remaining": rn[int(drm.argmin())]}
    # 4. triage features for flagged pairs
    lsui_by = {f"{r['split']}:{r['filename']}": r for r in lsui}
    ppath = {p["filename"]: p for p in allp}
    for r in rows:
        w = within[r["filename"]]
        r.update(w)
        r["flag_lsui"] = int(r["exact_match"] or r["flag_protocol"] or r["flag_cross"])
        r["flag_within_uieb"] = int(w["dhash_min_vs_frozen90"] <= THR or w["dhash_min_vs_remaining"] <= THR)
        u_in = P(ppath[r["filename"]]["input_path"])
        tri = {}
        if r["flag_lsui"]:
            L = lsui_by[r["closest_lsui_protocol"]]; Lc = lsui_by[r["closest_lsui_cross"]]
            tri["lsui_dhash16_input_input"] = int((dhash16(u_in) != dhash16(L["input_path"])).sum())
            tri["lsui_corr_input_input"] = float(np.mean(grey256(u_in) * grey256(L["input_path"])))
            tri["lsui_dhash16_input_gt_cross"] = int((dhash16(u_in) != dhash16(Lc["gt_path"])).sum())
            tri["lsui_corr_input_gt_cross"] = float(np.mean(grey256(u_in) * grey256(Lc["gt_path"])))
        if w["dhash_min_vs_frozen90"] <= THR:
            o = P(ppath[w["closest_frozen90"]]["input_path"])
            tri["frozen_dhash16"] = int((dhash16(u_in) != dhash16(o)).sum()); tri["frozen_corr"] = float(np.mean(grey256(u_in) * grey256(o)))
        if w["dhash_min_vs_remaining"] <= THR:
            o = P(ppath[w["closest_remaining"]]["input_path"])
            tri["remaining_dhash16"] = int((dhash16(u_in) != dhash16(o)).sum()); tri["remaining_corr"] = float(np.mean(grey256(u_in) * grey256(o)))
        r["triage"] = json.dumps(tri)
    # calibration on the two labelled frozen-90 checks (1 confirmed duplicate, 8 visually unrelated)
    calib = {}
    for u, l, lab in [("15_img_.png", "train:1013.jpg", "confirmed_duplicate"), ("136_img_.png", "train:2860.jpg", "unrelated"), ("104_img_.png", "train:1278.jpg", "unrelated"),
                      ("11064.png", "train:1670.jpg", "unrelated"), ("121_img_.png", "train:3306.jpg", "unrelated"), ("12348.png", "train:1014.jpg", "unrelated"),
                      ("102_img_.png", "train:2420.jpg", "unrelated"), ("107_img_.png", "train:1688.jpg", "unrelated"), ("131_img_.png", "train:1162.jpg", "unrelated")]:
        ui = P(ppath[u]["input_path"]); L = lsui_by[l]
        calib[f"{u} vs {l}"] = {"label": lab, "dhash16_input_input": int((dhash16(ui) != dhash16(L["input_path"])).sum()),
                                "corr_input_input": float(np.mean(grey256(ui) * grey256(L["input_path"])))}
    cols = list(rows[0].keys())
    with open(out / "remaining_pairs_audit.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(out / "remaining_pairs_all.json", "x", encoding="utf-8") as f:
        json.dump({"source_rule": "tests/test_pre_training_verification.py::verify_and_build_uieb_manifest (a2c0b5b), sorted, pairs[90:]", "pairs": rem}, f, indent=1)
    summary = {"input_dirs_found": red["input_dirs_found"], "gt_dirs_found": red["gt_dirs_found"], "n_input_files": n_inputs, "n_reference_files": n_refs,
               "total_pairs_rederived": len(allp), "frozen90_equals_first90": prov, "n_remaining": len(rem),
               "lsui": {k: summ[k] for k in ("n_lsui_records", "lsui_splits", "threshold", "n_exact_match", "n_flag_protocol", "n_flag_cross", "n_flag_any", "protocol_distance_hist_le_thr")},
               "within_uieb_exact_sha_duplicate_groups": exact_within,
               "n_remaining_flag_vs_frozen90": sum(1 for r in rows if r["dhash_min_vs_frozen90"] <= THR),
               "n_remaining_flag_vs_remaining": sum(1 for r in rows if r["dhash_min_vs_remaining"] <= THR),
               "n_remaining_any_flag": sum(1 for r in rows if r["flag_lsui"] or r["flag_within_uieb"]),
               "calibration_frozen90_labelled": calib, "seconds": round(time.time() - t0, 1)}
    with open(out / "summary.json", "x", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    s = main(sys.argv[1]); print(json.dumps({k: v for k, v in s.items() if k != "within_uieb_exact_sha_duplicate_groups"}, indent=1)); print("exact within-UIEB groups:", len(s["within_uieb_exact_sha_duplicate_groups"]))
