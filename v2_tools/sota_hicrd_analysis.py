"""Analysis for the SOTA HICRD-2000 comparison (spec: SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md, hash-checked). CPU only. PREPARED.
Inputs: Group R and Group T run folders (sota_hicrd_infer_r.py / _t.py), plus OUR frozen Stage 4 outputs (hash-pinned, read only):
  stage4_hicrd2000_best_valA/per_image.csv (square256 PSNR/SSIM, 12 models), stage4_hicrd2000_best_valA_G482/per_image.csv (G482),
  s5_e1r/e1r_per_image.csv (HICRD rows: E1r components of our 12 models at square256).
Registered contrasts (Amendment 02 B2): for each comparator m (within ONE group) and each of our arms a (3 seeds):
  d_{a,s}(i) = metric_{a,s}(i) - metric_m(i); per seed mean, seed mean, seed SD, t95 (df 2), 95 % percentile CI of the seed mean and of each seed
  from the Stage 4 draw list of the population (sota_hicrd_protocol.stage4_population_draws), share of images with d > 0 per seed.
  Group R: 6 x 4 = 24, Group T: 2 x 4 = 8 per metric (PSNR, SSIM). Group T rows carry the EXPLORATORY status. No cross-group contrast.
Populations: square256 -> ALL (primary), ALL_STRATIFIED, KNOWN, TEST300; aspect482 -> ALL (applicable comparators only).
Secondary: E1r component differences (dMSE, dA, dC, dG, dR) at square256 / ALL. U-Shape min-max sensitivity: separate table.
Bootstrap means use a draw-count matrix (mean over a drawn multiset == counts . values / n), tested equal to the direct computation.
Usage: python v2_tools/sota_hicrd_analysis.py --released DIR --retrained DIR --ushape-minmax DIR --released-g482 DIR --retrained-g482 DIR --out DIR [--dry-run]"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402

OURS = {"square256": ("paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv", "87fc8b6ee0b393002aa245c4ec08f947e504b8cf6dca0874bbd2c257f7f00c27"),
        "aspect482": ("paper1_uieb_runs/stage4_hicrd2000_best_valA_G482/per_image.csv", "d2353f95d1a5dfee3c4ddfe386d45b2f8b47f9cc6b7d982da564c6f1c3b3c436")}
OURS_E1R = ("paper1_uieb_runs/s5_e1r/e1r_per_image.csv", "384430561c71d7a1f340ff0b361c0cb371e30026e86943009b88684343dadfc4")
EXPECTED = {("R", "square256"): set(P.group_r_active()), ("R", "aspect482"): {"clahe", "udcp"},
            ("T", "square256"): {"waternet_lsui_s0", "wwe_uie_lsui_s0"}, ("T", "aspect482"): {"waternet_lsui_s0"}}
POPS = {"square256": ["ALL", "ALL_STRATIFIED", "KNOWN", "TEST300"], "aspect482": ["ALL"]}
COMPONENTS = ("MSE", "A", "C", "G", "R")
T95 = stats.t.ppf(0.975, 2)


def manifest_ids():
    return [p["id"] for p in json.load(open(ROOT / P.MANIFEST[0], encoding="utf-8"))["pairs"]]


def load_ours(geometry):
    rel, h = OURS[geometry]
    if K.sha(ROOT / rel) != h:
        raise SystemExit(f"STOP: frozen input {rel} hash differs")
    rows = list(csv.DictReader(open(ROOT / rel, newline="")))
    P.assert_aligned([r["id"] for r in rows])
    return {f"{a}_s{s}|{m}": np.array([float(r[f"{a}_s{s}_{m}"]) for r in rows]) for a in P.OUR_ARMS for s in P.SEEDS for m in ("psnr", "ssim")}


def load_ours_e1r(ids):
    if K.sha(ROOT / OURS_E1R[0]) != OURS_E1R[1]:
        raise SystemExit("STOP: frozen E1r input hash differs")
    by = {(r["key"], r["model"]): r for r in csv.DictReader(open(ROOT / OURS_E1R[0], newline="")) if r["dataset"] == "HICRD"}
    return {f"{a}_s{s}|{c}": np.array([float(by[(k, f"{a}_s{s}")][c]) for k in ids]) for a in P.OUR_ARMS for s in P.SEEDS for c in COMPONENTS}


def load_group(folder, group, geometry, ids):
    folder = Path(folder)
    files = {p.name[len("per_image_"):-4]: p for p in folder.glob("per_image_*.csv")}
    if set(files) != EXPECTED[(group, geometry)]:
        raise SystemExit(f"STOP: {folder} holds {sorted(files)}, expected exactly {sorted(EXPECTED[(group, geometry)])} for group {group}")
    out = {}
    for m, p in files.items():
        rows = K.read_per_image(p)
        if [r["id"] for r in rows] != ids:
            raise SystemExit(f"STOP: {p} rows not aligned with the manifest order")
        out[m] = {c: np.array([float(r[c]) for r in rows]) for c in ("psnr", "ssim") + COMPONENTS}
    return out


def count_matrix(draws, n):
    W = np.zeros((len(draws), n), dtype=np.float32)
    for b, ii in enumerate(draws):
        W[b] = np.bincount(ii, minlength=n)
    return W, np.array([len(ii) for ii in draws], dtype=np.float64)


def boot_means(W, lens, v):
    return (W.astype(np.float64) @ v) / lens


def pct(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def contrast(ours, comp, rows, W, lens):
    d = np.stack([ours[s][rows] - comp[rows] for s in range(3)])        # [3, n]
    per = d.mean(1); sm = float(per.mean()); sd = float(per.std(ddof=1)); h = T95 * sd / np.sqrt(3)
    return {"per_seed": per.tolist(), "seed_mean": sm, "seed_sd": sd, "t95": [sm - h, sm + h],
            "ci95_seed_mean": pct(boot_means(W, lens, d.mean(0))), "ci95_per_seed": [pct(boot_means(W, lens, d[s])) for s in range(3)],
            "share_d_gt_0_per_seed": [float((d[s] > 0).mean()) for s in range(3)]}


def analyse(released, retrained, ushape_mm, released_g, retrained_g):
    ids = manifest_ids()
    res = {"contrasts": [], "comparator_means": [], "e1r": [], "ushape_minmax": []}
    for geometry, rdir, tdir in (("square256", released, retrained), ("aspect482", released_g, retrained_g)):
        ours = load_ours(geometry)
        groups = {"R": load_group(rdir, "R", geometry, ids), "T": load_group(tdir, "T", geometry, ids)}
        for pop in POPS[geometry]:
            rows, draws = P.stage4_population_draws(pop)
            W, lens = count_matrix(draws, len(rows))
            for grp, comps in groups.items():
                status = P.GROUP_T_STATUS if grp == "T" else "descriptive (released model, n = 1)"
                for m, v in comps.items():
                    for metric in ("psnr", "ssim"):
                        res["comparator_means"].append({"geometry": geometry, "population": pop, "group": grp, "comparator": m, "metric": metric,
                                                        "mean": float(v[metric][rows].mean()), "ci95": pct(boot_means(W, lens, v[metric][rows]))})
                        for a in P.OUR_ARMS:
                            c = contrast([ours[f"{a}_s{s}|{metric}"] for s in P.SEEDS], v[metric], rows, W, lens)
                            res["contrasts"].append(dict(geometry=geometry, population=pop, group=grp, status=status, comparator=m, arm=a, metric=metric, **c))
            if geometry == "square256" and pop == "ALL":
                oe = load_ours_e1r(ids)
                for grp, comps in groups.items():
                    for m, v in comps.items():
                        for a in P.OUR_ARMS:
                            for comp in COMPONENTS:
                                c = contrast([oe[f"{a}_s{s}|{comp}"] for s in P.SEEDS], v[comp], rows, W, lens)
                                res["e1r"].append(dict(group=grp, comparator=m, arm=a, component="d" + comp, **c))
                if ushape_mm is not None:
                    mm = K.read_per_image(Path(ushape_mm) / "per_image_ushape_minmax.csv")
                    if [r["id"] for r in mm] != ids:
                        raise SystemExit("STOP: U-Shape min-max rows not aligned")
                    for metric in ("psnr", "ssim"):
                        vm = np.array([float(r[metric]) for r in mm])
                        for a in P.OUR_ARMS:
                            c = contrast([ours[f"{a}_s{s}|{metric}"] for s in P.SEEDS], vm, rows, W, lens)
                            res["ushape_minmax"].append(dict(comparator="ushape_minmax", arm=a, metric=metric, status="sensitivity (official min-max)", **c))
    return res


def write(res, out):
    out.mkdir(parents=True)
    json.dump(res, open(out / "analysis.json", "x"), indent=1)
    cols = ["geometry", "population", "group", "status", "comparator", "arm", "metric", "s0", "s1", "s2", "seed_mean", "seed_sd", "t95_lo", "t95_hi",
            "ci95_lo", "ci95_hi", "share_s0", "share_s1", "share_s2"]
    with open(out / "contrasts.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in res["contrasts"]:
            w.writerow([r["geometry"], r["population"], r["group"], r["status"], r["comparator"], r["arm"], r["metric"], *r["per_seed"], r["seed_mean"], r["seed_sd"],
                        *r["t95"], *r["ci95_seed_mean"], *r["share_d_gt_0_per_seed"]])
    with open(out / "comparator_means.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["geometry", "population", "group", "comparator", "metric", "mean", "ci95_lo", "ci95_hi"])
        for r in res["comparator_means"]:
            w.writerow([r["geometry"], r["population"], r["group"], r["comparator"], r["metric"], r["mean"], *r["ci95"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--released", "--retrained", "--released-g482", "--retrained-g482", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--ushape-minmax", required=True); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--test-mode", action="store_true")
    a = ap.parse_args(argv)
    K.verify_protocol_files()
    for geometry in OURS:
        if K.sha(ROOT / OURS[geometry][0]) != OURS[geometry][1]:
            raise SystemExit(f"STOP: frozen input {OURS[geometry][0]} hash differs")
    if K.sha(ROOT / OURS_E1R[0]) != OURS_E1R[1]:
        raise SystemExit("STOP: frozen E1r input hash differs")
    res_path = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    out = K.check_out(res_path(a.out), a.test_mode)
    plan = {"inputs": {k: str(res_path(getattr(a, k))) for k in ("released", "retrained", "released_g482", "retrained_g482", "ushape_minmax")},
            "out": str(out), "contrasts_per_metric": len(P.contrasts()), "populations": POPS}
    if a.dry_run:
        print("DRY RUN OK:", json.dumps(plan)); return {"dry_run": True, "plan": plan}
    res = analyse(*(res_path(getattr(a, k)) for k in ("released", "retrained", "ushape_minmax", "released_g482", "retrained_g482")))
    res["meta"] = dict(plan, protocol_sha256=K.PROTOCOL_V2[1], script_sha256=K.sha(__file__), label="descriptive; Group T exploratory; no ranking or superiority claim",
                       input_hashes={p.name + "@" + p.parent.name: K.sha(p) for d in plan["inputs"].values() for p in sorted(Path(d).glob("per_image_*.csv"))})
    write(res, out)
    print("written", out, {k: len(v) for k, v in res.items() if isinstance(v, list)})
    return res


if __name__ == "__main__":
    main()
