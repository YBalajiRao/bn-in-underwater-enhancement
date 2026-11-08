"""TUB Stage 5 analysis (EXPLORATORY; protocol reports/paper1_narrowed/TUB_stage5_preregistration_FROZEN_DRAFT.md, SHA-256 1e3382c0..., sections 4-7). CPU only.
Input: the per-image CSV of one tub_stage5_infer.py run (best_valA square256, last square256 or best_valA aspect455 = sensitivities 3 / 4 use the same code).
Outcome per pair and seed k, per architecture (marinenet, attrunk): y = IN value minus BN value of the same seed: d (PSNR, dB), dA, dC, dG, dR.
Primary estimator (e): per scene-camera series with >= MIN_N valid stages, Spearman (average ranks) of (NTU, y) over stages; mean over the scene's valid series;
  mean over scenes (equal weight). A series whose x or y is constant has no defined rho and is reported (excluded), as are series with < MIN_N stages.
Secondary estimator (d): within-series OLS slope: demean x and y within each valid series, pooled slope = sum(xd*yd) / sum(xd^2) over the cohort.
Uncertainty: scene-cluster bootstrap, B = 10,000, complete scenes drawn with replacement, draws from numpy SeedSequence(20261120) - one draw list per cohort
  (each cohort's list is generated from a fresh SeedSequence(20261120) over its sorted scene list), shared by all seeds, both architectures, all endpoints and both
  estimators within that cohort. 95 % percentile CIs. Seed level separately: the 3 per-seed estimates, their SD and a t-interval (df 2).
"consistent" (per architecture): the per-seed sign agrees with the stated direction in all 3 seeds AND the 95 % bootstrap CI of the seed mean excludes 0
  (on the stated side); else "not established". Endpoints without a stated direction (E-T4) get "no direction stated".
Endpoints: E-T1..E-T4 on the NTU cohort (1,022; 29 scenes) with (e) and (d); E-T5 scene-weighted means on the primary cohort (1,153; 33 scenes);
  supplementary: E-T1..E-T4 with g_ach = |m(ref) - m(in)| (evaluation-geometry channel means) in place of NTU on the primary cohort.
Sensitivities: (1) all 1,183: E-T5, and E-T1..E-T4 on the 1,183 minus the NTU-copied scenes; (2) shift > 1 px series removed (NTU endpoints and E-T5);
  (5) NTU ranges low [0,10], medium (10,20], high > 20 (dataset authors' ranges): scene-weighted means per range and seed; (6) scene-wise and camera-wise tables;
  (7) near-blank: (5) plus input luma-SD quantiles per range. (3) and (4) = this script on the G455 and last runs.
Usage: python v2_tools/tub_stage5_analysis.py --run DIR --out DIR [--dry-run]"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))
import tub_stage5_infer as I  # noqa: E402  (cohort loading and frozen hashes; no torch import at module level)

SEED = 20261120
B = 10_000
MIN_N = 5
ARCHS = ("marinenet", "attrunk")
SEEDS = (0, 1, 2)
OUTCOMES = {"d": "psnr", "dA": "A", "dC": "C", "dG": "G", "dR": "R"}
ENDPOINTS = {"E-T1": ("d", -1), "E-T2": ("dA", 1), "E-T3": ("dG", 1), "E-T4a": ("dC", 0), "E-T4b": ("dR", 0)}
SHIFT_SERIES = {(1, 2), (11, 4), (14, 4), (15, 1), (27, 2), (27, 3), (33, 4)}
NTU_BINS = (("low_0_10", 0.0, 10.0), ("medium_10_20", 10.0, 20.0), ("high_gt20", 20.0, np.inf))  # (lo, hi]; low includes 0
T95_DF2 = 4.302652729749462  # scipy.stats.t.ppf(0.975, 2)


# ----------------------------------------------------------------------------- pure estimators (tested on synthetic data)
def avg_ranks(v):
    from scipy.stats import rankdata
    return rankdata(v, method="average")


def spearman(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def group_series(scene, cam):
    keys = {}
    for i, sc in enumerate(zip(np.asarray(scene).tolist(), np.asarray(cam).tolist())):
        keys.setdefault(sc, []).append(i)
    return keys


def series_info(scene, cam, min_n=MIN_N):
    """{(s,c): n} and the list of series dropped for n < min_n (reported)."""
    g = group_series(scene, cam)
    return {k: len(v) for k, v in g.items()}, sorted([[int(k[0]), int(k[1]), len(v)] for k, v in g.items() if len(v) < min_n])


def scene_values_e(scene, cam, x, y, scenes, min_n=MIN_N):
    """Estimator (e) scene values: per scene the mean of its valid series' Spearman rho (NaN if none); plus undefined-rho series."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    per = {s: [] for s in scenes}; undefined = []
    for (s, c), idx in group_series(scene, cam).items():
        if len(idx) < min_n:
            continue
        r = spearman(x[idx], y[idx])
        if np.isnan(r):
            undefined.append([int(s), int(c)]); continue
        per[s].append(r)
    return np.array([np.mean(per[s]) if per[s] else np.nan for s in scenes]), {"undefined_rho_series": sorted(undefined),
                                                                               "valid_series_per_scene": {int(s): len(per[s]) for s in scenes}}


def scene_sums_d(scene, cam, x, y, scenes, min_n=MIN_N):
    """Estimator (d) per-scene sums of within-series demeaned cross products (Sxy) and squares (Sxx), series with >= min_n points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = {s: 0.0 for s in scenes}; sxx = {s: 0.0 for s in scenes}
    for (s, c), idx in group_series(scene, cam).items():
        if len(idx) < min_n:
            continue
        xd = x[idx] - x[idx].mean(); yd = y[idx] - y[idx].mean()
        sxy[s] += float(xd @ yd); sxx[s] += float(xd @ xd)
    return np.array([sxy[s] for s in scenes]), np.array([sxx[s] for s in scenes])


def scene_means(scene, y, scenes):
    """E-T5 scene values: mean over the scene's pairs (NaN if none)."""
    y = np.asarray(y, float); scene = np.asarray(scene)
    return np.array([y[scene == s].mean() if np.any(scene == s) else np.nan for s in scenes])


def draw_counts(n_scenes, b=B, seed=SEED):
    """Scene-cluster bootstrap: b x n_scenes multiplicity matrix of complete scenes drawn with replacement; deterministic in (seed, n_scenes)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    idx = rng.integers(0, n_scenes, size=(b, n_scenes))
    return np.stack([np.bincount(r, minlength=n_scenes) for r in idx]), idx


def est_mean(vals, w=None):
    """Weighted mean of scene values over scenes with a value; w = multiplicities (one row or a matrix)."""
    v = np.asarray(vals, float); ok = ~np.isnan(v)
    if w is None:
        return float(v[ok].mean()) if ok.any() else np.nan
    w = np.asarray(w, float)
    den = w @ ok.astype(float)
    return np.where(den > 0, (w @ np.where(ok, v, 0.0)) / np.where(den > 0, den, 1), np.nan)


def est_ratio(sxy, sxx, w=None):
    if w is None:
        return float(np.sum(sxy) / np.sum(sxx)) if np.sum(sxx) > 0 else np.nan
    den = np.asarray(w, float) @ sxx
    return np.where(den > 0, (np.asarray(w, float) @ sxy) / np.where(den > 0, den, 1), np.nan)


def seed_stats(per_seed):
    v = np.asarray(per_seed, float); m = float(v.mean()); sd = float(v.std(ddof=1)); h = T95_DF2 * sd / np.sqrt(len(v))
    return {"per_seed": [float(x) for x in v], "seed_mean": m, "seed_sd": sd, "t95": [m - h, m + h]}


def pct_ci(boot):
    b = np.asarray(boot, float); b = b[~np.isnan(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [np.nan, np.nan]


def verdict(per_seed, ci, direction):
    if direction == 0:
        return "no direction stated"
    signs_ok = all(np.sign(v) == direction for v in per_seed)
    ci_ok = (ci[0] > 0) if direction > 0 else (ci[1] < 0)
    return "consistent" if (signs_ok and ci_ok) else "not established"


def summarise(point_per_seed, boot_per_seed, direction):
    """point_per_seed: 3 floats; boot_per_seed: 3 x B arrays on the SAME draws -> seed-level stats, bootstrap CIs (per seed and seed mean), verdict."""
    bm = np.mean(np.stack(boot_per_seed), axis=0)
    ci = pct_ci(bm)
    return dict(seed_stats(point_per_seed), boot_ci95_seed_mean=ci, boot_ci95_per_seed=[pct_ci(b) for b in boot_per_seed], direction=direction,
                verdict=verdict(point_per_seed, ci, direction), n_boot_undefined=int(np.isnan(bm).sum()))


# ----------------------------------------------------------------------------- data assembly
def load_run(run, prep=I.PREP):
    """Read the per-image CSV; verify completeness (12 models x 1,183 pairs), cohort flags against the frozen manifests, exactness record; return arrays."""
    run = Path(run)
    rows = list(csv.DictReader(open(run / "tub_per_image.csv", newline="")))
    summ = json.load(open(run / "tub_summary.json"))
    allp, kp, kn = I.load_cohorts(prep)
    meta = {I.pair_key(p): p for p in allp}
    keys = [I.pair_key(p) for p in allp]
    by = {(r["pair"], r["model"]): r for r in rows}
    if len(rows) != 12 * 1183 or len(by) != len(rows) or {r["model"] for r in rows} != set(I.MODELS12) or {r["pair"] for r in rows} != set(keys):
        raise SystemExit("STOP: per-image CSV is not exactly 12 models x 1,183 pairs")
    if not all(v["exact"] and v["max_abs_psnr_frozen_vs_mse_dB"] <= I.PSNR_TOL for v in summ["checks"].values()) or len(summ["checks"]) != 12:
        raise SystemExit("STOP: run summary records a failed exactness / PSNR-consistency check")
    for r in rows:
        if int(r["in_primary"]) != (r["pair"] in kp) or int(r["in_ntu"]) != (r["pair"] in kn):
            raise SystemExit(f"STOP: cohort flag mismatch for {r['pair']}")
    D = {"key": np.array(keys), "scene": np.array([meta[k]["scene"] for k in keys]), "cam": np.array([meta[k]["cam"] for k in keys]),
         "stage": np.array([meta[k]["stage"] for k in keys]), "ntu": np.array([float(meta[k]["ntu_input"]) for k in keys]),
         "primary": np.array([k in kp for k in keys]), "ntu_ok": np.array([k in kn for k in keys])}
    r0 = [by[(k, I.MODELS12[0])] for k in keys]
    D["luma_sd"] = np.array([float(r["input_luma_sd_eval"]) for r in r0])
    mu_in = np.array([[float(r[f"mu_in_{c}"]) for c in "rgb"] for r in r0]); mu_ref = np.array([[float(r[f"mu_ref_{c}"]) for c in "rgb"] for r in r0])
    D["g_ach"] = np.abs(mu_ref.mean(1) - mu_in.mean(1))
    for k in I.MODELS12:
        rr = [by[(key, k)] for key in keys]
        for col in set(OUTCOMES.values()):
            D[f"{k}|{col}"] = np.array([float(r[col]) for r in rr])
    for arch in ARCHS:
        for s in SEEDS:
            for o, col in OUTCOMES.items():
                D[f"{arch}|{s}|{o}"] = D[f"{arch}_in_s{s}|{col}"] - D[f"{arch}_bn_s{s}|{col}"]
    return D, summ


def cohort_masks(D):
    copied = np.isin(D["scene"], sorted(I.NTU_COPIED_SCENES))
    shift = np.array([(int(s), int(c)) in SHIFT_SERIES for s, c in zip(D["scene"], D["cam"])])
    return {"NTU": D["ntu_ok"], "PRIMARY": D["primary"], "ALL": np.ones(len(D["key"]), bool), "ALL_minus_NTU_copied": ~copied,
            "NTU_minus_shift": D["ntu_ok"] & ~shift, "PRIMARY_minus_shift": D["primary"] & ~shift}


def cohort_draws(mask, D, b=B):
    scenes = sorted(set(D["scene"][mask].tolist()))
    counts, _ = draw_counts(len(scenes), b)
    return scenes, counts


def trend_endpoints(D, mask, xname, b=B):
    """E-T1..E-T4 with estimators (e) and (d) on one cohort; x = 'ntu' or 'g_ach'."""
    scenes, W = cohort_draws(mask, D, b)
    sc, cm, x = D["scene"][mask], D["cam"][mask], D[xname][mask]
    n_by_series, dropped = series_info(sc, cm)
    out = {"n_pairs": int(mask.sum()), "n_scenes": len(scenes), "scenes": scenes, "n_series": len(n_by_series), "series_dropped_lt_min_n": dropped,
           "missing_obs_per_scene": {int(s): int(4 * 9 - sum(n for (ss, _), n in n_by_series.items() if ss == s)) for s in scenes}, "endpoints": {}}
    for ep, (o, direction) in ENDPOINTS.items():
        for arch in ARCHS:
            pe, be, pd, bd, info = [], [], [], [], None
            for s in SEEDS:
                y = D[f"{arch}|{s}|{o}"][mask]
                v, info_s = scene_values_e(sc, cm, x, y, scenes)
                info = info or info_s
                if info_s["undefined_rho_series"] != info["undefined_rho_series"]:
                    info.setdefault("undefined_rho_series_by_seed", {})[s] = info_s["undefined_rho_series"]
                pe.append(est_mean(v)); be.append(est_mean(v, W))
                sxy, sxx = scene_sums_d(sc, cm, x, y, scenes)
                pd.append(est_ratio(sxy, sxx)); bd.append(est_ratio(sxy, sxx, W))
            out["endpoints"][f"{ep}|{arch}"] = {"outcome": o, "x": xname, "e_spearman": summarise(pe, be, direction), "d_ols_slope": summarise(pd, bd, direction),
                                                "n_contributing_scenes": int(sum(1 for v in info["valid_series_per_scene"].values() if v)),
                                                "n_valid_series": int(sum(info["valid_series_per_scene"].values())), "undefined_rho_series": info["undefined_rho_series"]}
    return out


def level_endpoints(D, mask, b=B):
    """E-T5: scene-weighted means (mean over pairs within scene, then over scenes) of d, dA, dC, dG, dR per seed."""
    scenes, W = cohort_draws(mask, D, b)
    out = {"n_pairs": int(mask.sum()), "n_scenes": len(scenes), "endpoints": {}}
    for arch in ARCHS:
        for o in OUTCOMES:
            pts, bs = [], []
            for s in SEEDS:
                v = scene_means(D["scene"][mask], D[f"{arch}|{s}|{o}"][mask], scenes)
                pts.append(est_mean(v)); bs.append(est_mean(v, W))
            out["endpoints"][f"E-T5|{arch}|{o}"] = summarise(pts, bs, 0)
    return out


def bin_of(ntu):
    for name, lo, hi in NTU_BINS:
        if (lo == 0.0 and 0.0 <= ntu <= hi) or (lo < ntu <= hi):
            return name
    raise ValueError(ntu)


def ntu_range_tables(D, mask):
    """Sensitivities 5 and 7 (descriptive): per NTU range, scene-weighted mean of each outcome per seed, plus input luma-SD quantiles."""
    bins = np.array([bin_of(v) for v in D["ntu"]])
    out = {}
    for name, _, _ in NTU_BINS:
        m = mask & (bins == name)
        sc = sorted(set(D["scene"][m].tolist()))
        row = {"n_pairs": int(m.sum()), "n_scenes": len(sc),
               "input_luma_sd_quantiles_10_25_50_75_90": (np.quantile(D["luma_sd"][m], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if m.any() else None)}
        for arch in ARCHS:
            for o in OUTCOMES:
                row[f"{arch}|{o}"] = seed_stats([est_mean(scene_means(D["scene"][m], D[f"{arch}|{s}|{o}"][m], sc)) for s in SEEDS]) if m.any() else None
        out[name] = row
    return out


def group_table(D, mask, by):
    """Sensitivity 6: per scene or per camera, plain mean over the group's pairs of each outcome per seed (descriptive)."""
    rows = []
    for g in sorted(set(D[by][mask].tolist())):
        m = mask & (D[by] == g)
        r = {by: int(g), "n_pairs": int(m.sum())}
        for arch in ARCHS:
            for o in OUTCOMES:
                for s in SEEDS:
                    r[f"{arch}|{o}|s{s}"] = float(D[f"{arch}|{s}|{o}"][m].mean())
        rows.append(r)
    return rows


def analyse(D, b=B):
    C = cohort_masks(D)
    counts = {k: int(v.sum()) for k, v in C.items()}
    if (counts["ALL"], counts["PRIMARY"], counts["NTU"]) != (1183, 1153, 1022):
        raise SystemExit(f"STOP: cohort counts {counts}")
    if np.isin(D["scene"][C["NTU"]], sorted(I.NTU_COPIED_SCENES)).any():
        raise SystemExit("STOP: NTU-copied scene in an NTU analysis")
    return {"cohort_counts": counts,
            "primary": {"trend_NTU": trend_endpoints(D, C["NTU"], "ntu", b), "E-T5_PRIMARY": level_endpoints(D, C["PRIMARY"], b)},
            "supplementary_g_ach": trend_endpoints(D, C["PRIMARY"], "g_ach", b),
            "sens1_all_1183": {"E-T5_ALL": level_endpoints(D, C["ALL"], b), "trend_NTU_ALL_minus_NTU_copied": trend_endpoints(D, C["ALL_minus_NTU_copied"], "ntu", b)},
            "sens2_shift_removed": {"trend_NTU": trend_endpoints(D, C["NTU_minus_shift"], "ntu", b), "E-T5": level_endpoints(D, C["PRIMARY_minus_shift"], b)},
            "sens5_7_ntu_ranges_NTU_cohort": ntu_range_tables(D, C["NTU"]),
            "sens6_scene_table_PRIMARY": group_table(D, C["PRIMARY"], "scene"), "sens6_camera_table_PRIMARY": group_table(D, C["PRIMARY"], "cam")}


def flat_rows(R):
    rows = []
    def add(section, ep, est, s):
        rows.append([section, ep, est] + [f"{v:.6g}" for v in s["per_seed"]] + [f"{s['seed_mean']:.6g}", f"{s['seed_sd']:.6g}", f"{s['t95'][0]:.6g}", f"{s['t95'][1]:.6g}",
                     f"{s['boot_ci95_seed_mean'][0]:.6g}", f"{s['boot_ci95_seed_mean'][1]:.6g}", s["direction"], s["verdict"]])
    for section, blk in (("primary_NTU", R["primary"]["trend_NTU"]), ("supp_g_ach", R["supplementary_g_ach"]), ("sens1_NTU", R["sens1_all_1183"]["trend_NTU_ALL_minus_NTU_copied"]),
                         ("sens2_NTU", R["sens2_shift_removed"]["trend_NTU"])):
        for ep, v in blk["endpoints"].items():
            add(section, ep, "e_spearman", v["e_spearman"]); add(section, ep, "d_ols_slope", v["d_ols_slope"])
    for section, blk in (("E-T5_PRIMARY", R["primary"]["E-T5_PRIMARY"]), ("sens1_E-T5_ALL", R["sens1_all_1183"]["E-T5_ALL"]), ("sens2_E-T5", R["sens2_shift_removed"]["E-T5"])):
        for ep, v in blk["endpoints"].items():
            add(section, ep, "scene_weighted_mean", v)
    return rows


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    I.verify_hashes()
    I.load_cohorts()
    run, out = ROOT / a.run, ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    if a.dry_run:
        print(f"DRY RUN OK: protocol / cohort / E1r hashes; cohorts 1,183 / 1,153 / 1,022; {out} absent; run folder "
              f"{'present' if (run / 'tub_per_image.csv').exists() else 'absent (expected before inference)'}; nothing read from images or written")
        return {"dry_run": True}
    D, summ = load_run(run)
    R = analyse(D)
    R["meta"] = {"run": str(run), "per_image_sha256": sha(run / "tub_per_image.csv"), "run_meta": summ["meta"], "script_sha256": sha(__file__), "bootstrap": {"B": B, "seed": SEED},
                 "min_series_n": MIN_N, "label": "EXPLORATORY (post hoc); no mechanism claims"}
    out.mkdir(parents=True)
    json.dump(R, open(out / "tub_analysis.json", "x"), indent=1, default=float)
    with open(out / "tub_endpoints.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "endpoint|arch", "estimator", "seed0", "seed1", "seed2", "seed_mean", "seed_sd", "t95_lo", "t95_hi", "boot_lo", "boot_hi", "direction", "verdict"])
        w.writerows(flat_rows(R))
    for name in ("sens6_scene_table_PRIMARY", "sens6_camera_table_PRIMARY"):
        with open(out / f"{name}.csv", "x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(R[name][0])); w.writeheader(); w.writerows(R[name])
    print("written", out)
    return R


if __name__ == "__main__":
    main()
