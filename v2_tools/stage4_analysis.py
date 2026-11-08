"""Stage 4 analysis: S4 v1 frozen protocol (2ea1f6ed) + Amendment 01 (8301b73d) + Amendment 02 (0b0753dc). Reads per-image PSNR (stage4_infer.py) and
input-only data (manifest 29402a7f: scene clusters; strata file 8bb91c66: sites); descriptors from the native input files. Writes one new folder.
x = channel-mean imbalance; b = mean luma; c = luma RMS contrast (native input, [0,1]); d_s = PSNR_IN - PSNR_BN per image, seed s in {0,1,2}.
Bootstrap units are ALWAYS complete T-clusters; one draw list per analysis is shared by all seeds, both architectures and all statistics of that analysis.
CI = two-sided 97.5 % percentile (1.25, 98.75): Bonferroni over the 2 architectures.
Analyses (Amendment 02 table B2):
  P1   ALL 2,000, pooled Spearman + partial(b,c); unstratified T30 cluster bootstrap rng 20261103 -> DECISION D1 (v1 rule = stage 3 decide()).
  P1s  ALL, same statistics, stratified bootstrap (6 sites + UNASSIGNED resampling block) rng 20261104 -> reported.
  A01_7cat  ALL, partial on 7 categories (incl. UNASSIGNED), P1s draws -> SUPERSEDED by A02, descriptive only, no decision field.
  P2   KNOWN-SITE 1,839, partial on 6 site indicators; stratified by 6 sites rng 20261106 -> DECISION D2 (decide_site()).
  P2c  KNOWN, pooled unadjusted, P2 draws.   P2b  KNOWN, partial on 6 sites + rank(b) + rank(c), P2 draws (sensitivity, not a decision).
  W    sites 4, 5, 7, 8 (A01 eligibility): within-site Spearman, cluster bootstrap within the site, one rng 20261105 used in that site order.
  U    UNASSIGNED 161: per-seed rho and mean d, imbalance range; no CI.
  S_T15, S_T60  ALL, pooled, unstratified bootstrap over T15 / T60 clusters, rng 20261103.
  S_cluster_level  T30 cluster means of x and d_s; Spearman over the 188 clusters; bootstrap over clusters rng 20261103.
  S_official_test300  official test split only; bootstrap over its T30 clusters rng 20261103.
  SUPP  fixed bins (frozen-90 cut-points), descriptor associations, mean d per site per seed.
Usage: python v2_tools/stage4_analysis.py --per-image CSV --out DIR [--label L] [--descriptors CSV]"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))

ARCH = {"MARINE-Net": ("marinenet_in", "marinenet_bn"), "AT-Trunk": ("attrunk_in", "attrunk_bn")}
SEEDS = (0, 1, 2)
B = 10_000
CI = (1.25, 98.75)
SESOI = 0.10  # operational threshold
FIXED_BINS = (0.14839622819495205, 0.30339161840200274)
SITES6 = ("site1", "site2", "site4", "site5", "site7", "site8")
UNASSIGNED = "TEST_UNASSIGNED"
WITHIN_SITES = ("site4", "site5", "site7", "site8")
MANIFEST = ROOT / "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json"
MANIFEST_SHA = "29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d"
STRATA = ROOT / "paper1_uieb_runs/stage4_prep/hicrd_site_strata_T30.json"
STRATA_SHA = "8bb91c6610431d41430b14260c63ba457cf0877e61a3855ec9d7fac0ca5da8fc"


# ----------------------------------------------------------------------------- pure statistics
def rk(v):
    return stats.rankdata(v)


def pearson(u, v):
    u = u - u.mean(); v = v - v.mean(); den = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / den) if den > 0 else 0.0


def resid(r, Z):
    beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
    return r - Z @ beta


def onehot(labels, cats):
    labels = np.asarray(labels)
    return np.column_stack([(labels == c).astype(float) for c in cats])


def design(kind, b, c, site, cats):
    """Covariate matrix on RANKS recomputed for the rows given. kind: None | 'bc' | 'site' | 'site_bc'."""
    n = len(b)
    if kind is None:
        return None
    if kind == "bc":
        return np.column_stack([np.ones(n), rk(b), rk(c)])
    if kind == "site":
        return onehot(site, cats)
    if kind == "site_bc":
        return np.column_stack([onehot(site, cats), rk(b), rk(c)])
    raise ValueError(kind)


def seed_rhos(x, Ds, Z):
    """Ds: list of [n,3] arrays (one per architecture). Returns list of 3-vectors. Z None -> Spearman; else partial Spearman on ranks."""
    rx = rk(x)
    ex = rx if Z is None else resid(rx, Z)
    out = []
    for D in Ds:
        r = []
        for s in range(D.shape[1]):
            rd = rk(D[:, s]); r.append(pearson(ex, rd if Z is None else resid(rd, Z)))
        out.append(np.array(r))
    return out


def cluster_index(cl):
    """cl: [n] cluster labels -> dict label -> np.array of row indices (complete clusters)."""
    d = {}
    for i, k in enumerate(cl):
        d.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in d.items()}


def draw_unstratified(cidx, rng, n_boot=None):
    keys = sorted(cidx); K = len(keys)
    for _ in range(n_boot or B):
        pick = rng.integers(0, K, size=K)
        yield np.concatenate([cidx[keys[j]] for j in pick])


def draw_stratified(cidx, cluster_stratum, rng, n_boot=None):
    """Within each stratum (sorted order), draw that stratum's clusters with replacement, as many as it has; union over strata."""
    by = {}
    for k in sorted(cidx):
        by.setdefault(cluster_stratum[k], []).append(k)
    order = sorted(by)
    for _ in range(n_boot or B):
        parts = []
        for st in order:
            ks = by[st]; pick = rng.integers(0, len(ks), size=len(ks))
            parts += [cidx[ks[j]] for j in pick]
        yield np.concatenate(parts)


def decide_d1(rho_s, lo, hi, rbar, p_lo, p_bar):
    """v1 rule (identical to stage 3)."""
    if rbar <= 0 or hi < SESOI:
        return "NOT SUPPORTED"
    core = bool(np.all(np.asarray(rho_s) > 0) and lo > 0 and rbar >= SESOI)
    if core and p_lo > 0 and p_bar >= 0.5 * rbar:
        return "SUPPORTED"
    if core:
        return "SUPPORTED, NOT CONFOUND-ROBUST"
    return "INCONCLUSIVE"


def decide_d2(prho_s, lo, hi, pbar):
    """Amendment 02 B3 rule on the 6-site-adjusted partial rho (known-site subset)."""
    if pbar <= 0 or hi < SESOI:
        return "WITHIN-SITE NOT SUPPORTED"
    if np.all(np.asarray(prho_s) > 0) and lo > 0 and pbar >= SESOI:
        return "WITHIN-SITE SUPPORTED"
    return "WITHIN-SITE INCONCLUSIVE"


def seed_level(r):
    sd = float(np.std(r, ddof=1)); h = stats.t.ppf(0.975, len(r) - 1) * sd / np.sqrt(len(r))
    return {"per_seed": [float(v) for v in r], "mean": float(np.mean(r)), "sd_across_seeds": sd, "t95_df2": [float(np.mean(r) - h), float(np.mean(r) + h)]}


def run(data, rows, kinds, draws, cats=None):
    """rows: row indices of the population. kinds: list of covariate kinds (None, 'bc', 'site', 'site_bc'). draws: iterable of row-index arrays INTO rows.
    Returns {kind: {arch: {...}}} with point estimates per seed, seed-level summary, bootstrap CI of the seed mean."""
    x, b, c, site = data["x"][rows], data["b"][rows], data["c"][rows], data["site"][rows]
    Ds = [data["D"][a][rows] for a in ARCH]
    point = {k: seed_rhos(x, Ds, design(k, b, c, site, cats)) for k in kinds}
    boot = {k: [[] for _ in ARCH] for k in kinds}
    nb = 0
    for ii in draws:
        nb += 1
        for k in kinds:
            r = seed_rhos(x[ii], [D[ii] for D in Ds], design(k, b[ii], c[ii], site[ii], cats))
            for j, v in enumerate(r):
                boot[k][j].append(v.mean())
    out = {"n_images": int(len(rows)), "n_boot": nb}
    for k in kinds:
        out[str(k)] = {}
        for j, a in enumerate(ARCH):
            bs = np.array(boot[k][j]) if nb else np.array([])
            ent = {"seed_level": seed_level(point[k][j])}
            if nb:
                ent["ci97_5"] = [float(v) for v in np.percentile(bs, CI)]
            out[str(k)][a] = ent
    return out


def analyse_all(data, n_boot=None):
    """data: x,b,c [n]; site [n] str (UNASSIGNED allowed); split [n] ('official_test'|'official_paired_train'); cl {'T15','T30','T60'}: [n] labels;
    D {arch: [n,3]}. Returns the full results dict."""
    n = len(data["x"]); allr = np.arange(n)
    known = np.where(data["site"] != UNASSIGNED)[0]; unas = np.where(data["site"] == UNASSIGNED)[0]
    res = {"populations": {"ALL": int(n), "KNOWN_SITE": int(len(known)), "UNASSIGNED": int(len(unas))}}
    cl30 = data["cl"]["T30"]
    # cluster integrity: each T30 cluster within one stratum
    cst = {}
    for i in range(n):
        cst.setdefault(cl30[i], set()).add(data["site"][i])
    assert all(len(v) == 1 for v in cst.values()), "a T30 cluster spans more than one stratum"
    cluster_stratum = {k: next(iter(v)) for k, v in cst.items()}
    res["n_clusters_T30"] = {"ALL": len(cst), "KNOWN_SITE": sum(1 for v in cluster_stratum.values() if v != UNASSIGNED), "UNASSIGNED": sum(1 for v in cluster_stratum.values() if v == UNASSIGNED)}

    def sub_cidx(rows, labels):
        """cluster index for a population, as positions within rows."""
        return cluster_index(labels[rows])
    # P1
    ci_all = sub_cidx(allr, cl30)
    P1 = run(data, allr, [None, "bc"], draw_unstratified(ci_all, np.random.default_rng(20261103), n_boot))
    res["P1_pooled_ALL"] = P1
    res["D1"] = {}
    for a in ARCH:
        pl, pb = P1["None"][a], P1["bc"][a]
        res["D1"][a] = decide_d1(pl["seed_level"]["per_seed"], *pl["ci97_5"], pl["seed_level"]["mean"], pb["ci97_5"][0], pb["seed_level"]["mean"])
    # P1s + superseded 7-category (same stratified draws)
    cats7 = sorted(set(data["site"]))
    draws_s = list(draw_stratified(ci_all, cluster_stratum, np.random.default_rng(20261104), n_boot))
    P1s = run(data, allr, [None, "bc"], iter(draws_s))
    res["P1s_pooled_ALL_stratified"] = P1s
    A7 = run(data, allr, ["site"], iter(draws_s), cats=cats7)
    A7["status"] = "SUPERSEDED by Amendment 02 (UNASSIGNED is not a site); descriptive only, no decision"
    A7["categories"] = cats7
    res["A01_7cat_SUPERSEDED"] = A7
    # P2, P2c, P2b on KNOWN
    ci_known = sub_cidx(known, cl30)
    cst_known = {k: cluster_stratum[k] for k in ci_known}
    P2all = run(data, known, ["site", None, "site_bc"], draw_stratified(ci_known, cst_known, np.random.default_rng(20261106), n_boot), cats=list(SITES6))
    res["P2_site_adjusted_KNOWN"] = {"n_images": P2all["n_images"], "n_boot": P2all["n_boot"], "site": P2all["site"]}
    res["P2c_pooled_KNOWN"] = {"n_images": P2all["n_images"], "n_boot": P2all["n_boot"], "None": P2all["None"]}
    res["P2b_site_bc_KNOWN_sensitivity"] = {"n_images": P2all["n_images"], "n_boot": P2all["n_boot"], "site_bc": P2all["site_bc"]}
    res["D2"] = {}
    for a in ARCH:
        p = P2all["site"][a]
        res["D2"][a] = decide_d2(p["seed_level"]["per_seed"], *p["ci97_5"], p["seed_level"]["mean"])
    res["claim_known_site_independent_of_site"] = {a: (res["D1"][a] == "SUPPORTED" and res["D2"][a] == "WITHIN-SITE SUPPORTED") for a in ARCH}
    # W
    rngW = np.random.default_rng(20261105); res["W_within_site"] = {}
    for st in WITHIN_SITES:
        rows = np.where(data["site"] == st)[0]
        res["W_within_site"][st] = run(data, rows, [None], draw_unstratified(sub_cidx(rows, cl30), rngW, n_boot))
        res["W_within_site"][st]["n_clusters"] = len(sub_cidx(rows, cl30))
    res["W_summary"] = {a: {"k_rho_bar_gt0": sum(res["W_within_site"][st]["None"][a]["seed_level"]["mean"] > 0 for st in WITHIN_SITES),
                            "m_ci_lower_gt0": sum(res["W_within_site"][st]["None"][a]["ci97_5"][0] > 0 for st in WITHIN_SITES)} for a in ARCH}
    # U
    U = run(data, unas, [None], iter([]))
    U["imbalance_range"] = [float(data["x"][unas].min()), float(data["x"][unas].max())] if len(unas) else None
    U["mean_d_per_seed"] = {a: data["D"][a][unas].mean(0).tolist() for a in ARCH} if len(unas) else None
    U["note"] = "UNASSIGNED is not an acquisition site; 3 clusters -> no CI"
    res["U_unassigned"] = U
    # S
    for T in ("T15", "T60"):
        res[f"S_{T}"] = run(data, allr, [None], draw_unstratified(sub_cidx(allr, data["cl"][T]), np.random.default_rng(20261103), n_boot))
    keys = sorted(ci_all); cx = {"x": [], "b": [], "c": [], "site": []}; cD = {a: [] for a in ARCH}
    for k in keys:
        rr = ci_all[k]
        cx["x"].append(data["x"][rr].mean()); cx["b"].append(data["b"][rr].mean()); cx["c"].append(data["c"][rr].mean()); cx["site"].append(cluster_stratum[k])
        for a in ARCH:
            cD[a].append(data["D"][a][rr].mean(0))
    cdata = {"x": np.array(cx["x"]), "b": np.array(cx["b"]), "c": np.array(cx["c"]), "site": np.array(cx["site"]), "D": {a: np.array(cD[a]) for a in ARCH}}
    rng = np.random.default_rng(20261103); K = len(keys)
    res["S_cluster_level"] = run(cdata, np.arange(K), [None], (rng.integers(0, K, size=K) for _ in range(n_boot or B)))
    test = np.where(data["split"] == "official_test")[0]
    res["S_official_test300"] = run(data, test, [None], draw_unstratified(sub_cidx(test, cl30), np.random.default_rng(20261103), n_boot))
    res["S_official_test300"]["n_clusters"] = len(sub_cidx(test, cl30))
    # SUPP
    bins = np.digitize(data["x"], FIXED_BINS)
    res["SUPP"] = {"fixed_bins_n": {lab: int((bins == k).sum()) for k, lab in enumerate(("low", "mid", "high"))},
                   "fixed_bins_mean_d_per_seed": {a: {lab: (data["D"][a][bins == k].mean(0).tolist() if (bins == k).any() else None) for k, lab in enumerate(("low", "mid", "high"))} for a in ARCH},
                   "rho_x_b": float(stats.spearmanr(data["x"], data["b"]).statistic), "rho_x_c": float(stats.spearmanr(data["x"], data["c"]).statistic),
                   "rho_d_b_per_seed": {a: [float(stats.spearmanr(data["D"][a][:, s], data["b"]).statistic) for s in SEEDS] for a in ARCH},
                   "rho_d_c_per_seed": {a: [float(stats.spearmanr(data["D"][a][:, s], data["c"]).statistic) for s in SEEDS] for a in ARCH},
                   "mean_d_per_site_per_seed": {a: {st: data["D"][a][data["site"] == st].mean(0).tolist() for st in sorted(set(data["site"]))} for a in ARCH}}
    return res


# ----------------------------------------------------------------------------- I/O
def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build_data(per_image_csv, descriptors_csv=None, out=None):
    if sha256(MANIFEST) != MANIFEST_SHA or sha256(STRATA) != STRATA_SHA:
        raise SystemExit("STOP: manifest or strata file hash differs")
    man = json.load(open(MANIFEST, encoding="utf-8"))["pairs"]; st = json.load(open(STRATA))["stratum"]
    per = {r["id"]: r for r in csv.DictReader(open(per_image_csv, encoding="utf-8"))}
    ids = [p["id"] for p in man]
    if set(per) != set(ids):
        raise SystemExit("STOP: per-image file and manifest list different images")
    if descriptors_csv:
        dsc = {r["id"]: {k: float(r[k]) for k in ("imbalance", "luma_mean", "luma_rms_contrast")} for r in csv.DictReader(open(descriptors_csv))}
    else:
        import stage3_h3_analysis as S3
        dsc = {p["id"]: S3.native_stats(ROOT / p["input_path"]) for p in man}
        if out is not None:
            with open(Path(out) / "descriptors_per_image.csv", "x", newline="") as f:
                w = csv.writer(f); w.writerow(["id", "imbalance", "luma_mean", "luma_rms_contrast"])
                [w.writerow([i, dsc[i]["imbalance"], dsc[i]["luma_mean"], dsc[i]["luma_rms_contrast"]]) for i in ids]
    data = {"ids": np.array(ids), "x": np.array([dsc[i]["imbalance"] for i in ids]), "b": np.array([dsc[i]["luma_mean"] for i in ids]),
            "c": np.array([dsc[i]["luma_rms_contrast"] for i in ids]), "site": np.array([st[i] for i in ids]), "split": np.array([p["split"] for p in man]),
            "cl": {T: np.array([p[f"scene_cluster_{T}"] for p in man]) for T in ("T15", "T30", "T60")},
            "D": {a: np.array([[float(per[i][f"{ii}_s{s}_psnr"]) - float(per[i][f"{bb}_s{s}_psnr"]) for s in SEEDS] for i in ids]) for a, (ii, bb) in ARCH.items()}}
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default=""); ap.add_argument("--descriptors")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    data = build_data(a.per_image, a.descriptors, out)
    res = analyse_all(data)
    res.update({"label": a.label, "per_image_file": str(a.per_image), "per_image_sha256": sha256(a.per_image), "manifest_sha256": MANIFEST_SHA, "strata_sha256": STRATA_SHA,
                "script_sha256": sha256(__file__), "B": B, "ci_percentiles": CI, "sesoi_operational": SESOI})
    with open(out / "results.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"D1": res["D1"], "D2": res["D2"], "claim_known_site_independent_of_site": res["claim_known_site_independent_of_site"]}, indent=1))
    return res


if __name__ == "__main__":
    main()
