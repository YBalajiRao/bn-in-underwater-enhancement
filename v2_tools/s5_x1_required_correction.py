"""S5-X1 (EXPLORATORY diagnostic follow-up, designed after Stage 4): is the per-image IN - BN PSNR difference associated with the global correction
that the paired reference demands? CPU only; no model; reads frozen per-image results and the native input/reference images. Writes one new folder.

Definitions (native RGB / 255, float64; mu = per-channel mean over all pixels; m(v) = mean of the three channel means):
  x = max(mu_in) - min(mu_in);  b = mean BT.601 luma of input;  c = population SD of input luma     (identical to stage3_h3_analysis.native_stats)
  g_ach = |m(ref) - m(in)|;  g_chr = || (mu_ref - m(ref)) - (mu_in - m(in)) ||_2 / sqrt(3);  g_tot = || mu_ref - mu_in ||_2 / sqrt(3)
  (exact identity g_tot^2 = g_ach^2 + g_chr^2);  x_ref = max(mu_ref) - min(mu_ref).
  Sensitivity: the same on the 256x256 bilinear (antialiased) resize used by the evaluator.
d_s = PSNR_IN - PSNR_BN per image, best_valA, seed s in {0,1,2}, from the frozen Stage 3 / Stage 4 per-image files.
Populations: UIEB = Stage 3 PRIMARY 787; HICRD = Stage 4 ALL 2,000 (T30 clusters; sites from the frozen strata file).
Bootstrap: B = 10,000; UIEB resamples images, HICRD resamples complete T30 clusters; 95 % percentile CI; draws shared by all seeds, both architectures and all
statistics of a block. Random numbers: numpy SeedSequence(20261110).spawn(10), blocks in this fixed order:
  0 X1-a/b UIEB | 1 X1-a/b HICRD | 2 X1-c and X1-d (joint UIEB-image + HICRD-cluster draws, shared so the change is paired) | 3-6 X1-e sites 4,5,7,8 |
  7 sensitivity-256 UIEB | 8 sensitivity-256 HICRD | 9 unused.
X1-c: coarsened exact matching on tertile bins of (x, b, c) with cut-points from the pooled UIEB-787 + HICRD-2000 inputs; common strata need >= 20 images in
each dataset; non-estimable if < 3 common strata. Estimand (per seed): Delta = sum_h w_h (mean d_HICRD,h - mean d_UIEB,h) on the common-support population;
w_h = min(n_U,h, n_H,h) / sum min (overlap weights), frozen on the full cohort; equal weights 1/H as descriptive sensitivity. A bootstrap resample in which a
common stratum is empty in either dataset is recorded as failed and excluded (count reported). X1-d: the same with g_tot tertiles added.
Usage: python v2_tools/s5_x1_required_correction.py --out DIR"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ARCH = {"MARINE-Net": ("marinenet_in", "marinenet_bn"), "AT-Trunk": ("attrunk_in", "attrunk_bn")}
SEEDS = (0, 1, 2)
B = 10_000
CI = (2.5, 97.5)
MIN_N = 20
MIN_STRATA = 3
SITES_W = ("site4", "site5", "site7", "site8")
SRC = {  # path -> registered / verified SHA-256
    "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json": "dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08",
    "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json": "d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c",
    "paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv": "a792f4e8ed12231c",          # prefix; committed at d2feaff, verified unchanged
    "paper1_uieb_runs/stage3_analysis_best_valA/descriptors_per_image.csv": "661fb30d849c94f7",
    "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json": "29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d",
    "paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv": "87fc8b6ee0b393002aa245c4ec08f947e504b8cf6dca0874bbd2c257f7f00c27",
    "paper1_uieb_runs/stage4_analysis_best_valA/descriptors_per_image.csv": "ca895bab71efb8beaab7a81cee1ca38f64dd61da562b287333878540aa14433e",
    "paper1_uieb_runs/stage4_prep/hicrd_site_strata_T30.json": "8bb91c6610431d41430b14260c63ba457cf0877e61a3855ec9d7fac0ca5da8fc",
}


# ------------------------------------------------------------------ descriptors
def global_correction(mu_in, mu_ref):
    mu_in = np.asarray(mu_in, float); mu_ref = np.asarray(mu_ref, float)
    m_in, m_ref = mu_in.mean(), mu_ref.mean()
    g_ach = abs(m_ref - m_in)
    g_chr = float(np.linalg.norm((mu_ref - m_ref) - (mu_in - m_in)) / np.sqrt(3))
    g_tot = float(np.linalg.norm(mu_ref - mu_in) / np.sqrt(3))
    return {"g_ach": float(g_ach), "g_chr": g_chr, "g_tot": g_tot, "x_ref": float(mu_ref.max() - mu_ref.min())}


def image_stats(path, size=None):
    """channel means (and luma mean / SD for inputs) of a native image or of its 256x256 bilinear resize (as train_v2.preload)."""
    im = Image.open(path).convert("RGB")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64) / 255.0
    mu = a.reshape(-1, 3).mean(0); y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return mu, float(y.mean()), float(y.std())


# ------------------------------------------------------------------ statistics
def rk(v):
    return stats.rankdata(v)


def corr(u, v):
    u = u - u.mean(); v = v - v.mean(); den = np.sqrt((u @ u) * (v @ v))
    return float(u @ v / den) if den > 0 else 0.0


def seed_spearman(v, D, Z=None):
    """per-seed Spearman (Z None) or partial Spearman given covariate matrix Z of ranks (with intercept)."""
    rv = rk(v)
    if Z is not None:
        rv = rv - Z @ np.linalg.lstsq(Z, rv, rcond=None)[0]
    out = []
    for s in range(D.shape[1]):
        rd = rk(D[:, s])
        if Z is not None:
            rd = rd - Z @ np.linalg.lstsq(Z, rd, rcond=None)[0]
        out.append(corr(rv, rd))
    return np.array(out)


def covars(*vs):
    return np.column_stack([np.ones(len(vs[0]))] + [rk(v) for v in vs])


def seed_summary(r):
    r = np.asarray(r, float); sd = float(r.std(ddof=1)); h = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
    return {"per_seed": r.tolist(), "mean": float(r.mean()), "sd_across_seeds": sd, "t95_df2": [float(r.mean() - h), float(r.mean() + h)]}


def draws_images(n, rng, n_boot):
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)


def cluster_groups(cl):
    g = {}
    for i, k in enumerate(cl):
        g.setdefault(k, []).append(i)
    keys = sorted(g)
    return [np.array(g[k]) for k in keys]


def draws_clusters(groups, rng, n_boot):
    K = len(groups)
    for _ in range(n_boot):
        yield np.concatenate([groups[j] for j in rng.integers(0, K, size=K)])


def tertile_cuts(pooled):
    return np.quantile(np.asarray(pooled, float), [1 / 3, 2 / 3])


def codes(cols, cuts):
    """stratum code per image from tertile bins of each column."""
    bins = [np.digitize(np.asarray(c, float), q) for c, q in zip(cols, cuts)]
    return np.array(["-".join(str(int(v)) for v in t) for t in zip(*bins)])


def common_support(cu, ch, min_n=MIN_N):
    su, sh = {}, {}
    for k in cu:
        su[k] = su.get(k, 0) + 1
    for k in ch:
        sh[k] = sh.get(k, 0) + 1
    common = sorted(k for k in su if sh.get(k, 0) >= min_n and su[k] >= min_n)
    return common, {k: [su.get(k, 0), sh.get(k, 0)] for k in sorted(set(su) | set(sh))}


def overlap_weights(counts, common):
    m = np.array([min(counts[k]) for k in common], float)
    return m / m.sum()


def matched_diff(DU, cu, DH, ch, common, w):
    """per-seed sum_h w_h (mean d_H,h - mean d_U,h). Returns None if a common stratum is empty in either sample."""
    out = np.zeros(DU.shape[1])
    for k, wk in zip(common, w):
        iu = cu == k; ih = ch == k
        if not iu.any() or not ih.any():
            return None
        out += wk * (DH[ih].mean(0) - DU[iu].mean(0))
    return out


def matched_block(DU, cu, DH, ch, common, w, draws):
    """returns point [3], aligned bootstrap array [n_boot, 3] with NaN rows for failed resamples, number of failures."""
    point = matched_diff(DU, cu, DH, ch, common, w)
    bs = []
    for iu, ih in draws:
        v = matched_diff(DU[iu], cu[iu], DH[ih], ch[ih], common, w)
        bs.append(np.full(DU.shape[1], np.nan) if v is None else v)
    bs = np.array(bs)
    return point, bs, int(np.isnan(bs[:, 0]).sum())


def ci(arr):
    arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
    return [float(v) for v in np.percentile(arr, CI)] if len(arr) else None


# ------------------------------------------------------------------ analyses (pure; arrays in, dict out)
def x1_within(name, D, desc, draws, extra_cov=True):
    """X1-a (g_tot) and X1-b (g_ach, g_chr, partial g_tot | x,b,c) for one dataset; D: {arch: [n,3]}; desc: dict of arrays."""
    stats_def = {"g_tot": (desc["g_tot"], None), "g_ach": (desc["g_ach"], None), "g_chr": (desc["g_chr"], None)}
    if extra_cov:
        stats_def["g_tot_partial_xbc"] = (desc["g_tot"], "xbc")
    point, boot = {}, {k: {a: [] for a in D} for k in stats_def}
    for k, (v, cov) in stats_def.items():
        Z = covars(desc["x"], desc["b"], desc["c"]) if cov else None
        point[k] = {a: seed_spearman(v, D[a], Z) for a in D}
    nb = 0
    for ii in draws:
        nb += 1
        Zc = covars(desc["x"][ii], desc["b"][ii], desc["c"][ii])
        for k, (v, cov) in stats_def.items():
            for a in D:
                boot[k][a].append(seed_spearman(v[ii], D[a][ii], Zc if cov else None).mean())
    return {k: {a: dict(seed_summary(point[k][a]), ci95=ci(np.array(boot[k][a]))) for a in D} for k in stats_def} | {"n": int(len(desc["g_tot"])), "n_boot": nb}


def x1_matched(DU, descU, DH, descH, clH, draws_list, with_g):
    cols = ["x", "b", "c"] + (["g_tot"] if with_g else [])
    cuts = [tertile_cuts(np.concatenate([descU[k], descH[k]])) for k in cols]
    cu = codes([descU[k] for k in cols], cuts); ch = codes([descH[k] for k in cols], cuts)
    common, counts = common_support(cu, ch)
    res = {"columns": cols, "cut_points": {k: q.tolist() for k, q in zip(cols, cuts)}, "min_n_per_dataset": MIN_N, "stratum_counts_uieb_hicrd": counts, "common_strata": common}
    if len(common) < MIN_STRATA:
        res["estimable"] = False; res["note"] = f"non-estimable: {len(common)} common strata < {MIN_STRATA}"
        return res, None
    inU = np.isin(cu, common); inH = np.isin(ch, common)
    res.update({"estimable": True, "retained_uieb": int(inU.sum()), "retained_hicrd": int(inH.sum()), "retained_hicrd_clusters": int(len(set(clH[inH]))),
                "excluded_uieb": int((~inU).sum()), "excluded_hicrd": int((~inH).sum())})
    w_ov = overlap_weights(counts, common); w_eq = np.full(len(common), 1 / len(common))
    res["weights_overlap"] = dict(zip(common, w_ov.tolist())); res["population"] = "common-support population of the matched strata (not all UIEB or HICRD images)"
    out = {}
    for a in DU:
        po, bo, fo = matched_block(DU[a], cu, DH[a], ch, common, w_ov, draws_list)
        pe, be, fe = matched_block(DU[a], cu, DH[a], ch, common, w_eq, draws_list)
        out[a] = {"overlap_weighted": dict(seed_summary(po), ci95_seed_mean=ci(bo.mean(1)), ci95_per_seed=[ci(bo[:, s]) for s in range(3)], failed_resamples=fo),
                  "equal_weighted_SENS": dict(seed_summary(pe), ci95_seed_mean=ci(be.mean(1)), failed_resamples=fe),
                  "_boot_overlap_seedmean": bo.mean(1)}
    return res, out


def unmatched(DU, DH, draws_list):
    out = {}
    for a in DU:
        p = DH[a].mean(0) - DU[a].mean(0)
        bs = np.array([DH[a][ih].mean(0) - DU[a][iu].mean(0) for iu, ih in draws_list])
        out[a] = dict(seed_summary(p), ci95_seed_mean=ci(bs.mean(1)), ci95_per_seed=[ci(bs[:, s]) for s in range(3)], _boot_seedmean=bs.mean(1))
    return out


def analyse(U, H, n_boot=B, seed=20261110):
    """U/H: dicts with D {arch: [n,3]}, desc {x,b,c,g_tot,g_ach,g_chr,(g_*_256...)}; H also cl (T30) and site arrays."""
    ss = np.random.SeedSequence(seed).spawn(10)
    R = {}
    R["X1ab_UIEB"] = x1_within("UIEB", U["D"], U["desc"], draws_images(len(U["desc"]["x"]), np.random.default_rng(ss[0]), n_boot))
    gH = cluster_groups(H["cl"])
    R["X1ab_HICRD"] = x1_within("HICRD", H["D"], H["desc"], draws_clusters(gH, np.random.default_rng(ss[1]), n_boot))
    r2 = np.random.default_rng(ss[2]); nU = len(U["desc"]["x"])
    joint = [(r2.integers(0, nU, size=nU), np.concatenate([gH[j] for j in r2.integers(0, len(gH), size=len(gH))])) for _ in range(n_boot)]
    R["UNMATCHED"] = unmatched(U["D"], H["D"], joint)
    c_info, c_out = x1_matched(U["D"], U["desc"], H["D"], H["desc"], H["cl"], joint, with_g=False)
    d_info, d_out = x1_matched(U["D"], U["desc"], H["D"], H["desc"], H["cl"], joint, with_g=True)
    R["X1c_info"], R["X1d_info"] = c_info, d_info
    R["X1c"] = {a: {k: v for k, v in c_out[a].items() if not k.startswith("_")} for a in c_out} if c_out else None
    R["X1d_SENS"] = {a: {k: v for k, v in d_out[a].items() if not k.startswith("_")} for a in d_out} if d_out else None
    R["CHANGES"] = {}
    for a in U["D"]:
        ch = {}
        bu = R["UNMATCHED"][a]["_boot_seedmean"]
        if c_out:
            bc = c_out[a]["_boot_overlap_seedmean"]
            ch["matched_xbc_minus_unmatched"] = {"point": float(c_out[a]["overlap_weighted"]["mean"] - R["UNMATCHED"][a]["mean"]), "ci95_paired": ci(bc - bu),
                                                 "note": "paired draws (failed X1-c resamples dropped); the two estimands have different target populations (common support vs all images)"}
        if c_out and d_out:
            bd = d_out[a]["_boot_overlap_seedmean"]
            ch["matched_xbcg_minus_matched_xbc"] = {"point": float(d_out[a]["overlap_weighted"]["mean"] - c_out[a]["overlap_weighted"]["mean"]), "ci95_paired": ci(bd - bc),
                                                    "note": "paired draws; supports differ between X1-c and X1-d, so the change mixes matching and support change; descriptive"}
        gap_ci = R["UNMATCHED"][a]["ci95_seed_mean"]
        ch["unmatched_gap_distinguishable_from_zero"] = bool(gap_ci[0] > 0 or gap_ci[1] < 0)
        R["CHANGES"][a] = ch
    for a in R["UNMATCHED"]:
        R["UNMATCHED"][a].pop("_boot_seedmean", None)
    R["X1e_sites"] = {}
    for j, st in enumerate(SITES_W):
        idx = np.where(H["site"] == st)[0]
        if len(idx) == 0:
            R["X1e_sites"][st] = {"n": 0, "note": "site not present"}
            continue
        sub = {k: v[idx] for k, v in H["desc"].items()}
        Dsub = {a: H["D"][a][idx] for a in H["D"]}
        g = cluster_groups(H["cl"][idx])
        R["X1e_sites"][st] = x1_within(st, Dsub, sub, draws_clusters(g, np.random.default_rng(ss[3 + j]), n_boot), extra_cov=False) | {"n_clusters": len(g)}
    if "g_tot_256" in U["desc"]:
        for nm, D_, desc_, k, dr in (("UIEB", U["D"], U["desc"], 7, None), ("HICRD", H["D"], H["desc"], 8, gH)):
            d256 = {"g_tot": desc_["g_tot_256"], "g_ach": desc_["g_ach_256"], "g_chr": desc_["g_chr_256"], "x": desc_["x_256"], "b": desc_["b_256"], "c": desc_["c_256"]}
            gen = draws_images(len(desc_["x"]), np.random.default_rng(ss[k]), n_boot) if dr is None else draws_clusters(dr, np.random.default_rng(ss[k]), n_boot)
            R[f"SENS256_X1ab_{nm}"] = x1_within(nm, D_, d256, gen)
    return R


# ------------------------------------------------------------------ I/O
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_sources():
    for p, exp in SRC.items():
        got = sha(ROOT / p)
        if not got.startswith(exp):
            raise SystemExit(f"STOP: source {p} sha256 {got} does not match registered {exp}")


def per_image_d(path, key):
    rows = {r[key]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    return rows


def build(out):
    verify_sources()
    mU = json.load(open(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json", encoding="utf-8"))["pairs"]
    excl = set(json.load(open(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"))["primary_exclusions"])
    mU = [p for p in mU if p["filename"] not in excl]
    assert len(mU) == 787, len(mU)
    mH = json.load(open(ROOT / "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json", encoding="utf-8"))["pairs"]
    assert len(mH) == 2000
    stH = json.load(open(ROOT / "paper1_uieb_runs/stage4_prep/hicrd_site_strata_T30.json"))["stratum"]
    perU = per_image_d(ROOT / "paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv", "filename")
    perH = per_image_d(ROOT / "paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv", "id")
    dsU = per_image_d(ROOT / "paper1_uieb_runs/stage3_analysis_best_valA/descriptors_per_image.csv", "filename")
    dsH = per_image_d(ROOT / "paper1_uieb_runs/stage4_analysis_best_valA/descriptors_per_image.csv", "id")
    integ = {"max_abs_diff_frozen_vs_recomputed_xbc": 0.0, "files_hash_checked": 0, "n_uieb": 787, "n_hicrd": 2000}
    rows = []

    def one(key, p, frozen):
        ip, gp = ROOT / p["input_path"], ROOT / p["gt_path"]
        for path, k in ((ip, "input_sha256"), (gp, "gt_sha256")):
            if sha(path) != p[k]:
                raise SystemExit(f"STOP: file hash mismatch {path}")
            integ["files_hash_checked"] += 1
        mi, bi, ci_ = image_stats(ip); mr, _, _ = image_stats(gp)
        mi2, bi2, ci2 = image_stats(ip, (256, 256)); mr2, _, _ = image_stats(gp, (256, 256))
        x = float(mi.max() - mi.min())
        dmax = max(abs(x - float(frozen["imbalance"])), abs(bi - float(frozen["luma_mean"])), abs(ci_ - float(frozen["luma_rms_contrast"])))
        integ["max_abs_diff_frozen_vs_recomputed_xbc"] = max(integ["max_abs_diff_frozen_vs_recomputed_xbc"], dmax)
        if dmax > 1e-9:
            raise SystemExit(f"STOP: recomputed input descriptors differ from frozen for {key} ({dmax})")
        g = global_correction(mi, mr); g2 = global_correction(mi2, mr2)
        return {"key": key, "x": x, "b": bi, "c": ci_, **g, "x_256": float(mi2.max() - mi2.min()), "b_256": bi2, "c_256": ci2,
                "g_ach_256": g2["g_ach"], "g_chr_256": g2["g_chr"], "g_tot_256": g2["g_tot"], "x_ref_256": g2["x_ref"]}
    rU = [dict(one(p["filename"], p, dsU[p["filename"]]), dataset="UIEB") for p in mU]
    rH = [dict(one(p["id"], p, dsH[p["id"]]), dataset="HICRD", site=stH[p["id"]], cluster_T30=p["scene_cluster_T30"]) for p in mH]
    with open(Path(out) / "x1_descriptors_per_image.csv", "x", newline="") as f:
        cols = ["dataset", "key", "x", "b", "c", "g_ach", "g_chr", "g_tot", "x_ref", "x_256", "b_256", "c_256", "g_ach_256", "g_chr_256", "g_tot_256", "x_ref_256", "site", "cluster_T30"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r.get(c, "") for c in cols}) for r in rU + rH]
    keys = ["x", "b", "c", "g_ach", "g_chr", "g_tot", "x_ref", "x_256", "b_256", "c_256", "g_ach_256", "g_chr_256", "g_tot_256"]
    U = {"desc": {k: np.array([r[k] for r in rU]) for k in keys},
         "D": {a: np.array([[float(perU[r["key"]][f"{i_}_s{s}_psnr"]) - float(perU[r["key"]][f"{b_}_s{s}_psnr"]) for s in SEEDS] for r in rU]) for a, (i_, b_) in ARCH.items()}}
    H = {"desc": {k: np.array([r[k] for r in rH]) for k in keys}, "cl": np.array([r["cluster_T30"] for r in rH]), "site": np.array([r["site"] for r in rH]),
         "D": {a: np.array([[float(perH[r["key"]][f"{i_}_s{s}_psnr"]) - float(perH[r["key"]][f"{b_}_s{s}_psnr"]) for s in SEEDS] for r in rH]) for a, (i_, b_) in ARCH.items()}}
    return U, H, integ


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    out.mkdir(parents=True)
    t0 = time.time()
    U, H, integ = build(out)
    R = analyse(U, H)
    R["integrity"] = integ
    R["descriptor_summaries"] = {nm: {k: np.quantile(D_["desc"][k], [0.1, 0.5, 0.9]).tolist() for k in ("x", "b", "c", "g_ach", "g_chr", "g_tot", "x_ref")} for nm, D_ in (("UIEB", U), ("HICRD", H))}
    R["meta"] = {"status": "EXPLORATORY diagnostic follow-up (designed after Stage 4); not confirmatory", "script_sha256": sha(__file__), "B": B, "ci_percentiles": CI,
                 "seed": 20261110, "sources": SRC, "seconds": round(time.time() - t0, 1)}
    with open(out / "x1_results.json", "x", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
