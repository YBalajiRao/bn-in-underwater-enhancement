"""Stage 3 H3-S3 analysis (S3 frozen protocol). Reads per-image PSNR (stage3_infer.py or the G1 per_image.csv, same column names) and the native
input images of a hash-checked UIEB manifest. Writes one new folder.
Per architecture a in {MARINE-Net, AT-Trunk}, per seed s in {0,1,2}:
  x_i = max(mu_R,mu_G,mu_B) - min(...) of the native input in [0,1]; d_is = PSNR_IN(i,s) - PSNR_BN(i,s)
  rho_s = Spearman(x, d_s) (average ranks); rho_bar = mean_s rho_s.
  Bootstrap: B = 10,000 resamples of image indices with replacement, numpy.default_rng(seed), ONE index matrix per cohort shared by both
  architectures, all seeds and the partial statistic; ranks are recomputed inside each resample; percentile CI at (1.25, 98.75) = 97.5 %.
  Partial Spearman: ranks of x, d_s, luma mean b, luma RMS contrast c; residualise rank(x) and rank(d_s) on [1, rank(b), rank(c)] by OLS;
  Pearson of the residuals; partial_bar = mean_s.
Decision (per architecture, primary cohort): see decide(). No pooling of image-seed rows.
Usage: python v2_tools/stage3_h3_analysis.py --per-image CSV --manifest M --manifest-sha256 H --expected-pairs N --cohorts JSON|none
       --out DIR [--rng-seed 20261102] [--label L]"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))

ARCH = {"MARINE-Net": ("marinenet_in", "marinenet_bn"), "AT-Trunk": ("attrunk_in", "attrunk_bn")}
SEEDS = (0, 1, 2)
B = 10_000
CI = (1.25, 98.75)
SESOI = 0.10  # operational threshold (protocol section 7), not an established minimum meaningful effect
FIXED_BINS = (0.14839622819495205, 0.30339161840200274)  # frozen-90 H3 tertile cut-points of native imbalance (h3_descriptors_per_image.csv)


def native_stats(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    mu = a.reshape(-1, 3).mean(0)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return {"imbalance": float(mu.max() - mu.min()), "luma_mean": float(y.mean()), "luma_rms_contrast": float(y.std())}


def rk(v):
    return stats.rankdata(v)


def pearson(u, v):
    u = u - u.mean(); v = v - v.mean(); den = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / den) if den > 0 else 0.0


def resid(r, Z):
    beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
    return r - Z @ beta


def stats_once(x, D, b, c):
    """x [n], D [n,S], b,c [n] -> (rho per seed, partial rho per seed)."""
    rx = rk(x); Z = np.column_stack([np.ones(len(x)), rk(b), rk(c)]); ex = resid(rx, Z)
    rho, prt = [], []
    for s in range(D.shape[1]):
        rd = rk(D[:, s]); rho.append(pearson(rx, rd)); prt.append(pearson(ex, resid(rd, Z)))
    return np.array(rho), np.array(prt)


def decide(rho_s, lo, hi, rbar, p_lo, p_bar):
    if rbar <= 0 or hi < SESOI:
        return "NOT SUPPORTED"
    core = bool(np.all(np.asarray(rho_s) > 0) and lo > 0 and rbar >= SESOI)
    if core and p_lo > 0 and p_bar >= 0.5 * rbar:
        return "SUPPORTED"
    if core:
        return "SUPPORTED, NOT CONFOUND-ROBUST"
    return "INCONCLUSIVE"


def analyse_cohort(names, desc, per, keep, rng_seed):
    sel = [n for n in names if n in keep]
    n = len(sel)
    x = np.array([desc[k]["imbalance"] for k in sel]); b = np.array([desc[k]["luma_mean"] for k in sel]); c = np.array([desc[k]["luma_rms_contrast"] for k in sel])
    idx = np.random.default_rng(rng_seed).integers(0, n, size=(B, n))
    out = {"n": n, "descriptor_associations": {"rho_x_luma_mean": float(stats.spearmanr(x, b).statistic), "rho_x_contrast": float(stats.spearmanr(x, c).statistic)}}
    for arch, (i_, b_) in ARCH.items():
        D = np.array([[float(per[k][f"{i_}_s{s}_psnr"]) - float(per[k][f"{b_}_s{s}_psnr"]) for s in SEEDS] for k in sel])
        rho, prt = stats_once(x, D, b, c)
        bs_r = np.empty((B, 3)); bs_p = np.empty((B, 3))
        for j in range(B):
            ii = idx[j]; bs_r[j], bs_p[j] = stats_once(x[ii], D[ii], b[ii], c[ii])
        rbar, pbar = float(rho.mean()), float(prt.mean())
        lo, hi = np.percentile(bs_r.mean(1), CI); plo, phi = np.percentile(bs_p.mean(1), CI)
        sd = float(rho.std(ddof=1)); th = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
        bins = np.digitize(x, FIXED_BINS)
        q = np.quantile(x, [1 / 3, 2 / 3]); tert = np.digitize(x, q)
        out[arch] = {
            "rho_per_seed": rho.tolist(), "rho_per_seed_ci97_5": [np.percentile(bs_r[:, s], CI).tolist() for s in range(3)],
            "rho_bar": rbar, "rho_bar_ci97_5": [float(lo), float(hi)],
            "partial_per_seed": prt.tolist(), "partial_bar": pbar, "partial_bar_ci97_5": [float(plo), float(phi)],
            "seed_level_descriptive": {"sd": sd, "t95_df2": [rbar - th, rbar + th]},
            "decision": decide(rho, lo, hi, rbar, plo, pbar),
            "rho_d_luma_mean_per_seed": [float(stats.spearmanr(D[:, s], b).statistic) for s in range(3)],
            "rho_d_contrast_per_seed": [float(stats.spearmanr(D[:, s], c).statistic) for s in range(3)],
            "mean_d_per_seed": D.mean(0).tolist(),
            "SUPP_fixed_bins_mean_d_per_seed": {lab: (D[bins == k].mean(0).tolist() if (bins == k).any() else None) for k, lab in enumerate(("low", "mid", "high"))},
            "SUPP_fixed_bins_n": {lab: int((bins == k).sum()) for k, lab in enumerate(("low", "mid", "high"))},
            "SUPP_tertiles_mean_d_per_seed": {lab: D[tert == k].mean(0).tolist() for k, lab in enumerate(("low", "mid", "high"))},
            "SUPP_tertile_cutpoints": q.tolist()}
    lowbin = out["AT-Trunk"]["SUPP_fixed_bins_mean_d_per_seed"]["low"]
    out["SUPP_AT_Trunk_low_bin_sign_flip_reproduced"] = (lowbin is not None and all(v < 0 for v in lowbin))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--per-image", "--manifest", "--manifest-sha256", "--cohorts", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--expected-pairs", type=int, required=True); ap.add_argument("--rng-seed", type=int, default=20261102); ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    import paper1_final_eval_v2 as V
    pairs, mh, _ = V.load_uieb_manifest2(a.manifest, a.manifest_sha256, a.expected_pairs, "UIEB", "stem")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    names = [Path(p["input_path"]).name for p in pairs]
    per = {r["filename"]: r for r in csv.DictReader(open(a.per_image, encoding="utf-8"))}
    assert set(per) == set(names), "per-image file and manifest list different images"
    desc = {Path(p["input_path"]).name: native_stats(p["input_path"]) for p in pairs}
    with open(out / "descriptors_per_image.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "imbalance", "luma_mean", "luma_rms_contrast"])
        for n in names:
            w.writerow([n, desc[n]["imbalance"], desc[n]["luma_mean"], desc[n]["luma_rms_contrast"]])
    if a.cohorts == "none":
        cohorts = {"all": set(names)}
    else:
        cj = json.load(open(a.cohorts, encoding="utf-8"))
        allset = set(names)
        cohorts = {"PRIMARY_787": allset - set(cj["primary_exclusions"]), "SENSITIVITY_A_800": allset, "SENSITIVITY_B_707": allset - set(cj["sensitivity_all_flagged_excluded"])}
        assert len(cohorts["PRIMARY_787"]) == 787 and len(cohorts["SENSITIVITY_A_800"]) == 800 and len(cohorts["SENSITIVITY_B_707"]) == 707, {k: len(v) for k, v in cohorts.items()}
    res = {"label": a.label, "manifest_sha256": mh, "per_image_file": str(a.per_image), "rng_seed": a.rng_seed, "B": B, "ci_percentiles": CI, "sesoi_operational": SESOI,
           "fixed_bins": FIXED_BINS, "script_sha256": V.E.sha256_file(__file__), "cohorts": {}}
    for k, keep in cohorts.items():
        res["cohorts"][k] = analyse_cohort(names, desc, per, keep, a.rng_seed)
    with open(out / "results.json", "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, r in res["cohorts"].items():
        print(k, r["n"], {arch: (r[arch]["decision"], round(r[arch]["rho_bar"], 3), [round(v, 3) for v in r[arch]["rho_per_seed"]]) for arch in ARCH})
    return res


if __name__ == "__main__":
    main()
