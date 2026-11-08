"""S5-E1r analysis (EXPLORATORY): IN - BN differences of the exact MSE components, per dataset, architecture and seed.
Delta_k,s = mean over images of k(IN_s) - mean of k(BN_s), k in {MSE, A, C, G, R, R_low, R_high}  (positive = IN has the larger error component).
Uncertainty: UIEB image bootstrap; HICRD complete-T30-cluster bootstrap; B = 10,000; numpy SeedSequence(20261111).spawn(2) -> [UIEB, HICRD]; one draw list
per dataset shared by all seeds, both architectures and all components; 95 % percentile CI. Seed level (n = 3) reported separately (SD, t95 df 2).
No ratios or shares are used for any statement. Descriptive: which component has the largest Delta in each seed.
Secondary (exploratory): per-seed Spearman(Delta_A_i, g_ach_i) and Spearman(Delta_C_i, g_chr_i) with g from S5-X1.
Integrity: PSNR = -10 log10(MSE) vs the frozen Stage 3 / Stage 4 per-image PSNR.
Usage: python v2_tools/s5_e1r_analysis.py --e1r DIR --out DIR"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ARCH = {"MARINE-Net": ("marinenet_in", "marinenet_bn"), "AT-Trunk": ("attrunk_in", "attrunk_bn")}
COMP = ("MSE", "A", "C", "G", "R", "R_low", "R_high")
MAIN4 = ("A", "C", "G", "R")
B = 10_000
CI = (2.5, 97.5)


def deltas(V, arch_pair):
    """V: {model: {comp: [n]}} -> {comp: [3] per seed} mean IN - mean BN."""
    i_, b_ = arch_pair
    return {k: np.array([V[f"{i_}_s{s}"][k].mean() - V[f"{b_}_s{s}"][k].mean() for s in range(3)]) for k in COMP}


def seed_summary(r):
    r = np.asarray(r, float); sd = float(r.std(ddof=1)); h = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
    return {"per_seed": r.tolist(), "mean": float(r.mean()), "sd_across_seeds": sd, "t95_df2": [float(r.mean() - h), float(r.mean() + h)]}


def cluster_groups(cl):
    g = {}
    for i, k in enumerate(cl):
        g.setdefault(k, []).append(i)
    return [np.array(g[k]) for k in sorted(g)]


def draws(n, groups, rng, n_boot):
    for _ in range(n_boot):
        if groups is None:
            yield rng.integers(0, n, size=n)
        else:
            yield np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])


def analyse_dataset(V, n, groups, rng, gdesc=None, n_boot=B):
    res = {}
    point = {a: deltas(V, p) for a, p in ARCH.items()}
    sec_point = {}
    if gdesc is not None:
        for a, (i_, b_) in ARCH.items():
            sec_point[a] = {"dA_vs_g_ach": np.array([stats.spearmanr(V[f"{i_}_s{s}"]["A"] - V[f"{b_}_s{s}"]["A"], gdesc["g_ach"]).statistic for s in range(3)]),
                            "dC_vs_g_chr": np.array([stats.spearmanr(V[f"{i_}_s{s}"]["C"] - V[f"{b_}_s{s}"]["C"], gdesc["g_chr"]).statistic for s in range(3)])}
    boot = {a: {k: [] for k in COMP} for a in ARCH}; bsec = {a: {"dA_vs_g_ach": [], "dC_vs_g_chr": []} for a in sec_point}
    for ii in draws(n, groups, rng, n_boot):
        sub = {m: {k: v[k][ii] for k in COMP} for m, v in V.items()}
        for a, p in ARCH.items():
            d = deltas(sub, p)
            for k in COMP:
                boot[a][k].append(d[k])
            if a in bsec:
                i_, b_ = p
                bsec[a]["dA_vs_g_ach"].append(np.mean([stats.spearmanr(sub[f"{i_}_s{s}"]["A"] - sub[f"{b_}_s{s}"]["A"], gdesc["g_ach"][ii]).statistic for s in range(3)]))
                bsec[a]["dC_vs_g_chr"].append(np.mean([stats.spearmanr(sub[f"{i_}_s{s}"]["C"] - sub[f"{b_}_s{s}"]["C"], gdesc["g_chr"][ii]).statistic for s in range(3)]))
    for a in ARCH:
        res[a] = {}
        for k in COMP:
            bs = np.array(boot[a][k])
            res[a][k] = dict(seed_summary(point[a][k]), ci95_seed_mean=np.percentile(bs.mean(1), CI).tolist(), ci95_per_seed=[np.percentile(bs[:, s], CI).tolist() for s in range(3)])
        lead = [MAIN4[int(np.argmax([point[a][k][s] for k in MAIN4]))] for s in range(3)]
        res[a]["largest_delta_component_per_seed"] = lead
        res[a]["base_levels_seed_mean"] = {"BN": {k: float(np.mean([V[f"{ARCH[a][1]}_s{s}"][k].mean() for s in range(3)])) for k in MAIN4 + ("MSE",)},
                                           "IN": {k: float(np.mean([V[f"{ARCH[a][0]}_s{s}"][k].mean() for s in range(3)])) for k in MAIN4 + ("MSE",)}}
        if a in sec_point:
            res[a]["secondary"] = {k: dict(seed_summary(sec_point[a][k]), ci95_seed_mean=np.percentile(bsec[a][k], CI).tolist()) for k in sec_point[a]}
    res["n_images"] = int(n); res["n_boot"] = n_boot
    return res


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--e1r", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    out.mkdir(parents=True)
    rows = list(csv.DictReader(open(ROOT / a.e1r / "e1r_per_image.csv")))
    V = {"UIEB": {}, "HICRD": {}}; keys = {"UIEB": [], "HICRD": []}
    for r in rows:
        V[r["dataset"]].setdefault(r["model"], {k: [] for k in COMP})
        for k in COMP:
            V[r["dataset"]][r["model"]][k].append(float(r[k]))
        if r["model"] == "marinenet_bn_s0":
            keys[r["dataset"]].append(r["key"])
    for ds in V:
        for m in V[ds]:
            V[ds][m] = {k: np.array(v) for k, v in V[ds][m].items()}
    # integrity: same image order for all models; PSNR vs frozen per-image files
    integ = {}
    frozen = {"UIEB": ({r["filename"]: r for r in csv.DictReader(open(ROOT / "paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv"))}),
              "HICRD": ({r["id"]: r for r in csv.DictReader(open(ROOT / "paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv"))})}
    for ds in V:
        mx = 0.0
        for m, v in V[ds].items():
            ps = -10 * np.log10(v["MSE"])
            fz = np.array([float(frozen[ds][k][f"{m}_psnr"]) for k in keys[ds]])
            mx = max(mx, float(np.max(np.abs(ps - fz))))
        integ[f"{ds}_max_abs_psnr_diff_vs_frozen_dB"] = mx
        integ[f"{ds}_n_images"] = len(keys[ds]); integ[f"{ds}_n_models"] = len(V[ds])
    # clusters and X1 descriptors
    man = json.load(open(ROOT / "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json"))["pairs"]
    clmap = {p["id"]: p["scene_cluster_T30"] for p in man}
    xd = {(r["dataset"], r["key"]): r for r in csv.DictReader(open(ROOT / "paper1_uieb_runs/s5_x1/x1_descriptors_per_image.csv"))}
    g = {ds: {k: np.array([float(xd[(ds, key)][k]) for key in keys[ds]]) for k in ("g_ach", "g_chr")} for ds in V}
    ss = np.random.SeedSequence(20261111).spawn(2)
    R = {"meta": {"status": "EXPLORATORY (S5-E1r, after Stage 4 and X1)", "B": B, "ci_percentiles": CI, "seed": 20261111, "script_sha256": sha(__file__),
                  "e1r_per_image_sha256": sha(ROOT / a.e1r / "e1r_per_image.csv")}, "integrity": integ}
    R["UIEB"] = analyse_dataset(V["UIEB"], len(keys["UIEB"]), None, np.random.default_rng(ss[0]), g["UIEB"])
    R["HICRD"] = analyse_dataset(V["HICRD"], len(keys["HICRD"]), cluster_groups([clmap[k] for k in keys["HICRD"]]), np.random.default_rng(ss[1]), g["HICRD"])
    with open(out / "e1r_results.json", "x") as f:
        json.dump(R, f, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
