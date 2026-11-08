"""Paper 1 open item O1 (DESCRIPTIVE, post hoc; computed from frozen saved outputs only; no inference).
Per-seed bootstrap CIs of the mean IN - BN PSNR difference d on UIEB-787 (Stage 3) and HICRD-2000 (Stage 4), using the FROZEN resampling units and draws:
  UIEB-787: image bootstrap, numpy.default_rng(20261102), one index matrix (B = 10,000) over the primary cohort in manifest order
            (identical to v2_tools/stage3_h3_analysis.analyse_cohort, which is imported and used unchanged for the reproduction check).
  HICRD:    complete T30 reef-area cluster bootstrap, unstratified, default_rng(20261103) (the Stage 4 P1 / D1 draws), plus the site-stratified
            P1s draws (default_rng(20261104)) as a sensitivity; draw generators imported unchanged from v2_tools/stage4_analysis.py.
Reproduction check (must pass, else STOP): with the same draws, the frozen rho_bar CIs saved in the Stage 3 / Stage 4 results.json are recomputed exactly.
Reported per seed and for the seed mean: point estimate, 97.5 % percentile CI (the registered level of Stages 3-4) and 95 % percentile CI. All CIs are
conditional on the three trained models per arm. Seed-level t95 (df 2) is reported separately and never combined with the bootstrap.
Usage: python v2_tools/paper1_o1_mean_d_cis.py --out paper1_uieb_runs/paper1_O1_mean_d_cis"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import stage3_h3_analysis as S3  # noqa: E402  (frozen; unchanged)
import stage4_analysis as S4  # noqa: E402  (frozen; unchanged)

FROZEN = {
    "v2_tools/stage3_h3_analysis.py": None, "v2_tools/stage4_analysis.py": None,  # hashes recorded at run time
    "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json": "dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08",
    "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json": "d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c",
}
UIEB_RUNS = {"best_valA": "paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv", "last": "paper1_uieb_runs/stage3_uieb800_last/per_image.csv"}
UIEB_SAVED = {"best_valA": "paper1_uieb_runs/stage3_analysis_best_valA/results.json", "last": "paper1_uieb_runs/stage3_analysis_last/results.json"}
HICRD_RUNS = {"best_valA": "paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv", "last": "paper1_uieb_runs/stage4_hicrd2000_last/per_image.csv",
              "best_valA_G482": "paper1_uieb_runs/stage4_hicrd2000_best_valA_G482/per_image.csv"}
HICRD_SAVED = {"best_valA": "paper1_uieb_runs/stage4_analysis_best_valA/results.json", "last": "paper1_uieb_runs/stage4_analysis_last/results.json",
               "best_valA_G482": "paper1_uieb_runs/stage4_analysis_G482/results.json"}
HICRD_DESC = "paper1_uieb_runs/stage4_analysis_best_valA/descriptors_per_image.csv"
ARCH = {"MARINE-Net": ("marinenet_in", "marinenet_bn"), "AT-Trunk": ("attrunk_in", "attrunk_bn")}
CI975, CI95 = (1.25, 98.75), (2.5, 97.5)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def summarise_mean_d(D, draws):
    """D: [n,3] per-image d; draws: list of index arrays. Returns per-seed and seed-mean point + CIs (97.5 %, 95 %), and seed-level t95."""
    bs = np.array([D[ii].mean(0) for ii in draws])  # [B,3]
    pt = D.mean(0); sm = bs.mean(1)
    sd = float(pt.std(ddof=1)); h = stats.t.ppf(0.975, 2) * sd / np.sqrt(3)
    return {"per_seed": pt.tolist(),
            "per_seed_ci97_5": [np.percentile(bs[:, s], CI975).tolist() for s in range(3)],
            "per_seed_ci95": [np.percentile(bs[:, s], CI95).tolist() for s in range(3)],
            "seed_mean": float(pt.mean()), "seed_mean_ci97_5": np.percentile(sm, CI975).tolist(), "seed_mean_ci95": np.percentile(sm, CI95).tolist(),
            "seed_level_sd": sd, "seed_level_t95_df2": [float(pt.mean() - h), float(pt.mean() + h)],
            "share_images_IN_gt_BN_per_seed": (D > 0).mean(0).tolist()}


def uieb(run):
    import paper1_final_eval_v2 as V
    pairs, mh, _ = V.load_uieb_manifest2(str(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json"),
                                          FROZEN["paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json"], 800, "UIEB", "stem")
    names = [Path(p["input_path"]).name for p in pairs]
    excl = set(json.load(open(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"))["primary_exclusions"])
    sel = [n for n in names if n not in excl]
    assert len(sel) == 787
    per = {r["filename"]: r for r in csv.DictReader(open(ROOT / UIEB_RUNS[run], encoding="utf-8"))}
    saved = json.load(open(ROOT / UIEB_SAVED[run]))
    assert saved["rng_seed"] == 20261102 and saved["B"] == S3.B and list(saved["ci_percentiles"]) == list(CI975)
    idx = np.random.default_rng(saved["rng_seed"]).integers(0, len(sel), size=(S3.B, len(sel)))  # exactly as analyse_cohort
    desc = {r["filename"]: r for r in csv.DictReader(open(ROOT / UIEB_SAVED[run].replace("results.json", "descriptors_per_image.csv")))}
    x = np.array([float(desc[k]["imbalance"]) for k in sel]); b = np.array([float(desc[k]["luma_mean"]) for k in sel]); c = np.array([float(desc[k]["luma_rms_contrast"]) for k in sel])
    out, check = {}, {}
    for arch, (i_, b_) in ARCH.items():
        D = np.array([[float(per[k][f"{i_}_s{s}_psnr"]) - float(per[k][f"{b_}_s{s}_psnr"]) for s in (0, 1, 2)] for k in sel])
        assert np.allclose(D.mean(0), saved["cohorts"]["PRIMARY_787"][arch]["mean_d_per_seed"], atol=1e-12)
        # reproduction check of the frozen draws: rho_bar CI (stats_once is the frozen statistic)
        bs_r = np.array([S3.stats_once(x[ii], D[ii], b[ii], c[ii])[0].mean() for ii in idx])
        ci = np.percentile(bs_r, CI975).tolist()
        check[arch] = {"recomputed_rho_bar_ci97_5": ci, "saved": saved["cohorts"]["PRIMARY_787"][arch]["rho_bar_ci97_5"],
                       "identical": bool(np.allclose(ci, saved["cohorts"]["PRIMARY_787"][arch]["rho_bar_ci97_5"], atol=1e-12))}
        out[arch] = summarise_mean_d(D, list(idx))
    return {"n": len(sel), "resampling_unit": "image", "rng": "default_rng(20261102)", "B": S3.B, "results": out, "draw_reproduction_check": check}


def hicrd(run):
    data = S4.build_data(str(ROOT / HICRD_RUNS[run]), descriptors_csv=str(ROOT / HICRD_DESC))
    saved = json.load(open(ROOT / HICRD_SAVED[run]))
    n = len(data["x"]); allr = np.arange(n)
    cidx = S4.cluster_index(data["cl"]["T30"][allr])
    cst = {}
    for i in range(n):
        cst.setdefault(data["cl"]["T30"][i], set()).add(data["site"][i])
    cluster_stratum = {k: next(iter(v)) for k, v in cst.items()}
    draws_u = list(S4.draw_unstratified(cidx, np.random.default_rng(20261103)))
    draws_s = list(S4.draw_stratified(cidx, cluster_stratum, np.random.default_rng(20261104)))
    check = {}
    for arch in ARCH:
        D = data["D"][arch]
        bs_r = np.array([S4.seed_rhos(data["x"][ii], [D[ii]], S4.design(None, data["b"][ii], data["c"][ii], data["site"][ii], None))[0].mean() for ii in draws_u])
        ci = np.percentile(bs_r, S4.CI).tolist(); sv = saved["P1_pooled_ALL"]["None"][arch]["ci97_5"]
        check[arch] = {"recomputed_P1_rho_bar_ci97_5": ci, "saved": sv, "identical": bool(np.allclose(ci, sv, atol=1e-12))}
    out = {a: {"T30_unstratified_P1_draws": summarise_mean_d(data["D"][a], draws_u), "T30_site_stratified_P1s_draws": summarise_mean_d(data["D"][a], draws_s)} for a in ARCH}
    return {"n": n, "n_clusters_T30": len(cidx), "resampling_unit": "complete T30 reef-area cluster", "rng": "P1 default_rng(20261103); P1s default_rng(20261104)",
            "B": S4.B, "results": out, "draw_reproduction_check": check}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    for rel, h in FROZEN.items():
        if h is not None and sha(ROOT / rel) != h:
            raise SystemExit(f"STOP: {rel} hash differs")
    inputs = {rel: sha(ROOT / rel) for rel in [*FROZEN, *UIEB_RUNS.values(), *UIEB_SAVED.values(), *HICRD_RUNS.values(), *HICRD_SAVED.values(), HICRD_DESC]}
    R = {"label": "O1 DESCRIPTIVE post hoc: per-seed CIs of mean IN-BN PSNR from frozen saved outputs; CIs conditional on the trained models",
         "script_sha256": sha(__file__), "inputs_sha256": inputs, "UIEB_787": {}, "HICRD_2000": {}}
    for run in UIEB_RUNS:
        R["UIEB_787"][run] = uieb(run); print("UIEB", run, {k: v["identical"] for k, v in R["UIEB_787"][run]["draw_reproduction_check"].items()}, flush=True)
    for run in HICRD_RUNS:
        R["HICRD_2000"][run] = hicrd(run); print("HICRD", run, {k: v["identical"] for k, v in R["HICRD_2000"][run]["draw_reproduction_check"].items()}, flush=True)
    ok = all(v["identical"] for blk in (R["UIEB_787"], R["HICRD_2000"]) for r in blk.values() for v in r["draw_reproduction_check"].values())
    R["all_draw_reproduction_checks_identical"] = ok
    if not ok:
        raise SystemExit("STOP: frozen draws not reproduced; nothing written")
    out.mkdir(parents=True)
    json.dump(R, open(out / "o1_results.json", "x"), indent=1)
    with open(out / "o1_mean_d_table.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "run", "arch", "draws", "seed", "mean_d_dB", "ci97_5_lo", "ci97_5_hi", "ci95_lo", "ci95_hi", "share_IN_gt_BN"])
        rows = [("UIEB_787", run, arch, "image", blk["results"][arch]) for run, blk in R["UIEB_787"].items() for arch in ARCH]
        rows += [("HICRD_2000", run, arch, dk, blk["results"][arch][dk]) for run, blk in R["HICRD_2000"].items() for arch in ARCH
                 for dk in ("T30_unstratified_P1_draws", "T30_site_stratified_P1s_draws")]
        for ds, run, arch, dk, e in rows:
            for s in range(3):
                w.writerow([ds, run, arch, dk, s, f"{e['per_seed'][s]:.4f}", *(f"{v:.4f}" for v in e["per_seed_ci97_5"][s]), *(f"{v:.4f}" for v in e["per_seed_ci95"][s]),
                            f"{e['share_images_IN_gt_BN_per_seed'][s]:.4f}"])
            w.writerow([ds, run, arch, dk, "mean", f"{e['seed_mean']:.4f}", *(f"{v:.4f}" for v in e["seed_mean_ci97_5"]), *(f"{v:.4f}" for v in e["seed_mean_ci95"]), ""])
            w.writerow([ds, run, arch, dk, "t95_df2(seed level)", "", f"{e['seed_level_t95_df2'][0]:.4f}", f"{e['seed_level_t95_df2'][1]:.4f}", "", "", ""])
    print("written", out, "all reproduction checks identical:", ok)


if __name__ == "__main__":
    main()
