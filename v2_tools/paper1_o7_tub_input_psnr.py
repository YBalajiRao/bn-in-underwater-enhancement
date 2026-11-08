"""Paper 1 open item O7 (DESCRIPTIVE, post hoc; CPU only; NO model inference).
Identity ("do-nothing") baseline on TUB: PSNR of each degraded input against its own Stage 01 reference, at the two evaluation geometries the models
were scored at (square256 = train_v2.preload; G455 = tub_stage5_infer.preload_geometry), with the frozen PSNR function (train_v2.compute_psnr),
plus the exact E1r decomposition of the identity error (A, C, G, R). Every image is hash-checked against the frozen cohort manifest before decoding.
Relates the baseline to measured NTU, per scene and per turbidity level, and to the saved model scores (square256 best_valA):
  gain over identity = model PSNR - input PSNR (BN and IN; seed mean and per seed).
Trends use the frozen TUB estimators and scene-bootstrap draw list (tub_stage5_analysis, unchanged; 95 % CIs conditional on the models).
Curves use the same post hoc NTU deciles as O2. Usage: python v2_tools/paper1_o7_tub_input_psnr.py --out paper1_uieb_runs/paper1_O7_tub_input_psnr"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import tub_stage5_infer as I  # noqa: E402  (frozen functions: load_cohorts, verify_files, preload_geometry, hashes)
import tub_stage5_analysis as A  # noqa: E402  (frozen estimators and draws)

ANALYSIS_SHA = "a01f5c668676bf40a4c15ab80e81de001f2320a46c165a87da414ff81383298d"
REF_RUN = "paper1_uieb_runs/tub_stage5_best_valA"
N_BINS = 10
ARCHS = {"marinenet": "MARINE-Net", "attrunk": "AT-Trunk"}
BN_COL, IN_COL, ID_COL = "#2a78d6", "#eb6834", "#3d3d3a"  # BN / IN validated categorical slots 1-2; identity in neutral ink (plus marker and label)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def identity_scores(allp, geometry):
    import torch
    import train_v2 as tv
    import s5_e1r_error_decomposition as Dm
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    X, Y = I.preload_geometry(allp, geometry)
    ps, parts = [], []
    for i in range(0, len(X), 32):
        x = X[i:i + 32].float() / 255.0; y = Y[i:i + 32].float() / 255.0
        ps += [tv.compute_psnr(x[j:j + 1], y[j:j + 1]) for j in range(len(x))]
        parts.append(Dm.decompose(x, y))
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ok, rel, _ = Dm.check_exact(d)
    if not ok:
        raise SystemExit(f"STOP: identity decomposition not exact ({geometry})")
    ps = np.array(ps, dtype=float); zero = d["MSE"] == 0  # input byte-identical to its reference (flagged IDENTICAL_TO_REFERENCE; outside both cohorts)
    zero_keys = [I.pair_key(allp[i]) for i in np.where(zero)[0]]
    if any(I.pair_key(p) in zero_keys and "IDENTICAL_TO_REFERENCE" not in p["flags"] for p in allp):
        raise SystemExit(f"STOP: zero-MSE pair not flagged identical-to-reference ({geometry}): {zero_keys}")
    dps = float(np.max(np.abs(ps[~zero] + 10 * np.log10(d["MSE"][~zero]))))
    if dps > I.PSNR_TOL:
        raise SystemExit(f"STOP: frozen PSNR vs decomposition PSNR {dps} dB ({geometry})")
    return ps, d, {"exact": ok, "max_rel_err": rel, "max_abs_psnr_vs_mse_dB_nonzero_mse": dps, "zero_mse_pairs": zero_keys,
                   "frozen_psnr_value_for_zero_mse": [float(v) for v in ps[zero]], "tensor_shape": list(X.shape)}


def binned(D, m, y, scenes, W, edges):
    b = np.clip(np.searchsorted(edges, D["ntu"], side="right") - 1, 0, N_BINS - 1)
    rows = []
    for k in range(N_BINS):
        mk = m & (b == k)
        ent = {"bin": k, "ntu_median": float(np.median(D["ntu"][mk])), "n_pairs": int(mk.sum()), "n_scenes": len(set(D["scene"][mk].tolist()))}
        for name, yy in y.items():
            v = A.scene_means(D["scene"][mk], yy[mk], scenes)
            bs = A.est_mean(v, W)
            ent[name] = {"mean": float(A.est_mean(v)), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
        rows.append(ent)
    return rows


def trend1(D, m, yy, scenes, W):
    v, info = A.scene_values_e(D["scene"][m], D["cam"][m], D["ntu"][m], yy[m], scenes)
    sxy, sxx = A.scene_sums_d(D["scene"][m], D["cam"][m], D["ntu"][m], yy[m], scenes)
    e, eb = A.est_mean(v), A.est_mean(v, W); s, sb = A.est_ratio(sxy, sxx), A.est_ratio(sxy, sxx, W)
    return {"e_spearman": float(e), "e_ci95": A.pct_ci(eb), "d_ols_slope_dB_per_NTU": float(s), "d_ci95": A.pct_ci(sb),
            "n_valid_series": int(sum(info["valid_series_per_scene"].values())), "undefined_rho_series": info["undefined_rho_series"]}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    if sha(ROOT / "v2_tools/tub_stage5_analysis.py") != ANALYSIS_SHA:
        raise SystemExit("STOP: frozen TUB analysis script hash differs")
    I.verify_hashes()
    allp, kp, kn = I.load_cohorts()
    I.verify_files(allp)  # 2,366 image hashes vs the frozen cohort manifest
    psq, dsq, chk_sq = identity_scores(allp, "square256"); print("square256 identity done", chk_sq, flush=True)
    p455, d455, chk_455 = identity_scores(allp, "aspect455"); print("G455 identity done", chk_455, flush=True)
    D, _ = A.load_run(ROOT / REF_RUN)  # frozen loader: verifies run completeness and cohort flags; rows in the cohort-manifest order
    assert list(D["key"]) == [I.pair_key(p) for p in allp]
    m = D["ntu_ok"]; scenes, W = A.cohort_draws(m, D)
    edges = np.quantile(D["ntu"][m], np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    R = {"label": "O7 DESCRIPTIVE post hoc: identity (input vs Stage 01 reference) PSNR on TUB; CPU only; no model inference",
         "script_sha256": sha(__file__), "analysis_script_sha256": ANALYSIS_SHA, "reference_run_per_image_sha256": sha(ROOT / REF_RUN / "tub_per_image.csv"),
         "identity_checks": {"square256": chk_sq, "G455": chk_455}, "ntu_decile_edges": edges.tolist()}
    # trends of the identity baseline vs NTU (NTU cohort)
    R["trend_identity_psnr_vs_ntu"] = {"square256": trend1(D, m, psq, scenes, W), "G455": trend1(D, m, p455, scenes, W)}
    R["trend_identity_components_vs_ntu_square256"] = {k: trend1(D, m, dsq[k], scenes, W) for k in ("A", "C", "G", "R")}
    # model gain over identity (square256 best_valA)
    R["gain_over_identity"] = {}
    curves_out = {}
    for arch in ARCHS:
        bn = np.mean([D[f"{arch}_bn_s{s}|psnr"] for s in (0, 1, 2)], axis=0); inn = np.mean([D[f"{arch}_in_s{s}|psnr"] for s in (0, 1, 2)], axis=0)
        g = {"BN_minus_input_seedmean": bn - psq, "IN_minus_input_seedmean": inn - psq}
        R["gain_over_identity"][arch] = {
            "trend": {k: trend1(D, m, v, scenes, W) for k, v in g.items()},
            "per_seed_trend": {f"{arm}_s{s}": trend1(D, m, D[f"{arch}_{arm.lower()}_s{s}|psnr"] - psq, scenes, W) for arm in ("BN", "IN") for s in (0, 1, 2)},
            "share_pairs_model_below_identity_NTU_cohort": {"BN": float((bn[m] < psq[m]).mean()), "IN": float((inn[m] < psq[m]).mean())}}
        curves_out[arch] = binned(D, m, {"identity": psq, "BN": bn, "IN": inn, "BN_gain": bn - psq, "IN_gain": inn - psq}, scenes, W, edges)
    R["decile_curves_square256_seedmean"] = curves_out
    # per turbidity level: dataset authors' ranges (scene-weighted)
    rng_b = np.array([A.bin_of(v) for v in D["ntu"]])
    R["ntu_ranges_identity"] = {}
    for name, _, _ in A.NTU_BINS:
        mk = m & (rng_b == name); sc = sorted(set(D["scene"][mk].tolist()))
        R["ntu_ranges_identity"][name] = {"n_pairs": int(mk.sum()), "n_scenes": len(sc),
                                          "identity_psnr_square256": float(A.est_mean(A.scene_means(D["scene"][mk], psq[mk], sc))),
                                          "identity_psnr_G455": float(A.est_mean(A.scene_means(D["scene"][mk], p455[mk], sc)))}
    # per stage (ordinal), all 1,183 and NTU cohort
    R["per_stage_identity_square256"] = {int(t): {"n_pairs_all": int((D["stage"] == t).sum()), "mean_all": float(psq[D["stage"] == t].mean()),
                                                 "median_ntu_ntu_cohort": float(np.median(D["ntu"][m & (D["stage"] == t)]))} for t in range(2, 11)}
    out.mkdir(parents=True)
    json.dump(R, open(out / "o7_results.json", "x"), indent=1, default=float)
    with open(out / "o7_per_pair.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "scene", "cam", "stage", "ntu_input", "in_primary", "in_ntu", "flags", "identity_psnr_square256", "identity_psnr_G455",
                    "id_MSE_sq256", "id_A_sq256", "id_C_sq256", "id_G_sq256", "id_R_sq256", "input_luma_sd_eval_sq256"]
                   + [f"{a}_{arm}_seedmean_psnr_sq256" for a in ARCHS for arm in ("bn", "in")])
        for i, p in enumerate(allp):
            k = I.pair_key(p)
            w.writerow([k, p["scene"], p["cam"], p["stage"], p["ntu_input"], int(k in kp), int(k in kn), "|".join(p["flags"]), f"{psq[i]:.6f}", f"{p455[i]:.6f}",
                        *(f"{dsq[c][i]:.10g}" for c in ("MSE", "A", "C", "G", "R")), f"{D['luma_sd'][i]:.6f}"]
                       + [f"{np.mean([D[f'{a}_{arm}_s{s}|psnr'][i] for s in (0, 1, 2)]):.6f}" for a in ARCHS for arm in ("bn", "in")])
    # per scene x stage (camera mean), all scenes; NTU flagged where copied
    with open(out / "o7_scene_stage.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "stage", "ntu_input", "ntu_reliable", "n_cams", "identity_psnr_sq256_cam_mean", "identity_psnr_G455_cam_mean"])
        for s in sorted(set(D["scene"].tolist())):
            for t in range(2, 11):
                mk = (D["scene"] == s) & (D["stage"] == t)
                if mk.any():
                    w.writerow([s, t, f"{D['ntu'][mk][0]:.4f}", int(s not in I.NTU_COPIED_SCENES), int(mk.sum()), f"{psq[mk].mean():.4f}", f"{p455[mk].mean():.4f}"])
    figures(R, D, m, psq, out)
    print("written", out)


def figures(R, D, m, psq, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3d3d3a",
                         "xtick.color": "#5c5c57", "ytick.color": "#5c5c57", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    # Fig 1: identity PSNR vs NTU per scene (camera mean), 29 NTU-cohort scenes, plus scene-weighted decile mean
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s in sorted(set(D["scene"][m].tolist())):
        xs, ys = [], []
        for t in range(2, 11):
            mk = m & (D["scene"] == s) & (D["stage"] == t)
            if mk.any():
                xs.append(D["ntu"][mk][0]); ys.append(psq[mk].mean())
        ax.plot(xs, ys, "-", color="#b8b7b0", lw=0.8)
    cv = R["decile_curves_square256_seedmean"]["marinenet"]
    x = [c["ntu_median"] for c in cv]; y = [c["identity"]["mean"] for c in cv]
    ax.fill_between(x, [c["identity"]["ci95"][0] for c in cv], [c["identity"]["ci95"][1] for c in cv], color=ID_COL, alpha=0.15, lw=0)
    ax.plot(x, y, "-", color=ID_COL, lw=2, marker="D", ms=4, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate("scene-weighted decile mean", (x[-1], y[-1]), xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=7, color="#3d3d3a")
    ax.annotate("one line per scene (camera mean)", (x[0], y[0]), xytext=(6, 14), textcoords="offset points", fontsize=7, color="#6b6b66")
    ax.set_xlabel("measured turbidity, NTU"); ax.set_ylabel("identity PSNR: input vs Stage 01 (dB)")
    ax.set_title("TUB: do-nothing baseline vs turbidity (square256; 29 scenes with reliable NTU)", fontsize=8, color="#3d3d3a")
    fig.tight_layout(); fig.savefig(out / "fig_o7_identity_vs_ntu_per_scene.png", dpi=200); plt.close(fig)
    # Fig 2: identity vs BN vs IN (seed mean), per architecture; Fig 3: gain over identity
    for kind in ("levels", "gain"):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
        for i, arch in enumerate(ARCHS):
            a = ax[i]; cv = R["decile_curves_square256_seedmean"][arch]; x = [c["ntu_median"] for c in cv]
            series = ((("identity", ID_COL, ":", "D", "identity (input)"), ("BN", BN_COL, "-", "o", "BN"), ("IN", IN_COL, "--", "s", "IN")) if kind == "levels"
                      else (("BN_gain", BN_COL, "-", "o", "BN − input"), ("IN_gain", IN_COL, "--", "s", "IN − input")))
            if kind == "gain":
                a.axhline(0, color="#8a8a85", lw=1)
            ends = []
            for key, col, ls, mk, lab in series:
                yv = [c[key]["mean"] for c in cv]
                a.fill_between(x, [c[key]["ci95"][0] for c in cv], [c[key]["ci95"][1] for c in cv], color=col, alpha=0.13, lw=0)
                a.plot(x, yv, ls, color=col, lw=2, marker=mk, ms=4, markeredgecolor="white", markeredgewidth=0.8, label=lab)
                ends.append([yv[-1], lab, x[-1]])
            ends.sort()
            for j in range(1, len(ends)):
                ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.45)
            for yl, lab, xl in ends:
                a.annotate(lab, (xl, yl), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color="#3d3d3a", annotation_clip=False)
            a.set_title(ARCHS[arch] + " (seed mean, best_valA, square256)", fontsize=8, color="#3d3d3a"); a.set_xlabel("measured turbidity, NTU (decile median)")
            a.legend(frameon=False, fontsize=7, loc="upper right" if kind == "levels" else "lower left")
            if i == 0:
                a.set_ylabel("PSNR vs Stage 01 (dB)" if kind == "levels" else "model PSNR − input PSNR (dB)")
        fig.suptitle("TUB: model PSNR against the do-nothing baseline" if kind == "levels" else "TUB: gain over doing nothing", fontsize=8, color="#3d3d3a")
        fig.tight_layout(rect=(0, 0, 0.95, 0.93)); fig.savefig(out / f"fig_o7_{kind}_vs_identity.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
