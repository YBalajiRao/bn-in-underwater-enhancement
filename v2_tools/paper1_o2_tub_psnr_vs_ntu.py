"""Paper 1 open item O2 (DESCRIPTIVE, post hoc; frozen saved TUB outputs only; no inference, no image read).
Absolute BN and IN PSNR versus measured NTU on the TUB NTU cohort (1,022 pairs, 29 scenes), per seed and architecture, plus the gap d = IN - BN.
Purpose: show whether the registered-direction reversal of E-T1 (d rises with NTU) comes from IN improving, BN degrading faster, or both converging.
Methods (reusing v2_tools/tub_stage5_analysis.py functions unchanged; that script's frozen SHA-256 is checked):
  - Trend per outcome (BN PSNR, IN PSNR, d): estimator (e) within-series Spearman vs NTU averaged to scene then over scenes, and estimator (d) pooled
    within-series OLS slope (dB / NTU); series >= 5 stages; scene-cluster bootstrap B = 10,000 with the frozen draw list (SeedSequence(20261120) over the
    sorted NTU-cohort scenes, identical to the frozen analysis); 95 % percentile CIs, conditional on the trained models; seed-level t95 separately.
  - Curves: NTU deciles over the 1,022 pairs (post hoc binning, equal-count); per bin the scene-weighted mean PSNR (mean over the scene's pairs in the
    bin, then over scenes), with 95 % scene-bootstrap bands from the same draw list; median input luma SD per bin.
Runs: square256 best_valA (reference), square256 last (exploratory), G455 best_valA (sensitivity), G455 last (supplementary).
Usage: python v2_tools/paper1_o2_tub_psnr_vs_ntu.py --out paper1_uieb_runs/paper1_O2_tub_psnr_vs_ntu"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools"))
import tub_stage5_analysis as A  # noqa: E402  (frozen, unchanged)

ANALYSIS_SHA = "a01f5c668676bf40a4c15ab80e81de001f2320a46c165a87da414ff81383298d"
RUNS = {"square256_best_valA": ("paper1_uieb_runs/tub_stage5_best_valA", "reference"),
        "square256_last": ("paper1_uieb_runs/tub_stage5_last", "exploratory"),
        "G455_best_valA": ("paper1_uieb_runs/tub_stage5_best_valA_G455", "sensitivity"),
        "G455_last": ("paper1_uieb_runs/tub_stage5_last_G455", "supplementary (batch-equivalence preflight not run)")}
ARCHS = {"marinenet": "MARINE-Net", "attrunk": "AT-Trunk"}
N_BINS = 10
BN_COL, IN_COL = "#2a78d6", "#eb6834"  # validated categorical slots 1-2 (dataviz reference palette; all checks PASS, light mode)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def trend(D, m, y, scenes, W):
    sc, cm, x = D["scene"][m], D["cam"][m], D["ntu"][m]
    pe, be, pd_, bd = [], [], [], []
    for yy in y:
        v, _ = A.scene_values_e(sc, cm, x, yy[m], scenes); pe.append(A.est_mean(v)); be.append(A.est_mean(v, W))
        sxy, sxx = A.scene_sums_d(sc, cm, x, yy[m], scenes); pd_.append(A.est_ratio(sxy, sxx)); bd.append(A.est_ratio(sxy, sxx, W))
    e = A.summarise(pe, be, 0); d = A.summarise(pd_, bd, 0)
    for z in (e, d):
        z.pop("verdict"); z.pop("direction")
    return {"e_spearman": e, "d_ols_slope_dB_per_NTU": d}


def curves(D, m, y, scenes, W, edges):
    b = np.clip(np.searchsorted(edges, D["ntu"], side="right") - 1, 0, N_BINS - 1)
    rows = []
    for k in range(N_BINS):
        mk = m & (b == k)
        ent = {"bin": k, "ntu_lo": float(edges[k]), "ntu_hi": float(edges[k + 1]), "ntu_median": float(np.median(D["ntu"][mk])), "n_pairs": int(mk.sum()),
               "n_scenes": len(set(D["scene"][mk].tolist())), "input_luma_sd_median": float(np.median(D["luma_sd"][mk]))}
        for name, yy in y.items():
            v = A.scene_means(D["scene"][mk], yy[mk], scenes)
            pt = A.est_mean(v); bs = A.est_mean(v, W)
            ent[name] = {"mean": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
        rows.append(ent)
    return rows


def analyse_run(run_dir):
    D, summ = A.load_run(ROOT / run_dir)
    m = D["ntu_ok"]
    assert m.sum() == 1022
    scenes, W = A.cohort_draws(m, D)
    assert len(scenes) == 29 and W.shape == (A.B, 29)
    edges = np.quantile(D["ntu"][m], np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    out = {"per_image_sha256": sha(ROOT / run_dir / "tub_per_image.csv"), "ntu_decile_edges": edges.tolist(), "arch": {}}
    for arch in ARCHS:
        res = {}
        for s in (0, 1, 2):
            bn, inn = D[f"{arch}_bn_s{s}|psnr"], D[f"{arch}_in_s{s}|psnr"]
            res[f"s{s}"] = {"curves": curves(D, m, {"BN": bn, "IN": inn, "d": inn - bn}, scenes, W, edges)}
        y = {k: [D[f"{arch}_{k.lower()}_s{s}|psnr"] for s in (0, 1, 2)] for k in ("BN", "IN")}
        y["d"] = [y["IN"][s] - y["BN"][s] for s in (0, 1, 2)]
        res["trend"] = {k: trend(D, m, v, scenes, W) for k, v in y.items()}
        # check: the d trend must equal the frozen E-T1 result of this run (same estimator, same draws)
        frozen = json.load(open(ROOT / run_dir.replace("tub_stage5_", "tub_stage5_analysis_") / "tub_analysis.json"))["primary"]["trend_NTU"]["endpoints"][f"E-T1|{arch}"]["e_spearman"]
        res["check_d_trend_equals_frozen_E_T1"] = bool(np.allclose(res["trend"]["d"]["e_spearman"]["per_seed"], frozen["per_seed"], atol=1e-12)
                                                         and np.allclose(res["trend"]["d"]["e_spearman"]["boot_ci95_seed_mean"], frozen["boot_ci95_seed_mean"], atol=1e-12))
        out["arch"][arch] = res
    return out


def figures(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3d3d3a",
                         "xtick.color": "#5c5c57", "ytick.color": "#5c5c57", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    files = []
    for run, (rd, lab) in RUNS.items():
        r = R["runs"][run]
        # Figure A: absolute PSNR, BN vs IN, small multiples arch x seed (one y-axis per panel, shared within a row)
        fig, ax = plt.subplots(2, 3, figsize=(9.0, 5.2), sharex=True, sharey="row")
        for i, arch in enumerate(ARCHS):
            for s in range(3):
                cv = r["arch"][arch][f"s{s}"]["curves"]; x = [c["ntu_median"] for c in cv]; a = ax[i, s]
                for key, col, ls, mk in (("BN", BN_COL, "-", "o"), ("IN", IN_COL, "--", "s")):
                    yv = [c[key]["mean"] for c in cv]; lo = [c[key]["ci95"][0] for c in cv]; hi = [c[key]["ci95"][1] for c in cv]
                    a.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
                    a.plot(x, yv, ls, color=col, lw=2, marker=mk, ms=4, markeredgecolor="white", markeredgewidth=0.8, label=key)
                    a.annotate(key, (x[-1], yv[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color="#3d3d3a")
                a.set_title(f"{ARCHS[arch]}  seed {s}", fontsize=8, color="#3d3d3a")
                if s == 0:
                    a.set_ylabel("PSNR vs Stage 01 (dB)")
                if i == 1:
                    a.set_xlabel("measured turbidity, NTU (decile median)")
        ax[0, 0].legend(frameon=False, loc="upper right", fontsize=7)
        fig.suptitle(f"TUB: BN and IN PSNR vs turbidity, {run.replace('_', ' ')} ({lab}). Scene-weighted decile means, 95 % scene-bootstrap bands (models fixed)",
                     fontsize=8, color="#3d3d3a")
        fig.tight_layout(rect=(0, 0, 0.98, 0.95)); p = out / f"fig_o2_abs_psnr_{run}.png"; fig.savefig(p, dpi=200); plt.close(fig); files.append(p)
        # Figure B: gap d = IN - BN per seed; neutral ink, seeds by marker + direct label; zero line
        fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
        for i, arch in enumerate(ARCHS):
            a = ax[i]; a.axhline(0, color="#8a8a85", lw=1)
            ends = []
            for s, (mk, gr) in enumerate((("o", "#3d3d3a"), ("s", "#6b6b66"), ("^", "#9a9a94"))):
                cv = r["arch"][arch][f"s{s}"]["curves"]; x = [c["ntu_median"] for c in cv]; yv = [c["d"]["mean"] for c in cv]
                a.plot(x, yv, "-", color=gr, lw=2, marker=mk, ms=4, markeredgecolor="white", markeredgewidth=0.8, label=f"seed {s}")
                ends.append([yv[-1], s, x[-1]])
            ends.sort(); gap = 0.28  # spread end labels so they never collide (data units, dB)
            for j in range(1, len(ends)):
                ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
            for yl, s, xl in ends:
                a.annotate(f"s{s}", (xl, yl), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color="#3d3d3a", annotation_clip=False)
            a.legend(frameon=False, loc="lower right", fontsize=7)
            a.set_title(ARCHS[arch], fontsize=8, color="#3d3d3a"); a.set_xlabel("measured turbidity, NTU (decile median)")
            if i == 0:
                a.set_ylabel("IN − BN PSNR (dB)")
        fig.suptitle(f"TUB: IN − BN gap vs turbidity per seed, {run.replace('_', ' ')} ({lab})", fontsize=8, color="#3d3d3a")
        fig.tight_layout(rect=(0, 0, 0.97, 0.92)); p = out / f"fig_o2_gap_{run}.png"; fig.savefig(p, dpi=200); plt.close(fig); files.append(p)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists")
    if sha(ROOT / "v2_tools/tub_stage5_analysis.py") != ANALYSIS_SHA:
        raise SystemExit("STOP: frozen TUB analysis script hash differs")
    R = {"label": "O2 DESCRIPTIVE post hoc: BN / IN PSNR vs NTU from saved TUB outputs; decile binning post hoc; CIs conditional on trained models",
         "script_sha256": sha(__file__), "analysis_script_sha256": ANALYSIS_SHA, "runs": {}}
    for run, (rd, lab) in RUNS.items():
        R["runs"][run] = dict(analyse_run(rd), status=lab)
        print(run, {k: v["check_d_trend_equals_frozen_E_T1"] for k, v in R["runs"][run]["arch"].items()}, flush=True)
    if not all(v["check_d_trend_equals_frozen_E_T1"] for r in R["runs"].values() for v in r["arch"].values()):
        raise SystemExit("STOP: d trend does not reproduce the frozen E-T1; nothing written")
    out.mkdir(parents=True)
    json.dump(R, open(out / "o2_results.json", "x"), indent=1, default=float)
    with open(out / "o2_trends.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "status", "arch", "outcome", "estimator", "seed0", "seed1", "seed2", "seed_mean", "t95_lo", "t95_hi", "boot95_lo", "boot95_hi"])
        for run, r in R["runs"].items():
            for arch in ARCHS:
                for o, t in r["arch"][arch]["trend"].items():
                    for est, e in t.items():
                        w.writerow([run, r["status"], arch, o, est, *(f"{v:.4f}" for v in e["per_seed"]), f"{e['seed_mean']:.4f}", *(f"{v:.4f}" for v in e["t95"]),
                                    *(f"{v:.4f}" for v in e["boot_ci95_seed_mean"])])
    with open(out / "o2_decile_curves.csv", "x", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "arch", "seed", "bin", "ntu_lo", "ntu_hi", "ntu_median", "n_pairs", "n_scenes", "input_luma_sd_median",
                                       "BN_mean", "BN_ci95_lo", "BN_ci95_hi", "IN_mean", "IN_ci95_lo", "IN_ci95_hi", "d_mean", "d_ci95_lo", "d_ci95_hi"])
        for run, r in R["runs"].items():
            for arch in ARCHS:
                for s in range(3):
                    for c in r["arch"][arch][f"s{s}"]["curves"]:
                        w.writerow([run, arch, s, c["bin"], f"{c['ntu_lo']:.3f}", f"{c['ntu_hi']:.3f}", f"{c['ntu_median']:.3f}", c["n_pairs"], c["n_scenes"], f"{c['input_luma_sd_median']:.4f}",
                                    *(f"{v:.4f}" for k in ("BN", "IN", "d") for v in (c[k]["mean"], *c[k]["ci95"]))])
    files = figures(R, out)
    print("written", out, [p.name for p in files])


if __name__ == "__main__":
    main()
