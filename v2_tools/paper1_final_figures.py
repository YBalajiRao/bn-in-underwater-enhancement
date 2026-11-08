"""Paper 1 final figures (main Figs 1-6, supplementary S1-S6). Reads SAVED results only: no training, no inference, no frozen file changed.
Every plotted number is logged in plotted_values.csv with its source and a verification status:
  SOT      = equal (1e-12 relative) to the row of paper1_previz_audit/source_of_truth.csv with that id;
  RECOMP   = recomputed here from per-image files and equal to the saved analysis field (control for the derivation);
  FIELD    = read from the named frozen field (no second derivation exists).
Interval styles: solid bar = 95 % bootstrap; dashed bar = 97.5 % bootstrap (registered rho); thin grey bar = seed-level t95 (df 2).
Usage: python v2_tools/paper1_final_figures.py --out paper1_uieb_runs/paper1_final_figures"""
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
RUNS = ROOT / "paper1_uieb_runs"
SEEDS = (0, 1, 2)
ARCH = {"MARINE-Net": ("marinenet_bn", "marinenet_in", "marinenet", "o"), "AT-Trunk": ("attrunk_bn", "attrunk_in", "attrunk", "s")}
BN_C, IN_C, ID_C = "#2a78d6", "#eb6834", "#52514e"   # blue = BN, orange = IN, grey = input; nothing else uses these
UIEB_C, HICRD_C = "#1baf7a", "#6250d6"                 # dataset hues (validated pair)
COMP_C = {"A": "#2a78d6", "C": "#eb6834", "G": "#1baf7a", "R": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 300
T95 = stats.t.ppf(0.975, 2)
LOG = []
SOT = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log(fig, panel, label, value, lo=None, hi=None, interval="", source="", sot_id=None, status="FIELD"):
    if sot_id is not None:
        r = SOT.get(sot_id)
        ok = r is not None and abs(float(r["value"]) - value) <= 1e-9 * max(1, abs(value))
        if ok and lo is not None and r["lo"] not in ("", None):
            tl = 5.0001e-7 if source == "table_method_level.csv" else 1e-9 * max(1, abs(lo), abs(hi))   # that table stores CI bounds at 6 decimals
            ok = abs(float(r["lo"]) - lo) <= tl and abs(float(r["hi"]) - hi) <= tl
        status = "SOT" if ok else "SOT-MISMATCH"
    LOG.append(dict(figure=fig, panel=panel, label=label, value=value, lo="" if lo is None else lo, hi="" if hi is None else hi, interval=interval,
                    source=source, sot_id=sot_id or "", status=status))


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "savefig.dpi": DPI, "pdf.fonttype": 42, "axes.titlesize": 7.5, "axes.titleweight": "bold"})
    return plt


def save(fig, out, stem, man):
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"; fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white"); man[p.name] = sha(p)


def tub_scene_weighted(rows, metric, bn, inn, s):
    """Scene-weighted mean (pairs within scene, then over scenes) of IN - BN on the primary cohort (frozen E-T5 estimator definition)."""
    by = {}
    for r in rows:
        if r["in_primary"] in ("1", "True", "true"):
            by.setdefault(r["pair"], {})[r["model"]] = r
    sc = {}
    for pair, m in by.items():
        if f"{bn}_s{s}" in m and f"{inn}_s{s}" in m:
            sc.setdefault(m[f"{bn}_s{s}"]["scene"], []).append(float(m[f"{inn}_s{s}"][metric]) - float(m[f"{bn}_s{s}"][metric]))
    return float(np.mean([np.mean(v) for v in sc.values()])), len(by), len(sc)


def ci_bar(ax, y, lo, hi, kind, color, horizontal=True):
    ls = {"95": "-", "97.5": (0, (3, 1.5)), "t95": "-"}[kind]; lw = {"95": 1.3, "97.5": 1.3, "t95": 0.6}[kind]
    c = "#9a9994" if kind == "t95" else color
    if horizontal:
        ax.plot([lo, hi], [y, y], color=c, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)
    else:
        ax.plot([y, y], [lo, hi], color=c, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    for r in csv.DictReader(open(RUNS / "paper1_previz_audit/source_of_truth.csv", encoding="utf-8")):
        SOT[r["id"]] = r
    o1 = json.load(open(RUNS / "paper1_O1_mean_d_cis/o1_results.json"))
    s3 = {k: json.load(open(RUNS / f"stage3_analysis_{k}/results.json")) for k in ("best_valA", "last")}
    s4 = json.load(open(RUNS / "stage4_analysis_best_valA/results.json"))
    pg = json.load(open(RUNS / "post_g1_stats/post_g1_stats.json"))["best_valA"]["full90 (registered primary)"]
    e1 = json.load(open(RUNS / "s5_e1r_analysis/e1r_results.json"))
    o7 = json.load(open(RUNS / "paper1_O7_tub_input_psnr/o7_results.json"))
    tub = {k: json.load(open(RUNS / f"tub_stage5_analysis_{k}/tub_analysis.json"))["primary"]["E-T5_PRIMARY"] for k in ("best_valA", "best_valA_G455", "last", "last_G455")}
    out.mkdir(parents=True); man = {}
    plt = plt_setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Patch

    # ---------- per-image data
    man3 = json.load(open(RUNS / "stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json", encoding="utf-8"))
    names = [Path(p["input_path"]).name for p in (man3["pairs"] if isinstance(man3, dict) else man3)]
    excl = set(json.load(open(RUNS / "stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"))["primary_exclusions"])
    sel = [n for n in names if n not in excl]
    U = {k: {r["filename"]: r for r in csv.DictReader(open(RUNS / f"stage3_uieb800_{k}/per_image.csv", encoding="utf-8"))} for k in ("best_valA", "last")}
    H = {g: list(csv.DictReader(open(RUNS / f, encoding="utf-8"))) for g, f in (("square256", "stage4_hicrd2000_best_valA/per_image.csv"),
                                                                                ("aspect482", "stage4_hicrd2000_best_valA_G482/per_image.csv"))}
    T = list(csv.DictReader(open(RUNS / "tub_stage5_best_valA/tub_per_image.csv", encoding="utf-8")))
    uv = lambda run, col: np.array([float(U[run][n][col]) for n in sel])
    hv = lambda g, col: np.array([float(r[col]) for r in H[g]])

    # ================= Fig 1: framework schematic (counts read from files)
    hic_cl = o1["HICRD_2000"]["best_valA"]["n_clusters_T30"]
    tc = json.load(open(RUNS / "tub_stage5_analysis_best_valA/tub_analysis.json"))["cohort_counts"]
    tn = json.load(open(RUNS / "tub_stage5_analysis_best_valA/tub_analysis.json"))["primary"]["trend_NTU"]
    res0 = json.load(open(ROOT / "runs_v2/marinenet_bn_r3_fp32_s0/result.json"))
    for lab, v in (("UIEB primary n", 787), ("HICRD clusters", hic_cl), ("TUB primary", tc["PRIMARY"]), ("TUB NTU cohort", tc["NTU"]), ("TUB all", tc["ALL"]),
                   ("TUB NTU scenes", tn["n_scenes"]), ("TUB primary scenes", tub["best_valA"]["n_scenes"])):
        log("Fig1", "schematic", lab, v, source="frozen counts (o1_results / tub_analysis)")
    assert len(sel) == 787 and tc["PRIMARY"] == 1153
    # independent recount of the T30 and TUB counts from the raw per-pair files
    import sota_hicrd_protocol as P
    hm = json.load(open(ROOT / P.MANIFEST[0], encoding="utf-8"))["pairs"]
    n_t30 = len({p["scene_cluster_T30"] for p in hm})
    tp = {}
    for r in T:
        tp[r["pair"]] = r
    prim = [r for r in tp.values() if r["in_primary"] in ("1", "True", "true")]; ntu = [r for r in tp.values() if r["in_ntu"] in ("1", "True", "true")]
    recount = {"HICRD pairs": (len(hm), 2000), "HICRD T30 clusters": (n_t30, hic_cl), "TUB all pairs": (len(tp), tc["ALL"]), "TUB primary pairs": (len(prim), tc["PRIMARY"]),
               "TUB primary scenes": (len({r["scene"] for r in prim}), tub["best_valA"]["n_scenes"]), "TUB NTU pairs": (len(ntu), tc["NTU"]),
               "TUB NTU scenes": (len({r["scene"] for r in ntu}), tn["n_scenes"])}
    # T30 definition check: rebuild the clusters from the saved same-scene edge list (union-find over pairs with >= 30 ORB-RANSAC inliers)
    sc_dir = RUNS / "stage4_prep/hicrd_scene_clusters"
    img_ids = [r["image"] for r in csv.DictReader(open(sc_dir / "clusters.csv", encoding="utf-8"))]
    parent = {k: k for k in img_ids}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]; k = parent[k]
        return k
    for r in csv.DictReader(open(sc_dir / "edges.csv", encoding="utf-8")):
        if int(r["inliers"]) >= 30:
            parent[find(r["a"])] = find(r["b"])
    recount["HICRD T30 clusters rebuilt from edges (>= 30 inliers)"] = (len({find(k) for k in img_ids}), json.load(open(sc_dir / "summary.json"))["thresholds"]["30"]["n_clusters"])
    recount["HICRD T30 recorded summary = analysis count"] = (json.load(open(sc_dir / "summary.json"))["thresholds"]["30"]["n_clusters"], hic_cl)
    for lab, (got, rec) in recount.items():
        log("Fig1", "recount", lab, got, source="raw manifest / tub_per_image.csv vs recorded count", status="RECOMP" if got == rec else "RECOMP-MISMATCH")
        if got != rec:
            raise SystemExit(f"STOP: Fig. 1 count {lab}: recounted {got}, recorded {rec}")
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-9, 50)

    def box(x, y, w, h, title, body, fc="#f4f3f0", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 1, y + h - 1.4, title, fontsize=6.6, fontweight="bold", va="top", color=INK)
        ax.text(x + 1, y + h - 4.6, body, fontsize=5.6, va="top", color=INK2, linespacing=1.35)
    box(1, 26, 22, 21, "Training (LSUI)", f"train manifest {res0['manifest_sha256']['train'][:8]}…\n2 architectures × {{BN, IN}}\n× 3 seeds = 12 checkpoints\n100 epochs; best Val-A\n(+ last as sensitivity)")
    box(27, 36, 34, 11, "UIEB frozen-90 (gate G1)", "90 curated pairs · image bootstrap\nH1 gate: IN ≥ BN + 1 dB every seed → failed")
    box(27, 24, 34, 11, "UIEB-787 (Stage 3, confirmatory)", "787 curated pairs · image bootstrap\nH3: ρ(d, imbalance) > 0 (97.5 % CI)")
    box(27, 11, 34, 11, f"HICRD-2000 (Stage 4)", f"2,000 pairs · computed references\n{hic_cl} T30 clusters · cluster bootstrap\ngeometry 256 × 256 and 482 × 256")
    box(64, 36, 35, 11, f"TUB turbidity series (Stage 5)", f"{tc['ALL']} pairs; primary {tc['PRIMARY']} ({tub['best_valA']['n_scenes']} scenes)\nNTU cohort {tc['NTU']} ({tn['n_scenes']} scenes) · scene bootstrap")
    box(64, 24, 35, 11, "Exploratory analyses", "X1 required correction · E1r exact MSE split\nO1 per-seed CIs · O2 / O7 TUB vs NTU")
    box(64, 11, 35, 11, "SOTA comparison (HICRD)", "Group R: 6 released methods (n = 1)\nGroup T: 2 LSUI retrains (n = 1, exploratory)")
    box(1, -8, 98, 16.5, "Outputs and uncertainty", "Per seed and seed mean of IN − BN (PSNR, SSIM, E1r components).\n"
        "Intervals: 95 % bootstrap by the unit of each cohort (image / T30 cluster / scene);\n97.5 % for registered ρ; seed-level t95 (df 2) shown separately.\n"
        f"T30 cluster = HICRD scene cluster: connected component of input images linked when ORB–RANSAC homography matching\n(500 features) finds ≥ 30 inliers; "
        f"{hic_cl} clusters, the resampling unit of the HICRD bootstrap.", fc="white")
    for y in (42.5, 29.5, 16.5):
        ax.annotate("", xy=(26.6, y), xytext=(23.6, 37), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7))
    ax.annotate("", xy=(81.5, 48.6), xytext=(12, 48.6), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, connectionstyle="angle,angleA=0,angleB=90"))
    ax.text(44, 49.3, "same 12 checkpoints evaluated on every cohort", fontsize=5.4, color=INK2, ha="center")
    save(fig, out, "Fig1_framework", man); plt.close(fig)

    # ================= Fig 2: cross-dataset IN - BN, PSNR (95 % bootstrap) and SSIM (seed points + t95 only)
    sets = [("UIEB-787", "image"), ("HICRD 256", "T30 cluster"), ("HICRD 482", "T30 cluster"), ("TUB 256", "scene")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, (arch, (bn, inn, tk, mk)) in enumerate(ARCH.items()):
        for j, (sname, unit) in enumerate(sets):
            y = j * 3 + k
            if sname == "UIEB-787":
                o = o1["UIEB_787"]["best_valA"]["results"][arch]; per, sm, ci, tt = o["per_seed"], o["seed_mean"], o["seed_mean_ci95"], o["seed_level_t95_df2"]
                sid = f"UIEB.dIN-BN.{arch}.psnr.mean"
            elif sname.startswith("HICRD"):
                geo = "square256" if "256" in sname else "aspect482"; run = "best_valA" if geo == "square256" else "best_valA_G482"
                o = o1["HICRD_2000"][run]["results"][arch]["T30_unstratified_P1_draws"]; per, sm, ci, tt = o["per_seed"], o["seed_mean"], o["seed_mean_ci95"], o["seed_level_t95_df2"]
                sid = f"HICRD.{geo}.dIN-BN.{arch}.psnr.mean"
            else:
                e = tub["best_valA"]["endpoints"][f"E-T5|{tk}|d"]; per, sm, ci, tt = e["per_seed"], e["seed_mean"], e["boot_ci95_seed_mean"], e["t95"]
                sid = f"TUB.dIN-BN.{arch}.psnr.mean"
            ax = axs[0]
            ci_bar(ax, y - 0.18, *tt, "t95", INK); ci_bar(ax, y, *ci, "95", INK)
            ax.scatter(per, [y] * 3, s=9, facecolor="white", edgecolor=INK, marker=mk, lw=0.7, zorder=3)
            ax.scatter([sm], [y], s=24, color=INK, marker=mk, zorder=4)
            log("Fig2", "a PSNR", f"{sname} {arch} seed mean", sm, *ci, f"95 % {unit} bootstrap", "O1 / tub_analysis", sid)
            for s in SEEDS:
                log("Fig2", "a PSNR", f"{sname} {arch} s{s}", per[s], source="O1 / tub_analysis", sot_id=sid.replace(".mean", f".s{s}"))
            # SSIM differences (no bootstrap exists): per-seed means recomputed from per-image files
            if sname == "UIEB-787":
                ds = [float((uv("best_valA", f"{inn}_s{s}_ssim") - uv("best_valA", f"{bn}_s{s}_ssim")).mean()) for s in SEEDS]; src = "stage3 per_image.csv (787)"
                sidS = f"UIEB.dIN-BN.{arch}.ssim.mean"
            elif sname.startswith("HICRD"):
                ds = [float((hv(geo, f"{inn}_s{s}_ssim") - hv(geo, f"{bn}_s{s}_ssim")).mean()) for s in SEEDS]; src = f"stage4 per_image.csv {geo}"; sidS = None
            else:
                ds = [tub_scene_weighted(T, "ssim", bn, inn, s)[0] for s in SEEDS]; src = "tub_per_image.csv, scene-weighted (E-T5 estimator)"; sidS = None
                pc = [tub_scene_weighted(T, "psnr", bn, inn, s)[0] for s in SEEDS]
                ok = np.allclose(pc, tub["best_valA"]["endpoints"][f"E-T5|{tk}|d"]["per_seed"], atol=1e-9)
                log("Fig2", "control", f"TUB {arch} scene-weighted PSNR d reproduces registered E-T5", float(np.mean(pc)), source="control", status="RECOMP" if ok else "RECOMP-MISMATCH")
            sm2 = float(np.mean(ds)); sd2 = float(np.std(ds, ddof=1)); t2 = [sm2 - T95 * sd2 / np.sqrt(3), sm2 + T95 * sd2 / np.sqrt(3)]
            ax = axs[1]
            ci_bar(ax, y, *t2, "t95", INK)
            ax.scatter(ds, [y] * 3, s=9, facecolor="white", edgecolor=INK, marker=mk, lw=0.7, zorder=3); ax.scatter([sm2], [y], s=24, color=INK, marker=mk, zorder=4)
            log("Fig2", "b SSIM", f"{sname} {arch} seed mean", sm2, *t2, "seed t95 (no bootstrap available)", src, sidS, status="COMPUTED")
    for ax, lab in ((axs[0], "IN − BN PSNR (dB)"), (axs[1], "IN − BN SSIM")):
        ax.axvline(0, color=INK, lw=0.6); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        ax.set_yticks([j * 3 + 0.5 for j in range(len(sets))]); ax.set_yticklabels([s for s, _ in sets]); ax.invert_yaxis()
    axs[0].set_title("a  PSNR: seed mean, 95 % bootstrap CI", loc="left"); axs[1].set_title("b  SSIM: seed mean, seed-level t95 only", loc="left")
    h = [Line2D([], [], color=INK, marker="o", ls="", label="MARINE-Net (seed mean; ○ seeds)"), Line2D([], [], color=INK, marker="s", ls="", label="AT-Trunk"),
         Line2D([], [], color=INK, lw=1.3, label="95 % bootstrap (image / cluster / scene)"), Line2D([], [], color="#9a9994", lw=0.6, label="seed-level t95 (df 2)")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07)); fig.tight_layout()
    save(fig, out, "Fig2_cross_dataset_IN_minus_BN", man); plt.close(fig)

    # ================= Fig 3: per-seed effects (a) per-seed d with 95 % CIs; (b) rho forest 97.5 %; (c) d vs imbalance; (d) collinearity
    fig = plt.figure(figsize=(7.4, 6.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.38, wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    rows3 = [("UIEB frozen-90", "best"), ("UIEB-787", "best"), ("UIEB-787", "last"), ("HICRD 256", "best"), ("HICRD 256", "last"), ("HICRD 482", "best"),
             ("TUB 256", "best"), ("TUB 256", "last"), ("TUB 455", "best"), ("TUB 455", "last")]
    yt = []
    for i, (sname, ck) in enumerate(rows3):
        for k, (arch, (bn, inn, tk, mk)) in enumerate(ARCH.items()):
            if sname == "UIEB frozen-90":
                hh = pg["H1_marinenet_in_minus_bn" if arch == "MARINE-Net" else "H1_attrunk_in_minus_bn"]; per, cis = hh["per_seed_mean"], hh["per_seed_boot95_images"]; src = "post_g1_stats full90"
            elif sname == "UIEB-787":
                o = o1["UIEB_787"]["best_valA" if ck == "best" else "last"]["results"][arch]; per, cis = o["per_seed"], o["per_seed_ci95"]; src = "O1"
            elif sname.startswith("HICRD"):
                run = {"HICRD 256": {"best": "best_valA", "last": "last"}, "HICRD 482": {"best": "best_valA_G482"}}[sname][ck]
                o = o1["HICRD_2000"][run]["results"][arch]["T30_unstratified_P1_draws"]; per, cis = o["per_seed"], o["per_seed_ci95"]; src = "O1"
            else:
                key = ("best_valA" if ck == "best" else "last") + ("_G455" if "455" in sname else "")
                e = tub[key]["endpoints"][f"E-T5|{tk}|d"]; per, cis = e["per_seed"], e["boot_ci95_per_seed"]; src = f"tub_analysis {key}"
            for s in SEEDS:
                y = i * 8 + k * 4 + s
                c = INK if arch == "MARINE-Net" else INK2
                ci_bar(ax, y, *cis[s], "95", c)
                ax.scatter([per[s]], [y], s=9, facecolor=c if arch == "MARINE-Net" else "white", edgecolor=c, marker=mk, lw=0.7, zorder=3)
                sid = None
                if sname == "UIEB-787" and ck == "best":
                    sid = f"UIEB.dIN-BN.{arch}.psnr.s{s}"
                elif sname == "HICRD 256" and ck == "best":
                    sid = f"HICRD.square256.dIN-BN.{arch}.psnr.s{s}"
                elif sname == "HICRD 482":
                    sid = f"HICRD.aspect482.dIN-BN.{arch}.psnr.s{s}"
                elif sname == "TUB 256" and ck == "best":
                    sid = f"TUB.dIN-BN.{arch}.psnr.s{s}"
                log("Fig3", "a", f"{sname} {ck} {arch} s{s}", per[s], *cis[s], "95 % bootstrap per seed", src, sid)
        yt.append((i * 8 + 3.5, f"{sname} ({ck})"))
    ax.axvline(0, color=INK, lw=0.6); ax.set_yticks([y for y, _ in yt]); ax.set_yticklabels([l for _, l in yt], fontsize=6); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("IN − BN PSNR per seed (dB), 95 % bootstrap CI"); ax.set_title("a  Per-seed effect by cohort, checkpoint and geometry", loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, marker="o", label="MARINE-Net s0 / s1 / s2"), Line2D([], [], color=INK2, marker="s", markerfacecolor="white", label="AT-Trunk s0 / s1 / s2")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=5.5)
    ax = fig.add_subplot(gs[0, 1])
    rho_rows = []
    for arch in ARCH:
        r = s3["best_valA"]["cohorts"]["PRIMARY_787"][arch]; rho_rows.append(("UIEB-787", arch, r["rho_per_seed"], r["rho_bar"], r["rho_bar_ci97_5"], "image", f"UIEB.rho.{arch}.mean"))
    for arch in ARCH:
        r = s4["P1_pooled_ALL"]["None"][arch]; rho_rows.append(("HICRD D1", arch, r["seed_level"]["per_seed"], r["seed_level"]["mean"], r["ci97_5"], "cluster", None))
    for arch in ARCH:
        r = s4["P2_site_adjusted_KNOWN"]["site"][arch]; rho_rows.append(("HICRD D2 (site-adj.)", arch, r["seed_level"]["per_seed"], r["seed_level"]["mean"], r["ci97_5"], "cluster", None))
    for i, (sn, arch, per, rb, ci, unit, sid) in enumerate(rho_rows):
        mk = ARCH[arch][3]; c = INK
        ci_bar(ax, i, *ci, "97.5", c); ax.scatter(per, [i] * 3, s=9, facecolor="white", edgecolor=c, marker=mk, lw=0.7, zorder=3)
        ax.scatter([rb], [i], s=24, color=c, marker=mk, zorder=4)
        log("Fig3", "b", f"{sn} {arch} rho_bar", rb, *ci, f"97.5 % {unit} bootstrap (registered)", "stage3/stage4 results.json", sid)
    ax.axvline(0, color=INK, lw=0.6); ax.set_yticks(range(len(rho_rows))); ax.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rho_rows], fontsize=5.8); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("Spearman ρ(d, channel-mean imbalance)"); ax.set_title("b  Registered ρ̄, 97.5 % CI (dashed)", loc="left")
    # (c) d vs imbalance, seed-mean d (U-4 default: seed mean, stated)
    ax = fig.add_subplot(gs[1, 0])
    du = {r["filename"]: r for r in csv.DictReader(open(RUNS / "stage3_analysis_best_valA/descriptors_per_image.csv"))}
    dh = {r["id"]: r for r in csv.DictReader(open(RUNS / "stage4_analysis_best_valA/descriptors_per_image.csv"))}
    xu = np.array([float(du[n]["imbalance"]) for n in sel]); xh = np.array([float(dh[r["id"]]["imbalance"]) for r in H["square256"]])
    dU = np.mean([uv("best_valA", f"marinenet_in_s{s}_psnr") - uv("best_valA", f"marinenet_bn_s{s}_psnr") for s in SEEDS], 0)
    dH = np.mean([hv("square256", f"marinenet_in_s{s}_psnr") - hv("square256", f"marinenet_bn_s{s}_psnr") for s in SEEDS], 0)
    ax.scatter(xh, dH, s=2, color=HICRD_C, alpha=0.35, lw=0, label=f"HICRD-2000 (n = {len(xh)})"); ax.scatter(xu, dU, s=2.5, color=UIEB_C, alpha=0.6, lw=0, label=f"UIEB-787 (n = {len(xu)})")
    ax.axhline(0, color=INK, lw=0.6); ax.set_xlabel("input channel-mean imbalance"); ax.set_ylabel("MARINE-Net IN − BN PSNR (dB), seed mean")
    ax.set_title("c  Per-image d vs imbalance (MARINE-Net)", loc="left"); ax.legend(frameon=False, fontsize=5.5, markerscale=4, loc="lower right")
    log("Fig3", "c", "UIEB mean of plotted d", float(dU.mean()), source="recomputed per-image", sot_id="UIEB.dIN-BN.MARINE-Net.psnr.mean")
    log("Fig3", "c", "HICRD mean of plotted d", float(dH.mean()), source="recomputed per-image", sot_id="HICRD.square256.dIN-BN.MARINE-Net.psnr.mean")
    # (d) collinearity
    ax = fig.add_subplot(gs[1, 1])
    lu = np.array([float(du[n]["luma_mean"]) for n in sel]); lh = np.array([float(dh[r["id"]]["luma_mean"]) for r in H["square256"]])
    ru, rh = stats.spearmanr(xu, lu)[0], stats.spearmanr(xh, lh)[0]
    okr = abs(ru - s3["best_valA"]["cohorts"]["PRIMARY_787"]["descriptor_associations"]["rho_x_luma_mean"]) < 1e-9 and abs(rh - s4["SUPP"]["rho_x_b"]) < 1e-9
    log("Fig3", "d", "UIEB ρ(imbalance, luma)", float(ru), source="recomputed vs results.json", status="RECOMP" if okr else "RECOMP-MISMATCH")
    log("Fig3", "d", "HICRD ρ(imbalance, luma)", float(rh), source="recomputed vs stage4 SUPP.rho_x_b", status="RECOMP" if okr else "RECOMP-MISMATCH")
    ax.scatter(xh, lh, s=2, color=HICRD_C, alpha=0.35, lw=0, label=f"HICRD ρ = {rh:.3f}"); ax.scatter(xu, lu, s=2.5, color=UIEB_C, alpha=0.6, lw=0, label=f"UIEB ρ = {ru:.3f}")
    ax.set_xlabel("input channel-mean imbalance"); ax.set_ylabel("input mean luma"); ax.set_title("d  Imbalance vs brightness (Spearman ρ)", loc="left")
    ax.legend(frameon=False, fontsize=5.5, markerscale=4)
    save(fig, out, "Fig3_per_seed_effects_and_colour_cast", man); plt.close(fig)

    # ================= Fig 4: E1r components
    ep = list(csv.DictReader(open(RUNS / "s5_e1r/e1r_per_image.csv", encoding="utf-8")))
    epi = {}
    for r in ep:
        epi.setdefault((r["dataset"], r["model"]), {})[r["key"]] = r
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    cols = [("UIEB", "MARINE-Net"), ("UIEB", "AT-Trunk"), ("HICRD", "MARINE-Net"), ("HICRD", "AT-Trunk"), ("TUB", "MARINE-Net"), ("TUB", "AT-Trunk")]
    for ci_, comp in enumerate("ACGR"):
        ax = axs[ci_]
        for j, (ds, arch) in enumerate(cols):
            bn, inn, tk, mk = ARCH[arch]
            if ds in ("UIEB", "HICRD"):
                r = e1[ds][arch][comp]; per, sm, ci = r["per_seed"], r["mean"], r["ci95_seed_mean"]; unit = "image" if ds == "UIEB" else "T30 cluster"
                keys = sorted(epi[(ds, f"{bn}_s0")]); rec = [np.mean([float(epi[(ds, f"{inn}_s{s}")][k][comp]) - float(epi[(ds, f"{bn}_s{s}")][k][comp]) for k in keys]) for s in SEEDS]
                st = "RECOMP" if np.allclose(rec, per, rtol=1e-9, atol=1e-15) else "RECOMP-MISMATCH"
            else:
                r = tub["best_valA"]["endpoints"][f"E-T5|{tk}|d{comp}"]; per, sm, ci = r["per_seed"], r["seed_mean"], r["boot_ci95_seed_mean"]; unit = "scene"
                rec = [tub_scene_weighted(T, comp, bn, inn, s)[0] for s in SEEDS]
                st = "RECOMP" if np.allclose(rec, per, rtol=1e-9, atol=1e-15) else "RECOMP-MISMATCH"
            c = INK
            ci_bar(ax, j, ci[0] * 1e3, ci[1] * 1e3, "95", c)
            ax.scatter([x * 1e3 for x in per], [j] * 3, s=8, facecolor="white", edgecolor=c, marker=mk, lw=0.7, zorder=3)
            ax.scatter([sm * 1e3], [j], s=22, color=c, marker=mk, zorder=4)
            log("Fig4", comp, f"{ds} {arch} Δ{comp} ×1e3", sm * 1e3, ci[0] * 1e3, ci[1] * 1e3, f"95 % {unit} bootstrap", "e1r_results / tub E-T5", status=st)
        ax.axvline(0, color=INK, lw=0.6); ax.grid(axis="y", visible=False)
        ax.set_title({"A": "ΔA brightness", "C": "ΔC colour", "G": "ΔG gain", "R": "ΔR residual"}[comp], loc="left"); ax.set_xlabel("IN − BN (×10⁻³ MSE)")
    axs[0].set_yticks(range(len(cols))); axs[0].set_yticklabels([f"{d} · {a}" for d, a in cols], fontsize=6); axs[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", ls="", label="MARINE-Net (seed mean; ○ seeds)"), Line2D([], [], color=INK, marker="s", ls="", label="AT-Trunk"),
                        Line2D([], [], color=INK, lw=1.3, label="95 % bootstrap (image / cluster / scene)")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, out, "Fig4_E1r_error_decomposition", man); plt.close(fig)

    # ================= Fig 5: TUB input / BN / IN vs NTU (O7 decile curves, square256, seed mean)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (arch, (bn, inn, tk, mk)) in zip(axs, ARCH.items()):
        cur = o7["decile_curves_square256_seedmean"][tk]; x = [b["ntu_median"] for b in cur]
        for key, c, lab in (("identity", ID_C, "input (do nothing)"), ("BN", BN_C, "BN"), ("IN", IN_C, "IN")):
            m = [b[key]["mean"] for b in cur]; lo = [b[key]["ci95"][0] for b in cur]; hi = [b[key]["ci95"][1] for b in cur]
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0); ax.plot(x, m, color=c, lw=2, marker="o", ms=3.5, label=lab)
            for b in cur:
                log("Fig5", arch, f"{key} decile {b['bin']} (NTU {b['ntu_median']})", b[key]["mean"], *b[key]["ci95"], "95 % scene bootstrap (O7, post hoc)", "o7_results decile_curves")
        ax.set_xlabel("turbidity, decile median (NTU)"); ax.set_title(f"{'ab'[list(ARCH).index(arch)]}  {arch}", loc="left")
    axs[0].set_ylabel("PSNR vs Stage 01 reference (dB)"); axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, out, "Fig5_TUB_PSNR_vs_NTU", man); plt.close(fig)

    # ================= Fig 6: qualitative, HICRD only (saved outputs); UIEB / TUB outputs were never saved
    import sota_hicrd_protocol as P
    import train_v2 as tv
    import stage4_infer as S4I
    vis = P.visual_selection()[:6]                                    # the 6 site-median images, protocol order (no choice made here)
    pairs = S4I.load_manifest(); ids = [p["id"] for p in pairs]; selp = [pairs[ids.index(k)] for k in vis]
    X, Y = tv.preload(selp); x = X.float().numpy() / 255.0; yref = Y.float().numpy() / 255.0
    arms = ["marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in"]
    imgs = {a_: np.load(RUNS / f"sota_hicrd_pub_figures_v2/ours_visual_{a_}_s0.npy")[:6] for a_ in arms}
    vchk = json.load(open(RUNS / "sota_hicrd_pub_figures_v2/visual_checks.json"))["annotations"]
    hrow = {r["id"]: r for r in H["square256"]}
    names6 = {"marinenet_bn": "MARINE-Net BN", "marinenet_in": "MARINE-Net IN", "attrunk_bn": "AT-Trunk BN", "attrunk_in": "AT-Trunk IN"}
    fig, ax = plt.subplots(6, 6, figsize=(7.2, 7.5))
    for i, k in enumerate(vis):
        for j, (lab, im) in enumerate([("Input", x[i]), ("Reference (computed)", yref[i])] + [(names6[a_], imgs[a_][i]) for a_ in arms]):
            aa = ax[i, j]; aa.imshow(np.clip(im.transpose(1, 2, 0), 0, 1)); aa.set_xticks([]); aa.set_yticks([]); aa.grid(False)
            for sp in aa.spines.values():
                sp.set_visible(False)
            if i == 0:
                aa.set_title(lab, fontsize=6, fontweight="normal")
            if j >= 2:
                a_ = arms[j - 2]; p_, s_ = float(hrow[k][f"{a_}_s0_psnr"]), float(hrow[k][f"{a_}_s0_ssim"])
                vp, vs = vchk[f"{a_}|{k}"]
                st = "RECOMP" if abs(vp - p_) <= 1e-4 and abs(vs - s_) <= 1e-5 else "RECOMP-MISMATCH"
                log("Fig6", k, f"{a_} s0 PSNR", p_, source="stage4 per_image.csv; recomputed from saved output in visual_checks.json", status=st)
                aa.set_xlabel(f"{p_:.2f} dB / {s_:.3f}", fontsize=5.5, color=INK2, labelpad=1.5)
            elif j == 0:
                aa.set_xlabel(k, fontsize=5.5, color=INK2, labelpad=1.5)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.2); save(fig, out, "Fig6_qualitative_HICRD", man); plt.close(fig)

    # ================= Supplementary S1-S6: verified copies of the HICRD SOTA v2 figures (generated 2026-11-03 from saved outputs)
    v2 = RUNS / "sota_hicrd_pub_figures_v2"; fm = json.load(open(v2 / "figure_manifest.json"))
    smap = {"S1_HICRD_SOTA_overall": "fig1_overall_psnr_ssim_square256", "S2_HICRD_SOTA_contrasts": "fig2_registered_contrasts_square256_ALL",
            "S3_HICRD_SOTA_E1r_composition": "fig3_error_decomposition_square256", "S4_HICRD_SOTA_geometry": "fig4_geometry_sensitivity",
            "S6a_HICRD_SOTA_visual_groupT": "fig6a_visual_groupT_images1-6", "S6b_HICRD_SOTA_visual_groupT": "fig6b_visual_groupT_images7-12"}
    # S5a-c are redrawn (title wording "released methods") from the SAVED arrays with the original panel code; crop boxes and annotations
    # must be identical to the recorded ones (visual_checks.json), otherwise STOP. Crops use nearest-neighbour display (source pixels unaltered).
    import sota_hicrd_pub_figures as SP
    vc = json.load(open(v2 / "visual_checks.json"))
    vis12 = P.visual_selection(); sel12 = [pairs[ids.index(k)] for k in vis12]
    X12, Y12 = tv.preload(sel12); x12 = X12.float() / 255.0; y12 = Y12.float() / 255.0
    boxes = [tuple(SP.crop_box(y12[i].numpy())) for i in range(12)]
    if [list(b) for b in boxes] != [list(vc["crop_boxes_yx"][k]) for k in vis12]:
        raise SystemExit("STOP: recomputed crop boxes differ from the recorded ones")
    log("S5", "crops", "crop boxes identical to visual_checks.json", 12.0, source="sota_hicrd_pub_figures.crop_box", status="RECOMP")
    simgs = {a_: np.load(v2 / f"ours_visual_{a_}_s0.npy") for a_ in SP.ARMS}
    simgs.update({m: np.load(RUNS / "sota_hicrd_released_square256" / f"visual_{m}.npy") for m in SP.R_ORDER})
    import torch
    ann = {}
    for m in SP.ARMS + SP.R_ORDER:
        for i, k in enumerate(vis12):
            t_ = torch.from_numpy(np.ascontiguousarray(simgs[m][i], dtype=np.float32))[None]
            ps, ss = float(tv.compute_psnr(t_, y12[i:i + 1])), float(tv.compute_ssim(t_, y12[i:i + 1]))
            rp_, rs_ = vc["annotations"][f"{m}|{k}"]
            if abs(ps - rp_) > 1e-9 or abs(ss - rs_) > 1e-9:
                raise SystemExit(f"STOP: S5 annotation {m} {k} differs from the recorded value")
            ann[(m, i)] = (ps, ss)
            log("S5", k, f"{m} PSNR", ps, source="recomputed from saved array = visual_checks.json", status="RECOMP")
    from matplotlib.patches import Rectangle
    SP.plt_setup()

    def s5_panel(idx, stem):
        cols = SP.ARMS + SP.R_ORDER; allc = ["input", "reference"] + cols
        fig, ax3 = plt.subplots(3 * len(idx), len(allc), figsize=(0.78 * len(allc), 0.86 * 2.3 * len(idx)), squeeze=False,
                                gridspec_kw={"height_ratios": [1, 1, 0.3] * len(idx), "hspace": 0.12, "wspace": 0.04})
        for sp_ax in ax3[2::3].ravel():
            sp_ax.axis("off")
        ax = np.concatenate([ax3[k:k + 2] for k in range(0, 3 * len(idx), 3)])
        for r, i in enumerate(idx):
            by, bx = boxes[i]
            for j, c in enumerate(allc):
                im = x12[i].numpy() if c == "input" else y12[i].numpy() if c == "reference" else simgs[c][i]
                a1, a2 = ax[2 * r, j], ax[2 * r + 1, j]
                a1.imshow(np.clip(im.transpose(1, 2, 0), 0, 1)); a1.add_patch(Rectangle((bx, by), 64, 64, fill=False, ec="#e34948", lw=0.6))
                a2.imshow(np.clip(im[:, by:by + 64, bx:bx + 64].transpose(1, 2, 0), 0, 1), interpolation="nearest")
                for aa in (a1, a2):
                    aa.set_xticks([]); aa.set_yticks([]); aa.grid(False)
                    for sp in aa.spines.values():
                        sp.set_visible(False)
                if r == 0:
                    a1.set_title(SP.NAME.get(c, c.capitalize()), fontsize=5.5, color=SP.INK, fontweight="normal")   # as in the original S5
                a2.set_xlabel(vis12[i] if c == "input" else "" if c == "reference" else f"{ann[(c, i)][0]:.2f} dB / {ann[(c, i)][1]:.3f}",
                              fontsize=5, color=SP.INK2, labelpad=1.5)
        fig.suptitle("Ours (seed 0) and Group R released methods (descriptive, n = 1). Below each output: PSNR (dB) / SSIM vs the computed HICRD reference.",
                     fontsize=6.5, color=SP.INK, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)
        SP.save(fig, out, stem, man); plt.close(fig)
    for p_, idx in enumerate((range(0, 4), range(4, 8), range(8, 12))):
        s5_panel(list(idx), f"S5{'abc'[p_]}_HICRD_SOTA_visual_groupR")
    plt_setup()
    for new, old in smap.items():
        for ext in ("png", "pdf"):
            src = v2 / f"{old}.{ext}"
            if sha(src) != fm[f"{old}.{ext}"]:
                raise SystemExit(f"STOP: {src.name} differs from its recorded hash")
            shutil.copy2(src, out / f"{new}.{ext}"); man[f"{new}.{ext}"] = sha(out / f"{new}.{ext}")
    # verify the S1 / S2 plotted data against the source of truth
    for r in csv.DictReader(open(v2 / "table_method_level.csv", encoding="utf-8")):
        if r["group"] in ("R", "T") and r["geometry"] == "square256":
            for met in ("psnr", "ssim"):
                lo, hi = (float(x) for x in r[f"ci95_{met}"].split("|"))
                log("S1", met, r["method"], float(r[f"mean_{met}"]), lo, hi, "95 % cluster bootstrap (registered)", "table_method_level.csv", f"HICRD.SOTA.square256.{r['method']}.{met}")
        if r["group"] == "ours" and r["seed"] == "mean":
            for met in ("psnr", "ssim"):
                lo, hi = (float(x) for x in r[f"ci95_{met}"].split("|"))
                log("S1", met, f"{r['method']} {r['geometry']} seed mean", float(r[f"mean_{met}"]), lo, hi, "95 % cluster bootstrap (POST HOC, not registered)", "table_method_level.csv")
    an = json.load(open(RUNS / "sota_hicrd_analysis/analysis.json"))
    for r in an["contrasts"]:
        if r["geometry"] == "square256" and r["population"] == "ALL":
            log("S2", r["metric"], f"{r['arm']} − {r['comparator']}", r["seed_mean"], *r["ci95_seed_mean"], "95 % cluster bootstrap (registered)", "analysis.json",
                f"HICRD.SOTA.square256.{r['arm']}-{r['comparator']}.{r['metric']}")

    # ---------- write logs
    with open(out / "plotted_values.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LOG[0].keys())); w.writeheader()
        for r in LOG:
            w.writerow({k: (f"{v:.12g}" if isinstance(v, float) else v) for k, v in r.items()})
    shutil.copy2(__file__, out / "paper1_final_figures.py")
    man["paper1_final_figures.py"] = sha(out / "paper1_final_figures.py"); man["plotted_values.csv"] = sha(out / "plotted_values.csv")
    json.dump(man, open(out / "figure_manifest.json", "x"), indent=1)
    cnt = {}
    for r in LOG:
        cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    print(json.dumps({"figures": sorted(k for k in man if k.endswith(".png")), "status_counts": cnt,
                      "mismatches": [r for r in LOG if "MISMATCH" in r["status"]][:10]}, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
