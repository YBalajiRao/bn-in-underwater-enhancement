"""Publication figures and tables for the SOTA HICRD-2000 comparison (after the frozen analysis; Group R descriptive, Group T EXPLORATORY).
Reads ONLY saved outputs: the five run folders (per-image CSVs + visual_<m>.npy), the frozen analysis output, our frozen Stage 4 per-image CSVs and
s5_e1r rows. The one computation on images: OUR four arms (seed 0, best_valA, frozen evaluator loader, square256, fp32, TF32 off) on the 12 registered
visual images, saved to ours_visual_<arm>_s0.npy; their PSNR / SSIM must equal the frozen Stage 4 per-image values (STOP otherwise), and every
comparator visual array must reproduce its own per-image CSV PSNR / SSIM (STOP otherwise). No image is swapped; the crop rule is fixed below.
Crop rule (fixed, method-independent): per image, the 64 x 64 window (stride 8) with the largest Sobel gradient energy of the REFERENCE grayscale.
Usage: python v2_tools/sota_hicrd_pub_figures.py --analysis paper1_uieb_runs/sota_hicrd_analysis --post-check DIR --out paper1_uieb_runs/sota_hicrd_pub_figures"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402
import sota_hicrd_analysis as AN  # noqa: E402

RUNS = ROOT / "paper1_uieb_runs"
FOLD = {("R", "square256"): "sota_hicrd_released_square256", ("R", "aspect482"): "sota_hicrd_released_G482",
        ("T", "square256"): "sota_hicrd_retrained_square256", ("T", "aspect482"): "sota_hicrd_retrained_G482"}
R_ORDER = ["ushape", "wwe_uie", "funie_gan", "nu2net", "clahe", "udcp"]
T_ORDER = ["waternet_lsui_s0", "wwe_uie_lsui_s0"]
ARMS = list(P.OUR_ARMS)
NAME = {"marinenet_bn": "MARINE-Net BN", "marinenet_in": "MARINE-Net IN", "attrunk_bn": "AT-Trunk BN", "attrunk_in": "AT-Trunk IN",
        "ushape": "U-Shape", "wwe_uie": "WWE-UIE", "funie_gan": "FUnIE-GAN", "nu2net": "NU²Net", "clahe": "CLAHE", "udcp": "UDCP-GF",
        "waternet_lsui_s0": "WaterNet-LSUI†", "wwe_uie_lsui_s0": "WWE-UIE-LSUI†", "ushape_minmax": "U-Shape (min–max)"}
BN_C, IN_C = "#2a78d6", "#eb6834"                      # validated categorical slots 1, 2 (BN / IN); arm family by marker
COMP_C = {"A": "#2a78d6", "C": "#eb6834", "G": "#1baf7a", "R": "#eda100"}   # slots 1-4, adjacent pairs validated
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 300


def arm_style(a):
    return dict(color=BN_C if a.endswith("_bn") else IN_C, marker="o" if a.startswith("marinenet") else "s")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": DPI, "pdf.fonttype": 42})
    return plt


def save(fig, out, stem, manifest):
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
        manifest[p.name] = K.sha(p)


def verify_inputs(post_check, analysis):
    K.verify_protocol_files()
    pc = json.load(open(post_check / "post_exec_check.json"))
    if pc["stage"] != "post_T482" or not pc["overall_pass"]:
        raise SystemExit("STOP: post-execution check (post_T482) did not pass")
    for f, r in pc["folders"].items():
        for m, v in r["files"].items():
            if K.sha(RUNS / f / f"per_image_{m}.csv") != v["sha256"]:
                raise SystemExit(f"STOP: {f}/{m} changed since the post-execution check")
    meta = json.load(open(analysis / "analysis.json"))["meta"]
    for k, h in meta["input_hashes"].items():
        name, folder = k.split("@")
        if K.sha(RUNS / folder / name) != h:
            raise SystemExit(f"STOP: analysis input {k} changed")
    return pc


def load_metrics(ids):
    comps = {}
    for (g, geo), f in FOLD.items():
        for m in (R_ORDER if g == "R" else T_ORDER):
            p = RUNS / f / f"per_image_{m}.csv"
            if p.exists():
                rows = K.read_per_image(p); assert [r["id"] for r in rows] == ids
                comps[(m, geo)] = {c: np.array([float(r[c]) for r in rows]) for c in ("psnr", "ssim", "MSE", "A", "C", "G", "R")}
    mm = K.read_per_image(RUNS / "sota_hicrd_released_square256_ushape_minmax" / "per_image_ushape_minmax.csv")
    comps[("ushape_minmax", "square256")] = {c: np.array([float(r[c]) for r in mm]) for c in ("psnr", "ssim")}
    ours = {geo: AN.load_ours(geo) for geo in ("square256", "aspect482")}
    oe = AN.load_ours_e1r(ids)
    return comps, ours, oe


def tables(comps, ours, oe, ids, out, manifest):
    rows, W = [], {}
    for geo in ("square256", "aspect482"):
        rr, draws = P.stage4_population_draws("ALL"); W[geo] = AN.count_matrix(draws, len(rr))
    for geo in ("square256", "aspect482"):
        Wm, lens = W[geo]
        for a in ARMS:
            for s in P.SEEDS:
                e = {c: float(oe[f"{a}_s{s}|{c}"].mean()) for c in ("MSE", "A", "C", "G", "R")} if geo == "square256" else {}
                rows.append(dict(group="ours", method=a, seed=s, geometry=geo, mean_psnr=float(ours[geo][f"{a}_s{s}|psnr"].mean()),
                                 mean_ssim=float(ours[geo][f"{a}_s{s}|ssim"].mean()), **{f"mean_{c}": v for c, v in e.items()}))
            sm = {m: np.mean([ours[geo][f"{a}_s{s}|{m}"] for s in P.SEEDS], 0) for m in ("psnr", "ssim")}
            rows.append(dict(group="ours", method=a, seed="mean", geometry=geo, mean_psnr=float(sm["psnr"].mean()), mean_ssim=float(sm["ssim"].mean()),
                             ci95_psnr=AN.pct(AN.boot_means(Wm, lens, sm["psnr"])), ci95_ssim=AN.pct(AN.boot_means(Wm, lens, sm["ssim"])),
                             **({f"mean_{c}": float(np.mean([oe[f"{a}_s{s}|{c}"] for s in P.SEEDS], 0).mean()) for c in ("MSE", "A", "C", "G", "R")}
                                if geo == "square256" else {})))
        for (m, g2), v in comps.items():
            if g2 != geo:
                continue
            grp = "T" if m in T_ORDER else ("R-sensitivity" if m == "ushape_minmax" else "R")
            rows.append(dict(group=grp, method=m, seed="n=1", geometry=geo, mean_psnr=float(v["psnr"].mean()), mean_ssim=float(v["ssim"].mean()),
                             ci95_psnr=AN.pct(AN.boot_means(Wm, lens, v["psnr"])), ci95_ssim=AN.pct(AN.boot_means(Wm, lens, v["ssim"])),
                             **({f"mean_{c}": float(v[c].mean()) for c in ("MSE", "A", "C", "G", "R")} if "MSE" in v else {})))
    cols = ["group", "method", "seed", "geometry", "mean_psnr", "ci95_psnr", "mean_ssim", "ci95_ssim", "mean_MSE", "mean_A", "mean_C", "mean_G", "mean_R"]
    with open(out / "table_method_level.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k][0]:.6f}|{r[k][1]:.6f}" if isinstance(r.get(k), list) else r.get(k, "")) for k in cols})
    manifest["table_method_level.csv"] = K.sha(out / "table_method_level.csv")
    return rows


def fig_overall(rows, plt, out, manifest):
    sq = [r for r in rows if r["geometry"] == "square256"]
    order = [("ours", a) for a in ARMS] + [("R", m) for m in R_ORDER] + [("R-sensitivity", "ushape_minmax")] + [("T", m) for m in T_ORDER]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    for ax, met, lab in ((axs[0], "psnr", "Mean PSNR (dB)"), (axs[1], "ssim", "Mean SSIM")):
        for y, (g, m) in enumerate(order):
            if g == "ours":
                sd = [r for r in sq if r["method"] == m and r["seed"] != "mean"]; mean = [r for r in sq if r["method"] == m and r["seed"] == "mean"][0]
                st = arm_style(m)
                ax.plot(mean[f"ci95_{met}"], [y, y], color=st["color"], lw=1.2, solid_capstyle="round")
                ax.scatter([r[f"mean_{met}"] for r in sd], [y] * 3, s=10, facecolor="white", edgecolor=st["color"], marker=st["marker"], lw=0.8, zorder=3)
                ax.scatter([mean[f"mean_{met}"]], [y], s=28, color=st["color"], marker=st["marker"], edgecolor="white", lw=0.8, zorder=4)
            else:
                r = [x for x in sq if x["method"] == m][0]
                ax.plot(r[f"ci95_{met}"], [y, y], color=INK2, lw=1.2)
                ax.scatter([r[f"mean_{met}"]], [y], s=26, marker="d" if g == "R-sensitivity" else "D", facecolor="white" if g == "T" else INK2, edgecolor=INK2, lw=0.9, zorder=4)
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        for yb in (3.5, 10.5):
            ax.axhline(yb, color=GRID, lw=0.8)
    axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([NAME[m] for _, m in order]); axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BN_C, marker="o", ls="", label="BN (seed mean; ○ seeds)"), Line2D([], [], color=IN_C, marker="o", ls="", label="IN"),
         Line2D([], [], color=INK2, marker="o", ls="", label="MARINE-Net (●) / AT-Trunk (■)", markerfacecolor="white"),
         Line2D([], [], color=INK2, marker="D", ls="", label="Group R (released, n = 1; ◆ thin = U-Shape min–max sensitivity)"),
         Line2D([], [], color=INK2, marker="D", ls="", markerfacecolor="white", label="Group T († exploratory, n = 1)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, out, "fig1_overall_psnr_ssim_square256", manifest); plt.close(fig)


def fig_contrasts(analysis, plt, out, manifest):
    rs = [r for r in json.load(open(analysis / "analysis.json"))["contrasts"] if r["geometry"] == "square256" and r["population"] == "ALL"]
    comps = R_ORDER + T_ORDER
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.6), sharey=True)
    for ax, met, lab in ((axs[0], "psnr", "ΔPSNR, ours − comparator (dB)"), (axs[1], "ssim", "ΔSSIM, ours − comparator")):
        ticks = []
        for i, m in enumerate(comps):
            for j, a in enumerate(ARMS):
                y = i * 5 + j; r = [x for x in rs if x["comparator"] == m and x["arm"] == a and x["metric"] == met][0]; st = arm_style(a)
                ax.plot(r["ci95_seed_mean"], [y, y], color=st["color"], lw=1.1)
                ax.scatter([r["seed_mean"]], [y], s=16, color=st["color"], marker=st["marker"], zorder=3)
            ticks.append(i * 5 + 1.5)
        ax.axvline(0, color=INK, lw=0.7); ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
        ax.axhline(len(R_ORDER) * 5 - 1, color=INK2, lw=0.6, ls=":")
    axs[0].set_yticks(ticks); axs[0].set_yticklabels([NAME[m] for m in comps]); axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker=mk, ls="-", label=NAME[a]) for a in ARMS for c, mk in [(arm_style(a)["color"], arm_style(a)["marker"])]]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(); save(fig, out, "fig2_registered_contrasts_square256_ALL", manifest); plt.close(fig)


def fig_e1r(rows, plt, out, manifest):
    sq = [r for r in rows if r["geometry"] == "square256" and "mean_A" in r and r["seed"] in ("mean", "n=1")]
    order = ARMS + R_ORDER + T_ORDER
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for y, m in enumerate(order):
        r = [x for x in sq if x["method"] == m][0]; left = 0.0
        for c in ("A", "C", "G", "R"):
            v = r[f"mean_{c}"]; ax.barh(y, v, left=left, color=COMP_C[c], height=0.62, edgecolor="white", linewidth=1.0)
            if v / r["mean_MSE"] > 0.09:
                ax.text(left + v / 2, y, c, ha="center", va="center", fontsize=6, color="white" if c in "AC" else INK)
            left += v
        ax.text(left + 0.0004, y, f"{r['mean_MSE']:.4f}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[m] for m in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean MSE and its exact E1r decomposition (A + C + G + R = MSE)")
    from matplotlib.patches import Patch
    lab = {"A": "A: achromatic (brightness) offset", "C": "C: chromatic (colour-cast) offset", "G": "G: per-channel gain / contrast", "R": "R: spatial residual"}
    ax.legend(handles=[Patch(color=COMP_C[c], label=lab[c]) for c in "ACGR"], loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, out, "fig3_error_decomposition_square256", manifest); plt.close(fig)


def fig_geometry(rows, plt, out, manifest):
    both = ARMS + ["clahe", "udcp", "waternet_lsui_s0"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, met, lab in ((axs[0], "psnr", "Mean PSNR (dB)"), (axs[1], "ssim", "Mean SSIM")):
        for y, m in enumerate(both):
            v = {geo: [x for x in rows if x["method"] == m and x["geometry"] == geo and x["seed"] in ("mean", "n=1")][0][f"mean_{met}"]
                 for geo in ("square256", "aspect482")}
            ax.plot([v["square256"], v["aspect482"]], [y, y], color=INK2, lw=1.0)
            ax.scatter([v["square256"]], [y], s=26, color=INK, zorder=3); ax.scatter([v["aspect482"]], [y], s=26, facecolor="white", edgecolor=INK, zorder=3)
            ax.text(max(v.values()), y - 0.3, f"  {v['aspect482'] - v['square256']:+.3f}", fontsize=6, color=INK2, va="bottom")
        ax.set_xlabel(lab); ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(both))); axs[0].set_yticklabels([NAME[m] for m in both]); axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", ls="", label="square 256 × 256"),
                        Line2D([], [], color=INK, marker="o", ls="", markerfacecolor="white", label="aspect 482 × 256 (Δ = 482 − 256)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); save(fig, out, "fig4_geometry_sensitivity", manifest); plt.close(fig)


def crop_box(ref_chw, size=64, stride=8):
    import cv2
    g = cv2.cvtColor((ref_chw.transpose(1, 2, 0) * 255).astype(np.float32), cv2.COLOR_RGB2GRAY)
    e = cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2
    best, bx = -1, (0, 0)
    for yy in range(0, g.shape[0] - size + 1, stride):
        for xx in range(0, g.shape[1] - size + 1, stride):
            s = float(e[yy:yy + size, xx:xx + size].sum())
            if s > best:
                best, bx = s, (yy, xx)
    return bx


def fig_visual(out, plt, manifest, ids, comps, ours_tab):
    import torch
    import train_v2 as tv
    import paper1_final_eval_v2 as V
    vis = P.visual_selection(); pairs = K.load_pairs(None); sel = [pairs[ids.index(k)] for k in vis]
    X, Y = tv.preload(sel); x = X.float() / 255.0; y = Y.float() / 255.0
    dev = torch.device("cuda"); K.set_tf32_off()
    imgs, checks = {}, {}
    for a in ARMS:
        k = f"{a}_s0"; path, _ = V.verify_checkpoint2(V.MODELS[k][0], "best_valA"); mdl = V.load_model2(k, path, "best_valA", dev)
        with torch.no_grad():
            raw = mdl(x.to(dev))["enhanced"]; K.check_finite(raw, k); imgs[a] = torch.clamp(raw, 0, 1).float().cpu().numpy()
        np.save(out / f"ours_visual_{k}.npy", imgs[a]); manifest[f"ours_visual_{k}.npy"] = K.sha(out / f"ours_visual_{k}.npy")
        del mdl; torch.cuda.empty_cache()
    for m in R_ORDER:
        imgs[m] = np.load(RUNS / FOLD[("R", "square256")] / f"visual_{m}.npy")
    for m in T_ORDER:
        imgs[m] = np.load(RUNS / FOLD[("T", "square256")] / f"visual_{m}.npy")
    vi = [ids.index(k) for k in vis]; ann = {}
    for m in ARMS + R_ORDER + T_ORDER:
        for i in range(len(vis)):
            t = torch.from_numpy(np.ascontiguousarray(imgs[m][i], dtype=np.float32))[None]
            ps, ss = float(tv.compute_psnr(t, y[i:i + 1])), float(tv.compute_ssim(t, y[i:i + 1]))
            ref_p = ours_tab["square256"][f"{m}_s0|psnr"][vi[i]] if m in ARMS else comps[(m, "square256")]["psnr"][vi[i]]
            ref_s = ours_tab["square256"][f"{m}_s0|ssim"][vi[i]] if m in ARMS else comps[(m, "square256")]["ssim"][vi[i]]
            checks[f"{m}|{vis[i]}"] = (ps - ref_p, ss - ref_s)
            if abs(ps - ref_p) > 1e-4 or abs(ss - ref_s) > 1e-5:
                raise SystemExit(f"STOP: visual {m} {vis[i]} PSNR/SSIM {ps:.6f}/{ss:.6f} != saved per-image {ref_p:.6f}/{ref_s:.6f}")
            ann[(m, i)] = (ps, ss)
    boxes = [crop_box(y[i].numpy()) for i in range(len(vis))]
    from matplotlib.patches import Rectangle

    def panel(cols, idx, stem, title):
        allc = ["input", "reference"] + cols
        fig, ax3 = plt.subplots(3 * len(idx), len(allc), figsize=(0.78 * len(allc), 0.86 * 2.3 * len(idx)), squeeze=False,
                                gridspec_kw={"height_ratios": [1, 1, 0.3] * len(idx), "hspace": 0.12, "wspace": 0.04})
        for sp_ax in ax3[2::3].ravel():
            sp_ax.axis("off")
        ax = np.concatenate([ax3[k:k + 2] for k in range(0, 3 * len(idx), 3)])
        for r, i in enumerate(idx):
            by, bx = boxes[i]
            for j, c in enumerate(allc):
                im = x[i].numpy() if c == "input" else y[i].numpy() if c == "reference" else imgs[c][i]
                a1, a2 = ax[2 * r, j], ax[2 * r + 1, j]
                a1.imshow(np.clip(im.transpose(1, 2, 0), 0, 1)); a1.add_patch(Rectangle((bx, by), 64, 64, fill=False, ec="#e34948", lw=0.6))
                a2.imshow(np.clip(im[:, by:by + 64, bx:bx + 64].transpose(1, 2, 0), 0, 1), interpolation="nearest")
                for aa in (a1, a2):
                    aa.set_xticks([]); aa.set_yticks([]); aa.grid(False)
                    for sp in aa.spines.values():
                        sp.set_visible(False)
                if r == 0:
                    a1.set_title(NAME.get(c, c.capitalize()), fontsize=5.5, color=INK)
                a2.set_xlabel(vis[i] if c == "input" else "" if c == "reference" else f"{ann[(c, i)][0]:.2f} dB / {ann[(c, i)][1]:.3f}",
                              fontsize=5, color=INK2, labelpad=1.5)
        fig.suptitle(title, fontsize=6.5, color=INK, y=0.995); fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)
        save(fig, out, stem, manifest); plt.close(fig)

    for p, idx in enumerate((range(0, 4), range(4, 8), range(8, 12))):
        panel(ARMS + R_ORDER, list(idx), f"fig5{'abc'[p]}_visual_groupR_images{idx[0] + 1}-{idx[-1] + 1}",
              "Ours (seed 0) and Group R released checkpoints (descriptive, n = 1). Below each output: PSNR (dB) / SSIM vs the computed HICRD reference.")
    for p, idx in enumerate((range(0, 6), range(6, 12))):
        panel(ARMS + T_ORDER, list(idx), f"fig6{'ab'[p]}_visual_groupT_images{idx[0] + 1}-{idx[-1] + 1}",
              "Ours (seed 0) and Group T LSUI retrains († EXPLORATORY, n = 1). Below each output: PSNR (dB) / SSIM.")
    json.dump({"crop_boxes_yx": dict(zip(vis, boxes)), "annotation_vs_saved_max_abs": [max(abs(v[0]) for v in checks.values()), max(abs(v[1]) for v in checks.values())],
               "annotations": {f"{m}|{vis[i]}": v for (m, i), v in ann.items()}}, open(out / "visual_checks.json", "x"), indent=1)
    manifest["visual_checks.json"] = K.sha(out / "visual_checks.json")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", required=True); ap.add_argument("--post-check", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rp = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    out = K.check_out(rp(a.out), False)
    verify_inputs(rp(a.post_check), rp(a.analysis))
    ids = AN.manifest_ids(); comps, ours, oe = load_metrics(ids)
    out.mkdir(parents=True); manifest = {"script_sha256": K.sha(__file__)}
    plt = plt_setup()
    rows = tables(comps, ours, oe, ids, out, manifest)
    fig_overall(rows, plt, out, manifest); fig_contrasts(rp(a.analysis), plt, out, manifest); fig_e1r(rows, plt, out, manifest); fig_geometry(rows, plt, out, manifest)
    fig_visual(out, plt, manifest, ids, comps, ours)
    json.dump(manifest, open(out / "figure_manifest.json", "x"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
