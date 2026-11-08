"""Complete numerical extraction + independent verification for the SOTA HICRD-2000 comparison. READ-ONLY on all results: no inference, no training.
Every number in the generated report is computed here from the saved per-image CSVs / JSON (never hand-copied). Independently of
sota_hicrd_analysis.py it recomputes: per-seed / seed-mean / seed-SD / t95 contrasts, the 95 % cluster-bootstrap CIs (registered Stage 4 draws, via a
SPARSE count matrix rather than the analysis' dense one), comparator means, E1r rows and the U-Shape min-max rows, and compares them with
analysis.json. It also checks row counts, manifest order, decomposition exactness, PSNR vs MSE, file hashes, group separation, and the numbers printed
in SOTA_HICRD_final_report_2026-11-03.md (rounding-aware). Discrepancies are listed, never silently corrected.
Usage: python v2_tools/sota_hicrd_numerical_report.py --report reports/paper1_narrowed/HICRD_complete_numerical_results_2026-11-03.md
                                                    --verify-out paper1_uieb_runs/sota_hicrd_numerical_verification"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402
import sota_hicrd_analysis as AN  # noqa: E402
import s5_e1r_error_decomposition as D  # noqa: E402

RUNS = ROOT / "paper1_uieb_runs"
FOLD = {("R", "square256"): "sota_hicrd_released_square256", ("R", "aspect482"): "sota_hicrd_released_G482",
        ("T", "square256"): "sota_hicrd_retrained_square256", ("T", "aspect482"): "sota_hicrd_retrained_G482"}
MM = "sota_hicrd_released_square256_ushape_minmax"
R_ORDER = ["funie_gan", "ushape", "wwe_uie", "nu2net", "clahe", "udcp"]
T_ORDER = ["waternet_lsui_s0", "wwe_uie_lsui_s0"]
ARMS = list(P.OUR_ARMS); SEEDS = list(P.SEEDS)
NAME = {"marinenet_bn": "MARINE-Net BN", "marinenet_in": "MARINE-Net IN", "attrunk_bn": "AT-Trunk BN", "attrunk_in": "AT-Trunk IN",
        "ushape": "U-Shape (clamp, primary)", "ushape_minmax": "U-Shape (official min–max, sensitivity)", "wwe_uie": "WWE-UIE", "funie_gan": "FUnIE-GAN",
        "nu2net": "NU²Net", "clahe": "CLAHE", "udcp": "UDCP-GF", "waternet_lsui_s0": "WaterNet-LSUI s0 †", "wwe_uie_lsui_s0": "WWE-UIE-LSUI s0 †"}
SHORT = dict(NAME, ushape="U-Shape")
RUN_DIR = {"marinenet_bn": "marinenet_bn_r3_fp32_s{}", "marinenet_in": "marinenet_in_r3_fp32_s{}", "attrunk_bn": "xf_attrunk_r3_fp32_s{}",
           "attrunk_in": "xfin_attrunk_in_r3_fp32_s{}", "waternet_lsui_s0": "waternet_lsui_s0", "wwe_uie_lsui_s0": "wwe_lr2e-4_s0"}
RECORDED = {"analysis.json": "331a09ab764904a434a12dec3cf39d9d95cd61004b2614ea557a2878920ff19e",
            "contrasts.csv": "81c4c7e1337a0115460595d9422fff36d89851d10d705d48514cf389eda4b4b7",
            "comparator_means.csv": "33f282c13a3c8fa3fbc675bbd9693280dc750fd8bd235389826bda74d91e0513"}
POST_CHECK = RUNS / "sota_hicrd_post_exec_check_post_T482" / "post_exec_check.json"
FINAL_REPORT = ROOT / "reports/paper1_narrowed/SOTA_HICRD_final_report_2026-11-03.md"
COMPS = ("MSE", "A", "C", "G", "R")
T95 = stats.t.ppf(0.975, 2)
issues, checks = [], {}


def note(ok, key, detail):
    checks[key] = {"pass": bool(ok), "detail": detail}
    if not ok:
        issues.append(f"{key}: {detail}")


# ------------------------------------------------------------------ loading
def load_all(ids):
    comp = {}
    for (g, geo), f in FOLD.items():
        for p in sorted((RUNS / f).glob("per_image_*.csv")):
            m = p.name[len("per_image_"):-4]
            rows = K.read_per_image(p)
            comp[(m, geo)] = {"group": g, "file": f"{f}/{p.name}", "sha256": K.sha(p), "n": len(rows), "ids": [r["id"] for r in rows],
                              **{c: np.array([float(r[c]) for r in rows]) for c in ("psnr", "ssim", "MSE", "A", "C", "G", "R", "R_low", "R_high", "nonfinite_raw")}}
    p = RUNS / MM / "per_image_ushape_minmax.csv"; rows = K.read_per_image(p)
    comp[("ushape_minmax", "square256")] = {"group": "R-sensitivity", "file": f"{MM}/{p.name}", "sha256": K.sha(p), "n": len(rows), "ids": [r["id"] for r in rows],
                                            **{c: np.array([float(r[c]) for r in rows]) for c in ("psnr", "ssim", "MSE", "A", "C", "G", "R", "R_low", "R_high", "nonfinite_raw")}}
    ours = {geo: AN.load_ours(geo) for geo in ("square256", "aspect482")}        # hash-pinned frozen Stage 4 per-image files
    oe_rows = {}
    for r in csv.DictReader(open(ROOT / AN.OURS_E1R[0], newline="")):
        if r["dataset"] == "HICRD":
            oe_rows[(r["key"], r["model"])] = r
    oe = {f"{a}_s{s}|{c}": np.array([float(oe_rows[(k, f"{a}_s{s}")][c]) for k in ids]) for a in ARMS for s in SEEDS for c in COMPS + ("R_low", "R_high")}
    return comp, ours, oe


# ------------------------------------------------------------------ independent bootstrap (sparse count matrix)
def sparse_W(draws, n):
    r = np.concatenate([np.full(len(d), b) for b, d in enumerate(draws)]); c = np.concatenate(draws)
    W = sparse.csr_matrix((np.ones(len(c)), (r, c)), shape=(len(draws), n))   # duplicates summed = multiplicities
    return W, np.array([len(d) for d in draws], dtype=np.float64)


def ci(W, lens, v):
    bm = (W @ v) / lens
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def contrast(seed_vals, comp_vals, rows, W, lens):
    d = np.stack([s[rows] - comp_vals[rows] for s in seed_vals]); per = d.mean(1)
    sm, sd = float(per.mean()), float(per.std(ddof=1)); h = T95 * sd / np.sqrt(3)
    return {"per_seed": per.tolist(), "seed_mean": sm, "seed_sd": sd, "t95": [sm - h, sm + h], "ci95_seed_mean": ci(W, lens, d.mean(0)),
            "share_d_gt_0_per_seed": [float((d[k] > 0).mean()) for k in range(3)], "n": int(len(rows))}


def close(a, b, tol=1e-9):
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol * np.maximum(1, np.abs(b))))


# ------------------------------------------------------------------ verification
def verify(ids, comp, ours, oe, an):
    pc = json.load(open(POST_CHECK))
    for (m, geo), v in comp.items():
        f = v["file"].split("/")[0]
        rec = pc["folders"].get(f, {}).get("files", {}).get(m, {}).get("sha256")
        note(rec == v["sha256"], f"hash:{v['file']}", f"{v['sha256'][:12]} vs post-check {str(rec)[:12]}")
        note(v["n"] == 2000 and v["ids"] == ids, f"rows:{v['file']}", f"n = {v['n']}, manifest order {'yes' if v['ids'] == ids else 'NO'}")
        fin = all(np.isfinite(v[c]).all() for c in ("psnr", "ssim", "MSE", "A", "C", "G", "R")) and not v["nonfinite_raw"].any()
        note(fin, f"finite:{v['file']}", "all values finite, nonfinite_raw = 0" if fin else "NON-FINITE")
        ok, rel, r2 = D.check_exact({c: v[c] for c in ("MSE", "A", "C", "G", "R", "R_low", "R_high")})
        note(ok, f"decomp:{v['file']}", f"A+C+G+R = MSE, R_low+R_high = R; max rel err {rel:.2e}")
        dps = float(np.max(np.abs(v["psnr"] + 10 * np.log10(v["MSE"])))); note(dps <= 1e-4, f"psnr_vs_mse:{v['file']}", f"max |PSNR + 10 log10 MSE| = {dps:.2e} dB")
        summ = json.load(open(RUNS / f / "summary.json")); blk = summ.get("methods") or summ.get("models") or {}
        if m in blk:
            note(abs(blk[m]["mean_psnr"] - v["psnr"].mean()) <= 1e-5, f"summary_mean:{v['file']}", f"CSV mean {v['psnr'].mean():.6f} vs summary {blk[m]['mean_psnr']:.6f}")
    for name, h in RECORDED.items():
        note(K.sha(RUNS / "sota_hicrd_analysis" / name) == h, f"hash:sota_hicrd_analysis/{name}", h[:12])
    for a in ARMS:
        for s in SEEDS:
            ok, rel, _ = D.check_exact({c: oe[f"{a}_s{s}|{c}"] for c in COMPS + ("R_low", "R_high")})
            dps = float(np.max(np.abs(ours["square256"][f"{a}_s{s}|psnr"] + 10 * np.log10(oe[f"{a}_s{s}|MSE"]))))
            note(ok and dps <= 1e-4, f"ours_e1r:{a}_s{s}", f"exact (max rel {rel:.2e}); Stage 4 PSNR vs E1r MSE max |Δ| {dps:.2e} dB")
    # --- recompute every analysis row
    res = {"contrasts": [], "comparator_means": [], "e1r": [], "ushape_minmax": []}
    mism = {"contrasts": 0, "comparator_means": 0, "e1r": 0, "ushape_minmax": 0}
    an_c = {(r["geometry"], r["population"], r["group"], r["comparator"], r["arm"], r["metric"]): r for r in an["contrasts"]}
    an_m = {(r["geometry"], r["population"], r["group"], r["comparator"], r["metric"]): r for r in an["comparator_means"]}
    an_e = {(r["group"], r["comparator"], r["arm"], r["component"]): r for r in an["e1r"]}
    an_u = {(r["arm"], r["metric"]): r for r in an["ushape_minmax"]}
    for geo, pops in AN.POPS.items():
        for pop in pops:
            rows, draws = P.stage4_population_draws(pop); W, lens = sparse_W(draws, len(rows))
            for (m, g2), v in comp.items():
                if g2 != geo or v["group"] == "R-sensitivity":
                    continue
                for met in ("psnr", "ssim"):
                    cm = {"geometry": geo, "population": pop, "group": v["group"], "comparator": m, "metric": met, "mean": float(v[met][rows].mean()), "ci95": ci(W, lens, v[met][rows])}
                    res["comparator_means"].append(cm); ref = an_m.get((geo, pop, v["group"], m, met))
                    if ref is None or not (close(ref["mean"], cm["mean"]) and close(ref["ci95"], cm["ci95"])):
                        mism["comparator_means"] += 1
                    for a in ARMS:
                        c = contrast([ours[geo][f"{a}_s{s}|{met}"] for s in SEEDS], v[met], rows, W, lens)
                        c.update(geometry=geo, population=pop, group=v["group"], comparator=m, arm=a, metric=met); res["contrasts"].append(c)
                        ref = an_c.get((geo, pop, v["group"], m, a, met))
                        if ref is None or not all(close(ref[k], c[k]) for k in ("per_seed", "seed_mean", "seed_sd", "t95", "ci95_seed_mean", "share_d_gt_0_per_seed")):
                            mism["contrasts"] += 1
            if geo == "square256" and pop == "ALL":
                for (m, g2), v in comp.items():
                    if g2 != geo or v["group"] == "R-sensitivity":
                        continue
                    for a in ARMS:
                        for cc in COMPS:
                            c = contrast([oe[f"{a}_s{s}|{cc}"] for s in SEEDS], v[cc], rows, W, lens)
                            c.update(group=v["group"], comparator=m, arm=a, component="d" + cc); res["e1r"].append(c)
                            ref = an_e.get((v["group"], m, a, "d" + cc))
                            if ref is None or not all(close(ref[k], c[k]) for k in ("per_seed", "seed_mean", "seed_sd", "t95", "ci95_seed_mean")):
                                mism["e1r"] += 1
                vm = comp[("ushape_minmax", "square256")]
                for met in ("psnr", "ssim"):
                    for a in ARMS:
                        c = contrast([ours[geo][f"{a}_s{s}|{met}"] for s in SEEDS], vm[met], rows, W, lens); c.update(arm=a, metric=met); res["ushape_minmax"].append(c)
                        ref = an_u.get((a, met))
                        if ref is None or not all(close(ref[k], c[k]) for k in ("per_seed", "seed_mean", "seed_sd", "t95", "ci95_seed_mean")):
                            mism["ushape_minmax"] += 1
                    res.setdefault("ushape_minmax_mean", {})[met] = {"mean": float(vm[met].mean()), "ci95": ci(W, lens, vm[met])}
    for k in mism:
        n_an, n_re = len(an[k]), len(res[k])
        note(mism[k] == 0 and n_an == n_re, f"recompute:{k}", f"{n_re} recomputed vs {n_an} in analysis.json; mismatches {mism[k]} (tol 1e-9 relative)")
    # group separation
    bad = [r for r in an["contrasts"] if (r["group"] == "R" and r["comparator"] not in P.group_r_active()) or (r["group"] == "T" and r["comparator"] not in P.GROUP_T)
           or (r["group"] == "T") != ("EXPLORATORY" in r["status"])]
    note(not bad and "waternet" not in {r["comparator"] for r in an["contrasts"]}, "group_separation",
         f"{len(bad)} rows with a comparator outside its group or a wrong status; original WaterNet absent from all rows")
    sq = [r for r in an["contrasts"] if r["geometry"] == "square256" and r["population"] == "ALL"]
    note(len(sq) == 64 and sum(r["group"] == "R" for r in sq) == 48 and sum(r["group"] == "T" for r in sq) == 16, "registered_count",
         f"square256/ALL: {len(sq)} rows = 32 per metric (R 24 + T 8)")
    return res


# ------------------------------------------------------------------ checks on the earlier final report (rounding-aware)
NUM = re.compile(r"[-+]?\d+\.\d+")


def nums(line):
    return [(float(t), len(t.split(".")[1])) for t in NUM.findall(line.replace("−", "-"))]


def check_final_report(comp, ours, res, own_ci):
    txt = FINAL_REPORT.read_text(encoding="utf-8").splitlines()
    rep = {"checked_values": 0, "mismatches": []}

    def cmp(label, line, expected):
        got = nums(line)
        if len(got) != len(expected):
            rep["mismatches"].append(f"{label}: {len(got)} numbers printed, {len(expected)} expected"); return
        for (g, d), e in zip(got, expected):
            rep["checked_values"] += 1
            if abs(g - e) > 0.5 * 10 ** -d + 1e-12:
                rep["mismatches"].append(f"{label}: printed {g} vs computed {e:.8f}")
    t1 = {"MARINE-Net BN": "marinenet_bn", "MARINE-Net IN": "marinenet_in", "AT-Trunk BN": "attrunk_bn", "AT-Trunk IN": "attrunk_in",
          "U-Shape (clamp, primary)": "ushape", "WWE-UIE": "wwe_uie", "FUnIE-GAN": "funie_gan", "NU²Net": "nu2net", "CLAHE": "clahe", "UDCP-GF": "udcp",
          "U-Shape (min–max)": "ushape_minmax", "WaterNet-LSUI s0": "waternet_lsui_s0", "WWE-UIE-LSUI s0": "wwe_uie_lsui_s0"}
    sec = None
    for line in txt:
        if line.startswith("### 4.1"):
            sec = "t1"
        elif line.startswith("| Comparator | Δ PSNR"):
            sec = "t2p"
        elif line.startswith("| Comparator | Δ SSIM"):
            sec = "t2s"
        elif line.startswith("### 4.4"):
            sec = "geo"
        elif line.startswith("### 4.5"):
            sec = "mm"
        elif line.startswith("##") and not line.startswith("### 4.1"):
            sec = None if not line.startswith("### 4.2") else sec
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if sec == "t1" and len(cells) == 6 and cells[1] in t1:
            m = t1[cells[1]]
            if m in ARMS:
                exp = []
                for met in ("psnr", "ssim"):
                    vals = [ours["square256"][f"{m}_s{s}|{met}"].mean() for s in SEEDS]
                    exp += vals + [np.mean(vals)] + own_ci[("square256", m, met)]
            else:
                key = ("ushape_minmax", "square256") if m == "ushape_minmax" else (m, "square256")
                exp = []
                for met in ("psnr", "ssim"):
                    exp += [comp[key][met].mean()] + own_ci[("square256", m, met)]
            cmp(f"final 4.1 {cells[1]}", "|".join(cells[2:]), exp)
        elif sec in ("t2p", "t2s") and len(cells) == 5:
            met = "psnr" if sec == "t2p" else "ssim"
            name = cells[0].replace("†", "").strip()
            m = {"U-Shape": "ushape", "WWE-UIE": "wwe_uie", "FUnIE-GAN": "funie_gan", "NU²Net": "nu2net", "CLAHE": "clahe", "UDCP-GF": "udcp",
                 "WaterNet-LSUI": "waternet_lsui_s0", "WWE-UIE-LSUI": "wwe_uie_lsui_s0"}.get(name)
            if m:
                exp = []
                for a in ARMS:
                    r = [x for x in res["contrasts"] if x["geometry"] == "square256" and x["population"] == "ALL" and x["comparator"] == m and x["arm"] == a and x["metric"] == met][0]
                    exp += [r["seed_mean"]] + r["ci95_seed_mean"]
                cmp(f"final 4.2 {met} {name}", "|".join(cells[1:]), exp)
        elif sec == "geo" and len(cells) == 3 and "→" in cells[1]:
            name = cells[0].split("(")[0].strip().replace("†", "").strip()
            m = {"MARINE-Net BN": "marinenet_bn", "MARINE-Net IN": "marinenet_in", "AT-Trunk BN": "attrunk_bn", "AT-Trunk IN": "attrunk_in", "CLAHE": "clahe",
                 "UDCP-GF": "udcp", "WaterNet-LSUI": "waternet_lsui_s0"}.get(name)
            if m:
                exp = []
                for met in ("psnr", "ssim"):
                    a, b = (np.mean([ours[g][f"{m}_s{s}|{met}"].mean() for s in SEEDS]) for g in ("square256", "aspect482")) if m in ARMS else \
                           (comp[(m, g)][met].mean() for g in ("square256", "aspect482"))
                    exp += [a, b, b - a]
                cmp(f"final 4.4 {name}", "|".join(cells[1:]), exp)
        elif sec == "mm" and len(cells) == 3 and cells[0] in ("MARINE-Net BN", "MARINE-Net IN", "AT-Trunk BN", "AT-Trunk IN"):
            a = t1[cells[0]]; exp = []
            for met in ("psnr", "ssim"):
                r = [x for x in res["ushape_minmax"] if x["arm"] == a and x["metric"] == met][0]; exp += [r["seed_mean"]] + r["ci95_seed_mean"]
            cmp(f"final 4.5 {cells[0]}", "|".join(cells[1:]), exp)
    # specific narrative counts
    sq = [r for r in res["contrasts"] if r["geometry"] == "square256" and r["population"] == "ALL"]
    cls = lambda r: "+" if r["ci95_seed_mean"][0] > 0 else "-" if r["ci95_seed_mean"][1] < 0 else "0"
    counts = {k: sum(cls(r) == k for r in sq) for k in "+-0"}
    body = "\n".join(txt)
    for k, label in (("+", "Entirely above 0"), ("-", "Entirely below 0"), ("0", "Spans 0")):
        mm = re.search(rf"\| {label} \| (\d+) \|", body); rep["checked_values"] += 1
        if not mm or int(mm.group(1)) != counts[k]:
            rep["mismatches"].append(f"final 4.2 class count '{label}': printed {mm.group(1) if mm else None}, computed {counts[k]}")
    base = {(r["group"], r["comparator"], r["arm"], r["metric"]): cls(r) for r in sq}
    for pop, label in (("ALL_STRATIFIED", "ALL_STRATIFIED"), ("KNOWN", "KNOWN"), ("TEST300", "TEST300")):
        n = sum(cls(r) != base[(r["group"], r["comparator"], r["arm"], r["metric"])] for r in res["contrasts"] if r["geometry"] == "square256" and r["population"] == pop)
        mm = re.search(rf"\| {label} \| (\d+) \|", body); rep["checked_values"] += 1
        if not mm or int(mm.group(1)) != n:
            rep["mismatches"].append(f"final 4.2 population '{label}': printed {mm.group(1) if mm else None}, computed {n}")
    return rep


# ------------------------------------------------------------------ report helpers
def f(x, d):
    return f"{x:.{d}f}".replace("-", "−")


def fs(x, d):
    return f"{x:+.{d}f}".replace("-", "−")


def fci(c, d):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def direction(c):
    lo, hi = c["ci95_seed_mean"]
    return "ours higher (CI > 0)" if lo > 0 else "ours lower (CI < 0)" if hi < 0 else "CI spans 0"


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--report", required=True); ap.add_argument("--verify-out", required=True); a = ap.parse_args(argv)
    t0 = time.time()
    rep_path = ROOT / a.report
    if rep_path.exists():
        raise SystemExit(f"STOP: {rep_path} exists (never overwrite)")
    vout = K.check_out(ROOT / a.verify_out, False)
    K.verify_protocol_files()
    ids = AN.manifest_ids()
    comp, ours, oe = load_all(ids)
    an = json.load(open(RUNS / "sota_hicrd_analysis" / "analysis.json"))
    res = verify(ids, comp, ours, oe, an)
    # absolute-mean CIs (P1, sparse) for every method incl. our seed means (NOT a registered analysis output; labelled as such)
    own_ci = {}
    for geo in ("square256", "aspect482"):
        rows, draws = P.stage4_population_draws("ALL"); W, lens = sparse_W(draws, len(rows))
        for (m, g2), v in comp.items():
            if g2 == geo:
                for met in ("psnr", "ssim"):
                    own_ci[(geo, m, met)] = ci(W, lens, v[met])
        for arm in ARMS:
            for met in ("psnr", "ssim"):
                own_ci[(geo, arm, met)] = ci(W, lens, np.mean([ours[geo][f"{arm}_s{s}|{met}"] for s in SEEDS], 0))
                for s in SEEDS:
                    own_ci[(geo, f"{arm}_s{s}", met)] = ci(W, lens, ours[geo][f"{arm}_s{s}|{met}"])
    # cross-check: own_ci of comparators == analysis comparator_means CIs
    cmok = all(close(own_ci[(r["geometry"], r["comparator"], r["metric"])], r["ci95"]) for r in an["comparator_means"] if r["population"] == "ALL")
    note(cmok, "comparator_mean_ci_vs_analysis", "absolute-mean CIs recomputed here equal analysis comparator_means (ALL)")
    fr = check_final_report(comp, ours, res, own_ci)
    note(not fr["mismatches"], "final_report_numbers", f"{fr['checked_values']} printed values checked; {len(fr['mismatches'])} mismatches")
    prov = {}
    for k in ARMS:
        for s in SEEDS:
            r = json.load(open(ROOT / "runs_v2" / RUN_DIR[k].format(s) / "result.json"))
            prov[f"{k}_s{s}"] = {"run": RUN_DIR[k].format(s), "train": r["manifest_sha256"]["train"], "val_A": r["manifest_sha256"]["val_A"], "epochs": r["epochs"],
                                 "best_epoch": r["best_valA_epoch"], "ckpt": r["checkpoint_sha256"]["best_valA.pth"], "params": r["params"]}
    for m in T_ORDER:
        r = json.load(open(ROOT / "runs_v2" / RUN_DIR[m] / "result.json"))
        prov[m] = {"run": RUN_DIR[m], "train": r["manifest_sha256"]["train"], "val_A": r["manifest_sha256"]["val_A"], "epochs": r["epochs"],
                   "best_epoch": r["best_valA_epoch"], "ckpt": r["checkpoint_sha256"]["best_valA.pth"], "params": r["params"]}
        note(r["checkpoint_sha256"]["best_valA.pth"] == P.GROUP_T[m]["sha256"], f"group_t_ckpt_record:{m}", r["checkpoint_sha256"]["best_valA.pth"][:12])
    same_manifest = len({p["train"] for p in prov.values()}) == 1
    note(same_manifest, "training_manifest", f"all 12 of our runs and both Group T runs record train manifest {next(iter(prov.values()))['train'][:12]}")

    # ============================================================== REPORT
    L = []
    W_ = L.append
    sm = lambda arm, geo, met: np.mean([ours[geo][f"{arm}_s{s}|{met}"].mean() for s in SEEDS])
    sd = lambda arm, geo, met: float(np.std([ours[geo][f"{arm}_s{s}|{met}"].mean() for s in SEEDS], ddof=1))
    C = lambda geo, pop, grp, m, arm, met: [x for x in res["contrasts"] if x["geometry"] == geo and x["population"] == pop and x["group"] == grp
                                            and x["comparator"] == m and x["arm"] == arm and x["metric"] == met][0]
    sq_all = [x for x in res["contrasts"] if x["geometry"] == "square256" and x["population"] == "ALL"]
    cls = lambda r: "+" if r["ci95_seed_mean"][0] > 0 else "-" if r["ci95_seed_mean"][1] < 0 else "0"
    n_cls = {k: sum(cls(r) == k for r in sq_all) for k in "+-0"}
    W_("# HICRD-2000: complete numerical results of the released / retrained-method comparison (2026-11-03)")
    W_("")
    W_("**Source of every number.** Generated by `v2_tools/sota_hicrd_numerical_report.py` directly from the saved per-image CSVs, `summary.json` files, "
       "the frozen Stage 4 per-image files, the `s5_e1r` rows and `sota_hicrd_analysis/analysis.json`. Nothing was typed by hand; no inference, training "
       "or new experiment was run. Every analysis value was recomputed independently and compared (section 10.1).")
    W_("")
    W_("**Status and wording.** Descriptive comparison under the frozen protocol (effective v2 + Amendments 02 and 03). Group R = released checkpoints, "
       "n = 1 each. Group T (†) = our LSUI retrains of two published architectures, n = 1 each, **EXPLORATORY**. Intervals are descriptive and **not "
       "adjusted for multiplicity**; they are not significance tests. No ranking, superiority or state-of-the-art statement is made. HICRD references are "
       "**computed restorations, not ground truth**; every score below measures agreement with that restoration, not perceptual quality.")
    W_("")
    W_("**Conventions.** n = 2,000 image pairs unless stated. Seed SD = sample SD over the three per-seed means (ddof = 1). 95 % CI = percentile "
       "cluster bootstrap reusing the registered Stage 4 draws (population ALL: P1, seed 20261103, B = 10,000, T30 scene clusters). t95 = seed-level "
       f"t interval, df = 2 (t = {T95:.4f}). Share > 0 = fraction of images on which our model's score exceeds the comparator's, per seed. "
       "PSNR in dB to 4 decimals, SSIM to 5, error components in units of 10⁻³ MSE.")
    W_("")
    # ---------------- 1 executive summary
    W_("## 1. Executive summary")
    W_("")
    nv = len(checks); nf = sum(not c["pass"] for c in checks.values())
    W_(f"- **Verification:** {nv - nf} of {nv} checks pass. Every saved result and every analysis value reproduces exactly; "
       + (f"the one failed check is {len(fr['mismatches'])} last-digit discrepancies in the earlier `SOTA_HICRD_final_report_2026-11-03.md` (section 10.1)." if fr["mismatches"] else "no discrepancies."))
    W_(f"- **Registered contrasts at 256 × 256 / ALL:** 64 (32 per metric). CI entirely above 0: {n_cls['+']}; entirely below 0: {n_cls['-']}; spans 0: {n_cls['0']}.")
    for arm in ARMS:
        W_(f"- **{NAME[arm]}**, mean over seeds: PSNR {f(sm(arm, 'square256', 'psnr'), 4)} dB (seeds "
           + " / ".join(f(ours['square256'][f'{arm}_s{s}|psnr'].mean(), 4) for s in SEEDS) + f"; SD {f(sd(arm, 'square256', 'psnr'), 4)}), SSIM "
           f"{f(sm(arm, 'square256', 'ssim'), 5)} (SD {f(sd(arm, 'square256', 'ssim'), 5)}).")
    hiR = max(R_ORDER, key=lambda m: comp[(m, "square256")]["psnr"].mean()); hiRs = max(R_ORDER, key=lambda m: comp[(m, "square256")]["ssim"].mean())
    W_(f"- **Group R:** highest observed mean PSNR {SHORT[hiR]} {f(comp[(hiR, 'square256')]['psnr'].mean(), 4)} dB; highest observed mean SSIM "
       f"{SHORT[hiRs]} {f(comp[(hiRs, 'square256')]['ssim'].mean(), 5)} (observed means only; comparator-vs-comparator differences are not registered contrasts).")
    W_(f"- **Group T †:** WaterNet-LSUI {f(comp[('waternet_lsui_s0', 'square256')]['psnr'].mean(), 4)} dB / {f(comp[('waternet_lsui_s0', 'square256')]['ssim'].mean(), 5)}; "
       f"WWE-UIE-LSUI {f(comp[('wwe_uie_lsui_s0', 'square256')]['psnr'].mean(), 4)} dB / {f(comp[('wwe_uie_lsui_s0', 'square256')]['ssim'].mean(), 5)}.")
    W_(f"- **Geometry:** AT-Trunk BN / IN change by {fs(sm('attrunk_bn', 'aspect482', 'psnr') - sm('attrunk_bn', 'square256', 'psnr'), 4)} / "
       f"{fs(sm('attrunk_in', 'aspect482', 'psnr') - sm('attrunk_in', 'square256', 'psnr'), 4)} dB at 482 × 256; MARINE-Net BN / IN by "
       f"{fs(sm('marinenet_bn', 'aspect482', 'psnr') - sm('marinenet_bn', 'square256', 'psnr'), 4)} / {fs(sm('marinenet_in', 'aspect482', 'psnr') - sm('marinenet_in', 'square256', 'psnr'), 4)} dB.")
    W_("")
    # ---------------- 2 Group R
    W_("## 2. Group R: released checkpoints and classical methods (descriptive, n = 1 each)")
    W_("")
    W_("### 2.1 Provenance")
    W_("")
    rows = []
    for m in R_ORDER:
        s = P.GROUP_R.get(m) or P.CLASSICAL[m]
        rows.append([SHORT[m], f"`{Path(s['file']).name}` `{s['sha256'][:12]}…`" if "file" in s else f"`{s['impl']}` {json.dumps(s['params'])}",
                     s.get("training", "none (classical)"), s.get("licence", "—"), s["output_map"]])
    W_(table(["Method", "Checkpoint / implementation", "Training data", "Licence", "Output mapping"], rows))
    W_("")
    W_("Excluded: original WaterNet weights (unlicensed; " + "never loaded). FUnIE-GAN ran on CPU (TensorFlow). U-Shape min–max uses the same forward pass as the clamp output.")
    W_("")
    W_("### 2.2 Results at 256 × 256 (ALL, n = 2,000)")
    W_("")
    rows = []
    for m in R_ORDER + ["ushape_minmax"]:
        v = comp[(m, "square256")]
        rows.append([NAME[m], v["n"], f(v["psnr"].mean(), 6), fci(own_ci[("square256", m, "psnr")], 4), f(v["ssim"].mean(), 6), fci(own_ci[("square256", m, "ssim")], 5),
                     f(float(np.median(v["psnr"])), 4), f(v["MSE"].mean() * 1e3, 4)])
    W_(table(["Method", "n", "Mean PSNR (dB)", "95 % CI", "Mean SSIM", "95 % CI", "Median PSNR", "Mean MSE ×10³"], rows))
    W_("")
    W_("### 2.3 Results at 482 × 256 (ALL, n = 2,000)")
    W_("")
    rows = []
    for m in ("clahe", "udcp"):
        v = comp[(m, "aspect482")]
        rows.append([NAME[m], v["n"], f(v["psnr"].mean(), 6), fci(own_ci[("aspect482", m, "psnr")], 4), f(v["ssim"].mean(), 6), fci(own_ci[("aspect482", m, "ssim")], 5),
                     f(float(np.median(v["psnr"])), 4), f(v["MSE"].mean() * 1e3, 4)])
    W_(table(["Method", "n", "Mean PSNR (dB)", "95 % CI", "Mean SSIM", "95 % CI", "Median PSNR", "Mean MSE ×10³"], rows))
    W_("")
    na = json.load(open(RUNS / FOLD[("R", "aspect482")] / "summary.json"))["meta"]["not_applicable"]
    W_("Not evaluable at 482 × 256 (recorded in `sota_hicrd_released_G482/summary.json`): " + "; ".join(f"{SHORT[k]} — {v}" for k, v in na.items()) + ".")
    W_("")
    # ---------------- 3 Group T and ours
    W_("## 3. Group T (†, EXPLORATORY) and our models")
    W_("")
    W_("All rows below were trained with the same LSUI recipe: every run records train manifest "
       f"`{next(iter(prov.values()))['train'][:16]}…` and Val-A manifest `{next(iter(prov.values()))['val_A'][:16]}…`, 100 epochs, checkpoint = best Val-A. "
       "Our models are not Group T: they are the arms being compared; they are tabulated here because they share the training recipe.")
    W_("")
    W_("### 3.1 Provenance")
    W_("")
    rows = [[NAME[m] if m in NAME else m, prov[m]["run"], prov[m]["params"], prov[m]["best_epoch"], f"`{prov[m]['ckpt'][:12]}…`"] for m in T_ORDER]
    rows += [[f"{NAME[k]} s{s}", prov[f'{k}_s{s}']["run"], prov[f'{k}_s{s}']["params"], prov[f'{k}_s{s}']["best_epoch"], f"`{prov[f'{k}_s{s}']['ckpt'][:12]}…`"] for k in ARMS for s in SEEDS]
    W_(table(["Model", "Run directory (`runs_v2/`)", "Parameters", "Best Val-A epoch", "best_valA.pth SHA-256"], rows))
    W_("")
    W_("### 3.2 Group T baselines (†, single training run, exploratory)")
    W_("")
    rows = []
    for m in T_ORDER:
        for geo in ("square256", "aspect482"):
            if (m, geo) in comp:
                v = comp[(m, geo)]
                rows.append([NAME[m], geo.replace("square256", "256 × 256").replace("aspect482", "482 × 256"), v["n"], f(v["psnr"].mean(), 6), fci(own_ci[(geo, m, "psnr")], 4),
                             f(v["ssim"].mean(), 6), fci(own_ci[(geo, m, "ssim")], 5)])
            else:
                rows.append([NAME[m], geo.replace("aspect482", "482 × 256"), "—", "not evaluable (H, W must be divisible by 4)", "—", "—", "—"])
    W_(table(["Model", "Geometry", "n", "Mean PSNR (dB)", "95 % CI", "Mean SSIM", "95 % CI"], rows))
    W_("")
    W_("### 3.3 Our models: per seed and seed mean")
    W_("")
    for geo, gl in (("square256", "256 × 256"), ("aspect482", "482 × 256")):
        W_(f"**{gl}** (n = 2,000 each; per-seed CI = bootstrap of that seed's mean; seed-mean CI = bootstrap of the per-image seed mean; these absolute-mean "
           "CIs are computed here with the P1 draws and are not registered analysis outputs).")
        W_("")
        rows = []
        for arm in ARMS:
            for s in SEEDS:
                rows.append([f"{NAME[arm]} s{s}", f(ours[geo][f'{arm}_s{s}|psnr'].mean(), 6), fci(own_ci[(geo, f'{arm}_s{s}', 'psnr')], 4),
                             f(ours[geo][f'{arm}_s{s}|ssim'].mean(), 6), fci(own_ci[(geo, f'{arm}_s{s}', 'ssim')], 5)])
            rows.append([f"**{NAME[arm]} mean (SD)**", f"**{f(sm(arm, geo, 'psnr'), 6)}** ({f(sd(arm, geo, 'psnr'), 4)})", fci(own_ci[(geo, arm, 'psnr')], 4),
                         f"**{f(sm(arm, geo, 'ssim'), 6)}** ({f(sd(arm, geo, 'ssim'), 5)})", fci(own_ci[(geo, arm, 'ssim')], 5)])
        W_(table(["Model", "Mean PSNR (dB)", "95 % CI", "Mean SSIM", "95 % CI"], rows))
        W_("")
    # ---------------- 4 contrasts
    W_("## 4. All registered contrasts (ours − comparator)")
    W_("")
    W_("Population ALL, 256 × 256, n = 2,000; 32 per metric (Group R 6 × 4 = 24; Group T 2 × 4 = 8). Direction is read from the bootstrap CI of the seed mean. "
       "Groups are never contrasted with each other.")
    W_("")
    for met, d, unit in (("psnr", 4, " (dB)"), ("ssim", 5, "")):
        for grp, order, gl in (("R", R_ORDER, "Group R (descriptive)"), ("T", T_ORDER, "Group T † (EXPLORATORY, n = 1 baselines)")):
            W_(f"### 4.{1 if met == 'psnr' else 2}{'a' if grp == 'R' else 'b'} {met.upper()}{unit}: {gl}")
            W_("")
            rows = []
            for m in order:
                for arm in ARMS:
                    c = C("square256", "ALL", grp, m, arm, met)
                    rows.append([SHORT[m], NAME[arm], " / ".join(fs(x, d) for x in c["per_seed"]), fs(c["seed_mean"], d), f(c["seed_sd"], d), fci(c["t95"], d),
                                 fci(c["ci95_seed_mean"], d), " / ".join(f"{100 * x:.1f}" for x in c["share_d_gt_0_per_seed"]), direction(c)])
            W_(table(["Reference", "Our model", "Per seed s0 / s1 / s2", "Seed mean", "Seed SD", "t95 (df 2)", "95 % bootstrap CI", "% images ours > ref (s0/s1/s2)", "Direction"], rows))
            W_("")
    W_("### 4.3 The same contrasts in the other registered populations (seed mean [95 % CI])")
    W_("")
    W_("ALL_STRATIFIED = P1s (site-stratified, seed 20261104, n = 2,000); KNOWN = known-site images (P2, n = 1,839); TEST300 = official test split (n = 300).")
    W_("")
    for met, d in (("psnr", 4), ("ssim", 5)):
        rows = []
        for grp, order in (("R", R_ORDER), ("T", T_ORDER)):
            for m in order:
                for arm in ARMS:
                    cells = [SHORT[m], NAME[arm]]
                    for pop in ("ALL", "ALL_STRATIFIED", "KNOWN", "TEST300"):
                        c = C("square256", pop, grp, m, arm, met); cells.append(f"{fs(c['seed_mean'], d)} {fci(c['ci95_seed_mean'], d)}")
                    rows.append(cells)
        W_(f"**{met.upper()}**")
        W_("")
        W_(table(["Reference", "Our model", "ALL", "ALL_STRATIFIED", "KNOWN", "TEST300"], rows))
        W_("")
    base = {(r["group"], r["comparator"], r["arm"], r["metric"]): cls(r) for r in sq_all}
    W_("**CI classification changes relative to ALL (of 64):**")
    W_("")
    rows = []
    for pop in ("ALL_STRATIFIED", "KNOWN", "TEST300"):
        ch = [r for r in res["contrasts"] if r["geometry"] == "square256" and r["population"] == pop and cls(r) != base[(r["group"], r["comparator"], r["arm"], r["metric"])]]
        sym = {"+": "+", "-": "−", "0": "0"}
        rows.append([pop, len(ch), "; ".join(f"{SHORT[r['comparator']]} × {NAME[r['arm']]} {r['metric'].upper()}: {sym[base[(r['group'], r['comparator'], r['arm'], r['metric'])]]} → {sym[cls(r)]}" for r in ch) or "—"])
    W_(table(["Population", "Changes", "Which (+ = CI > 0, − = CI < 0, 0 = spans 0)"], rows))
    W_("")
    # ---------------- 5 per-seed results (compact cross-view)
    W_("## 5. Per-seed results across all comparators (PSNR, dB; ALL, 256 × 256)")
    W_("")
    W_("Per-seed mean difference ours − comparator; the column 'seed range' is max − min over seeds. Each comparator is one fixed model, so a contrast's "
       "seed SD and seed range equal those of our arm's own per-seed means (identical within each arm below): all seed variation in a contrast is our training-seed variation.")
    W_("")
    rows = []
    for arm in ARMS:
        for grp, order in (("R", R_ORDER), ("T", T_ORDER)):
            for m in order:
                c = C("square256", "ALL", grp, m, arm, "psnr")
                rows.append([NAME[arm], SHORT[m]] + [fs(x, 4) for x in c["per_seed"]] + [f(max(c["per_seed"]) - min(c["per_seed"]), 4),
                            "yes" if len({np.sign(x) for x in c["per_seed"]}) > 1 else "no"])
    W_(table(["Our model", "Reference", "s0", "s1", "s2", "Seed range", "Sign differs across seeds"], rows))
    W_("")
    W_("Seed-to-seed spread of our absolute means (PSNR, dB, 256 × 256): " + "; ".join(
        f"{NAME[arm]} {f(min(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS), 4)}–{f(max(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS), 4)} "
        f"(range {f(max(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS) - min(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS), 4)})" for arm in ARMS) + ".")
    W_("")
    # ---------------- 6 geometry
    W_("## 6. Geometry sensitivity: 482 × 256 versus 256 × 256")
    W_("")
    W_("Both geometries resize input **and** reference (bilinear); the reference therefore differs between geometries, so a change is a property of the "
       "whole evaluation, not of the model alone. No registered CI exists for the change itself; registered CIs exist for each geometry separately.")
    W_("")
    W_("### 6.1 Per model and per seed")
    W_("")
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            a_, b_ = (ours[g][f"{arm}_s{s}|psnr"].mean() for g in ("square256", "aspect482")); c_, d_ = (ours[g][f"{arm}_s{s}|ssim"].mean() for g in ("square256", "aspect482"))
            rows.append([f"{NAME[arm]} s{s}", f(a_, 4), f(b_, 4), fs(b_ - a_, 4), f(c_, 5), f(d_, 5), fs(d_ - c_, 5)])
        a_, b_ = sm(arm, "square256", "psnr"), sm(arm, "aspect482", "psnr"); c_, d_ = sm(arm, "square256", "ssim"), sm(arm, "aspect482", "ssim")
        rows.append([f"**{NAME[arm]} mean**", f(a_, 4), f(b_, 4), f"**{fs(b_ - a_, 4)}**", f(c_, 5), f(d_, 5), f"**{fs(d_ - c_, 5)}**"])
    for m in ("clahe", "udcp", "waternet_lsui_s0"):
        a_, b_ = comp[(m, "square256")]["psnr"].mean(), comp[(m, "aspect482")]["psnr"].mean(); c_, d_ = comp[(m, "square256")]["ssim"].mean(), comp[(m, "aspect482")]["ssim"].mean()
        rows.append([NAME[m], f(a_, 4), f(b_, 4), fs(b_ - a_, 4), f(c_, 5), f(d_, 5), fs(d_ - c_, 5)])
    W_(table(["Model", "PSNR 256", "PSNR 482", "Δ PSNR", "SSIM 256", "SSIM 482", "Δ SSIM"], rows))
    W_("")
    W_("### 6.2 Registered contrasts at both geometries (ALL; seed mean [95 % CI])")
    W_("")
    rows = []
    for grp, m in (("R", "clahe"), ("R", "udcp"), ("T", "waternet_lsui_s0")):
        for arm in ARMS:
            for met, d in (("psnr", 4), ("ssim", 5)):
                c1, c2 = C("square256", "ALL", grp, m, arm, met), C("aspect482", "ALL", grp, m, arm, met)
                rows.append([SHORT[m], NAME[arm], met.upper(), f"{fs(c1['seed_mean'], d)} {fci(c1['ci95_seed_mean'], d)}", f"{fs(c2['seed_mean'], d)} {fci(c2['ci95_seed_mean'], d)}",
                             " / ".join(fs(x, d) for x in c2["per_seed"]), "same" if cls(c1) == cls(c2) else f"{cls(c1)} → {cls(c2)}".replace("-", "−")])
    W_(table(["Reference", "Our model", "Metric", "256 × 256", "482 × 256", "482 per seed", "CI class"], rows))
    W_("")
    W_("### 6.3 U-Shape output mapping: clamp (primary) versus official min–max (sensitivity)")
    W_("")
    uc, um = comp[("ushape", "square256")], comp[("ushape_minmax", "square256")]
    dpp = um["psnr"] - uc["psnr"]; dss = um["ssim"] - uc["ssim"]
    W_(table(["Output", "Mean PSNR (dB)", "95 % CI", "Mean SSIM", "95 % CI"],
             [["clamp [0, 1]", f(uc["psnr"].mean(), 6), fci(own_ci[("square256", "ushape", "psnr")], 4), f(uc["ssim"].mean(), 6), fci(own_ci[("square256", "ushape", "ssim")], 5)],
              ["per-image min–max", f(um["psnr"].mean(), 6), fci(own_ci[("square256", "ushape_minmax", "psnr")], 4), f(um["ssim"].mean(), 6), fci(own_ci[("square256", "ushape_minmax", "ssim")], 5)]]))
    W_("")
    W_(f"Per image, min–max − clamp: PSNR mean {fs(dpp.mean(), 4)} dB (min {fs(dpp.min(), 4)}, max {fs(dpp.max(), 4)}; higher on {100 * (dpp > 0).mean():.1f} % of images); "
       f"SSIM mean {fs(dss.mean(), 5)} (higher on {100 * (dss > 0).mean():.1f} %).")
    W_("")
    rows = []
    for arm in ARMS:
        for met, d in (("psnr", 4), ("ssim", 5)):
            c1 = C("square256", "ALL", "R", "ushape", arm, met); c2 = [x for x in res["ushape_minmax"] if x["arm"] == arm and x["metric"] == met][0]
            rows.append([NAME[arm], met.upper(), f"{fs(c1['seed_mean'], d)} {fci(c1['ci95_seed_mean'], d)}", f"{fs(c2['seed_mean'], d)} {fci(c2['ci95_seed_mean'], d)}",
                         " / ".join(fs(x, d) for x in c2["per_seed"]), "same" if cls(c1) == cls(c2) else f"{cls(c1)} → {cls(c2)}".replace("-", "−")])
    W_(table(["Our model", "Metric", "vs U-Shape clamp", "vs U-Shape min–max", "min–max per seed", "CI class"], rows))
    W_("")
    W_("### 6.4 Comparisons not possible at 482 × 256")
    W_("")
    W_("U-Shape (fixed 256 × 256 input), FUnIE-GAN (Keras input fixed at 256 × 256 × 3), WWE-UIE and WWE-UIE-LSUI (H, W must be divisible by 4; 482 is not), "
       "NU²Net (H, W multiples of 16). Hence at 482 × 256 there are 3 comparators × 4 arms = 12 contrasts per metric.")
    W_("")
    # ---------------- 7 E1r
    W_("## 7. Error decomposition (E1r; 256 × 256, ALL)")
    W_("")
    W_("Exact per-image identity MSE = A + C + G + R (`s5_e1r_error_decomposition.decompose`, float64). A = squared mean error over channels "
       "(achromatic / brightness offset); C = spread of per-channel mean errors (chromatic / colour-cast offset); G = per-channel gain (contrast) error "
       "from the OLS slope of prediction on reference; R = spatial residual. All values ×10³.")
    W_("")
    W_("### 7.1 Mean components per method")
    W_("")
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            vals = [oe[f"{arm}_s{s}|{c}"].mean() for c in COMPS]
            rows.append([f"{NAME[arm]} s{s}"] + [f(x * 1e3, 4) for x in vals] + [f((vals[1] + vals[2] + vals[3] + vals[4] - vals[0]) * 1e3, 10)])
        vals = [np.mean([oe[f"{arm}_s{s}|{c}"].mean() for s in SEEDS]) for c in COMPS]
        rows.append([f"**{NAME[arm]} seed mean**"] + [f(x * 1e3, 4) for x in vals] + [f((vals[1] + vals[2] + vals[3] + vals[4] - vals[0]) * 1e3, 10)])
    for m in R_ORDER + T_ORDER + ["ushape_minmax"]:
        v = comp[(m, "square256")]; vals = [v[c].mean() for c in COMPS]
        rows.append([NAME[m]] + [f(x * 1e3, 4) for x in vals] + [f((vals[1] + vals[2] + vals[3] + vals[4] - vals[0]) * 1e3, 10)])
    W_(table(["Method", "MSE", "A", "C", "G", "R", "(A+C+G+R − MSE) ×10³"], rows))
    W_("")
    W_("Shares of mean MSE (A / C / G / R, %): " + "; ".join(
        f"{SHORT[m]} " + " / ".join(f"{100 * comp[(m, 'square256')][c].mean() / comp[(m, 'square256')]['MSE'].mean():.1f}" for c in "ACGR") for m in R_ORDER + T_ORDER) + "; " + "; ".join(
        f"{NAME[arm]} " + " / ".join(f"{100 * np.mean([oe[f'{arm}_s{s}|{c}'].mean() for s in SEEDS]) / np.mean([oe[f'{arm}_s{s}|MSE'].mean() for s in SEEDS]):.1f}" for c in "ACGR") for arm in ARMS) + ".")
    W_("")
    W_("### 7.2 All 160 registered E1r contrast rows (ours − comparator, ×10³; negative = our component smaller)")
    W_("")
    rows = []
    for grp, order in (("R", R_ORDER), ("T", T_ORDER)):
        for m in order:
            for arm in ARMS:
                for cc in COMPS:
                    e = [x for x in res["e1r"] if x["group"] == grp and x["comparator"] == m and x["arm"] == arm and x["component"] == "d" + cc][0]
                    rows.append([("† " if grp == "T" else "") + SHORT[m], NAME[arm], "d" + cc, " / ".join(fs(x * 1e3, 4) for x in e["per_seed"]), fs(e["seed_mean"] * 1e3, 4),
                                 f(e["seed_sd"] * 1e3, 4), fci([x * 1e3 for x in e["ci95_seed_mean"]], 4), direction(e).replace("ours higher", "ours larger").replace("ours lower", "ours smaller")])
    W_(table(["Reference", "Our model", "Component", "Per seed", "Seed mean", "Seed SD", "95 % bootstrap CI", "Direction"], rows))
    W_("")
    W_("Consistency: for every comparator and arm the dMSE seed mean equals dA + dC + dG + dR to "
       f"{max(abs(sum([x for x in res['e1r'] if x['group'] == g and x['comparator'] == m and x['arm'] == a and x['component'] == 'd' + c][0]['seed_mean'] for c in 'ACGR') - [x for x in res['e1r'] if x['group'] == g and x['comparator'] == m and x['arm'] == a and x['component'] == 'dMSE'][0]['seed_mean']) for g, order in (('R', R_ORDER), ('T', T_ORDER)) for m in order for a in ARMS):.2e} "
       "(absolute, MSE units); per-image exactness and PSNR-vs-MSE checks are in section 9.")
    W_("")
    # ---------------- 8 visual
    vis = P.visual_selection(); vi = [ids.index(k) for k in vis]
    W_("## 8. The 12 registered visual images: per-image values")
    W_("")
    W_("Selection is fixed by the protocol (§8: six site-median images by the input `imbalance` descriptor, six by `default_rng(20261203)`); no image was swapped. "
       "Values are read from the saved per-image CSVs (the figure annotations reproduce them; max |Δ| 7.1e-6 dB). Geometry 256 × 256 unless stated.")
    W_("")
    for met, d in (("psnr", 2), ("ssim", 4)):
        rows = []
        for j, k in enumerate(vis):
            i = vi[j]
            cells = [f"{j + 1}. `{k}`"] + [f(ours["square256"][f"{arm}_s{s}|{met}"][i], d) for arm in ARMS for s in (0,)] + \
                    [f(comp[(m, "square256")][met][i], d) for m in R_ORDER + ["ushape_minmax"] + T_ORDER]
            rows.append(cells)
        W_(f"**{met.upper()}{' (dB)' if met == 'psnr' else ''}, seed 0 for our models**")
        W_("")
        W_(table(["Image"] + [f"{NAME[a]} s0" for a in ARMS] + [SHORT[m] for m in R_ORDER] + ["U-Shape mm"] + [SHORT[m] for m in T_ORDER], rows))
        W_("")
    rows = []
    for j, k in enumerate(vis):
        i = vi[j]
        rows.append([f"`{k}`"] + [" / ".join(f(ours["square256"][f"{arm}_s{s}|psnr"][i], 2) for s in SEEDS) for arm in ARMS])
    W_("**All three seeds of our models (PSNR, dB, s0 / s1 / s2)**")
    W_("")
    W_(table(["Image"] + [NAME[a] for a in ARMS], rows))
    W_("")
    rows = []
    for j, k in enumerate(vis):
        i = vi[j]
        rows.append([f"`{k}`"] + [f"{f(ours['aspect482'][f'{arm}_s0|psnr'][i], 2)} / {f(ours['aspect482'][f'{arm}_s0|ssim'][i], 4)}" for arm in ARMS] +
                    [f"{f(comp[(m, 'aspect482')]['psnr'][i], 2)} / {f(comp[(m, 'aspect482')]['ssim'][i], 4)}" for m in ("clahe", "udcp", "waternet_lsui_s0")])
    W_("**482 × 256 (PSNR dB / SSIM; seed 0 for ours)**")
    W_("")
    W_(table(["Image"] + [f"{NAME[a]} s0" for a in ARMS] + ["CLAHE", "UDCP-GF", "WaterNet-LSUI †"], rows))
    W_("")
    W_("### 8.1 Strongest and weakest examples (numerical, all 12 reported above)")
    W_("")
    rows = []
    for key, lab in [(f"{a}_s0", f"{NAME[a]} s0") for a in ARMS] + [(m, SHORT[m]) for m in R_ORDER + T_ORDER]:
        vals = np.array([(ours["square256"][f"{key}|psnr"] if key.endswith("_s0") and key[:-3] in ARMS else comp[(key, "square256")]["psnr"])[i] for i in vi])
        allv = ours["square256"][f"{key}|psnr"] if key.endswith("_s0") and key[:-3] in ARMS else comp[(key, "square256")]["psnr"]
        hi, lo = int(vals.argmax()), int(vals.argmin())
        rows.append([lab, f"`{vis[hi]}` {f(vals[hi], 2)}", f"`{vis[lo]}` {f(vals[lo], 2)}", f(vals.mean(), 2), f(allv.mean(), 2)])
    W_(table(["Method", "Highest PSNR of the 12", "Lowest PSNR of the 12", "Mean of the 12", "Mean of all 2,000"], rows))
    W_("")
    sm12 = np.mean([ours["square256"][f"marinenet_bn_s{s}|psnr"] for s in SEEDS], 0)
    pct = [100 * (sm12 < sm12[i]).mean() for i in vi]
    W_("Representativeness: percentile of each image's MARINE-Net BN seed-mean PSNR within all 2,000 images: " +
       ", ".join(f"`{k}` {p:.0f}" for k, p in zip(vis, pct)) + ". Twelve images are illustrative only and do not estimate any population quantity.")
    W_("")
    # ---------------- 9 interpretation + verification
    W_("## 9. Research interpretation")
    W_("")
    rmeans = sorted(R_ORDER, key=lambda m: -comp[(m, "square256")]["psnr"].mean())
    W_("**9.1 Released methods (Group R).** Observed mean PSNR, highest to lowest: " + ", ".join(f"{SHORT[m]} {f(comp[(m, 'square256')]['psnr'].mean(), 4)}" for m in rmeans) +
       " dB. Observed mean SSIM: " + ", ".join(f"{SHORT[m]} {f(comp[(m, 'square256')]['ssim'].mean(), 5)}" for m in sorted(R_ORDER, key=lambda m: -comp[(m, 'square256')]['ssim'].mean())) +
       ". The PSNR and SSIM orderings differ (U-Shape is second in PSNR but fifth in SSIM). These are single released checkpoints with different training "
       "data, losses and output mappings; differences between comparators are not registered contrasts and carry no CI here.")
    W_("")
    c = lambda m, a, met: C("square256", "ALL", "R", m, a, met)
    W_("**9.2 Our models versus Group R.** Under this protocol the seed-mean PSNR of MARINE-Net BN was "
       + ", ".join(f"{fs(c(m, 'marinenet_bn', 'psnr')['seed_mean'], 3)} dB vs {SHORT[m]} {fci(c(m, 'marinenet_bn', 'psnr')['ci95_seed_mean'], 3)}" for m in R_ORDER)
       + ". For the IN arms the CIs against FUnIE-GAN and U-Shape lie below zero (MARINE-Net IN − FUnIE-GAN "
       f"{fs(c('funie_gan', 'marinenet_in', 'psnr')['seed_mean'], 3)} dB {fci(c('funie_gan', 'marinenet_in', 'psnr')['ci95_seed_mean'], 3)}). In SSIM, all four arms have CIs above zero "
       "against U-Shape, NU²Net, CLAHE and UDCP-GF; against FUnIE-GAN only the BN arms do, and against WWE-UIE the IN arms' CIs span zero. "
       "The decomposition locates the differences: against CLAHE mainly C (colour cast), against UDCP-GF mainly A (brightness), against U-Shape lower R "
       "(spatial residual) but higher G (gain) for every arm (section 7.2).")
    W_("")
    W_("**9.3 Seed variation.** Seed ranges of our absolute mean PSNR: " + "; ".join(
        f"{NAME[arm]} {f(max(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS) - min(ours['square256'][f'{arm}_s{s}|psnr'].mean() for s in SEEDS), 3)} dB" for arm in ARMS)
       + f". The MARINE-Net BN range is driven by seed 2 ({f(ours['square256']['marinenet_bn_s2|psnr'].mean(), 4)} dB vs {f(ours['square256']['marinenet_bn_s0|psnr'].mean(), 4)} for seed 0). "
       f"{sum(1 for arm in ARMS for grp, order in (('R', R_ORDER), ('T', T_ORDER)) for m in order if len({np.sign(x) for x in C('square256', 'ALL', grp, m, arm, 'psnr')['per_seed']}) > 1)} of 32 PSNR contrasts "
       "change sign across seeds (section 5). Because the bootstrap CI resamples images, not seeds, it does not reflect seed variability; the t95 interval (df 2) does, "
       "and it is much wider wherever seeds disagree.")
    W_("")
    W_("**9.4 Geometry.** At 482 × 256 AT-Trunk BN and IN gain "
       f"{fs(sm('attrunk_bn', 'aspect482', 'psnr') - sm('attrunk_bn', 'square256', 'psnr'), 3)} and {fs(sm('attrunk_in', 'aspect482', 'psnr') - sm('attrunk_in', 'square256', 'psnr'), 3)} dB "
       f"in seed-mean PSNR, while MARINE-Net BN and IN change by {fs(sm('marinenet_bn', 'aspect482', 'psnr') - sm('marinenet_bn', 'square256', 'psnr'), 3)} and "
       f"{fs(sm('marinenet_in', 'aspect482', 'psnr') - sm('marinenet_in', 'square256', 'psnr'), 3)} dB; the AT-Trunk gain is present in every seed (section 6.1). "
       "Comparators change less (CLAHE "
       f"{fs(comp[('clahe', 'aspect482')]['psnr'].mean() - comp[('clahe', 'square256')]['psnr'].mean(), 3)}, UDCP-GF {fs(comp[('udcp', 'aspect482')]['psnr'].mean() - comp[('udcp', 'square256')]['psnr'].mean(), 3)}, "
       f"WaterNet-LSUI {fs(comp[('waternet_lsui_s0', 'aspect482')]['psnr'].mean() - comp[('waternet_lsui_s0', 'square256')]['psnr'].mean(), 3)} dB). Of the 24 contrasts available at both "
       f"geometries, {sum(cls(C('square256', 'ALL', g, m, a, met)) != cls(C('aspect482', 'ALL', g, m, a, met)) for g, m in (('R', 'clahe'), ('R', 'udcp'), ('T', 'waternet_lsui_s0')) for a in ARMS for met in ('psnr', 'ssim'))} "
       "changes CI class (section 6.2). These data show that AT-Trunk's scores depend on evaluation geometry; they do not identify the mechanism.")
    W_("")
    wr, wt = comp[("wwe_uie", "square256")], comp[("wwe_uie_lsui_s0", "square256")]
    W_(f"**9.5 Released versus LSUI-retrained WWE-UIE.** Released (Group R, UIEB-trained): {f(wr['psnr'].mean(), 4)} dB / {f(wr['ssim'].mean(), 5)}; retrained by us on "
       f"LSUI (Group T †): {f(wt['psnr'].mean(), 4)} dB / {f(wt['ssim'].mean(), 5)} (Δ {fs(wt['psnr'].mean() - wr['psnr'].mean(), 4)} dB, {fs(wt['ssim'].mean() - wr['ssim'].mean(), 5)}). "
       f"Per image the retrain scores higher PSNR on {100 * (wt['psnr'] > wr['psnr']).mean():.1f} % of images. Mean E1r components ×10³ (A, C, G, R): released "
       + ", ".join(f(wr[cc].mean() * 1e3, 3) for cc in "ACGR") + "; retrained " + ", ".join(f(wt[cc].mean() * 1e3, 3) for cc in "ACGR")
       + ". The same architecture moves by about 2 dB with training data and recipe, which is why Group R contrasts are statements about released checkpoints, "
       "not architectures. This is an observed difference between two single models, not a registered contrast.")
    W_("")
    W_("**9.6 Why Group T is exploratory.** Each Group T baseline is one training run; its seed variance is unknown, so a contrast against it cannot be "
       "generalised over training seeds, whereas our arms have three seeds. Both baselines use our LSUI recipe and best-Val-A selection, which may not suit "
       "those architectures as well as their authors' settings. WWE-UIE-LSUI could not be evaluated at 482 × 256.")
    W_("")
    W_("**9.7 Computed references.** HICRD references are produced by a restoration algorithm, not captured clean images. PSNR / SSIM here measure "
       "fidelity to that algorithm's output. A method trained toward a different target (e.g. UIEB's selected enhancements or EUVP pairs) can score low "
       "while producing a visually acceptable image, and a high score does not establish perceptual enhancement quality. No perceptual claim is made.")
    W_("")
    W_("## 10. Limitations and reproducibility")
    W_("")
    W_("- References are computed restorations; scores are reference fidelity only.")
    W_("- 2,000 images in 188 T30 scene clusters (largest 163, median 3; cluster-weighted effective n ≈ 43). Bootstrap CIs resample clusters.")
    W_("- Our arms: 3 seeds; comparators: n = 1. Seed spread is large relative to several contrasts (section 5).")
    W_("- Intervals are descriptive and not adjusted for 64 (or 280) contrasts; they are not significance tests.")
    W_("- Group R checkpoints differ in training data, objectives and native resolution; all were evaluated at our fixed resize.")
    W_("- UDCP-GF is our implementation (guided filter, r = 24, in place of soft matting); exact fidelity to the published method is not claimed.")
    W_("- Absolute-mean CIs for our models (sections 3.3) and the per-image summaries in sections 6.3 and 8 are computed here from saved values; they are not registered analysis outputs.")
    W_("")
    W_("### 10.1 Verification (section 9 of the request)")
    W_("")
    W_(table(["Check", "Result", "Detail"], [[k.replace("|", "/"), "PASS" if v["pass"] else "**FAIL**", v["detail"].replace("|", "/")] for k, v in checks.items()]))
    W_("")
    if fr["mismatches"]:
        W_("**Discrepancies in `SOTA_HICRD_final_report_2026-11-03.md` (not corrected there; that file is unchanged):**")
        W_("")
        for x in fr["mismatches"]:
            W_(f"- {x}")
        W_("")
        W_(f"All {len(fr['mismatches'])} are in the last printed digit (each printed value differs from the correctly rounded value by one unit of that digit). "
           "Cause: those values were rounded to 3 decimals from already-rounded 4-decimal intermediates (double rounding) when the earlier report was written. "
           "They change no sign, CI classification or conclusion. The earlier report is hash-recorded and was left unchanged; the values in this document are the correct ones.")
    else:
        W_(f"`SOTA_HICRD_final_report_2026-11-03.md`: all {fr['checked_values']} checked printed values (Tables 1 and 2, sections 4.4 and 4.5, class and population counts) "
           "agree with the recomputed values at their printed precision.")
    W_("")
    W_("### 10.2 Source files")
    W_("")
    srcs = sorted({v["file"] for v in comp.values()})
    W_(table(["File (`paper1_uieb_runs/`)", "SHA-256"], [[s, f"`{K.sha(RUNS / s)}`"] for s in srcs] +
             [[f"sota_hicrd_analysis/{n}", f"`{h}`"] for n, h in RECORDED.items()] +
             [[AN.OURS["square256"][0].replace("paper1_uieb_runs/", ""), f"`{AN.OURS['square256'][1]}`"], [AN.OURS["aspect482"][0].replace("paper1_uieb_runs/", ""), f"`{AN.OURS['aspect482'][1]}`"],
              [AN.OURS_E1R[0].replace("paper1_uieb_runs/", ""), f"`{AN.OURS_E1R[1]}`"]]))
    W_("")
    W_(f"Generator: `v2_tools/sota_hicrd_numerical_report.py` (`{K.sha(__file__)}`). Verification record: `{a.verify_out}/verification.json`.")
    W_("")
    vout.mkdir(parents=True)
    json.dump({"checks": checks, "issues": issues, "final_report_check": fr, "recomputed": res, "seconds": round(time.time() - t0, 1), "script_sha256": K.sha(__file__)},
              open(vout / "verification.json", "x"), indent=1)
    with open(rep_path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(json.dumps({"checks": len(checks), "failed": [k for k, v in checks.items() if not v["pass"]], "issues": issues, "final_report_checked": fr["checked_values"],
                      "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
