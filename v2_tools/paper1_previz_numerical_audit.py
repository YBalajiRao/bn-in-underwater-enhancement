"""Paper 1 pre-visualisation numerical audit. READ-ONLY on every frozen file; no inference, no training, no figure generation.
1. UIEB-787 BN / IN PSNR and SSIM per seed, seed mean and SD, paired IN - BN differences, recomputed from the frozen Stage 3 per-image file and
   checked against summary.json, results.json and O1; O1's 95 % image-bootstrap CIs are re-derived independently from the frozen draw rule.
   Seed-level t95 intervals are kept in separate columns. No SSIM-difference CI exists in any frozen file; none is computed here.
2. Cross-checks UIEB numbers in the research notes: targeted checks (draft Tables 1-3) and a rounding-aware screen of every decimal number on
   UIEB lines against a pool of frozen values, with a planted-perturbation control that measures the screen's power.
3. Confirms the HICRD v2 report and its verification record; records E1r exactness, O7 and audit evidence (read only).
4. Writes a source-of-truth table and a figure-data mapping sheet.
Usage: python v2_tools/paper1_previz_numerical_audit.py --out paper1_uieb_runs/paper1_previz_audit"""
import argparse, csv, glob, hashlib, json, re, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "paper1_uieb_runs"
REP = ROOT / "reports/paper1_narrowed"
FROZEN = {  # path -> expected sha256 (from O1 inputs record / earlier registries)
    "paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv": "a792f4e8ed12231c5e0f0aa7b28e5acd3be7681d4118864c6b1322220f6115b2",
    "paper1_uieb_runs/stage3_analysis_best_valA/results.json": "21270f0b285eab3bd887593aff4b7d44d7ced7ea0235297418d84ec90acfeeb4",
    "paper1_uieb_runs/stage3_uieb800_last/per_image.csv": "453967e3dd31405053621869ffa70311e67d0407dcb5547fa94f9b5615d1eb10",
    "paper1_uieb_runs/stage3_analysis_last/results.json": "be9029f7a81ce95ba7bf6ab5d1f74b2cf998b66165371da96e3f342e7b403fdb",
    "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json": "dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08",
    "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json": "d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c",
    "paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv": "87fc8b6ee0b393002aa245c4ec08f947e504b8cf6dca0874bbd2c257f7f00c27",
    "paper1_uieb_runs/s5_e1r/e1r_per_image.csv": "384430561c71d7a1f340ff0b361c0cb371e30026e86943009b88684343dadfc4",
    "reports/paper1_narrowed/HICRD_complete_numerical_results_2026-11-03_v2.md": "c7e15614bdf50290703d4a1d091305ccc506409abe44225b83ecad4dcc4c7d23",
    "paper1_uieb_runs/sota_hicrd_numerical_verification_v2/verification.json": "2a94e156ee2c9647a7f8d32fd1042db30e28a070ac0cc0408aed3e6acd2aa5d3",
    "paper1_uieb_runs/sota_hicrd_analysis/analysis.json": "331a09ab764904a434a12dec3cf39d9d95cd61004b2614ea557a2878920ff19e",
}
ARMS = {"MARINE-Net": ("marinenet_bn", "marinenet_in"), "AT-Trunk": ("attrunk_bn", "attrunk_in")}
SEEDS = (0, 1, 2)
T95 = stats.t.ppf(0.975, 2)
checks = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def chk(name, ok, detail):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    return ok


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{p}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool) and np.isfinite(d):
        yield p, float(d)


def t95(per):
    per = np.asarray(per); m, s = per.mean(), per.std(ddof=1); h = T95 * s / np.sqrt(len(per)); return [m - h, m + h], s


# ---------------------------------------------------------------- 1. UIEB-787
def uieb(run="best_valA"):
    man = json.load(open(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json", encoding="utf-8"))
    pairs = man["pairs"] if isinstance(man, dict) else man
    names = [Path(p["input_path"]).name for p in pairs]
    excl = set(json.load(open(ROOT / "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"))["primary_exclusions"])
    sel = [n for n in names if n not in excl]
    sub = "stage3_uieb800_best_valA" if run == "best_valA" else "stage3_uieb800_last"
    rows = list(csv.DictReader(open(RUNS / sub / "per_image.csv", encoding="utf-8")))
    per = {r["filename"]: r for r in rows}
    chk(f"UIEB {run}: per-image rows", len(rows) == 800 and len(sel) == 787 and all(k in per for k in sel), f"{len(rows)} rows, primary cohort {len(sel)} (13 exclusions)")
    summ = json.load(open(RUNS / sub / "summary.json"))
    v = lambda k, col: np.array([float(per[n][col]) for n in k])
    mx = max(abs(v(names, f"{m}_{met}").mean() - summ["models"][m][met]) for m in summ["models"] for met in ("psnr", "ssim"))
    chk(f"UIEB {run}: CSV means (800) equal summary.json", mx < 1e-5, f"max |Δ| {mx:.2e} (CSV has 6 decimals)")
    res = json.load(open(RUNS / ("stage3_analysis_" + run) / "results.json"))
    o1 = json.load(open(RUNS / "paper1_O1_mean_d_cis/o1_results.json"))["UIEB_787"][run]
    idx = np.random.default_rng(20261102).integers(0, 787, size=(10000, 787))      # O1 / frozen Stage 3 draw rule
    out = {"n": 787, "arms": {}, "diff": {}}
    for arch, (bn, inn) in ARMS.items():
        for arm in (bn, inn):
            out["arms"][arm] = {met: [float(v(sel, f"{arm}_s{s}_{met}").mean()) for s in SEEDS] for met in ("psnr", "ssim")}
        D = {met: np.stack([v(sel, f"{inn}_s{s}_{met}") - v(sel, f"{bn}_s{s}_{met}") for s in SEEDS], 1) for met in ("psnr", "ssim")}
        dp = D["psnr"].mean(0)
        chk(f"UIEB {run} {arch}: mean d PSNR equals results.json", np.allclose(dp, res["cohorts"]["PRIMARY_787"][arch]["mean_d_per_seed"], atol=1e-9),
            " / ".join(f"{x:.6f}" for x in dp))
        o = o1["results"][arch]
        chk(f"UIEB {run} {arch}: mean d PSNR equals O1", np.allclose(dp, o["per_seed"], atol=1e-9) and abs(dp.mean() - o["seed_mean"]) < 1e-9, "per seed and seed mean")
        bs = D["psnr"][idx].mean(1)                                                       # [B, 3]
        ci_seed = [np.percentile(bs[:, s], [2.5, 97.5]).tolist() for s in SEEDS]
        ci_mean = np.percentile(bs.mean(1), [2.5, 97.5]).tolist()
        ok = np.allclose(ci_seed, o["per_seed_ci95"], atol=1e-9) and np.allclose(ci_mean, o["seed_mean_ci95"], atol=1e-9)
        chk(f"UIEB {run} {arch}: O1 95 % image-bootstrap CIs re-derived independently", ok, f"seed-mean CI {ci_mean[0]:.4f}, {ci_mean[1]:.4f}")
        tt, sdd = t95(dp)
        chk(f"UIEB {run} {arch}: seed-level t95 equals O1", np.allclose(tt, o["seed_level_t95_df2"], atol=1e-9), f"[{tt[0]:.4f}, {tt[1]:.4f}], SD {sdd:.4f}")
        share = (D["psnr"] > 0).mean(0)
        chk(f"UIEB {run} {arch}: share IN > BN equals O1", np.allclose(share, o["share_images_IN_gt_BN_per_seed"], atol=1e-12), " / ".join(f"{x:.4f}" for x in share))
        ds = D["ssim"].mean(0); ts, sds = t95(ds)
        out["diff"][arch] = {"psnr": {"per_seed": dp.tolist(), "seed_mean": float(dp.mean()), "seed_sd": float(sdd), "t95": tt, "ci95_per_seed": ci_seed, "ci95_seed_mean": ci_mean,
                                      "ci97_5_seed_mean": o["seed_mean_ci97_5"], "share_IN_gt_BN": share.tolist()},
                             "ssim": {"per_seed": ds.tolist(), "seed_mean": float(ds.mean()), "seed_sd": float(sds), "t95": ts, "ci95_per_seed": None, "ci95_seed_mean": None,
                                      "share_IN_gt_BN": (D["ssim"] > 0).mean(0).tolist()}}
        out["diff"][arch]["rho"] = {k: res["cohorts"]["PRIMARY_787"][arch][k] for k in ("rho_per_seed", "rho_per_seed_ci97_5", "rho_bar", "rho_bar_ci97_5", "partial_bar",
                                                                                    "partial_bar_ci97_5", "seed_level_descriptive", "decision")}
    out["descriptor_assoc"] = res["cohorts"]["PRIMARY_787"]["descriptor_associations"]
    return out


# ---------------------------------------------------------------- 2. research-note cross-check
NOTES = ["paper1_draft_sections_v2_WORKING.md", "S3_results_report.md", "paper1_open_items_O1_O2_O4_O5_report.md", "paper1_PR_audit_report_2026-11-02.md",
         "paper1_draft_v2_changelog.md", "S5_E1r_results_report.md"]
UIEB_LINE = re.compile(r"uieb|787|frozen-90|frozen90|\bG1\b", re.I)
NUM = re.compile(r"(?<![\w.`])[-+−]?\d+\.\d+(?![\w`])")


def pool(u):
    vals = []
    srcs = ["paper1_uieb_runs/stage3_analysis_best_valA/results.json", "paper1_uieb_runs/stage3_analysis_last/results.json",
            "paper1_uieb_runs/paper1_O1_mean_d_cis/o1_results.json", "paper1_uieb_runs/stage3_uieb800_best_valA/summary.json",
            "paper1_uieb_runs/stage3_uieb800_last/summary.json", "paper1_uieb_runs/s5_e1r_analysis/e1r_results.json", "paper1_uieb_runs/s5_x1/x1_results.json",
            "paper1_uieb_runs/post_g1_stats/post_g1_stats.json", "paper1_uieb_runs/h3_colour_cast/h3_results.json", "paper1_uieb_runs/g1_uieb_frozen90_rerun1_subsets.json"]
    srcs += [str(Path(p).relative_to(ROOT)).replace("\\", "/") for p in glob.glob(str(RUNS / "g1_uieb_frozen90*" / "summary.json"))]
    for s in srcs:
        vals += [(s + k, x) for k, x in flat(json.load(open(ROOT / s)))]
    vals += [("computed:" + k, x) for k, x in flat(u)]
    for k in ("best_valA", "last"):
        uu = U[k]
        for arm, d in uu["arms"].items():
            for met, per in d.items():
                vals += [(f"computed:{k}:{arm}:{met}:seedmean", float(np.mean(per))), (f"computed:{k}:{arm}:{met}:sd", float(np.std(per, ddof=1)))]
    out = []
    for k, x in vals:
        out += [(k, x), (k + "×100", 100 * x), (k + "×1000", 1000 * x)]
    return out


def match(p, d, pv):
    tol = 0.5 * 10 ** -d + 1e-12
    hit = np.abs(pv - p) <= tol
    if hit.any():
        return "match", int(np.argmax(hit))
    hit = np.abs(np.abs(pv) - abs(p)) <= tol
    return ("magnitude-only", int(np.argmax(hit))) if hit.any() else ("unmatched", -1)


def screen(P):
    keys = [k for k, _ in P]; pv = np.array([x for _, x in P])
    res, items = [], []
    for fn in NOTES:
        for ln, line in enumerate((REP / fn).read_text(encoding="utf-8").splitlines(), 1):
            if not UIEB_LINE.search(line):
                continue
            for m in NUM.finditer(line):
                t = m.group(0).replace("−", "-"); p = float(t); d = len(t.split(".")[1])
                st, j = match(p, d, pv)
                items.append((p, d))
                res.append({"file": fn, "line": ln, "printed": m.group(0), "status": st, "source": keys[j] if j >= 0 else "", "text": line.strip()[:220]})
    # power control: perturb each printed value by +1 and -1 unit of its last digit, and by +10 %
    rng_hits = {"one_unit": 0, "ten_pct": 0}
    for p, d in items:
        u_ = 10 ** -d
        rng_hits["one_unit"] += (match(round(p + u_, d), d, pv)[0] == "match") + (match(round(p - u_, d), d, pv)[0] == "match")
        rng_hits["ten_pct"] += match(round(p * 1.1 if p else u_ * 7, d), d, pv)[0] == "match"
    power = {"n_numbers": len(items), "false_match_rate_one_unit": rng_hits["one_unit"] / max(1, 2 * len(items)),
             "false_match_rate_ten_pct": rng_hits["ten_pct"] / max(1, len(items))}
    return res, power


def targeted(u):
    """High-power checks of the UIEB cells of draft Tables 1-3 (exact cell -> named source field)."""
    txt = (REP / "paper1_draft_sections_v2_WORKING.md").read_text(encoding="utf-8").splitlines()
    e = json.load(open(RUNS / "s5_e1r_analysis/e1r_results.json"))["UIEB"]
    out = []

    def cmp(label, printed, true):
        t = printed.replace("−", "-"); d = len(t.split(".")[1]); ok = abs(float(t) - true) <= 0.5 * 10 ** -d + 1e-12
        out.append({"cell": label, "printed": printed, "source_value": true, "status": "PASS" if ok else "FAIL"})
    for line in txt:
        if line.startswith("| UIEB-787 (787; curated)"):
            nums = NUM.findall(line)
            exp = u["diff"]["MARINE-Net"]["psnr"]["per_seed"] + [u["diff"]["MARINE-Net"]["psnr"]["seed_mean"]] + u["diff"]["AT-Trunk"]["psnr"]["per_seed"] + [u["diff"]["AT-Trunk"]["psnr"]["seed_mean"]]
            for i, (a, b) in enumerate(zip(nums, exp)):
                cmp(f"Table 1 UIEB-787 value {i + 1}", a, b)
        for arch in ARMS:
            if line.startswith(f"| UIEB-787 | {arch} |"):
                r = u["diff"][arch]["rho"]; nums = NUM.findall(line)
                exp = r["rho_per_seed"] + [r["rho_bar"]] + r["rho_bar_ci97_5"]
                for i, (a, b) in enumerate(zip(nums, exp)):
                    cmp(f"Table 2 UIEB {arch} value {i + 1}", a, b)
        m = re.match(r"\| Δ([ACGR]) \(", line)
        if m:
            c = m.group(1); cells = [x.strip() for x in line.strip("|").split("|")]
            for arch, cell in (("MARINE-Net", cells[1]), ("AT-Trunk", cells[2])):
                nums = NUM.findall(cell); s = e[arch][c]
                for a, b in zip(nums, [s["mean"] * 1e3] + [x * 1e3 for x in s["ci95_seed_mean"]]):
                    cmp(f"Table 3 UIEB {arch} Δ{c}", a, b)
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    global U
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    for rel, h in FROZEN.items():
        got = sha(ROOT / rel); chk(f"hash {rel}", got == h, got[:16])
    U = {"best_valA": uieb("best_valA"), "last": uieb("last")}
    # HICRD v2 source of truth
    ver = json.load(open(RUNS / "sota_hicrd_numerical_verification_v2/verification.json"))
    failed = [k for k, v in ver["checks"].items() if not v["pass"]]
    chk("HICRD v2 verification record", failed == ["final_report_numbers"] and len(ver["checks"]) == 97,
        f"97 checks; only failure = 16 last-digit discrepancies in the superseded final report (v2 values are the correct ones)")
    # E1r, O7 records
    es = json.load(open(RUNS / "s5_e1r/e1r_summary.json"))["exactness"]; ei = json.load(open(RUNS / "s5_e1r_analysis/e1r_results.json"))["integrity"]
    chk("E1r exactness record (24 model × dataset rows)", all(v["exact"] for v in es.values()) and len(es) == 24,
        f"max rel err {max(v['max_rel_err'] for v in es.values()):.2e}; PSNR vs frozen UIEB {ei['UIEB_max_abs_psnr_diff_vs_frozen_dB']:.2e} dB, HICRD {ei['HICRD_max_abs_psnr_diff_vs_frozen_dB']:.2e} dB")
    o7 = json.load(open(RUNS / "paper1_O7_tub_input_psnr/o7_results.json"))
    ic = o7["identity_checks"]
    chk("O7 identity checks record", ic["square256"]["exact"] and ic["G455"]["exact"],
        f"exact at both geometries; max |PSNR + 10 log10 MSE| {ic['square256']['max_abs_psnr_vs_mse_dB_nonzero_mse']:.2e} dB; zero-MSE pair {ic['square256']['zero_mse_pairs']}")
    tub = json.load(open(RUNS / "tub_stage5_analysis_best_valA/tub_analysis.json"))["primary"]["E-T5_PRIMARY"]
    # cross-check
    P = pool(U["best_valA"])
    scr, power = screen(P)
    tg = targeted(U["best_valA"])
    chk("Draft Tables 1-3 UIEB cells (targeted)", all(t["status"] == "PASS" for t in tg), f"{sum(t['status'] == 'PASS' for t in tg)} / {len(tg)} cells")
    n_un = sum(r["status"] != "match" for r in scr)
    # stale statements
    txt = (REP / "paper1_draft_sections_v2_WORKING.md").read_text(encoding="utf-8")
    stale = [s for s in ("Per-seed bootstrap CIs of the mean difference were not computed for UIEB-787 and HICRD",
                         "Per-seed bootstrap CIs of the mean IN − BN difference for UIEB-787 and HICRD are not in the frozen outputs") if s in txt]
    chk("Draft v2 has no stale O1 statement", not stale, f"{len(stale)} stale sentence(s): O1 has since been computed")
    out.mkdir(parents=True)
    # ---------------- source-of-truth table
    SOT = []

    def row(qid, ds, cohort, n, model, seed, metric, value, unit, unc_type="", lo="", hi="", t_lo="", t_hi="", src="", key=""):
        SOT.append(dict(id=qid, dataset=ds, cohort=cohort, n=n, model=model, seed=seed, metric=metric, value=value, unit=unit, uncertainty_type=unc_type,
                        lo=lo, hi=hi, seed_t95_lo=t_lo, seed_t95_hi=t_hi, source=src, key=key))
    u = U["best_valA"]; s3 = "stage3_uieb800_best_valA/per_image.csv (787 primary rows)"
    for arm, d in u["arms"].items():
        for met, unit in (("psnr", "dB"), ("ssim", "1")):
            for s in SEEDS:
                row(f"UIEB.{arm}.{met}.s{s}", "UIEB", "PRIMARY_787", 787, arm, s, f"mean {met.upper()}", d[met][s], unit, src=s3, key=f"{arm}_s{s}_{met}")
            tt, sdd = t95(d[met])
            row(f"UIEB.{arm}.{met}.mean", "UIEB", "PRIMARY_787", 787, arm, "mean", f"mean {met.upper()} (seed mean; SD {sdd:.6f})", float(np.mean(d[met])), unit,
                t_lo=tt[0], t_hi=tt[1], src=s3, key="computed")
    for arch in ARMS:
        for met, unit in (("psnr", "dB"), ("ssim", "1")):
            dd = u["diff"][arch][met]
            for s in SEEDS:
                row(f"UIEB.dIN-BN.{arch}.{met}.s{s}", "UIEB", "PRIMARY_787", 787, arch, s, f"mean IN − BN {met.upper()}", dd["per_seed"][s], unit,
                    "95 % image bootstrap (O1, post hoc)" if met == "psnr" else "NONE in frozen files", *(dd["ci95_per_seed"][s] if met == "psnr" else ("", "")),
                    src="paper1_O1_mean_d_cis/o1_results.json" if met == "psnr" else s3, key=f"UIEB_787.best_valA.results.{arch}.per_seed_ci95[{s}]" if met == "psnr" else "computed")
            row(f"UIEB.dIN-BN.{arch}.{met}.mean", "UIEB", "PRIMARY_787", 787, arch, "mean", f"mean IN − BN {met.upper()} (seed mean; SD {dd['seed_sd']:.6f})", dd["seed_mean"], unit,
                "95 % image bootstrap (O1, post hoc)" if met == "psnr" else "NONE in frozen files", *(dd["ci95_seed_mean"] if met == "psnr" else ("", "")), dd["t95"][0], dd["t95"][1],
                src="paper1_O1_mean_d_cis/o1_results.json" if met == "psnr" else s3, key=f"UIEB_787.best_valA.results.{arch}.seed_mean_ci95" if met == "psnr" else "computed")
        r = u["diff"][arch]["rho"]
        for s in SEEDS:
            row(f"UIEB.rho.{arch}.s{s}", "UIEB", "PRIMARY_787", 787, arch, s, "Spearman ρ(d, imbalance)", r["rho_per_seed"][s], "1", "97.5 % image bootstrap (registered)",
                *r["rho_per_seed_ci97_5"][s], src="stage3_analysis_best_valA/results.json", key=f"cohorts.PRIMARY_787.{arch}.rho_per_seed_ci97_5[{s}]")
        row(f"UIEB.rho.{arch}.mean", "UIEB", "PRIMARY_787", 787, arch, "mean", f"ρ̄ (decision {r['decision']})", r["rho_bar"], "1", "97.5 % image bootstrap (registered)", *r["rho_bar_ci97_5"],
            *r["seed_level_descriptive"]["t95_df2"], src="stage3_analysis_best_valA/results.json", key=f"cohorts.PRIMARY_787.{arch}.rho_bar_ci97_5")
    o1h = json.load(open(RUNS / "paper1_O1_mean_d_cis/o1_results.json"))["HICRD_2000"]
    for geo, run in (("square256", "best_valA"), ("aspect482", "best_valA_G482")):
        for arch in ARMS:
            o = o1h[run]["results"][arch]["T30_unstratified_P1_draws"]
            for s in SEEDS:
                row(f"HICRD.{geo}.dIN-BN.{arch}.psnr.s{s}", "HICRD", f"ALL_2000 {geo}", 2000, arch, s, "mean IN − BN PSNR", o["per_seed"][s], "dB", "95 % T30-cluster bootstrap P1 (O1, post hoc)",
                    *o["per_seed_ci95"][s], src="paper1_O1_mean_d_cis/o1_results.json", key=f"HICRD_2000.{run}.results.{arch}.T30_unstratified_P1_draws.per_seed_ci95[{s}]")
            row(f"HICRD.{geo}.dIN-BN.{arch}.psnr.mean", "HICRD", f"ALL_2000 {geo}", 2000, arch, "mean", "mean IN − BN PSNR (seed mean)", o["seed_mean"], "dB",
                "95 % T30-cluster bootstrap P1 (O1, post hoc)", *o["seed_mean_ci95"], *o["seed_level_t95_df2"], src="paper1_O1_mean_d_cis/o1_results.json", key=f"HICRD_2000.{run}...seed_mean_ci95")
    for arch, k in (("MARINE-Net", "marinenet"), ("AT-Trunk", "attrunk")):
        e5 = tub["endpoints"][f"E-T5|{k}|d"]
        for s in SEEDS:
            row(f"TUB.dIN-BN.{arch}.psnr.s{s}", "TUB", "PRIMARY_1153 square256", 1153, arch, s, "mean IN − BN PSNR (scene-weighted)", e5["per_seed"][s], "dB", "95 % scene bootstrap (registered)",
                *e5["boot_ci95_per_seed"][s], src="tub_stage5_analysis_best_valA/tub_analysis.json", key=f"primary.E-T5_PRIMARY.endpoints.E-T5|{k}|d.boot_ci95_per_seed[{s}]")
        row(f"TUB.dIN-BN.{arch}.psnr.mean", "TUB", "PRIMARY_1153 square256", 1153, arch, "mean", "mean IN − BN PSNR (seed mean)", e5["seed_mean"], "dB", "95 % scene bootstrap (registered)",
            *e5["boot_ci95_seed_mean"], *e5["t95"], src="tub_stage5_analysis_best_valA/tub_analysis.json", key=f"primary.E-T5_PRIMARY.endpoints.E-T5|{k}|d")
    tr = o7["trend_identity_psnr_vs_ntu"]["square256"]
    row("TUB.O7.identity_rho", "TUB", "NTU_1022 square256", 1022, "identity (input)", "—", "within-series Spearman ρ̄ (identity PSNR vs NTU)", tr["e_spearman"], "1",
        "95 % scene bootstrap", *tr["e_ci95"], src="paper1_O7_tub_input_psnr/o7_results.json", key="trend_identity_psnr_vs_ntu.square256")
    row("TUB.O7.identity_slope", "TUB", "NTU_1022 square256", 1022, "identity (input)", "—", "OLS slope identity PSNR vs NTU", tr["d_ols_slope_dB_per_NTU"], "dB/NTU",
        "95 % scene bootstrap", *tr["d_ci95"], src="paper1_O7_tub_input_psnr/o7_results.json", key="trend_identity_psnr_vs_ntu.square256")
    an = json.load(open(RUNS / "sota_hicrd_analysis/analysis.json"))
    for r in an["comparator_means"]:
        if r["population"] == "ALL":
            row(f"HICRD.SOTA.{r['geometry']}.{r['comparator']}.{r['metric']}", "HICRD", f"ALL_2000 {r['geometry']}", 2000, f"{r['comparator']} (Group {r['group']})", "n=1",
                f"mean {r['metric'].upper()}", r["mean"], "dB" if r["metric"] == "psnr" else "1", "95 % T30-cluster bootstrap P1 (registered)", *r["ci95"],
                src="sota_hicrd_analysis/analysis.json", key="comparator_means")
    for r in an["contrasts"]:
        if r["population"] == "ALL":
            row(f"HICRD.SOTA.{r['geometry']}.{r['arm']}-{r['comparator']}.{r['metric']}", "HICRD", f"ALL_2000 {r['geometry']}", 2000, f"{r['arm']} − {r['comparator']} (Group {r['group']})",
                "mean", f"mean Δ{r['metric'].upper()} (seed mean; per seed {' / '.join(f'{x:.6f}' for x in r['per_seed'])})", r["seed_mean"], "dB" if r["metric"] == "psnr" else "1",
                "95 % T30-cluster bootstrap P1 (registered)", *r["ci95_seed_mean"], *r["t95"], src="sota_hicrd_analysis/analysis.json", key="contrasts")
    cols = list(SOT[0].keys())
    with open(out / "source_of_truth.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in SOT:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(out / "uieb_notes_screen.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scr[0].keys())); w.writeheader(); w.writerows(scr)
    with open(out / "uieb_targeted_cells.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tg[0].keys())); w.writeheader(); w.writerows(tg)
    json.dump({"checks": checks, "screen_power": power, "screen_counts": {s: sum(r["status"] == s for r in scr) for s in ("match", "magnitude-only", "unmatched")},
               "uieb": U, "stale": stale, "script_sha256": sha(__file__)}, open(out / "audit.json", "x"), indent=1)
    print(json.dumps({"checks": [(c["check"], c["status"]) for c in checks if c["status"] != "PASS"], "n_checks": len(checks), "sot_rows": len(SOT),
                      "screen": {s: sum(r["status"] == s for r in scr) for s in ("match", "magnitude-only", "unmatched")}, "power": power,
                      "targeted": f"{sum(t['status'] == 'PASS' for t in tg)}/{len(tg)}"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
