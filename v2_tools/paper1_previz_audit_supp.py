"""Supplement to paper1_previz_numerical_audit.py (read-only): targeted checks of the UIEB claims outside draft Tables 1-3, hashes of the completed
audit records, and the figure-data mapping sheet. The first script's rounding screen had little power (one-unit perturbations still matched 89 %
of the time), so every UIEB claim here is checked against its NAMED source field instead.
Usage: python v2_tools/paper1_previz_audit_supp.py --out paper1_uieb_runs/paper1_previz_audit_supp"""
import argparse, csv, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "paper1_uieb_runs"; REP = ROOT / "reports/paper1_narrowed"
DRAFT = REP / "paper1_draft_sections_v2_WORKING.md"
NUM = re.compile(r"[-+−]?\d+\.\d+")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    lines = DRAFT.read_text(encoding="utf-8").splitlines()
    pg = json.load(open(RUNS / "post_g1_stats/post_g1_stats.json"))["best_valA"]["full90 (registered primary)"]
    x1 = json.load(open(RUNS / "s5_x1/x1_results.json")); e1 = json.load(open(RUNS / "s5_e1r_analysis/e1r_results.json"))
    s3 = json.load(open(RUNS / "stage3_analysis_best_valA/results.json"))["cohorts"]["PRIMARY_787"]
    s4 = json.load(open(RUNS / "stage4_analysis_best_valA/results.json"))
    info = x1["X1c_info"]; cnt = info["stratum_counts_uieb_hicrd"]; mn = info["min_n_per_dataset"]
    common = [v for v in cnt.values() if v[0] >= mn and v[1] >= mn]
    cov_u = sum(v[0] for v in common) / sum(v[0] for v in cnt.values()); cov_h = sum(v[1] for v in common) / sum(v[1] for v in cnt.values())
    m, t = pg["H1_marinenet_in_minus_bn"], pg["H1_attrunk_in_minus_bn"]
    # (draft line text fragment, ordered expected values, source description)
    targets = [
        ("| UIEB frozen-90 (90; curated)", m["per_seed_mean"] + [x for ci in m["per_seed_boot95_images"] for x in ci] + [m["seed_mean"]] + m["seed_level"]["t95"]
         + t["per_seed_mean"] + [t["seed_mean"]] + t["seed_level"]["t95"], "post_g1_stats.json best_valA/full90 H1_*"),
        ("seed-mean Spearman ρ̄ = 0.30 and 0.18", [s3["MARINE-Net"]["rho_bar"], s3["AT-Trunk"]["rho_bar"]], "stage3 results.json rho_bar"),
        ("UIEB 0.155 [", [x1["X1ab_UIEB"]["g_chr"]["MARINE-Net"]["mean"], *x1["X1ab_UIEB"]["g_chr"]["MARINE-Net"]["ci95"],
                          x1["X1ab_UIEB"]["g_chr"]["AT-Trunk"]["mean"], *x1["X1ab_UIEB"]["g_chr"]["AT-Trunk"]["ci95"]], "x1_results X1ab_UIEB g_chr"),
        ("UIEB AT-Trunk −0.105", [x1["X1ab_UIEB"]["g_ach"]["AT-Trunk"]["mean"], *x1["X1ab_UIEB"]["g_ach"]["AT-Trunk"]["ci95"]], "x1_results X1ab_UIEB g_ach AT-Trunk"),
        ("UIEB MARINE-Net −0.052", [x1["X1ab_UIEB"]["g_ach"]["MARINE-Net"]["mean"], *x1["X1ab_UIEB"]["g_ach"]["MARINE-Net"]["ci95"]], "x1_results X1ab_UIEB g_ach MARINE-Net"),
        ("median reference imbalance 0.045 vs 0.118", [x1["descriptor_summaries"]["HICRD"]["x_ref"][1], x1["descriptor_summaries"]["UIEB"]["x_ref"][1]],
         "x1_results descriptor_summaries x_ref[1] (median), HICRD then UIEB"),
        ("Imbalance and mean luma had ρ = 0.989 on HICRD, against −0.24 on UIEB", [s4["SUPP"]["rho_x_b"], s3["descriptor_associations"]["rho_x_luma_mean"]],
         "stage4 results SUPP.rho_x_b; stage3 descriptor_associations.rho_x_luma_mean"),
        ("consistent only for UIEB AT-Trunk (ρ = −0.182)", [e1["UIEB"]["AT-Trunk"]["secondary"]["dC_vs_g_chr"]["mean"]], "e1r_results UIEB/AT-Trunk/secondary/dC_vs_g_chr"),
    ]
    rows = []
    for frag, exp, src in targets:
        hits = [ln for ln in lines if frag in ln]
        if len(hits) != 1:
            rows.append({"claim": frag, "printed": "", "source_value": "", "status": f"FAIL (line found {len(hits)} times)", "source": src}); continue
        nums = [n for n in NUM.findall(hits[0])]
        if frag.startswith("| UIEB frozen-90"):
            nums = [n for n in nums if n not in ("90",)]
        if frag.startswith("seed-mean"):
            nums = nums[:2]
        if frag.startswith("Imbalance and mean luma"):
            nums = nums[:2]
        if len(nums) < len(exp):
            rows.append({"claim": frag, "printed": " ".join(nums), "source_value": "", "status": f"FAIL ({len(nums)} numbers, {len(exp)} expected)", "source": src}); continue
        for p, v in zip(nums, exp):
            q = p.replace("−", "-"); d = len(q.split(".")[1]); ok = abs(float(q) - v) <= 0.5 * 10 ** -d + 1e-12
            rows.append({"claim": frag, "printed": p, "source_value": f"{v:.10g}", "status": "PASS" if ok else "FAIL", "source": src})
    # 42 % / 19 % coverage (integers; recomputed from the X1 stratum counts)
    hit = [ln for ln in lines if "covers 42 % of UIEB and 19 % of HICRD" in ln]
    for lab, pv, v in (("coverage UIEB", 42, cov_u), ("coverage HICRD", 19, cov_h)):
        rows.append({"claim": lab, "printed": f"{pv} %", "source_value": f"{100 * v:.4f} %", "status": "PASS" if hit and abs(100 * v - pv) <= 0.5 else "FAIL",
                     "source": f"x1_results X1c_info stratum_counts (strata with both counts ≥ {mn})"})
    out.mkdir(parents=True)
    with open(out / "uieb_targeted_claims_supp.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # completed audit / evidence records (hashes only; read, not rerun)
    recs = ["reports/paper1_narrowed/S5_E1r_results_report.md", "reports/paper1_narrowed/S5_E1r_implementation_freeze_addendum.md", "paper1_uieb_runs/s5_e1r/e1r_summary.json",
            "paper1_uieb_runs/s5_e1r_analysis/e1r_results.json", "paper1_uieb_runs/paper1_O7_tub_input_psnr/O7_report.md", "paper1_uieb_runs/paper1_O7_tub_input_psnr/o7_results.json",
            "reports/paper1_narrowed/TUB_dataset_audit_and_suitability_report.md", "reports/paper1_narrowed/paper1_PR_audit_report_2026-11-02.md",
            "reports/paper1_narrowed/paper1_open_items_O1_O2_O4_O5_report.md", "reports/paper1_narrowed/S3_results_report_erratum.md",
            "reports/paper1_narrowed/SOTA_final_protocol_audit_report.md", "reports/paper1_narrowed/SOTA_HICRD_final_report_2026-11-03.md",
            "reports/paper1_narrowed/HICRD_complete_numerical_results_2026-11-03_v2.md", "paper1_uieb_runs/sota_hicrd_numerical_verification_v2/verification.json",
            "paper1_uieb_runs/sota_hicrd_post_exec_check_post_T482/post_exec_check.json", "paper1_uieb_runs/paper1_O1_mean_d_cis/o1_results.json",
            "paper1_uieb_runs/tub_stage5_analysis_best_valA/tub_analysis.json", "paper1_uieb_runs/paper1_previz_audit/audit.json",
            "paper1_uieb_runs/paper1_previz_audit/source_of_truth.csv"]
    with open(out / "evidence_hashes.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "sha256"]); [w.writerow([r, sha(ROOT / r)]) for r in recs]
    # figure-data mapping sheet
    F = [
        ("Fig. 1", "UIEB-787 panel", "mean IN − BN PSNR per seed + seed mean", "dB", "PRIMARY_787 (n = 787), best_valA and last", "95 % image bootstrap (O1, post hoc, default_rng(20261102), B = 10,000); seed t95 df 2 shown separately", "paper1_O1_mean_d_cis/o1_results.json UIEB_787.{run}.results.{arch}", "ready"),
        ("Fig. 1", "UIEB frozen-90 panel", "mean IN − BN PSNR per seed", "dB", "G1 frozen-90 (n = 90; first 90 filenames, not random)", "95 % image bootstrap per seed (registered post-G1); seed t95", "post_g1_stats/post_g1_stats.json best_valA.full90.H1_*", "ready"),
        ("Fig. 1", "HICRD panel", "mean IN − BN PSNR per seed + seed mean", "dB", "ALL 2,000; square256 and G482; best_valA and last", "95 % T30-cluster bootstrap P1 (O1, post hoc; Stage 4 draws); seed t95", "paper1_O1_mean_d_cis/o1_results.json HICRD_2000.{run}", "ready"),
        ("Fig. 1", "TUB panel", "mean IN − BN PSNR per seed + seed mean (scene-weighted)", "dB", "PRIMARY 1,153 pairs, 33 scenes; square256 (G455 in tub_stage5_analysis_*_G455)", "95 % scene bootstrap (registered); seed t95", "tub_stage5_analysis_best_valA/tub_analysis.json primary.E-T5_PRIMARY.endpoints.E-T5|{arch}|d", "ready"),
        ("Fig. 2", "forest of per-seed ρ", "Spearman ρ(d, channel-mean imbalance) per seed and ρ̄", "1", "UIEB-787; HICRD ALL 2,000 (and sensitivity cohorts as Table 2)", "UIEB 97.5 % image bootstrap; HICRD 97.5 % cluster bootstrap (registered) — label the level; do NOT mix with 95 % intervals", "stage3_analysis_best_valA/results.json cohorts.*.{arch}.rho_*; stage4_analysis_best_valA/results.json", "ready"),
        ("Fig. 3", "per-image scatter", "per-image d (IN − BN PSNR) vs input imbalance", "dB vs 1", "UIEB-787; HICRD 2,000 coloured by site", "none (points); decide: seed-mean d or one panel per seed", "stage3_uieb800_best_valA/per_image.csv + stage3_analysis_best_valA/descriptors_per_image.csv; stage4 per_image.csv + descriptors_per_image.csv", "DECISION NEEDED (seed display)"),
        ("Fig. 4", "descriptor collinearity", "imbalance vs mean luma per image", "1 vs 1", "HICRD 2,000 (ρ = 0.98947); UIEB-787 (ρ = −0.24101)", "none (ρ annotated)", "stage4_analysis_best_valA/results.json SUPP.rho_x_b; stage3 results.json descriptor_associations", "ready"),
        ("Fig. 5", "E1r components", "ΔA, ΔC, ΔG, ΔR (IN − BN) seed mean + per seed", "×10⁻³ MSE", "UIEB-787 (image), HICRD 2,000 (T30 cluster), TUB PRIMARY 1,153 (scene)", "95 % bootstrap by unit as stated; seed t95 separate", "s5_e1r_analysis/e1r_results.json {UIEB,HICRD}.{arch}.{A,C,G,R}; tub_analysis.json E-T5 components", "ready"),
        ("Fig. 6a", "TUB d vs NTU", "per-pair d with series lines and scene means", "dB vs NTU", "NTU cohort 1,022 pairs, 29 scenes, 114 series", "within-series ρ̄ with 95 % scene bootstrap (registered)", "tub_stage5_analysis_best_valA/tub_endpoints.csv; tub_analysis.json primary.trend_NTU", "ready"),
        ("Fig. 6b", "TUB components vs NTU", "ΔA, ΔG, ΔR vs NTU", "×10⁻³ MSE vs NTU", "NTU cohort", "95 % scene bootstrap (registered)", "tub_analysis.json primary.trend_NTU", "ready"),
        ("Fig. 6c", "TUB per-arm PSNR vs NTU and identity", "BN, IN and do-nothing PSNR by NTU decile", "dB", "NTU cohort 1,022 (post hoc deciles)", "95 % scene bootstrap (O2/O7, post hoc)", "paper1_O2_tub_psnr_vs_ntu/*; paper1_O7_tub_input_psnr/o7_results.json decile_curves_square256_seedmean", "ready (post hoc; label)"),
        ("Fig. S1", "SOTA overall", "mean PSNR / SSIM, ours per seed + seed mean, comparators n = 1", "dB / 1", "HICRD ALL 2,000 square256", "comparators: registered 95 % cluster bootstrap; OUR absolute-mean CIs are NOT registered (computed with P1 draws)", "sota_hicrd_analysis/comparator_means.csv; sota_hicrd_pub_figures_v2/table_method_level.csv", "rendered (v2); label unregistered CIs"),
        ("Fig. S2", "SOTA contrasts", "32 registered contrasts per metric", "dB / 1", "HICRD ALL 2,000 square256", "registered 95 % cluster bootstrap; unadjusted", "sota_hicrd_analysis/analysis.json contrasts", "rendered (v2)"),
        ("Fig. S3", "SOTA E1r composition", "mean A, C, G, R", "MSE", "HICRD ALL 2,000 square256", "none", "per_image CSVs; s5_e1r/e1r_per_image.csv", "rendered (v2)"),
        ("Fig. S4", "SOTA geometry", "mean PSNR / SSIM at 256 and 482", "dB / 1", "HICRD ALL 2,000", "none (registered CIs exist per geometry, not for the change)", "table_method_level.csv", "rendered (v2)"),
        ("Fig. S5–S6", "SOTA visual panels", "per-image PSNR / SSIM annotations", "dB / 1", "12 registered images", "none", "sota_hicrd_pub_figures_v2/visual_checks.json", "rendered (v2)"),
        ("(any)", "SSIM IN − BN with CIs", "mean IN − BN SSIM", "1", "UIEB-787 / HICRD", "NOT AVAILABLE: no SSIM-difference bootstrap in any frozen file", "—", "BLOCKED (needs approval to compute, or omit)"),
    ]
    with open(out / "figure_data_mapping.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["figure", "panel", "values", "unit", "cohort / n", "uncertainty", "source (paper1_uieb_runs/)", "status"]); w.writerows(F)
    n_fail = sum(r["status"] != "PASS" for r in rows)
    print(json.dumps({"targeted": f"{len(rows) - n_fail}/{len(rows)}", "fails": [r for r in rows if r["status"] != "PASS"], "coverage": [cov_u, cov_h]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
