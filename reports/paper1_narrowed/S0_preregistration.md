# Paper 1 (narrowed): pre-registration S0

**Date:** 2026-11-01. Written **before** any UIEB data, filename or manifest was read.

**Status: FROZEN at commit.** The SHA-256 of this file is recorded in `S0_S1_stage_report.md`. Any later change needs a new, dated amendment file; this file is never edited.

**Decision basis:** `decision/AT_MARINE_FINAL_CONTINUE_PIVOT_ABANDON_DECISION.md` (NARROW).

**Precedence:** this document supersedes the endpoint section (§3, "Primary endpoint") of `reports/paper1_uieb_evaluation_protocol_proposal.md`. All other sections of that proposal remain in force (metrics, resize candidate, integrity, limitations).

---

## 1. Question

Does the choice of normalisation, batch (BN) or per-image instance (IN), determine:
- (a) restoration fidelity and seed stability; and
- (b) the **measured value of other UIE components**?

The components in (b) are a global colour-affine head, a physics-guided gate and a white-balance front-end.

## 2. Hypotheses

**H1.** IN > BN in paired per-image PSNR on audited UIEB, and the seed SD is smaller under IN.
- Primary architectures: MARINE-Net, AT-Trunk.
- Descriptive only: WWE-UIE (n = 1 per arm at G1).

**H2 (central).** A component that models per-image colour statistics has a large gain under BN and a gain near zero under IN.
- The **interaction contrast** for a component *c* is:
  `I_c = (gain of c under BN) − (gain of c under IN)`
- Each gain is the paired per-image PSNR difference "with c − without c", same seed.
- G1 tests *c* = colour heads using existing checkpoints. S4 and S5 test *c* = physics-guided gate (PGA) and *c* = white balance (WB).

**H3 (mechanism).** The per-image BN − IN gap increases with the magnitude of the image's global colour cast. Pre-declared descriptors, computed on the **input** image:
- channel-mean imbalance `max(μ_R, μ_G, μ_B) − min(μ_R, μ_G, μ_B)`;
- gray-world angle between `(μ_R, μ_G, μ_B)` and `(1, 1, 1)`.

Test: Spearman ρ over images, per seed. Reported descriptively, with a bootstrap 95 % CI.

## 3. Data and checkpoints

- **Test set:** UIEB paired images.
  - The subset definition ("all 890 pairs" vs `datasets/splits/candidate_v1/uieb_test_frozen_90.json`) is **a decision for the user**, recorded before evaluation.
  - The overlap audit (`v2_tools/uieb_overlap_audit.py`) runs first.
  - The primary analysis uses **all pairs**. The unflagged subset is a separate, labelled sensitivity table.
- **Checkpoints:**
  - **Primary:** `best_valA.pth`. Val-A selection is independent of UIEB, so it introduces no UIEB selection bias.
  - **Sensitivity:** `last.pth`. Needs an evaluator extension; hashes are recorded in `reports/data/paper1_provenance.csv`.
- **Protocol:** the frozen evaluator, Resize (256, 256), candidate protocol A. Scores are not comparable with published UIEB numbers.

**Checkpoints for G1:**

| Status | Runs |
|---|---|
| Already in the evaluator inventory (17) | `marinenet_{bn,in}_r3_fp32_s{0,1,2}`<br>`xf_attrunk_r3_fp32_s*`, `xfin_attrunk_in_r3_fp32_s*`, `armcin_idres_r3_fp32_s*`<br>`wwe_lr2e-4_s0`, `waternet_lsui_s0` |
| To be added in the next stage (14) | `stage4a_at_trunk_s{0,1,2}`, `armc_idres_s{0,1,2}`<br>`armd_global_s{0,1,2}`, `armd_grid_s{0,1,2}`<br>`xn_wwe_nowb_r3_s0`, `xnbn_wwe_nowb_bn_r3_s0` |

## 4. Gate G1 (existing checkpoints only, evaluated once)

The thresholds are pre-declared conventions, not derived from data. All differences are paired per-image means in dB on all audited pairs.

- **G1-a (normalisation transfers).**
  - MARINE-Net IN − BN ≥ 1.0 dB in **each** of seeds 0–2.
  - **and** AT-Trunk IN − BN (XFIN − XF) ≥ 1.0 dB in **each** of seeds 0–2.
- **G1-b (colour-head crossing).**
  - BN gain = `stage4a_at_trunk` − `armc_idres` (both AMP) ≥ 1.0 dB, mean over seeds.
  - **and** |IN gain| = |XFIN − ArmC-IN| ≤ 0.30 dB, mean over seeds.
  - **and** I_colour ≥ 1.0 dB.

| G1 outcome | Action |
|---|---|
| G1-a and G1-b pass | proceed to S3 (training block, separate approval) |
| G1-a passes, G1-b fails | H1 only; no PGA/WB training; venue IEEE JOE / IVC; re-decide with the user |
| G1-a fails | **stop training plans**; report as an in-distribution finding (short paper or note); re-decide the research question |

## 5. Gate G2 (after S4, PGA × norm, n = 5 per cell)

- **PGA crossing present:** I_PGA ≥ 0.5 dB and its seed-level 95 % CI excludes 0.
- **Otherwise:** PGA is reported as an inert component with the saturation mechanism.
- **Pattern Recognition is targeted only if** the crossing holds for at least 2 of the 3 components (colour, PGA, WB).

## 6. Analysis (fixed)

- **Unit of inference:** training seed. Per-seed paired mean differences; mean over seeds with a t-interval (n = 3 at G1, n = 5 after S3).
- **Image-level uncertainty:** paired bootstrap over images, 10,000 resamples, seed 20261028, nested within seed.
- **Multiplicity:** Holm across the component interaction contrasts.
- **Equivalence:** TOST with δ = 0.10 dB for any "≈ 0 under IN" statement. Sensitivity at δ = 0.05 and 0.20 dB.
- **Secondary metrics:** SSIM, LPIPS (AlexNet v0.1, reported separately), and CIEDE2000 if added by a reviewed change.
- **Not used for decisions:** UIQM, UCIQE, Wilcoxon p-values.
- **Disclosure:** every evaluation attempt, including failed attempts, is reported.

## 7. Claims that will not be made

- State of the art, "first", "best" or "superior".
- Comparisons with published UIEB numbers.
- Physical meaning of PGA parameters.
- Native-resolution performance.
- Any claim from Val-A numbers.
