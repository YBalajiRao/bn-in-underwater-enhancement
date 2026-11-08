# Paper 1: strict pre-submission audit (Pattern Recognition / IEEE standard)

Date: 2026-11-02. Status: read-only audit. Nothing was run, modified, re-frozen or committed. This file is new and uncommitted.

**Written for:** the researcher, as an internal adversarial review before drafting. It is not text for submission.

**Labels used**
- **VERIFIED:** checked against a file during this audit.
- **REPORTED:** copied from a frozen report that was independently audited earlier; not re-derived here.
- **UNVERIFIED:** not checkable from the repository.
- **RECOMMENDATION:** a proposed correction.

**What this audit checked directly**
- The working draft's numbers against the Stage 3 result JSON.
- The git line-ending state of the frozen artefacts.
- The EUVP-in-LSUI evidence wording.
- The commit and protocol chain.

**What it did not do**
- Re-derive Stage 3–4 statistics. They were independently recomputed in the Stage 4 audit and, for the TUB runs, earlier this session.
- Search the literature. **Novelty against prior work is therefore UNVERIFIED.**

**Sources reviewed**
- Draft: `paper1_draft_sections_v1_WORKING.md`, `paper1_draft_revision_proposal_v1.md`.
- Protocols and reports: `S0_preregistration.md`, `S2_post_G1_statistical_report.md`, `S2_seed_pairing_validity_audit.md`, `S3_results_report.md`, `S4_results_audit_and_research_decision.md`, `S4prep_dataset_inventory_comparison_exposure.md`, `S5_X1_results_report.md`, `S5_E1r_results_report.md`, `S5_research_decision_update_after_X1.md`.
- TUB: `TUB_dataset_audit_and_suitability_report.md`, `TUB_stage5_preregistration_FROZEN_DRAFT.md`, both TUB amendments and implementation addenda.
- Outputs: `paper1_uieb_runs/tub_stage5_*` (Square256 and G455 inference and analysis) and `paper1_uieb_runs/G455_execution_deviations.md`.

---

## 0. Verdict in ten lines

1. **Not ready for Pattern Recognition.** The working draft is a good scaffold, but it predates X1, E1r and TUB. Its Discussion hypothesis has since failed an out-of-sample test that it does not report.
2. **Defensible core:** a hash-locked, audited, matched-seed evaluation of BN→IN in two architectures across three external evaluation sets. The sign of the IN − BN effect is **positive only on UIEB** and **negative on HICRD and TUB**, in every seed, both architectures and both checkpoint choices.
3. **The colour-cast story is narrower than the draft implies.** It is confound-robust on UIEB only. On HICRD it is not identifiable (cast ≈ brightness, ρ = 0.989). On TUB the registered dose-response predictions are contradicted.
4. **The brightness-information-loss hypothesis is not supported out of sample.** It is the Discussion's leading candidate and was built from X1 and E1r. On TUB, E-T1, E-T2 and E-T3 have the opposite sign to the registered directions, in all 3 seeds and both architectures, with bootstrap CIs excluding 0 on the opposite side.
5. **The error attribution is dataset-specific.** HICRD: IN's extra error is brightness plus gain (A + G). UIEB: IN's saving is colour plus gain (C + G). TUB: IN's extra error is mostly the spatial residual (R), and its gain term is smaller than BN's.
6. **Evidence is thin where it matters for a Q1 claim:**
   - only 3 training seeds;
   - two closely related architectures;
   - one training set;
   - PSNR-led metrics;
   - no mechanism experiment (E2r not run);
   - no related work or citations.
7. **Statistical practice is a strength:** pre-registration with hashes, matched seeds, cluster and site-aware resampling, independent recomputation, and disclosed failures. It is undercut by mixed interval levels (95 % vs 97.5 %), conditional bootstraps, and a "not established" label that hides contradictions.
8. **One concrete reproducibility defect (VERIFIED):** the repository has no `.gitattributes`, `core.autocrlf=true`, and the frozen hashes were computed on working-copy bytes whose line endings differ from the committed blobs. A fresh clone can fail the scripts' own hash checks (issue D1).
9. **Recommended scope:** a rigorous empirical study with sign reversal and exact error attribution as the findings, written without a mechanism claim. Target a focused venue first. Pattern Recognition only after the "required before submission" items in §11.
10. **No state-of-the-art, causal or mechanism claim is supportable.** None should appear in any version.

---

## 1. Central contribution and defensible novelty

### 1.1 What the evidence supports as the contribution

| # | Contribution | Evidence | Status |
|---|---|---|---|
| C1 | A pre-registered, hash-locked, audited protocol for comparing BN vs IN: verified matched seeds, hash-checked cohorts, overlap audits, cluster- and site-aware resampling, independent recomputation, a ledger of every attempt | `S2_seed_pairing_validity_audit.md` (init weights bit-identical in every pair; same data order and augmentation); `S3_preregistration_FROZEN.md`; `S4_*` protocols; `S4_results_audit_and_research_decision.md` §2 | REPORTED; strong. It is a methodological strength, not a method contribution. |
| C2 | **The sign of the IN − BN PSNR difference differs across external datasets**: positive on UIEB (MARINE-Net +2.19 / +1.16 / +1.21 dB; AT-Trunk +0.25 / +0.70 / +0.47), negative on HICRD (MARINE-Net −3.43 / −3.91 / −1.40; AT-Trunk −1.69 / −3.19 / −1.44) and on TUB (§2). | Stage 3 and 4 reports; TUB `tub_analysis.json` | REPORTED and VERIFIED (TUB, this session). **Descriptive, not a pre-registered hypothesis.** |
| C3 | The exact error attribution (A, C, G, R) shows **where** IN's difference lies on each dataset, and that it is not the same everywhere (§2.3). | `S5_E1r_results_report.md`; TUB `E-T5` rows | REPORTED; exploratory. |
| C4 | Contradicted predictions are reported rather than hidden: G1 failed; the HICRD confound check failed; the TUB dose-response directions were opposite. | `1a7aace`, `S4_*`, TUB analyses | Strong for credibility. |
| C5 | A leakage finding: 482 of 515 EUVP test pairs are perceptually identical (256-bit dHash 0) to LSUI-train images. | `S4prep_dataset_inventory_comparison_exposure.md` line 55 | REPORTED. A benchmark-hygiene note, not a core result. |

### 1.2 What is not defensibly novel or must not be claimed

- **That BN vs IN matters, or that IN removes per-image statistics.** This is textbook; no claim.
- **Colour-cast-conditioned normalisation, or an adaptive-normalisation method.** The S4 decision rated this "not supported".
- **Any mechanism.** E2r was not run. E1r gives attribution, not cause.
- **State-of-the-art or "best" performance.** No comparison with published numbers is valid: the protocol resizes to 256², and S0 §7 forbids it.
- **Novelty relative to the literature.** UNVERIFIED. There is no related-work section and no citation in the draft (all `[REF?]`).
- **Generalisation beyond 2 related architectures, 3 seeds and 1 training set (LSUI).**

**Honest novelty statement.** "An audited empirical characterisation showing that the effect of replacing BatchNorm with InstanceNorm on PSNR in two LSUI-trained UIE networks changes sign across external evaluation sets, with an exact decomposition of where the error lies and a record of predictions that failed." Whether that is new is for a literature check to decide.

---

## 2. Unified BN vs IN comparison across the three datasets

### 2.1 Mean IN − BN PSNR (dB), per seed, best_valA, Square256

| Dataset (n; reference type) | MARINE-Net per seed | mean | AT-Trunk per seed | mean |
|---|---|---|---|---|
| UIEB frozen-90 (90; curated). **G1 gate cohort, not a random sample** (first 90 filenames) | +2.56 / +0.95 / +0.89 | +1.47 | +0.25 / +0.32 / +0.10 | +0.23 |
| UIEB-787 (787; curated) | +2.19 / +1.16 / +1.21 | +1.52 | +0.25 / +0.70 / +0.47 | +0.47 |
| HICRD-2000 (2,000; physically computed restorations; 188 reef-area clusters, ≈ 43 effective) | −3.43 / −3.91 / −1.40 | −2.91 | −1.69 / −3.19 / −1.44 | −2.11 |
| TUB primary (1,153; same-scene clear-water photograph; 33 scenes) | −3.53 / −3.09 / −3.23 | −3.28 | −3.79 / −2.72 / −2.26 | −2.92 |

The "mean" column is an arithmetic mean of the per-seed values, computed for this table.

**Sources**
- UIEB frozen-90: `S2_post_G1_statistical_report.md` §1.
- UIEB-787: `stage3_analysis_best_valA/results.json` (VERIFIED: +2.194 / +1.162 / +1.215, so the third value is +1.21; the +1.22 in the S3 report table is a rounding slip, and the draft's +1.21 is correct; see A8).
- HICRD: S4 report §4.
- TUB: `tub_analysis.json` E-T5 `d` rows (seed-mean 95 % scene-bootstrap CIs: MARINE-Net [−3.48, −3.11]; AT-Trunk [−3.13, −2.73]).

**Consistency across settings**
- Last-epoch checkpoints: same signs on every dataset (TUB `last`: −3.18 and −3.09 dB).
- Aspect-preserving geometry: same signs (HICRD G482: MARINE-Net −4.33 / −3.15 / −2.08; AT-Trunk −1.29 / −2.98 / −1.65; TUB G455: −3.04 and −2.87).
- HICRD: IN is worse in all 12 seed × architecture × checkpoint combinations (S4 §0.4).

**What differs between the datasets** (listed, not interpreted):
- the reference type;
- the degradation: UIEB mixed and curated; HICRD reef scenes with computed restoration; TUB a controlled achromatic turbidity series;
- the cast range (S4);
- the resolution;
- and, for TUB, a reference that is itself a (low-turbidity) photograph, not an enhancement target.

**Which of these causes the sign change is UNKNOWN.**

### 2.2 The colour-cast association (per-image Spearman ρ of IN − BN with input channel-mean imbalance)

| | UIEB-787 | HICRD D1 pooled | HICRD D2 (6-site adjusted) | HICRD G482 pooled |
|---|---|---|---|---|
| MARINE-Net ρ per seed | 0.346 / 0.322 / 0.227 | 0.004 / 0.268 / 0.445 | −0.026 / 0.181 / 0.460 | not independently tabulated |
| MARINE-Net ρ̄ [97.5 % CI] | 0.298 [0.236, 0.358] | 0.239 [0.081, 0.388] | 0.205 [0.035, 0.367] | 0.174 |
| AT-Trunk ρ per seed | 0.181 / 0.197 / 0.150 | 0.190 / 0.292 / 0.326 | 0.240 / 0.235 / 0.278 | not independently tabulated |
| AT-Trunk ρ̄ [97.5 % CI] | 0.176 [0.110, 0.241] | 0.269 [0.085, 0.441] | 0.251 [0.044, 0.448] | 0.069 |
| Registered label | SUPPORTED, confound-robust (both) | SUPPORTED, NOT confound-robust (both) | MARINE-Net INCONCLUSIVE; AT-Trunk SUPPORTED | INCONCLUSIVE (both) |
| Partial (brightness, contrast) ρ̄ | 0.279 / 0.184 | −0.113 / 0.013 | — | — |

- **UIEB** is the only dataset on which the association is both registered-supported and separable from brightness and contrast.
- **HICRD** cast ≈ brightness (ρ = 0.989), so the check is uninformative.
- **TUB** has no channel-imbalance dose (the degradation is achromatic; imbalance is flat). It therefore tests the brightness-side hypotheses instead (§2.4).

### 2.3 Exact error attribution (E1r and TUB; exploratory; units 10⁻³ MSE; Δ = IN − BN; positive = IN has more error)

| Component | UIEB MARINE-Net | UIEB AT-Trunk | HICRD MARINE-Net | HICRD AT-Trunk | TUB MARINE-Net | TUB AT-Trunk |
|---|---|---|---|---|---|---|
| ΔA (brightness offset) | +0.18 [−0.15, 0.50] | +0.07 [−0.36, 0.49] | **+2.79 [1.88, 3.73]** | **+1.55 [0.77, 2.30]** | **+8.19 [7.30, 9.35]** | **+5.86 [4.59, 7.36]** |
| ΔC (colour offset) | **−0.66 [−0.93, −0.44]** | **−0.96 [−1.50, −0.56]** | +0.01 [−0.08, 0.11] | **+0.21 [0.10, 0.30]** | +0.78 [0.70, 0.87] | +1.48 [1.38, 1.59] |
| ΔG (gain error) | **−0.63 [−0.77, −0.48]** | **−0.21 [−0.38, −0.05]** | **+2.00 [1.80, 2.18]** | **+1.58 [1.37, 1.77]** | **−5.21 [−6.01, −4.22]** | **−4.71 [−5.50, −3.72]** |
| ΔR (spatial residual) | **−0.27 [−0.34, −0.20]** | +0.02 [−0.04, 0.08] | −0.02 [−0.06, 0.01] | +0.17 [0.15, 0.18] | **+15.18 [13.95, 16.37]** | **+15.12 [14.05, 16.13]** |

- UIEB and HICRD CIs are 95 % bootstrap intervals (E1r §2–3).
- TUB values are seed-means of the scene-weighted E-T5 means, with 95 % scene-bootstrap CIs from `tub_analysis.json`.
- **TUB CI units:** the A, C, G and R rows are in 10⁻³ MSE (the printed values ×10³). Bounds are rounded from `tub_analysis.json` (checked for the AT-Trunk ΔA and ΔG rows and MARINE-Net ΔA).

**Reading (attribution, not explanation)**
- The three attributions do **not** coincide: HICRD is dominated by A + G; TUB by R, with G working in IN's favour; UIEB by C + G in IN's favour.
- Any single-component account of "why IN loses" (such as "IN loses absolute brightness") is therefore not supported across datasets. It fits HICRD at most.

### 2.4 TUB dose-response (the registered, post hoc, exploratory test; NTU cohort, 1,022 pairs, 29 scenes; within-series Spearman; seed-mean [95 % scene-bootstrap CI]; per-seed signs)

| Endpoint (registered direction) | MARINE-Net | AT-Trunk | Per-seed signs | Registered label |
|---|---|---|---|---|
| E-T1: d vs NTU (**negative**) | **+0.524 [+0.443, +0.603]** | **+0.654 [+0.585, +0.722]** | all positive | not established |
| E-T2: ΔA vs NTU (**positive**) | **−0.714 [−0.773, −0.649]** | **−0.804 [−0.849, −0.755]** | all negative | not established |
| E-T3: ΔG vs NTU (**positive**) | **−0.749 [−0.834, −0.653]** | **−0.740 [−0.828, −0.644]** | all negative | not established |
| E-T4a: ΔC vs NTU (none) | −0.275 | +0.083 | mixed in sign across seeds | no direction stated |
| E-T4b: ΔR vs NTU (none) | +0.923 | +0.915 | all positive | no direction stated |

**Square256, best_valA, best-checkpoint values; the `last` checkpoint and the G455 geometry give the same signs.**

- **Contradiction, in plain terms.** E-T1 to E-T3 were registered with the opposite sign to what was observed. For both architectures, all three seeds and both checkpoint types, the CI lies on the opposite side of 0.
- **Reading:** IN's PSNR deficit relative to BN **shrinks** as turbidity rises, and its brightness and gain excess shrink as well.
- **Do not claim support.** The registered labels stand.
- **Seed-level caveat (G455):** MARINE-Net E-T1 seed 0 = +0.148, and its 3-seed t95 interval is [−0.24, +1.00], so it includes 0 (G455 `best_valA`).

---

## 3. Statistical results, intervals and seed-level uncertainty

### 3.1 The three uncertainty layers in the project

| Layer | Used in | What it covers | What it ignores |
|---|---|---|---|
| Image bootstrap, 97.5 % (two architectures) | S3 UIEB ρ̄ CIs | resampling of the cohort's images | training-seed variation; any image grouping |
| Cluster / site-stratified bootstrap, 97.5 % | S4 HICRD | reef-area clusters (≈ 43 effective) | training-seed variation |
| Scene bootstrap, 95 % | TUB, E1r (HICRD), X1 | complete scenes or clusters | training-seed variation |
| Seed-level t-interval, df = 2 | all, descriptive | training-seed variation (n = 3) | image sampling |

- **Bootstrap CIs are conditional on three fixed trained models per arm.** They quantify cohort sampling only.
- **Seed t-intervals with df = 2 are wide.** Examples:
  - UIEB frozen-90: MARINE-Net IN − BN +1.47 [−0.89, 3.83], p = 0.116.
  - TUB MARINE-Net E-T1 [−0.24, +1.00] (G455 best_valA).
- **The seed count is the binding limitation of the whole paper.** `S2_seed_pairing_validity_audit.md` shows that with n = 3, interval exclusion of 0 can depend on the analysis choice (paired vs Welch); no conclusion should rest on p ≈ 0.02–0.06 at n = 3.

### 3.2 Per-seed spread worth showing in the paper

- UIEB-787 MARINE-Net d ranges from +1.16 to +2.19 dB across seeds (a factor of about 1.9).
- HICRD MARINE-Net ranges from −1.40 to −3.91.
- HICRD ρ_s for MARINE-Net ranges from 0.004 to 0.445.
- E1r HICRD ΔA for MARINE-Net ranges from 1.33 to 5.53 × 10⁻³.
- Therefore: **the sign of the dataset effect is stable across seeds, but its magnitude is not.** State both.

### 3.3 Statistical findings

See the issue register: B1–B10.

---

## 4. Confirmatory vs exploratory evidence (strict separation)

### 4.1 Status register

| Analysis | Registered before data? | Status to state in the paper | Outcome |
|---|---|---|---|
| G1-a, G1-b (S0 §4) | yes (S0, frozen 2026-11-01, commit `4a462f8`) | **Confirmatory gate** | **Failed** → `stop_training_plans` (`1a7aace`) |
| H1 / H2 seed-level estimates, Holm m = 1 (S0 §6) | estimation method registered; no α or test | **Estimates with uncertainty**, not confirmatory tests (corrected in `S2_seed_pairing_validity_audit.md` §4) | I_colour = 0.96 dB, seed t95 [−0.03, 1.95], p = 0.053; below the 1.0 dB threshold; "suggestive, not established" |
| H3 on frozen-90 (colour-cast association) | **No.** Exploratory, protocol note committed before computing (`b0003cf`), `ef89978` | **Exploratory** | Positive in both architectures; AT-Trunk sign change by tertile **not reproduced** later |
| **Stage 3 H3-S3** (UIEB-787) | yes (`ee8f0ab`, before any model output) | **Confirmatory replication on disjoint data** | **SUPPORTED**, both architectures; confound-robust |
| **Stage 4 D1 / D2** (HICRD) | yes (v1 + A01 `7c78bbb` + A02 `c8d7069`, before inference) | **Confirmatory** | D1: SUPPORTED, NOT confound-robust (both). D2: AT-Trunk SUPPORTED; MARINE-Net INCONCLUSIVE. Joint site-independent claim not permitted |
| Sign reversal of mean IN − BN | **No** (descriptive; S4 §4) | **Descriptive** | stable sign; unexplained |
| X1 (required correction) | No (designed after Stage 4; frozen before execution) | **Exploratory** | primary g_tot not established; secondary g_chr (+) and g_ach (−) associations |
| E1r (exact decomposition) | No (designed after X1; frozen before execution) | **Exploratory** | A + G on HICRD; C + G on UIEB |
| **TUB Stage 5** (E-T1 to E-T5; sensitivities) | Frozen before inference (`771e678`); **post hoc relative to HICRD, X1 and E1r**; file named "FROZEN_DRAFT" | **Exploratory, pre-specified** | E-T1 to E-T3 opposite to registered directions |
| TUB G455 and `last.pth` G455 | Amendments 01 and 02 plus an explicit user instruction | **Sensitivity** (`last` G455 is supplementary; its batch-equivalence preflight was not run) | same signs and labels as Square256 |

### 4.2 Rules for the manuscript

- "Pre-registered" may be used only for G1, S3 and S4. Use "pre-specified" for TUB and "exploratory" for H3 (frozen-90), X1 and E1r.
- State once that **the hypothesis for S3 and S4 originated from the exploratory frozen-90 analysis after G1 failed**, and give the commit chain: `1a7aace` → `b0003cf` → `ef89978` → `ee8f0ab` → `d2feaff` → `929060c` / `7c78bbb` / `c8d7069` → `2e2c944`.
- No exploratory result enters the Abstract's first sentence or the Conclusion as a finding.
- Report every failed gate and every amendment.

---

## 5. Geometry sensitivity: Square256 vs G455 (TUB) and G482 (HICRD)

### 5.1 TUB (best_valA; seed mean; Square256 → G455)

| Endpoint | MARINE-Net | AT-Trunk |
|---|---|---|
| E-T1 ρ̄ (d vs NTU) | +0.524 → +0.381 (Δ −0.144) | +0.654 → +0.696 (Δ +0.042) |
| E-T2 ρ̄ (ΔA) | −0.714 → −0.495 (Δ +0.219) | −0.804 → −0.807 (Δ −0.003) |
| E-T3 ρ̄ (ΔG) | −0.749 → −0.719 (Δ +0.030) | −0.740 → −0.717 (Δ +0.023) |
| E-T5 mean d (dB) | −3.28 → −3.04 | −2.92 → −2.87 |
| E-T5 ΔA / ΔG / ΔR (×10⁻³) | +8.19 / −5.21 / +15.18 → +6.68 / −5.04 / +14.63 | +5.86 / −4.71 / +15.12 → +5.15 / −4.81 / +14.29 |

- **Direction:** no sign changes in per-seed Spearman patterns, in any E-T5 component mean, or in any registered label, for either checkpoint type.
- **Magnitude:** MARINE-Net E-T1 and E-T2 weaken (0.14–0.22); AT-Trunk is stable (≤ 0.04 on E-T1 to E-T3).
- **Meaning:** the contradiction of the registered directions is **not** an artefact of the 1.78 anisotropy.
- **Limit:** G455 changes only the aspect ratio. It leaves unchanged the models' training resolution (256²) and the 1920 × 1080 → small-image downsampling, so it does not test native-resolution behaviour.

### 5.2 HICRD (Square256 vs G482; REPORTED)

- The association ρ̄ falls from 0.239 to 0.174 (MARINE-Net) and from 0.269 to **0.069 (AT-Trunk)**. All D1 and D2 decisions become INCONCLUSIVE.
- The mean IN − BN sign is unchanged (negative in all seeds).
- AT-Trunk's HICRD association is therefore **geometry-sensitive** and cannot be presented as a robust property.

### 5.3 Reading

- **The sign of the average effect** is robust to geometry on both datasets.
- **The correlational effect sizes** (cast on HICRD, dose-response on TUB for MARINE-Net) are geometry-sensitive.
- The paper should show geometry as a sensitivity axis in the main sensitivity table, not hide it in an appendix.

---

## 6. Protocol deviations, limitations and reproducibility

### 6.1 Deviation register (every item should appear in an appendix)

| # | Deviation / event | Effect | Evidence |
|---|---|---|---|
| 1 | G1-a failed; training plans stopped | Original central claim (colour-head × normalisation crossing) not supported | `1a7aace`; `S2_G1_uieb_frozen90_report.md` |
| 2 | First G1 attempt failed (WaterNet white-balance 0/0); fix verified, rerun disclosed | none on conclusions | `96c42e1`, `262f8b5` |
| 3 | Post-G1 wording corrections: "nominal" seed pairing was too weak (pairing is real); H1/H2 are estimates, not confirmatory tests; equivalence "≈ 0 under IN" cannot be claimed | claim narrowing | `S2_seed_pairing_validity_audit.md` §4 |
| 4 | Frozen-90 is the first 90 filenames, not random; 1 confirmed train/test near-duplicate (15_img_ ↔ LSUI train 1013) | representativeness | `S2_seed_pairing_validity_audit.md` §3 (frozen-90 rule); `S2_post_G1_statistical_report.md` §4 (near-duplicate) |
| 5 | Stage 4 amendments A01 and A02 (before inference) | disclosed | `7c78bbb`, `c8d7069` |
| 6 | Stage 4 ran with 1 modified tracked file and 14 untracked paths in the tree (none in the dependency chain) | none | `S4_results_audit_and_research_decision.md` §2 |
| 7 | Reporting errors acknowledged: wrong start-time estimate; GPU memory under-estimate | none on results | `S4_*` §2 |
| 8 | E1r report note: "largest Δ" field in `e1r_results.json` mislabels the largest contributor on UIEB | presentation | `S5_E1r_results_report.md` §3 |
| 9 | TUB: protocol was a "frozen draft"; hypotheses derived post hoc from E1r | exploratory status | `TUB_stage5_preregistration_FROZEN_DRAFT.md` |
| 10 | TUB G455 needed Amendments 01 (batch 8, memory failure) and 02 (batch 4) | disclosed | `04361ee`, `d530acb` |
| 11 | In-run GPU-memory probe ran before TF32 was disabled → under-measured memory (also in the Square256 runs); per-model reserved memory not recorded for the real runs | none on outputs; weaker no-spill evidence | `paper1_uieb_runs/G455_execution_deviations.md` |
| 12 | `last.pth` G455 not registered; its batch-1 vs batch-4 preflight was not run | supplementary only | same file |
| 13 | S3 report table shows +1.22 where the JSON gives +1.2148 (= +1.21) | cosmetic | VERIFIED this audit |

### 6.2 Limitations to state (none may be omitted)

**Models and training**
- Two architectures, one derived from the other.
- n = 3 seeds per arm, the only training replication.
- One training set (LSUI, 3,463 pairs).
- The BN → IN swap changes batch coupling, inference statistics and per-image normalisation together.

**Evaluation**
- PSNR (and SSIM) at 256² (Square256 or the G455 / G482 sensitivities); no perceptual or task-based evaluation; no native resolution.
- Scores are not comparable with published numbers.

**Datasets**
- **UIEB:** curated references; remainder is a census, not a random sample; 3 train/test near-duplicates confirmed and handled.
- **HICRD:** computed restorations; one location and camera; cast ≈ brightness; ≈ 43 effective clusters.
- **TUB:**
  - one tank and lighting set-up, so low ecological validity;
  - the reference is a low-turbidity photograph with its own cast and vignetting (1.0–8.7 NTU), not an enhancement target;
  - NTU metadata defects in scenes 10, 15, 28, 33 and a phantom Exp34;
  - no EXIF, so auto-exposure and white balance are unknown and brightness is not a pure function of NTU;
  - 3 misaligned series, 2 byte-duplicate observations and 1 image identical to its reference (all flagged; the primary cohort drops them).

**Inference**
- Exploratory analyses are many and unadjusted.
- All bootstraps are conditional on the trained models.

### 6.3 Reproducibility concerns

See D1–D6 in the register. The most serious is the line-ending / hash-lock defect.

---

## 7. Issue register: unsupported claims, contradictions, defects

Severity: **Critical** = would be a reject-level objection or a false statement; **Major** = reviewers will raise it; **Minor** = polish.

### A. Claims and contradictions

**A1 (Critical). The working draft omits X1, E1r and all of TUB.**
- *Evidence:* `paper1_draft_sections_v1_WORKING.md` ends with "testable without training (proposal S5, E1)". That test has now been run (E1r), and a further out-of-sample test (TUB) failed its registered directions.
- *Impact:* the Discussion states a prediction the project has since tested, and does not report the result.
- *Exact correction:* add Results subsections "Exact error attribution (exploratory)" (Table 3) and "Controlled turbidity series (TUB; exploratory, pre-specified)" (Table 4, Fig. 5). Delete the sentence "This predicts that IN's HICRD deficit is concentrated in global colour offset; this is testable without training" and replace it with the measured result: on HICRD the deficit is in the brightness (A) and gain (G) terms, not colour (C).

**A2 (Critical). The Discussion's leading hypothesis is not supported out of sample, and the draft's own HICRD-based wording is wrong.**
- *Evidence:*
  - The draft's hypothesis says the HICRD deficit is "global colour offset"; E1r shows A + G, with ΔC ≈ 0 for MARINE-Net.
  - The brightness-loss hypothesis from X1/E1r predicts a deficit that grows with turbidity, with ΔA and ΔG rising with NTU (E-T1 negative, E-T2 and E-T3 positive). TUB shows all three with the opposite sign in every seed, both architectures, both checkpoint types and both geometries.
  - TUB E-T5 shows ΔG negative (IN's gain error is smaller than BN's) and R dominant (+15.2 × 10⁻³ vs A +8.2).
- *Impact:* presenting the brightness-information-loss idea as a mechanism, or even as "the leading candidate", would conflict with the project's own data.
- *Exact correction:* Discussion text: "On HICRD, IN's additional error was concentrated in brightness and gain terms. A registered, out-of-sample test of the brightness-information-loss idea on a controlled turbidity series was **not supported**: the deficit shrank, not grew, with turbidity, and the attribution on that dataset was dominated by the spatial residual. We do not have an explanation for the sign reversal." Remove the BN-running-statistics sentence or label it "untested, equally unsupported".

**A3 (Critical). Within-dataset and within-series associations of the same variable point in opposite directions, and the draft does not mention it.**
- *Evidence:*
  - X1 (between images; HICRD): ρ(g_ach, d) = −0.257 (MARINE-Net) and −0.349 (AT-Trunk), both consistent.
  - TUB (within series; supplementary analysis with g_ach in place of NTU, primary cohort, best_valA): E-T1 ρ̄ = **+0.488** and **+0.608**.
  - These are different estimands (between-image vs within-scene dose-response). They are not proven contradictory.
- *Impact:* a reader will combine them. Any "achromatic correction hurts IN" statement fails the within-series test.
- *Exact correction:* add one sentence and one table row: "The association of d with the achromatic correction demanded by the reference is negative across images (X1) and positive within turbidity series (TUB); the two estimands are different and the discrepancy is unresolved." Do not combine them in a single claim.

**A4 (Major). "Replicated" association overstated (Contribution 3).**
- *Evidence:* HICRD D1 is "SUPPORTED, NOT CONFOUND-ROBUST"; MARINE-Net D2 is INCONCLUSIVE with seed-0 ρ = 0.004 / −0.026; G482 makes all decisions INCONCLUSIVE; AT-Trunk ρ̄ falls to 0.069.
- *Impact:* a reviewer reading "replicated" will check Table 2 and find the contradiction.
- *Exact correction:* "The association is positive in direction on both datasets; it is confound-robust on UIEB; on HICRD it cannot be separated from brightness, is inconclusive for MARINE-Net after site adjustment, and is geometry-sensitive for AT-Trunk."

**A5 (Major). "IN better than BN on UIEB" is not uniform.**
- *Evidence:* AT-Trunk on frozen-90 is +0.23 dB, seed t95 [−0.05, 0.51] and every per-seed image CI includes 0; `last.pth` seed 2 is negative. On UIEB-787 the AT-Trunk gain is +0.25 / +0.70 / +0.47 (per-seed CIs not reported in the draft).
- *Impact:* the contribution statement implies both architectures benefit on both UIEB sets.
- *Exact correction:* "On UIEB-787, IN scores higher than BN in all three seeds for both architectures (MARINE-Net +1.16 to +2.19 dB; AT-Trunk +0.25 to +0.70 dB). The AT-Trunk effect was not distinguishable from zero on the frozen-90 cohort." Add per-seed image-bootstrap CIs to Table 1.

**A6 (Major). "Pre-registered" in the title and Contribution 1.**
- *Evidence:* G1 failed; the S0 colour-head hypothesis was dropped; S3 and S4 hypotheses came from an exploratory result; TUB, X1 and E1r are post hoc.
- *Impact:* a reviewer who reads the S0 file will see that the registered central claim failed.
- *Exact correction:* title without "pre-registered": e.g. *"Instance versus batch normalisation in underwater image enhancement: an audited cross-dataset evaluation of matched-seed network pairs"*. In Methods: a "Registration status" table (§4.1) with commit hashes.

**A7 (Major). "Dataset-dependent" or "domain-dependent" implies a cause the data cannot give.**
- *Evidence:* three datasets differ in at least five ways (§2.1); UIEB is the only positive one.
- *Impact:* the phrase suggests a domain mechanism. It could equally be reference style, curation, or any of the confounds in A2 and A3.
- *Exact correction:* "IN scored higher than BN on UIEB and lower on HICRD and TUB; the datasets differ in reference type, degradation and acquisition, and we do not identify which difference matters."

**A8 (Minor). Numerical discrepancy.**
- *Evidence:* `S3_results_report.md` §1 table: "+1.22" for MARINE-Net seed 2; `results.json` `mean_d_per_seed[2]` = 1.2148 (VERIFIED). The draft uses +1.21 (correct).
- *Impact:* a reviewer comparing the repository with the paper will see a mismatch.
- *Exact correction:* do not edit the frozen report. Add `S3_results_report_erratum.md` listing the rounding slip, and cite the JSON as the source for the paper.

**A9 (Major). E1r's UIEB interpretation is not cast-specific.**
- *Evidence:* E1r §3 attributes IN's UIEB advantage to ΔC and ΔG, but §4 shows the per-image link between ΔC and the required chromatic correction is consistent only for AT-Trunk (ρ = −0.182) and is mixed for MARINE-Net.
- *Impact:* "IN removes colour offset" would be over-read.
- *Exact correction:* state the aggregate attribution only and add "the per-image link to the demanded correction is consistent only for UIEB AT-Trunk".

### B. Statistics

**B1 (Major). Mixed interval levels and meanings.**
- *Evidence:* 97.5 % (S3, S4) vs 95 % (X1, E1r, TUB); image vs cluster vs scene bootstraps; seed t95 with df = 2.
- *Impact:* tables cannot be compared by eye, and 97.5 % intervals look wider than they are in context.
- *Exact correction:* every table caption states the level and the resampling unit. Put a "Uncertainty layers" box (§3.1) in Methods. Either report all at 95 % with a note, or keep 97.5 % only where registered.

**B2 (Major). The training seed is the only replication, and n = 3.**
- *Evidence:* §3.1–3.2.
- *Impact:* a reviewer will ask for more seeds; claims at the population-of-trainings level are weak.
- *Exact correction:* put the per-seed values in the main tables; state conclusions per seed ("same sign in 3/3 seeds"); never say "IN improves PSNR by X dB" without "for these three trained pairs".

**B3 (Major). Multiplicity in the exploratory analyses.**
- *Evidence:* X1 (several descriptors × 2 datasets × 2 architectures), E1r (5 components × 2 × 2), TUB (5 endpoints × 2 architectures × 2 estimators × several cohorts × 3 runs); no family-wise control.
- *Impact:* uncorrected exclusions of 0 will be questioned.
- *Exact correction:* label all exploratory intervals "unadjusted"; report the number of endpoints examined per section; keep confirmatory claims limited to the registered ones.

**B4 (Major). The label "not established" hides contradictions.**
- *Evidence:* TUB E-T1 to E-T3: the registered rule gives "not established" whether the effect is absent or reversed.
- *Impact:* a reader sees "not established" and does not realise the data point the other way.
- *Exact correction:* keep the registered label unchanged and add a descriptive column "Observed vs registered direction": "opposite, 3/3 seeds, CI on opposite side". Do not rename any classification.

**B5 (Major). Outcome-scale dependence.**
- *Evidence:* TUB E-T1 is in dB and rises with NTU, while E-T4b (ΔR in MSE units) also rises with NTU (ρ ≈ +0.92) and the E-T5 MSE components are in absolute units.
- *Impact:* PSNR is logarithmic and the MSE components are not; conclusions about "growing" or "shrinking" differ by scale.
- *Exact correction:* state the scale with every claim. Add an exploratory column with each component normalised by the BN model's MSE, labelled post hoc.

**B6 (Major). A floor effect on TUB is not excluded.**
- *Evidence:* both models' PSNR is lower at high turbidity; the shrinking gap could reflect both converging, and this was not tested. The per-arm PSNR-vs-NTU curves are not shown.
- *Impact:* E-T1's opposite sign could be a scale artefact.
- *Exact correction:* add a descriptive figure of BN and IN PSNR (each) versus NTU from the existing per-image files (no new inference; separate approval as an exploratory analysis). Until then, do not interpret the E-T1 reversal.

**B7 (Moderate). Image-level independence on UIEB.**
- *Evidence:* UIEB-787 uses an image bootstrap; 11 repeats were flagged and retained in the 800 sensitivity set.
- *Impact:* slightly optimistic CIs.
- *Exact correction:* state it; the 707-pair sensitivity (all 93 flagged excluded) already exists, so cite it in the main text.

**B8 (Moderate). Thresholds are conventions.**
- *Evidence:* ρ̄ ≥ 0.10 operational; 1.0 dB G1; TOST margin 0.10 dB; none justified externally (S0 states this).
- *Impact:* "SUPPORTED" is a statement about passing a convention.
- *Exact correction:* write "met the pre-specified operational criterion (ρ̄ ≥ 0.10)" and say the criterion is conventional.

**B9 (Moderate). Effect-size interpretation.**
- *Evidence:* ρ̄ = 0.18–0.30: cast explains a modest part of image-to-image variation.
- *Impact:* "increases with cast" can read as a strong relation.
- *Exact correction:* report ρ² or the bin means (MARINE-Net high-cast tertile +1.9 to +4.1 dB vs low +0.1 to +0.8; AT-Trunk +1.2 to +1.7 vs −0.2 to +0.3, Stage 3 tertiles) next to ρ̄.

**B10 (Moderate). Metrics.**
- *Evidence:* PSNR is the only endpoint for every confirmatory result; SSIM secondary on frozen-90 only; UIQM and UCIQE excluded by S0.
- *Impact:* reviewers in UIE expect perceptual or no-reference evidence.
- *Exact correction:* add SSIM (already computed for all cohorts) to Tables 1 and 2; add LPIPS only if it can be run under a new, registered, inference-only protocol.

### C. Design and validity

**C1 (Critical for Pattern Recognition). Scope: two related architectures, one training set, three seeds.**
- *Evidence:* AT-Trunk is derived from MARINE-Net.
- *Impact:* the most likely stated reason for rejection ("insufficient generality").
- *Exact correction:* either add at least one architecture family and more seeds (§8, E3), or scope the title and Abstract to "two related networks".

**C2 (Major). Reference types differ and cannot be separated from the dataset.**
- *Evidence:* curated (UIEB), computed (HICRD), same-scene clear photograph (TUB).
- *Impact:* "dataset-dependent" may be "reference-dependent".
- *Exact correction:* Table 2 lists the reference type per dataset; the Limitations section says the design cannot separate reference from scene content.

**C3 (Major). TUB is not an enhancement benchmark.**
- *Evidence:* TUB report §3–5, §8: Stage 01 is 1.0–8.7 NTU with its own cast and vignetting; no EXIF; the four NTU defects.
- *Impact:* scores vs Stage 01 penalise legitimate colour correction by LSUI-trained models, and absolute PSNR is not comparable with the other datasets.
- *Exact correction:* "TUB is used as a controlled dose-response series, not as ground truth." Never compare absolute PSNR across datasets.

**C4 (Major). The sign-reversal finding was not registered as a hypothesis.**
- *Evidence:* S4 §4: "descriptive; not a Stage 4 hypothesis".
- *Impact:* if it is the paper's headline, it is a post hoc finding.
- *Exact correction:* present it as "observed in confirmatory runs, but not registered as a hypothesis" and give its replication across TUB (also post hoc) as supporting evidence only.

**C5 (Moderate). Geometry.**
- *Evidence:* §5; models trained at 256²; 16:9 → 1:1 squashing.
- *Impact:* HICRD AT-Trunk association vanishes under G482.
- *Exact correction:* §5 table in the main text; no claim about "cast" or "dose" that is not shown at both geometries.

**C6 (Moderate). Checkpoint selection.**
- *Evidence:* best_valA (Val-A selection independent of external sets) and `last.pth` agree on every decision.
- *Impact:* positive; mention it.
- *Exact correction:* one sentence.

### D. Reproducibility

**D1 (Major, VERIFIED). Hash-locked artefacts and git line endings.**
- *Evidence:* there is no `.gitattributes`; `git config core.autocrlf` is `true`.
  - `git ls-files --eol` shows `i/lf w/crlf` for `paper1_uieb_runs/tub_stage5_prep/TUB_ALL_1183.json` and `v2_tools/tub_stage5_infer.py` (working copy CRLF, committed blob LF).
  - It shows `i/lf w/lf` for `S0_preregistration.md`, `S3_preregistration_FROZEN.md` and `TUB_stage5_preregistration_FROZEN_DRAFT.md` (working copy LF).
  - The frozen hashes were computed on working-copy bytes. The same blob checks out as CRLF on a Windows `autocrlf=true` machine and as LF elsewhere. Hash-pinned files (protocols, cohort JSON, outputs) can therefore differ from their frozen hashes in a fresh clone, and the scripts' own `verify_hashes()` would STOP.
  - Earlier this session: blob hashes of CSV and JSON outputs also differed from the working-file hashes (same bytes except carriage returns).
- *Impact:* "hash-locked" claims are not independently verifiable from the repository alone.
- *Exact correction:*
  1. Add a `.gitattributes` with `-text` (or `binary`) for every frozen artefact path and for `paper1_uieb_runs/**` outputs.
  2. Re-add those files so the blobs equal the hashed bytes, as a new commit; do not rewrite history.
  3. Test: clone to a fresh directory on Windows and on Linux, run `verify_hashes()` and the dry runs.
  4. Meanwhile, document in the Reproducibility statement that hashes refer to working-copy bytes.

**D2 (Major). Data and checkpoints are not in the repository.**
- *Evidence:* `.gitignore` excludes `datasets/` and `*.pth` (VERIFIED); `original_repo/marine_net_model.py` is a pinned, git-ignored file (`8c3a2654…`).
- *Impact:* an outside reader cannot re-run anything.
- *Exact correction:* archive checkpoints (24), cohort manifests and per-image outputs in a persistent repository (e.g. Zenodo) with the manifests of hashes; write a data-acquisition note per dataset with exact revision or DOI (TUB revision `6214f008…`, CC-BY-4.0; HICRD DOI `10.25919/ahhq-0827`); state the EUVP licence status (UNVERIFIED: only an uploader-declared CC0 copy was used); state the redistribution status of the pinned original model file.

**D3 (Moderate). Training is not bitwise reproducible.**
- *Evidence:* `cudnn.deterministic` and `use_deterministic_algorithms` are not set (`S2_seed_pairing_validity_audit.md` §1).
- *Impact:* retraining will not reproduce the checkpoints exactly.
- *Exact correction:* state it; release the checkpoints; claim reproducibility of the evaluation only.

**D4 (Moderate). Environment.**
- *Evidence:* Windows 11, one RTX 3070 (8,192 MiB), torch 2.7.1+cu118, torchvision 0.22.1+cu118, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0, Python 3.12; the GPU memory ceiling forced Amendments 01 and 02.
- *Impact:* reviewers will ask whether results depend on batch size or TF32; they do not (batch equivalence max difference 5.5 × 10⁻⁶ on outputs for 12/12 best_valA checkpoints; `last.pth` untested).
- *Exact correction:* list the environment; report the equivalence table in the appendix; state that `last.pth` equivalence was not tested.

**D5 (Moderate). Ledger and logs.**
- *Evidence:* every inference attempt is in `paper1_uieb_runs/ledger/` (VERIFIED for TUB).
- *Impact:* a strength; state it.
- *Exact correction:* include the ledger summary in the supplementary material.

**D6 (Minor). The memory-probe ordering defect (deviation 11).**
- *Exact correction:* disclose as in `G455_execution_deviations.md`; if any run is repeated, move the TF32-off lines before the probe and re-freeze.

### E. Manuscript

**E1 (Critical). No related work or verified citations.** Every reference in the draft is `[REF?]`. *Exact correction:* run a documented literature search (databases, queries, dates, what failed to load), verify each entry by DOI or arXiv id, and write Related Work covering BN, IN, normalisation in restoration, test-time statistics adaptation, UIE datasets and benchmark leakage. Do not state novelty until this is done.

**E2 (Major). No figures or tables exist.** See §9.

**E3 (Major). Descriptor naming.** "Colour cast" is operationally `max(μ) − min(μ)` (channel-mean imbalance). *Exact correction:* use "channel-mean imbalance (colour-cast proxy)" and define it once; do not use "colour cast" for the TUB achromatic degradation.

**E4 (Moderate). Licensing and ethics statements.** TUB CC-BY-4.0 requires attribution (arXiv 2606.26295); the HICRD licence and the EUVP licence status must be stated. *Exact correction:* a Data statement with dataset, revision, licence, count after filtering.

---

## 8. Missing experiments, in priority order (all need separate approval; none run here)

| # | Experiment | Why | Cost | Status |
|---|---|---|---|---|
| E0 | **Per-arm PSNR vs NTU curves from existing TUB per-image outputs** (descriptive; no inference) | Rules in or out the floor effect (B6) | CPU minutes | not run |
| E1 | **Training-free mechanism checks beyond E1r:** E2r (BN with per-image mean only, per-image variance only; IN with running statistics) | Only experiment that can address the normalisation mechanism | < 1.5 GPU-h (S4 estimate, inference only) | not approved |
| E2 | Additional seeds (≥ 5; about 7 for MARINE-Net per Stage 3 note) | Reduces the n = 3 limitation | 48–54 GPU-h | not approved |
| E3 | A third architecture family, from a different lineage than MARINE-Net | Addresses C1 | training; not estimated here | not approved |
| E4 | A second curated-reference dataset with an audited overlap | Tests whether "positive only on UIEB" is a reference-style effect | audit plus inference | not approved |
| E5 | Aspect-preserving or native-resolution evaluation on UIEB and in the main tables | Geometry coverage is currently HICRD and TUB only | inference | not approved |
| E6 | Perceptual or task-based evaluation under a new registered protocol | Reviewer expectation | inference | not approved |
| E7 | Normalised error components (component / BN MSE) from saved E1r and TUB outputs | Addresses B5 | CPU minutes | not run |

**Not recommended:** an adaptive-normalisation method (S4 rated it "not supported"); training a new family only to chase the colour-cast story.

---

## 9. Required tables and figures (exact contents)

All figures must be generated from frozen result files by a script that is itself hash-recorded. None exists yet.

**Tables**

- **T1. Study design and registration status.** Rows: G1, S3, S4, X1, E1r, TUB, G455 sensitivity. Columns: purpose; registered before data (Y/N); commit hash of the protocol; commit hash of the results; label (confirmatory / pre-specified exploratory / exploratory / descriptive); outcome in one phrase. Source: §4.1.
- **T2. Datasets and cohorts.** Rows: UIEB frozen-90, UIEB-787, HICRD-2000 (1,839 known-site / 161 UNASSIGNED), TUB (1,183 / 1,153 / 1,022). Columns: n pairs; clusters or scenes; reference type; resolution; cast or dose range; exclusions and reasons; overlap audit result (EUVP-in-LSUI note); licence and revision.
- **T3. Mean IN − BN (dB) per seed.** Rows: dataset × architecture. Columns: seeds 0 / 1 / 2; seed mean; seed t95 (df 2); the 95 % bootstrap CI (state the unit); share of images with IN > BN; checkpoint (best_valA, last); geometry. Source: §2.1.
- **T4. Cast association.** Columns as §2.2, plus partial ρ̄ and the registered decision.
- **T5. Exact error attribution.** §2.3, with the seed range and an "unadjusted" note; add the normalised variant (E7) as a separate block labelled post hoc.
- **T6. TUB dose-response.** §2.4 for Square256 best_valA, `last`, G455 best_valA, G455 `last`: per-seed ρ, seed mean, bootstrap CI, t95, registered direction, observed vs registered direction, registered label.
- **T7. Sensitivity matrix.** Rows: population × checkpoint × geometry; columns: key estimate and registered decision for each dataset. Include UIEB 800 / 787 / 707; HICRD site-adjusted, G482, last; TUB all-1,183, shift-removed, g_ach, G455, last.
- **T8. Matched-seed verification.** Shared init (MARINE-Net 214/214 parameters, AT-Trunk 197/197), data order and augmentation, with the control (different seeds: 2/56 identical conv tensors).
- **T9 (appendix). Deviation register.** §6.1.
- **T10 (appendix). Geometry and batch equivalence.** The batch-1 vs batch-4 table (12 checkpoints: output, PSNR, SSIM, relative MSE differences) and the memory table.

**Figures**

- **F1. Sign reversal.** x = dataset (UIEB-787, HICRD, TUB), y = mean IN − BN (dB); one point per seed per architecture; horizontal zero line; checkpoint and geometry as marker styles.
- **F2. Forest plot.** Per-seed ρ with bootstrap CIs for UIEB-787, HICRD D1, D2, G482, and the registered thresholds.
- **F3. Per-image scatter.** d vs imbalance for UIEB-787 and HICRD, per seed, HICRD coloured by site.
- **F4. Collinearity.** HICRD imbalance vs mean luma (ρ = 0.989) beside UIEB (ρ = −0.24).
- **F5. Error attribution.** Stacked or paired bars of ΔA, ΔC, ΔG, ΔR per dataset and architecture with CIs; a second panel normalised by BN MSE.
- **F6. TUB dose-response.** (a) BN and IN PSNR vs NTU separately (new descriptive; E0); (b) d vs NTU with series lines (grey) and scene means; (c) ΔA, ΔG, ΔR vs NTU; per architecture; registered direction annotated.
- **F7. Geometry.** Paired points (Square256 vs G455 and G482) for ρ̄ and mean d.

---

## 10. Recommended Results, Discussion and Limitations structure

**Results**
1. Evaluation sets and audited cohorts (T2).
2. Primary outcome: IN − BN PSNR per seed on each dataset (T3, F1). State the sign pattern first, then magnitudes.
3. Confirmatory association analyses: UIEB-787 (T4, F2, F3); HICRD (D1, D2, collinearity F4).
4. Robustness: checkpoint, geometry, sensitivity cohorts (T7, F7).
5. Exploratory: exact error attribution (T5, F5).
6. Exploratory: controlled turbidity series (T6, F6), including the failed registered directions and the opposite-sign association (A3).
7. Negative and inconclusive results: G1 failure, I_colour, HICRD confound check, MARINE-Net D2, G482 (one short subsection, not an appendix).

**Discussion**
1. What was measured (three sentences, no mechanism).
2. What is established vs not established (§4 status; per-dataset).
3. Candidate explanations **as hypotheses**, each with the test that would decide it and the existing evidence for and against: reference style; input distribution (X1 matching: gap unchanged in the high-cast subpopulation); BN running-statistics fit (untested); brightness-information loss (partly consistent with HICRD attribution, **not supported on TUB**).
4. Implications: "choose normalisation per target domain, with a held-out check" is the only defensible practical statement.
5. Relation to prior work (after E1).

**Limitations**: §6.2, as a bulleted block, with the three limitations a reviewer will lead with (2 related architectures; 3 seeds; reference types) first.

**Reproducibility statement**: hashes, environment, ledger, what is archived, what is not (D1–D5).

---

## 11. Honest assessment for a Q1 journal such as Pattern Recognition

**Strengths a reviewer will credit**
- Unusually rigorous process (registered hypotheses with hashes, matched-seed verification, overlap and leakage audits, cluster-aware inference, independent recomputation, a ledger of every attempt).
- A sign reversal that is stable across 3 datasets × 2 architectures × 3 seeds × 2 checkpoints × 2 geometries.
- Exact attribution that gives a quantitative picture of where error lies.
- Willingness to report failed gates and contradicted predictions.

**Weaknesses a reviewer will lead with**
1. **Generality:** two related architectures, one training set, three seeds.
2. **No mechanism and no method.** The paper characterises; it does not explain or improve. For Pattern Recognition this is the most likely ground for rejection.
3. **Evaluation-set validity:** HICRD references are computed; TUB's reference is a photograph, not a target; UIEB is a curated set. Absolute scores are not comparable with published work.
4. **The story changed after the registered claim failed.** The reviewer will read the registration, see G1 failure and the pivot, and discount the later exploratory work.
5. **PSNR-only primary outcome at 256²** with geometric distortion.
6. **No related work; novelty unverified.**
7. **Reproducibility from the repository alone is not currently possible** (D1, D2).

**Verdict**
- **In its present state: not suitable for submission to Pattern Recognition**, and I would expect rejection or a major revision on points 1–2.
- **A focused empirical paper** (an audited negative-and-positive result with no mechanism claim) **is defensible now**, once A1–A7, B1–B4, D1, E1 and E2 are corrected. The earlier plan named IEEE JOE and IVC as fallbacks; that remains reasonable.
- **Pattern Recognition becomes plausible only if** (a) E2r (or an equivalent mechanism experiment) yields a clear, reproducible explanation of the sign reversal, **and** (b) a third architecture family and ≥ 5 seeds are added, **and** (c) a literature review shows the finding is new. Even then the outcome is uncertain; I cannot estimate an acceptance probability from the repository.
- **Highest-value, lowest-cost actions:** E0, E7 (CPU only), D1, A1–A3 text, and the literature search.

---

## 12. Prioritised action list

| Priority | Action | Issues | Approval needed? |
|---|---|---|---|
| 1 | Rewrite the draft's Results and Discussion to include X1, E1r, TUB; remove the brightness-loss "prediction"; report A3 | A1, A2, A3 | no (writing) |
| 2 | Fix the contribution statements and title (A4–A7); add the registration-status table | A4–A7 | no |
| 3 | Add `.gitattributes`, re-add frozen artefacts, test a fresh clone on both OS | D1 | no experiments; a new commit |
| 4 | Unify uncertainty reporting and add a "observed vs registered direction" column | B1, B4, B5 | no |
| 5 | Run the literature search and write Related Work with verified citations | E1 | no |
| 6 | Archive checkpoints, manifests and outputs; write data statements | D2, E4 | no |
| 7 | E0 and E7 (CPU analyses of existing outputs) | B5, B6 | yes (exploratory) |
| 8 | Decide on E2r; decide on scope (E2, E3) | C1 | yes |
| 9 | Generate T1–T10 and F1–F7 from frozen files | E2 | no |
| 10 | Add `S3_results_report_erratum.md` for the +1.22 / +1.21 slip | A8 | no |

*End of report. Nothing was run, modified or committed in producing it; this file is the only addition.*
