# TUB Stage 5: external-evaluation protocol, FROZEN DRAFT (EXPLORATORY), 2026-11-02

**Status.**
- This is an **exploratory external evaluation designed after observing Stages 3 and 4, S5-X1 and S5-E1r.**
- Freezing it before inference prevents analytic flexibility. It does **not** make it confirmatory, and no result may be described as confirming a mechanism.
- Its SHA-256 is recorded in `TUB_stage5_preregistration_FROZEN_DRAFT.sha256.txt`.
- **Implementation and inference require separate approval.** The implementation will be frozen by an addendum (script, test and configuration hashes) that cannot change anything below.

**Fixed context.** G1 = `stop_training_plans`; Stages 1–4, S5-X1, S5-E1r and every earlier protocol are unchanged. No training, no Val-B, no E2r.

**Data.** TUB (vapaau/TUB @ `6214f00807df5dbbfd389cddb4cecc41e9d62110`, CC-BY-4.0; Ismiroglou et al., arXiv 2606.26295); audit: `TUB_dataset_audit_and_suitability_report.md` (commit `050c773`).

**Terminology.**
- The Stage-01 image is the **corresponding low-turbidity reference** (1.0–8.7 NTU), **not absolute ground truth**.
- Turbidity is always the **measured NTU**, never the stage number.

## 1. Questions (exploratory)

d = PSNR(IN) − PSNR(BN) per pair and matched seed. **Negative d means BN is better.**

| | Question |
|---|---|
| Q1 | Does d change systematically with measured NTU **within the same scene and camera**? Stated direction: **d decreases (BN's advantage grows) with NTU.** |
| Q2 | Does the IN − BN difference in **global brightness-offset error (ΔA)** increase with NTU? (stated direction: increases) |
| Q3 | Does the difference in **global gain or contrast error (ΔG)** increase with NTU? (stated direction: increases) |
| Q4 | Does the **colour-offset difference (ΔC)** change with NTU? **No direction stated.** ΔR is also reported, with no direction stated. |
| Q5 | Do MARINE-Net and AT-Trunk differ? (descriptive) |
| Q6 | Is each pattern consistent across the 3 matched seeds? |

## 2. Cohorts (frozen; files in `paper1_uieb_runs/tub_stage5_prep/`)

| Population | Pairs | Scenes | Camera series | File |
|---|---|---|---|---|
| All valid pairs (before exclusions) | 1,183 | 33 | 132 | `TUB_ALL_1183.json` |
| **Primary paired cohort** | **1,153** | 33 | 129 | `TUB_PRIMARY_1153.json` |
| **Reliable-NTU subset** (for every NTU-based endpoint) | **1,022** | 29 | 114 | `TUB_NTU_1022.json` |

**Pair definition.** Each pair is a Stage 02–10 input with the Stage-01 image of the **same scene and same camera**. Per-file SHA-256 is recorded in each manifest. The HF train/test split is **not** used.

**Exclusions** (`TUB_exclusions_and_flags.csv`; nothing silently removed):

| Scene / camera / stage | Reason | Effect |
|---|---|---|
| 11 / 1 / 02 | input byte-identical to its Stage-01 reference | −1 pair from primary |
| 21 / 3 / 09 and 10 | byte-identical to each other; which one is genuine is unknown | −2 pairs |
| 21 / 4 / 02–10 | Stage01-vs-02/03 shift 3–4 px at 640 px width (> 3) | −9 pairs (series) |
| 28 / 4 / 02–10 | camera moved (−51 / −63 px; visually confirmed) | −9 pairs |
| 29 / 4 / 02–10 | shift −4 / −6 px | −9 pairs |
| scenes **10, 15, 28, 33** (all cameras) | NTU metadata copied (Exp15 = Exp33 in all stages; Exp10, Exp28 and phantom Exp34 share stages 1–3/4) | kept in primary; **excluded from all NTU-based endpoints** (−131 more pairs → 1,022) |
| 26 / 3 / 10 and 33 / 1–4 / 10 | file missing in the release | no pair exists (counted in the 1,320 → 1,183 reconciliation) |

**Representativeness.** 33 scenes from **one tank, one lighting set-up and one camera model** are **not** 33 independent underwater domains.

## 3. Models and preprocessing (frozen)

**Checkpoints.** The 12 best_valA fp32 checkpoints: `marinenet_{bn,in}_s{0,1,2}`, `attrunk_{bn,in}_s{0,1,2}`. Hashes are verified against `result.json` before loading and re-hashed after the run.

**Image format.** TUB images: all 1920 × 1080 RGB JPEG (verified), with input and reference of identical size.

**Primary geometry.** `train_v2.preload`, unchanged:
- PIL RGB, then torchvision `Resize((256, 256))` (BILINEAR, antialias = True), applied to **input and reference identically**;
- uint8 → /255; batch 16; fp32; TF32 off (cuDNN and matmul); output = clamp(model(x)["enhanced"], 0, 1);
- **Distortion:** anisotropy 1.778 (16:9 squashed to 1:1), the same convention as Stages 1–4. The models were trained at 256².

**Sensitivity geometry G455.** Input and reference both PIL BILINEAR to **455 × 256** (anisotropy 1.0003; all four architectures accept 256 × 455, verified on random tensors).

**Metrics.**
- PSNR and SSIM from the frozen evaluator functions (`tools.eval_reproduce_baseline` via `train_v2`), computed per image on the same clamped outputs.
- Error components from `v2_tools/s5_e1r_error_decomposition.decompose` and `check_exact`, **imported unchanged** (SHA-256 `24ef8a69…`). MSE = A + C + G + R is verified per image with tolerance 1e-10·MSE + 1e-18, otherwise STOP.
- **Integrity check:** PSNR from the decomposition MSE must equal the frozen PSNR within 1e-4 dB.

## 4. Estimand and estimator

**Notation.** s = scene, c = camera, t = stage. Within scene s, the measured NTU of stage t is N_{s,t} (shared by all cameras of a scene). Per pair and seed k, the outcome is y_{s,c,t,k} ∈ {d, ΔA, ΔC, ΔG, ΔR}, each the IN model's value minus the BN model's value for the same seed.

**Candidate estimators compared.**

| Candidate | Uses within-series change? | Robust to unequal NTU spacing and non-linearity? | Assumptions / problem |
|---|---|---|---|
| (a) pooled Spearman over all pairs | no (mixes between-scene brightness / baseline) | rank-based | **rejected:** between-scene confounding |
| (b) pooled OLS of y on N | no | no | rejected: same as (a), plus linearity |
| (c) mixed model, random scene/camera intercepts and slopes | yes | no (linear) | distributional assumptions; unstable with 29 clusters and 9 points per series |
| (d) within (fixed-effects) OLS slope: demean y and N within each series, then pooled slope | yes | no (linear; weights series by NTU spread) | interpretable (dB per NTU); sensitive to saturation at high NTU |
| **(e) within-series Spearman, averaged to scene then over scenes** | **yes** | **yes** (ranks of measured NTU) | needs ≥ 5 points per series; ignores magnitude |

**Primary estimator (e).**
1. For each scene-camera series with ≥ 5 valid stages in the cohort: ρ_{s,c,k} = Spearman over t of (N_{s,t}, y_{s,c,t,k}), average ranks.
2. Scene value: ρ_{s,k} = mean over the scene's valid cameras.
3. Estimate: **ρ̂_k = mean over scenes** (each scene weighted equally).
4. Reported per architecture and per seed k ∈ {0, 1, 2}, plus the seed mean.

**Why (e).**
- It uses only **within-series** variation, so the scene and camera baseline (brightness, arrangement) cancels.
- It uses measured NTU ranks, so **no equal spacing or linearity** is assumed.
- Scenes are the independent unit and weighted equally, so cameras and stages are not treated as replications.

**Secondary estimator (d).** The within-series OLS slope (dB or MSE units per NTU), for effect size.

**Supplementary analyses** (all 33 scenes, using the **reference-required achromatic correction**):
- g_ach = |m(ref) − m(in)| replaces N, removing the dependence on NTU metadata;
- the same estimator on the primary cohort;
- labelled supplementary: g_ach is an image property, not an optical measurement.

## 5. Uncertainty

**Scene-cluster bootstrap.**
- 10,000 resamples of scenes with replacement: 29 for NTU endpoints, 33 for g_ach and descriptive endpoints.
- `numpy.random.SeedSequence(20261120)`, one draw list per cohort. It is **shared by all seeds, both architectures and all endpoints** within that cohort.
- Each resampled scene enters with its complete set of series and stages.
- 95 % percentile CIs.

**Seed level.** Per-seed estimates, their SD and a t-interval (df = 2) are reported **separately**. Three seeds are the only training replication; the bootstrap adds none.

**Reported counts.** Contributing scenes, valid camera series, and missing observations per scene.

**Power and uncertainty.**
- **29 scene clusters, with about 4 correlated cameras each.** The effective number of independent units is at most 29, and lower if cameras within a scene behave alike.
- With 9 points per series, the series-level Spearman is coarse (resolution about 0.017).
- Moderate effects may give wide CIs; **null results are not evidence of no effect.**

## 6. Endpoints and interpretation (no thresholds)

| Endpoint | Cohort | Estimator | Stated direction |
|---|---|---|---|
| **E-T1 (primary): d vs NTU** | NTU 1,022 | (e) | negative |
| E-T2: ΔA vs NTU | NTU 1,022 | (e) | positive |
| E-T3: ΔG vs NTU | NTU 1,022 | (e) | positive |
| E-T4: ΔC vs NTU; ΔR vs NTU | NTU 1,022 | (e) | none |
| E-T5: overall mean d and mean ΔA / ΔC / ΔG / ΔR per seed | primary 1,153 | scene-weighted mean (mean over pairs within scene, then over scenes) | descriptive |
| Secondary: within-series OLS slopes for E-T1 to E-T4 | NTU 1,022 | (d) | as above |
| Supplementary: E-T1 to E-T4 with g_ach in place of NTU | primary 1,153 | (e) | as above |

**Interpretation rule (per architecture).** A pattern is described as **"consistent"** only if the per-seed sign agrees with the stated direction in all 3 seeds **and** the 95 % CI of the seed mean excludes 0. Otherwise it is **"not established"**.

**Statements about mechanism are not permitted.** E-T2 and E-T3 are attributions of where the error lies, not explanations.

## 7. Sensitivity analyses (pre-specified)

1. All 1,183 valid pairs, including the flagged pairs: d and components, descriptive; NTU endpoints on the 1,183 minus the NTU-copied scenes.
2. Primary minus series with *any* Stage01-vs-02/03 shift > 1 px: Exp1 Cam2, Exp11 Cam4, Exp14 Cam4, Exp15 Cam1, Exp27 Cam2, Exp27 Cam3, Exp33 Cam4.
3. Geometry G455, best_valA: full E-T1 to E-T5.
4. Checkpoint last.pth: full E-T1 to E-T5.
5. **Turbidity ranges as defined by the dataset authors** (fixed before any prediction): low 0–10, medium 10–20, high > 20 NTU. Mean d and Δ components per range per seed, descriptive.
6. Scene-wise and camera-wise descriptive tables.
7. **Near-blank behaviour:** E-T5 per NTU range, plus input luma-SD distribution per range (descriptive).

## 8. Failure criteria (STOP and report)

- any checkpoint, manifest or file hash mismatch;
- any non-finite output;
- the decomposition not exact;
- frozen PSNR vs decomposition PSNR > 1e-4 dB;
- a cohort count different from 1,183 / 1,153 / 1,022;
- an output folder that already exists.

## 9. Limitations (to be reported with any result)

- **Post hoc, exploratory design.**
- **References:** low-turbidity references with their own tank cast and vignetting, 1.0–8.7 NTU; not enhancement targets. LSUI-trained models also colour-correct, and that is penalised against these references.
- **Measurement:** NTU metadata defects (4 scenes); no EXIF, so camera auto exposure and white balance are UNKNOWN; brightness is not a pure function of NTU.
- **Content:** near-blank inputs at high NTU (restoration is prior-driven there); one tank and one lighting set-up (low ecological validity).
- **Statistical:** 29–33 scene clusters; 3 seeds; two related architectures; the BN → IN swap changes several properties at once.
- **Geometry:** 16:9 → 1:1 distortion in the primary geometry.
- **Comparability:** absolute PSNR is **not comparable** with UIEB or HICRD (different reference standards).

## 10. Implementation plan (for approval; not started)

**Scripts.**
- `v2_tools/tub_stage5_infer.py`: verifies cohorts and hashes; runs `train_v2.preload` (or the G455 resize); 12 models; per pair PSNR, SSIM, A, C, G, R, R_low, R_high, MSE; exactness and PSNR-consistency checks; checkpoint re-hash; ledger label `tub_stage5`; `--dry-run`.
- `v2_tools/tub_stage5_analysis.py`: estimators (e) and (d), scene bootstrap, E-T1 to E-T5, sensitivities 1–7.

**Unit tests** (`tests_v2/test_tub_stage5.py`):
1. cohort loading reproduces 1,183 / 1,153 / 1,022, with scene and series counts;
2. pairing is the same scene and camera, never Stage01 as input;
3. estimator (e) on synthetic series with a known monotone relation returns ρ = ±1, and is invariant to adding a series-specific constant (baseline cancellation) and to monotone NTU re-spacing;
4. estimator (d) returns the known slope;
5. series with < 5 points are dropped and reported;
6. scene bootstrap draws complete scenes with equal multiplicity; draws shared across seeds, architectures and endpoints; deterministic with the seed;
7. the decomposition is reused unchanged (hash check) and is exact at 256² and 256 × 455;
8. frozen PSNR from the reused functions equals `score_model` on synthetic tensors;
9. the per-seed / seed-mean / t95 bookkeeping;
10. the "consistent" rule on synthetic outcomes;
11. checkpoint and weights unchanged after a forward pass (random tensors);
12. the dry run writes nothing.

**Hash-freeze checklist (addendum before execution).** Scripts, tests and test output; the three cohort manifests and the exclusions CSV; the reused `s5_e1r_error_decomposition.py`; `train_v2.py`, `paper1_final_eval*.py` and the model files; `original_repo/marine_net_model.py` (pinned); the TUB download manifest; this protocol's hash; the output paths (must be absent).

**Candidate commands** (names fixed now; run only after approval):
```
python v2_tools/tub_stage5_infer.py --checkpoint best_valA --geometry square256 --out paper1_uieb_runs/tub_stage5_best_valA
python v2_tools/tub_stage5_infer.py --checkpoint last      --geometry square256 --out paper1_uieb_runs/tub_stage5_last
python v2_tools/tub_stage5_infer.py --checkpoint best_valA --geometry aspect455 --out paper1_uieb_runs/tub_stage5_best_valA_G455
python v2_tools/tub_stage5_analysis.py --run paper1_uieb_runs/tub_stage5_best_valA --out paper1_uieb_runs/tub_stage5_analysis_best_valA   (likewise for last, G455)
```
Inference covers all 1,183 valid pairs once; cohorts are selected in the analysis.

**Compute estimate.**

| Run | Estimate | Peak GPU |
|---|---|---|
| square256, best_valA or last | decode 2 × 1,183 JPEGs at 1920 × 1080 (≈ 2–3 min) + 12 models × 1,183 at 256² (≈ 4–5 min) + decomposition: **≈ 8–10 min each** | ≈ 6 GB (as Stage 4 A/B) |
| G455 | **≈ 15 min** | ≈ 8 GB (Stage 4 run C peaked at 7.9 GB of 8.2 at 256 × 482) |
| **Total GPU** | **≈ 35 min** | |
| Analysis (CPU, per run) | ≈ 5 min (series ρ computed once; the bootstrap re-weights scenes) | — |

The implementation itself (scripts, tests, freeze) takes about 2–3 h of work.
