# Stage 4 pre-registration: statistical protocol FROZEN (implementation pending approval), 2026-11-02

**Status.**
- The statistical protocol below is fixed **before any model prediction on HICRD exists**. Its SHA-256 is recorded in `S4_preregistration_FROZEN_DRAFT.sha256.txt` in the same commit.
- After the user approves, the inference and analysis scripts will be written to implement exactly this protocol, tested, hashed and recorded in a freeze addendum **before inference**. The addendum may add only implementation hashes and preflight results; it changes nothing below.

**Fixed context.**
- G1 = `stop_training_plans`; Stages 1–3 are unchanged.
- Inference only, existing checkpoints, no training.

## 1. Recommended dataset and justification

**Primary Stage 4 dataset: HICRD, all 2,000 paired images** (official test 300 + official paired train 1,700).
- Source: CSIRO DAP collection 49488 v2, DOI 10.25919/ahhq-0827.
- Archive SHA-256 verified against CSIRO's checksum.
- Licence: CSIRO Data Licence (non-commercial, attribution, no redistribution).

**Why HICRD:**
1. **Independent of model training and selection.** No exact or perceptual match to LSUI train or Val-A, UIEB-890 or EUVP for any of the 2,000 pairs. No evaluated checkpoint has seen HICRD.
2. **Documented provenance and licence**, unlike the local EUVP mirror.
3. **Real underwater inputs** from a different location, camera and year than LSUI and UIEB, so it is a genuine cross-dataset shift.
4. **Usable colour-cast variation once the paired-train split is included**: imbalance 10th–90th percentile 0.14–0.36, with sites 1, 2 and 8 at medians 0.41–0.55.

**Why not the alternatives:**
- **EUVP-515:** 482/515 pairs are in the models' training data (LSUI).
- **HICRD official test alone:** about 4 scenes, narrow cast range.
- **LSUI test:** same dataset as training.
- **Downloads:** not needed (see the part 1 report, §6).

**Why the official HICRD split is not used as the population boundary:**
- Our models never saw any HICRD image, so the train/test label carries no exposure information here.
- The official test set is almost entirely a few scenes (clusters of 88, 90 and 56 test images).
- 6 scene clusters span HICRD's own test and train.

**Known weaknesses, accepted and handled below:**
1. **References are computational restorations** (the authors' physical model with measured Kd applied to the input, manual filtering), not ground truth. PSNR here measures agreement with that restoration.
2. **Severe pseudo-replication:** continuous reef-transect footage; 188 scene clusters, cluster-weighted effective n ≈ 43.
3. **One location and one camera.**
4. **Cast range narrower than UIEB:** median 0.23 vs 0.26; 90th percentile 0.36 vs 0.53.
5. **Squash to 256 × 256** from aspect ratio 1.88. This is the same squash as UIEB in Stages 1–3; geometry is not preserved.

## 2. Hypothesis (directional, one descriptor)

**H3-S4.** For each architecture separately (MARINE-Net, AT-Trunk), the per-image IN − BN PSNR advantage **increases** with the input's native-resolution channel-mean imbalance.

## 3. Data, populations and units

**Manifest.** `paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json`, SHA-256 `29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d`. Each entry records split, site (UNKNOWN for the official test), input and reference paths with SHA-256, and scene cluster IDs at T15, T30 and T60.

**Scene clusters.**
- Built from input images only (no model output): ORB-RANSAC same-scene graph (`v2_tools/stage4_scene_overlap.py`, CLAHE, 500 ORB features, ratio 0.75, RANSAC 5 px). Connected components at **≥ 30 inliers (primary)**, 15 and 60 (sensitivity).
- Calibration: visually labelled same-scene pairs 122–291 inliers; cross-site pairs ≤ 5.
- **Primary: 188 clusters**, largest 163, 57 singletons.

**No exclusions.** There are no exact or perceptual duplicates with any training set. Within-HICRD repetition is handled by clustering, not by removal. Nothing is silently removed.

| Population | Definition | Role |
|---|---|---|
| **PRIMARY** | all 2,000 pairs; bootstrap unit = T30 scene cluster | decision |
| S1 | all 2,000, clusters at T15 / T60 | sensitivity (clustering) |
| S2 | **cluster-level:** per T30 cluster, the mean of x and of d per seed; Spearman over the 188 clusters; bootstrap over clusters | sensitivity (pseudo-replication) |
| S3 | official test 300 only, cluster bootstrap | sensitivity (conventional split; expected to be very underpowered) |
| S4 | within-site, descriptive: Spearman within site4, site5, site7, site8 and the official test stream separately | sensitivity (site confounding) |

## 4. Models and inference

**Models.** The same 12 checkpoints as Stage 3: `marinenet_{bn,in}_s{0,1,2}` and `attrunk_{bn,in}_s{0,1,2}`.
- All fp32, matched seeds (shared initial weights, data order and augmentation). **Primary:** best_valA. **Cross-check:** last.pth.
- **Precision:** there is no AMP/fp32 confound among these 12. The AMP-BN vs fp32-IN confound concerns only the colour-head arms, which are not used.
- **Mechanistic confound (recorded):** BN → IN also removes batch coupling and the train/inference statistics mismatch.

**Preprocessing (frozen evaluator, unchanged).**
- Input and reference converted to RGB.
- Both resized with the same bilinear `Resize((256, 256))` (aspect not preserved: 1842 × 980 → 256 × 256).
- ToTensor; output clamped to [0, 1]; PSNR and SSIM from the frozen evaluator; TF32 off.
- Every per-image SHA-256 is checked before decoding.
- **Descriptors** (§5) are computed on the native-resolution input file.

**Outcome.** d_is = PSNR_IN(i, s) − PSNR_BN(i, s).

## 5. Primary analysis (exact, per architecture)

**Variables (native input, RGB / 255, float64).**
- x_i = max(μ_R, μ_G, μ_B) − min(μ_R, μ_G, μ_B).
- Confounds: b_i = mean of Y = 0.299 R + 0.587 G + 0.114 B; c_i = population SD of Y.

**Statistics.**
1. ρ_s = Spearman(x, d_·s) over **all 2,000 images**, for each matched seed s ∈ {0, 1, 2}. ρ̄ = mean of the three. Image × seed rows are never pooled.
2. **Cluster bootstrap.**
   - 10,000 resamples: draw the 188 T30 clusters with replacement, `numpy.random.default_rng(20261103)`, and take all images of the drawn clusters (with multiplicity).
   - The **same cluster draws** are used for all seeds, both architectures and the partial statistic. Ranks are recomputed in each resample.
   - **Two-sided 97.5 % percentile CI** (1.25th–98.75th): Bonferroni over 2 architectures.
3. **Partial Spearman.** Ranks of x, d_·s, b, c. Residualise rank(x) and rank(d_·s) on [1, rank(b), rank(c)] by OLS. Take the Pearson correlation of the residuals; partial ρ̄ = mean over seeds. Same cluster bootstrap.
4. **Seed level, descriptive:** SD of ρ_s and a t-interval with df = 2.

## 6. Decision rule (per architecture, PRIMARY, best_valA). Identical to Stage 3, with the cluster bootstrap

| Outcome | Condition (evaluated in order) |
|---|---|
| NOT SUPPORTED | ρ̄ ≤ 0, **or** the 97.5 % CI upper bound < 0.10 |
| SUPPORTED | ρ_s > 0 in all 3 seeds **and** CI lower bound > 0 **and** ρ̄ ≥ 0.10 **and** confound-robust (partial-ρ̄ CI lower bound > 0 **and** partial ρ̄ ≥ 0.5 ρ̄) |
| SUPPORTED, NOT CONFOUND-ROBUST | the first three SUPPORTED conditions hold, the confound condition fails |
| INCONCLUSIVE | otherwise |

**Threshold.** 0.10 is an **operational threshold**, not an established minimum meaningful effect.

**Multiplicity.** Bonferroni across the 2 architectures through the 97.5 % intervals. Sensitivity analyses carry no decision.

**Interpretation constraints, fixed now.**
- **The narrow cast range:** a NOT SUPPORTED or INCONCLUSIVE result on HICRD is **not** evidence against Stage 3. It may reflect range restriction, computational references or single-site domain shift.
- **Site confounding:** a SUPPORTED result driven mainly by between-site differences (sites 1/2/8 vs others) will be reported as such, based on S4. Site, water type and scene content are confounded with cast in HICRD.

## 7. Supplementary (descriptive only)

- **Fixed bins** from the frozen-90 H3 cut-points (0.148396 / 0.303392): mean d per bin per seed.
- **Associations:** ρ(x, b), ρ(x, c), ρ(d, b), ρ(d, c).
- **Mean d per site.**
- **The same analysis on last.pth.**

## 8. Three-seed limitation

Three matched seeds per arm. Every result concerns these 12 trained networks, not the training procedure. Seed-level testing would need about 7 (MARINE-Net) or 3–6 (AT-Trunk) seeds; see the S3 protocol, §12.

## 9. Execution plan and resources (after approval only)

1. Write `v2_tools/stage4_infer.py`: 12 models, explicit HICRD pair list with hashes, no UIEB phrase, path marker "HICRD", ledger label `hicrd_stage4`.
2. Write `v2_tools/stage4_h3_analysis.py`: cluster bootstrap, S1–S4.
3. Preflight:
   - synthetic tests;
   - reproduction of the Stage 3 primary ρ_s from the Stage 3 per-image file, using image-level units (tolerance 1e-6);
   - inference-script reproduction of 5 HICRD-free UIEB images vs Stage 3 per-image PSNR (tolerance 1e-4 dB).
4. Freeze addendum with the script hashes. Then inference with best_valA and last, then analysis, then report and stop.

**Estimated resources.**

| Item | Estimate |
|---|---|
| Image loading | 2,000 pairs at 1842 × 980 PNG (about 2.9 MB each, about 11.5 GB read): decode and resize about 5–8 min per run |
| Scoring | 12 models at 256 × 256: about 4–5 min per checkpoint |
| **Total GPU** | **about 25–30 min** for best_valA + last (RTX 3070; peak VRAM < 3 GB) |
| RAM | about 0.8 GB of image tensors |
| Analysis | cluster bootstrap about 5–10 min CPU |
| Disk | under 5 MB of outputs; images already extracted (about 11.5 GB, outside git) |

All under the 2-hour limit.

## 10. Decision table

| Stage 4 result (primary) | Interpretation | Next step (user approval required) |
|---|---|---|
| Both SUPPORTED, confound-robust, and S4 shows within-site association | Cross-dataset generalisation for these models, not only site-driven | Write-up with Stages 3 + 4. Optional confirmatory seed replication (about 48–54 GPU-h). |
| SUPPORTED, but S4 shows the association only between sites | Consistent with the hypothesis but confounded with site and water | Report as site-level; no strong cross-dataset claim |
| One architecture SUPPORTED | Architecture-specific transfer | Report; no training |
| NOT SUPPORTED / INCONCLUSIVE | Not evidence against Stage 3 (range, reference type, single site) | Report as a limitation of cross-dataset evidence. Consider a dataset with real references and wide cast range, if one with clear licence and independence exists. |
| SUPPORTED, NOT CONFOUND-ROBUST | Not separable from brightness or contrast | Reframe as general degradation dependence |
