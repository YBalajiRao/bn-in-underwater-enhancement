# Stage 4 protocol amendment 01 (pre-inference), 2026-11-02

**Amends.** `S4_preregistration_FROZEN_DRAFT.md`, SHA-256 `2ea1f6ed3f8b20ff889f79c27f0dfaa795e2c2a7d9d66257b92236751b4a4934`. That file is **unchanged and remains in force** except where this amendment explicitly adds or overrides.

**This amendment.** Its SHA-256 is recorded in `S4_protocol_amendment_01.sha256.txt` in the same commit.

**Timing.** Written after the user's review and **before any model prediction on HICRD exists**. Only input images, descriptors, site labels and scene clusters were inspected.

**Reason.** The user identified that the colour-cast magnitude is confounded with acquisition site. Several further items needed to be specified exactly: effective cluster count, detector validation, cohort terminology, preprocessing, and the separation of uncertainty sources.

| # | Change | Type |
|---|---|---|
| A1 | Cohort terminology | clarification |
| A2 | Site strata and site-wise descriptor table | addition |
| A3 | Site-adjusted partial Spearman (secondary decision) and within-site analysis | addition |
| A4 | Site-stratified scene-cluster bootstrap | addition (the original unstratified CI is kept for the original pooled decision) |
| A5 | Interpretation rules: pooled vs within-site | addition |
| A6 | Effective cluster count: definition and terminology | correction of wording |
| A7 | Scene clusters renamed "overlap-connected reef-area clusters"; validation report | clarification |
| A8 | Preprocessing: exact primary transform; geometry-preserving sensitivity | addition |
| A9 | Reporting of uncertainty sources and scope of the BN/IN claim | clarification |

## A1. Cohort terminology

| Name | n | Definition |
|---|---|---|
| Official HICRD training split | 1,700 pairs | `trainA_paired` / `trainB_paired` |
| Official HICRD test split | 300 pairs | `testA` / `testB` |
| **Combined external HICRD evaluation cohort** | **2,000 pairs** | both splits. Eligible because no evaluated checkpoint was trained or selected on HICRD. |

**The 2,000 are never called the official test set.** The official-test result is a separate sensitivity analysis (S3 of the original protocol).

## A2. Site strata (input-only, fixed now)

**Assignment** (`paper1_uieb_runs/stage4_prep/hicrd_site_strata_T30.json`, SHA-256 `8bb91c66…`):
- official training images take the site in their filename;
- an official test image takes the site of the training images in its T30 cluster if that site is unique, otherwise **TEST_UNASSIGNED**.

**Every T30 cluster lies within one stratum** (verified: 0 clusters span two sites).

| Stratum | Images | T30 clusters | Imbalance min / Q1 / median / Q3 / max | IQR | Within-site eligible? |
|---|---|---|---|---|---|
| site1 | 25 | 16 | 0.174 / 0.365 / 0.555 / 0.626 / 0.642 | 0.261 | no (n < 50) |
| site2 | 16 | 9 | 0.407 / 0.461 / 0.535 / 0.629 / 0.668 | 0.168 | no (n < 50, clusters < 10) |
| site4 | 159 | 20 | 0.132 / 0.156 / 0.187 / 0.206 / 0.351 | 0.050 | **yes** |
| site5 (incl. 139 official-test images) | 1,084 | 39 | 0.094 / 0.165 / 0.233 / 0.285 / 0.454 | 0.119 | **yes** |
| site7 | 457 | 75 | 0.123 / 0.191 / 0.224 / 0.251 / 0.361 | 0.060 | **yes** |
| site8 | 98 | 26 | 0.251 / 0.366 / 0.413 / 0.509 / 0.627 | 0.143 | **yes** |
| TEST_UNASSIGNED (161 official-test images) | 161 | 3 | 0.173 / 0.216 / 0.222 / 0.233 / 0.247 | 0.018 | no (clusters < 10, IQR < 0.04) |

**Within-site eligibility rule.**
- A stratum is eligible if it has ≥ 50 images, ≥ 10 T30 clusters, and imbalance IQR ≥ 0.04.
- This was set from input descriptors only, before any model output. It is meant to require enough images, enough independent areas and enough cast variation for a within-stratum rank correlation to be meaningful.

## A3. Site-adjusted and within-site analyses (per architecture)

**Site-adjusted partial Spearman (secondary decision).**
- Rank-transform x and d_·s over the whole cohort.
- Residualise rank(x) and rank(d_·s) by OLS on the 7 stratum indicators (one-hot, no separate intercept).
- partial ρ_s^site = Pearson correlation of the residuals; partial ρ̄^site = mean over seeds.
- **Also reported, not decisive:** adjustment for strata + rank(b) + rank(c).

**Within-site Spearman, eligible strata only (descriptive).**
- For each eligible stratum: ρ_s computed within the stratum, ρ̄ = mean over seeds, and a 97.5 % CI from a cluster bootstrap within that stratum.
- Also reported: site-wise mean d per seed, and site-wise n and imbalance range (table A2).

## A4. Site-stratified scene-cluster bootstrap

**Unit.** One T30 overlap-connected cluster (all its images).

**Stratification.** Within each of the 7 strata, draw that stratum's clusters with replacement, as many as it has (16, 9, 20, 39, 75, 26, 3). The resample is the union over strata. Complete clusters are always kept together.

**Random numbers.** 10,000 resamples, `numpy.random.default_rng(20261104)`. The same draws are used for all seeds, both architectures, the pooled, site-adjusted and confound statistics. Ranks and residualisation are recomputed in every resample.

**Use.**

| Statistic | CI used for the decision | Also reported |
|---|---|---|
| Pooled ρ̄ (original primary) | **original unstratified cluster bootstrap** (`rng 20261103`), unchanged | stratified CI. Any disagreement between the two is disclosed. |
| Site-adjusted ρ̄^site | stratified | — |
| Within-site ρ̄ | cluster bootstrap within the stratum (`rng 20261105`) | — |

## A5. Interpretation rules (fixed now)

**Decisions.**
1. **POOLED decision:** the original rule (SUPPORTED / SUPPORTED, NOT CONFOUND-ROBUST / NOT SUPPORTED / INCONCLUSIVE), unchanged.
2. **SITE-ADJUSTED decision** on partial ρ̄^site (stratified 97.5 % CI):

| Outcome | Condition |
|---|---|
| WITHIN-SITE SUPPORTED | all three partial ρ_s^site > 0 **and** CI lower bound > 0 **and** partial ρ̄^site ≥ 0.10 |
| WITHIN-SITE NOT SUPPORTED | partial ρ̄^site ≤ 0, **or** CI upper bound < 0.10 |
| WITHIN-SITE INCONCLUSIVE | otherwise |

**Allowed statements.**
- **"Colour-cast magnitude is associated with the IN − BN advantage independently of site"** only if POOLED = SUPPORTED **and** SITE-ADJUSTED = WITHIN-SITE SUPPORTED. This is an intersection-union rule: no extra alpha is spent.
- POOLED SUPPORTED with site-adjusted NOT SUPPORTED or INCONCLUSIVE gives **"pooled association, possibly between-site; not attributable to colour cast independently of site."** Site, water body, depth, scene content and cast vary together in HICRD.
- **A positive pooled correlation alone is never interpreted as evidence that cast explains the advantage independently of site.**
- Within-site results per stratum are descriptive, reported as "k of 4 eligible strata with ρ̄ > 0; m with CI > 0".

**Multiplicity.** Bonferroni over the 2 architectures within each decision (97.5 % intervals). The two decisions answer different questions and are combined only by the intersection-union rule above.

## A6. Effective number of clusters (corrected wording)

The earlier "effective sample size ≈ 43" is the **effective number of clusters**, not an effective sample size of images:

**n_eff,clusters = (Σ_k m_k)² / Σ_k m_k²**, where m_k is the number of images in cluster k.

This is the number of equally sized clusters that would carry the same weight concentration; it assumes complete within-cluster dependence.

| Threshold | Clusters | Size min / median / max | Largest-cluster share | n_eff,clusters |
|---|---|---|---|---|
| T15 | 146 | 1 / 5 / 213 | 0.106 | 29.9 |
| **T30 (primary)** | **188** | 1 / 3 / 163 | 0.082 | **43.2** |
| T60 | 303 | 1 / 2 / 163 | 0.082 | 57.1 |

**Resampling.** Resampling unit and stratification as in A4.

## A7. Scene clusters (validation)

See `S4prep_scene_detector_validation.md`:
- no false split observed (sampled);
- 1 inconclusive possible direct false merge;
- transitive chaining merges different views of contiguous reef areas, which is conservative.

Clusters are hereafter "overlap-connected reef-area clusters". The manifest and clusters are unchanged.

## A8. Preprocessing (frozen)

**Primary transform (identical to Stages 1–3).** This is the frozen evaluator's `train_v2.preload`: torchvision 0.22.1 `T.Resize((256, 256))` with defaults (interpolation BILINEAR, antialias = True, max_size = None) on a PIL RGB image.
- Verified bit-identical to `PIL.Image.resize((256, 256), BILINEAR)` on a HICRD image.
- **The input and the reference receive the same transform.**
- Then uint8 → /255; model output clamped to [0, 1]; frozen PSNR and SSIM; TF32 off.

**Distortion.**

| Axis | Scale |
|---|---|
| Horizontal | 256 / 1842 = 0.13898 |
| Vertical | 256 / 980 = 0.26122 |

**Anisotropy 1.880:** scene content is compressed horizontally 1.88× more than vertically. UIEB inputs had mixed aspects and were squashed the same way. The models were trained on 256 × 256 squashes of LSUI.

**Geometry-preserving sensitivity (G).**
- Input and reference both resized with the same PIL BILINEAR (antialiased) to **482 × 256** (W × H). Scales 0.26167 / 0.26122; **anisotropy 1.0017**.
- All four architectures accept 256 × 482 tensors (verified with untrained models on random input; no data). PSNR and SSIM are computed on the 256 × 482 images.
- The models were not trained at this geometry, so G is a sensitivity analysis only. It is computed for best_valA, with the full pooled and site-adjusted analysis, and is never decisive.

**Evaluator.** The frozen evaluator is **not modified**. Stage 4 inference uses a separate script that calls `tv.preload` for the primary transform and its own documented resize for G.

## A9. Uncertainty sources and scope

**Each primary output table reports separately:**
- (i) per-seed estimates (s0, s1, s2) and their SD and t-interval (df = 2) as **training-seed variation**;
- (ii) the cluster-bootstrap CI as **image and reef-area sampling uncertainty conditional on these three trained models**;
- (iii) the pooled vs site-adjusted association.

The 10,000 bootstrap resamples do not increase the number of training replicates: n = 3 seeds per arm.

**Scope of the BN/IN claim.** Any statement concerns **the normalisation replacement as a whole** (BN → IN on all 40 sites). Batch coupling, running statistics and per-image normalisation are not isolated.

**Not changed.** Hypothesis, descriptor, confound definitions, threshold 0.10 (operational), the original decision rule and its CI, models, manifest (`29402a7f…`), clusters (`97f83af4…`), seeds and checkpoints.
