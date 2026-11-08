# SOTA HICRD-2000 comparison protocol: Amendment 02 (2026-11-02)

**Amends.**
- The frozen draft, SHA-256 `24a096435fe7683c714be2507f07c18e3f86b1bf0f3404f3ca89666dc0052f7b`.
- Amendment 01, SHA-256 `5e269005c19dd1f6c3e34c397777553dad8a2f5a6cb1dd07f8c1a0b1bc791631`.

Neither file is edited.

**Source.** The final scientific audit (`SOTA_final_protocol_audit_report.md`) and the user's instructions of 2026-11-02.

**Written before any HICRD inference** of any external method. Its SHA-256 is recorded in `SOTA_HICRD_comparison_protocol_AMENDMENT_02.sha256.txt`.

**Precedence:** where it differs from earlier text, **this amendment governs**. The consolidated text is `SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md`.

## B1. WaterNet's original checkpoint is excluded from Group R

**Why.**
- The original WaterNet weights come from Li-Chongyi/Water-Net_Code, which has **no licence file** (GitHub API: no licence).
- The tnwei port's MIT licence covers the **port's code**, not the converted original weights.
- **No documented permission** to use these weights has been obtained, so research-use permission is **not assumed**.

**Consequences.**
- WaterNet (original weights) is **removed from Group R**.
- The downloaded file (`daa0eeb7…`) stays on disk, unused, and is recorded as excluded in the registry.
- The tnwei "replicated" weights are **not** substituted: they are a different model, and their own licence terms are not separately documented.
- WaterNet remains represented **only** by our LSUI retrain in **Group T**, which uses the MIT-licensed port's code and our own training.
- **Reinstatement** in Group R needs documented permission (for example written permission from the original authors, or a licence added to the original repository), recorded in a further amendment.

## B2. Group R composition and contrast count

**Group R: 6 comparators.**
- U-Shape Transformer;
- WWE-UIE (official UIEB weight);
- FUnIE-GAN (official Keras);
- NU²Net (with-URanker variant);
- CLAHE;
- UDCP (our implementation; guided-filter refinement).

**Group T: 2 comparators.** WaterNet-LSUI s0 and WWE-UIE-LSUI s0.

**Registered contrasts per metric (PSNR, SSIM):**

| Group | Comparators × arms | Contrasts |
|---|---|---|
| R | 6 × 4 | **24** |
| T | 2 × 4 | **8** |
| Total | | **32** (replacing Amendment 01's 36) |

**Arms:** MARINE-Net BN, MARINE-Net IN, AT-Trunk BN, AT-Trunk IN, three seeds each.

**Multiplicity.** These 64 comparator × arm × metric contrasts are descriptive and **unadjusted for multiplicity**; this is stated.

## B3. Status and permitted wording

**Group T (exploratory).**
- Each Group T baseline is **one training run (n = 1)**.
- Contrasts against it describe **these specific trained models only**. They **cannot support any training-seed-level superiority claim**: the baseline's seed variability is unknown, and the seed-level t95 reflects only our own three seeds.
- Group T is labelled **"exploratory"** in every table, figure and sentence.

**Group R (descriptive).** Each comparator is one released model, trained by its authors on different data. Results describe published models under this protocol, not architectures.

**Permitted wording.** "Under this protocol, the seed-mean PSNR of ⟨our arm⟩ on HICRD-2000 was X dB higher / lower than ⟨comparator⟩ (95 % cluster-bootstrap CI […]; per-seed values …)."

**Not permitted:**
- "outperforms", "superior", "better than", "state of the art";
- any claim generalising over training seeds for a comparison involving an n = 1 baseline;
- any claim about architectures from Group R.

## B4. Bootstrap draws by population (Stage 4 lists, reused exactly)

**Draw lists:**

| Population | Stage 4 analysis | Rows | Draws |
|---|---|---|---|
| ALL 2,000, primary | P1 | all | complete T30 clusters, unstratified, `default_rng(20261103)` |
| ALL 2,000, sensitivity | P1s | all | site-stratified, `default_rng(20261104)` |
| **KNOWN-site 1,839** | P2 | the known-site rows | stratified by the 6 sites, `default_rng(20261106)` |
| **Official test 300** | S_official_test300 | its rows | unstratified over its T30 clusters, `default_rng(20261103)` |

Implemented as `sota_hicrd_protocol.stage4_population_draws`. **Tests check that each list reproduces the frozen Stage 4 interval exactly.**

**Estimator.**
- For draw b with row multiset I_b, the bootstrap value of the seed-mean contrast is mean_{i ∈ I_b} [ (1/3) Σ_s d_{a,s}(i) ].
- Per-seed values: mean_{i ∈ I_b} d_{a,s}(i).
- Comparator means: mean_{i ∈ I_b} metric_m(i).
- **95 % percentile CI**, B = 10,000.
- The share of images with d > 0 is a point value, with no CI.

## B5. Metric computation details

- **Dtype:** every output is cast to a **float32** tensor [1, 3, H, W] in [0, 1] (`to_metric_tensor`) before the frozen `compute_psnr` / `compute_ssim` (`tools/eval_reproduce_baseline.py`, SHA-256 `6e861f53…`; its Gaussian window is float32). UDCP's float64 output and FUnIE-GAN's NumPy output are cast.
- **Decomposition:** E1r casts to float64 internally (unchanged).
- **Alignment:** every method's per-image rows must equal the manifest ids in manifest order (`assert_aligned`), otherwise STOP. This is needed for the shared draws and the per-image pairing with our models' Stage 4 rows.

## B6. Preprocessing, exactly

- **Primary:** `train_v2.preload` (uint8 [N, 3, 256, 256]).
- **G482:** `stage4_infer.preload_g` (PIL RGB, `resize((482, 256), BILINEAR)`, input and reference alike). Only for the applicable methods: our models (existing outputs), CLAHE, UDCP, WaterNet-LSUI (Group T). U-Shape, FUnIE-GAN, WWE-UIE (R and T) and NU²Net are "not applicable".
- **CLAHE:** receives the uint8 HWC image taken directly from the preloaded tensor, with no further resizing.
- **UDCP:** receives that image / 255 as float64.
- **FUnIE-GAN:** receives (uint8 / 127.5 − 1) as float32.

## B7. FUnIE-GAN execution

- CPU, TensorFlow 2.21 / tf_keras 2.21 in `venv_sota`, with `TF_ENABLE_ONEDNN_OPTS=0` set and recorded.
- **Determinism check:** the first batch is run twice and must be identical, otherwise STOP.

## B8. Visual panels

- **Group R panels:** input, reference, U-Shape, WWE-UIE, FUnIE-GAN, NU²Net, CLAHE, UDCP, and our MARINE-Net BN and IN (seed 0).
- **Group T panels (labelled "exploratory, n = 1 baselines"):** input, reference, WaterNet-LSUI, WWE-UIE-LSUI, and our MARINE-Net BN and IN (seed 0).
- The 12 selected images are unchanged (Amendment 01 A7).

## B9. Not changed by this amendment

- The primary clamp and the U-Shape min–max sensitivity (A1).
- UDCP r = 24 and its disclosure (A4).
- Float metrics (A2); input sizes (A3).
- The E1r caveat (A8); devices (A10); licences of the remaining methods (A11); U-Shape parameter count (A12).
- STOP rules and output paths.

## B10. Before execution approval (not yet done)

1. **Implement** the inference and analysis scripts for this protocol, with tests and a dry run.
2. **Freeze** them in an implementation addendum (script, test and dependency hashes; output paths absent).
3. **Obtain** separate execution approval.
