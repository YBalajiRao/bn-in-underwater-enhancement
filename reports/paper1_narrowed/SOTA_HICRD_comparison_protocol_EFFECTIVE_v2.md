# SOTA comparison on HICRD-2000: effective protocol v2 (2026-11-02)

**Composition.** The frozen draft (`24a096435fe7683c…`) with Amendment 01 (`5e269005c19dd1f6…`) and **Amendment 02** applied. It is consolidated for review and contains no decisions beyond those amendments. The SHA-256 of Amendment 02 is in its `.sha256.txt`.

**Precedence.** If this text and the source files differ, the source files govern.

**It supersedes** `SOTA_HICRD_comparison_protocol_EFFECTIVE_v1.md`, which is not edited.

**Status.**
- **Binding after approval. Not run.**
- **Label:** descriptive comparison. **Group T is exploratory.**
- **No state-of-the-art, superiority, ranking or architecture claim.**
- **Implementation:** `v2_tools/sota_hicrd_protocol.py`, `v2_tools/sota_classical.py`.
- **Tests:**
  - `tests_v2/test_sota_hicrd_protocol.py`: Amendment 01 checks;
  - `tests_v2/test_sota_hicrd_protocol_v2.py`: Amendment 02 checks;
  - `tests_v2/test_sota_classical.py`.

## 1. Cohort

**Data:**
- **Manifest:** HICRD-2000 (`29402a7f…`): 1,700 paired-train + 300 official test.
- **Clusters and sites:** T30 clusters (188) and sites (`8bb91c66…`).
- **Hash checks:** every image hash is verified before decoding.

**Populations:**
- **Primary:** ALL 2,000.
- **Sensitivity:** KNOWN-site 1,839.
- **Descriptive:** official test 300.

**Leakage and references.**
- **Leakage:** no overlap with UIEB, EUVP or LSUI (Stage 4 prep audit), and no evaluated method trained on HICRD.
- **References:** computed restorations, not ground truth (stated with every table).

## 2. Evaluation groups (never merged, never cross-contrasted)

**Group R: released checkpoints as published (descriptive; n = 1 each; not an architecture comparison)**

| Comparator | Checkpoint / implementation | Training | Licence |
|---|---|---|---|
| U-Shape Transformer | `generator_795.pth`, `c595c2aa…` (zip `32df84c6…`) | LSUI | MIT |
| WWE-UIE | official `best_model.pth`, `36c3e59e…` | UIEB | Apache-2.0 |
| FUnIE-GAN | official TF-Keras h5 `3054abfc…` + json `cbbfe3b1…` | EUVP paired | MIT |
| NU²Net | `NU2Net_ckpt.pth`, `2e1401a7…`; with-URanker variant | URankerSet / UIEB | **CC BY-NC 4.0** |
| CLAHE | OpenCV; Lab-L, clip 2.0, 8×8 | — | — |
| UDCP | **our implementation; guided-filter refinement** (§5) | — | — |

**Excluded from Group R:** **WaterNet (original authors' weights)**.
- The original repository has no licence, and **no documented permission** was obtained, so permission is not assumed.
- The file `daa0eeb7…` is kept on disk, unused.
- Reinstatement only through documented permission and a further amendment.

**Group T: retrained under our LSUI recipe (EXPLORATORY)**

| Model | Checkpoint | Seeds | Note |
|---|---|---|---|
| WaterNet-LSUI s0 | `af168d31…` | **1** | MIT port code, our training |
| WWE-UIE-LSUI s0 | `fc9ebbea…` | **1** | — |
| our 12 models | existing Stage 4 checkpoints | 3 each | per-image outputs reused unchanged |

**Group T contrasts describe these specific trained models only and cannot support any training-seed-level superiority claim.** The baselines' seed variability is unknown.

**Conditions for all:** research use only; no checkpoint or derived weights redistributed; all sources cited.

## 3. Input, output, precision

**Primary input.**
- `train_v2.preload`: identical 256 × 256 uint8 input and reference for every method.
- **Official resizes, not used (disclosed):** U-Shape `cv2.resize`; FUnIE-GAN PIL bicubic; NU²Net multiples of 16.

**Method inputs:**

| Method | Input |
|---|---|
| U-Shape, WWE-UIE, NU²Net | x / 255 (float) |
| FUnIE-GAN | x / 127.5 − 1 (float32) |
| CLAHE | uint8 HWC from the same tensor |
| UDCP | that image / 255 (float64) |
| WaterNet-LSUI | WB / GC / HE from the port's Python transforms (not the original MATLAB transforms; disclosed) |

**Output: primary.** The method's own final transform, then **clamp to [0, 1]**.
- FUnIE-GAN: (y + 1) / 2.
- U-Shape: `model(x)[3]`.
- NU²Net: its in-network min–max is kept.

**Output: sensitivity (U-Shape only).** Official per-image min–max, (y − min) / max(max − min, 1e-5), equal to `save_image(normalize=True)`. Separate table.

**Precision:**
- float outputs, with no 8-bit quantisation (CLAHE: uint8 / 255);
- **every output is cast to float32 [1, 3, H, W] for the frozen PSNR / SSIM** (whose window is float32);
- the E1r decomposition runs in float64.

**Execution:**

| Item | Setting |
|---|---|
| Precision | fp32; TF32 off (before any memory probe) |
| Batch | Group R: 1. Group T: 16 (frozen evaluator) |
| PyTorch methods | GPU |
| FUnIE-GAN | CPU in `venv_sota` with `TF_ENABLE_ONEDNN_OPTS=0`; first batch run twice and must match, else STOP |
| Cross-device tolerance | none claimed |

**Sensitivity geometry G482.** `stage4_infer.preload_g`: PIL `resize((482, 256), BILINEAR)`, input and reference alike.

| Methods | G482 |
|---|---|
| our 12 models (existing outputs), CLAHE, UDCP, WaterNet-LSUI | applicable |
| U-Shape, FUnIE-GAN (fixed 256) | **not applicable** |
| WWE-UIE, R and T (dimensions divisible by 4) | **not applicable** |
| NU²Net (multiples of 16) | **not applicable** |

**Alignment.** Every method's per-image rows must equal the manifest ids in manifest order, otherwise STOP.

## 4. Metrics

**Primary.**
- Per-image PSNR and SSIM from the frozen `compute_psnr` / `compute_ssim` (`tools/eval_reproduce_baseline.py`, `6e861f53…`).
- **Integrity check:** PSNR recomputed from MSE must agree within 1e-4 dB.

**Secondary.**
- The exact E1r decomposition (A, C, G, R; `24ef8a69…`; exactness 1e-10 · MSE + 1e-18) on the primary outputs.
- **Caveat:** NU²Net's in-network normalisation partly shapes its A and G terms. The U-Shape min–max sensitivity is not decomposed.

**Not used:** UIQM, UCIQE, LPIPS.

**Parameters.** Counted on the evaluated model:

| Method | Parameters |
|---|---|
| U-Shape | 31,589,775. The published 65.6 M and 22.8 M are not reproducible from the released code; cited as unverified |
| WWE-UIE | 734,059 |
| NU²Net | 3,145,923 |
| FUnIE-GAN | 4,215,843 |
| WaterNet-LSUI | 1,090,668 |

## 5. UDCP (our implementation)

**Paper elements implemented** (Drews et al., ICCVW 2013, read directly):
- G–B dark channel;
- Ω = 15 × 15;
- the airlight by He et al.'s rule on the G–B dark channel;
- t̃ = 1 − min_Ω min_{G,B} I_c / A_c, with no ω;
- J = (I − A) / max(t, t₀) + A.

**Our choices:**
- **guided-filter refinement, r = 24, ε = 1e-3, grey guide. This substitutes for the paper's spectral (soft) matting**;
- t₀ = 0.1;
- Ω kept at the paper's value, chosen for 640 × 480 images, applied to our 256².

**Reporting.**
- Reported as "UDCP (our implementation; guided-filter refinement)".
- **No claim of exact implementation fidelity.**
- IBLA is not used.

## 6. Statistical uncertainty

**Stage 4 draw lists, reused exactly** (B = 10,000, complete T30 clusters):

| Population | Stage 4 list | Seed | Stratification |
|---|---|---|---|
| ALL, primary | P1 | `default_rng(20261103)` | unstratified |
| ALL, sensitivity | P1s | `default_rng(20261104)` | site-stratified |
| KNOWN 1,839 | P2 | `default_rng(20261106)` | stratified by 6 sites |
| Official test 300 | S_official_test300 | `default_rng(20261103)` | unstratified over its clusters |

**Estimator.**
- **Seed-mean contrast:** for draw b with row multiset I_b, mean_{i ∈ I_b} [ (1/3) Σ_s d_{a,s}(i) ].
- **Per seed:** mean_{i ∈ I_b} d_{a,s}(i).
- **Comparator means:** mean_{i ∈ I_b} metric_m(i).
- **Interval:** 95 % percentile CI, conditional on the evaluated checkpoints.
- **Share of images with d > 0:** point value only.

**Seed-level.**
- t95 (df 2) for our arms only.
- n = 1 comparators have unknown seed variability.

**HICRD has about 43 effective clusters.**

## 7. Registered contrasts

**Per comparator:**
- mean PSNR and SSIM, with 95 % CI.

**For each comparator m and each of our 4 arms a** (MARINE-Net BN, MARINE-Net IN, AT-Trunk BN, AT-Trunk IN; 3 seeds):
- the per-image difference d_{a,s}(i) = metric_{a,s}(i) − metric_m(i), for each seed s;
- reported per seed, with:
  - the seed mean;
  - the seed t95;
  - the 95 % CI;
  - the share of images with d > 0.

**Count:**

| Group | Contrasts per metric |
|---|---|
| R | 6 × 4 = **24** |
| T | 2 × 4 = **8** |
| Total | **32** |

**Rules:**
- **No cross-group contrast.**
- Descriptive and **unadjusted for multiplicity** (stated).
- **Secondary:** E1r component differences per contrast.

**Permitted wording.** "Under this protocol, the seed-mean PSNR of ⟨arm⟩ on HICRD-2000 was X dB higher / lower than ⟨comparator⟩ (95 % CI […]; per-seed …)."

**Forbidden:**
- "outperforms", "superior", "better than", "SOTA";
- seed-generalising claims with n = 1 baselines;
- architecture claims from Group R;
- p-values, significance tests, bold-best formatting.

## 8. Visual comparison (fixed from inputs only)

**Rule:**
- **Source:** `descriptors_per_image.csv` (`ca895bab…`, column `imbalance`).
- **Site picks:** per site (1, 2, 4, 5, 7, 8), the image closest to that site's median; ties go to the lowest manifest index.
- **Random picks:** the first 6 new indices of `default_rng(20261203).permutation(2000)`.

**Selected:**

| # | Id | Selection |
|---|---|---|
| 1 | `site1_good_2` | site median |
| 2 | `site2_good_7` | site median |
| 3 | `site4_good_140` | site median |
| 4 | `site5_good_641` | site median |
| 5 | `site7_good_303` | site median |
| 6 | `site8_good_42` | site median |
| 7 | `site5_good_708` | random |
| 8 | `site5_good_257` | random |
| 9 | `site7_good_164` | random |
| 10 | `site7_good_225` | random |
| 11 | `site5_good_173` | random |
| 12 | `test94` | random |

**Panels:**
- **Group R:** input, reference, U-Shape, WWE-UIE, FUnIE-GAN, NU²Net, CLAHE, UDCP, and our MARINE-Net BN and IN (seed 0).
- **Group T, labelled "exploratory, n = 1 baselines":** input, reference, WaterNet-LSUI, WWE-UIE-LSUI, and our MARINE-Net BN and IN (seed 0).
- 256², with per-image PSNR under each.

No swapping after outputs are seen.

## 9. Integrity and STOP rules

**STOP on any of:**
- a hash mismatch (manifest, strata, image, checkpoint);
- a non-finite output;
- inexact decomposition;
- PSNR inconsistency > 1e-4 dB;
- row misalignment;
- an existing output folder;
- non-determinism in FUnIE-GAN's first batch;
- a method needing more than I/O wrappers. It is excluded and reported, not patched.

**Execution records:**
- a ledger entry per attempt;
- **no-spill rule:** peak allocated × 1.10 and peak reserved ≤ free dedicated memory.

**Eligibility gate**, all must hold:
1. checkpoint hashed with provenance recorded;
2. it runs in the recorded environment;
3. an author-example sanity check;
4. no HICRD in its training data;
5. **documented** research-use licence or permission.

## 10. Not done

- No retraining of external methods.
- No tuning on HICRD.
- No comparison with literature numbers.
- No UIEB-787 or TUB use. UIEB-787 needs its own amendment; TUB is descriptive only.

## 11. Output paths (must be absent at run time)

- `paper1_uieb_runs/sota_hicrd_released_square256/`
- `paper1_uieb_runs/sota_hicrd_released_square256_ushape_minmax/`
- `paper1_uieb_runs/sota_hicrd_released_G482/`
- `paper1_uieb_runs/sota_hicrd_retrained_square256/`
- `paper1_uieb_runs/sota_hicrd_retrained_G482/`
- `paper1_uieb_runs/sota_hicrd_analysis/`
- `paper1_uieb_runs/sota_hicrd_visual/`

## 12. Before execution approval

1. Implement the inference and analysis scripts, with tests and a dry run.
2. Freeze them in an implementation addendum.
3. Obtain separate execution approval.
