# Paper 1: HICRD-2000 comparison with released and retrained methods — consolidated report (2026-11-03)

**Binding protocol:** effective v2 (`2d29442248a57ac6…`), Amendment 02 (`23cf50fb…`) and Amendment 03 (`0bd811b3…`; Group T 482 × 256 batch 4).

**Implementation freezes:** `SOTA_HICRD_implementation_freeze_addendum.md` (`876a8d14…`) and `…_addendum_A03.md` (`cb2a64ca…`).

**Status of all results:** descriptive. Group T is **EXPLORATORY** (each baseline is a single training run, n = 1). Under protocol §7 there is:
- no superiority, ranking or state-of-the-art claim;
- no p-values or significance tests;
- no bold-best formatting;
- no cross-group contrast;
- no adjustment for multiplicity.

**References** are computed restorations, not ground truth. That applies to every table below.

---

## 1. What was run

| Step | Output folder (`paper1_uieb_runs/`) | Wall-clock | Script seconds | Exit |
|---|---|---|---|---|
| Group R 256 × 256 (6 methods, plus U-Shape min–max) | `sota_hicrd_released_square256`, `…_ushape_minmax` | 10:18:38 PM → 10:44:09 PM (Nov 2) | 1,418.6 | 0 |
| Group R 482 × 256 (CLAHE, UDCP-GF) | `sota_hicrd_released_G482` | 10:44:36 PM → 11:03:09 PM | 1,005.7 | 0 |
| Group T 256 × 256 (batch 16) | `sota_hicrd_retrained_square256` | 11:03:23 PM → 11:17:00 PM | 698.5 | 0 |
| Batch-4 preflight (WaterNet-LSUI, 482) | `sota_hicrd_t_g482_preflight_b4` | 11:17:17 PM → 11:19:13 PM | 9.2 | 0 |
| Amendment 03 implementation and tests | `sota_prep_checks_A03` | 11:28:58 PM → 11:34:11 PM | — | see §6 |
| Post-execution check, before T482 | `sota_hicrd_post_exec_check_pre_T482` | 11:49:16 PM → 11:49:43 PM | — | 0 |
| Group T WaterNet-LSUI 482 × 256 (batch 4) | `sota_hicrd_retrained_G482` | 11:50:29 PM → 12:04:23 AM (Nov 3) | 814.8 | 0 |
| Post-execution check, after T482 | `sota_hicrd_post_exec_check_post_T482` | 12:04:35 AM → 12:06:27 AM | — | 0 |
| Frozen analysis | `sota_hicrd_analysis` | 12:06:32 AM → 12:07:45 AM | — | 0 |
| Protocol visual panels (frozen script) | `sota_hicrd_visual` | 12:07:54 AM → 12:08:24 AM | — | 0 |
| Publication figures v1 (**superseded**, see §6) | `sota_hicrd_pub_figures` | 12:08:24 AM → 12:09:21 AM | — | 0 |
| Publication figures v2 (**use these**) | `sota_hicrd_pub_figures_v2` | 12:09:58 AM → 12:10:55 AM | — | 0 |

**GPU jobs** ran one after another; none in parallel.

**Run logs** are outside the repository, in `D:\MARINE_NET_RESEARCH_sota_assets\check\`:
- `run_R_square256.log`, `run_R_G482.log`, `run_T_square256.log`;
- `run_T_G482.log`, `run_analysis.log`, `run_visual.log`;
- `run_pub_figures.log`, `run_pub_figures_v2.log`.

## 2. GPU memory (no-spill rule: peak allocated × 1.10 ≤ free, and peak reserved ≤ free; TF32 off before the probe)

| Run | Batch | Free (MiB) | Peak allocated (MiB) | Peak reserved (MiB) | Pass |
|---|---|---|---|---|---|
| U-Shape, 256 | 1 | — | — | 276 | yes |
| WWE-UIE, 256 | 1 | — | — | 128 | yes |
| NU²Net, 256 | 1 | — | — | 126 | yes |
| FUnIE-GAN, CLAHE, UDCP-GF | CPU | — | — | — | n/a |
| WaterNet-LSUI, 256 | 16 | 7,090 | 3,382 | 5,672 | yes |
| WWE-UIE-LSUI, 256 | 16 | 6,570 | 1,275 | 1,412 | yes |
| WaterNet-LSUI, 482 (preflight) | 4 | 7,090 | 753 | 794 | yes |
| **WaterNet-LSUI, 482 (full run)** | **4** | **7,090** | **753** | **794** | **yes** |
| WaterNet-LSUI, 482, batch 16 (finding B-1; not used) | 16 | 7,062 | 6,381 | 10,646 | **no** |

**Byte values for the full run** (`sota_hicrd_retrained_G482/summary.json`):

| Quantity | Bytes |
|---|---|
| Free | 7,434,403,840 |
| Peak allocated | 789,326,848 |
| Peak reserved | 832,569,344 |

These are identical to the preflight.

**Batch 16 at 482** was re-measured in a process that had run only a 6-image test, and gave the same violation. So the "non-fresh process" caveat in Amendment 03 C2 did not matter.

## 3. Integrity evidence

**Post-execution check, after T482** (`post_exec_check.json`, `035990f8…`; script `sota_hicrd_post_exec_check.py`, `9fc3713d…`): **PASS.**

**Hashes:**
- protocol v2, Amendment 02, Amendment 03, the E1r script and the metric file;
- Group R checkpoints and repo pins;
- Group T checkpoints (`af168d31…` WaterNet-LSUI s0, `fc9ebbea…` WWE-UIE-LSUI s0);
- the frozen manifest (`29402a7f…`) and all 4,000 image hashes.

The original, unlicensed WaterNet checkpoint was not loaded by any step.

**All five result folders:**
- each per-image CSV and `summary.json` hash equals the value recorded at completion;
- 2,000 rows, in manifest order;
- every value finite, and `nonfinite_raw` = 0;
- exact E1r decomposition (largest relative error 1.33e-15);
- largest |PSNR + 10 log10 MSE| = 2.23e-6 dB, against a 1e-4 dB limit;
- each mean PSNR equals its summary value;
- visual arrays 12 × 3 × H × W, finite, in [0, 1];
- one `started` and one `completed` ledger event per run, and no `failed` event.

**Controls** (each fault STOPs; the clean file passes): swapped rows, a dropped row, a NaN, MSE off by 0.1 %, PSNR off by 1e-3 dB, a non-finite flag, a wrong recorded hash.

**Visual annotations.** Every PSNR / SSIM printed on a figure was recomputed from the saved arrays and matched the saved per-image values. The largest differences were 7.1e-6 dB and 9.2e-7; the script STOPs above 1e-4 dB or 1e-5.

This includes our four arms re-run at seed 0 on the 12 registered images, which reproduce the frozen Stage 4 per-image values.

**Output hashes (SHA-256):**

| File | Hash |
|---|---|
| `sota_hicrd_retrained_G482/per_image_waternet_lsui_s0.csv` | `bd8090a593868165fa3ad7c6e6e88913f0a8f8ca5cfae553365ac52e0cd275e6` |
| `sota_hicrd_retrained_G482/summary.json` | `e51e2d9a11e051ca669fd705bfe3019cc76245614ec79b8c676d24ce21da8884` |
| `sota_hicrd_retrained_G482/visual_waternet_lsui_s0.npy` | `4aec2a4552a565d979a790ba393688634387697601f41bac53207e269ba9b913` |
| `sota_hicrd_analysis/analysis.json` | `331a09ab764904a434a12dec3cf39d9d95cd61004b2614ea557a2878920ff19e` |
| `sota_hicrd_analysis/contrasts.csv` | `81c4c7e1337a0115460595d9422fff36d89851d10d705d48514cf389eda4b4b7` |
| `sota_hicrd_analysis/comparator_means.csv` | `33f282c13a3c8fa3fbc675bbd9693280dc750fd8bd235389826bda74d91e0513` |
| `sota_hicrd_visual/panels_group_R.png` / `panels_group_T.png` / `visual_meta.json` | `29440a24…` / `2af0fcee…` / `44c02c82…` |

The earlier folders keep their recorded hashes; the check above verified them. The publication-figure hashes are in `sota_hicrd_pub_figures_v2/figure_manifest.json`.

## 4. Results

**Conventions for this section:**
- **Means and CIs:** mean over 2,000 images. The 95 % CI is a percentile bootstrap that reuses the registered Stage 4 draws: ALL is P1, seed 20261103, B = 10,000, resampling T30 scene clusters.
- **Ours:** per-seed values (s0, s1, s2) and the seed mean.
- **Group R and Group T:** n = 1 each.
- **†** marks Group T: exploratory, n = 1.

### 4.1 Method-level results, 256 × 256, ALL 2,000 (Table 1)

| Group | Method | PSNR s0 / s1 / s2 | PSNR mean [95 % CI] | SSIM s0 / s1 / s2 | SSIM mean [95 % CI] |
|---|---|---|---|---|---|
| Ours | MARINE-Net BN | 23.888 / 23.377 / 21.316 | 22.860 [22.362, 23.342] | 0.9352 / 0.9122 / 0.8905 | 0.9126 [0.9075, 0.9174] |
| Ours | MARINE-Net IN | 20.462 / 19.466 / 19.919 | 19.949 [19.514, 20.398] | 0.8770 / 0.8712 / 0.8751 | 0.8744 [0.8671, 0.8811] |
| Ours | AT-Trunk BN | 21.535 / 22.366 / 21.805 | 21.902 [21.059, 22.732] | 0.9144 / 0.9269 / 0.9091 | 0.9168 [0.9085, 0.9245] |
| Ours | AT-Trunk IN | 19.842 / 19.172 / 20.368 | 19.794 [19.254, 20.433] | 0.8811 / 0.8761 / 0.8799 | 0.8790 [0.8729, 0.8849] |
| R | U-Shape (clamp, primary) | — | 21.720 [21.238, 22.174] | — | 0.7637 [0.7550, 0.7729] |
| R | WWE-UIE | — | 18.852 [18.279, 19.369] | — | 0.8689 [0.8555, 0.8805] |
| R | FUnIE-GAN | — | 23.243 [22.270, 24.133] | — | 0.8867 [0.8678, 0.9028] |
| R | NU²Net | — | 18.210 [17.830, 18.610] | — | 0.8371 [0.8274, 0.8461] |
| R | CLAHE | — | 18.066 [17.545, 18.542] | — | 0.7962 [0.7798, 0.8118] |
| R | UDCP-GF | — | 15.709 [15.115, 16.298] | — | 0.6814 [0.6530, 0.7055] |
| R (sensitivity) | U-Shape (min–max) | — | 20.873 [20.373, 21.381] | — | 0.7666 [0.7579, 0.7749] |
| T† | WaterNet-LSUI s0 | — | 22.347 [21.807, 22.848] | — | 0.9102 [0.9044, 0.9155] |
| T† | WWE-UIE-LSUI s0 | — | 20.934 [20.300, 21.610] | — | 0.9013 [0.8949, 0.9072] |

**Sources:**
- Comparator means and CIs are from `sota_hicrd_analysis/comparator_means.csv`.
- Per-seed values of our arms are from the frozen Stage 4 `per_image.csv`.
- The CI of our seed-mean metric is computed in `sota_hicrd_pub_figures.py` with the same P1 draws (`table_method_level.csv`); the frozen analysis registers contrasts, not our absolute means.

### 4.2 Registered contrasts, 256 × 256, ALL (32 per metric; Table 2)

**Definitions:**
- Δ = ours − comparator, the seed mean of the three per-seed mean differences.
- The 95 % CI is the bootstrap CI of the seed-mean difference (P1 draws).
- Per-seed values are in `contrasts.csv`, together with the t95 interval (df 2) and the share of images with Δ > 0 per seed.

| Comparator | Δ PSNR, dB — MARINE-Net BN | MARINE-Net IN | AT-Trunk BN | AT-Trunk IN |
|---|---|---|---|---|
| U-Shape | +1.140 [0.459, 1.792] | −1.771 [−2.278, −1.269] | +0.182 [−0.730, 1.011] | −1.926 [−2.583, −1.259] |
| WWE-UIE | +4.009 [3.368, 4.708] | +1.097 [0.629, 1.577] | +3.050 [2.085, 4.103] | +0.942 [0.288, 1.600] |
| FUnIE-GAN | −0.382 [−1.680, 0.900] | −3.294 [−4.537, −1.911] | −1.341 [−2.953, 0.248] | −3.449 [−4.814, −1.870] |
| NU²Net | +4.651 [4.137, 5.193] | +1.739 [1.545, 1.914] | +3.692 [2.951, 4.503] | +1.584 [1.225, 1.938] |
| CLAHE | +4.794 [4.472, 5.158] | +1.883 [1.378, 2.393] | +3.836 [3.394, 4.311] | +1.728 [1.311, 2.152] |
| UDCP-GF | +7.152 [6.175, 8.062] | +4.241 [3.282, 5.247] | +6.193 [4.927, 7.395] | +4.086 [3.036, 5.266] |
| WaterNet-LSUI† | +0.513 [−0.277, 1.379] | −2.398 [−2.906, −1.947] | −0.446 [−1.468, 0.600] | −2.553 [−3.217, −1.933] |
| WWE-UIE-LSUI† | +1.927 [1.294, 2.556] | −0.985 [−1.274, −0.677] | +0.968 [0.359, 1.612] | −1.140 [−1.353, −0.916] |

| Comparator | Δ SSIM — MARINE-Net BN | MARINE-Net IN | AT-Trunk BN | AT-Trunk IN |
|---|---|---|---|---|
| U-Shape | +0.149 [0.137, 0.160] | +0.111 [0.099, 0.122] | +0.153 [0.139, 0.167] | +0.115 [0.103, 0.127] |
| WWE-UIE | +0.044 [0.033, 0.056] | +0.006 [−0.004, 0.016] | +0.048 [0.036, 0.062] | +0.010 [−0.000, 0.022] |
| FUnIE-GAN | +0.026 [0.006, 0.048] | −0.012 [−0.034, 0.012] | +0.030 [0.007, 0.055] | −0.008 [−0.029, 0.017] |
| NU²Net | +0.076 [0.066, 0.085] | +0.037 [0.032, 0.043] | +0.080 [0.068, 0.091] | +0.042 [0.035, 0.049] |
| CLAHE | +0.116 [0.104, 0.129] | +0.078 [0.065, 0.093] | +0.121 [0.112, 0.130] | +0.083 [0.071, 0.096] |
| UDCP-GF | +0.231 [0.203, 0.262] | +0.193 [0.164, 0.226] | +0.235 [0.204, 0.270] | +0.198 [0.169, 0.231] |
| WaterNet-LSUI† | +0.003 [−0.005, 0.010] | −0.036 [−0.042, −0.030] | +0.007 [−0.004, 0.017] | −0.031 [−0.038, −0.025] |
| WWE-UIE-LSUI† | +0.011 [0.006, 0.017] | −0.027 [−0.029, −0.025] | +0.015 [0.008, 0.022] | −0.022 [−0.025, −0.020] |

**How the 64 contrasts at ALL fall** (32 per metric × 2):

| CI position | Count |
|---|---|
| Entirely above 0 | 41 |
| Entirely below 0 | 12 |
| Spans 0 | 11 |

This is descriptive, unadjusted for multiplicity, and not a test.

**Seed dependence.** Per-seed values show it clearly. For example, MARINE-Net BN against U-Shape gives +2.168 / +1.657 / −0.404 dB; seed 2 of MARINE-Net BN is lower on every comparator. The t95 intervals (df 2) are wide wherever the seeds disagree; they are in `contrasts.csv`.

**Population sensitivity.** The other registered populations are ALL_STRATIFIED (P1s), KNOWN-site (n = 1,839, P2) and official test (n = 300). Comparing each with ALL (64 contrasts each):

| Population | CI classifications that change |
|---|---|
| ALL_STRATIFIED | 1 |
| KNOWN | 2 |
| TEST300 | 8 |

- **ALL_STRATIFIED and KNOWN:** every change is WWE-UIE SSIM for an IN arm, moving from spanning 0 to above 0.
- **TEST300:** every change is from a sign-definite CI to one spanning 0 (FUnIE-GAN SSIM for both BN arms, and several PSNR contrasts). That is consistent with the smaller, more clustered population.

### 4.3 Error decomposition (secondary; 256 × 256, ALL)

**Decomposition:** MSE = A + C + G + R exactly, where:
- **A** = achromatic (brightness) offset;
- **C** = chromatic (colour-cast) offset;
- **G** = per-channel gain / contrast error;
- **R** = spatial residual.

**Mean composition** (Fig. 3, `table_method_level.csv`):

| Method | Mean MSE | Main component(s) |
|---|---|---|
| UDCP-GF | 0.0329 | A (≈ 0.029) |
| CLAHE | 0.0179 | C, then R |
| WWE-UIE, NU²Net | 0.0164, 0.0163 | A |
| U-Shape | 0.0085 | a larger R share than any of our arms |
| Our arms | 0.0066–0.0120 | A and G, with a small R |

**Differences in our MSE relative to comparators.** These are registered secondary contrasts (`analysis.json` → `e1r`), seed mean [95 % CI]:
- **Against CLAHE:** dominated by C, about −0.008 for every arm.
- **Against UDCP-GF:** dominated by A, from −0.022 to −0.026.
- **Against U-Shape:** R is lower for every arm (about −0.0021). G is higher for every arm (+0.0014 to +0.0034).
- **IN arms against FUnIE-GAN and WaterNet-LSUI†:** dMSE is positive, mainly through A and G. For example, MARINE-Net IN against WaterNet-LSUI† has dA +0.0028 and dG +0.0018.

### 4.4 Geometry sensitivity, 482 × 256 against 256 × 256 (Fig. 4)

**Applicable at 482:** CLAHE, UDCP-GF, WaterNet-LSUI† and our arms. The other Group R models need fixed or divisible input sizes (recorded in `summary.json` → `not_applicable`).

| Method | Mean PSNR 256 → 482 (Δ) | Mean SSIM 256 → 482 (Δ) |
|---|---|---|
| MARINE-Net BN (seed mean) | 22.860 → 23.001 (+0.141) | 0.9126 → 0.9109 (−0.002) |
| MARINE-Net IN | 19.949 → 19.813 (−0.136) | 0.8744 → 0.8723 (−0.002) |
| AT-Trunk BN | 21.902 → 23.050 (+1.148) | 0.9168 → 0.9268 (+0.010) |
| AT-Trunk IN | 19.794 → 21.076 (+1.282) | 0.8790 → 0.9029 (+0.024) |
| CLAHE | 18.066 → 18.272 (+0.206) | 0.7962 → 0.8080 (+0.012) |
| UDCP-GF | 15.709 → 15.362 (−0.347) | 0.6814 → 0.6633 (−0.018) |
| WaterNet-LSUI† | 22.347 → 22.611 (+0.263) | 0.9102 → 0.9158 (+0.006) |

**Registered contrasts at 482** (ALL; 3 comparators × 4 arms × 2 metrics = 24): only one CI classification differs from 256 × 256. AT-Trunk BN against WaterNet-LSUI† in SSIM moves from spanning 0 at 256 (+0.007 [−0.004, 0.017]) to above 0 at 482 (+0.011 [0.003, 0.019]).

**AT-Trunk gains about 1.1–1.3 dB at 482**, while MARINE-Net changes by under 0.15 dB. So for AT-Trunk, contrasts involving it depend on the evaluation geometry.

### 4.5 U-Shape output-normalisation sensitivity

U-Shape with its official per-image min–max mapping, compared with the clamp primary:

| | Clamp (primary) | Min–max |
|---|---|---|
| Mean PSNR | 21.720 | 20.873 |
| Mean SSIM | 0.7637 | 0.7666 |

**Contrasts against min–max** (seed mean [CI]):

| Arm | Δ PSNR (dB) | Δ SSIM |
|---|---|---|
| MARINE-Net BN | +1.987 [1.267, 2.724] | +0.146 [0.135, 0.157] |
| MARINE-Net IN | −0.925 [−1.305, −0.576] | +0.108 [0.099, 0.116] |
| AT-Trunk BN | +1.028 [0.086, 2.016] | +0.150 [0.137, 0.163] |
| AT-Trunk IN | −1.080 [−1.640, −0.583] | +0.112 [0.103, 0.122] |

**AT-Trunk BN vs U-Shape** changes classification: its PSNR CI spans 0 under clamp and is above 0 under min–max. All other signs are unchanged.

### 4.6 Group R and Group T compared (description only; never cross-contrasted)

- **Group R** answers: "how do our checkpoints compare with methods *as released* (different training data, objectives and input sizes)?"
- **Group T** answers: "how do they compare with two architectures *retrained by us on LSUI* under our pipeline?"

**The difference matters.** WWE-UIE as released (Group R) has mean PSNR 18.852; retrained on LSUI (Group T) it reaches 20.934, and its SSIM rises from 0.8689 to 0.9013. So the released-checkpoint numbers depend on each release's training setup. They are not evidence about the architecture.

## 5. Figures (`paper1_uieb_runs/sota_hicrd_pub_figures_v2/`, PNG at 300 dpi plus vector PDF)

**Captions:**

**Figure 1** (`fig1_overall_psnr_ssim_square256`). Mean PSNR (left) and SSIM (right) on HICRD-2000 (2,000 pairs, 256 × 256).
- Our arms: filled marker = seed mean, hollow = individual seeds (0–2). Blue = BN, orange = IN; circle = MARINE-Net, square = AT-Trunk.
- Diamonds: comparators. Filled = Group R (released checkpoints, n = 1); thin = U-Shape min–max sensitivity; hollow † = Group T (our LSUI retrains, exploratory, n = 1).
- Bars: 95 % cluster-bootstrap CIs (registered Stage 4 draws, B = 10,000).
- References are computed restorations, not ground truth.

**Figure 2** (`fig2_registered_contrasts_square256_ALL`). The 32 registered contrasts per metric: seed-mean difference (ours − comparator) with 95 % bootstrap CI.
- Group R above the dotted line; Group T (†, exploratory) below.
- Descriptive and unadjusted for multiplicity; no cross-group contrast.

**Figure 3** (`fig3_error_decomposition_square256`). Mean MSE split exactly into A (achromatic offset), C (chromatic offset), G (per-channel gain / contrast) and R (spatial residual).
- Ours: seed mean of seeds 0–2.
- Numbers: mean MSE. Letters label segments that hold more than 9 % of their bar.

**Figure 4** (`fig4_geometry_sensitivity`). Mean PSNR and SSIM at 256 × 256 (filled) and 482 × 256 (hollow), for the methods valid at both. The label is the change (482 − 256).

**Figures 5a–c** (`fig5{a,b,c}_visual_groupR_images*`). The 12 registered images (protocol §8: 6 site-median picks and 6 random picks; none swapped).
- Each image has two rows: the full image, then a 64 × 64 crop at the same place for every method.
- Crop rule: the window with the most Sobel gradient energy in the reference, which does not depend on any method.
- Columns: input, reference, our four arms (seed 0), and the six Group R methods.
- Under each output: PSNR (dB) / SSIM against the computed reference, recomputed from the saved outputs and matching the saved per-image values.

**Figures 6a–b** (`fig6{a,b}_visual_groupT_images*`). The same layout with the two Group T retrains († exploratory).

**Registered protocol figure** (`sota_hicrd_visual/panels_group_{R,T}.png`, 160 dpi, MARINE-Net BN / IN only). Produced by the frozen visual script as registered and kept unchanged. The publication figures above extend it with the AT-Trunk arms, SSIM and crops.

**Tables:** `table_method_level.csv` (per seed and per geometry), `sota_hicrd_analysis/contrasts.csv` and `comparator_means.csv`.

## 6. Failures, deviations and corrections (disclosed)

1. **The frozen execution test suite** (`test_sota_hicrd_exec.py`) passed 42 checks, none failed, then halted at D1 (exit 1).
   - D1–D4 dry-run the real configuration and require the run folders to be absent. After the approved runs they exist, so the never-overwrite guard refused.
   - The frozen test was not edited. D1–D4 were replaced by `sota_hicrd_post_exec_check.py`.
   - I stopped and reported this; the user then approved continuing.
2. **Test-writing fault:** the first version of the batch-map control did not catch the expected batch-16 no-spill STOP. It was fixed and re-run (14 / 14 pass).
3. **Checker fault:** the first run of the post-execution check crashed (TypeError) on the U-Shape min–max `summary.json`, which has a single `check` block. It was fixed before any result was recorded; nothing was written by the failed run.
4. **Publication figures v1** (`sota_hicrd_pub_figures/`) are kept but **superseded**:
   - Fig. 1 drew the CI bars wrongly (a plotting-argument error) and styled the U-Shape min–max marker as Group T;
   - Fig. 5 annotation labels overlapped the next image row.

   Both were fixed and regenerated in `_v2`. The data, checks and annotations are identical; only the drawing changed.
5. **The Amendment 03 file name** keeps "PROPOSED" because frozen files are not renamed. The approval is recorded in `…_addendum_A03.md`.

There were no hash, memory, non-finite or decomposition failures in any inference or analysis run.

## 7. Limitations

- **References:** HICRD references are computed restorations, not ground truth. PSNR / SSIM measure agreement with that restoration.
- **Effective sample size:** the 2,000 images form 188 T30 scene clusters. The largest has 163 images and the median 3; the cluster-weighted effective n is about 43. The bootstrap resamples clusters, and the CIs should be read at that effective size.
- **Seeds:** our arms have 3 seeds, and the seed spread is large (MARINE-Net BN: 21.3–23.9 dB). Group R and Group T comparators are n = 1. Group T contrasts describe these particular trained models only and cannot support training-seed-level statements.
- **Group R:** released checkpoints differ in training data, loss, resolution and output mapping. Contrasts are not architecture comparisons. Their geometry is our fixed 256 × 256 resize, not each method's native size.
- **Implementations:** UDCP-GF is our implementation (a guided filter, r = 24, in place of soft matting); exact fidelity to the paper is not claimed. NU²Net's licence is CC BY-NC 4.0.
- **Multiplicity:** contrasts are descriptive and unadjusted. TEST300 changes 8 of 64 CI classifications.
- **Geometry:** at 482 × 256 only CLAHE, UDCP-GF and WaterNet-LSUI† apply. AT-Trunk is geometry-sensitive (about +1.1 to +1.3 dB at 482).
- **Visuals:** the 12 visual images are illustrative, not representative estimates.

## 8. Manuscript-ready Results subsection (draft)

> **Comparison with released and retrained methods on HICRD-2000.** Under a pre-registered protocol (Supplementary §S-x), we evaluated six released methods (Group R: U-Shape, WWE-UIE, FUnIE-GAN, NU²Net, CLAHE and our guided-filter UDCP implementation) and two architectures retrained on LSUI with our pipeline (Group T: WaterNet and WWE-UIE; one training run each, reported as exploratory) on all 2,000 HICRD pairs at 256 × 256, against HICRD's computed reference restorations. Table 1 lists the mean PSNR and SSIM with 95 % cluster-bootstrap CIs, and per-seed values for our arms. Seed-mean PSNR of MARINE-Net BN was 22.86 dB (seeds 23.89, 23.38, 21.32) and of AT-Trunk BN 21.90 dB; the IN arms were 19.95 and 19.79 dB. Table 2 and Fig. 2 give the 32 registered contrasts per metric. Under this protocol, the seed-mean PSNR of MARINE-Net BN was 4.79 dB higher than CLAHE (95 % CI [4.47, 5.16]), 4.65 dB higher than NU²Net ([4.14, 5.19]) and 1.14 dB higher than U-Shape ([0.46, 1.79]; per seed +2.17, +1.66, −0.40), and 0.38 dB lower than FUnIE-GAN ([−1.68, 0.90]). The IN arms were lower than FUnIE-GAN and U-Shape in PSNR (e.g. MARINE-Net IN − FUnIE-GAN: −3.29 dB, [−4.54, −1.91]). Against the exploratory LSUI retrains, MARINE-Net BN differed from WaterNet-LSUI by +0.51 dB ([−0.28, 1.38]) and both IN arms were lower (−2.40 and −2.55 dB). SSIM contrasts were positive for every arm against U-Shape, NU²Net, CLAHE and UDCP-GF. The exact error decomposition (Fig. 3) attributes the differences from CLAHE mainly to the chromatic component (≈ −0.008 in MSE) and from UDCP-GF to the achromatic component; relative to U-Shape, our arms had lower spatial residual and higher gain error. At 482 × 256 (Fig. 4), AT-Trunk gained 1.15–1.28 dB while MARINE-Net changed by under 0.15 dB, so contrasts involving AT-Trunk depend on evaluation geometry. Contrast signs were stable in the site-stratified and known-site populations (1–2 of 64 CI classifications changed) and less so in the 300-image official test split (8 of 64). These contrasts are descriptive, unadjusted for multiplicity, and based on about 43 effective scene clusters; released checkpoints differ in training data and objectives, so they do not compare architectures.

**Table 1 caption.** Mean PSNR (dB) and SSIM on HICRD-2000 (2,000 pairs, 256 × 256; references are computed restorations, not ground truth). Ours: three seeds and their mean. Comparators: one model each. Group R uses released checkpoints; Group T (†) uses our LSUI retrains, exploratory. 95 % CIs from a scene-cluster bootstrap (B = 10,000) reusing the Stage 4 draws. U-Shape min–max is an output-mapping sensitivity row.

**Table 2 caption.** Registered contrasts (ours − comparator, seed mean of three per-seed mean differences) in PSNR (dB) and SSIM, with 95 % scene-cluster bootstrap CIs. Groups R and T are reported separately and never compared with each other. Group T (†) contrasts are exploratory (n = 1 baseline). Descriptive; unadjusted for multiplicity; per-seed values in the supplementary file `contrasts.csv`.
