# S5-E1r results: exact error decomposition of the IN − BN difference (EXPLORATORY), 2026-11-02

**Status.** An exploratory diagnostic, designed after Stage 4 and X1.

**What E1r is.** An **exact attribution** of the evaluation error to four additive components. It says **where** IN's extra or reduced error lies, not **why** it occurs.

**Fixed context.** G1, Stages 1–4, all protocols and earlier S5 documents are unchanged. No E2r, no training, no new datasets, no Val-B.

**Run.**
- Scripts: `v2_tools/s5_e1r_error_decomposition.py` (`24ef8a69…`) and `v2_tools/s5_e1r_analysis.py` (`432e0afa…`), frozen in `S5_E1r_implementation_freeze_addendum.md` (1:33:09 PM) before execution.
- Inference and decomposition 1:33:18 → 1:53:00 PM (rc 0, 19.7 min GPU); analysis → 1:55:34 PM (rc 0).
- Ledger: `20261102T133533_started_c669555d`, `20261102T135259_completed_d42f0f4a` (dataset `s5_e1r`).
- GPU memory: **not sampled** for this run (UNKNOWN). Same model and batch path as Stage 3/4, which peaked at about 6.1 GB.

## 1. Integrity and exactness

| Check | Result |
|---|---|
| Tests (`tests_v2/test_s5_e1r.py`) | 17/17 PASS: exactness; equality with an independent numpy implementation; single-component cases; constant and near-constant reference channels; perturbation detected; weights and checkpoint unchanged by a forward pass; cluster draws; determinism |
| Dry run | 12 checkpoint hashes; UIEB 787 and HICRD 2,000 pairs, all per-file hashes verified; Val-B removed from the manifest table |
| **Decomposition exact for every image × model** (787 × 12 + 2,000 × 12) | **yes:** max relative error **3.4e-15** (tolerance 1e-10); R_low + R_high = R |
| Constant reference channels | 0 |
| Checkpoint files after the run | re-hashed, **unchanged** |
| PSNR from E1r MSE vs frozen per-image PSNR | max \|Δ\| **3.4e-6 dB** (UIEB), **2.7e-6 dB** (HICRD): same predictions as Stages 3/4 |

**Definitions** (per image, 256 × 256, e = ŷ − y):
- **A:** global brightness offset;
- **C:** global colour offset;
- **G:** per-channel global gain or contrast error;
- **R:** spatial residual (split exactly into R_low ≤ 1/16 cycles per pixel and R_high);
- **MSE = A + C + G + R.**

**Δk = mean_IN(k) − mean_BN(k), per seed. Positive means IN has more error.** CIs are 95 % bootstrap: UIEB by image, HICRD by complete T30 cluster, draws shared across seeds, architectures and components.

## 2. HICRD (IN worse than BN): where is IN's extra error?

**MARINE-Net.** Seed-mean MSE: BN 0.00657, IN 0.01135.

| Δ (× 10⁻³) | s0 / s1 / s2 | Mean [95 % CI] | Seed SD |
|---|---|---|---|
| ΔMSE | 4.62 / 7.44 / 2.28 | 4.78 [3.77, 5.83] | 2.59 |
| **ΔA (brightness)** | 1.33 / 5.53 / 1.51 | **2.79 [1.88, 3.73]** | 2.37 |
| ΔC (colour) | 0.18 / −0.10 / −0.03 | 0.01 [−0.08, 0.11] | 0.15 |
| **ΔG (gain / contrast)** | 3.12 / 2.02 / 0.85 | **2.00 [1.80, 2.18]** | 1.14 |
| ΔR (residual) | −0.02 / −0.00 / −0.04 | −0.02 [−0.06, 0.01] | 0.02 |
| ΔR_low / ΔR_high | — | −0.07 [−0.10, −0.05] / 0.05 [0.04, 0.06] | — |

**AT-Trunk.** Seed-mean MSE: BN 0.00847, IN 0.01198.

| Δ (× 10⁻³) | s0 / s1 / s2 | Mean [95 % CI] | Seed SD |
|---|---|---|---|
| ΔMSE | 2.78 / 5.82 / 1.93 | 3.51 [2.67, 4.32] | 2.04 |
| **ΔA** | 1.14 / 2.97 / 0.55 | **1.55 [0.77, 2.30]** | 1.26 |
| ΔC | 0.03 / 0.33 / 0.28 | 0.21 [0.10, 0.30] | 0.16 |
| **ΔG** | 1.47 / 2.32 / 0.94 | **1.58 [1.37, 1.77]** | 0.69 |
| ΔR | 0.13 / 0.21 / 0.16 | 0.17 [0.15, 0.18] | 0.04 |
| ΔR_low / ΔR_high | — | 0.03 [0.02, 0.04] / 0.14 [0.13, 0.15] | — |

**Attribution (exact, HICRD).** For both architectures and **every seed**, IN's extra error lies almost entirely in the **global brightness offset (A)** and the **global gain or contrast term (G)**.
- Both are positive in all 3 seeds, with seed-mean CIs excluding 0.
- One per-seed CI includes 0: AT-Trunk ΔA seed 2.
- The component with the largest Δ is G or A in every seed (MARINE-Net G / A / A; AT-Trunk G / A / G).
- The global colour offset C and the spatial residual R change little. They are about 0 for MARINE-Net and small but non-zero for AT-Trunk (ΔC 0.21, ΔR 0.17 × 10⁻³; mostly high-frequency residual).
- Even under BN, A and G are the dominant HICRD error components (A 3.1–5.1, G 2.4–2.6 × 10⁻³); IN increases both.

## 3. UIEB (IN better than BN): where is IN's error reduction?

| Δ (× 10⁻³), mean [95 % CI]; per seed | MARINE-Net | AT-Trunk |
|---|---|---|
| ΔMSE | −1.37 [−1.87, −0.90]; −2.06 / −0.57 / −1.49 | −1.08 [−1.95, −0.33]; −0.57 / −1.63 / −1.05 |
| ΔA (brightness) | 0.18 [−0.15, 0.50]; 0.11 / 0.68 / −0.26 | 0.07 [−0.36, 0.49]; 0.42 / −0.13 / −0.07 |
| **ΔC (colour)** | **−0.66 [−0.93, −0.44]**; −0.58 / −0.72 / −0.66 | **−0.96 [−1.50, −0.56]**; −0.77 / −1.33 / −0.80 |
| **ΔG (gain)** | **−0.63 [−0.77, −0.48]**; −1.18 / −0.29 / −0.41 | **−0.21 [−0.38, −0.05]**; −0.25 / −0.22 / −0.16 |
| ΔR (residual) | −0.27 [−0.34, −0.20] | 0.02 [−0.04, 0.08] |

**Attribution (exact, UIEB).** IN's advantage lies in a **reduced global colour offset (C)**, negative in all seeds for both architectures, and a **reduced gain error (G)**, also negative in all seeds for both. MARINE-Net's residual R is reduced too. **The brightness offset (A) is not reduced**: CIs include 0 and the per-seed signs are mixed.

**Note on the "largest Δ" field in `e1r_results.json`.** It is the most *positive* Δ (the largest IN excess). On UIEB, where the Δs are mostly negative, it picks the least-negative component, so it is **not** a "largest contributor" there. This report uses the per-component table instead.

## 4. Secondary (exploratory): per-image ΔA vs g_ach and ΔC vs g_chr (from X1)

| | UIEB MARINE-Net | UIEB AT-Trunk | HICRD MARINE-Net | HICRD AT-Trunk |
|---|---|---|---|---|
| ρ(ΔA, g_ach) | 0.021 [−0.035, 0.076]; seeds mixed | **0.156 [0.098, 0.214]**; all seeds > 0 | −0.033 [−0.197, 0.132]; mixed | −0.028 [−0.231, 0.166]; mixed |
| ρ(ΔC, g_chr) | −0.050 [−0.113, 0.011]; mixed | **−0.182 [−0.237, −0.127]**; all < 0 | −0.041; mixed | 0.045; mixed |

**Consistent only for UIEB AT-Trunk.** There, images whose reference demands more brightness change show more IN brightness-offset excess, and images demanding more colour change show a larger IN colour-offset reduction. **Not established elsewhere.**

## 5. Interpretation (attribution vs explanation)

**Measured (exact attribution):**
- **The UIEB–HICRD reversal decomposes into different error components on the two datasets.**
  - On UIEB, IN mainly **reduces colour-offset and gain errors**, with no consistent change in the brightness offset.
  - On HICRD, IN mainly **adds brightness-offset and gain errors**, with little change in the colour offset.
- **Both architectures and all seeds show the same pattern.** Magnitudes vary across seeds; for example, MARINE-Net HICRD ΔA ranges 1.3–5.5 × 10⁻³.

**Consistency with X1 (exploratory).** X1 found that d rises with the reference-demanded *chromatic* correction and falls with the demanded *achromatic* correction. E1r shows the corresponding pattern **at the error level**:
- IN's gains are chromatic and gain-related (UIEB);
- its losses are brightness- and gain-related (HICRD).

**Not explained (hypotheses only):**
- **Why IN produces larger global brightness and gain errors on HICRD.** A candidate: IN removes each feature map's per-image mean and variance at all 40 sites, so absolute brightness and contrast must be re-inferred. On HICRD's darker, low-contrast inputs (median luma 0.28, contrast 0.085), and against references whose brightness differs from the input, that may fail more often.
- **The competing candidate:** BN's LSUI running statistics may encode a brightness and contrast prior that happens to suit HICRD.
- **E1r cannot separate these.** E2r-b's IR (per-image mean only) and RI (per-image variance only) conditions are designed to address the mean-vs-variance part of exactly this question. This is noted, **not proposed here**.
- **Gain error G worsens on HICRD but improves on UIEB.** Whether that reflects reference style (HICRD's computed restorations vs UIEB's curated contrast) or input contrast is not identifiable from E1r.

## 6. Limitations

- **Exploratory, post hoc;** many endpoints; no family-wise control.
- **n = 3 seeds**, the only training replication; seed spread is large for MARINE-Net.
- **Two related architectures.**
- **HICRD references are computed restorations; UIEB's are curated.** "Error" is relative to each dataset's reference style.
- **Global channel statistics:** A, C and G are global, and spatially varying brightness or colour errors fall into R.
- **G is a per-channel linear gain about the mean,** so non-linear tone errors also fall into R.
- **256 × 256 square geometry only** (aspect-preserving geometry not decomposed).
- **GPU memory for this run was not recorded.**

## 7. Answer to the question asked

**Is the HICRD IN deficit predominantly associated with brightness, colour, gain or spatial residual errors?**

**Brightness offset (A) and global gain or contrast (G).**
- Together they account for essentially all of ΔMSE: A + G is about 4.8 of 4.8 × 10⁻³ for MARINE-Net and about 3.1 of 3.5 × 10⁻³ for AT-Trunk.
- Both are positive in every seed, with CIs excluding 0.
- Colour offset and spatial residual contribute little.

This is an exact **attribution of error**, not an explanation of its cause.
