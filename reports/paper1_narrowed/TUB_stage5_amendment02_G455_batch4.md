# TUB Stage 5: Amendment 02 (PROPOSED). G455 sensitivity at batch 4

Additive to Amendment 01: `reports/paper1_narrowed/TUB_stage5_amendment01_G455_batch8.md`, SHA-256 `58ce787d29fd12e1d002bb6d3231ddc55c912d94e23bfc1bbc2d9862af8d0fea`, committed in `04361ee`. Amendment 01 is not modified.

The protocol (`1e3382c0…`) and the implementation freeze (`5a8609e`) are unchanged.

Written 2026-11-02. **G455 only. No G455 inference has been run.**

## 1. Reason

The Amendment 01 preflight (`paper1_uieb_runs/tub_stage5_G455_preflight/`) gave two results:
- **P2 passed:** batch-1 and batch-8 outputs were equivalent for all 12 checkpoints.
- **P1 failed:** MARINE-Net peak reserved memory at batch 8 was 7,264 / 7,292 MiB, against 7,114 / 7,002 MiB free dedicated memory. This breaks the no-spill rule.

## 2. Change

**The only change is the G455 inference batch: 8 → 4.** Everything else is preserved exactly as in Amendment 01:
- fp32 precision, TF32 off;
- G455 geometry and preprocessing (PIL BILINEAR 455 × 256, input and reference alike);
- the 12 registered best_valA checkpoints;
- the frozen 16-pair sample (indices [17, 74, 338, 427, 625, 638, 686, 744, 831, 862, 912, 950, 1061, 1130, 1131, 1154], from `SeedSequence(20261121)`);
- the numerical-equivalence tolerances;
- the no-spill memory rule;
- all cohort and analysis definitions;
- the output paths.

**Proposed code change (not made until approved):** in `v2_tools/tub_stage5_infer.py`, use a per-geometry batch, `{"square256": 16, "aspect455": 4}`. A test asserts the mapping. Script and test hashes would be re-frozen before any G455 run.

## 3. Preflight (batch 4)

**P1. Memory** (extended to all 12 checkpoints).
- **Measurement:** the full `score_batch` (forward + frozen PSNR/SSIM + float64 decomposition) at batch 4, 256 × 455, on random tensors, for each of the 12 real best_valA checkpoints. Both peak allocated and peak reserved memory are recorded (`torch.cuda` allocator statistics).
- **Pass (unchanged rule):** for every checkpoint, peak allocated × 1.10 ≤ free dedicated memory **and** peak reserved ≤ free dedicated memory.

**P2. Batch-1 vs batch-4 equivalence.**
- **Procedure:** for each of the 12 checkpoints, the frozen sample runs in 16 batches of 1 and in 4 batches of 4, through the frozen `score_batch`.
- **Quantities:**
  - max and mean |Δ| of the clamped outputs;
  - max |ΔPSNR| and max |ΔSSIM|;
  - max relative |ΔMSE|;
  - non-finite count.
- Only differences are reported.

**Tolerances (unchanged from Amendment 01; required for every checkpoint).**

| Quantity | Tolerance |
|---|---|
| max \|Δ output\| | ≤ 3.9e-4 (0.1 / 255) |
| mean \|Δ output\| | ≤ 1e-5 |
| max \|ΔPSNR\| | ≤ 1e-3 dB |
| max \|ΔSSIM\| | ≤ 1e-4 |
| max relative \|ΔMSE\| | ≤ 1e-4 |
| non-finite outputs | 0 |

**Outcome.**
- **If P1 and P2 pass:** report the evidence and request approval for the code change and the G455 run. Full G455 inference is **not** started by this amendment.
- **If either fails:** STOP. Batch 2, mixed precision, CPU execution and other workarounds are not attempted without separate approval.

## 4. Records

- Script: `v2_tools/tub_stage5_g455_preflight_b4.py`. It reuses the committed Amendment 01 preflight functions and the inference functions unchanged.
- Outputs, written exclusively to `paper1_uieb_runs/tub_stage5_G455_preflight_b4/`:
  - `preflight_results.json`
  - `preflight_per_checkpoint.csv`
  - `preflight_console.log`
- The SHA-256 hashes of these are reported.
- Checkpoints are re-hashed after the preflight.
