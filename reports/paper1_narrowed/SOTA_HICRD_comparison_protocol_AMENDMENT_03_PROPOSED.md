# SOTA HICRD-2000 comparison protocol: Amendment 03, PROPOSED (2026-11-02)

**Amends** effective protocol v2 (`2d29442248a57ac6…`), which is not edited.

**Status: proposed, awaiting separate approval.** Nothing in it has been applied.
- The full WaterNet-LSUI 256 × 482 inference has **not** been run.
- The code change in C3 has **not** been made.

## C1. Change

**Group T, aspect482 (256 × 482) only:** the batch for WaterNet-LSUI s0 changes from **16 to 4**.

**Unchanged:**
- Group T square256 batch 16, which has already run and passed;
- Group R batch 1;
- all other Group T settings and all other protocol content.

## C2. Evidence

**Blocker** (`SOTA_HICRD_implementation_freeze_addendum.md`, B-1). At batch 16, 256 × 482, the memory guard measured:

| Quantity | Value |
|---|---|
| Free dedicated memory | 7,395 MB |
| Peak allocated | 6,699 MB |
| Peak reserved | 11,163 MB |

That violates the no-spill rule.

**Caveat.** That measurement was taken inside the test process, after other models had already run there, so allocator caching may have inflated it. It was not re-measured in a fresh process.

**Preflight.** Plan `SOTA_HICRD_T_G482_batch4_preflight_plan.md`, SHA-256 `b75bd4fc4ea9a713…`, frozen before testing. Output `paper1_uieb_runs/sota_hicrd_t_g482_preflight_b4/` (`preflight_results.json` `6db29f84…`, `preflight_per_image.csv` `8170c660…`).

- **Sample:** 16 frozen HICRD pairs (`default_rng(20261204)`), loaded through the frozen G482 path, with all manifest hashes verified.
- **Memory at batch 4, 256 × 482** (TF32 off before the probe; both flags confirmed `false`):

  | Quantity | Value |
  |---|---|
  | Free dedicated memory | 7,090 MB |
  | Peak allocated | 753 MB |
  | Peak reserved | 794 MB |

  **The no-spill rule passes.**
- **Equivalence, batch 1 vs batch 4:**
  - max and mean |Δ output| = **0.0**;
  - |ΔPSNR| = 0.0;
  - |ΔSSIM| = 0.0;
  - relative ΔMSE = 0.0;
  - non-finite outputs = 0.

  The outputs are **bit-identical**, within all frozen tolerances (3.9e-4, 1e-5, 1e-3 dB, 1e-4, 1e-4).
- **Checkpoint** `af168d31…` unchanged after the run.

## C3. Implementation change needed (not made)

- **The change:** in `v2_tools/sota_hicrd_infer_t.py`, replace the global `BATCH = 16` with a per-geometry batch, `{"square256": 16, "aspect482": 4}`, applied in the memory guard, the inference loop and the recorded meta.
- **A test** asserts the mapping.
- **Re-freezing:** the script, test and addendum hashes are re-frozen before the run.
- **Square256 results** (already produced with batch 16) are unaffected.

## C4. After approval

1. Make and test the C3 change.
2. Run `sota_hicrd_infer_t.py --geometry aspect482 --out paper1_uieb_runs/sota_hicrd_retrained_G482`.
3. Only then, and after separate approval, run the analysis and the visual step, which need all four run folders.
