# Erratum to `S3_results_report.md` (numerical rounding slip)

Written 2026-11-02. **The original report is not modified.**

- Its SHA-256 is still `b4902efebe102e264d2129bd8a0e61d416675cd736584a74381b0514882a77be` (working copy), committed in `d2feaff`.
- This erratum only corrects the presentation of one number.
- No result, decision, statistic or protocol changes.

## The slip

**Location:** `S3_results_report.md`, §1 primary table, row "Mean d = IN − BN (dB) per seed", MARINE-Net column (line 38).

| | As printed | Correct value | Source |
|---|---|---|---|
| MARINE-Net, seed 2 | **+1.22** | **+1.21** (1.2148475463786546) | `paper1_uieb_runs/stage3_analysis_best_valA/results.json` → `cohorts.PRIMARY_787.MARINE-Net.mean_d_per_seed[2]` |
| MARINE-Net, seeds 0 and 1 | +2.19 / +1.16 | unchanged (2.1943833697585755 / 1.1616609555273194) | same file |
| AT-Trunk, all seeds | +0.25 / +0.70 / +0.47 | unchanged (0.2478 / 0.6954 / 0.4722) | same file |

The saved value, 1.2148…, rounds to +1.21 at two decimals, not +1.22. The corrected row is:

> Mean d = IN − BN (dB) per seed | **+2.19 / +1.16 / +1.21** | +0.25 / +0.70 / +0.47

## Scope

- **Unaffected:** the H3-S3 decisions (both architectures SUPPORTED), all ρ, partial ρ and CI values, and every sensitivity result in the report. None of them use this number.
- **The two other "+1.22" values in the project are a different quantity.** They are the BN global-head gain from the post-G1 analysis (`S2_post_G1_statistical_report.md` line 89; `S2_seed_pairing_validity_audit.md` line 92), and they are not affected.
- **Manuscript drafts:** `paper1_draft_sections_v1_WORKING.md` and `paper1_draft_sections_v2_WORKING.md` already use +1.21.
- **Rule for future text:** per-seed values are quoted from `results.json`, not from the S3 report table.

## Verification

The value was read directly from the frozen `results.json` on 2026-11-02 (`mean_d_per_seed` = [2.1943833697585755, 1.1616609555273194, 1.2148475463786546]). It was confirmed again by `v2_tools/paper1_o1_mean_d_cis.py`, which asserts equality with the per-image file to 1e-12 (`paper1_uieb_runs/paper1_O1_mean_d_cis/o1_results.json`).
