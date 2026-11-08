# Paper 1 open items O1, O2, O4, O5: results (2026-11-02)

**Scope**
- Computed from existing frozen outputs only.
- No training, no inference, no change to frozen results or statistical protocols, nothing committed.
- O1 and O2 are **descriptive, post hoc** analyses. They do not change any registered classification.

**Where everything is**

| Item | Script (new) | Outputs (new folder) |
|---|---|---|
| O1 | `v2_tools/paper1_o1_mean_d_cis.py` | `paper1_uieb_runs/paper1_O1_mean_d_cis/` |
| O2 | `v2_tools/paper1_o2_tub_psnr_vs_ntu.py` | `paper1_uieb_runs/paper1_O2_tub_psnr_vs_ntu/` (8 figures) |
| O4 | — | `reports/paper1_narrowed/S3_results_report_erratum.md` |
| O5 | `v2_tools/eol_hash_audit.py`, `v2_tools/eol_fix_prepare.py` | `paper1_uieb_runs/paper1_O5_eol_fix/` (with `O5_report.md`) |

---

## O1. Per-seed CIs of mean IN − BN PSNR (UIEB-787, HICRD-2000)

### Method

**Frozen resampling units and draws:**
- **UIEB-787:** image bootstrap, `default_rng(20261102)`, one index matrix (B = 10,000) over the primary cohort in manifest order, as in Stage 3.
- **HICRD:** complete T30 reef-area clusters, unstratified, `default_rng(20261103)` (the Stage 4 P1 / D1 draws). The site-stratified P1s draws (`default_rng(20261104)`) are a sensitivity.

**Reproduction check (passed).** With the re-created draws, the frozen ρ̄ CIs saved in each `results.json` are reproduced **exactly**: 10/10 checks, covering UIEB best_valA and last, and HICRD best_valA, last and G482, for both architectures. The draws are therefore the registered ones.

**Interval levels.** 97.5 % is the registered level of Stages 3–4; 95 % is also given. Both are conditional on the trained models. Seed-level t95 (df 2) is shown separately.

### Table O1 (best_valA, Square256); d in dB

| Set | Arch | Seed 0 [97.5 %] | Seed 1 [97.5 %] | Seed 2 [97.5 %] | Seed mean [97.5 %] / [95 %] | Seed t95 (df 2) | Share of images IN > BN |
|---|---|---|---|---|---|---|---|
| UIEB-787 | MARINE-Net | +2.19 [+1.87, +2.53] | +1.16 [+0.91, +1.41] | +1.21 [+0.98, +1.45] | +1.52 [+1.30, +1.76] / [+1.33, +1.73] | [+0.08, +2.97] | 0.69 / 0.66 / 0.65 |
| UIEB-787 | AT-Trunk | +0.25 **[−0.01, +0.51]** (95 %: [+0.02, +0.47]) | +0.70 [+0.40, +0.99] | +0.47 [+0.22, +0.74] | +0.47 [+0.25, +0.70] / [+0.27, +0.67] | **[−0.08, +1.03]** | 0.52 / 0.57 / 0.55 |
| HICRD-2000 | MARINE-Net | −3.43 [−4.36, −2.52] | −3.91 [−4.66, −3.14] | −1.40 [−1.98, −0.78] | −2.91 [−3.56, −2.31] / [−3.48, −2.39] | **[−6.22, +0.40]** | 0.17 / 0.08 / 0.35 |
| HICRD-2000 | AT-Trunk | −1.69 [−2.33, −1.12] | −3.19 [−3.99, −2.53] | −1.44 [−2.33, −0.59] | −2.11 [−2.85, −1.50] / [−2.75, −1.55] | **[−4.47, +0.25]** | 0.26 / 0.10 / 0.34 |

- **HICRD CIs** use T30 cluster draws (unstratified). The site-stratified draws give CIs within ±0.1 dB of these.

**Sensitivities (seed mean [97.5 %]; t95):**

| Set and run | MARINE-Net | AT-Trunk |
|---|---|---|
| UIEB `last` | +1.51 [+1.28, +1.75]; t95 [−0.25, +3.26] | +0.47 [+0.25, +0.70]; t95 [−0.21, +1.15]. Seed 0: +0.23 [−0.03, +0.49] at 97.5 %, [+0.003, +0.46] at 95 % |
| HICRD `last` | −3.57 [−4.31, −2.80]; t95 [−8.61, +1.48] | −2.08 [−2.79, −1.48]; t95 [−4.00, −0.15] |
| HICRD G482 | −3.19 [−3.81, −2.61]; t95 [−5.99, −0.39] | −1.97 [−2.63, −1.30]; t95 [−4.19, +0.24] |

### Reading

- **Image- or cluster-level uncertainty, models fixed.** Every per-seed CI excludes 0, with one exception: UIEB AT-Trunk seed 0. At the registered 97.5 % level its interval is [−0.01, +0.51] (best_valA) and [−0.03, +0.49] (`last`). It excludes 0 only at 95 %.
- **Seed-level uncertainty.** The t95 interval **includes 0** for UIEB AT-Trunk (both checkpoints), UIEB MARINE-Net `last`, and HICRD best_valA for both architectures.
- **Consequence:** the sign difference between datasets is consistent in every seed, but with 3 seeds it is not established at the level of training replication by the t-interval.

**Manuscript implications (not yet applied to v2):**
- Table 1 should carry these per-seed CIs and the t95 column.
- The statement "IN scored higher than BN on UIEB-787 in all three seeds for AT-Trunk" is true in sign. It needs the qualifier "seed 0's 97.5 % interval includes 0".

---

## O2. TUB: absolute BN and IN PSNR versus turbidity, and the gap

### Method

- **Data:** the NTU cohort (1,022 pairs, 29 scenes) from the saved per-image outputs.
- **Trends:** per outcome (BN PSNR, IN PSNR, d = IN − BN), the frozen estimators were reused unchanged:
  - (e) within-series Spearman, averaged to scene then over scenes;
  - (d) pooled within-series OLS slope (dB / NTU);
  - the frozen scene-bootstrap draw list (B = 10,000) with 95 % CIs, conditional on the models.
- **Check (passed):** the d trend reproduces the frozen E-T1 result exactly in all four runs.
- **Curves:** NTU deciles (equal-count; **binning chosen post hoc**), scene-weighted means with 95 % scene-bootstrap bands.
- **Run status:**
  - Square256 best_valA = reference;
  - `last` = exploratory;
  - G455 best_valA = sensitivity;
  - G455 `last` = supplementary (its batch-equivalence preflight was not run).

### Table O2: within-series trend vs NTU (Square256 best_valA); seed mean [95 % scene bootstrap]; per-seed values

| Outcome | MARINE-Net Spearman | MARINE-Net slope (dB / NTU) | AT-Trunk Spearman | AT-Trunk slope (dB / NTU) |
|---|---|---|---|---|
| **BN PSNR** | −0.839 [−0.910, −0.743]; −0.89 / −0.81 / −0.82 | −0.084 [−0.100, −0.070] | −0.848 [−0.900, −0.789]; −0.88 / −0.76 / −0.91 | −0.089 [−0.108, −0.075] |
| **IN PSNR** | −0.060 [−0.221, +0.108]; +0.10 / −0.14 / −0.13 | −0.013 [−0.031, +0.009] | +0.090 [−0.052, +0.236]; +0.13 / +0.34 / −0.20 | +0.007 [−0.009, +0.027] |
| d = IN − BN (= frozen E-T1) | +0.524 [+0.443, +0.603] | +0.071 [+0.051, +0.095] | +0.654 [+0.585, +0.722] | +0.097 [+0.075, +0.124] |

### Decile curves (Square256 best_valA; scene-weighted means, dB)

| Decile | MARINE-Net BN (s0 / s1 / s2) | MARINE-Net IN | AT-Trunk BN | AT-Trunk IN |
|---|---|---|---|---|
| Lowest (median 4.1 NTU) | 20.46 / 17.90 / 18.77 | 14.08 / 13.99 / 13.95 | 19.49 / 17.67 / 18.73 | 13.75 / 12.89 / 14.33 |
| Highest (median 33.9 NTU) | 17.41 / 16.50 / 16.77 | 14.67 / 13.73 / 13.76 | 16.67 / 15.90 / 16.13 | 14.07 / 14.15 / 14.44 |

**The gap d:**

| | Lowest decile (dB) | Highest decile (dB) |
|---|---|---|
| MARINE-Net | −6.39 / −3.90 / −4.82 | −2.73 / −2.77 / −3.01 |
| AT-Trunk | −5.74 / −4.78 / −4.40 | −2.60 / −1.74 / −1.68 |

- **IN curves:** in several seeds IN's curve rises to a maximum at about 15–20 NTU and then falls (see `fig_o2_abs_psnr_*`).
- **Input contrast:** the median input luma SD falls from 0.107 (lowest decile) to 0.042 (highest).

### Sensitivity

- **`last`:** essentially identical (BN ρ̄ −0.865 / −0.866; IN −0.058 / +0.091).
- **G455 best_valA:**
  - BN declines as before (ρ̄ −0.868 / −0.926; slopes −0.090 / −0.114 dB/NTU);
  - **MARINE-Net IN also declines** (ρ̄ −0.353 [−0.504, −0.186]; slope −0.051 [−0.066, −0.033]), but less steeply than BN;
  - AT-Trunk IN stays flat (−0.101 [−0.257, +0.063]);
  - this is why MARINE-Net's E-T1 is weaker at G455.

### Reading (descriptive)

**Answer to O2.**
- At Square256, the shrinking IN deficit (E-T1's direction, opposite to the registered one) occurs because **BN's PSNR falls steadily with turbidity while IN's is approximately flat**: within-series trend CIs include 0 and the per-seed signs are mixed.
- **The two arms do not converge to a common floor within the measured range.** At the highest-turbidity decile, BN still exceeds IN by about 1.7–3.0 dB, and BN is still declining.
- A floor-convergence account of the reversal is therefore **not supported by these curves**.
- BN's advantage is largest on the least turbid inputs.

**Not explained.**
- Why BN's advantage is largest at low turbidity is not identified here.
- **One hypothesis (untested):** at low turbidity the input is close to its Stage 01 reference, so a model that changes the input's global statistics less would score higher there.
- **Test:** compare with the input-vs-reference PSNR (identity baseline). This needs the images to be read (no inference) and is not done (open item O7).

**Status.** E-T1 to E-T3 remain "not established" (opposite direction). O2 adds description, not support.

---

## O4. Stage 3 erratum

`reports/paper1_narrowed/S3_results_report_erratum.md`:
- MARINE-Net seed 2 mean d is **1.2148…** (`results.json`); it rounds to **+1.21**, not the printed **+1.22**.
- The original report is not modified (SHA-256 `b4902efe…`).
- Decisions are unaffected. The project's other "+1.22" values are a different quantity (the BN global-head gain) and are correct.

---

## O5. Line-ending / hash reproducibility

Details are in `paper1_uieb_runs/paper1_O5_eol_fix/O5_report.md`.

**Defect (verified).**
- 103 registered paths: 99 tracked, 4 git-ignored by design.
- **In fresh clones under every git configuration tested, 49 of the 99 tracked registered files fail byte-exact hash verification.** That covers Windows git 2.47 and Linux (WSL Ubuntu) git 2.43, each with `core.autocrlf` true / false / input.
- 1 file with mixed line endings (`tools/eval_reproduce_baseline.py`) fails under all of them.
- The original working tree matches 103 / 103.

**Fix (verified in scratch clones; not applied).**
- A `.gitattributes` with `* -text`, plus a one-time re-add of the original bytes (840 files, line-ending-only changes, confirmed with `--ignore-cr-at-eol`).
- **Result: 99 / 99 RAW_MATCH in all six fresh-clone configurations.**
- The exact real-repository commands were rehearsed: they exclude the modified paired-results CSV and all untracked files.

**Recorded hash-normalisation strategy.**
- Canonical = exact frozen bytes, enforced by storage.
- **No normalisation** in frozen verifiers.
- The EOL-variant match is a diagnostic only, for pre-fix commits.

**Needs your approval:** applying it creates one commit (commands in `O5_report.md` §4).

---

## Unresolved issues

1. **O5 not applied.** Applying it creates one commit and needs approval.
   - Commits before the fix keep LF-normalised blobs (verifiable only with the EOL-variant diagnostic).
   - The 4 git-ignored registered files (HICRD extraction manifests, TUB download manifest, `original_repo/marine_net_model.py`) cannot be verified from any clone and need an archive (audit D2).
2. **Manuscript v2 not yet updated with O1 and O2.** Table 1 needs the per-seed CIs and t95, plus the AT-Trunk seed-0 caveat. §4.6 needs the O2 description in place of "reported without interpretation", and the floor-effect open item should be marked as addressed descriptively.
3. **O1 seed-level uncertainty.** With 3 seeds, the t95 interval of the mean IN − BN includes 0 for UIEB AT-Trunk and for HICRD (best_valA) in both architectures. The cross-dataset sign difference is consistent per seed, but not established at the training-replication level.
4. **O2 limits.**
   - Decile binning is post hoc.
   - The trend CIs are conditional on the models.
   - The identity baseline (input-vs-reference PSNR) needed to test the low-turbidity hypothesis is not computed (proposed O7; reads images, no inference; needs approval).
   - `last` and G455 `last` are exploratory and supplementary.
5. **O5 coverage.** The audit finds registered hashes by pattern (103 paths); hashes recorded in unusual formats could be missed.
6. **Carried over.**
   - O3: related work and citations.
   - O6: E2r, more seeds, a third architecture.
   - Main-paper figures F1–F7 (O2 provides the TUB part of F6).
   - Audit items A8–E not yet addressed.
7. **New files are uncommitted:** 4 scripts, 3 output folders, this report, the erratum and the O5 report.
