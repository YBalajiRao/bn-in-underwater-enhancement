# S5-E1r implementation freeze addendum (recorded 2026-11-02 01:33:09 PM, BEFORE execution)

**Status.** EXPLORATORY (designed after Stage 4 and X1). Implements E1r of `S5_proposal_REVISED_v1_DRAFT.md` §4, approved by the user 2026-11-02 after commit `2e2c944`.

**Not authorised:** E2r, training, new datasets, Val-B.

## Hashes

| Item | SHA-256 |
|---|---|
| `v2_tools/s5_e1r_error_decomposition.py` | `24ef8a6919f90d25dcbd7e318ad4491920b2fed16d2c9a8e4c5da8e3d61ff0c1` |
| `v2_tools/s5_e1r_analysis.py` | `432e0afa978b8ffeca55b585b9aed660d686c11ef62fa4d43bd1342e19af1aeb` |
| `tests_v2/test_s5_e1r.py` | `4fed9aec617e81874be6ecb5ed6d471f93e8054712e92a701ad4cefa86cd5e57` |
| `v2_tools/paper1_final_eval_v2.py` | `347e9533c9bf8118113721b67c052d451c6d6aefe1e9d593190164b8ad37a514` |
| `v2_tools/paper1_final_eval.py` | `0607d763dad5f1507ab3327a9af7bf36c90095fa43f8d6d623e2b7d487162c0f` |
| `v2_tools/stage4_infer.py` | `2b07877f3e31fe97d1e08afd29c4b3cfdce314b5d7418e7832145408eaa99e1b` |
| `train_v2.py` | `f0fb8b77ee42d8cd410a5f9aba3fc3266c966d401291dec7d67036c36ea51c8f` |
| `models/marine_net_in.py` | `3d09066f164175d6862501e7597a87be7a2101a0ec051f4ad253f40a821de81f` |
| `models/at_marine.py` | `cd1422a2072d6792bfcef5c4d28a9e515fdc1ba06e37d7556b84bdbb6e1f5c80` |
| `models/at_trunk_in.py` | `0f69dfea834f14dae98059a155b63e9eb9b667b327d7bc4c7fdc0ba582633600` |
| `original_repo/marine_net_model.py` | `8c3a265408c2d71dfdc61d299503d6174549ac5ef58d03231369c94f3dad0c91` |
| `paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json` | `dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08` |
| `paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json` | `d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c` |
| `paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json` | `29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d` |
| `paper1_uieb_runs/stage3_uieb800_best_valA/per_image.csv` | `a792f4e8ed12231c5e0f0aa7b28e5acd3be7681d4118864c6b1322220f6115b2` |
| `paper1_uieb_runs/stage4_hicrd2000_best_valA/per_image.csv` | `87fc8b6ee0b393002aa245c4ec08f947e504b8cf6dca0874bbd2c257f7f00c27` |
| `paper1_uieb_runs/s5_x1/x1_descriptors_per_image.csv` | `4f0aca5be5d5b2179a7c87e8d00e5e2bd8683dd0fb005204a7c0cc4fd9bc2004` |
| `paper1_uieb_runs/s5_e1r_preflight/test_s5_e1r_output.txt` | `61cd66594f2ae49bc47babcbdf3d8c8c1ee8bdfc7b28c2515b453604194b7e58` |

**Configuration.**
- Geometry: `train_v2.preload` 256 × 256 (frozen). Batch 16, clamp [0, 1], TF32 off (conv and matmul), float64 decomposition.
- `EPS_CONST` = 1e-12 (reference-channel variance at or below it: constant, gain term 0).
- Exactness tolerance: |A+C+G+R − MSE| ≤ 1e-10·MSE + 1e-18 per image and model, otherwise STOP. R_low / R_high cut-off 1/16 cycles per pixel.
- Cohorts: UIEB = Stage 3 PRIMARY 787; HICRD = Stage 4 ALL 2,000. Checkpoints: the 12 best_valA (hashes verified pre-run; files re-hashed after the run).
- Analysis: B = 10,000, 95 % percentile CI, `SeedSequence(20261111)` → [UIEB image draws, HICRD T30-cluster draws]; draws shared across seeds, architectures and components.
- No ratios or thresholds; secondary Spearman(ΔA, g_ach) and (ΔC, g_chr).
- Outputs: `paper1_uieb_runs/s5_e1r/`, `paper1_uieb_runs/s5_e1r_analysis/` (both absent before the run).

**Tests.** 17/17 PASS. Dry run OK (12 checkpoint hashes; UIEB 787; HICRD 2,000; all file hashes; nothing written).
