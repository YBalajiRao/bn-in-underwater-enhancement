# TUB (Turbid Underwater Baseline): acquisition, integrity audit and suitability for Paper 1 (2026-11-02)

**Scope.** Dataset acquisition and read-only audit only.
- No inference, no training, no checkpoint changes, no Val-B, no E2r.
- G1, Stages 1–4, S5-X1 and S5-E1r untouched: E1r outputs re-verified against their registry (0 mismatches) before starting.

**Labels.**
- **VERIFIED:** checked against files.
- **UNVERIFIED:** not checkable from the available evidence.
- **RECOMMENDATION:** a proposal.

## 1. Acquisition and download integrity (VERIFIED)

| Item | Value |
|---|---|
| Source | official Aalborg University VAP repository, https://huggingface.co/datasets/vapaau/TUB (project page https://vap.aau.dk/tub/ links to it; the page says "site under construction") |
| Pinned revision | `6214f00807df5dbbfd389cddb4cecc41e9d62110` (last modified 2026-04-13) |
| Licence | **CC-BY-4.0** (README front-matter and HF API); attribution required (cite arXiv 2606.26295) |
| Files | **2,634**: 1,315 images, 1,315 label `.txt`, `av_turbidity.csv`, `labels.json`, `README.md`, `.gitattributes`; **227,693,748 bytes** |
| Verification | every file checked against the official LFS SHA-256 (or git blob SHA-1 for non-LFS): **0 mismatches**; sizes equal; **no zero-byte, truncated (all JPEGs end with an EOI marker) or undecodable files**; no temporary or partial files |
| Location | `datasets/source_copies/TUB/hf_vapaau_TUB_rev6214f00807df/` (git-ignored). No archive, so no extraction step; files kept as released |
| Download manifest | `datasets/source_copies/TUB/DOWNLOAD_MANIFEST.json` (SHA-256 `2c1f63e5…`): per-file SHA-256 and verification status |
| Originals after the audit | re-hashed: **unchanged** |

## 2. Counts and missing images (VERIFIED)

| Source | Count | Note |
|---|---|---|
| Paper (arXiv 2606.26295) | 1,320 | = 33 scenes × 4 cameras × 10 stages ("40 images per scene") |
| HF README | 1,315 | **matches the files** |
| Dataset viewer | 1,319 rows | not reproduced; the viewer configs cover only 27 scenes (see §6). UNVERIFIED how 1,319 arises |
| **Downloaded** | **1,315 images** | 33 scenes (Exp1–Exp33), cameras 1–4, stages 1–10, all 1920 × 1080 |

**The 5 missing images (all degraded, none a Stage-01 reference):**
- Exp26 Cam3 Stage10;
- Exp33 Cam1, Cam2, Cam3 and Cam4 Stage10.

**Effect:** **0 missing references.** The missing degraded images remove exactly 5 pairs, and no series loses its reference.

**Exact duplicates inside TUB (byte-identical files):**
- **`Exp11_Cam1_Stage01` = `Exp11_Cam1_Stage02`.** The "degraded" Stage-02 input is the clear reference itself.
- **`Exp21_Cam3_Stage09` = `Exp21_Cam3_Stage10`.** One of these two observations is not genuine; which one is UNKNOWN.

## 3. Turbidity metadata (VERIFIED, with defects)

**Files.** `av_turbidity.csv` gives the mean and SD of three nephelometer readings per scene × stage. `labels.json` NTU values agree with the CSV exactly (max |Δ| = 0).

**Defects found:**
- **The CSV contains an extra experiment, Exp34** (10 rows), with **no images**.
- **The NTU series of Exp15 and Exp33 are identical in all 10 stages** (mean and SD to 4 decimals). The images show **different scenes** (panel B), so this is a **metadata copy error**: at least one of the two series is not that scene's measurement.
- **Exp10, Exp28 and Exp34 share stages 1–3** (Exp10 and Exp28 also stage 4). The images are different scenes, so this is also copied metadata.

**Clarity and spacing of the stages:**
- **Stage 01 ("clear water") is not uniformly clear:** 1.0–8.7 NTU across scenes (Exp1 8.72, Exp2 4.52; most 1.0–3.6).
- **Stages are not equally spaced in NTU**, and the ranges overlap heavily across scenes: stage 2 spans 2.3–12.7 NTU; stage 10 spans 11.3–58.3.
- **NTU decreases between consecutive stages 19 times** (largest −6.2 NTU, Exp15/Exp33 stage 7→8, which is also the copied series).

**Consequence.** Stage numbers are only ordinal. NTU must be used as measured, but **NTU is unreliable for scenes 10, 15, 28, 33** (and Exp34 has no images).

## 4. Pairing (VERIFIED)

**Rule.** Every Stage 02–10 image is paired explicitly with Stage 01 of the **same scene and same camera** (`build_pairs`, tested).
- **1,183 raw pairs** = 1,320 − 132 references − 5 missing degraded.
- **132 complete scene-camera series.**
- No scene or camera mismatches; all pairs have the same 1920 × 1080 geometry.

**Alignment** (phase correlation, Stage 01 vs Stage 02/03 of the same camera, at 640-px width; 264 checks):
- 248 within 1 px; 16 above 1 px; **5 above 3 px**.
- **Exp28 Cam4:** −51 / −63 px with low correlation peaks. **The camera moved between Stage 01 and Stage 02** (visually confirmed: board in a different position).
- **Exp29 Cam4:** −4 / −6 px. **Exp21 Cam4:** 3 / 4 px.
- **Limit:** alignment can only be measured at low turbidity. Movement at stages ≥ 4 is not detectable and is UNVERIFIED.

**Candidate evaluation manifest** (`paper1_uieb_runs/tub_audit/PROPOSED_tub_eval_manifest_v1_flagged.json`, SHA-256 `b3a31b58…`). Every pair is kept and flagged; nothing is silently removed.

| Flag | Pairs |
|---|---|
| input identical to reference (Exp11 Cam1 St02) | 1 |
| byte-duplicate observation (Exp21 Cam3 St09 / St10) | 2 |
| misaligned series, shift > 3 px (Exp21 Cam4, Exp28 Cam4, Exp29 Cam4) | 27 |
| NTU metadata copied (scenes 10, 15, 28, 33) | 140 (9 also misaligned) |

| Eligibility | Pairs | Scenes | Camera series |
|---|---|---|---|
| **Primary-eligible** | **1,153** | 33 | 129 |
| **Eligible for NTU-based analyses** | **1,022** | 29 | — |

## 5. Camera metadata (VERIFIED absence)

**None of the 1,315 JPEGs carries EXIF**: exposure time, ISO, white balance, exposure mode and camera model are **UNKNOWN for every image**. They were stripped or never written. Whether the GoPros used auto exposure or auto white balance is **UNKNOWN**; it is not documented in the paper text read.

**Measured global image statistics** (median over all series):

| Stage | NTU | Luma | Imbalance |
|---|---|---|---|
| 1 | 1.8 | 0.448 | 0.150 |
| 10 | 31.7 | 0.517 | 0.147 |

- Luma **rises monotonically**; channel imbalance **does not change**.
- The reference-required correction is almost purely **achromatic**: g_ach ≈ g_tot, 0.037 at stage 2 to 0.071 at stage 10.
- **Whether part of the brightness change is camera automation rather than scattering cannot be determined** (no EXIF). Brightness is therefore a measured image property, **not** a pure function of NTU.

## 6. Splits (VERIFIED from files and metadata)

**HF config** (`tub_split_matrix_by_scene.csv`):
- train = 24 scenes, test = 9 scenes;
- **6 scenes in both: 15, 18, 20, 22, 24, 29**;
- **6 scenes in neither: 16, 21, 23, 25, 30, 33**;
- the union covers only 27 of 33 scenes.

**Paper.** A scene-level split of 8 validation / 25 training scenes; **IDs not listed**. Figures captioned as validation show scenes 10, 15 and 20.

**Conflicts:**
- **scene 10 is "validation" in the paper but HF-train only;**
- scenes 15 and 20 are in both HF splits.

**Conclusion.** **Neither split is usable as an independence guarantee.** Our models never saw TUB anyway, so the **whole verified cohort** is the external evaluation set. The HF split is not used.

## 7. Leakage and checkpoint exposure

**Screen.** Exact SHA-256 and 8 × 9 dHash ≤ 10 against LSUI train + Val-A (7,258), UIEB-890 (1,780), HICRD-2000 (4,000) and EUVP (31,515).
- **0 exact matches.**
- 3,859 perceptual flags; strongest dHash-256 = 19.
- The strongest candidates were **visually unrelated** (panel C): near-uniform turbid frames colliding with hazy images.
- The flag volume is inflated because high-turbidity TUB images are almost featureless.

**Provenance.** TUB was released in 2026, after LSUI, EUVP, UIEB and HICRD, so it cannot be contained in those datasets.

**Checkpoint exposure.**
- The 12 checkpoints were trained from scratch on the LSUI-train manifest only (`b4871a98…`) and selected on Val-A (Stage 4 exposure report).
- Their training postdates TUB's release, so **release dates do not exclude exposure**. Exposure is excluded by the verified **training manifests** together with 0 exact and no genuine perceptual matches.
- No other exposure path is known. Anything beyond the manifest evidence is formally **UNKNOWN**.

**Repeated scenes across scene IDs.** 2,112 same-camera Stage-01 pairs; closest dHash-256 = 45 (median 90). The top candidates are the same tank and baseboard with **different object arrangements** (panel D), so no merging is needed. Exp15/Exp33 and Exp10/Exp28 (shared NTU) are visually different scenes. **33 verified scenes.**

## 8. Scientific suitability for Paper 1

**What TUB adds that UIEB and HICRD cannot.** A **controlled, within-scene dose-response for an achromatic degradation**:
- the same static scene and camera, with measured turbidity increasing;
- degradation = whitening and contrast loss with **no added colour cast** (imbalance flat, g_chr ≈ 0).

E1r and X1 suggested that IN's relative performance is worse where **brightness and contrast** must be restored and better where **colour** must be corrected. TUB isolates exactly the former. The **directional prediction** to pre-register: IN − BN becomes more negative as NTU, or as the required achromatic correction, increases, within scene. That is a genuinely new test, not just another dataset.

**Limitations (severe; must be stated):**
1. **33 independent scenes**, 4 cameras and 9 stages nested within each. Effective independent units ≈ 33 scenes. The 1,153 pairs are not independent.
2. **References are clear-water photographs with their own yellowish tank cast and vignetting**, not colour-corrected enhancement targets. Our LSUI-trained models also colour-correct, and that is penalised against these references. PSNR therefore mixes "dehazing" with "unwanted colour change", so the error decomposition (E1r-style A/C/G/R) is essential here.
3. **Stage-01 references range 1.0–8.7 NTU** (not uniformly clear).
4. **At high NTU the scene is nearly invisible** (contact sheets), so "restoration" becomes prior-driven guessing.
5. **NTU metadata defects** (scenes 10, 15, 28, 33).
6. **No EXIF:** camera automation is UNKNOWN.
7. **Byte duplicates and 3 misaligned series.**
8. **Artificial tank, one lighting setup, LEGO baseboard, small objects:** very limited ecological validity.
9. **Square 256 × 256 resize distorts 16:9** (1.78×).
10. **The TUB paper's own focus is information loss**, a framing to cite.

## 9. Recommendation

**TUB is suitable for a clearly labelled EXPLORATORY, inference-only dose-response evaluation, not for a confirmatory claim.**

**RECOMMENDATION: protocol outline** (to be frozen before any inference):
- **Cohort:** 1,153 primary-eligible pairs (flags as in §4); NTU analyses on the 1,022 NTU-eligible pairs.
- **Models:** the 12 best_valA checkpoints; frozen 256 × 256 evaluation (aspect-preserving 455 × 256 as sensitivity).
- **Outcomes:**
  - d = PSNR_IN − PSNR_BN per pair;
  - E1r components ΔA / ΔC / ΔG / ΔR per pair (exact decomposition already validated).
- **Primary exploratory endpoint, per architecture and seed:**
  - the within-scene association of d with NTU, as a **scene-level slope** (per scene, Spearman or OLS of d on NTU across its stages and cameras), then summarised over 29 NTU-eligible scenes;
  - plus the same with the reference-required achromatic correction g_ach instead of NTU (available for all 33 scenes).
- **Uncertainty:** a **scene-cluster bootstrap** (resample the 33 or 29 scenes), 10,000 resamples; cameras and stages are never treated as independent. Per-seed reporting; seed SD and t95 (df = 2) shown separately.
- **Secondary:** ΔA and ΔG vs NTU (does IN's brightness and gain error grow with turbidity?).
- **Pre-specified direction:** d decreases and ΔA + ΔG increase with turbidity.
- **Interpretation limits as in §8.**
- **Compute:** 1,153 + 132 images at 1920 × 1080 → about 10–15 GPU-min per checkpoint set (similar to HICRD).

**RECOMMENDATION: the one scientific question TUB can answer.** *Within the same scene and camera, does BN's advantage over IN grow as a purely achromatic (scattering) degradation increases, and is the additional IN error located in the global brightness and gain components?*

## 10. Files

**Outputs** (`paper1_uieb_runs/tub_audit/`):
- `tub_inventory.csv` (scene, camera, stage, NTU, NTU SD, dimensions, SHA-256, EXIF fields = UNKNOWN);
- `tub_global_stats_per_image.csv`;
- `tub_alignment_stage01_vs_02_03.csv`;
- `tub_leakage_flags.csv`;
- `tub_split_matrix_by_scene.csv`;
- `PROPOSED_tub_pairs_manifest.json` (1,183 raw pairs);
- `PROPOSED_tub_eval_manifest_v1_flagged.json` (flags and eligibility);
- `tub_audit_summary.json`.

**Preflight** (`paper1_uieb_runs/tub_audit_preflight/`): test output (10/10), Phase-1 integrity log, pre-execution hashes, audit stdout.

**Visual panels** (scratchpad; not committed):
- contact sheets for scenes 1, 17, 33;
- A (duplicates and alignment), B (copied-NTU scenes), C (leakage), D (cross-scene).

**Scripts:** `v2_tools/tub_dataset_audit.py` (`d18cff15…`), `tests_v2/test_tub_audit.py` (`1b96826c…`).
