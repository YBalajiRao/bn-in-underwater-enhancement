# Stage 4 protocol amendment 02 (pre-inference), 2026-11-02

**Amends.**
- `S4_preregistration_FROZEN_DRAFT.md` (v1, SHA-256 `2ea1f6ed3f8b20ff889f79c27f0dfaa795e2c2a7d9d66257b92236751b4a4934`);
- `S4_protocol_amendment_01.md` (SHA-256 `8301b73d10ddf574b25b447e686314b6f0905eeb5d7bbe80bf42e872bff8bad3`).

Both remain **unchanged, visible and in force** except where this amendment explicitly overrides a named clause.

**This amendment.** Its SHA-256 is recorded in `S4_protocol_amendment_02.sha256.txt`.

**Timing.** Written before any model prediction on HICRD exists. Only input-derived counts were used: site labels and cluster sizes.

**Reason** (user review). The 161 official-test images without an identified site were grouped as `TEST_UNASSIGNED` and used as a seventh category in the site-adjusted partial Spearman (Amendment 01, A3). A shared "unassigned" label is not an acquisition site. Adjusting for it is not equivalent to controlling for a known site.

## B1. Populations (input-only facts)

| Population | Images | T30 clusters | Cluster size min / median / max | Largest-cluster share | n_eff,clusters |
|---|---|---|---|---|---|
| **ALL**: combined external HICRD evaluation cohort | 2,000 | 188 | 1 / 3 / 163 | 0.082 | 43.2 |
| **KNOWN-SITE**: images with an identified acquisition site | **1,839** | 185 | 1 / 3 / 163 | 0.089 | 41.5 |
| **UNASSIGNED**: official-test images with no identified site | 161 | 3 | 17 / 56 / 88 | 0.547 | 2.3 |

**KNOWN-SITE composition.** site1 25, site2 16, site4 159, site5 1,084 (incl. 139 official-test images placed by cluster), site7 457, site8 98.

**UNASSIGNED** is **not** a site. It is never used as an adjustment category.

## B2. Analyses after this amendment (per architecture; d = IN − BN PSNR per image and seed)

| # | Analysis | Population | Bootstrap (complete clusters only) | Status |
|---|---|---|---|---|
| P1 | **Pooled Spearman** ρ̄ (original v1 primary) | ALL 2,000 | **v1 unstratified cluster bootstrap** (`rng 20261103`), decision CI | **Primary decision 1** (v1 rule, unchanged). The v1 confound clause (partial for luma and contrast) is computed on ALL. |
| P1s | Same, stratified CI | ALL 2,000 | resampling strata = 6 sites + one "UNASSIGNED block" (`rng 20261104`). The block is **only a resampling stratum, never a covariate**. | reported alongside P1 |
| P2 | **Site-adjusted partial Spearman** ρ̄^site: residualise rank(x) and rank(d_·s) on **6 site indicators** | KNOWN-SITE 1,839 (ranks computed within this population) | stratified by the 6 sites (`rng 20261106`) | **Primary decision 2**. **Overrides Amendment 01 A3/A5**, which used 7 categories on ALL. |
| P2c | Pooled Spearman, no site adjustment | KNOWN-SITE 1,839 | same draws as P2 | comparison: shows whether dropping UNASSIGNED alone changes the pooled estimate |
| P2b | Partial for 6 sites + rank(b) + rank(c) | KNOWN-SITE 1,839 | same draws as P2 | reported, not decisive |
| W | Within-site Spearman, sites 4, 5, 7, 8 (A01 eligibility unchanged) | each site | cluster bootstrap within the site (`rng 20261105`) | descriptive |
| U | UNASSIGNED descriptive: per-seed ρ_s, mean d per seed, imbalance range | UNASSIGNED 161 | **no CI**: 3 clusters (n_eff 2.3) cannot support a resampling interval | descriptive only; no site interpretation |
| S | v1 sensitivities (T15/T60, cluster-level, official test 300, last.pth) and A01-A8 geometry G | as before | as before | unchanged |
| A01-7cat | The Amendment 01 7-category adjustment | ALL 2,000 | as A01 | **retained for transparency only**, labelled "superseded by Amendment 02 (P2)"; not decisive |

**Resampling.** Individual images are **never** resampled independently. Every bootstrap draws complete T30 clusters. The same draw index is used across seeds and architectures within each analysis.

## B3. Decision rule and allowed statements (overrides A01 A5 where they conflict)

| Decision | Statistic and population | Rule |
|---|---|---|
| **D1 POOLED** | P1, ALL 2,000 | v1 rule unchanged: SUPPORTED / SUPPORTED, NOT CONFOUND-ROBUST / NOT SUPPORTED / INCONCLUSIVE |
| **D2 SITE-ADJUSTED** | P2, KNOWN-SITE 1,839 | WITHIN-SITE SUPPORTED: all three partial ρ_s^site > 0 **and** stratified 97.5 % CI lower bound > 0 **and** partial ρ̄^site ≥ 0.10. NOT SUPPORTED: partial ρ̄^site ≤ 0 or CI upper bound < 0.10. Otherwise INCONCLUSIVE. |

**Allowed statements.**
- **"In the known-site HICRD subset (1,839 images), the colour-cast association with the IN − BN advantage holds after adjustment for the six acquisition sites"** only if D1 = SUPPORTED **and** D2 = WITHIN-SITE SUPPORTED (intersection-union; no extra alpha). The site-adjusted conclusion **applies to the known-site subset**. It does not automatically extend to all 2,000 images or to the 161 unassigned images.
- D1 SUPPORTED with D2 not supported: "pooled association across all 2,000 images; not shown to hold within sites."
- A positive pooled correlation alone is never evidence that cast explains the advantage independently of site.
- **W** (sites 4, 5, 7, 8) and **U** are descriptive. W is summarised as "k of 4 sites with ρ̄ > 0; m with CI > 0". U is never described as a site.

**Multiplicity.** Bonferroni over 2 architectures within D1 and within D2 (97.5 % intervals). D1 and D2 are combined only by the intersection-union rule.

## B4. Separation of uncertainty sources (unchanged from A01 A9, restated)

Each table reports separately:
- (i) per-seed values s0 / s1 / s2 with their SD and t-interval (df = 2): **variation across the three trained seeds**, the unit of training replication, n = 3;
- (ii) cluster-bootstrap CIs: **image and reef-area sampling uncertainty conditional on the three trained models**;
- (iii) pooled (ALL), pooled (KNOWN-SITE), site-adjusted (KNOWN-SITE) and within-site results.

The bootstrap does not add training replicates. The BN/IN statement concerns the normalisation replacement as a whole.

## B5. Unchanged

- Hypothesis; descriptor (native channel-mean imbalance); confound definitions; threshold 0.10 (operational).
- v1 decision rule and its CI; models, seeds and checkpoints.
- Manifest `29402a7f…`, clusters `97f83af4…`, strata file `8bb91c66…` (UNASSIGNED stays a label in that file; this amendment governs its use).
- Preprocessing (A01 A8). The v1 and A01 documents themselves.
