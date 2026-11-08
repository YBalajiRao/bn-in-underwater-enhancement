"""SOTA HICRD-2000 comparison: protocol constants and helpers (Amendment 01 + Amendment 02, 2026-11-02). PREPARED; no inference is run by this module.
Encodes: the two evaluation groups (R = released checkpoints as published; T = retrained under our LSUI recipe), fixed input geometry, output
mapping (primary clamp; U-Shape official min-max as a separate sensitivity), checkpoint registry and licences, the reused Stage 4 bootstrap draws,
the registered contrasts, and the input-only visual-selection rule."""
import hashlib, json, os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(os.environ.get("SOTA_ASSETS", "sota_assets"))

# ---- cohort (unchanged from Stage 4)
MANIFEST = ("paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json", "29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d")
STRATA = ("paper1_uieb_runs/stage4_prep/hicrd_site_strata_T30.json", "8bb91c6610431d41430b14260c63ba457cf0877e61a3855ec9d7fac0ca5da8fc")
DESCRIPTORS = ("paper1_uieb_runs/stage4_analysis_best_valA/descriptors_per_image.csv", "ca895bab71efb8beaab7a81cee1ca38f64dd61da562b287333878540aa14433e")
SITES6 = ("site1", "site2", "site4", "site5", "site7", "site8")

# ---- bootstrap: the Stage 4 draw lists, reused exactly (stage4_analysis.draw_unstratified / draw_stratified over complete T30 clusters)
BOOT_PRIMARY_SEED = 20261103      # Stage 4 P1 (unstratified)
BOOT_SENSITIVITY_SEED = 20261104  # Stage 4 P1s (site-stratified)
B = 10_000
CI = (2.5, 97.5)

# ---- geometry
PRIMARY_HW = (256, 256)
G482_HW = (256, 482)

# ---- Group R: released checkpoints (registry: reports/paper1_narrowed/SOTA_checkpoint_registry.md)
GROUP_R = {
    "waternet": {"file": "checkpoints/waternet_exported_state_dict-daa0ee.pt", "sha256": "daa0eeb74271c402b6772843a61dc751fcb243553e6650891285ad29ce7c7bc9",
                 "licence": "port MIT; original authors' weights unlicensed (no redistribution)", "training": "random 800 UIEB",
                 "input": "RGB float [0,1], 256x256; WB/GC/HE from the port's transforms", "output_map": "clamp", "g482": True},
    "ushape": {"file": "checkpoints/UShape_extracted/saved_models/G/generator_795.pth", "sha256": "c595c2aa49cd9623d02379b3990ad955c46d65dd3095b2b1c5612744981d8fae",
               "zip_sha256": "32df84c63043aa1522d6b55640d56ab296d78733b7700ad777a27d1a76e696e2", "licence": "MIT", "training": "LSUI",
               "input": "RGB float [0,1], fixed 256x256; output = model(x)[3]", "output_map": "clamp (primary); per-image min-max (sensitivity)", "g482": False,
               "params_counted": 31_589_775, "params_published": "65.6M (paper Table III); not reproducible from the released code"},
    "wwe_uie": {"file": "checkpoints/WWE-UIE_best_model.pth", "sha256": "36c3e59e556ed69a31e3f83ae7e15b86409eeb951de049ebc2207728c210027d",
                "licence": "Apache-2.0", "training": "UIEB", "input": "RGB float [0,1], 256x256 (H, W divisible by 4)", "output_map": "clamp", "g482": False},
    "funie_gan": {"file": "checkpoints/model_15320_.h5", "sha256": "3054abfc7b77706b99fd9a2e92ed8df86f6e041ac2af3d6f4a4d52292b974782",
                  "json": "checkpoints/model_15320_.json", "json_sha256": "cbbfe3b117a81d74b6382f01cb700453e739a319a3a6f38b77419bc9a1e6cf83",
                  "licence": "MIT", "training": "EUVP paired", "input": "RGB, fixed 256x256, x*2-1 (official preprocess x/127.5-1)",
                  "output_map": "(y+1)/2 then clamp", "g482": False, "device": "CPU (TensorFlow, no Windows GPU build)"},
    "nu2net": {"file": "checkpoints/NU2Net_ckpt.pth", "sha256": "2e1401a70e88d7dbc37bd77f709ff74ec52cc30b6d8a698b4cdf8842a15821d5",
               "licence": "CC BY-NC 4.0 (non-commercial)", "training": "URankerSet / UIEB random 800", "variant": "with URanker loss (epoch 237; 22.669 dB / 0.9246)",
               "input": "RGB float [0,1], 256x256 (multiple of 16)", "output_map": "clamp (in-model per-channel min-max is part of the network)", "g482": False},
}
CLASSICAL = {"clahe": {"impl": "v2_tools/sota_classical.py:clahe", "params": {"clipLimit": 2.0, "tileGridSize": [8, 8], "space": "Lab L channel"},
                       "input": "uint8 RGB 256x256 (from train_v2.preload)", "output_map": "uint8/255 (inherently 8-bit)", "g482": True},
             "udcp": {"impl": "v2_tools/sota_classical.py:udcp", "params": {"patch": 15, "t0": 0.1, "gf_radius": 24, "gf_eps": 1e-3},
                      "input": "float RGB [0,1] 256x256", "output_map": "float, clipped [0,1]", "g482": True,
                      "fidelity": "guided filter substitutes for the paper's spectral (soft) matting; not an exact reimplementation"}}
UDCP_GF_RADIUS = 24

# ---- Amendment 02: exclusions and status
GROUP_R_EXCLUDED = {"waternet": "original authors' weights (Li-Chongyi/Water-Net_Code) carry no licence; no documented permission for use. "
                                "Excluded from Group R. The tnwei port's MIT licence covers the port's code, not the converted original weights. "
                                "WaterNet remains represented only by our LSUI retrain in Group T."}
GROUP_T_STATUS = ("EXPLORATORY: each Group T baseline is a single training run (n = 1). Contrasts against it describe these specific trained models only "
                  "and cannot support any training-seed-level superiority claim; the baseline's seed variability is unknown.")

# ---- Group T: retrained under our LSUI recipe (never mixed with Group R)
GROUP_T = {"waternet_lsui_s0": {"model_key": "waternet_s0", "sha256": "af168d31cf1632bb2f9c550e32670eab1c39f28714c2a60bdfedbfc30c3565b3", "seeds": 1, "g482": True},
           "wwe_uie_lsui_s0": {"model_key": "wwe_uie_s0", "sha256": "fc9ebbeaffb1f56ca4297ef5d9a5b914274e5d6db32845cfd067fad490ca4bf5", "seeds": 1, "g482": False}}
OUR_ARMS = ("marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in")  # 3 seeds each; existing Stage 4 per-image outputs reused unchanged
SEEDS = (0, 1, 2)


def group_r_active():
    return [m for m in GROUP_R if m not in GROUP_R_EXCLUDED] + list(CLASSICAL)


def contrasts(version=2):
    """Registered contrasts (version 1 = Amendment 01, 36 per metric; version 2 = Amendment 02, WaterNet excluded from R, 32 per metric). Primary: for each comparator m and each of our 4 arms a, d_a(i) = PSNR_a,seed(i) - PSNR_m(i) per image;
    reported per seed, seed mean, seed-level t95 (df 2), and the cluster-bootstrap 95 % CI of the seed mean (Stage 4 draws). Same for SSIM.
    Comparators are taken within one group only."""
    rows = []
    comps_r = (list(GROUP_R) + list(CLASSICAL)) if version == 1 else group_r_active()
    for grp, comps in (("R", comps_r), ("T", list(GROUP_T))):
        for m in comps:
            for a in OUR_ARMS:
                rows.append({"group": grp, "comparator": m, "our_arm": a, "metrics": ("psnr", "ssim"), "estimand": "mean over images of (ours - comparator), per seed and seed mean",
                             "status": GROUP_T_STATUS if grp == "T" else "descriptive (released model, n = 1)"})
    return rows


def map_primary(y):
    """Primary output handling for every method: clamp to [0, 1] (float; no 8-bit quantisation)."""
    import torch
    return torch.clamp(y, 0.0, 1.0)


def map_funie(y_tanh):
    return np.clip((np.asarray(y_tanh, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)


def map_ushape_minmax(y):
    """U-Shape official sensitivity: torchvision save_image(normalize=True) on one image = global min-max over that image (all channels)."""
    import torch
    out = []
    for img in y:
        lo, hi = float(img.min()), float(img.max())
        out.append((img.clamp(lo, hi) - lo) / max(hi - lo, 1e-5))
    return torch.stack(out)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_group_r():
    res = {}
    for m, s in GROUP_R.items():
        ok = sha(ASSETS / s["file"]) == s["sha256"]
        if "json" in s:
            ok = ok and sha(ASSETS / s["json"]) == s["json_sha256"]
        res[m] = ok
    return res


def stage4_draws(stratified=False):
    """The exact Stage 4 draw list (row-index arrays into the 2,000-pair manifest order)."""
    import sys
    sys.path.insert(0, str(ROOT / "v2_tools"))
    import stage4_analysis as S4
    man = json.load(open(ROOT / MANIFEST[0], encoding="utf-8"))["pairs"]; st = json.load(open(ROOT / STRATA[0]))["stratum"]
    cl = np.array([p["scene_cluster_T30"] for p in man]); site = np.array([st[p["id"]] for p in man])
    cidx = S4.cluster_index(cl)
    if not stratified:
        return list(S4.draw_unstratified(cidx, np.random.default_rng(BOOT_PRIMARY_SEED)))
    cst = {}
    for i in range(len(cl)):
        cst.setdefault(cl[i], set()).add(site[i])
    return list(S4.draw_stratified(cidx, {k: next(iter(v)) for k, v in cst.items()}, np.random.default_rng(BOOT_SENSITIVITY_SEED)))


def stage4_population_draws(pop):
    """Population-specific Stage 4 draw lists, reused exactly (indices are positions within the population's rows, in manifest order):
    ALL -> P1 (unstratified, 20261103); ALL_STRATIFIED -> P1s (site-stratified, 20261104); KNOWN -> P2 (stratified by the 6 sites, 20261106);
    TEST300 -> S_official_test300 (unstratified over its T30 clusters, 20261103). Returns (rows, draws)."""
    import sys
    sys.path.insert(0, str(ROOT / "v2_tools"))
    import stage4_analysis as S4
    man = json.load(open(ROOT / MANIFEST[0], encoding="utf-8"))["pairs"]; st = json.load(open(ROOT / STRATA[0]))["stratum"]
    cl = np.array([p["scene_cluster_T30"] for p in man]); site = np.array([st[p["id"]] for p in man]); split = np.array([p["split"] for p in man])
    allr = np.arange(len(man))
    if pop in ("ALL", "ALL_STRATIFIED"):
        return allr, stage4_draws(stratified=(pop == "ALL_STRATIFIED"))
    rows = np.where(site != S4.UNASSIGNED)[0] if pop == "KNOWN" else np.where(split == "official_test")[0]
    cidx = S4.cluster_index(cl[rows])
    if pop == "KNOWN":
        cst = {k: site[rows][v[0]] for k, v in cidx.items()}
        return rows, list(S4.draw_stratified(cidx, cst, np.random.default_rng(20261106)))
    return rows, list(S4.draw_unstratified(cidx, np.random.default_rng(20261103)))


def to_metric_tensor(img_hwc_or_chw):
    """Every output is converted to a float32 torch tensor [1,3,H,W] in [0,1] before the frozen PSNR/SSIM (whose Gaussian window is float32)."""
    import torch
    a = np.asarray(img_hwc_or_chw)
    if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] != 3:
        a = np.transpose(a, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]


def assert_aligned(method_ids):
    """STOP unless a method's per-image rows are exactly the manifest ids in manifest order (required before using the shared draws)."""
    ids = [p["id"] for p in json.load(open(ROOT / MANIFEST[0], encoding="utf-8"))["pairs"]]
    if list(method_ids) != ids:
        raise SystemExit("STOP: per-image rows not aligned with the manifest order")
    return True


def visual_selection():
    """Input-only, fixed rule. 6 site picks: per known site, the image whose native input channel-mean imbalance is closest to that site's median
    (ties -> lowest manifest index). 6 random: default_rng(20261203).permutation(2000), first 6 indices not already selected. Returns ids in order."""
    import csv
    man = json.load(open(ROOT / MANIFEST[0], encoding="utf-8"))["pairs"]; st = json.load(open(ROOT / STRATA[0]))["stratum"]
    ids = [p["id"] for p in man]
    imb = {r["id"]: float(r["imbalance"]) for r in csv.DictReader(open(ROOT / DESCRIPTORS[0]))}
    picks = []
    for s in SITES6:
        idx = [i for i, k in enumerate(ids) if st[k] == s]
        med = float(np.median([imb[ids[i]] for i in idx]))
        best = min(idx, key=lambda i: (abs(imb[ids[i]] - med), i))
        picks.append(best)
    for i in np.random.default_rng(20261203).permutation(len(ids)):
        if len(picks) == 12:
            break
        if int(i) not in picks:
            picks.append(int(i))
    return [ids[i] for i in picks]
