"""MARINENETMS configuration: matched MARINE-Net baselines under the XF/XFIN fp32 recipe. Two variants, each for seeds 0, 1 and 2 (six dedicated runs, one run per driver invocation):
  bn  MARINE-Net, ORIGINAL architecture (original_repo.marine_net_model.MARINENet(), 40 BatchNorm2d), model name marine_default (the existing train_v2 registry entry), recipe name XF_R3_fp32_floor
  in  MARINE-Net with the 40 BatchNorm2d replaced by full-channel InstanceNorm2d (models/marine_net_in.py), model name marine_default_in, recipe name XFIN_R3_fp32_floor_IN
Both use the recipe values of XF/XFIN exactly (xf_config.recipe_args: R3 with ONLY amp False and eta_min 2e-8 - never retyped): AdamW lr 2e-4, weight decay 1e-5, 3-epoch warm-up, cosine to 2e-8, clip 1.0, batch 4,
no accumulation, 100 epochs, paired flips (no jitter), train 3,463 + Val-A 166 only, frozen evaluator, best-Val-A-PSNR selection. The data order and the flips are functions of the seed only, so each run sees the
same order and flips as the XF / XFIN run of the same seed.
Question (exploratory): does the BatchNorm -> InstanceNorm effect seen in AT-Trunk (XF vs XFIN) also appear in the PARENT architecture MARINE-Net? Together with XF / XFIN this gives a 2 x 2 design (architecture:
MARINE-Net vs AT-Trunk; normalisation: BatchNorm vs InstanceNorm), each cell with seeds 0, 1, 2. The design does NOT isolate single components: MARINE-Net and AT-Trunk differ in several things at once.
ARCHITECTURE-SPECIFIC LOSS DIFFERENCE (documented, not removed): MARINELoss is the same object with the same weights, but AT-Trunk is final_output_only, so its two auxiliary L1 terms (0.3 each) are fed the target and are
exactly zero, whereas MARINE-Net returns real cnn_output / inr_output tensors, so those terms are ACTIVE. This is part of the MARINE-Net design and is deliberately not altered.
REPORTING (Option A, descriptive only; fixed BEFORE any run): no class, threshold, margin, tolerance, significance or equivalence claim. See REPORT_* below and v2_tools/marinenetms_paired_summary.py.
Import-light and side-effect free (no manifest, checkpoint or image access). Val-B and UIEB are never referenced. Running any of these runs is NOT approved by this file."""
import copy
from pathlib import Path

from v2_tools import xf_config as xf
from v2_tools import xfin_config as xi

ROOT = xf.ROOT
VARIANTS = ("bn", "in")
ALLOWED_SEEDS = (0, 1, 2)
PAIRED_SEEDS = ALLOWED_SEEDS
MODEL = {"bn": "marine_default", "in": "marine_default_in"}
RECIPE_NAME = {"bn": xf.RECIPE_NAME, "in": xi.RECIPE_NAME}  # the same recipe VALUES under the XF / XFIN names, so recorded args differ from the reference's only in model / run_name
RUN_PREFIX = {"bn": "marinenet_bn_r3_fp32", "in": "marinenet_in_r3_fp32"}
EXPECTED_PARAMS = 2_609_401  # trainable parameters (reports/stage_1_report.md); the tooling ALSO compares it with the instantiated models
EXPECTED_STATE_ELEMENTS = {"bn": 2_622_241, "in": 2_609_401}  # BatchNorm adds 12,840 buffer elements
N_NORM_LAYERS = xi.N_NORM_LAYERS
EPOCHS, STEPS_PER_EPOCH, EXPECTED_TOTAL_STEPS = xf.EPOCHS, xf.STEPS_PER_EPOCH, xf.EXPECTED_TOTAL_STEPS
EXPECTED_N = dict(xf.EXPECTED_N)
ALLOWED_MANIFESTS = tuple(xf.ALLOWED_MANIFESTS)
EXPECTED_HASH_PREFIXES = dict(xf.EXPECTED_HASH_PREFIXES)
UNTRACKED_DEPENDENCY = xf.UNTRACKED_DEPENDENCY
PINNED_ORIGINAL_SHA256 = xf.PINNED_ORIGINAL_SHA256
R3_WEIGHTS = dict(xf.R3_WEIGHTS)
LOSS_CONFIG_FILE = "marinenet_loss_config.json"
CLIP_LOG_FILE = xf.CLIP_LOG_FILE  # the XF recorder writes this file name inside the run directory
GATE_LOG_FILE = "marinenet_gate.log"
GATE_JSON_FILE = "marinenet_gate.json"
sha256_file = xf.sha256_file
recipe_diff = xf.recipe_diff
verify_original_dependency = xf.verify_original_dependency
check_runtime = xf.check_runtime
check_clip_log = xf.check_clip_log
recipe_args = xf.recipe_args  # R3 with ONLY amp False and eta_min 2e-8: never retyped

# ----------------------------------------------------------------------------- references (pinned result.json files: train + val_A manifests only)
XF_RESULT_SHA256 = {0: "ecdaf4dcc1a1f6f6f906a8dc0bbe71c6e0e6e5baebdd5a37bccd384bc22cdc18",  # runs_v2/xf_attrunk_r3_fp32_s0/result.json (committed in e6310e7)
                    1: "9d576b1c82caa1e802abebe4f1ff7199843f6cc4e1d8b8e211017c5b67be3eb0",
                    2: "3d2f5cd854d37fe6d9e2344f266f24ab37566758fb8d0081cce5a8bee6ac058a"}
XFIN_RESULT_SHA256 = {0: "36d6641cae178ebcf5d91bc9607df32a16d5088a29c49b5918c76dc2a0b48ce1",  # runs_v2/xfin_attrunk_in_r3_fp32_s0/result.json
                      1: "ae6a03cb87885e578d43e71f891e4de99f472b8a82f8240c6e7e51e47f6250a5",
                      2: "5a904344e45cd901369647a4e715352e19cdb2d046ea46298a8ed6700361268b"}
EXPECTED_DIFF_VS_REFERENCE = {"model", "run_name"}  # seed and recipe name are equal by construction: bn vs XF of the same seed, in vs XFIN of the same seed

# commits that hold the reviewed tooling this tooling imports (must be ancestors of HEAD; the imported files must equal them at launch)
XF_COMMIT, XFIN_COMMIT = xi.XF_COMMIT, "5f04019654996076886ca4e8124da67117362131"
XFINMS_COMMIT = "ee59b3230707ecbdb0fd667dabff22be7d861548"  # v2_tools/xfinms_config.py + xfms_config.py (the per-seed XF / XFIN metadata checks)
XF_IMPORTED_FILES = list(xf.NEW_TOOLING_FILES)
XFIN_IMPORTED_FILES = list(xi.NEW_TOOLING_FILES)  # includes models/at_trunk_in.py (replace_batch_norms)
XFINMS_IMPORTED_FILES = ["v2_tools/xfinms_config.py", "v2_tools/xfms_config.py"]
NEW_TOOLING_FILES = ["models/marine_net_in.py", "v2_tools/marinenetms_config.py", "v2_tools/train_v2_no_valb_marinenetms.py", "v2_tools/marinenetms_gate.py", "v2_tools/run_marinenetms_single.py",
                     "v2_tools/marinenetms_paired_summary.py", "tests_v2/test_marinenetms_pipeline.py"]
PINNED_COMMITS = (("XF", XF_COMMIT, XF_IMPORTED_FILES), ("XFIN", XFIN_COMMIT, XFIN_IMPORTED_FILES), ("XFINMS", XFINMS_COMMIT, XFINMS_IMPORTED_FILES))
# tracked + unmodified at launch (wrapper and driver both check). The new files are only tracked once committed: a preflight before that commit fails by design.
PROVENANCE_FILES = list(dict.fromkeys(NEW_TOOLING_FILES + XFINMS_IMPORTED_FILES + xi.PROVENANCE_FILES))

# ----------------------------------------------------------------------------- estimates (NOT measurements)
ESTIMATED_RUN_HOURS = {"bn": 6.0, "in": 6.7}
ESTIMATE_BASIS = ("EXTRAPOLATION: MARINE-Net R3 AMP runs took about 3.6 h (earlier run records); the fp32 / cosine-floor change multiplied AT-Trunk's time by about 1.67 (3.07 h -> 5.12 h) and InstanceNorm by about "
                  "1.11 (5.12 h -> 5.67 h); no MARINE-Net fp32 run has been timed, and MARINE-Net's per-image INR loop may change the factor")

# ----------------------------------------------------------------------------- descriptive paired report (Option A): conventions only, NO rule
REPORT_METRICS = ("last10_psnr", "best_psnr", "best_ssim", "last10_ssim")  # the first is the primary metric
# (name, first run family, second run family): delta = first minus second of the SAME seed, from the exact values in each run's history.jsonl; positive = the first is higher (PSNR in dB and SSIM alike)
REPORT_PAIRS = (("norm_effect_in_marinenet", "marinenet_in", "marinenet_bn"),   # InstanceNorm vs BatchNorm inside MARINE-Net
                ("norm_effect_in_attrunk", "xfin", "xf"),                        # InstanceNorm vs BatchNorm inside AT-Trunk (the existing finding)
                ("arch_effect_under_bn", "xf", "marinenet_bn"),                  # AT-Trunk vs MARINE-Net with BatchNorm
                ("arch_effect_under_in", "xfin", "marinenet_in"))                # AT-Trunk vs MARINE-Net with InstanceNorm
REPORT_CONTRAST = ("interaction_contrast", "norm_effect_in_attrunk", "norm_effect_in_marinenet")  # per seed: (XFIN - XF) - (MARINE-IN - MARINE-BN); positive = the normalisation gain is larger in AT-Trunk
DELTA_CONVENTION = ("delta = FIRST minus SECOND run family of the SAME seed, computed from the exact values in each run's history.jsonl; for PSNR (dB) and SSIM alike a positive delta means the first family is higher. "
                    "Pairs: " + "; ".join(f"{n} = {a} - {b}" for n, a, b in REPORT_PAIRS) + ". Contrast: " + f"{REPORT_CONTRAST[0]} = {REPORT_CONTRAST[1]} - {REPORT_CONTRAST[2]} (per seed)")
REPORT_BANNER = ("DESCRIPTIVE PAIRED REPORT (Option A): per-seed exact deltas and simple summaries only; no class, no threshold, no margin; no significance, equivalence, generalisation or state-of-the-art claim; "
                 "Val-A numbers only (Val-A also selects the checkpoint); one pair per seed (n = 3); MARINE-Net and AT-Trunk differ in several components at once, so no single component effect is isolated")

REMAINING_CONFOUNDERS = [
    "MARINE-Net and AT-Trunk differ in several components at once (physics-guided attention vs none, INR branch vs none, scalar fusion vs affine colour heads + detail branch, parameter count 2.61 M vs 1.95 M): the "
    "2 x 2 design separates 'architecture family' from 'normalisation', NOT any single component",
    "the auxiliary CNN / INR L1 loss terms (0.3 each) are ACTIVE for MARINE-Net and exactly zero for AT-Trunk (final_output_only); this is an inherent property of the two designs and is not removed",
    "the initial weights differ between MARINE-Net and AT-Trunk (different constructors / random-number streams); within a variant the BatchNorm and InstanceNorm MARINE-Net models share bit-identical convolution, "
    "linear and INR initialisation for the same seed",
    "the normalisation mechanism is not separated: the swap removes batch coupling and the train / eval statistics mismatch AND adds per-image channel normalisation",
    "MARINE-Net was designed with BatchNorm; no tuning for InstanceNorm was done, and full-channel InstanceNorm2d on all 40 sites is only one possible replacement",
    "the existing MARINE-Net R3 AMP runs (cosine floor 1e-6) are NOT matched baselines and are not opened by this tooling; these runs replace them as the matched baseline",
    "n = 3 seeds per cell, no significance claim possible; Val-A selects the checkpoint (best-epoch selection inflates noisy curves); cuDNN is not bitwise deterministic; the fp32 convolutions may use TF32 "
    "(PyTorch default, recorded in marinenet_loss_config.json)",
    "MARINE-Net's seed-to-seed spread was large in the earlier AMP runs (about 1.8 dB); the fp32 spread is unknown until the runs exist",
]


# ----------------------------------------------------------------------------- names and validation
def validate_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {list(VARIANTS)}")
    return variant


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed {seed!r} is not allowed: MARINENETMS is defined for seeds {list(ALLOWED_SEEDS)} only")
    return seed


def run_name(variant, seed):
    validate_variant(variant)
    validate_seed(seed)
    return f"{RUN_PREFIX[variant]}_s{seed}"


def all_run_names():
    return [run_name(v, s) for v in VARIANTS for s in ALLOWED_SEEDS]


def parse_run_name(name):
    """-> (variant, seed) for one of the six dedicated run names; anything else (an XF / XFIN / ArmC-IN / MARINE-Net Stage 3 run name or a typo) raises ValueError."""
    for v in VARIANTS:
        for s in ALLOWED_SEEDS:
            if name == run_name(v, s):
                return v, s
    raise ValueError(f"run name {name!r} is not one of {all_run_names()}")


def variant_of_model(model):
    for v, m in MODEL.items():
        if model == m:
            return v
    raise ValueError(f"model {model!r} is not one of {sorted(MODEL.values())}")


def reference_run_name(variant, seed):
    """The AT-Trunk run of the same seed that anchors the args pairing: XF (BatchNorm) for 'bn', XFIN (InstanceNorm) for 'in'."""
    validate_variant(variant)
    validate_seed(seed)
    return f"xf_attrunk_r3_fp32_s{seed}" if variant == "bn" else f"xfin_attrunk_in_r3_fp32_s{seed}"


def reference_pin(variant, seed):
    validate_variant(variant)
    validate_seed(seed)
    return XF_RESULT_SHA256[seed] if variant == "bn" else XFIN_RESULT_SHA256[seed]


def reference_metadata_problems(variant, seed, rec):
    """Problems unless the parsed reference result.json describes the XF (bn) / XFIN (in) run of this seed. Uses the reviewed per-seed checks of xf / xfms / xfin / xfinms (pure)."""
    from v2_tools import xfinms_config as xs
    from v2_tools import xfms_config as xm
    validate_variant(variant)
    validate_seed(seed)
    if not isinstance(rec, dict):
        return ["the reference result.json is not a JSON object"]
    if variant == "bn":
        return xf.check_result_metadata(rec) if seed == 0 else xm.check_result_metadata(rec, seed)
    return xi.check_result_metadata(rec) if seed == 0 else xs.check_result_metadata(rec, seed)


def expected_argv(variant, seed):
    validate_variant(variant)
    validate_seed(seed)
    return {"--recipe": RECIPE_NAME[variant], "--model": MODEL[variant], "--seed": str(seed), "--run_name": run_name(variant, seed)}


def expected_args(variant, recipe, seed):
    """The full argparse namespace train_v2.main records in result.json['args']: the record of the AT-Trunk reference run of the same seed (XF for bn, XFIN for in) with ONLY the model and the run name replaced. Pure."""
    validate_variant(variant)
    validate_seed(seed)
    a = xf.expected_args(recipe) if variant == "bn" else xi.expected_args(recipe)
    a.update(model=MODEL[variant], run_name=run_name(variant, seed), seed=seed, recipe=RECIPE_NAME[variant])
    return a


# ----------------------------------------------------------------------------- loss record
def loss_config(variant, seed, git_commit_full, recipe, original_sha256, runtime):
    """The exact record written to runs_v2/<run>/marinenet_loss_config.json (and re-derived by the gate): XF's record (xf_config.loss_config, unchanged) with the experiment-specific fields replaced. Pure."""
    validate_variant(variant)
    validate_seed(seed)
    rec = xf.loss_config(git_commit_full, recipe, original_sha256, runtime)
    kind = "original architecture (40 BatchNorm2d)" if variant == "bn" else "its 40 BatchNorm2d layers replaced by full-channel InstanceNorm2d (affine, no running statistics)"
    rec["experiment"] = (f"MARINENETMS-{variant.upper()}: MARINE-Net {kind}, trained under the recipe of XF / XFIN seed {seed} (R3 loss, fp32, cosine floor 2e-8); compared with the AT-Trunk run "
                         f"{reference_run_name(variant, seed)} and, across variants, with the other MARINE-Net variant of the same seed")
    rec.update(run_name=run_name(variant, seed), recipe_name=RECIPE_NAME[variant], model=MODEL[variant], params=EXPECTED_PARAMS, seed=seed)
    rec["parameter_count"] = {"trainable": EXPECTED_PARAMS, "state_dict_elements": EXPECTED_STATE_ELEMENTS[variant], "bn_buffer_elements": 12_840 if variant == "bn" else 0,
                              "source": "reports/stage_1_report.md (2,609,401 trainable + 12,840 BatchNorm buffers = 2,622,241 state elements); the tooling also compares them with the instantiated models"}
    rec["baseline_reference"] = {"primary": reference_run_name(variant, seed), "primary_recipe": RECIPE_NAME[variant], "result_json_read_by_tooling": True,
                                 "note": "the result.json of the AT-Trunk run of this seed (train + val_A manifests only) is read by the preflight and the gate for the args pairing and must have its pinned SHA-256; no other "
                                         "run's result.json is opened (the MARINE-Net R3 AMP runs are deliberately not opened)",
                                 "result_json_sha256": reference_pin(variant, seed),
                                 "comparison_rule": "descriptive paired report only (Option A): no class, threshold or margin; all Val-A numbers are exploratory"}
    rec.pop("decision_rule", None)
    rec["decision_rule"] = {"status": "NONE (Option A: descriptive paired reporting only)",
                            "reporting": "v2_tools/marinenetms_paired_summary.py: exact per-seed deltas, mean, range, sign pattern, largest absolute delta; no class, threshold, margin, significance or equivalence claim",
                            "delta_convention": DELTA_CONVENTION, "pairs": [list(p) for p in REPORT_PAIRS], "contrast": list(REPORT_CONTRAST)}
    loss = rec["loss"]
    loss["implementation"] = ("original_repo/marine_net_model.py :: MARINELoss (unchanged, SHA-256 pinned) via train_v2.build exactly as for the earlier MARINE-Net R3 runs: MARINE-Net has NO final_output_only attribute, so "
                              "the loss is MARINELoss itself (NOT FinalOutputLoss) and train_v2.ZeroLoss replaces the zero-weight perceptual term")
    loss["equation"] = ("1.0*L1(clamp(pred),clamp(y)) + 0.5*(1-SSIM11) + 0.4*GradientL1 + 0.2*ColourStats(mean MSE + std MSE) + 0.3*(L1(cnn_output,y) + L1(inr_output,y)) [ACTIVE: real branch outputs] [+ 0.0*perceptual]")
    loss["clamp"] = "pred and target are clamped to [0,1] inside the loss; MARINE-Net clamps enhanced, cnn_output and inr_output inside the model as well"
    loss["auxiliary_terms"] = ("ACTIVE for MARINE-Net (cnn_output and inr_output are real tensors); exactly zero for AT-Trunk (final_output_only). This difference is inherent to the two designs and is not removed")
    if variant == "in":
        rec["normalisation"] = {"module": "models/marine_net_in.py :: MARINENetIN (models/at_trunk_in.py::replace_batch_norms, imported unchanged)", "sites": N_NORM_LAYERS,
                                "channels_in_module_order": "(80,)*8 + (160,)*8 + (320,)*8 + (160,)*8 + (80,)*8",
                                "initialisation": "construction and random-number use identical to MARINENet(); BatchNorm2d and InstanceNorm2d both start at weight 1 / bias 0 without drawing random numbers, so all "
                                                  "convolution / linear / INR parameters are bit-identical to MARINENet()'s for the same seed"}
    rec.pop("differs_from_xf", None)
    rec["differs_from_reference"] = {"reference": reference_run_name(variant, seed), "recorded_args_differ_in": sorted(EXPECTED_DIFF_VS_REFERENCE),
                                     "architecture": "MARINE-Net (physics-guided attention, INR branch, scalar fusion) instead of AT-Trunk (colour heads + detail branch, no PGA, no INR): several components differ at once",
                                     "loss": "auxiliary CNN / INR L1 terms active (MARINE-Net) vs zero (AT-Trunk)", "parameters": f"{EXPECTED_PARAMS:,} trainable vs 1,952,768"}
    rec["unchanged"] = [
        "recipe values: AdamW lr 2e-4, weight decay 1e-5, 3-epoch warm-up, cosine floor 2e-8, grad-clip 1.0, batch 4, no accumulation, 100 epochs, fp32 (amp False, no GradScaler): exactly those of XF / XFIN",
        "MARINELoss (R3 weights, SHA-256 pinned) with train_v2.ZeroLoss as perceptual term; NOT wrapped in FinalOutputLoss",
        "data: the LSUI train (3,463) and Val-A (166) manifests; paired horizontal flips only (no jitter); the data order (sampler seeded with seed*10007+epoch) and the per-sample flip decisions are derived from the "
        f"seed only, so they equal those of the XF / XFIN run of seed {seed}",
        "frozen evaluator train_v2.evaluate (256x256, fp32, TF32 off, clamp [0,1]) and Val-A best-PSNR checkpoint selection",
        "the MARINE-Net architecture and its forward(); " + ("the 40 BatchNorm2d layers" if variant == "bn" else "all components except that the 40 BatchNorm2d layers are replaced"),
    ]
    rec["remaining_confounders"] = list(REMAINING_CONFOUNDERS)
    rec["code"] = dict(rec["code"], xfin_commit=XFIN_COMMIT, xfinms_commit=XFINMS_COMMIT)
    rec["code"].pop("xr_commit", None)
    return copy.deepcopy(rec)


def check_result_metadata(r, variant, seed):
    """Problems if result.json does not describe the MARINE-Net run of this variant and seed (seed, run name, recipe, model, params, fp32, eta_min, no resume, manifest scope); [] if it does."""
    validate_variant(variant)
    validate_seed(seed)
    a = r.get("args") or {}
    out = []
    if r.get("seed") != seed or a.get("seed") != seed:
        out.append(f"seed {r.get('seed')!r}/{a.get('seed')!r} != {seed}")
    if a.get("run_name") != run_name(variant, seed):
        out.append(f"run_name {a.get('run_name')!r} != {run_name(variant, seed)!r}")
    if a.get("recipe") != RECIPE_NAME[variant]:
        out.append(f"recipe {a.get('recipe')!r} != {RECIPE_NAME[variant]!r}")
    if r.get("model") != MODEL[variant] or a.get("model") != MODEL[variant]:
        out.append(f"model {r.get('model')!r}/{a.get('model')!r} != {MODEL[variant]!r}")
    if r.get("params") != EXPECTED_PARAMS:
        out.append(f"params {r.get('params')!r} != {EXPECTED_PARAMS:,}")
    if a.get("amp") is not False:
        out.append(f"amp is {a.get('amp')!r}, expected False (fp32)")
    if not isinstance(a.get("eta_min"), (int, float)) or isinstance(a.get("eta_min"), bool) or abs(a["eta_min"] - 2e-8) > 1e-15:
        out.append(f"eta_min {a.get('eta_min')!r} != 2e-08")
    if a.get("resume") is not False:
        out.append(f"resume flag is {a.get('resume')!r}, expected False")
    m = r.get("manifest_sha256")
    if not isinstance(m, dict) or set(m) != set(ALLOWED_MANIFESTS):
        out.append(f"manifest_sha256 keys {sorted(m) if isinstance(m, dict) else m!r} != {sorted(ALLOWED_MANIFESTS)} (only train and val_A may be hashed)")
    else:
        for k, prefix in EXPECTED_HASH_PREFIXES.items():
            if not str(m.get(k, "")).startswith(prefix):
                out.append(f"manifest {k} hash does not start with {prefix}")
    return out


def check_reference_hash(path, expected):
    """Problems (list of str) unless the reference result.json at `path` is readable and has the pinned SHA-256 (fail closed). Reads only that one file."""
    p = Path(path)
    try:
        h = sha256_file(p)
    except OSError as e:
        return [f"{p.parent.name}/result.json cannot be read ({type(e).__name__}): the reference cannot be verified"]
    return [] if h == expected else [f"{p.parent.name}/result.json SHA-256 {h[:12]}... != the pinned {expected[:12]}...: the reference changed"]
