"""MARINE-Net with its 40 BatchNorm2d layers replaced by InstanceNorm2d (study MARINENETMS). Sibling of models/at_trunk_in.py (NOT modified; its replace_batch_norms helper is imported unchanged) and of
original_repo/marine_net_model.py (git-ignored, NOT modified).

The network is built EXACTLY as original_repo.marine_net_model.MARINENet() builds it (defaults: base_channels 80, num_blocks 4: frequency-aware decomposition, three-scale depthwise-separable encoder / decoder,
physics-guided attention, coordinate-only INR branch, scalar fusion), and then each `nn.BatchNorm2d(c)` is replaced in place by
    nn.InstanceNorm2d(c, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)       (same channel count c, same eps as BatchNorm2d's default)
  * Locations (all inside LightweightResidualBlock.conv = [depthwise 3x3, BN, ReLU, 1x1, BN], original_repo/marine_net_model.py:190-202): 2 per block x 4 blocks x 5 stages (3 encoder stages with 80 / 160 / 320
    channels, 2 decoder stages with 160 / 80 channels) = 40 layers, the SAME 40 sites (and channel order) as in AT-Trunk, which reuses this trunk. The frequency decomposition, the physics-guided attention, the INR
    branch and the reconstruction head contain no normalisation layer. The replacement asserts exactly these 40 sites and that no BatchNorm2d is left over.
  * FULL-channel instance normalisation (every channel of every site), as in models/at_trunk_in.py.
  * Trainable parameters: unchanged, 2,609,401 (BatchNorm2d(c, affine=True) and InstanceNorm2d(c, affine=True) both hold weight and bias of size c). Buffers: -12,840 (2 x 6,400 running statistics + 40
    num_batches_tracked counters); InstanceNorm2d(track_running_stats=False) has none. So the state dictionary has 2,622,241 elements for the BatchNorm model and 2,609,401 for this one.
  * Initialisation: both normalisation types initialise weight = 1 and bias = 0 without drawing a random number, and the replacement happens AFTER the official constructor ran, so for the same seed every
    convolution / linear / INR parameter is bit-identical to the one of MARINENet().
  * Train / eval: InstanceNorm2d uses per-image statistics in both modes (no running statistics), so there is no train / eval normalisation mismatch and no dependence on the other images of the batch.
  * forward() is inherited unchanged: the output dictionary {enhanced, cnn_output, inr_output, turbidity} is the original one, and the model has NO final_output_only attribute, so train_v2.build uses the full
    MARINELoss (auxiliary CNN / INR L1 terms active) exactly as for the MARINE-Net R3 AMP runs.
Importing this module has no side effects."""
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.at_trunk_in import N_NORM_LAYERS, NORM_CHANNELS, norm_layers, replace_batch_norms  # noqa: E402,F401
from original_repo.marine_net_model import MARINENet  # noqa: E402

EXPECTED_PARAMS = 2_609_401  # trainable parameters of MARINENet() (reports/stage_1_report.md); identical for the BatchNorm and the InstanceNorm model
BN_BUFFER_ELEMENTS = 12_840  # 2 x sum(NORM_CHANNELS) running statistics + 40 num_batches_tracked
EXPECTED_STATE_ELEMENTS_BN = EXPECTED_PARAMS + BN_BUFFER_ELEMENTS  # 2,622,241 (the figure quoted in the project plan section 3)
EXPECTED_STATE_ELEMENTS_IN = EXPECTED_PARAMS


class MARINENetIN(MARINENet):
    """MARINENet (default configuration only) with InstanceNorm2d in place of its 40 BatchNorm2d layers. forward() is inherited."""

    def __init__(self):
        super().__init__()  # identical construction (and random-number use) to MARINENet()
        replace_batch_norms(self)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def n_state_elements(model):
    return sum(v.numel() for v in model.state_dict().values())


def has_forbidden_attributes(model):
    """Problems (list of str): the model must keep the original MARINE-Net output interface (no final_output_only), i.e. the auxiliary CNN / INR loss terms stay active."""
    out = []
    if getattr(model, "final_output_only", False):
        out.append("final_output_only is set: the auxiliary CNN / INR loss terms would be zeroed")
    for name in ("inr", "physics_attention", "freq_decompose", "fusion_weight"):
        if not hasattr(model, name):
            out.append(f"the MARINE-Net component {name!r} is missing")
    return out


def model_problems(model, variant):
    """Problems (list of str) unless `model` is the exact MARINE-Net model of the variant: 'bn' = MARINENet with 40 BatchNorm2d / 0 InstanceNorm2d, 'in' = MARINENetIN with 40 InstanceNorm2d / 0 BatchNorm2d;
    2,609,401 trainable parameters; state-dict element count 2,622,241 (bn) / 2,609,401 (in); the MARINE-Net components present and no final_output_only. Pure."""
    out = []
    if variant not in ("bn", "in"):
        return [f"unknown variant {variant!r}"]
    want_cls = MARINENet if variant == "bn" else MARINENetIN
    if type(model) is not want_cls:
        out.append(f"model is {type(model).__name__}, expected {want_cls.__name__}")
    n = n_params(model)
    if n != EXPECTED_PARAMS:
        out.append(f"parameter count {n:,} != {EXPECTED_PARAMS:,}")
    want_norms = (0, N_NORM_LAYERS) if variant == "bn" else (N_NORM_LAYERS, 0)
    if norm_layers(model) != want_norms:
        out.append(f"normalisation layers (InstanceNorm, BatchNorm) = {norm_layers(model)}, expected {want_norms}")
    ne = n_state_elements(model)
    want_ne = EXPECTED_STATE_ELEMENTS_BN if variant == "bn" else EXPECTED_STATE_ELEMENTS_IN
    if ne != want_ne:
        out.append(f"state-dict elements {ne:,} != {want_ne:,}")
    if any(isinstance(m, nn.InstanceNorm2d) and (m.track_running_stats or not m.affine) for m in model.modules()):
        out.append("an InstanceNorm2d layer is not affine without running statistics")
    out += has_forbidden_attributes(model)
    return out
