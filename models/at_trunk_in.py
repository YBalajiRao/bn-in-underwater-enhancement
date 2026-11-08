"""AT-Trunk with its 40 BatchNorm2d layers replaced by InstanceNorm2d (study XFIN). Sibling of models/at_marine.py, which is NOT modified.

The network is built EXACTLY as models.at_marine.ATMarineTrunk() builds it (defaults: base_channels 80, num_blocks 4, use_fad, no PGA, global + grid colour heads, detail branch), and then each
`nn.BatchNorm2d(c)` is replaced in place by
    nn.InstanceNorm2d(c, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)       (same channel count c, same eps as BatchNorm2d's default)
  * Locations (all inside LightweightResidualBlock.conv = [depthwise 3x3, BN, ReLU, 1x1, BN], original_repo/marine_net_model.py:190-202): 2 per block x 4 blocks x 5 stages
    (3 encoder stages with 80 / 160 / 320 channels, 2 decoder stages with 160 / 80 channels) = 40 layers; channels in module order (80,)*8 + (160,)*8 + (320,)*8 + (160,)*8 + (80,)*8.
    The replacement asserts exactly these 40 sites and that no BatchNorm2d is left over. The stem (initial_conv + frequency decomposition), the colour heads and the detail head contain no normalisation.
  * FULL-channel instance normalisation (every channel of every site), unlike WWE-UIE, which normalises only the first half of the channels of 10 layers (model.py:47,52-53).
  * Parameters: unchanged, 1,952,768 (BatchNorm2d(c, affine=True) and InstanceNorm2d(c, affine=True) both hold weight and bias of size c). Buffers: -120 (running_mean, running_var and num_batches_tracked of
    each BatchNorm2d disappear; InstanceNorm2d(track_running_stats=False) has none).
  * Initialisation: both normalisation types initialise weight = 1 and bias = 0 without drawing a random number, and the replacement happens AFTER the official constructor ran, so for the same seed every
    convolution / linear parameter is bit-identical to the one of ATMarineTrunk() (the initial state is the same exact identity map for the whole network, as at_marine documents).
  * Train / eval: InstanceNorm2d uses per-image statistics in both modes (no running statistics), so there is no train / eval normalisation mismatch and no dependence on the other images of the batch.
  * forward() is inherited unchanged. final_output_only stays True (class attribute of ATMarineTrunk), so train_v2.build wraps the R3 loss exactly as for AT-Trunk.
The original models/at_marine.py and original_repo/marine_net_model.py are not modified; ATMarineTrunk keeps its BatchNorm2d layers."""
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.at_marine import ATMarineTrunk  # noqa: E402

EXPECTED_PARAMS_IN = 1_952_768  # == ATMarineTrunk's count
N_NORM_LAYERS = 40
NORM_CHANNELS = (80,) * 8 + (160,) * 8 + (320,) * 8 + (160,) * 8 + (80,) * 8  # module order: encoder stages 1-3, decoder stages 1-2 (8 BatchNorm2d each)
BN_BUFFERS_REMOVED = 3 * N_NORM_LAYERS  # running_mean, running_var, num_batches_tracked per replaced layer


def make_instancenorm(c):
    return nn.InstanceNorm2d(c, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)


def replace_batch_norms(module):
    """In-place: every nn.BatchNorm2d below `module` becomes an InstanceNorm2d with the same channel count. Returns the channel counts in module order.
    Raises unless every BatchNorm2d is affine with running statistics (as in the official block) and exactly the 40 expected sites were replaced."""
    channels = []
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                if not child.affine or not child.track_running_stats:
                    raise RuntimeError("unexpected BatchNorm2d configuration (expected affine=True, track_running_stats=True)")
                setattr(parent, name, make_instancenorm(child.num_features))
                channels.append(child.num_features)
    if len(channels) != N_NORM_LAYERS or tuple(channels) != NORM_CHANNELS:
        raise RuntimeError(f"expected {N_NORM_LAYERS} BatchNorm2d layers with channels {NORM_CHANNELS}, replaced {channels}")
    return channels


class ATMarineTrunkIN(ATMarineTrunk):
    """ATMarineTrunk (default configuration only) with InstanceNorm2d in place of its 40 BatchNorm2d layers. forward() and final_output_only are inherited."""

    def __init__(self):
        super().__init__()  # identical construction (and random-number use) to ATMarineTrunk()
        replace_batch_norms(self)


def norm_layers(model):
    """-> (n_instance_norm, n_batch_norm) among all submodules."""
    mods = list(model.modules())
    return sum(isinstance(m, nn.InstanceNorm2d) for m in mods), sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in mods)
