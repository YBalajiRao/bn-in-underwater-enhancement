"""Compatibility wrapper for the official WWE-UIE network (external/WWE-UIE, pinned commit below).

Only `external/WWE-UIE/model.py` is loaded (by file path, under a private module name): it imports torch only. Nothing from
the repository's `utils/` package is imported, so no loss, dataset, metric or weight file is touched (utils/loss_funcs.py
needs torchvision VGG weights and a CUDA device; utils/dataset.py pairs files by sorted index - neither is used here).
The wrapper changes nothing in the network: it returns the authors' unbounded output under the key "enhanced"
(train_v2.evaluate clamps to [0,1]). Constructor arguments are the ones train.py / test.py use.
"""
import importlib.util
import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
WWE_DIR = ROOT / "external" / "WWE-UIE"
WWE_COMMIT = "fd0697558203f9f849800cc13b804116bb3f921c"
WWE_URL = "https://github.com/chingheng0808/WWE-UIE.git"
EXPECTED_PARAMS = 734_059  # counted from the instantiated network (sum of p.numel(); buffers excluded)

_spec = importlib.util.spec_from_file_location("_wwe_uie_official_model", WWE_DIR / "model.py")
_mod = importlib.util.module_from_spec(_spec)
_prev_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True  # keep the pinned checkout pristine (no __pycache__)
try:
    _spec.loader.exec_module(_mod)
finally:
    sys.dont_write_bytecode = _prev_bytecode
myModel = _mod.myModel


class WWEUIE(nn.Module):
    """RGB [B,3,H,W] in [0,1] -> {"enhanced": [B,3,H,W]} (unclamped). H and W must be multiples of 4 (two 2x down-samplings)."""

    def __init__(self):
        super().__init__()
        self.net = myModel(in_channels=3, feature_channels=32, use_white_balance=True)  # as in train.py / test.py

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError(f"WWE-UIE needs H and W divisible by 4, got {h}x{w}")
        return {"enhanced": self.net(x)}


def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)
