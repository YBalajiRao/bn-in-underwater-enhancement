"""WWE-UIE composite loss rebuilt from the pinned official sources (external/WWE-UIE @ fd06975).

Official classes reused unchanged (loaded by file path as package `_wwe_utils`; CIDNet / dataset / metrics are NOT imported):
  L1_Charbonnier_loss, SSIMLoss(window_size=5), EdgeAwareLoss('l2'), RGB_HVI.HVIT.
Re-implemented identically except for device handling: PerceptualLoss (official constructor hard-codes .cuda() and pretrained=True;
this version takes any device and an injectable feature extractor, so CPU tests need no weights). As in the official code, the
VGG16 input is NOT ImageNet-normalised (images in [0,1] go straight in).
Total = 1.0*Charbonnier + 0.5*HVI-Charbonnier + 0.1*(1-SSIM) + 0.1*VGG16 + 0.1*edge-L2   (train.py, lines 155-166).
Deviation D6: the HVI transform's density_k is the RGB_HVI default 0.2, not the value stored in the unloaded LOLv2 CIDNet checkpoint.
As in train.py the prediction is clamped to [0,1] only inside the HVI term; the other terms see the raw (unbounded) output."""
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.wwe_uie import WWE_DIR  # noqa: E402
from v2_tools.wwe_config import DENSITY_K, LOSS_WEIGHTS  # noqa: E402


def _load_official():
    name = "_wwe_utils"
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(WWE_DIR / "utils")]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(name + ".loss_funcs"), importlib.import_module(name + ".HVI_transform")
    finally:
        sys.dont_write_bytecode = prev


OFFICIAL_LOSS, OFFICIAL_HVI = _load_official()


def default_vgg16_features():
    """Equivalent of the official `vgg16(pretrained=True).features[:16]` (torchvision cache; must already be present - never downloads here
    unless the torchvision cache is empty, which the caller must check)."""
    from torchvision.models import VGG16_Weights, vgg16
    cached = Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth"
    if not cached.exists():
        raise FileNotFoundError(f"VGG16 weights not in the torch hub cache ({cached}); refusing to download")
    return vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]


class PerceptualVGG16(nn.Module):
    NAMES = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def __init__(self, features):
        super().__init__()
        self.vgg_layers = features
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

    def _feats(self, x):
        out = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.NAMES:
                out.append(x)
        return out

    def forward(self, pred, gt):
        loss = [F.mse_loss(a, b) for a, b in zip(self._feats(pred), self._feats(gt))]
        return sum(loss) / len(loss)


class WWEUIELoss(nn.Module):
    """crit(out, y) -> dict(total, l1, ssim, gradient, color, + hvi, perceptual); see wwe_config.LOG_KEY_MAP for the key mapping."""

    def __init__(self, device, density_k=DENSITY_K, vgg_features=None):
        super().__init__()
        device = torch.device(device)
        self.charb = OFFICIAL_LOSS.L1_Charbonnier_loss()
        self.ssimL = OFFICIAL_LOSS.SSIMLoss(device=device, window_size=5)
        self.edgeL = OFFICIAL_LOSS.EdgeAwareLoss(loss_type="l2", device=device)
        self.hvi = OFFICIAL_HVI.RGB_HVI()
        with torch.no_grad():
            self.hvi.density_k.fill_(density_k)
        self.hvi.density_k.requires_grad = False
        self.perc = PerceptualVGG16(vgg_features if vgg_features is not None else default_vgg16_features())
        self.to(device)
        self.w = LOSS_WEIGHTS

    def forward(self, out, y):
        pred = out["enhanced"]
        with torch.no_grad():
            label_hvi = self.hvi.HVIT(y)
        pred_hvi = self.hvi.HVIT(pred.clamp(0.0, 1.0))
        hvi = self.charb(pred_hvi, label_hvi)
        l1 = self.charb(pred, y)
        perc = self.perc(pred, y)
        ssim = self.ssimL(pred, y)
        edge = self.edgeL(pred, y)
        w = self.w
        total = w["charbonnier"] * l1 + w["hvi"] * hvi + w["ssim"] * ssim + w["perceptual"] * perc + w["edge"] * edge
        return {"total": total, "l1": l1, "ssim": ssim, "gradient": edge, "color": hvi, "hvi": hvi, "perceptual": perc}
