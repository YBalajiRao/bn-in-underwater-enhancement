"""WaterNet training loss rebuilt from the pinned port (external/waternet @ 2091896, train.py lines ~96-130):
    total = 0.05 * perceptual + mse,   everything on the 0-255 scale
    mse        = mean((255 * (out - ref))^2)
    perceptual = mean((255 * (F(norm(out)) - F(norm(ref))))^2),  F = torchvision VGG19 .features[:-1] (final max-pool dropped; stride 16, 512 channels),
                 norm = ImageNet mean/std normalisation, applied to the UNCLAMPED output (as upstream)
Differences (documented in waternet_config): VGG weights come from a pinned local file (no download at run time), VGG parameters are frozen
(upstream leaves requires_grad=True but excludes them from the optimizer, so the optimisation is identical), the feature extractor is injectable for CPU tests.
Returns the dict train_v2.compute_loss expects: total / l1 (=mse) / ssim (0) / gradient (=perceptual, unweighted) / color (0)."""
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from v2_tools import waternet_config as cfg  # noqa: E402


def vgg19_weights_path():
    import torch.hub
    return Path(torch.hub.get_dir()) / "checkpoints" / cfg.VGG19_FILE


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_vgg19_features(path=None):
    """torchvision vgg19 (weights=None) + the pinned state dict, hash-verified; returns features[:-1] in eval mode with frozen parameters."""
    import torchvision
    path = Path(path) if path else vgg19_weights_path()
    if sha256_file(path) != cfg.VGG19_SHA256:
        raise RuntimeError(f"{path} SHA-256 != pinned {cfg.VGG19_SHA256}")
    m = torchvision.models.vgg19(weights=None)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    f = nn.Sequential(*list(m.features.children())[:-1])
    for p in f.parameters():
        p.requires_grad_(False)
    return f.eval()


class WaterNetLoss(nn.Module):
    def __init__(self, device, extractor=None):
        super().__init__()
        self.vgg = (extractor if extractor is not None else build_vgg19_features()).to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(cfg.IMAGENET_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor(cfg.IMAGENET_STD).view(1, 3, 1, 1).to(device))

    def train(self, mode=True):
        super().train(mode)
        self.vgg.eval()  # the feature extractor is always in eval mode
        return self

    def forward(self, out, y):
        pred = out["enhanced"]
        s = cfg.PIXEL_SCALE
        mse = torch.mean(torch.square(s * (pred - y)))
        fx = self.vgg((pred - self.mean) / self.std)
        fy = self.vgg((y - self.mean) / self.std)
        perceptual = torch.mean(torch.square(s * (fx - fy)))
        total = cfg.LOSS_WEIGHTS["mse"] * mse + cfg.LOSS_WEIGHTS["perceptual"] * perceptual
        z = pred.new_zeros(())
        return {"total": total, "l1": mse, "ssim": z, "gradient": perceptual, "color": z}
