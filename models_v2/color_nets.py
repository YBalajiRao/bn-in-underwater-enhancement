"""Stage 3 colour-transform models (the project plan Section 5, Stage 3).

M1 GlobalColorNet: small CNN on a 64x64 thumbnail -> GAP -> MLP -> 12 numbers
   (3x3 matrix A + 3 bias b). Output = clip(A * I + b), applied per pixel at full resolution.
M2 GlobalLocalColorNet: M1 + a 16x16 grid of 12-number affine residuals (predicted from the
   16x16 thumbnail feature map), bounded by 0.2*tanh, bilinearly upsampled to full size.
   Output = clip(A(x) * I + b(x)) with A(x) = A + dA(x), b(x) = b + db(x).
Both start as the identity mapping (output == input): the last MLP layer has zero weights and
bias = identity parameters; the grid head's last conv is zero-initialised (tanh(0) = 0).
No normalisation layers (no BatchNorm). Forward returns {"enhanced": ...} only; train_v2 then
uses MARINELoss on the final output with the CNN/INR auxiliary terms set to zero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IDENTITY_12 = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.])  # A (row-major) then b


def _conv(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.2, inplace=True))


def apply_affine(img, p):
    """img [B,3,H,W]; p [B,12] (global) or [B,12,H,W] (per pixel). Returns A*img + b (unclipped).
    Always computed in fp32 (autocast off), so the identity mapping is exact even under AMP."""
    with torch.autocast("cuda", enabled=False):
        img, p = img.float(), p.float()
        if p.dim() == 2:
            p = p[:, :, None, None]
        A = p[:, :9].reshape(p.shape[0], 3, 3, *p.shape[2:])
        b = p[:, 9:]
        return (torch.einsum("bij...,bj...->bi...", A, img) + b).contiguous()  # standard NCHW layout for the frozen metrics


class GlobalColorNet(nn.Module):
    final_output_only = True  # train_v2: MARINELoss with zero auxiliary (cnn/inr) terms

    def __init__(self, thumb=64, width=(16, 32, 64, 128), hidden=64):
        super().__init__()
        self.thumb = thumb
        c1, c2, c3, c4 = width
        self.stem = nn.Sequential(_conv(3, c1, 2), _conv(c1, c2, 2))        # 64 -> 32 -> 16
        self.down = nn.Sequential(_conv(c2, c3, 2), _conv(c3, c4, 2))       # 16 -> 8 -> 4
        self.mlp = nn.Sequential(nn.Linear(c4, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(IDENTITY_12)

    def thumbnail(self, x):
        return F.interpolate(x, size=(self.thumb, self.thumb), mode="bilinear", align_corners=False, antialias=True)

    def features(self, x):
        f16 = self.stem(self.thumbnail(x))
        g = self.mlp(self.down(f16).mean(dim=(2, 3)))
        return f16, g

    def forward(self, x):
        _, g = self.features(x)
        return {"enhanced": torch.clamp(apply_affine(x, g), 0, 1), "global_params": g}


class GlobalLocalColorNet(GlobalColorNet):
    def __init__(self, grid=16, bound=0.2, **kw):
        super().__init__(**kw)
        c2 = self.stem[-1][0].out_channels
        self.grid, self.bound = grid, bound
        self.grid_head = nn.Sequential(_conv(c2, c2, 1), nn.Conv2d(c2, 12, 3, 1, 1))
        nn.init.zeros_(self.grid_head[-1].weight); nn.init.zeros_(self.grid_head[-1].bias)

    def forward(self, x):
        f16, g = self.features(x)
        if f16.shape[-1] != self.grid:
            f16 = F.adaptive_avg_pool2d(f16, self.grid)
        d = self.bound * torch.tanh(self.grid_head(f16))                          # [B,12,16,16]
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        p = g[:, :, None, None] + d
        return {"enhanced": torch.clamp(apply_affine(x, p), 0, 1), "global_params": g, "grid_residual": d}
