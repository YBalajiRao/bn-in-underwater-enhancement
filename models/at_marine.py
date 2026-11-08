"""AT-MARINE (Adaptive-Transform MARINE-Net), Stage 4A variants (the project plan Sections 5 and 7).

Design (Stage 3 decision): M2's colour branch + a spatial detail-restoration branch.
    J = clip( A(x) * I + b(x) + s * R(x) ),   A(x) = A_global + dA(x),  b(x) = b_global + db(x)
  * global part : 12 numbers (3x3 matrix + 3 bias) from pooled features; last layer zero weights, bias = identity.
  * grid part   : G x G x 12 affine residual, 0.2*tanh bounded, bilinearly upsampled; last conv zero-initialised.
  * detail part : full-resolution residual R(x); its FINAL layer is zero-initialised; s is a learnable scalar
                  (init 0.1, the project plan Stage 4). The affine is evaluated in fp32 (no autocast).
  => at initialisation the whole model is exactly the identity: clip(I) == I for I in [0, 1].
  * NO INR branch. No loss-related code here: train_v2 trains these with the frozen R3 loss on "enhanced".

Variants (selected by name in train_v2.MODEL_REGISTRY, or built directly):
  at_tiny  (ATMarineTiny)  : M2's colour branch (GlobalLocalColorNet, reused unchanged) + a light
                             depthwise-separable U-shaped detail branch applied to the input image.
  at_trunk (ATMarineTrunk) : MARINE trunk (initial_conv, freq_decompose, encoder/decoder blocks, same module
                             names as original_repo MARINENet, no INR); colour parameters from the trunk
                             BOTTLENECK; detail residual from the trunk DECODER output.
Optional components (flags): use_global, use_grid, grid_size, use_detail (both); use_fad, use_pga (at_trunk).
With use_detail=False and both colour parts on, at_tiny computes exactly M2's function (same parameter names
under `.colour`).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "original_repo"))
from models_v2.color_nets import GlobalLocalColorNet, IDENTITY_12, apply_affine  # noqa: E402
from original_repo.marine_net_model import (AdaptivePhysicsGuidedAttention, FrequencyAwareDecomposition,  # noqa: E402
                                            LightweightResidualBlock)

DETAIL_SCALE_INIT = 0.1


def _compose(x, g, d, detail, scale, use_detail):
    """J = clip(A(x) I + b(x) + s R). g [B,12] or None (-> identity), d [B,12,H,W] or None, detail [B,3,H,W] or None."""
    p = (g if g is not None else IDENTITY_12.to(x.device, torch.float32).expand(x.shape[0], 12))[:, :, None, None]
    if d is not None:
        p = p + d
    out = apply_affine(x, p)
    if use_detail:
        out = out + scale.float() * detail.float()
    return torch.clamp(out, 0, 1)


# ================================================================================ at_tiny
class DSBlock(nn.Module):
    """Depthwise-separable residual block (no normalisation, as in the M1/M2 colour nets)."""

    def __init__(self, c):
        super().__init__()
        self.dw, self.pw = nn.Conv2d(c, c, 3, 1, 1, groups=c), nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x + F.leaky_relu(self.pw(self.dw(x)), 0.2)


class DSDown(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw, self.pw = nn.Conv2d(cin, cin, 3, 2, 1, groups=cin), nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return F.leaky_relu(self.pw(self.dw(x)), 0.2)


class TinyDetailBranch(nn.Module):
    """Light U-shaped branch, full-resolution output: stem -> [DS x2] -> down -> [DS x2] -> down -> [DS x2]
    -> (bilinear up + 1x1 + skip) x2 -> final 1x1 conv to 3 channels (zero-initialised)."""

    def __init__(self, width=(32, 64, 128), blocks=2):
        super().__init__()
        c1, c2, c3 = width
        self.stem = nn.Conv2d(3, c1, 3, 1, 1)
        self.e1 = nn.Sequential(*[DSBlock(c1) for _ in range(blocks)])
        self.d1, self.e2 = DSDown(c1, c2), nn.Sequential(*[DSBlock(c2) for _ in range(blocks)])
        self.d2, self.e3 = DSDown(c2, c3), nn.Sequential(*[DSBlock(c3) for _ in range(blocks)])
        self.u2, self.u1 = nn.Conv2d(c3, c2, 1), nn.Conv2d(c2, c1, 1)
        self.dec2, self.dec1 = DSBlock(c2), DSBlock(c1)
        self.out = nn.Conv2d(c1, 3, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        f1 = self.e1(F.leaky_relu(self.stem(x), 0.2))
        f2 = self.e2(self.d1(f1))
        f3 = self.e3(self.d2(f2))
        y2 = self.dec2(f2 + self.u2(F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)))
        y1 = self.dec1(f1 + self.u1(F.interpolate(y2, size=f1.shape[-2:], mode="bilinear", align_corners=False)))
        return self.out(y1)


class ATMarineTiny(nn.Module):
    final_output_only = True  # train_v2: MARINELoss on "enhanced" only (aux terms zero), as for M1/M2

    def __init__(self, use_global=True, use_grid=True, grid_size=16, use_detail=True, detail_width=(32, 64, 128), detail_blocks=2):
        super().__init__()
        self.use_global, self.use_grid, self.use_detail, self.grid_size = use_global, use_grid, use_detail, grid_size
        self.colour = GlobalLocalColorNet(grid=grid_size)  # M2 colour branch, reused unchanged
        if use_detail:
            self.detail = TinyDetailBranch(detail_width, detail_blocks)
            self.detail_scale = nn.Parameter(torch.tensor(DETAIL_SCALE_INIT))

    def forward(self, x):
        c = self.colour
        f16, g = c.features(x)
        d = None
        if self.use_grid:
            if f16.shape[-1] != c.grid:
                f16 = F.adaptive_avg_pool2d(f16, c.grid)
            d = F.interpolate(c.bound * torch.tanh(c.grid_head(f16)), size=x.shape[-2:], mode="bilinear", align_corners=False)
        R = self.detail(x) if self.use_detail else None
        out = _compose(x, g if self.use_global else None, d, R, getattr(self, "detail_scale", None), self.use_detail)
        return {"enhanced": out, "global_params": g, "grid_residual": d, "detail": R}


# ================================================================================ at_trunk
class ATMarineTrunk(nn.Module):
    final_output_only = True

    def __init__(self, base_channels=80, num_blocks=4, use_fad=True, use_pga=False, use_global=True, use_grid=True,
                 grid_size=16, use_detail=True, bound=0.2, hidden=64):
        super().__init__()
        self.use_fad, self.use_pga, self.use_global, self.use_grid = use_fad, use_pga, use_global, use_grid
        self.use_detail, self.grid_size, self.bound = use_detail, grid_size, bound
        # ---- MARINE trunk (module names identical to MARINENet) ----
        self.initial_conv = nn.Conv2d(3, base_channels, 3, padding=1)
        if use_fad:
            self.freq_decompose = FrequencyAwareDecomposition(base_channels)
        self.encoder_blocks, self.downsample = nn.ModuleList(), nn.ModuleList()
        ch = base_channels
        for i in range(3):
            self.encoder_blocks.append(nn.Sequential(*[LightweightResidualBlock(ch) for _ in range(num_blocks)]))
            if i < 2:
                self.downsample.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)); ch *= 2
        self.bott_ch = ch
        if use_pga:
            self.physics_attention = AdaptivePhysicsGuidedAttention(ch)
        if use_detail:
            self.decoder_blocks, self.upsample = nn.ModuleList(), nn.ModuleList()
            for i in range(2):
                ch //= 2
                self.upsample.append(nn.ConvTranspose2d(ch * 2, ch, 3, stride=2, padding=1, output_padding=1))
                self.decoder_blocks.append(nn.Sequential(*[LightweightResidualBlock(ch) for _ in range(num_blocks)]))
            self.detail_head = nn.Sequential(nn.Conv2d(base_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1))
            nn.init.zeros_(self.detail_head[-1].weight); nn.init.zeros_(self.detail_head[-1].bias)
            self.detail_scale = nn.Parameter(torch.tensor(DETAIL_SCALE_INIT))
        # ---- colour heads on the bottleneck ----
        if use_global:
            self.global_head = nn.Sequential(nn.Linear(self.bott_ch, hidden), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden, 12))
            nn.init.zeros_(self.global_head[-1].weight)
            with torch.no_grad():
                self.global_head[-1].bias.copy_(IDENTITY_12)
        if use_grid:
            self.grid_head = nn.Sequential(nn.Conv2d(self.bott_ch, hidden, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(hidden, 12, 3, 1, 1))
            nn.init.zeros_(self.grid_head[-1].weight); nn.init.zeros_(self.grid_head[-1].bias)

    def forward(self, x):
        xin = torch.clamp(x, 0, 1)
        f = self.initial_conv(xin)
        if self.use_fad:
            f = f + self.freq_decompose(xin)[0]
        skips = []
        for blocks, down in zip(self.encoder_blocks[:-1], self.downsample):
            f = blocks(f); skips.append(f); f = down(f)
        bott = self.encoder_blocks[-1](f)
        if self.use_pga:
            bott = self.physics_attention(bott, F.interpolate(xin, size=bott.shape[-2:], mode="bilinear", align_corners=False))
        g = self.global_head(bott.mean(dim=(2, 3))) if self.use_global else None
        d = None
        if self.use_grid:
            d = self.bound * torch.tanh(self.grid_head(F.adaptive_avg_pool2d(bott, self.grid_size)))
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        R = None
        if self.use_detail:
            h = bott
            for i, (up, blocks) in enumerate(zip(self.upsample, self.decoder_blocks)):
                h = up(h)
                if h.shape != skips[-(i + 1)].shape:
                    h = F.interpolate(h, size=skips[-(i + 1)].shape[-2:], mode="bilinear", align_corners=False)
                h = blocks(h + skips[-(i + 1)])
            R = self.detail_head(h)
        out = _compose(xin, g, d, R, getattr(self, "detail_scale", None), self.use_detail)
        return {"enhanced": out, "global_params": g, "grid_residual": d, "detail": R}
