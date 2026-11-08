"""Compatibility wrapper for the UNOFFICIAL PyTorch port of WaterNet (external/waternet = github.com/tnwei/waternet, pinned commit below).

The port re-implements Li et al., "An Underwater Image Enhancement Benchmark Dataset and Beyond", IEEE TIP 2019 (official code: TensorFlow 1.x + MATLAB).
Only `waternet/net.py` (torch only) and `waternet/data.py` (numpy + cv2) are loaded, by file path under private module names. Nothing from the port's
training_utils.py (PNG-only, UIEB-only dataset class) or train.py is imported or executed.

Contract: forward(x) with x = RGB float32 [B,3,H,W] in [0,1] -> {"enhanced": [B,3,H,W]} (UNCLAMPED: the refiners end in ReLU, so the output is >= 0 but has
no upper bound; train_v2.evaluate clamps to [0,1]). The three pre-processed inputs are computed inside forward() from x exactly as the port's
dataset does (data.transform on the HWC uint8 RGB image):
    transform(rgb) -> (wb, gc, he)            # NOTE the return order: white balance, gamma, histogram-equalisation (LAB-CLAHE)
    WaterNet.forward(x, wb, ce, gc)           # NOTE the argument order: ce (= he) comes BEFORE gc
    port's train.py:  model(rgb_ten, wb_ten, he_ten, gc_ten)   -> this wrapper calls net(x, wb, he, gc)
Known difference to the original: the port uses OpenCV CLAHE (clipLimit 0.1, 8x8 tiles, on the L channel in LAB) instead of MATLAB `adapthisteq`
(the port's own README calls it 'close but not equivalent'). Weights are NOT loaded: the network is trained from the port's default initialisation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
WATERNET_DIR = ROOT / "external" / "waternet"
WATERNET_COMMIT = "2091896881fe0dfa5fb96007f29021a6067259cb"
WATERNET_URL = "https://github.com/tnwei/waternet.git"
EXPECTED_PARAMS = 1_090_668  # sum of p.numel() of the instantiated network (checked by the tests; see count_params)


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep the pinned checkout pristine (no __pycache__)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


_net = _load("_waternet_port_net", WATERNET_DIR / "waternet" / "net.py")
_data = _load("_waternet_port_data", WATERNET_DIR / "waternet" / "data.py")
WaterNet = _net.WaterNet
upstream_transform = _data.transform


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def collapsed_wb_channels(im):
    """Channels whose white-balance clipping range collapses (all values equal after clipping), replicating the port's / SimplestColorBalance.m's
    saturation levels and sequential clipping exactly. There, (x - bottom) * 255 / (top - bottom) is 0/0 = NaN for every pixel."""
    v = im.reshape(-1, 3).astype(np.float64)
    tot = v.sum(0)
    ratio = tot.max() / tot
    out = []
    for ch in range(3):
        t = np.quantile(v[:, ch], [0.005 * ratio[ch], 1 - 0.005 * ratio[ch]])
        temp = v[:, ch].copy()
        temp[temp < t[0]] = t[0]
        temp[temp > t[1]] = t[1]
        if temp.min() == temp.max():
            out.append(ch)
    return out


def preprocess_batch(x):
    """x float [B,3,H,W] in [0,1] (exactly k/255 on our uint8-backed pipeline) -> (wb, he, gc) float32 [B,3,H,W] in [0,1] on x.device.
    Round-trips through uint8 like the port's dataset (arr2ten divides by 255). Any other NaN/overflow in the port's code still raises
    (errstate 'raise') instead of silently producing garbage.
    Collapsed white-balance channel (e.g. UIEB 139_img_.png, red): the original MATLAB code yields NaN there and uint8(NaN) = 0, so that channel
    of wb is set to 0, matching MATLAB. Images without a collapsed channel take the unchanged path."""
    u8 = (x.detach().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    wbs, gcs, hes = [], [], []
    for im in u8:
        im = np.ascontiguousarray(im)
        bad = collapsed_wb_channels(im)
        if not bad:
            with np.errstate(all="raise"):
                wb, gc, he = upstream_transform(im)  # returns (wb, gc, he)
        else:
            with np.errstate(all="raise", invalid="ignore"):  # only the expected 0/0 and NaN->uint8 cast are tolerated
                wb, gc, he = upstream_transform(im)
            wb = wb.copy()
            wb[:, :, bad] = 0
        wbs.append(wb); gcs.append(gc); hes.append(he)

    def ten(a):
        return torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float().div(255.0).to(x.device)

    return ten(wbs), ten(hes), ten(gcs)


class WaterNetWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = WaterNet()

    def forward(self, x):
        wb, he, gc = preprocess_batch(x)
        return {"enhanced": self.net(x, wb, he, gc)}  # WaterNet.forward(x, wb, ce, gc): he is the 'ce' argument
