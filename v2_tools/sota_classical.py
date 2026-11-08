"""Classical baselines for the SOTA comparison (PREPARED; not yet run on any cohort). Pure numpy / OpenCV; no learned weights.

UDCP: Drews Jr., Nascimento, Moraes, Botelho, Campos, "Transmission Estimation in Underwater Single Images", ICCV Workshops 2013, pp. 825-830,
doi:10.1109/ICCVW.2013.113 (read directly from the CVF open-access PDF). Implemented from the paper, NOT from any third-party code:
  - underwater dark channel  J_UDCP(x) = min_{y in Omega(x)} ( min_{c in {G,B}} J_c(y) )                         (paper Eq. 9)
  - Omega = 15 x 15                                                                                                ("We defined the Omega(x) = 15 x 15")
  - airlight A: "the brightest pixel in the underwater dark channel", using He et al.'s rule restricted to G-B: among the
    top 0.1 % pixels of the underwater dark channel, the pixel of highest G-B intensity ((G + B) / 2); A takes that pixel's RGB value.
  - transmission  t~(x) = 1 - min_{y in Omega(x)} ( min_{c in {G,B}} I_c(y) / A_c )                                  (paper Eq. 10; no omega factor)
  - restoration by inverting I = J t + A (1 - t):  J = (I - A) / max(t, t0) + A                                   (paper Eq. 4 rearranged)
Stated substitutions (the paper does not specify them; recorded in the protocol): refinement by the guided filter (He, Sun, Tang, TPAMI 2013) on the
grey-level guide instead of soft matting; t0 = 0.1; output clipped to [0, 1].

CLAHE (Zuiderveld 1994) via OpenCV: L channel of CIE-Lab (8-bit OpenCV convention), clipLimit = 2.0, tileGridSize = (8, 8); a, b untouched."""
import cv2
import numpy as np

UDCP_PATCH = 15
UDCP_TOP_FRACTION = 0.001
UDCP_T0 = 0.1
GF_EPS = 1e-3
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)


def min_filter(a, patch):
    """Square min filter of side `patch` (erosion); borders use only in-image pixels."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(a.astype(np.float64), k, borderType=cv2.BORDER_CONSTANT, borderValue=np.inf)


def dark_channel(img, patch=UDCP_PATCH, channels=(1, 2)):
    """img: HxWx3 RGB float in [0,1]. Underwater dark channel over the given channels (G=1, B=2)."""
    return min_filter(img[:, :, list(channels)].min(axis=2), patch)


def airlight(img, dark, top_fraction=UDCP_TOP_FRACTION, channels=(1, 2)):
    n = max(1, int(round(top_fraction * dark.size)))
    idx = np.argpartition(dark.ravel(), -n)[-n:]                       # top 0.1 % of the dark channel
    inten = img[:, :, list(channels)].mean(axis=2).ravel()[idx]        # G-B intensity of those pixels
    k = idx[np.argmax(inten)]
    return img.reshape(-1, 3)[k].astype(np.float64)


def transmission_raw(img, A, patch=UDCP_PATCH, channels=(1, 2), omega=1.0):
    ratio = img[:, :, list(channels)] / np.maximum(A[list(channels)], 1e-6)
    return 1.0 - omega * min_filter(ratio.min(axis=2), patch)


def guided_filter(guide, p, r, eps=GF_EPS):
    """He et al. guided filter, grey guide, box radius r (window 2r+1)."""
    I = guide.astype(np.float64); p = p.astype(np.float64); w = (2 * r + 1, 2 * r + 1)
    box = lambda a: cv2.boxFilter(a, -1, w, normalize=True, borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def udcp(img, r, patch=UDCP_PATCH, t0=UDCP_T0, refine=True, channels=(1, 2), omega=1.0, return_parts=False):
    """img: HxWx3 RGB float64 in [0,1] -> restored HxWx3 float64 in [0,1]. r: guided-filter radius (fixed by the protocol)."""
    img = np.asarray(img, dtype=np.float64)
    dark = dark_channel(img, patch, channels)
    A = airlight(img, dark, channels=channels)
    t = transmission_raw(img, A, patch, channels, omega)
    if refine:
        grey = img @ np.array([0.299, 0.587, 0.114])
        t = guided_filter(grey, t, r)
    t_used = np.maximum(t, t0)
    J = np.clip((img - A) / t_used[..., None] + A, 0.0, 1.0)
    return (J, {"dark": dark, "A": A, "t": t, "t_used": t_used}) if return_parts else J


def clahe(img_rgb_uint8, clip=CLAHE_CLIP, tiles=CLAHE_TILES):
    """img: HxWx3 RGB uint8 -> RGB uint8. CLAHE on the L channel of Lab only."""
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    lab[:, :, 0] = op.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
