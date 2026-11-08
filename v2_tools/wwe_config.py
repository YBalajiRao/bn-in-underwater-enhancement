"""WWE-UIE adapted-protocol configuration (documentation + single source of truth for wrapper, gate and tests).

Authors' recipe (external/WWE-UIE @ fd06975, train.py), kept here for the record:
  AdamW lr 2e-3 betas (0.9,0.999) wd 1e-5; CosineAnnealingLR(T_max=epochs, eta_min=lr*1e-4) stepped per epoch, no warm-up;
  100 epochs, batch 24, fp32 (no AMP), no gradient clipping, seed 7, best epoch by validation PSNR;
  loss = Charbonnier + 0.5*HVI-Charbonnier + 0.1*(1-SSIM, window 5) + 0.1*VGG16(relu1_2,2_2,3_3) MSE + 0.1*Sobel-edge L2.

Adapted protocol (deviations, all intentional and to be reported):
  D1 data: our manifest pairs (3,463 train, Val-A selection), 256x256 Resize, paired horizontal flip only (authors: vertical flip +
     mirror + 90/180/270 rotations); no ColorJitter.
  D2 batch 4 (authors 24) -> 865.75 -> 866 iterations/epoch, 86,600 total (authors ~14,500).
  D3 lr: not the authors' 2e-3 (batch-24 value); candidates below, selection plan in the report; eta_min = lr*1e-4 kept as authors.
  D4 3-epoch linear warm-up and gradient clipping at 1.0 (project loop, R3 parity); the authors use neither.
  D5 best checkpoint chosen on Val-A with the frozen evaluator (authors: their own validation folder, skimage metrics).
  D6 HVI loss uses density_k = 0.2 (RGB_HVI default); the authors' loss takes it from utils/CIDNet_weight_LOLv2_bestSSIM.pth, which was
     deliberately NOT loaded; no documentation states the value (README, train.py, commit log checked) -> unverified default.
  D7 seeds 0/1/2 (authors 7).
fp32 (amp=False) as the authors; the project loop still applies its loss in fp32 via compute_loss.
"""
LR_CANDIDATES = {"lr2e-4": 2e-4,    # R3 value used for every other arm (parity)
                 "lr3p3e-4": 3.3e-4,  # authors' 2e-3 linearly scaled by batch 4/24 (3.33e-4)
                 "lr8e-4": 8e-4}      # authors' 2e-3 scaled by sqrt(4/24) (8.2e-4)
BASE = dict(epochs=100, warmup=3, bs=4, accum=1, wd=1e-5, clip=1.0, amp=False, loss_in_autocast=False,
            skip_zero_weight_perceptual=False, no_jitter=True, stop_after=None, select_on="val_A")
RECIPES = {f"WWE_{tag}": {**BASE, "lr": lr, "eta_min": lr * 1e-4} for tag, lr in LR_CANDIDATES.items()}
LOSS_WEIGHTS = dict(charbonnier=1.0, hvi=0.5, ssim=0.1, perceptual=0.1, edge=0.1)
DENSITY_K = 0.2
EXPECTED_N = {"train": 3463, "val_A": 166}
# train_v2 logs only these loss keys; WWE terms are mapped onto them (history.jsonl "l1/ssim/gradient/color"):
LOG_KEY_MAP = {"l1": "charbonnier", "ssim": "1-SSIM(w5)", "gradient": "edge L2", "color": "HVI-Charbonnier"}  # perceptual is in total only
