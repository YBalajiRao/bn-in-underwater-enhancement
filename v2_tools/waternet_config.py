"""WaterNet (UNOFFICIAL PyTorch port tnwei/waternet @ 2091896) adapted-protocol configuration: single source of truth for wrapper, loss, gate, driver and tests.

Upstream recipe (external/waternet/train.py, kept here for the record):
  Adam lr 1e-3 (betas default, no weight decay, no gradient clipping), StepLR(step_size=10000 optimizer steps, gamma=0.1) stepped per BATCH,
  400 epochs, batch 16, 112x112 (cv2.resize), random_split(890, [800, 90]) of UIEB, augmentation H-flip + V-flip + rot90 (each p=0.5, albumentations),
  loss = 0.05 * perceptual + mse, both on the 0-255 scale:
      mse        = mean((255*(out-ref))^2)
      perceptual = mean((255*(VGG19.features[:-1](imagenet_norm(out)) - VGG19.features[:-1](imagenet_norm(ref))))^2)   (stride-16 conv5_4 relu features)
  Network input order: model(raw, wb, he, gc). No sigmoid/clamp on the output.

Adapted protocol (deviations, all intentional and to be reported):
  D1 data: our manifest pairs (3,463 train; Val-A selection), 256x256 PIL Resize (not cv2.resize at 112), paired horizontal flip only (upstream: + vertical flip
     + rot90), no ColorJitter. The port's PNG-only UIEBDataset is bypassed; WB/GC/HE are computed inside the wrapper on the (flipped) uint8 image, as upstream does
     after augmentation.
  D2 batch 4 (upstream 16): lr scaled linearly, 1e-3 * 4/16 = 2.5e-4 (not tuned). Samples seen: 100 x 3463 = 346,300 vs upstream 400 x 800 = 320,000 (about equal);
     optimizer steps 86,600 vs upstream 20,000.
  D3 100 epochs (upstream 400 on 800 images). LR decay x0.1 at epoch 50: upstream's StepLR fires once inside its run at step 10,000 of 20,000 = 50% of training;
     the schedule is applied per epoch here (the project loop steps the scheduler once per epoch) instead of per batch.
  D4 Adam with no weight decay, no warm-up, gradient clipping DISABLED (max_norm 1e30; the project loop needs a number, and its logged grad-norm is kept) - as upstream.
  D5 best checkpoint = best Val-A PSNR with the frozen evaluator (upstream keeps only last.pt and validates on a random UIEB split).
  D6 loss unchanged (0.05 / 255-scale / VGG19 conv5_4 relu features, ImageNet-normalised, output not clamped). VGG weights: torchvision vgg19-dcbb9e9d.pth,
     loaded from a file, parameters frozen (upstream leaves requires_grad=True but excludes VGG from the optimizer: same optimisation, less work).
  D7 shuffle: the project loop uses a seeded RandomSampler (upstream train loader is not shuffled); validation-loss accumulation bug of the port is not reproduced
     because validation uses the frozen evaluator (no val loss is computed).
  D8 CLAHE: OpenCV (as the port), not MATLAB adapthisteq (the original's tool).
  D9 fp32, TF32 follows the project loop (training default, evaluation forces TF32 off).
  D10 seeds 0/1/2 (upstream seed 0, changeable by --seed which also changes its data split).
"""
LR = 2.5e-4
LR_DECAY_EPOCH = 50
LR_DECAY_GAMMA = 0.1
CLIP_DISABLED = 1e30
RECIPE = "WATERNET_lsui_lr2p5e-4"
BASE = dict(epochs=100, warmup=0, bs=4, accum=1, lr=LR, wd=0.0, eta_min=0.0, clip=CLIP_DISABLED, amp=False, loss_in_autocast=False,
            skip_zero_weight_perceptual=False, no_jitter=True, stop_after=None, select_on="val_A",
            lr_decay_epoch=LR_DECAY_EPOCH, lr_decay_gamma=LR_DECAY_GAMMA)
RECIPES = {RECIPE: dict(BASE)}
LOSS_WEIGHTS = dict(mse=1.0, perceptual=0.05)
PIXEL_SCALE = 255.0
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXPECTED_N = {"train": 3463, "val_A": 166}
VGG19_FILE = "vgg19-dcbb9e9d.pth"
VGG19_URL = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
VGG19_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
# train_v2 logs only these loss keys; WaterNet terms are mapped onto them (history.jsonl "l1/ssim/gradient/color"):
LOG_KEY_MAP = {"l1": "mse (0-255 scale)", "gradient": "perceptual (VGG19, unweighted, 0-255 scale)", "ssim": "n/a (always 0)", "color": "n/a (always 0)"}
