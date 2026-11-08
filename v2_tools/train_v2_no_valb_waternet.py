"""WaterNet (unofficial PyTorch port) wrapper around train_v2.main(): val_B removed BEFORE main() (never hashed or loaded), WaterNet recipe / model / loss / optimizer
registered, strict pair validation on the FULL train and val_A lists. train_v2.py and the frozen evaluator are not modified.
Usage (NOT yet approved to run on GPU): python v2_tools/train_v2_no_valb_waternet.py --recipe WATERNET_lsui_lr2p5e-4 --model waternet --seed 0 --run_name waternet_lsui_s0"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.waternet_wrap import EXPECTED_PARAMS, WATERNET_COMMIT, WATERNET_URL, WaterNetWrap, count_params  # noqa: E402
from v2_tools import waternet_config as cfg  # noqa: E402
from v2_tools.waternet_loss import WaterNetLoss, sha256_file, vgg19_weights_path  # noqa: E402
from v2_tools.waternet_pairing import checked_loader  # noqa: E402

assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
tv.MANIFESTS.pop("val_B")
assert "val_B" not in tv.MANIFESTS and not any("val_B" in str(p) for p, _, _ in tv.MANIFESTS.values())
assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
assert "--recipe" in sys.argv and sys.argv[sys.argv.index("--recipe") + 1] in cfg.RECIPES, f"wrapper only runs recipes {sorted(cfg.RECIPES)}"
assert "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "waternet"
for k, v in cfg.RECIPES.items():
    assert k not in tv.RECIPES, "recipe name already taken"
    tv.RECIPES[k] = dict(v)
assert "waternet" not in tv.MODEL_REGISTRY
tv.MODEL_REGISTRY["waternet"] = lambda: WaterNetWrap()
_m = tv.MODEL_REGISTRY["waternet"]()  # CPU build, no data access
assert count_params(_m) == EXPECTED_PARAMS, f"parameter count != {EXPECTED_PARAMS:,}"
del _m
tv.load_pairs = checked_loader(tv.load_pairs, cfg.EXPECTED_N)  # basename / count / duplicate / restricted-path assertions on the full list


def build(a, device):
    return WaterNetWrap().to(device), WaterNetLoss(device)


def make_optim(model, a):
    """Upstream optimizer: plain Adam (no weight decay), lr 1e-3 * 4/16, x0.1 once at 50% of training (upstream StepLR step 10,000 of 20,000 optimizer steps)."""
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[a.lr_decay_epoch], gamma=a.lr_decay_gamma)
    return opt, sched


tv.build = build
tv.make_optim = make_optim


def _write_loss_config(run_name):
    d = ROOT / "runs_v2" / run_name
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"waternet_commit": WATERNET_COMMIT, "waternet_url": WATERNET_URL, "implementation": "UNOFFICIAL PyTorch port (tnwei/waternet), MIT; original is TF1.x + MATLAB",
               "loss_weights": cfg.LOSS_WEIGHTS, "pixel_scale": cfg.PIXEL_SCALE, "vgg19_file": cfg.VGG19_FILE, "vgg19_url": cfg.VGG19_URL,
               "vgg19_sha256_pinned": cfg.VGG19_SHA256, "vgg19_sha256_observed": sha256_file(vgg19_weights_path()), "log_key_map": cfg.LOG_KEY_MAP,
               "params": EXPECTED_PARAMS, "optimizer": f"Adam lr {cfg.LR}, MultiStepLR [{cfg.LR_DECAY_EPOCH}] gamma {cfg.LR_DECAY_GAMMA}, clip disabled (1e30)",
               "preprocessing": "data.transform(rgb uint8) -> (wb, gc, he); net(x, wb, he, gc); OpenCV CLAHE (clipLimit 0.1, 8x8, LAB L) - not MATLAB adapthisteq"},
              open(d / "waternet_loss_config.json", "w"), indent=1)


if __name__ == "__main__":
    if "--run_name" in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        _write_loss_config(sys.argv[sys.argv.index("--run_name") + 1])
    tv.main()
