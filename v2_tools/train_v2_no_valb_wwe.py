"""WWE-UIE wrapper around train_v2.main(): val_B removed BEFORE main() (never hashed or loaded), WWE recipes / model / composite loss registered,
strict pair validation on the FULL train and val_A lists. train_v2.py and the frozen evaluator are not modified.
Usage (NOT yet approved to run): python v2_tools/train_v2_no_valb_wwe.py --recipe WWE_lr2e-4 --model wwe_uie --seed 0 --run_name wwe_lr2e-4_s0"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_v2 as tv  # noqa: E402
from models.wwe_uie import EXPECTED_PARAMS, WWE_COMMIT, WWE_URL, WWEUIE  # noqa: E402
from v2_tools import wwe_config as cfg  # noqa: E402
from v2_tools.wwe_loss import WWEUIELoss  # noqa: E402
from v2_tools.wwe_pairing import checked_loader  # noqa: E402

assert "val_B" in tv.MANIFESTS, "unexpected: val_B not in MANIFESTS (train_v2 changed?)"
tv.MANIFESTS.pop("val_B")
assert "val_B" not in tv.MANIFESTS and not any("val_B" in str(p) for p, _, _ in tv.MANIFESTS.values())
assert not any("uieb" in str(p).lower() for p, _, _ in tv.MANIFESTS.values()), "a manifest path refers to UIEB"
assert "--recipe" in sys.argv and sys.argv[sys.argv.index("--recipe") + 1] in cfg.RECIPES, f"wrapper only runs recipes {sorted(cfg.RECIPES)}"
assert "--model" in sys.argv and sys.argv[sys.argv.index("--model") + 1] == "wwe_uie"
for k, v in cfg.RECIPES.items():
    assert k not in tv.RECIPES, "recipe name already taken"
    tv.RECIPES[k] = dict(v)
assert "wwe_uie" not in tv.MODEL_REGISTRY
tv.MODEL_REGISTRY["wwe_uie"] = lambda: WWEUIE()
_m = tv.MODEL_REGISTRY["wwe_uie"]()  # CPU build, no data access
assert sum(p.numel() for p in _m.parameters()) == EXPECTED_PARAMS, f"parameter count != {EXPECTED_PARAMS:,}"
del _m
tv.load_pairs = checked_loader(tv.load_pairs, cfg.EXPECTED_N)  # basename / count / duplicate / restricted-path assertions on the full list


def build(a, device):
    return WWEUIE().to(device), WWEUIELoss(device)


tv.build = build


def _write_loss_config(run_name):
    d = ROOT / "runs_v2" / run_name
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"wwe_commit": WWE_COMMIT, "wwe_url": WWE_URL, "loss_weights": cfg.LOSS_WEIGHTS, "density_k": cfg.DENSITY_K,
               "density_k_source": "RGB_HVI default (LOLv2 CIDNet checkpoint not loaded; deviation D6)", "log_key_map": cfg.LOG_KEY_MAP,
               "perceptual": "VGG16 relu1_2/2_2/3_3 MSE, no ImageNet normalisation (as official)"},
              open(d / "wwe_loss_config.json", "w"), indent=1)


if __name__ == "__main__":
    if "--run_name" in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        _write_loss_config(sys.argv[sys.argv.index("--run_name") + 1])
    tv.main()
