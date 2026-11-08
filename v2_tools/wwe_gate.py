"""WWE-UIE validation gate for ONE finished run (adapted from v2_tools/armd_gate.py; NOT run yet - it re-scores on GPU and reads Val-A).
Usage: python v2_tools/wwe_gate.py --run wwe_lr2e-4_s0     (run name = wwe_<lr tag>_s<seed>, tag in wwe_config.LR_CANDIDATES)
Nine checks (same order and meaning as the Arm D gate; differences marked *):
  1 files exist   2 model == wwe_uie, params == 734,059   3 recipe args == the named WWE recipe*   4 100 contiguous epochs
  5 no non-finite steps / finite val-A   6 manifest hashes as expected, val_B absent, AND strict pair validation of the train and val_A lists*
  7 best epoch == argmax history   8 checkpoint SHA-256 == result.json   9 independent Val-A re-score == logged (1e-4)
Checks 2-5 and 7 are pure functions of result.json / history.jsonl (`static_checks`, CPU-tested with synthetic dicts).
val_B is removed from MANIFESTS before anything is hashed or loaded; UIEB is never referenced."""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.wwe_uie import EXPECTED_PARAMS, WWEUIE  # noqa: E402
from v2_tools import wwe_config as cfg  # noqa: E402
from v2_tools.wwe_pairing import validate_pairs  # noqa: E402

EXPECTED_HASHES = {"train": "b4871a98", "val": "c034978c", "val_clean": "dd48c348", "val_A": "2d8daae6"}
TOL = 1e-4


def recipe_name(run):
    tag, seed = run[len("wwe_"):].rsplit("_s", 1)
    assert run.startswith("wwe_") and f"WWE_{tag}" in cfg.RECIPES and seed.isdigit(), f"bad run name {run!r}"
    return f"WWE_{tag}"


def static_checks(r, hist, run):
    """-> list of (name, passed, info); no file or GPU access."""
    out = []
    rn = recipe_name(run)
    out.append(("model/params", r["model"] == "wwe_uie" and r["params"] == EXPECTED_PARAMS, f"{r['model']} {r['params']:,}"))
    want = cfg.RECIPES[rn]
    bad = {k: r["args"].get(k) for k, v in want.items() if r["args"].get(k) != v}
    out.append(("recipe args == WWE recipe", r["args"].get("recipe") == rn and not bad, bad or rn))
    out.append(("100 contiguous epochs", [h["epoch"] for h in hist] == list(range(100)), len(hist)))
    out.append(("no non-finite loss steps / finite val-A", all(h["train"]["nonfinite"] == 0 and np.isfinite(h["val_A"]["psnr_mean"]) and np.isfinite(h["train"]["total"]) for h in hist),
                f"inf-grad steps total {sum(h['train']['inf_grad_steps'] for h in hist)}"))
    best = max(hist, key=lambda h: h["val_A"]["psnr_mean"])
    out.append(("best epoch == argmax history", best["epoch"] == r["best_valA_epoch"] and abs(best["val_A"]["psnr_mean"] - r["best_valA"]["psnr_mean"]) < 1e-9, f"epoch {best['epoch']}"))
    return out


def main():
    import torch
    import train_v2 as tv
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    tv.MANIFESTS.pop("val_B", None)
    assert "val_B" not in tv.MANIFESTS
    d = ROOT / "runs_v2" / a.run
    checks, ok = {}, True

    def rec(name, cond, info=""):
        nonlocal ok
        checks[name] = {"pass": bool(cond), "info": str(info)}; ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)

    files = ["result.json", "history.jsonl", "best_valA.pth", "last.pth", "wwe_loss_config.json"]
    rec("files exist", all((d / f).exists() for f in files))
    if not all((d / f).exists() for f in files):
        json.dump({"pass": False, "checks": checks}, open(d / "wwe_gate.json", "w"), indent=1); sys.exit(1)
    r = json.load(open(d / "result.json")); hist = [json.loads(l) for l in open(d / "history.jsonl")]
    for name, cond, info in static_checks(r, hist, a.run)[:4]:
        rec(name, cond, info)
    mh = r["manifest_sha256"]
    try:
        for k in ("train", "val_A"):
            validate_pairs(tv.load_pairs(k), cfg.EXPECTED_N[k], k)
        pairs_ok, pinfo = True, "strict pair validation passed"
    except AssertionError as e:
        pairs_ok, pinfo = False, str(e)
    rec("manifest hashes as expected, val_B absent, pairs valid", "val_B" not in mh and all(mh.get(k, "").startswith(v) for k, v in EXPECTED_HASHES.items()) and pairs_ok, pinfo)
    nm, c, i = static_checks(r, hist, a.run)[4]
    rec(nm, c, i)
    rec("checkpoint SHA-256 == result.json", all(tv.sha256(d / f) == h for f, h in r["checkpoint_sha256"].items()) and set(r["checkpoint_sha256"]) == {"best_valA.pth", "last.pth"})
    dev = torch.device("cuda")
    m = WWEUIE().to(dev)
    m.load_state_dict(torch.load(d / "best_valA.pth", map_location=dev, weights_only=False)["model_state_dict"], strict=True)
    Xv, Yv = tv.preload(tv.load_pairs("val_A"))
    ps, ss = tv.evaluate(m, Xv, Yv, dev)
    s = tv.summarize(ps, ss)
    dp, ds = abs(s["psnr_mean"] - r["best_valA"]["psnr_mean"]), abs(s["ssim_mean"] - r["best_valA"]["ssim_mean"])
    rec("val-A re-score == logged", s["n"] == 166 and dp < TOL and ds < TOL, f"n={s['n']} PSNR {s['psnr_mean']:.6f} (diff {dp:.2e}) SSIM {s['ssim_mean']:.6f} (diff {ds:.2e})")
    json.dump({"run": a.run, "pass": ok, "checks": checks, "rescore": s}, open(d / "wwe_gate.json", "w"), indent=1)
    print("GATE", "PASSED" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
