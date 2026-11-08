"""Visual panels for the SOTA HICRD-2000 comparison (spec: EFFECTIVE_v2 section 8; selection fixed by sota_hicrd_protocol.visual_selection).
PREPARED; part of the approved execution only. Uses the saved float outputs (visual_<method>.npy) of the Group R / Group T runs and runs OUR
MARINE-Net BN and IN (seed 0, best_valA, frozen evaluator loader) on the same 12 images only. Two figures (Group R; Group T labelled exploratory),
256x256, per-image PSNR (frozen function) under each tile. No image is swapped.
Usage: python v2_tools/sota_hicrd_visual.py --released DIR --retrained DIR --out paper1_uieb_runs/sota_hicrd_visual [--dry-run]"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402

R_ORDER = ["ushape", "wwe_uie", "funie_gan", "nu2net", "clahe", "udcp"]
T_ORDER = ["waternet_lsui_s0", "wwe_uie_lsui_s0"]
OURS = ["marinenet_bn_s0", "marinenet_in_s0"]


def panels(rows, cols, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.6 * len(cols), 1.75 * len(rows)), squeeze=False)
    for i, (imgs, labels) in enumerate(rows):
        for j, c in enumerate(cols):
            ax[i, j].imshow(np.clip(imgs[c].transpose(1, 2, 0), 0, 1)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_xlabel(labels[c], fontsize=6)
            if i == 0:
                ax[i, j].set_title(c, fontsize=6)
    fig.suptitle(title, fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--released", required=True); ap.add_argument("--retrained", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--test-manifest", default=None)
    a = ap.parse_args(argv)
    test = a.test_manifest is not None
    K.verify_protocol_files()
    rp = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    out = K.check_out(rp(a.out), test)
    pairs = K.load_pairs(a.test_manifest); ids = [p["id"] for p in pairs]
    vis = P.visual_selection() if not test else ids[:2]
    plan = {"visual_ids": vis, "out": str(out), "released": str(rp(a.released)), "retrained": str(rp(a.retrained))}
    if a.dry_run:
        print("DRY RUN OK:", json.dumps(plan)); return {"dry_run": True, "plan": plan}
    import torch
    import train_v2 as tv
    import paper1_final_eval_v2 as V
    sel = [pairs[ids.index(k)] for k in vis]
    X, Y = tv.preload(sel); x = X.float() / 255.0; y = Y.float() / 255.0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); K.set_tf32_off()
    ours = {}
    for k in OURS:
        path, _ = V.verify_checkpoint2(V.MODELS[k][0], "best_valA")
        m = V.load_model2(k, path, "best_valA", dev)
        with torch.no_grad():
            raw = m(x.to(dev))["enhanced"]; K.check_finite(raw, k); ours[k] = torch.clamp(raw, 0, 1).float().cpu()
    out.mkdir(parents=True)
    for grp, order, folder, title in (("R", R_ORDER, rp(a.released), "Group R: released checkpoints as published (descriptive; n = 1 each)"),
                                      ("T", T_ORDER, rp(a.retrained), "Group T: our LSUI retrains (EXPLORATORY; n = 1 baselines)")):
        outs = {m: np.load(folder / f"visual_{m}.npy") for m in order}
        rows = []
        for i in range(len(vis)):
            imgs = {"input": x[i].numpy(), "reference": y[i].numpy(), **{m: outs[m][i] for m in order}, **{k: ours[k][i].numpy() for k in OURS}}
            labels = {"input": vis[i], "reference": "reference"}
            for c in list(order) + OURS:
                labels[c] = f"{tv.compute_psnr(torch.from_numpy(np.ascontiguousarray(imgs[c], dtype=np.float32))[None], y[i:i + 1]):.2f} dB"
            rows.append((imgs, labels))
        panels(rows, ["input", "reference"] + list(order) + OURS, title, out / f"panels_group_{grp}.png")
    json.dump(dict(plan, protocol_sha256=K.PROTOCOL_V2[1]), open(out / "visual_meta.json", "x"), indent=1)
    return {"written": str(out)}


if __name__ == "__main__":
    main()
