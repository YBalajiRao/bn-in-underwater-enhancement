"""Stage 3 inference: the 12 matched fp32 checkpoints (MARINE-Net and AT-Trunk, BN and IN, seeds 0-2) on a UIEB manifest. PSNR and SSIM per image
with the frozen evaluator's score_model (Resize 256, clamp, TF32 off); no LPIPS (not part of the Stage 3 analysis).
Order of checks, all before any UIEB pixel is decoded: phrase; frozen Stage 3 protocol hash; Val-B-free manifest table; all 12 checkpoint hashes;
UIEB manifest hash, path/marker checks and every recorded per-image hash (paper1_final_eval_v2.load_uieb_manifest2, unchanged).
Ledger: one 'started' and one 'completed'/'failed' event in paper1_uieb_runs/ledger (dataset label 'uieb_stage3'). Output folder is created exclusively.
Usage: python v2_tools/stage3_infer.py --uieb-phrase "UIEB approved" --manifest M --manifest-sha256 H --expected-pairs N --checkpoint best_valA|last
       --protocol P --protocol-sha256 HP --label L --out DIR"""
import argparse, csv, json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import paper1_final_eval_v2 as V  # noqa: E402

E = V.E
MODELS12 = [f"{a}_s{s}" for a in ("marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in") for s in (0, 1, 2)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("--uieb-phrase", "--manifest", "--manifest-sha256", "--checkpoint", "--protocol", "--protocol-sha256", "--label", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--expected-pairs", type=int, required=True)
    a = ap.parse_args(argv)
    if a.uieb_phrase != "UIEB approved":
        raise SystemExit("STOP: UIEB phrase missing or wrong")
    if E.sha256_file(a.protocol) != a.protocol_sha256.lower():
        raise SystemExit(f"STOP: protocol {a.protocol} hash differs from {a.protocol_sha256}")
    if a.checkpoint not in V.CKPT_FILES:
        raise SystemExit("STOP: checkpoint must be best_valA or last")
    import torch
    import train_v2 as tv
    E.restrict_manifests(tv.MANIFESTS)  # Val-B (and every split except val_A) removed before anything is read
    ckpts = {k: V.verify_checkpoint2(V.MODELS[k][0], a.checkpoint) for k in MODELS12}
    pairs, mh, extra = V.load_uieb_manifest2(a.manifest, a.manifest_sha256, a.expected_pairs, "UIEB", "stem")
    out = E.exclusive_dir(ROOT / a.out)
    rec = {"dataset": "uieb_stage3", "label": a.label, "out": str(out), "checkpoint": a.checkpoint, "manifest_sha256": mh, "protocol_sha256": a.protocol_sha256,
           "script_sha256": E.sha256_file(__file__), "models": MODELS12}
    V.ledger_write(V.LEDGER_DIR, dict(rec, event="started", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        X, Y = tv.preload(pairs)
        assert X.shape == Y.shape == (len(pairs), 3, 256, 256) and X.dtype == Y.dtype == torch.uint8
        dev = torch.device("cuda")
        per, res = {}, {}
        for k in MODELS12:
            m = V.load_model2(k, ckpts[k][0], a.checkpoint, dev)
            ps, ss, _, nf = E.score_model(m, X, Y, dev, None)
            if nf != 0 or not (np.isfinite(ps).all() and np.isfinite(ss).all()):
                raise SystemExit(f"STOP: non-finite output for {k}")
            per[k] = (ps, ss)
            res[k] = {"run": V.MODELS[k][0], "checkpoint_sha256": ckpts[k][1], "psnr": float(ps.mean()), "ssim": float(ss.mean()), "nonfinite_before_clamp": nf}
            print(k, round(float(ps.mean()), 5), flush=True)
            del m; torch.cuda.empty_cache()
        names = [Path(p["input_path"]).name for p in pairs]
        with open(out / "per_image.csv", "x", newline="") as f:
            w = csv.writer(f); w.writerow(["filename"] + [f"{k}_{m}" for k in MODELS12 for m in ("psnr", "ssim")])
            for i, n in enumerate(names):
                w.writerow([n] + [f"{per[k][j][i]:.6f}" for k in MODELS12 for j in (0, 1)])
        meta = dict(rec, n_images=len(pairs), created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__, numpy=np.__version__,
                    gpu=torch.cuda.get_device_name(0), recorded_hashes_checked=extra.get("recorded_hashes_checked"))
        with open(out / "summary.json", "x", encoding="utf-8") as f:
            json.dump({"meta": meta, "models": res}, f, indent=1)
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="completed", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    except BaseException as e:
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="failed", time=time.strftime("%Y-%m-%d %H:%M:%S"), error=f"{type(e).__name__}: {e}",
                                          trace=traceback.format_exc()[-2000:]))
        raise


if __name__ == "__main__":
    main()
