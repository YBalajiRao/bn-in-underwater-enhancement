"""Stage 4 inference on the combined external HICRD evaluation cohort (2,000 pairs): the 12 matched fp32 checkpoints (MARINE-Net / AT-Trunk x BN / IN x
seeds 0-2). Per-image PSNR and SSIM with the frozen evaluator's score_model (clamp, TF32 off); no LPIPS.
Geometry (protocol A01 A8):
  square256  : PRIMARY. train_v2.preload unchanged = torchvision Resize((256,256)) BILINEAR antialias on PIL RGB, input and reference alike.
  aspect482  : sensitivity G. PIL resize((482, 256), BILINEAR) (antialiased), input and reference alike; best_valA only.
Order of checks before any HICRD pixel is decoded: the three protocol documents' SHA-256; Val-B-free manifest table; all 12 checkpoint hashes;
manifest SHA-256, pair count 2,000, every path under datasets/source_copies/HICRD and outside LSUI/UIEB, every recorded per-file SHA-256.
--dry-run stops after these checks (no pixel decoded, no model loaded on GPU, nothing written except nothing).
Ledger: 'started' and 'completed'/'failed' events in paper1_uieb_runs/ledger (dataset label 'hicrd_stage4'). Output folder created exclusively.
Usage: python v2_tools/stage4_infer.py --checkpoint best_valA|last --geometry square256|aspect482 --label L --out DIR [--dry-run]"""
import argparse, csv, hashlib, json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import paper1_final_eval_v2 as V  # noqa: E402

E = V.E
MODELS12 = [f"{a}_s{s}" for a in ("marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in") for s in (0, 1, 2)]
MANIFEST = ROOT / "paper1_uieb_runs/stage4_prep/PROPOSED_hicrd_stage4_2000_manifest.json"
MANIFEST_SHA = "29402a7f706dc926effa4a2e05949b93cef664086c40439516d7b6f1eef5621d"
PROTOCOL = {"reports/paper1_narrowed/S4_preregistration_FROZEN_DRAFT.md": "2ea1f6ed3f8b20ff889f79c27f0dfaa795e2c2a7d9d66257b92236751b4a4934",
            "reports/paper1_narrowed/S4_protocol_amendment_01.md": "8301b73d10ddf574b25b447e686314b6f0905eeb5d7bbe80bf42e872bff8bad3",
            "reports/paper1_narrowed/S4_protocol_amendment_02.md": "0b0753dcf62182a7b472764cf993dec830ddc0b9afb249d5e24c11f0fa2743f2"}
HICRD_ROOT = (ROOT / "datasets/source_copies/HICRD").resolve()
FORBIDDEN = [(ROOT / "datasets/source_copies/LSUI").resolve(), (ROOT / "datasets/source_copies/UIEB").resolve()]
G_SIZE = (482, 256)  # (W, H)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_protocol(root=ROOT):
    for rel, exp in PROTOCOL.items():
        got = sha256(Path(root) / rel)
        if got != exp:
            raise SystemExit(f"STOP: protocol file {rel} sha256 {got} != frozen {exp}")


def load_manifest(path=MANIFEST, expected_sha=MANIFEST_SHA, verify_files=True):
    if sha256(path) != expected_sha:
        raise SystemExit("STOP: HICRD manifest hash differs")
    m = json.load(open(path, encoding="utf-8"))["pairs"]
    if len(m) != 2000 or len({p["id"] for p in m}) != 2000:
        raise SystemExit("STOP: manifest must hold 2,000 unique pairs")
    out = []
    for p in m:
        q = dict(p)
        for k in ("input_path", "gt_path"):
            a = (ROOT / p[k]).resolve()
            if HICRD_ROOT not in a.parents or any(f == a or f in a.parents for f in FORBIDDEN):
                raise SystemExit(f"STOP: path outside HICRD or inside LSUI/UIEB: {a}")
            q[k] = str(a)
        if verify_files:
            for k in ("input", "gt"):
                if sha256(q[f"{k}_path"]) != p[f"{k}_sha256"]:
                    raise SystemExit(f"STOP: per-file hash mismatch {p['id']} {k}")
        out.append(q)
    return out


def preload_g(pairs, size=G_SIZE):
    """Sensitivity G: aspect-preserving PIL BILINEAR (antialiased) resize to (W,H)=size for input and reference; uint8 [N,3,H,W]."""
    import torch
    from PIL import Image
    W, H = size
    X = torch.empty(len(pairs), 3, H, W, dtype=torch.uint8); Y = torch.empty_like(X)
    for i, p in enumerate(pairs):
        for dst, path in ((X, p["input_path"]), (Y, p["gt_path"])):
            im = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
            dst[i] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    return X, Y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, choices=sorted(V.CKPT_FILES)); ap.add_argument("--geometry", required=True, choices=["square256", "aspect482"])
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.geometry == "aspect482" and a.checkpoint != "best_valA":
        raise SystemExit("STOP: geometry sensitivity G is registered for best_valA only")
    check_protocol()
    import torch
    import train_v2 as tv
    E.restrict_manifests(tv.MANIFESTS)  # Val-B and every split except val_A removed before anything is read
    ckpts = {k: V.verify_checkpoint2(V.MODELS[k][0], a.checkpoint) for k in MODELS12}
    pairs = load_manifest()
    if a.dry_run:
        print(f"DRY RUN OK: protocol hashes, 12 checkpoint hashes ({a.checkpoint}), manifest {MANIFEST_SHA[:12]}, 2,000 pairs, 4,000 file hashes verified; nothing decoded or written")
        return {"dry_run": True, "n_pairs": len(pairs)}
    out = E.exclusive_dir(ROOT / a.out)
    rec = {"dataset": "hicrd_stage4", "label": a.label, "out": str(out), "checkpoint": a.checkpoint, "geometry": a.geometry, "manifest_sha256": MANIFEST_SHA,
           "protocol_sha256": PROTOCOL, "script_sha256": sha256(__file__), "models": MODELS12}
    V.ledger_write(V.LEDGER_DIR, dict(rec, event="started", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        X, Y = (tv.preload(pairs) if a.geometry == "square256" else preload_g(pairs))
        H, W = (256, 256) if a.geometry == "square256" else (G_SIZE[1], G_SIZE[0])
        assert X.shape == Y.shape == (len(pairs), 3, H, W) and X.dtype == Y.dtype == torch.uint8
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
        with open(out / "per_image.csv", "x", newline="") as f:
            w = csv.writer(f); w.writerow(["id"] + [f"{k}_{m}" for k in MODELS12 for m in ("psnr", "ssim")])
            for i, p in enumerate(pairs):
                w.writerow([p["id"]] + [f"{per[k][j][i]:.6f}" for k in MODELS12 for j in (0, 1)])
        meta = dict(rec, n_images=len(pairs), created=time.strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
                    tensor_hw=[H, W])
        with open(out / "summary.json", "x", encoding="utf-8") as f:
            json.dump({"meta": meta, "models": res}, f, indent=1)
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="completed", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    except BaseException as e:
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="failed", time=time.strftime("%Y-%m-%d %H:%M:%S"), error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]))
        raise


if __name__ == "__main__":
    main()
