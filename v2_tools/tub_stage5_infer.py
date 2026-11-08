"""TUB Stage 5 inference (EXPLORATORY; protocol reports/paper1_narrowed/TUB_stage5_preregistration_FROZEN_DRAFT.md, SHA-256 1e3382c0...).
12 best_valA (or last) fp32 checkpoints on all 1,183 valid TUB pairs (input Stage02-10, reference = Stage01 of the same scene AND camera). Per pair and model:
frozen PSNR / SSIM (train_v2.compute_psnr / compute_ssim on the clamped output, exactly as paper1_final_eval.score_model), and the exact E1r decomposition
(MSE, A, C, G, R, R_low, R_high via v2_tools/s5_e1r_error_decomposition.decompose, imported unchanged and hash-checked). Also the input luma SD at the
evaluation geometry (near-blank sensitivity) and the per-channel means of input and reference at the evaluation geometry (supplementary g_ach). Geometries: square256 = train_v2.preload (Resize 256x256 BILINEAR antialias, input and reference alike);
aspect455 = PIL BILINEAR to 455x256 (W x H), input and reference alike. Batch 16 (square256) / 4 (aspect455; Amendment 02, SHA-256 a1457407...), fp32, TF32 off, clamp [0,1].
Checks before any pixel is decoded: protocol hash; cohort manifest hashes and counts (1,183 / 1,153 / 1,022, nesting, no NTU-copied scene in the NTU cohort);
E1r script hash; TUB download-manifest hash; every input / reference file hash; Val-B removed; 12 checkpoint hashes. GPU-memory guard for every run (no-spill: peak allocated x 1.10 and peak reserved both within free dedicated memory): a probe
forward of each of the 4 architectures (seed 0) at the registered batch and shape must fit in the free device memory with a 10 % margin, else STOP
(batch, precision and preprocessing are never changed).
Checks during / after: non-finite outputs -> STOP; decomposition not exact -> STOP; |frozen PSNR - (-10 log10 MSE)| > 1e-4 dB -> STOP; checkpoints re-hashed.
Ledger: 'tub_stage5' started / completed / failed. The output folder is created exclusively.
Usage: python v2_tools/tub_stage5_infer.py --checkpoint best_valA|last --geometry square256|aspect455 --out DIR [--dry-run]"""
import argparse, csv, hashlib, json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))

MODELS12 = [f"{a}_s{s}" for a in ("marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in") for s in (0, 1, 2)]
TUB_ROOT = ROOT / "datasets/source_copies/TUB/hf_vapaau_TUB_rev6214f00807df"
PREP = ROOT / "paper1_uieb_runs/tub_stage5_prep"
HASHES = {
    "reports/paper1_narrowed/TUB_stage5_preregistration_FROZEN_DRAFT.md": "1e3382c0797de8031d9cf97ee4ca11e6ae731e76ea02ad56334baa0ae8dfa3d6",
    "paper1_uieb_runs/tub_stage5_prep/TUB_ALL_1183.json": "c4a5947a5e5f2dba240e6ede086300f834c338591aa2db2d0b23608005791ad4",
    "paper1_uieb_runs/tub_stage5_prep/TUB_PRIMARY_1153.json": "d0da780a68ddb7e4ed4d7c351776d43a1981c20a7dd4efed109f1072a654e2d1",
    "paper1_uieb_runs/tub_stage5_prep/TUB_NTU_1022.json": "45729d6190dabb04da08f498a5b9836b3f1d1704aa7c44fc7b8a180472cb832e",
    "v2_tools/s5_e1r_error_decomposition.py": "24ef8a6919f90d25dcbd7e318ad4491920b2fed16d2c9a8e4c5da8e3d61ff0c1",
    "datasets/source_copies/TUB/DOWNLOAD_MANIFEST.json": "2c1f63e5bb23b299005b889fe13cb2b526784a54a4d9c48d63173f83702fd63a",
}
NTU_COPIED_SCENES = {10, 15, 28, 33}
G455 = (455, 256)  # (W, H)
BATCH = {"square256": 16, "aspect455": 4}  # aspect455 batch 4: Amendment 02 (batch 16 and 8 fail the no-spill memory rule for MARINE-Net)
AMENDMENT02 = ("reports/paper1_narrowed/TUB_stage5_amendment02_G455_batch4.md", "a1457407e9f2d42b35f7f89c14502bef558aefdfb881f0d9443e29629ae3c680")
PSNR_TOL = 1e-4
PROBE_MODELS = ["marinenet_bn_s0", "marinenet_in_s0", "attrunk_bn_s0", "attrunk_in_s0"]
COLUMNS = ["pair", "scene", "cam", "stage", "model", "psnr", "ssim", "MSE", "A", "C", "G", "R", "R_low", "R_high", "input_luma_sd_eval",
           "mu_in_r", "mu_in_g", "mu_in_b", "mu_ref_r", "mu_ref_g", "mu_ref_b", "in_primary", "in_ntu"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_hashes(root=ROOT):
    for rel, exp in HASHES.items():
        if sha(Path(root) / rel) != exp:
            raise SystemExit(f"STOP: {rel} hash differs from frozen {exp}")


def pair_key(p):
    return f"Exp{p['scene']}_Cam{p['cam']}_Stage{p['stage']:02d}"


def load_cohorts(prep=PREP):
    allp = json.load(open(Path(prep) / "TUB_ALL_1183.json"))["pairs"]
    prim = json.load(open(Path(prep) / "TUB_PRIMARY_1153.json"))["pairs"]
    ntu = json.load(open(Path(prep) / "TUB_NTU_1022.json"))["pairs"]
    ka, kp, kn = ({pair_key(p) for p in x} for x in (allp, prim, ntu))
    if (len(allp), len(prim), len(ntu)) != (1183, 1153, 1022) or len(ka) != 1183:
        raise SystemExit("STOP: cohort counts differ from 1,183 / 1,153 / 1,022")
    if not (kn <= kp <= ka):
        raise SystemExit("STOP: cohorts not nested (NTU within primary within all)")
    if any(p["scene"] in NTU_COPIED_SCENES for p in ntu):
        raise SystemExit("STOP: an NTU-copied scene is in the NTU cohort")
    for p in allp:
        if p["stage"] < 2 or not p["reference"].endswith(f"Exp{p['scene']}_Cam{p['cam']}_Stage01.jpg") or not p["input"].endswith(f"{pair_key(p)}.jpg"):
            raise SystemExit(f"STOP: pairing rule violated for {pair_key(p)}")
    return allp, kp, kn


def verify_files(allp):
    for p in allp:
        for rel, h in ((p["input"], p["input_sha256"]), (p["reference"], p["reference_sha256"])):
            if sha(TUB_ROOT / rel) != h:
                raise SystemExit(f"STOP: file hash mismatch {rel}")


def preload_geometry(pairs, geometry):
    import torch
    import train_v2 as tv
    from PIL import Image
    if geometry == "square256":
        return tv.preload([{"input_path": str(TUB_ROOT / p["input"]), "gt_path": str(TUB_ROOT / p["reference"])} for p in pairs])
    W, H = G455
    X = torch.empty(len(pairs), 3, H, W, dtype=torch.uint8); Y = torch.empty_like(X)
    for i, p in enumerate(pairs):
        for dst, rel in ((X, p["input"]), (Y, p["reference"])):
            dst[i] = torch.from_numpy(np.asarray(Image.open(TUB_ROOT / rel).convert("RGB").resize((W, H), Image.BILINEAR))).permute(2, 0, 1)
    return X, Y


def memory_probe(model_fn, shape, dev):
    """peak allocated bytes of one no-grad score_batch (forward + frozen metrics + float64 decomposition) at the registered batch / shape on random tensors,
    and free device memory before it."""
    import torch
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    free, total = torch.cuda.mem_get_info(dev)
    m = model_fn()
    with torch.no_grad():
        score_batch(m, torch.rand(*shape, device=dev), torch.rand(*shape, device=dev))
    peak = torch.cuda.max_memory_allocated(dev); reserved = torch.cuda.max_memory_reserved(dev)
    del m; torch.cuda.empty_cache()
    return {"free_before_bytes": int(free), "total_bytes": int(total), "peak_allocated_bytes": int(peak), "peak_reserved_bytes": int(reserved)}


def score_batch(m, x, y):
    """frozen PSNR / SSIM per image (as score_model) + exact decomposition, on the same clamped output."""
    import torch
    import train_v2 as tv
    import s5_e1r_error_decomposition as D
    raw = m(x)["enhanced"]
    nonfinite = int((~torch.isfinite(raw)).sum())
    out = torch.clamp(raw, 0.0, 1.0)
    ps = np.array([tv.compute_psnr(out[j:j + 1], y[j:j + 1]) for j in range(len(x))])
    ss = np.array([tv.compute_ssim(out[j:j + 1], y[j:j + 1]) for j in range(len(x))])
    d = D.decompose(out, y)
    return ps, ss, d, nonfinite


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, choices=["best_valA", "last"]); ap.add_argument("--geometry", required=True, choices=["square256", "aspect455"])
    ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    verify_hashes()
    if sha(ROOT / AMENDMENT02[0]) != AMENDMENT02[1]:
        raise SystemExit("STOP: Amendment 02 hash differs from the frozen value")
    bs = BATCH[a.geometry]
    import torch
    import train_v2 as tv
    import paper1_final_eval_v2 as V
    import s5_e1r_error_decomposition as D
    V.E.restrict_manifests(tv.MANIFESTS)
    if sorted(tv.MANIFESTS) != ["val_A"]:
        raise SystemExit("STOP: manifest table not restricted")
    ckpts = {k: V.verify_checkpoint2(V.MODELS[k][0], a.checkpoint) for k in MODELS12}
    allp, kp, kn = load_cohorts()
    verify_files(allp)
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    if a.dry_run:
        print(f"DRY RUN OK: protocol / cohort / E1r / download-manifest hashes; 1,183 / 1,153 / 1,022 nested; pairing rule; 2,366 file hashes; "
              f"12 checkpoint hashes ({a.checkpoint}); {out} absent; nothing decoded or written")
        return {"dry_run": True}
    dev = torch.device("cuda")
    H, W = (256, 256) if a.geometry == "square256" else (G455[1], G455[0])
    probe = {k: memory_probe(lambda k=k: V.load_model2(k, ckpts[k][0], a.checkpoint, dev), (bs, 3, H, W), dev) for k in PROBE_MODELS}
    if any(v["peak_allocated_bytes"] * 1.10 > v["free_before_bytes"] or v["peak_reserved_bytes"] > v["free_before_bytes"] for v in probe.values()):  # no-spill rule
        raise SystemExit(f"STOP: insufficient GPU memory for batch {bs} at {H}x{W}: {probe}")
    out.mkdir(parents=True)
    rec = {"dataset": "tub_stage5", "out": str(out), "checkpoint": a.checkpoint, "geometry": a.geometry, "script_sha256": sha(__file__), "protocol_sha256": HASHES[next(iter(HASHES))], "models": MODELS12}
    V.ledger_write(V.LEDGER_DIR, dict(rec, event="started", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        t0 = time.time()
        X, Y = preload_geometry(allp, a.geometry)
        assert X.shape == Y.shape == (len(allp), 3, H, W) and X.dtype == torch.uint8
        xf = X.float() / 255.0
        luma_sd = (0.299 * xf[:, 0] + 0.587 * xf[:, 1] + 0.114 * xf[:, 2]).flatten(1).std(1, unbiased=False).numpy()
        mu_in = xf.mean(dim=(2, 3)).double().numpy(); mu_ref = (Y.double() / 255.0).mean(dim=(2, 3)).numpy(); del xf
        rows, checks, peaks = [], {}, {}
        for k in MODELS12:
            torch.cuda.reset_peak_memory_stats(dev)
            m = V.load_model2(k, ckpts[k][0], a.checkpoint, dev)
            parts = []
            with torch.no_grad():
                for i in range(0, len(X), bs):
                    x = X[i:i + bs].to(dev).float() / 255.0; y = Y[i:i + bs].to(dev).float() / 255.0
                    ps, ss, d, nf = score_batch(m, x, y)
                    if nf:
                        raise SystemExit(f"STOP: non-finite output for {k}")
                    parts.append((ps, ss, d))
            ps = np.concatenate([p[0] for p in parts]); ss = np.concatenate([p[1] for p in parts])
            d = {kk: np.concatenate([p[2][kk] for p in parts]) for kk in parts[0][2]}
            ok, rel, _ = D.check_exact(d)
            dpsnr = float(np.max(np.abs(ps - (-10 * np.log10(d["MSE"])))))
            checks[k] = {"exact": ok, "max_rel_err": rel, "max_abs_psnr_frozen_vs_mse_dB": dpsnr}
            if not ok:
                raise SystemExit(f"STOP: decomposition not exact for {k}")
            if dpsnr > PSNR_TOL:
                raise SystemExit(f"STOP: frozen PSNR vs decomposition PSNR differ by {dpsnr} dB for {k}")
            peaks[k] = int(torch.cuda.max_memory_allocated(dev))
            for i, p in enumerate(allp):
                rows.append([pair_key(p), p["scene"], p["cam"], p["stage"], k, f"{ps[i]:.6f}", f"{ss[i]:.6f}"] + [f"{d[kk][i]:.17g}" for kk in ("MSE", "A", "C", "G", "R", "R_low", "R_high")]
                            + [f"{luma_sd[i]:.6f}"] + [f"{v:.10f}" for v in (*mu_in[i], *mu_ref[i])] + [int(pair_key(p) in kp), int(pair_key(p) in kn)])
            print(k, f"mean PSNR {ps.mean():.4f}", flush=True)
            del m; torch.cuda.empty_cache()
        with open(out / "tub_per_image.csv", "x", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)
        unchanged = all(sha(ckpts[k][0]) == ckpts[k][1] for k in MODELS12)
        if not unchanged:
            raise SystemExit("STOP: a checkpoint file changed during the run")
        json.dump({"meta": dict(rec, seconds=round(time.time() - t0, 1), torch=torch.__version__, gpu=torch.cuda.get_device_name(0), tensor_hw=[H, W], batch=bs,
                                memory_probe=probe, peak_allocated_bytes_per_model=peaks, checkpoints_unchanged_after_run=True, n_pairs=len(allp)), "checks": checks},
                  open(out / "tub_summary.json", "x"), indent=1)
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="completed", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    except BaseException as e:
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="failed", time=time.strftime("%Y-%m-%d %H:%M:%S"), error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]))
        raise


if __name__ == "__main__":
    main()
