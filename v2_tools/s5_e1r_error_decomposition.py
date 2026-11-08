"""S5-E1r (EXPLORATORY): exact per-image decomposition of the evaluation MSE of the 12 best_valA fp32 checkpoints on UIEB-787 and HICRD-2000.
Frozen geometry: train_v2.preload (Resize 256x256, BILINEAR antialias, input and reference alike); output = clamp(model(x)['enhanced'], 0, 1); TF32 off.
Decomposition (float64, per image, per model; e = yhat - y over P pixels, channels c = R,G,B; Var = population variance over pixels):
  e_bar_c = mean_p e_c ;  m = mean_c e_bar_c
  A = m^2                                   global achromatic (brightness) offset
  C = mean_c (e_bar_c - m)^2                global chromatic offset
  yc = y_c - mean(y_c), pc = yhat_c - mean(yhat_c), v_c = mean(yc^2)
  a_c = mean(pc*yc) / v_c  (OLS slope; if v_c <= EPS_CONST the channel is treated as constant: a_c := 1, so its gain term is 0 and pc goes to R)
  G = mean_c (a_c - 1)^2 v_c                global per-channel gain / contrast error
  r_c = pc - a_c*yc ;  R = mean_c mean(r_c^2)   spatial residual
  exact: MSE = A + C + G + R (verified per image: |A+C+G+R-MSE| <= 1e-10*MSE + 1e-18, else STOP)
  R split exactly by Parseval into R_low (radial frequency <= 1/16 cycles/pixel, DC excluded since r_c has zero mean) and R_high.
Also recorded: per-image PSNR from this MSE (float64), compared with the frozen per-image PSNR of Stage 3 / Stage 4 (float32 path) as an integrity check.
Order of checks before any pixel is decoded: Val-B removed from the manifest table; 12 checkpoint hashes; UIEB S3 manifest hash + per-file hashes (via
load_uieb_manifest2) and the frozen exclusion list; HICRD S4 manifest hash + per-file hashes. Checkpoint files are re-hashed after the run (must be unchanged).
Usage: python v2_tools/s5_e1r_error_decomposition.py --out DIR [--dry-run]"""
import argparse, csv, hashlib, json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))

MODELS12 = [f"{a}_s{s}" for a in ("marinenet_bn", "marinenet_in", "attrunk_bn", "attrunk_in") for s in (0, 1, 2)]
EPS_CONST = 1e-12
TOL_REL, TOL_ABS = 1e-10, 1e-18
F_CUT = 1 / 16  # cycles per pixel
UIEB_MANIFEST = "paper1_uieb_runs/stage3_prep/PROPOSED_uieb_stage3_remaining800_manifest.json"
UIEB_MANIFEST_SHA = "dcab4ac2589586dec9eb90d25ec2bb1390c368b46fbd220df3ca0a87a46fdf08"
UIEB_EXCL = "paper1_uieb_runs/stage3_prep/PROPOSED_primary_exclusions_and_sensitivity_sets.json"
UIEB_EXCL_SHA = "d80aa72a39abf0fb18ad4d6ac066a59b166c789430a7b5112ee61a600c0c503c"


def decompose(yhat, y):
    """torch or numpy arrays [N,3,H,W] in [0,1] -> dict of float64 arrays [N] (MSE, A, C, G, R, R_low, R_high, n_const_channels)."""
    import torch
    yh = torch.as_tensor(yhat).double(); yy = torch.as_tensor(y).double()
    N, Cn, H, W = yy.shape
    e = yh - yy
    mse = (e * e).mean(dim=(1, 2, 3))
    ebar = e.mean(dim=(2, 3))                      # [N,3]
    m = ebar.mean(1)
    A = m * m
    Cc = ((ebar - m[:, None]) ** 2).mean(1)
    yc = yy - yy.mean(dim=(2, 3), keepdim=True); pc = yh - yh.mean(dim=(2, 3), keepdim=True)
    v = (yc * yc).mean(dim=(2, 3))                # [N,3]
    const = v <= EPS_CONST
    a = torch.where(const, torch.ones_like(v), (pc * yc).mean(dim=(2, 3)) / torch.where(const, torch.ones_like(v), v))
    G = (((a - 1) ** 2) * torch.where(const, torch.zeros_like(v), v)).mean(1)
    r = pc - a[:, :, None, None] * yc
    R = (r * r).mean(dim=(1, 2, 3))
    Fr = torch.fft.fft2(r)                          # Parseval: mean(r^2) = sum |F|^2 / (H*W)^2
    fy = torch.fft.fftfreq(H, device=r.device, dtype=torch.float64)[:, None]; fx = torch.fft.fftfreq(W, device=r.device, dtype=torch.float64)[None, :]
    low = (torch.sqrt(fy ** 2 + fx ** 2) <= F_CUT)
    P2 = (Fr.real ** 2 + Fr.imag ** 2) / float(H * W) ** 2
    R_low = (P2 * low).sum(dim=(2, 3)).mean(1); R_high = (P2 * (~low)).sum(dim=(2, 3)).mean(1)
    out = {"MSE": mse, "A": A, "C": Cc, "G": G, "R": R, "R_low": R_low, "R_high": R_high, "n_const_channels": const.sum(1).double()}
    return {k: v.cpu().numpy() for k, v in out.items()}


def check_exact(d):
    tot = d["A"] + d["C"] + d["G"] + d["R"]
    err = np.abs(tot - d["MSE"]); lim = TOL_REL * d["MSE"] + TOL_ABS
    err2 = np.abs(d["R_low"] + d["R_high"] - d["R"]); lim2 = TOL_REL * d["R"] + TOL_ABS
    return bool(np.all(err <= lim) and np.all(err2 <= lim2)), float(np.max(err / np.maximum(d["MSE"], 1e-300))), float(np.max(err2))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_cohorts():
    import paper1_final_eval_v2 as V
    import stage4_infer as S4
    if sha(ROOT / UIEB_EXCL) != UIEB_EXCL_SHA:
        raise SystemExit("STOP: UIEB exclusion file hash differs")
    excl = set(json.load(open(ROOT / UIEB_EXCL))["primary_exclusions"])
    pu, mh, _ = V.load_uieb_manifest2(str(ROOT / UIEB_MANIFEST), UIEB_MANIFEST_SHA, 800, "UIEB", "stem")
    pu = [dict(p, key=Path(p["input_path"]).name) for p in pu if Path(p["input_path"]).name not in excl]
    if len(pu) != 787:
        raise SystemExit(f"STOP: UIEB primary cohort has {len(pu)} != 787")
    ph = [dict(p, key=p["id"]) for p in S4.load_manifest()]
    return {"UIEB": pu, "HICRD": ph}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    import torch
    import train_v2 as tv
    import paper1_final_eval_v2 as V
    E = V.E
    E.restrict_manifests(tv.MANIFESTS)
    if sorted(tv.MANIFESTS) != ["val_A"]:
        raise SystemExit("STOP: manifest table not restricted")
    ckpts = {k: V.verify_checkpoint2(V.MODELS[k][0], "best_valA") for k in MODELS12}
    cohorts = load_cohorts()
    if a.dry_run:
        print(f"DRY RUN OK: 12 checkpoint hashes; UIEB {len(cohorts['UIEB'])} pairs; HICRD {len(cohorts['HICRD'])} pairs; all file hashes verified; nothing decoded or written")
        return {"dry_run": True}
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    out.mkdir(parents=True)
    rec = {"dataset": "s5_e1r", "out": str(out), "checkpoint": "best_valA", "script_sha256": sha(__file__), "models": MODELS12}
    V.ledger_write(V.LEDGER_DIR, dict(rec, event="started", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    try:
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        dev = torch.device("cuda")
        rows, checks, t0 = [], {}, time.time()
        for ds, pairs in cohorts.items():
            X, Y = tv.preload(pairs)
            assert X.shape == Y.shape == (len(pairs), 3, 256, 256) and X.dtype == torch.uint8
            for k in MODELS12:
                m = V.load_model2(k, ckpts[k][0], "best_valA", dev)
                parts = []
                with torch.no_grad():
                    for i in range(0, len(X), 16):
                        x = X[i:i + 16].to(dev).float() / 255.0; y = Y[i:i + 16].to(dev).float() / 255.0
                        raw = m(x)["enhanced"]
                        if not torch.isfinite(raw).all():
                            raise SystemExit(f"STOP: non-finite output {ds} {k}")
                        parts.append(decompose(torch.clamp(raw, 0.0, 1.0), y))
                d = {kk: np.concatenate([p[kk] for p in parts]) for kk in parts[0]}
                ok, rel, err2 = check_exact(d)
                checks[f"{ds}|{k}"] = {"exact": ok, "max_rel_err": rel, "max_abs_err_Rsplit": err2, "n_const_channels": int(d["n_const_channels"].sum())}
                if not ok:
                    raise SystemExit(f"STOP: decomposition not exact for {ds} {k} (max rel {rel})")
                for i, p in enumerate(pairs):
                    rows.append([ds, p["key"], k] + [f"{d[kk][i]:.17g}" for kk in ("MSE", "A", "C", "G", "R", "R_low", "R_high")] + [int(d["n_const_channels"][i])])
                print(ds, k, f"mean MSE {d['MSE'].mean():.6g}", flush=True)
                del m; torch.cuda.empty_cache()
            del X, Y
        with open(out / "e1r_per_image.csv", "x", newline="") as f:
            w = csv.writer(f); w.writerow(["dataset", "key", "model", "MSE", "A", "C", "G", "R", "R_low", "R_high", "n_const_channels"]); w.writerows(rows)
        after = {k: sha(ckpts[k][0]) == ckpts[k][1] for k in MODELS12}
        if not all(after.values()):
            raise SystemExit("STOP: a checkpoint file changed during the run")
        json.dump({"meta": dict(rec, seconds=round(time.time() - t0, 1), torch=torch.__version__, gpu=torch.cuda.get_device_name(0), checkpoints_unchanged_after_run=True,
                                n_uieb=len(cohorts["UIEB"]), n_hicrd=len(cohorts["HICRD"])), "exactness": checks},
                  open(out / "e1r_summary.json", "x"), indent=1)
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="completed", time=time.strftime("%Y-%m-%d %H:%M:%S")))
    except BaseException as e:
        V.ledger_write(V.LEDGER_DIR, dict(rec, event="failed", time=time.strftime("%Y-%m-%d %H:%M:%S"), error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]))
        raise


if __name__ == "__main__":
    main()
