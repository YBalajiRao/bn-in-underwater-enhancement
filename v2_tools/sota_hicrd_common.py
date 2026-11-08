"""Shared execution helpers for the SOTA HICRD-2000 comparison (protocol: reports/paper1_narrowed/SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md).
PREPARED; nothing here runs inference by itself. Reuses frozen functions unchanged:
  stage4_infer.load_manifest (manifest + 4,000 image hashes), train_v2.preload (square256), stage4_infer.preload_g (G482),
  tools/eval_reproduce_baseline.compute_psnr / compute_ssim (via train_v2), s5_e1r_error_decomposition.decompose / check_exact.
Fail-fast checks: non-finite outputs, decomposition exactness, PSNR consistency (1e-4 dB), manifest alignment, GPU memory (no-spill rule, TF32 off
before the probe), protocol / checkpoint hashes, exclusive output folders."""
import csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))

PROTOCOL_V2 = ("reports/paper1_narrowed/SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md", "2d29442248a57ac6a164c7e2e3e09420e781a972fdde95b9466748b417da1000")
AMENDMENT_02 = ("reports/paper1_narrowed/SOTA_HICRD_comparison_protocol_AMENDMENT_02.md", "23cf50fb4e22e92abad9434b78536fead46aa821d79ffe639cd166e92825fe69")
E1R = ("v2_tools/s5_e1r_error_decomposition.py", "24ef8a6919f90d25dcbd7e318ad4491920b2fed16d2c9a8e4c5da8e3d61ff0c1")
METRICS = ("tools/eval_reproduce_baseline.py", "6e861f53dbf74aaa")  # prefix as recorded in the protocol
PSNR_TOL = 1e-4
FORBIDDEN_OUT_PREFIX = ROOT / "paper1_uieb_runs"
COLUMNS = ["id", "psnr", "ssim", "MSE", "A", "C", "G", "R", "R_low", "R_high", "nonfinite_raw"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_protocol_files():
    for rel, h in (PROTOCOL_V2, AMENDMENT_02, E1R):
        if sha(ROOT / rel) != h:
            raise SystemExit(f"STOP: {rel} hash differs from the frozen value")
    if not sha(ROOT / METRICS[0]).startswith(METRICS[1]):
        raise SystemExit("STOP: frozen metric file hash differs")


def check_out(out, test_mode):
    out = Path(out).resolve()
    if out.exists():
        raise SystemExit(f"STOP: {out} exists (never overwrite)")
    if test_mode and (FORBIDDEN_OUT_PREFIX.resolve() in out.parents):
        raise SystemExit("STOP: test mode may not write under paper1_uieb_runs")
    return out


def load_pairs(test_manifest=None):
    """Real run: the frozen HICRD manifest via stage4_infer.load_manifest (hash-checks all 4,000 files). Test mode: a synthetic manifest
    {'pairs': [{'id','input_path','gt_path'}]} (absolute paths), never HICRD."""
    if test_manifest is None:
        import stage4_infer as S4I
        return S4I.load_manifest()
    m = json.load(open(test_manifest))["pairs"]
    if any("HICRD" in p["input_path"] or "HICRD" in p["gt_path"] for p in m):
        raise SystemExit("STOP: test manifest must not reference HICRD")
    return m


def preload(pairs, geometry):
    import train_v2 as tv
    import stage4_infer as S4I
    if geometry == "square256":
        return tv.preload(pairs)
    if geometry == "aspect482":
        return S4I.preload_g(pairs)
    raise SystemExit(f"STOP: unknown geometry {geometry}")


def to_metric_tensor(x):
    """float32 torch [N,3,H,W]."""
    import torch
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.to(torch.float32)


def score(out01, ref01):
    """out01, ref01: [N,3,H,W] in [0,1] (any float dtype). Frozen PSNR / SSIM per image on float32 tensors; exact E1r decomposition (float64).
    Returns (psnr[N], ssim[N], decomposition dict)."""
    import torch
    import train_v2 as tv
    import s5_e1r_error_decomposition as D
    o = to_metric_tensor(out01).cpu(); r = to_metric_tensor(ref01).cpu()
    ps = np.array([tv.compute_psnr(o[j:j + 1], r[j:j + 1]) for j in range(len(o))])
    ss = np.array([tv.compute_ssim(o[j:j + 1], r[j:j + 1]) for j in range(len(o))])
    d = D.decompose(o.double(), r.double())
    return ps, ss, d


def check_scores(ps, d, label):
    """STOP on inexact decomposition or PSNR inconsistency."""
    import s5_e1r_error_decomposition as D
    ok, rel, _ = D.check_exact(d)
    if not ok:
        raise SystemExit(f"STOP: decomposition not exact for {label} (max rel {rel})")
    nz = d["MSE"] > 1e-10
    dps = float(np.max(np.abs(ps[nz] + 10 * np.log10(d["MSE"][nz])))) if nz.any() else 0.0
    if dps > PSNR_TOL:
        raise SystemExit(f"STOP: PSNR inconsistency {dps} dB for {label}")
    return {"exact": True, "max_rel_err": rel, "max_abs_psnr_vs_mse_dB": dps}


def check_finite(raw, label):
    a = np.asarray(raw.detach().cpu() if hasattr(raw, "detach") else raw)
    n = int((~np.isfinite(a)).sum())
    if n:
        raise SystemExit(f"STOP: {n} non-finite output values for {label}")
    return 0


def set_tf32_off():
    import torch
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False


def memory_guard(forward, shape, dev, label, mem_info=None):
    """TF32 off FIRST, then one no-grad forward at the registered batch/shape on random input. No-spill rule:
    peak_allocated * 1.10 <= free and peak_reserved <= free (free = dedicated memory before the probe). mem_info is injectable for tests."""
    import torch
    set_tf32_off()
    if dev.type != "cuda":
        return {"device": "cpu", "checked": False}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    free, total = (mem_info or torch.cuda.mem_get_info)(dev)
    with torch.no_grad():
        forward(torch.rand(*shape, device=dev))
    pa, pr = torch.cuda.max_memory_allocated(dev), torch.cuda.max_memory_reserved(dev)
    rep = {"device": str(dev), "free_bytes": int(free), "total_bytes": int(total), "peak_allocated_bytes": int(pa), "peak_reserved_bytes": int(pr)}
    if pa * 1.10 > free or pr > free:
        raise SystemExit(f"STOP: no-spill rule violated for {label}: {rep}")
    torch.cuda.empty_cache()
    return rep


def write_per_image(path, ids, ps, ss, d, nonfinite):
    with open(path, "x", newline="") as f:
        w = csv.writer(f); w.writerow(COLUMNS)
        for i, k in enumerate(ids):
            w.writerow([k, f"{ps[i]:.6f}", f"{ss[i]:.6f}"] + [f"{d[c][i]:.17g}" for c in ("MSE", "A", "C", "G", "R", "R_low", "R_high")] + [int(nonfinite)])


def read_per_image(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if list(rows[0].keys()) != COLUMNS:
        raise SystemExit(f"STOP: unexpected schema in {path}")
    return rows


def ledger(event, rec, test_mode):
    if test_mode:
        return
    import paper1_final_eval_v2 as V
    V.ledger_write(V.LEDGER_DIR, dict(rec, event=event, time=time.strftime("%Y-%m-%d %H:%M:%S")))
