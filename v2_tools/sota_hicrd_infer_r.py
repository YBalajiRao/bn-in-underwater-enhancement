"""Group R inference (released checkpoints + classical methods) for the SOTA HICRD-2000 comparison.
Binding spec: reports/paper1_narrowed/SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md (hash-checked at start). PREPARED; requires separate approval to run.
Run inside D:\\MARINE_NET_RESEARCH_sota_assets\\venv_sota (FUnIE-GAN needs TensorFlow there; the PyTorch methods use the same environment).
One invocation per geometry processes every applicable Group R method and writes ONE exclusive folder:
  square256: U-Shape, WWE-UIE, FUnIE-GAN, NU2Net, CLAHE, UDCP  (+ U-Shape official min-max sensitivity, written to --out-ushape-minmax)
  aspect482: CLAHE, UDCP only (U-Shape, FUnIE-GAN fixed 256; WWE-UIE needs /4; NU2Net needs /16: 'not applicable', recorded)
Per method: per_image_<method>.csv (frozen PSNR/SSIM on float32 + exact E1r components), float outputs of the 12 registered visual images (.npy),
checks in summary.json. WaterNet (original weights) is EXCLUDED (Amendment 02 B1) and Group T models are refused.
Usage (real, after approval):
  venv_sota\\Scripts\\python.exe v2_tools/sota_hicrd_infer_r.py --geometry square256 --out paper1_uieb_runs/sota_hicrd_released_square256 \\
        --out-ushape-minmax paper1_uieb_runs/sota_hicrd_released_square256_ushape_minmax
  venv_sota\\Scripts\\python.exe v2_tools/sota_hicrd_infer_r.py --geometry aspect482 --out paper1_uieb_runs/sota_hicrd_released_G482
Add --dry-run to verify everything and write nothing. --test-manifest (synthetic images only) is for the test suite."""
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # Amendment 02 B7; must be set before TensorFlow is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, json, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402

ASSETS = P.ASSETS
REPOS = {"ushape": ("repos/U-shape_Transformer_for_Underwater_Image_Enhancement", "41f9897669d2e6f6718e00772ece2523d0f149a2"),
         "nu2net": ("repos/UnderwaterRanker", "f9bf3508121c5bd658a2f2c3fa8d4d3826ed1ce0"),
         "funie_gan": ("repos/FUnIE-GAN", "8f934c834c94e007b00866186b9ee624dc2b7b69")}
WWE_COMMIT = "fd0697558203f9f849800cc13b804116bb3f921c"
METHODS_BY_GEOMETRY = {"square256": ["ushape", "wwe_uie", "funie_gan", "nu2net", "clahe", "udcp"], "aspect482": ["clahe", "udcp"]}
NOT_APPLICABLE_G482 = {"ushape": "fixed 256x256 input", "funie_gan": "Keras input fixed (256,256,3)", "wwe_uie": "H, W must be divisible by 4",
                       "nu2net": "H, W must be multiples of 16"}


def git_head(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def verify_sources(methods):
    for m in methods:
        if m in P.GROUP_R_EXCLUDED or m in P.GROUP_T or m not in (set(P.GROUP_R) | set(P.CLASSICAL)):
            raise SystemExit(f"STOP: {m} is not an active Group R method")
    hashes = P.verify_group_r()
    for m in methods:
        if m in P.GROUP_R and not hashes[m]:
            raise SystemExit(f"STOP: checkpoint hash mismatch for {m}")
        if m in REPOS and git_head(ASSETS / REPOS[m][0]) != REPOS[m][1]:
            raise SystemExit(f"STOP: {m} code repository is not at the pinned commit")
    if "wwe_uie" in methods and git_head(ROOT / "external/WWE-UIE") != WWE_COMMIT:
        raise SystemExit("STOP: WWE-UIE code is not at the pinned commit")


def safe_load(p):
    import torch
    from numpy._core.multiarray import scalar
    with torch.serialization.safe_globals([(scalar, "numpy.core.multiarray.scalar"), np.dtype, (type(np.dtype("float64")), "numpy.dtypes.Float64DType")]):
        return torch.load(p, map_location="cpu", weights_only=True)


def build_torch(method, dev):
    """Returns (model, forward(x) -> raw output on the [0,1] scale before clamping)."""
    if method == "ushape":
        repo = ASSETS / REPOS["ushape"][0]; sys.path.insert(0, str(repo)); sys.dont_write_bytecode = True
        from net.Ushape_Trans import Generator
        m = Generator(); m.load_state_dict(safe_load(ASSETS / P.GROUP_R["ushape"]["file"]), strict=True)
        return m.to(dev).eval(), lambda x: m(x)[3]
    if method == "nu2net":
        repo = ASSETS / REPOS["nu2net"][0]; sys.path.insert(0, str(repo)); sys.dont_write_bytecode = True
        from all_model.NU2Net.model import NU2Net
        m = NU2Net(basic_channel=64); m.load_state_dict(safe_load(ASSETS / P.GROUP_R["nu2net"]["file"])["net"], strict=True)
        return m.to(dev).eval(), lambda x: m(x)
    if method == "wwe_uie":
        from models.wwe_uie import WWEUIE
        m = WWEUIE(); m.net.load_state_dict(safe_load(ASSETS / P.GROUP_R["wwe_uie"]["file"]), strict=True)
        return m.to(dev).eval(), lambda x: m(x)["enhanced"]
    raise SystemExit(f"STOP: no torch builder for {method}")


def build_funie():
    import tf_keras
    m = tf_keras.models.model_from_json(open(ASSETS / P.GROUP_R["funie_gan"]["json"]).read())
    m.load_weights(str(ASSETS / P.GROUP_R["funie_gan"]["file"]))
    return m


def run_method(method, X, dev, forward=None):
    """X: uint8 [N,3,H,W]. Returns (primary float32 [N,3,H,W] in [0,1], extra dict). STOP on non-finite raw output.
    `forward` (tests only) replaces the model forward for torch methods."""
    import torch
    import sota_classical as SC
    N = len(X); extra = {}
    if method == "clahe":
        out = np.stack([SC.clahe(np.ascontiguousarray(X[i].permute(1, 2, 0).numpy())) for i in range(N)]).astype(np.float32) / 255.0
        return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous(), extra
    if method == "udcp":
        out = np.stack([SC.udcp(X[i].permute(1, 2, 0).numpy().astype(np.float64) / 255.0, r=P.UDCP_GF_RADIUS) for i in range(N)])
        K.check_finite(out, method)
        return torch.from_numpy(out).permute(0, 3, 1, 2).to(torch.float32).contiguous(), extra
    if method == "funie_gan":
        m = build_funie()
        x0 = X[:1].permute(0, 2, 3, 1).numpy().astype(np.float32) / 127.5 - 1.0
        if not np.array_equal(m.predict(x0, verbose=0), m.predict(x0, verbose=0)):
            raise SystemExit("STOP: FUnIE-GAN first batch not deterministic")
        extra["determinism_first_batch"] = "identical"
        outs = []
        for i in range(N):
            y = m.predict(X[i:i + 1].permute(0, 2, 3, 1).numpy().astype(np.float32) / 127.5 - 1.0, verbose=0)
            K.check_finite(y, method)
            outs.append(np.clip((y + 1.0) / 2.0, 0.0, 1.0))
        return torch.from_numpy(np.concatenate(outs)).permute(0, 3, 1, 2).to(torch.float32).contiguous(), extra
    model, fwd = (None, forward) if forward is not None else build_torch(method, dev)
    H, W = X.shape[2:]
    if forward is None:
        extra["memory"] = K.memory_guard(fwd, (1, 3, H, W), dev, method)
    K.set_tf32_off()
    prim, mm = [], []
    with torch.no_grad():
        for i in range(N):
            raw = fwd(X[i:i + 1].to(dev).float() / 255.0)
            K.check_finite(raw, method)
            prim.append(torch.clamp(raw, 0.0, 1.0).float().cpu())
            if method == "ushape":
                mm.append(P.map_ushape_minmax(raw.float().cpu()))
    if mm:
        extra["ushape_minmax"] = torch.cat(mm)
    return torch.cat(prim), extra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, choices=sorted(METHODS_BY_GEOMETRY)); ap.add_argument("--out", required=True)
    ap.add_argument("--out-ushape-minmax", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-manifest", default=None); ap.add_argument("--methods", default=None, help="test mode only")
    a = ap.parse_args(argv)
    test = a.test_manifest is not None
    if a.methods and not test:
        raise SystemExit("STOP: --methods is only allowed in test mode (a real run processes every applicable method)")
    methods = a.methods.split(",") if a.methods else METHODS_BY_GEOMETRY[a.geometry]
    if a.geometry == "aspect482" and any(m in NOT_APPLICABLE_G482 for m in methods):
        raise SystemExit("STOP: a requested method is not applicable at aspect482")
    needs_mm = "ushape" in methods and a.geometry == "square256"
    if needs_mm and not a.out_ushape_minmax:
        raise SystemExit("STOP: --out-ushape-minmax is required when U-Shape runs at square256")
    K.verify_protocol_files()
    verify_sources(methods)
    pairs = K.load_pairs(a.test_manifest)
    ids = [p["id"] for p in pairs]
    if not test:
        P.assert_aligned(ids)
    out = K.check_out(ROOT / a.out if not Path(a.out).is_absolute() else a.out, test)
    out_mm = K.check_out(ROOT / a.out_ushape_minmax if not Path(a.out_ushape_minmax).is_absolute() else a.out_ushape_minmax, test) if needs_mm else None
    vis = P.visual_selection() if not test else ids[:2]
    plan = {"geometry": a.geometry, "methods": methods, "not_applicable": ({k: v for k, v in NOT_APPLICABLE_G482.items()} if a.geometry == "aspect482" else {}),
            "n_pairs": len(pairs), "out": str(out), "out_ushape_minmax": str(out_mm) if out_mm else None, "visual_ids": vis, "test_mode": test}
    if a.dry_run:
        print("DRY RUN OK:", json.dumps(plan)); return {"dry_run": True, "plan": plan}
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = {"dataset": "sota_hicrd_R", "out": str(out), "geometry": a.geometry, "methods": methods, "script_sha256": K.sha(__file__)}
    out.mkdir(parents=True)
    if out_mm:
        out_mm.mkdir(parents=True)
    K.ledger("started", rec, test)
    try:
        t0 = time.time()
        X, Y = K.preload(pairs, a.geometry)
        assert X.dtype == torch.uint8 and X.shape == Y.shape
        Yf = Y.float() / 255.0
        vis_idx = [ids.index(k) for k in vis]
        summary = {"meta": dict(plan, protocol_sha256=K.PROTOCOL_V2[1], torch=torch.__version__, device=str(dev),
                                tf_enable_onednn_opts=os.environ.get("TF_ENABLE_ONEDNN_OPTS"), udcp_gf_radius=P.UDCP_GF_RADIUS), "methods": {}}
        for m in methods:
            tm = time.time()
            prim, extra = run_method(m, X, dev)
            ps, ss, d = K.score(prim, Yf)
            chk = K.check_scores(ps, d, m)
            K.write_per_image(out / f"per_image_{m}.csv", ids, ps, ss, d, 0)
            np.save(out / f"visual_{m}.npy", prim[vis_idx].numpy())
            summary["methods"][m] = dict(chk, n=len(ids), mean_psnr=float(ps.mean()), seconds=round(time.time() - tm, 1),
                                         **{k: v for k, v in extra.items() if k != "ushape_minmax"})
            if "ushape_minmax" in extra:
                ps2, ss2, d2 = K.score(extra["ushape_minmax"], Yf)
                chk2 = K.check_scores(ps2, d2, "ushape_minmax")
                K.write_per_image(out_mm / "per_image_ushape_minmax.csv", ids, ps2, ss2, d2, 0)
                json.dump({"meta": dict(plan, source="same U-Shape forward pass; official per-image min-max"), "check": chk2}, open(out_mm / "summary.json", "x"), indent=1)
            print(m, f"mean PSNR {ps.mean():.4f}", flush=True)
        summary["meta"]["seconds"] = round(time.time() - t0, 1)
        json.dump(summary, open(out / "summary.json", "x"), indent=1, default=str)
        K.ledger("completed", rec, test)
        return summary
    except BaseException as e:
        K.ledger("failed", dict(rec, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]), test)
        raise


if __name__ == "__main__":
    main()
