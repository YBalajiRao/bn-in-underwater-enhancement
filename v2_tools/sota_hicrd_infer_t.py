"""Group T inference (our LSUI retrains, EXPLORATORY n = 1 baselines) for the SOTA HICRD-2000 comparison.
Binding spec: SOTA_HICRD_comparison_protocol_EFFECTIVE_v2.md (hash-checked). PREPARED; requires separate approval to run.
Uses the frozen evaluator path: paper1_final_eval_v2.verify_checkpoint2 (hash vs result.json) + load_model2 (keys waternet_s0, wwe_uie_s0),
best_valA, batch per geometry (square256 16; aspect482 4 per Amendment 03), fp32, TF32 off, clamp [0,1]; scoring via sota_hicrd_common.score (frozen PSNR/SSIM on float32 + exact E1r).
  square256: waternet_lsui_s0, wwe_uie_lsui_s0;   aspect482: waternet_lsui_s0 only (WWE-UIE needs H, W divisible by 4).
Our 12 models are NOT re-run: their Stage 4 per-image outputs are reused by the analysis. Group R methods are refused.
Usage (real, after approval; system Python or venv_sota):
  python v2_tools/sota_hicrd_infer_t.py --geometry square256 --out paper1_uieb_runs/sota_hicrd_retrained_square256
  python v2_tools/sota_hicrd_infer_t.py --geometry aspect482 --out paper1_uieb_runs/sota_hicrd_retrained_G482
Add --dry-run to verify everything and write nothing."""
import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2_tools")); sys.path.insert(0, str(ROOT))
import sota_hicrd_common as K  # noqa: E402
import sota_hicrd_protocol as P  # noqa: E402

MODELS_BY_GEOMETRY = {"square256": ["waternet_lsui_s0", "wwe_uie_lsui_s0"], "aspect482": ["waternet_lsui_s0"]}
BATCH_BY_GEOMETRY = {"square256": 16, "aspect482": 4}   # Amendment 03 (batch-4 preflight passed); square256 unchanged
AMENDMENT_03 = ("reports/paper1_narrowed/SOTA_HICRD_comparison_protocol_AMENDMENT_03_PROPOSED.md", "0bd811b34d702bbf882dbf2874006fee1d08539834d4315c9495b50a72a433d7")


def verify_group_t(models):
    import paper1_final_eval_v2 as V
    ck = {}
    for m in models:
        if m not in P.GROUP_T:
            raise SystemExit(f"STOP: {m} is not a Group T model (Group R methods are refused here)")
        key = P.GROUP_T[m]["model_key"]
        path, h = V.verify_checkpoint2(V.MODELS[key][0], "best_valA")
        if h != P.GROUP_T[m]["sha256"]:
            raise SystemExit(f"STOP: {m} checkpoint hash differs from the registry")
        ck[m] = (key, path, h)
    return ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, choices=sorted(MODELS_BY_GEOMETRY)); ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--test-manifest", default=None); ap.add_argument("--models", default=None, help="test mode only")
    a = ap.parse_args(argv)
    test = a.test_manifest is not None
    if a.models and not test:
        raise SystemExit("STOP: --models is only allowed in test mode")
    models = a.models.split(",") if a.models else MODELS_BY_GEOMETRY[a.geometry]
    if any(m not in MODELS_BY_GEOMETRY[a.geometry] for m in models if m in P.GROUP_T):
        raise SystemExit("STOP: a requested Group T model is not applicable at this geometry")
    K.verify_protocol_files()
    if K.sha(ROOT / AMENDMENT_03[0]) != AMENDMENT_03[1]:
        raise SystemExit("STOP: Amendment 03 hash differs from the frozen value")
    BATCH = BATCH_BY_GEOMETRY[a.geometry]
    ck = verify_group_t(models)
    pairs = K.load_pairs(a.test_manifest); ids = [p["id"] for p in pairs]
    if not test:
        P.assert_aligned(ids)
    out = K.check_out(ROOT / a.out if not Path(a.out).is_absolute() else a.out, test)
    vis = P.visual_selection() if not test else ids[:2]
    plan = {"geometry": a.geometry, "models": models, "status": P.GROUP_T_STATUS, "n_pairs": len(pairs), "out": str(out), "batch": BATCH, "test_mode": test,
            "checkpoints": {m: c[2] for m, c in ck.items()}}
    if a.dry_run:
        print("DRY RUN OK:", json.dumps(plan)); return {"dry_run": True, "plan": plan}
    import torch
    import train_v2 as tv
    import paper1_final_eval_v2 as V
    V.E.restrict_manifests(tv.MANIFESTS)
    if sorted(tv.MANIFESTS) != ["val_A"]:
        raise SystemExit("STOP: manifest table not restricted")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = {"dataset": "sota_hicrd_T", "out": str(out), "geometry": a.geometry, "models": models, "script_sha256": K.sha(__file__)}
    out.mkdir(parents=True)
    K.ledger("started", rec, test)
    try:
        t0 = time.time()
        X, Y = K.preload(pairs, a.geometry)
        H, W = X.shape[2:]
        vis_idx = [ids.index(k) for k in vis]
        summary = {"meta": dict(plan, protocol_sha256=K.PROTOCOL_V2[1], torch=torch.__version__, device=str(dev)), "models": {}}
        for m in models:
            key, path, h = ck[m]
            model = V.load_model2(key, path, "best_valA", dev)
            mem = K.memory_guard(lambda x: model(x)["enhanced"], (BATCH, 3, H, W), dev, m)
            K.set_tf32_off()
            outs = []
            with torch.no_grad():
                for i in range(0, len(X), BATCH):
                    raw = model(X[i:i + BATCH].to(dev).float() / 255.0)["enhanced"]
                    K.check_finite(raw, m)
                    outs.append(torch.clamp(raw, 0.0, 1.0).float().cpu())
            prim = torch.cat(outs)
            ps, ss, d = K.score(prim, Y.float() / 255.0)
            chk = K.check_scores(ps, d, m)
            K.write_per_image(out / f"per_image_{m}.csv", ids, ps, ss, d, 0)
            np.save(out / f"visual_{m}.npy", prim[vis_idx].numpy())
            after = K.sha(path) == h
            if not after:
                raise SystemExit(f"STOP: {m} checkpoint changed during the run")
            summary["models"][m] = dict(chk, n=len(ids), mean_psnr=float(ps.mean()), memory=mem, checkpoint_unchanged=after)
            print(m, f"mean PSNR {ps.mean():.4f}", flush=True)
            del model; torch.cuda.empty_cache()
        summary["meta"]["seconds"] = round(time.time() - t0, 1)
        json.dump(summary, open(out / "summary.json", "x"), indent=1, default=str)
        K.ledger("completed", rec, test)
        return summary
    except BaseException as e:
        K.ledger("failed", dict(rec, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:]), test)
        raise


if __name__ == "__main__":
    main()
