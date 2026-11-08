"""train_v2.py - Stage 2 training pipeline (the project plan).

Official MARINE-Net recipe (original_repo/train_marine_net.py) re-implemented with:
model registry, AMP, grad accumulation, 10-epoch warmup + cosine, seed control, paired
ColorJitter/flip, checkpoint by best val PSNR, full logging, manifest hash guard, resume.
Metrics: frozen compute_psnr / compute_ssim imported from tools/eval_reproduce_baseline.py.
Old scripts are NOT modified. Outputs go to runs_v2/<run_name>/.

Deviations from the official script (documented, minimal):
  1. checkpoint selection by val PSNR (the project plan Stage 2), not val loss.
  2. a trailing partial accumulation group is flushed at epoch end (official lets its
     gradients leak into the next epoch).
  3. images are pre-resized once (Resize((256,256)) happens before jitter in the official
     pipeline, so this is equivalent) and kept in RAM; jitter/flip run on uint8 tensors.
  4. --skip_zero_weight_perceptual (optional): the official perceptual term has weight 0.0,
     skipping it changes no gradient, only time. Off by default.
  5. Under AMP the model forward runs in autocast but the ENTIRE loss is computed in fp32
     (autocast disabled, outputs/targets cast to float32). Reason (Stage 2 diagnostics): the
     pytorch-msssim SSIM term in fp16 gives ~190x inflated gradients -> GradScaler collapse.
     --loss_in_autocast restores the official (broken) behaviour for reproduction only.
  6. --eta_min (default 1e-7 = official) added for recipe tests (Recipe R2 uses 1e-6).
  7. --no_jitter (off by default) drops ColorJitter, keeps the paired flip (Stage 2 jitter test).
  8. --stop_after N trains only epochs 0..N-1 of the --epochs LR schedule (schedule unchanged).
  9. --recipe R3: frozen Stage-3 recipe (RECIPES dict); overrides the arguments it lists.
 10. --select_on val_A (set by R3): evaluate and select on val_A only; val_B is never loaded here.
     Single-output models (M1/M2, final_output_only) use FinalOutputLoss (MARINELoss, aux terms = 0).
"""
import argparse, hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "original_repo"))

from original_repo.marine_net_model import MARINENet, MARINELoss  # noqa: E402
from tools.eval_reproduce_baseline import compute_psnr, compute_ssim  # noqa: E402  (frozen)
from models_v2.color_nets import GlobalColorNet, GlobalLocalColorNet  # noqa: E402  (Stage 3 M1/M2)
from models.at_marine import ATMarineTiny, ATMarineTrunk  # noqa: E402  (Stage 4A variants)

SPLITS = ROOT / "datasets/splits"
MANIFESTS = {
    "train": (SPLITS / "candidate_v1/lsui_train_3463.json", "train_pairs", "b4871a98"),
    "val": (SPLITS / "candidate_v1/lsui_val_368.json", "val_pairs", "c034978c"),
    "val_clean": (SPLITS / "derived_v2/lsui_val_clean.json", "val_pairs", "dd48c348"),
    # Stage 3 split of val_clean by near-duplicate cluster (v2_tools/make_val_ab.py). val_A = checkpoint
    # selection only. val_B = one-time Stage 3 comparison; NEVER loaded by this training script.
    "val_A": (SPLITS / "derived_v3/lsui_val_A.json", "val_pairs", "2d8daae6"),
    "val_B": (SPLITS / "derived_v3/lsui_val_B.json", "val_pairs", "4b2e1c41"),
}

# ----------------------------------------------------------------------------- registry
MODEL_REGISTRY = {
    "marine_default": lambda: MARINENet(),  # default MARINENet() == official checkpoint architecture (M3)
    "m1_global": lambda: GlobalColorNet(),  # Stage 3 M1
    "m2_globallocal": lambda: GlobalLocalColorNet(),  # Stage 3 M2
    "at_tiny": lambda: ATMarineTiny(),  # Stage 4A: M2 colour branch + light depthwise-separable detail branch
    "at_trunk": lambda: ATMarineTrunk(),  # Stage 4A: MARINE trunk, colour from bottleneck, detail from decoder
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_manifests():
    hashes = {}
    for k, (p, _, prefix) in MANIFESTS.items():
        h = sha256(p)
        if not h.startswith(prefix):
            sys.exit(f"STOP: manifest {k} hash {h[:8]} != expected {prefix} ({p})")
        hashes[k] = h
    return hashes


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def env_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    return {"git_commit": commit, "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "command": " ".join(sys.argv)}


# ----------------------------------------------------------------------------- data
def load_pairs(key, limit=None):
    p, k, _ = MANIFESTS[key]
    pairs = json.load(open(p))[k]
    return pairs[:limit] if limit else pairs


def preload(pairs, size=256):
    """Resize((256,256)) once; uint8 [N,3,H,W] for input and target (ToTensor == /255)."""
    rs = T.Resize((size, size))
    X = torch.empty(len(pairs), 3, size, size, dtype=torch.uint8)
    Y = torch.empty_like(X)
    for i, p in enumerate(pairs):
        for dst, path in ((X, p["input_path"]), (Y, p["gt_path"])):
            im = rs(Image.open(path).convert("RGB"))
            dst[i] = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)
    return X, Y


class PairedAugDataset(Dataset):
    """Flip + ColorJitter(0.1,0.1,0.1,0.05) applied IDENTICALLY to input and target
    (same seed re-set before each call, as in the official script)."""

    def __init__(self, X, Y, base_seed, jitter=True):
        self.X, self.Y = X.share_memory_(), Y.share_memory_()
        self.base_seed = base_seed
        self.epoch = torch.zeros(1, dtype=torch.long).share_memory_()
        # flip is first, so with jitter=False each sample gets the same flip decision as with jitter=True
        self.tf = T.Compose([T.RandomHorizontalFlip(p=0.5)] +
                            ([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)] if jitter else []))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seed = (self.base_seed * 1_000_003 + int(self.epoch.item()) * 7_919 + i) % (2**31 - 1)
        torch.manual_seed(seed); x = self.tf(self.X[i])
        torch.manual_seed(seed); y = self.tf(self.Y[i])
        return x.float() / 255.0, y.float() / 255.0


# ----------------------------------------------------------------------------- eval (frozen metrics)
@torch.no_grad()
def evaluate(model, X, Y, device, bs=16):
    """True FP32 (TF32 off), clamp [0,1], per-image PSNR/SSIM from the frozen functions."""
    model.eval()
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 convs add ~5e-4 output noise (~0.005 dB)
    ps, ss = [], []
    for i in range(0, len(X), bs):
        x = (X[i:i + bs].to(device).float() / 255.0)
        y = (Y[i:i + bs].to(device).float() / 255.0)
        out = torch.clamp(model(x)["enhanced"], 0.0, 1.0)
        for j in range(len(x)):
            ps.append(compute_psnr(out[j:j + 1], y[j:j + 1]))
            ss.append(compute_ssim(out[j:j + 1], y[j:j + 1]))
    torch.backends.cudnn.allow_tf32 = prev_tf32
    return np.array(ps), np.array(ss)


def summarize(ps, ss, mask=None):
    if mask is not None:
        ps, ss = ps[mask], ss[mask]
    return {"n": int(len(ps)), "psnr_mean": float(ps.mean()), "psnr_median": float(np.median(ps)),
            "ssim_mean": float(ss.mean())}


# ----------------------------------------------------------------------------- LR schedule (official)
def make_optim(model, a):
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs - a.warmup, eta_min=getattr(a, "eta_min", 1e-7))
    return opt, sched


def apply_warmup(opt, epoch, a):
    if epoch < a.warmup:
        for g in opt.param_groups:
            g["lr"] = a.lr * (epoch + 1) / a.warmup


# ----------------------------------------------------------------------------- training
def compute_loss(crit, out, y, loss_in_autocast=False):
    """MARINELoss in fp32: autocast disabled, model outputs and target cast to float32."""
    if loss_in_autocast:
        return crit(out, y)
    with torch.autocast("cuda", enabled=False):
        return crit({k: v.float() for k, v in out.items()}, y.float())


def train_one_epoch(model, crit, opt, scaler, loader, a, device, max_iters=None, timer=None):
    model.train()
    agg = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "gradient": 0.0, "color": 0.0}
    n, nonfinite, gnorms, pending = 0, 0, [], 0
    opt.zero_grad(set_to_none=True)
    for bi, (x, y) in enumerate(loader):
        if max_iters is not None and bi >= max_iters:
            break
        if timer is not None:
            timer(bi, "start")
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lia = getattr(a, "loss_in_autocast", False)
        with torch.autocast("cuda", enabled=a.amp):
            out = model(x)
            if lia:
                ld = compute_loss(crit, out, y, loss_in_autocast=True)
        if not lia:
            ld = compute_loss(crit, out, y)
        loss = ld["total"] / a.accum
        if not torch.isfinite(loss):
            nonfinite += 1
            continue
        scaler.scale(loss).backward() if a.amp else loss.backward()
        pending += 1
        if pending == a.accum:
            gnorms.append(step(model, opt, scaler, a)); pending = 0
        for k in agg:
            agg[k] += float(ld[k])
        n += 1
        if timer is not None:
            timer(bi, "end")
    if pending:  # deviation 2: flush trailing partial group
        gnorms.append(step(model, opt, scaler, a))
    g = np.array(gnorms, dtype=np.float64) if gnorms else np.array([0.0])
    inf_grad_steps = int((~np.isfinite(g)).sum())  # under AMP: GradScaler overflow -> step skipped
    gf = g[np.isfinite(g)] if np.isfinite(g).any() else np.array([0.0])
    res = {k: v / max(n, 1) for k, v in agg.items()}
    res.update(nonfinite=nonfinite, inf_grad_steps=inf_grad_steps, gnorm_mean=float(gf.mean()),
               gnorm_max=float(gf.max()), steps=len(gnorms))
    return res


def step(model, opt, scaler, a):
    if a.amp:
        scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
    if a.amp:
        scaler.step(opt); scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return float(gn)


class FinalOutputLoss(nn.Module):
    """MARINELoss for models with a single output (M1/M2): the CNN/INR auxiliary L1 terms are fed the
    target itself, so they are exactly 0; every other term and weight is the unchanged MARINELoss."""

    def __init__(self, crit):
        super().__init__(); self.crit = crit

    def forward(self, out, y):
        t = y.detach()
        return self.crit({"enhanced": out["enhanced"], "cnn_output": t, "inr_output": t}, y)


class ZeroLoss(nn.Module):
    def forward(self, p, t):
        return p.new_zeros(())


def build(a, device):
    model = MODEL_REGISTRY[a.model]().to(device)
    crit = MARINELoss().to(device)
    if not getattr(crit, "ssim_loss", None):
        sys.exit("STOP: SSIM loss disabled (pytorch-msssim missing) - ask the user.")
    if a.skip_zero_weight_perceptual:
        crit.perceptual = ZeroLoss()
    if getattr(model, "final_output_only", False):
        crit = FinalOutputLoss(crit)
    return model, crit


def make_loader(ds, a, epoch_seed):
    g = torch.Generator().manual_seed(epoch_seed)
    return DataLoader(ds, batch_size=a.bs, sampler=RandomSampler(ds, generator=g), num_workers=a.workers,
                      pin_memory=True, persistent_workers=a.workers > 0)


# Frozen recipes (the project plan Section 3). --recipe overrides the listed arguments; defaults stay official.
RECIPES = {
    # R3 (Stage 3, frozen 2026-10-13) = Recipe R2 without ColorJitter (paired flips kept).
    "R3": dict(epochs=100, warmup=3, bs=4, accum=1, lr=2e-4, wd=1e-5, eta_min=1e-6, clip=1.0, amp=True,
               loss_in_autocast=False, skip_zero_weight_perceptual=True, no_jitter=True, stop_after=None,
               select_on="val_A"),
}


def apply_recipe(a):
    for k, v in RECIPES[a.recipe].items():
        setattr(a, k, v)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marine_default", choices=list(MODEL_REGISTRY))
    ap.add_argument("--run_name", default="stage2_marine_default")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--eta_min", type=float, default=1e-7, help="cosine floor (official 1e-7)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_amp", action="store_false", dest="amp")
    ap.add_argument("--loss_in_autocast", action="store_true", help="reproduce official fp16 loss (unstable)")
    ap.add_argument("--skip_zero_weight_perceptual", action="store_true")
    ap.add_argument("--no_jitter", action="store_true", help="disable ColorJitter (paired flips kept)")
    ap.add_argument("--stop_after", type=int, default=None, help="train only the first N epochs of the --epochs schedule")
    ap.add_argument("--time_iters", type=int, default=0, help="time N iterations then exit (no training run)")
    ap.add_argument("--smoke", type=int, default=0, help="run N iterations of epoch 0 + eval on 16 val imgs, then exit")
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--underfit_n", type=int, default=300)
    ap.add_argument("--select_on", default="val368", choices=["val368", "val_A"],
                    help="val368: eval val_368 (+val_clean subset), select on val_368 (Stage 2). val_A: eval/select on val_A only")
    ap.add_argument("--recipe", default=None, choices=list(RECIPES), help="frozen recipe; overrides its listed args")
    a = ap.parse_args()
    if a.recipe:
        apply_recipe(a)

    device = torch.device("cuda")
    hashes = verify_manifests()
    seed_everything(a.seed)
    out_dir = ROOT / "runs_v2" / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, crit = build(a, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={a.model} params={n_params:,} amp={a.amp} ssim_loss=ON hashes={ {k: v[:8] for k, v in hashes.items()} }")

    t0 = time.time()
    lim = a.train_limit or (64 if a.smoke else 300 if a.time_iters else None)
    Xtr, Ytr = preload(load_pairs("train", lim))
    selA = a.select_on == "val_A"
    val_pairs = load_pairs("val_A" if selA else "val", 16 if a.smoke else None)
    Xv, Yv = preload(val_pairs)
    if selA:
        vmask = None
        print(f"data loaded in {time.time() - t0:.1f}s  train={len(Xtr)} val_A={len(Xv)} (val_B not loaded)")
    else:
        clean_names = {p["filename"] for p in load_pairs("val_clean")}
        vmask = np.array([p["filename"] in clean_names for p in val_pairs])
        print(f"data loaded in {time.time() - t0:.1f}s  train={len(Xtr)} val={len(Xv)} val_clean(in val)={int(vmask.sum())}")

    ds = PairedAugDataset(Xtr, Ytr, a.seed, jitter=not a.no_jitter)
    opt, sched = make_optim(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    env = env_info()

    # ---- timing mode -------------------------------------------------------------------------
    if a.time_iters:
        loader = make_loader(ds, a, a.seed)
        times = []
        def timer(bi, ev):
            torch.cuda.synchronize()
            if ev == "start": timer.t = time.perf_counter()
            else: times.append(time.perf_counter() - timer.t)
        train_one_epoch(model, crit, opt, scaler, loader, a, device, max_iters=a.time_iters + 10, timer=timer)
        t = np.array(times[10:])  # drop 10 warm-up iterations
        full_train = load_pairs("train"); n_it = int(np.ceil(len(full_train) / a.bs))
        t_ev = time.perf_counter(); evaluate(model, Xv, Yv, device); torch.cuda.synchronize(); ev = (time.perf_counter() - t_ev)
        ev_full = ev / len(Xv) * len(load_pairs("val_A" if selA else "val"))
        epoch_s = n_it * t.mean() + ev_full
        res = {"iters_timed": int(len(t)), "iter_ms_mean": float(t.mean() * 1e3), "iter_ms_median": float(np.median(t) * 1e3),
               "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20, "iters_per_epoch": n_it,
               "val_eval_s_est_full": ev_full, "model": a.model, "params": n_params, "epoch_s_est": epoch_s, "skip_perceptual": a.skip_zero_weight_perceptual, "amp": a.amp,
               "estimate_hours": {str(e): e * epoch_s / 3600 for e in (100, 150, 200)}, "env": env}
        print(json.dumps(res, indent=1))
        json.dump(res, open(out_dir / f"timing_perc{'OFF' if a.skip_zero_weight_perceptual else 'ON'}.json", "w"), indent=1)
        return

    # ---- smoke mode --------------------------------------------------------------------------
    if a.smoke:
        loader = make_loader(ds, a, a.seed)
        r = train_one_epoch(model, crit, opt, scaler, loader, a, device, max_iters=a.smoke)
        ps, ss = evaluate(model, Xv, Yv, device)
        print("smoke train:", {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
        print("smoke eval (untrained after few iters):", summarize(ps, ss))
        return

    # ---- full run ----------------------------------------------------------------------------
    hist_path, last_path = out_dir / "history.jsonl", out_dir / "last.pth"
    start, best368, bestclean = 0, -1.0, -1.0
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start = ck["epoch"] + 1; best368, bestclean = ck["best368"], ck["bestclean"]
        print(f"resumed at epoch {start}")
    for epoch in range(start, min(a.epochs, a.stop_after or a.epochs)):
        te = time.time()
        apply_warmup(opt, epoch, a)
        lr = opt.param_groups[0]["lr"]
        ds.epoch[0] = epoch
        loader = make_loader(ds, a, a.seed * 10_007 + epoch) if epoch == start else loader
        loader.sampler.generator.manual_seed(a.seed * 10_007 + epoch)
        tr = train_one_epoch(model, crit, opt, scaler, loader, a, device)
        ps, ss = evaluate(model, Xv, Yv, device)
        if epoch >= a.warmup:
            sched.step()
        if selA:
            vA = summarize(ps, ss)
            rec = {"epoch": epoch, "lr": lr, "train": tr, "val_A": vA,
                   "scaler_scale": float(scaler.get_scale()) if a.amp else None, "epoch_s": time.time() - te}
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if vA["psnr_mean"] > best368:
                best368 = vA["psnr_mean"]; torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_A": vA}, out_dir / "best_valA.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "best368": best368, "bestclean": bestclean}, last_path)
            print(f"ep {epoch:3d} lr {lr:.2e} loss {tr['total']:.4f} valA {vA['psnr_mean']:.3f}/{vA['ssim_mean']:.4f} "
                  f"gn {tr['gnorm_mean']:.2f} nf {tr['nonfinite']}/{tr['inf_grad_steps']} scale {rec['scaler_scale']} {rec['epoch_s']:.0f}s", flush=True)
            continue
        v368, vcl = summarize(ps, ss), summarize(ps, ss, vmask)
        rec = {"epoch": epoch, "lr": lr, "train": tr, "val_368": v368, "val_clean": vcl,
               "scaler_scale": float(scaler.get_scale()) if a.amp else None, "epoch_s": time.time() - te}
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if v368["psnr_mean"] > best368:
            best368 = v368["psnr_mean"]; torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_368": v368, "val_clean": vcl}, out_dir / "best_val368.pth")
        if vcl["psnr_mean"] > bestclean:
            bestclean = vcl["psnr_mean"]; torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_368": v368, "val_clean": vcl}, out_dir / "best_valclean.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best368": best368, "bestclean": bestclean}, last_path)
        print(f"ep {epoch:3d} lr {lr:.2e} loss {tr['total']:.4f} val368 {v368['psnr_mean']:.3f}/{v368['ssim_mean']:.4f} "
              f"clean {vcl['psnr_mean']:.3f}/{vcl['ssim_mean']:.4f} gn {tr['gnorm_mean']:.2f} nf {tr['nonfinite']}/{tr['inf_grad_steps']} scale {rec['scaler_scale']} {rec['epoch_s']:.0f}s", flush=True)

    # ---- final: underfitting check on random train images + result JSON -------------------------
    best_name = "best_valA.pth" if selA else "best_val368.pth"
    ck = torch.load(out_dir / best_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    full = load_pairs("train")
    idx = np.random.RandomState(0).choice(len(full), a.underfit_n, replace=False)
    Xu, Yu = preload([full[i] for i in idx])
    pu, su = evaluate(model, Xu, Yu, device)
    result = {"model": a.model, "params": n_params, "seed": a.seed, "epochs": a.epochs, "args": vars(a),
              "train_300_random": summarize(pu, su), "manifest_sha256": hashes, "env": env}
    if selA:
        result.update(best_valA_epoch=ck["epoch"], best_valA=ck["val_A"],
                      checkpoint_sha256={"best_valA.pth": sha256(out_dir / "best_valA.pth"), "last.pth": sha256(last_path)},
                      run_seconds=sum(json.loads(l)["epoch_s"] for l in open(hist_path)))
    else:
        result.update(best_val368_epoch=ck["epoch"], best_val368=ck["val_368"], best_val368_on_clean=ck["val_clean"],
                      checkpoint_sha256={"best_val368.pth": sha256(out_dir / "best_val368.pth"), "best_valclean.pth": sha256(out_dir / "best_valclean.pth")})
    json.dump(result, open(out_dir / "result.json", "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
