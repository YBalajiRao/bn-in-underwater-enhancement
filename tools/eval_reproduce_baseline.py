import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import json
import time
import hashlib
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from original_repo.marine_net_model import MARINENet

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()

def create_window(window_size: int, channel: int):
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    _1d_window = g.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window.expand(channel, 1, window_size, window_size).contiguous()

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def evaluate_manifest(model, manifest_path: Path, device: torch.device):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pairs = data.get("test_pairs", data.get("pairs", []))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    per_image = []
    latencies = []
    
    print(f"Evaluating {len(pairs)} pairs from {manifest_path.name}...")
    
    for idx, item in enumerate(pairs):
        inp_path_str = item.get("input_path", item.get("image_path", ""))
        gt_path_str = item.get("gt_path", item.get("target_path", ""))
        
        inp_img = Image.open(inp_path_str).convert("RGB")
        gt_img = Image.open(gt_path_str).convert("RGB")
        
        inp_t = transform(inp_img).unsqueeze(0).to(device)
        gt_t = transform(gt_img).unsqueeze(0).to(device)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(inp_t)
            out_full = torch.clamp(outputs["enhanced"], 0.0, 1.0)
            out_cnn = torch.clamp(outputs["cnn_output"], 0.0, 1.0)
            out_inr = torch.clamp(outputs["inr_output"], 0.0, 1.0)
            
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        
        psnr_full = compute_psnr(out_full, gt_t)
        ssim_full = compute_ssim(out_full, gt_t)
        
        psnr_cnn = compute_psnr(out_cnn, gt_t)
        ssim_cnn = compute_ssim(out_cnn, gt_t)
        
        psnr_inr = compute_psnr(out_inr, gt_t)
        ssim_inr = compute_ssim(out_inr, gt_t)
        
        img_id = item.get("id", idx)
        per_image.append({
            "image_id": img_id,
            "filename": Path(inp_path_str).name,
            "psnr": float(psnr_full),
            "ssim": float(ssim_full),
            "psnr_cnn": float(psnr_cnn),
            "ssim_cnn": float(ssim_cnn),
            "psnr_inr": float(psnr_inr),
            "ssim_inr": float(ssim_inr),
            "latency_ms": float(lat_ms),
        })

    steady_lat = latencies[10:] if len(latencies) > 10 else latencies
    mean_lat = float(sum(steady_lat) / len(steady_lat))
    throughput = float(1000.0 / mean_lat)
    
    summary = {
        "n": len(per_image),
        "psnr_mean": float(sum(x["psnr"] for x in per_image) / len(per_image)),
        "ssim_mean": float(sum(x["ssim"] for x in per_image) / len(per_image)),
        "psnr_cnn_mean": float(sum(x["psnr_cnn"] for x in per_image) / len(per_image)),
        "ssim_cnn_mean": float(sum(x["ssim_cnn"] for x in per_image) / len(per_image)),
        "psnr_inr_mean": float(sum(x["psnr_inr"] for x in per_image) / len(per_image)),
        "ssim_inr_mean": float(sum(x["ssim_inr"] for x in per_image) / len(per_image)),
        "latency_ms_mean": mean_lat,
        "throughput_fps": throughput,
    }
    
    return {"summary": summary, "per_image": per_image}

def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    ckpt_path = root / "checkpoints/official/kaggle/working/checkpoints/marine_net_lsui_finetuned/best_finetuned.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
    
    model = MARINENet().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully with strict=True.")
    
    # 1. Evaluate canonical test manifest
    canon_manifest_path = root / "datasets/splits/lsui_test_canonical_400.json"
    res_canon = evaluate_manifest(model, canon_manifest_path, device)
    res_canon_path = root / "results/lsui_baseline_reproduced_canonical.json"
    res_canon_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_canon_path, "w", encoding="utf-8") as f:
        json.dump(res_canon, f, indent=2)
    print(f"Saved canonical reproduction to: {res_canon_path}")
    print(f"  Canonical Summary: PSNR={res_canon['summary']['psnr_mean']:.4f} dB, SSIM={res_canon['summary']['ssim_mean']:.4f}")
    
    # 2. Evaluate candidate test manifest
    cand_manifest_path = root / "datasets/splits/candidate_v1/lsui_test_400.json"
    res_cand = evaluate_manifest(model, cand_manifest_path, device)
    res_cand_path = root / "results/lsui_baseline_reproduced_candidate.json"
    with open(res_cand_path, "w", encoding="utf-8") as f:
        json.dump(res_cand, f, indent=2)
    print(f"Saved candidate reproduction to: {res_cand_path}")
    print(f"  Candidate Summary: PSNR={res_cand['summary']['psnr_mean']:.4f} dB, SSIM={res_cand['summary']['ssim_mean']:.4f}")

if __name__ == "__main__":
    main()
