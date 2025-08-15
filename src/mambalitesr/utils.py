import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
try:
    from piq import psnr as piq_psnr, ssim as piq_ssim, fid as piq_fid, lpips as piq_lpips
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False
    print("Warning: piq not installed. Install with 'pip install piq' for real metrics.")

def _psnr_manual(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Manual PSNR calculation as fallback"""
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def _ssim_manual(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Simplified SSIM calculation as fallback"""
    # Very basic approximation - not accurate but better than dummy values
    mu1 = torch.mean(sr)
    mu2 = torch.mean(hr)
    sigma1_sq = torch.var(sr)
    sigma2_sq = torch.var(hr)
    sigma12 = torch.mean((sr - mu1) * (hr - mu2))
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return numerator / denominator

def calculate_psnr(sr, hr, data_range=1.0):
    """Calculate PSNR in [0,1] range"""
    if PIQ_AVAILABLE:
        return piq_psnr(sr, hr, data_range=data_range).item()
    else:
        return _psnr_manual(sr, hr, data_range).item()

def calculate_ssim(sr, hr, data_range=1.0):
    """Calculate SSIM in [0,1] range"""
    if PIQ_AVAILABLE:
        return piq_ssim(sr, hr, data_range=data_range).item()
    else:
        return _ssim_manual(sr, hr, data_range).item()

def calculate_lpips(sr, hr):
    """Calculate LPIPS (perceptual similarity)"""
    if PIQ_AVAILABLE:
        return piq_lpips(sr, hr, reduction='mean').item()
    else:
        # Fallback: use L2 distance as rough perceptual proxy
        return torch.mean((sr - hr) ** 2).item()

def calculate_fid(sr, hr):
    """Calculate FID (requires batch of images)"""
    if PIQ_AVAILABLE:
        return piq_fid(sr, hr).item()
    else:
        # Fallback: use feature distance approximation
        sr_flat = sr.flatten(1).mean(1)
        hr_flat = hr.flatten(1).mean(1)
        return torch.mean((sr_flat - hr_flat) ** 2).item() * 100


def _rgb_to_y(t: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [0,1] to Y (luma) channel using BT.601 coefficients.

    Accepts (B,3,H,W) or (3,H,W). Returns (B,1,H,W) or (1,H,W) respectively.
    """
    if t.dim() == 3:
        # (C,H,W)
        r, g, b = t[0:1], t[1:2], t[2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y
    elif t.dim() == 4:
        # (B,C,H,W)
        r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y
    else:
        raise ValueError("Expected tensor with 3 or 4 dims (C,H,W) or (B,C,H,W)")


def calculate_psnr_y(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR on Y channel in [0,1]. Accepts (B,3,H,W) or (3,H,W)."""
    y_sr = _rgb_to_y(sr)
    y_hr = _rgb_to_y(hr)
    # Ensure shapes match piq expected channel dimension
    if y_sr.dim() == 3:
        y_sr = y_sr.unsqueeze(0)
        y_hr = y_hr.unsqueeze(0)
    return psnr(y_sr, y_hr, data_range=data_range).item()


def calculate_ssim_y(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> float:
    """SSIM on Y channel in [0,1]. Accepts (B,3,H,W) or (3,H,W)."""
    y_sr = _rgb_to_y(sr)
    y_hr = _rgb_to_y(hr)
    if y_sr.dim() == 3:
        y_sr = y_sr.unsqueeze(0)
        y_hr = y_hr.unsqueeze(0)
    return ssim(y_sr, y_hr, data_range=data_range).item()

def save_comparison(lr, sr, hr, path):
    """Save visual comparison of LR, SR, and HR images"""
    # Convert tensors to PIL Images
    lr_img = to_pil_image(lr[0].cpu().clamp(0, 1))
    sr_img = to_pil_image(sr[0].cpu().clamp(0, 1))
    hr_img = to_pil_image(hr[0].cpu().clamp(0, 1))
    
    # Create canvas
    width = lr_img.width + sr_img.width + hr_img.width
    height = max(lr_img.height, sr_img.height, hr_img.height)
    canvas = Image.new('RGB', (width, height), (255, 255, 255))
    
    # Paste images
    canvas.paste(lr_img, (0, (height - lr_img.height) // 2))
    canvas.paste(sr_img, (lr_img.width, (height - sr_img.height) // 2))
    canvas.paste(hr_img, (lr_img.width + sr_img.width, (height - hr_img.height) // 2))
    
    # Save result
    canvas.save(path)


def save_checkpoint(model, optimizer, step, path):
    """Save model checkpoint to path."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
    }, path)


def load_checkpoint(model, path, map_location="cpu"):
    """Load model weights from checkpoint and return saved step (if any)."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    return state.get("step", 0)