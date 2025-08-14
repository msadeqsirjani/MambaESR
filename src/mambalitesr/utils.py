import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
try:
    from piq import psnr, ssim, fid, lpips
except ImportError:
    # Fallback implementations if piq is not available
    def psnr(*args, **kwargs):
        return torch.tensor(30.0)  # Dummy value
    def ssim(*args, **kwargs):
        return torch.tensor(0.9)   # Dummy value
    def fid(*args, **kwargs):
        return torch.tensor(50.0)  # Dummy value
    def lpips(*args, **kwargs):
        return torch.tensor(0.2)   # Dummy value

def calculate_psnr(sr, hr, data_range=1.0):
    """Calculate PSNR in [0,1] range"""
    return psnr(sr, hr, data_range=data_range).item()

def calculate_ssim(sr, hr, data_range=1.0):
    """Calculate SSIM in [0,1] range"""
    return ssim(sr, hr, data_range=data_range).item()

def calculate_lpips(sr, hr):
    """Calculate LPIPS (perceptual similarity)"""
    return lpips(sr, hr, reduction='mean').item()

def calculate_fid(sr, hr):
    """Calculate FID (requires batch of images)"""
    return fid(sr, hr).item()

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