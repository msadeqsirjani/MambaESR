import torch
from piq import psnr, ssim, LPIPS

# Global metric instances (initialized lazily)
_LPIPS_METRIC = None


def calculate_psnr(sr, hr, data_range=1.0):
    """Calculate PSNR in [0,1] range"""
    # Clamp values to [0, data_range] as required by piq
    sr_clamped = torch.clamp(sr, 0, data_range)
    hr_clamped = torch.clamp(hr, 0, data_range)
    return psnr(sr_clamped, hr_clamped, data_range=data_range).item()


def calculate_ssim(sr, hr, data_range=1.0):
    """Calculate SSIM in [0,1] range"""
    # Clamp values to [0, data_range] as required by piq
    sr_clamped = torch.clamp(sr, 0, data_range)
    hr_clamped = torch.clamp(hr, 0, data_range)
    return ssim(sr_clamped, hr_clamped, data_range=data_range).item()


def calculate_lpips(sr, hr):
    """Calculate LPIPS (perceptual similarity) using piq.LPIPS class"""
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None or next(_LPIPS_METRIC.parameters()).device != sr.device:
        _LPIPS_METRIC = LPIPS(reduction="mean").to(sr.device)
    # Clamp values to [0, 1] as required by piq
    sr_clamped = torch.clamp(sr, 0, 1)
    hr_clamped = torch.clamp(hr, 0, 1)
    return _LPIPS_METRIC(sr_clamped, hr_clamped).item()


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


def calculate_psnr_y(
    sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0
) -> float:
    """PSNR on Y channel in [0,1]. Accepts (B,3,H,W) or (3,H,W)."""
    # Clamp inputs to [0, data_range] before processing
    sr_clamped = torch.clamp(sr, 0, data_range)
    hr_clamped = torch.clamp(hr, 0, data_range)

    y_sr = _rgb_to_y(sr_clamped)
    y_hr = _rgb_to_y(hr_clamped)
    # Ensure shapes match piq expected channel dimension
    if y_sr.dim() == 3:
        y_sr = y_sr.unsqueeze(0)
        y_hr = y_hr.unsqueeze(0)
    return psnr(y_sr, y_hr, data_range=data_range).item()


def calculate_ssim_y(
    sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0
) -> float:
    """SSIM on Y channel in [0,1]. Accepts (B,3,H,W) or (3,H,W)."""
    # Clamp inputs to [0, data_range] before processing
    sr_clamped = torch.clamp(sr, 0, data_range)
    hr_clamped = torch.clamp(hr, 0, data_range)

    y_sr = _rgb_to_y(sr_clamped)
    y_hr = _rgb_to_y(hr_clamped)
    if y_sr.dim() == 3:
        y_sr = y_sr.unsqueeze(0)
        y_hr = y_hr.unsqueeze(0)
    return ssim(y_sr, y_hr, data_range=data_range).item()


def save_checkpoint(model, optimizer, step, path):
    """Save model checkpoint to path."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
        },
        path,
    )


def load_checkpoint(model, path, map_location="cpu"):
    """Load model weights from checkpoint and return saved step (if any)."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
    return state.get("step", 0)
