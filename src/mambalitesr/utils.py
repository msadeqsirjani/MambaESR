from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def rgb_to_y_channel(img: np.ndarray) -> np.ndarray:
    # img: HWC, RGB, [0,255]
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 16.0 + (65.738 * r + 129.057 * g + 25.064 * b) / 256.0
    return np.clip(y, 16.0, 235.0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    # t: BCHW in [0,1]
    t = t.detach().clamp(0, 1)
    if t.dim() == 4:
        t = t[0]
    img = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return img


def calculate_psnr_y(sr: torch.Tensor, hr: torch.Tensor, shave: int = 4) -> float:
    sr_img = tensor_to_image(sr)
    hr_img = tensor_to_image(hr)
    if shave > 0:
        sr_img = sr_img[shave:-shave, shave:-shave]
        hr_img = hr_img[shave:-shave, shave:-shave]
    y_sr = rgb_to_y_channel(sr_img)
    y_hr = rgb_to_y_channel(hr_img)
    mse = np.mean((y_sr - y_hr) ** 2)
    if mse <= 1e-10:
        return 100.0
    return 10 * math.log10((235.0 - 16.0) ** 2 / mse)


def calculate_ssim_y(sr: torch.Tensor, hr: torch.Tensor, shave: int = 4) -> float:
    sr_img = tensor_to_image(sr)
    hr_img = tensor_to_image(hr)
    if shave > 0:
        sr_img = sr_img[shave:-shave, shave:-shave]
        hr_img = hr_img[shave:-shave, shave:-shave]
    y_sr = rgb_to_y_channel(sr_img)
    y_hr = rgb_to_y_channel(hr_img)
    return float(ssim(y_hr, y_sr, data_range=235.0 - 16.0))


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(model: torch.nn.Module, path: Path, map_location: str | torch.device = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"], strict=True)
