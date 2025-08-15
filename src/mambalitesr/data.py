from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import random
from PIL import Image
from torchvision.transforms.functional import to_tensor as tv_to_tensor
from torchvision.transforms.functional import hflip, vflip

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def _load_image_as_tensor(path: Path) -> torch.Tensor:
    # Returns CxHxW float tensor in [0,1]
    img = Image.open(path).convert("RGB")
    return tv_to_tensor(img)


def _bicubic_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    # image: CxHxW in [0,1]
    c, h, w = image.shape
    nh = (h // scale) * 1
    nw = (w // scale) * 1
    image = image[:, : (nh * scale), : (nw * scale)]
    image = image.unsqueeze(0)
    lr = F.interpolate(image, scale_factor=1.0 / scale, mode="bicubic", align_corners=False)
    return lr.squeeze(0).clamp(0.0, 1.0)


class DF2KDataset(Dataset):
    """DIV2K/DF2K-style HR dataset with on-the-fly LR generation and random patches for training."""

    def __init__(self, hr_dir: Path, scale: int, lr_patch_size: int | None = None) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        self.lr_patch_size = lr_patch_size

        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")

        self.hr_images = sorted([p for p in self.hr_dir.rglob("*") if p.is_file() and _is_image_file(p)])
        if len(self.hr_images) == 0:
            raise RuntimeError(f"No images found in {self.hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_images)

    def _random_crop_hr(self, hr: torch.Tensor) -> torch.Tensor:
        if self.lr_patch_size is None:
            return hr
        hr_patch = self.lr_patch_size * self.scale
        _, h, w = hr.shape
        if h < hr_patch or w < hr_patch:
            # Center crop to the largest multiple of scale possible
            h_new = (h // self.scale) * self.scale
            w_new = (w // self.scale) * self.scale
            top = max(0, (h - h_new) // 2)
            left = max(0, (w - w_new) // 2)
            return hr[:, top : top + h_new, left : left + w_new]

        top = random.randint(0, h - hr_patch)
        left = random.randint(0, w - hr_patch)
        return hr[:, top : top + hr_patch, left : left + hr_patch]

    def __getitem__(self, idx: int) -> dict:
        hr_path = self.hr_images[idx]
        hr = _load_image_as_tensor(hr_path)
        hr = self._random_crop_hr(hr)
        
        # Data augmentation for training
        if self.lr_patch_size is not None:  # Training mode
            # Random horizontal flip
            if random.random() > 0.5:
                hr = hflip(hr)
            # Random vertical flip
            if random.random() > 0.5:
                hr = vflip(hr)
            # Random 90-degree rotation
            if random.random() > 0.5:
                hr = torch.rot90(hr, k=random.randint(1, 3), dims=[-2, -1])
        
        lr = _bicubic_downsample(hr, self.scale)
        return {"lr": lr, "hr": hr}


class EvalImageFolder(Dataset):
    """Evaluation dataset that loads HR images and generates LR by bicubic downsampling."""

    def __init__(self, hr_dir: Path, scale: int) -> None:
        self.hr_dir = Path(hr_dir)
        self.scale = int(scale)
        if not self.hr_dir.exists():
            raise FileNotFoundError(f"HR directory not found: {self.hr_dir}")
        self.hr_images = sorted([p for p in self.hr_dir.rglob("*") if p.is_file() and _is_image_file(p)])
        if len(self.hr_images) == 0:
            raise RuntimeError(f"No images found in {self.hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int) -> dict:
        hr_path = self.hr_images[idx]
        hr = _load_image_as_tensor(hr_path)
        # Ensure dimensions are multiples of scale for clean downsampling
        c, h, w = hr.shape
        h_new = (h // self.scale) * self.scale
        w_new = (w // self.scale) * self.scale
        hr = hr[:, :h_new, :w_new]
        lr = _bicubic_downsample(hr, self.scale)
        return {"lr": lr, "hr": hr}


def _resolve_default_dirs(data_root: str | Path) -> Tuple[Path, Path]:
    root = Path(data_root)
    # Common layouts
    candidates = [
        (root / "DIV2K_train_HR", root / "DIV2K_valid_HR"),
        (root / "train" / "HR", root / "val" / "HR"),
        (root / "HR" / "train", root / "HR" / "val"),
    ]
    for tr, va in candidates:
        if tr.exists() and va.exists():
            return tr, va
    # If root itself is an HR dir, use it for both (fallback)
    return root, root


def create_dataloaders(
    data_root: str | Path,
    scale: int,
    batch_size: int,
    lr_patch_size: int | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for SR training.

    Expects one of the following layouts under data_root:
      - DIV2K_train_HR / DIV2K_valid_HR
      - train/HR / val/HR
      - HR/train / HR/val
    """
    train_hr_dir, val_hr_dir = _resolve_default_dirs(data_root)

    train_ds = DF2KDataset(train_hr_dir, scale=scale, lr_patch_size=lr_patch_size)
    val_ds = DF2KDataset(val_hr_dir, scale=scale, lr_patch_size=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    return train_loader, val_loader


__all__ = ["DF2KDataset", "EvalImageFolder", "create_dataloaders"]