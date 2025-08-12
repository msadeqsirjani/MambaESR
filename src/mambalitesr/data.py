from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class PatchConfig:
    scale: int = 4
    lr_patch_size: int = 64  # yields hr 256 for x4
    augment: bool = True


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


def imread_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w]


def downscale_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return lr


def random_augment_pair(lr: np.ndarray, hr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Horizontal flip
    if random.random() < 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()
    # Vertical flip
    if random.random() < 0.5:
        lr = lr[::-1, :, :].copy()
        hr = hr[::-1, :, :].copy()
    # 90-degree rotations
    k = random.randint(0, 3)
    if k:
        lr = np.rot90(lr, k).copy()
        hr = np.rot90(hr, k).copy()
    return lr, hr


def extract_aligned_patch_pair(lr: np.ndarray, hr: np.ndarray, lr_patch_size: int, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    h_lr, w_lr = lr.shape[:2]
    x = random.randint(0, w_lr - lr_patch_size)
    y = random.randint(0, h_lr - lr_patch_size)
    lr_patch = lr[y : y + lr_patch_size, x : x + lr_patch_size]
    hr_patch = hr[y * scale : (y + lr_patch_size) * scale, x * scale : (x + lr_patch_size) * scale]
    return lr_patch, hr_patch


def to_tensor(img: np.ndarray) -> torch.Tensor:
    # Convert to float tensor in [0,1], CHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return tensor


class DF2KDataset(Dataset):
    """
    - Reads HR images from DF2K/HR
    - Generates LR on-the-fly by bicubic downscale by `scale`
    - For each HR image, yields `num_patches_per_image` random aligned patches (LR 64x64, HR 256x256 by default)
    - Augmentations: random flips and rotations
    """

    def __init__(self, root: Path | str, patch: PatchConfig, num_patches_per_image: int = 8):
        super().__init__()
        root = Path(root)
        self.hr_dir = root / "DF2K" / "HR"
        self.hr_images = list_images(self.hr_dir)
        if not self.hr_images:
            raise RuntimeError(f"No HR images found in {self.hr_dir}")
        self.patch = patch
        self.num_patches_per_image = max(1, num_patches_per_image)
        # Virtually expand dataset by repeating each image `num_patches_per_image` times
        self.index_map: List[int] = []
        for idx in range(len(self.hr_images)):
            self.index_map.extend([idx] * self.num_patches_per_image)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, i: int):
        hr_idx = self.index_map[i]
        hr_path = self.hr_images[hr_idx]
        hr = imread_rgb(hr_path)
        hr = mod_crop(hr, self.patch.scale)
        lr = downscale_bicubic(hr, self.patch.scale)
        lr_patch, hr_patch = extract_aligned_patch_pair(lr, hr, self.patch.lr_patch_size, self.patch.scale)
        if self.patch.augment:
            lr_patch, hr_patch = random_augment_pair(lr_patch, hr_patch)
        lr_t = to_tensor(lr_patch)
        hr_t = to_tensor(hr_patch)
        return {"lr": lr_t, "hr": hr_t}


class EvalImageFolder(Dataset):
    """Evaluation dataset that reads full-resolution HR images and creates matched LR images by bicubic downscale."""

    def __init__(self, hr_dir: Path | str, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.hr_paths = list_images(Path(hr_dir))
        if not self.hr_paths:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, i: int):
        hr_path = self.hr_paths[i]
        hr = imread_rgb(hr_path)
        hr = mod_crop(hr, self.scale)
        lr = downscale_bicubic(hr, self.scale)
        lr_t = to_tensor(lr)
        hr_t = to_tensor(hr)
        return {"lr": lr_t, "hr": hr_t, "name": hr_path.stem}


def create_dataloaders(
    data_root: Path | str,
    scale: int,
    batch_size: int,
    lr_patch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    patch_cfg = PatchConfig(scale=scale, lr_patch_size=lr_patch_size, augment=True)
    train_ds = DF2KDataset(root=data_root, patch=patch_cfg, num_patches_per_image=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    val_hr_dir = Path(data_root) / "DIV2K_valid_HR"
    val_ds = EvalImageFolder(val_hr_dir, scale=scale)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
