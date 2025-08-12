from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mambalitesr.config import CONFIG, TEACHER
from src.mambalitesr.data import create_dataloaders
from src.mambalitesr.model import MambaLiteSR
from src.mambalitesr.losses import DistillationLoss
from src.mambalitesr.utils import calculate_psnr_y, calculate_ssim_y, save_checkpoint, load_checkpoint


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs: List[float] = []
    ssims: List[float] = []
    with torch.inference_mode():
        for batch in val_loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)
            psnrs.append(calculate_psnr_y(sr, hr))
            ssims.append(calculate_ssim_y(sr, hr))
    return float(sum(psnrs) / len(psnrs)), float(sum(ssims) / len(ssims))


def main() -> None:
    cfg = CONFIG
    tcfg = TEACHER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_dataloaders(
        data_root=cfg.data_root,
        scale=cfg.scale,
        batch_size=cfg.batch_size,
        lr_patch_size=cfg.patch_size,
        num_workers=cfg.num_workers,
    )

    student = MambaLiteSR(
        scale=cfg.scale,
        embed_dim=cfg.embed_dim,
        num_rmmb=cfg.num_rmmb,
        mixers_per_block=cfg.mixers_per_block,
        low_rank=cfg.low_rank,
    ).to(device)

    teacher = None
    if cfg.use_kd:
        teacher = MambaLiteSR(
            scale=cfg.scale,
            embed_dim=tcfg.embed_dim,
            num_rmmb=tcfg.num_rmmb,
            mixers_per_block=tcfg.mixers_per_block,
            low_rank=tcfg.low_rank,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if tcfg.ckpt:
            load_checkpoint(teacher, Path(tcfg.ckpt), map_location=device)

    criterion = DistillationLoss(alpha=cfg.alpha)
    optimizer = optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.milestones), gamma=cfg.gamma)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            with torch.set_grad_enabled(True):
                if teacher is not None:
                    with torch.no_grad():
                        teacher_sr = teacher(lr)
                else:
                    teacher_sr = None
                sr = student(lr)
                loss = criterion(sr, hr, teacher_sr=teacher_sr)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()

        val_psnr, val_ssim = evaluate(student, val_loader, device)
        print(f"Validation PSNR-Y: {val_psnr:.3f} dB | SSIM-Y: {val_ssim:.4f}")
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(student, optimizer, global_step, out_dir / "best.pt")
        save_checkpoint(student, optimizer, global_step, out_dir / "last.pt")


if __name__ == "__main__":
    main()
