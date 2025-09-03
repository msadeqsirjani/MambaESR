#!/usr/bin/env python3
"""
Train a teacher model (larger, no knowledge distillation)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import CONFIG, TEACHER
from src.data import create_dataloaders
from src.model import MambaESR
from src.utils import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    save_checkpoint,
)
from src.experiment_manager import ExperimentManager


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    out_dir: Path,
) -> Dict[str, float]:
    """Comprehensive model evaluation with metrics and visual samples"""
    model.eval()
    psnrs, ssims, lpips_scores = [], [], []

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)

            # Calculate metrics (skip LPIPS due to SSL issues)
            psnrs.append(calculate_psnr(sr, hr))
            ssims.append(calculate_ssim(sr, hr))
            lpips_scores.append(calculate_lpips(sr, hr))

    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims),
        "lpips": np.mean(lpips_scores),
    }


def main() -> None:
    cfg = CONFIG
    tcfg = TEACHER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    print(f"Using device: {device}")
    print("Training TEACHER model (large, no KD)")

    # Create experiment manager for teacher
    experiment = ExperimentManager("teacher")
    out_dir = experiment.get_experiment_dir()

    # Save teacher configuration
    config_dict = {
        "model": {
            "embed_dim": tcfg.embed_dim,
            "num_rmmb": tcfg.num_rmmb,
            "mixers_per_block": tcfg.mixers_per_block,
            "low_rank": tcfg.low_rank,
            "scale": cfg.scale,
        },
        "training": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "patch_size": cfg.patch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "loss_type": "L1",  # Teacher uses simple L1 loss
        },
    }
    experiment.save_config(config_dict)

    writer = SummaryWriter(out_dir / "logs")

    # Create dataloaders with original batch size (simplified selective scan allows this)
    train_loader, val_loader = create_dataloaders(
        data_root=cfg.data_root,
        scale=cfg.scale,
        batch_size=cfg.batch_size,
        lr_patch_size=cfg.patch_size,
        num_workers=cfg.num_workers,
    )
    print(
        f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}"
    )

    # Create teacher model (large)
    teacher = MambaESR(
        scale=cfg.scale,
        embed_dim=tcfg.embed_dim,
        num_rmmb=tcfg.num_rmmb,
        mixers_per_block=tcfg.mixers_per_block,
        low_rank=tcfg.low_rank,
        use_gradient_checkpointing=False,  # Disabled for speed
    ).to(device)
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    # Simple L1 loss for teacher (no KD)
    criterion = nn.L1Loss()

    # Optimizer and scheduler with warmup
    optimizer = optim.Adam(
        teacher.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
    )

    # Main scheduler
    main_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.milestones), gamma=cfg.scheduler_gamma
    )

    # Combined scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    # Training state
    best_psnr = float("-inf")
    global_step = 0
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        teacher.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                sr = teacher(lr)
                loss = criterion(sr, hr)

            # Backpropagation (mixed precision)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update progress
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # Update learning rate
        scheduler.step()

        # Validation and metrics
        metrics = evaluate(teacher, val_loader, device, epoch, out_dir)
        print(
            f"Teacher Epoch {epoch} | "
            f"PSNR: {metrics['psnr']:.2f} dB | "
            f"SSIM: {metrics['ssim']:.4f} | "
            f"LPIPS: {metrics['lpips']:.4f}"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train", epoch_loss / len(train_loader), epoch)
        writer.add_scalar("PSNR", metrics["psnr"], epoch)
        writer.add_scalar("SSIM", metrics["ssim"], epoch)
        writer.add_scalar("LPIPS", metrics["lpips"], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best checkpoint and early stopping
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            epochs_without_improvement = 0
            save_checkpoint(teacher, optimizer, global_step, out_dir / "best.pt")
            print(f"Saved best teacher model with PSNR: {best_psnr:.2f} dB")
            writer.add_text("checkpoint/best", str(out_dir / "best.pt"), epoch)
            experiment.log_training_metric(
                epoch, {"best_checkpoint": str(out_dir / "best.pt")}
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_patience:
                print(
                    f"Early stopping after {epoch} epochs (no improvement for {cfg.early_stop_patience} epochs)"
                )
                break

        # Always save latest
        save_checkpoint(teacher, optimizer, global_step, out_dir / "latest.pt")
        writer.add_text("checkpoint/latest", str(out_dir / "latest.pt"), epoch)

    print("Teacher training completed!")
    print(f"Best teacher PSNR: {best_psnr:.2f} dB")
    print(f"Teacher checkpoint saved to: {out_dir / 'best.pt'}")
    writer.close()


if __name__ == "__main__":
    main()
