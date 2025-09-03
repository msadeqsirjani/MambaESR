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
from src.losses import FeatureDistillationLoss
from src.discriminator import PatchDiscriminator
from src.utils import (
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    save_checkpoint,
    load_checkpoint,
)
from src.experiment_manager import (
    ExperimentManager,
    find_latest_teacher_checkpoint,
)


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

            # Calculate metrics
            psnrs.append(calculate_psnr(sr, hr))
            ssims.append(calculate_ssim(sr, hr))
            lpips_scores.append(calculate_lpips(sr, hr))

    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims),
        "lpips": np.mean(lpips_scores),
    }


def main() -> None:
    # Configuration setup
    cfg = CONFIG
    tcfg = TEACHER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    print(f"Using device: {device}")

    # Create experiment manager for versioned runs
    experiment = ExperimentManager("student")
    out_dir = experiment.get_experiment_dir()

    # Save experiment configuration
    config_dict = {
        "model": {
            "embed_dim": cfg.embed_dim,
            "num_rmmb": cfg.num_rmmb,
            "mixers_per_block": cfg.mixers_per_block,
            "low_rank": cfg.low_rank,
            "scale": cfg.scale,
        },
        "training": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "patch_size": cfg.patch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "use_kd": cfg.use_kd,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "adv_gamma": cfg.adv_gamma,
        },
    }
    experiment.save_config(config_dict)

    writer = SummaryWriter(out_dir / "logs")

    # Create dataloaders
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

    # Create student model
    student = MambaESR(
        scale=cfg.scale,
        embed_dim=cfg.embed_dim,
        num_rmmb=cfg.num_rmmb,
        mixers_per_block=cfg.mixers_per_block,
        low_rank=cfg.low_rank,
    ).to(device)
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Create teacher model (if using knowledge distillation)
    teacher = None
    if cfg.use_kd:
        teacher = MambaESR(
            scale=cfg.scale,
            embed_dim=tcfg.embed_dim,
            num_rmmb=tcfg.num_rmmb,
            mixers_per_block=tcfg.mixers_per_block,
            low_rank=tcfg.low_rank,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

        # Auto-find latest teacher checkpoint
        teacher_checkpoint = find_latest_teacher_checkpoint()

        if teacher_checkpoint:
            load_checkpoint(teacher, teacher_checkpoint, map_location=device)
            print(f"✅ Loaded teacher weights from {teacher_checkpoint}")
        elif tcfg.ckpt and Path(tcfg.ckpt).exists():
            load_checkpoint(teacher, Path(tcfg.ckpt), map_location=device)
            print(f"✅ Loaded teacher weights from {tcfg.ckpt}")
        else:
            print(
                "⚠️  No teacher checkpoint found - training with random teacher weights"
            )
            print("   This will significantly reduce performance!")

    # Create discriminator (if using adversarial training)
    discriminator = None
    d_optimizer = None
    if cfg.use_gan:
        discriminator = PatchDiscriminator().to(device)
        d_optimizer = optim.Adam(
            discriminator.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999)
        )
        print(
            f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}"
        )

    # Loss function with configurable weights
    criterion = FeatureDistillationLoss(
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.adv_gamma, temperature=cfg.temperature
    )

    # Optimizer and scheduler with warmup
    optimizer = optim.Adam(
        student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
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
        student.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            # --- Discriminator Training ---
            d_loss = 0.0
            if cfg.use_gan:
                d_optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    # Real images
                    real_pred = discriminator(hr)
                    real_loss = F.binary_cross_entropy_with_logits(
                        real_pred, torch.ones_like(real_pred)
                    )
                    # Fake images (with student detach)
                    with torch.no_grad():
                        student_sr = student(lr)
                    fake_pred = discriminator(student_sr.detach())
                    fake_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, torch.zeros_like(fake_pred)
                    )
                    d_loss = (real_loss + fake_loss) * 0.5
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)

            # --- Generator Training ---
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # Forward passes with feature extraction
                teacher_features = None
                if teacher is not None:
                    with torch.no_grad():
                        _, teacher_features = teacher(lr, return_features=True)
                student_sr, student_features = student(lr, return_features=True)
                # Calculate loss
                loss = criterion(
                    student_sr,
                    hr,
                    teacher_features=teacher_features,
                    student_features=student_features,
                    discriminator=discriminator,
                )
            # Backpropagation (mixed precision)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update progress
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "d_loss": f"{d_loss.item():.4f}" if cfg.use_gan else "N/A",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # Update learning rate
        scheduler.step()

        # Validation and metrics
        metrics = evaluate(student, val_loader, device, epoch, out_dir)
        print(
            f"Epoch {epoch} | "
            f"PSNR: {metrics['psnr']:.2f} dB | "
            f"SSIM: {metrics['ssim']:.4f} | "
            f"LPIPS: {metrics['lpips']:.4f}"
        )

        # Log metrics to experiment tracker
        experiment.log_training_metric(
            epoch,
            {
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "train_loss": epoch_loss / len(train_loader),
                "learning_rate": scheduler.get_last_lr()[0],
            },
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
            save_checkpoint(student, optimizer, global_step, out_dir / "best.pt")
            print(f"Saved best model with PSNR: {best_psnr:.2f} dB")
            # Log checkpoint path
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
                # Log final training results
                total_params = sum(p.numel() for p in student.parameters())
                experiment.log_final_training_results(
                    best_psnr, epoch, total_params, early_stopped=True
                )
                break

        # Always save latest
        save_checkpoint(student, optimizer, global_step, out_dir / "latest.pt")
        # Log latest checkpoint path
        writer.add_text("checkpoint/latest", str(out_dir / "latest.pt"), epoch)
        experiment.log_training_metric(
            epoch, {"latest_checkpoint": str(out_dir / "latest.pt")}
        )

        if cfg.use_gan:
            torch.save(
                discriminator.state_dict(), out_dir / f"discriminator_epoch_{epoch}.pth"
            )

    # Log final training results if not early stopped
    if epochs_without_improvement < cfg.early_stop_patience:
        total_params = sum(p.numel() for p in student.parameters())
        experiment.log_final_training_results(
            best_psnr, cfg.epochs, total_params, early_stopped=False
        )

    print("Training completed!")
    print(f"📊 Metrics saved to: {experiment.metrics_file}")
    print(f"📁 Experiment directory: {out_dir}")
    writer.close()


if __name__ == "__main__":
    main()
