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
from src.mambalitesr.config import CONFIG, TEACHER
from src.mambalitesr.data import create_dataloaders
from src.mambalitesr.model import MambaLiteSR
from src.mambalitesr.losses import FeatureDistillationLoss
from src.mambalitesr.discriminator import PatchDiscriminator
from src.mambalitesr.utils import (
    calculate_psnr, calculate_ssim, calculate_lpips, calculate_fid,
    save_comparison, save_checkpoint, load_checkpoint
)

def evaluate(model: nn.Module, 
            val_loader: DataLoader, 
            device: torch.device,
            epoch: int,
            out_dir: Path) -> Dict[str, float]:
    """Comprehensive model evaluation with metrics and visual samples"""
    model.eval()
    psnrs, ssims, lpips_scores = [], [], []
    all_sr, all_hr = [], []
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model(lr)
            
            # Calculate metrics
            psnrs.append(calculate_psnr(sr, hr))
            ssims.append(calculate_ssim(sr, hr))
            lpips_scores.append(calculate_lpips(sr, hr))
            
            # Collect samples for FID
            all_sr.append(sr.cpu())
            all_hr.append(hr.cpu())
            
            # Save visual comparisons
            if i < 3:
                save_comparison(
                    lr, sr, hr,
                    out_dir / f"compare_epoch{epoch}_sample{i}.png"
                )
    
    # Calculate FID (requires all samples)
    sr_tensor = torch.cat(all_sr, dim=0).to(device)
    hr_tensor = torch.cat(all_hr, dim=0).to(device)
    fid_score = calculate_fid(sr_tensor, hr_tensor)
    
    return {
        "psnr": np.mean(psnrs),
        "ssim": np.mean(ssims),
        "lpips": np.mean(lpips_scores),
        "fid": fid_score
    }

def main() -> None:
    # Configuration setup
    cfg = CONFIG
    tcfg = TEACHER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "logs")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_root=cfg.data_root,
        scale=cfg.scale,
        batch_size=cfg.batch_size,
        lr_patch_size=cfg.patch_size,
        num_workers=cfg.num_workers,
    )
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create student model
    student = MambaLiteSR(
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
        print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
        
        if tcfg.ckpt:
            load_checkpoint(teacher, Path(tcfg.ckpt), map_location=device)
            print(f"Loaded teacher weights from {tcfg.ckpt}")
    
    # Create discriminator (if using adversarial training)
    discriminator = None
    d_optimizer = None
    if cfg.use_gan:
        discriminator = PatchDiscriminator().to(device)
        d_optimizer = optim.Adam(
            discriminator.parameters(), 
            lr=cfg.d_lr, 
            betas=(0.5, 0.999)
        )
        print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Loss function with configurable weights
    criterion = FeatureDistillationLoss(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        temperature=cfg.temperature
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        student.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=list(cfg.milestones), 
        gamma=cfg.gamma
    )
    
    # Training state
    best_psnr = 0.0
    global_step = 0
    
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
                d_loss.backward()
                d_optimizer.step()
            
            # --- Generator Training ---
            optimizer.zero_grad(set_to_none=True)
            
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
                discriminator=discriminator
            )
            
            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "d_loss": f"{d_loss.item():.4f}" if cfg.use_gan else "N/A",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Update learning rate
        scheduler.step()
        
        # Validation and metrics
        metrics = evaluate(student, val_loader, device, epoch, out_dir)
        print(f"Epoch {epoch} | "
              f"PSNR: {metrics['psnr']:.2f} dB | "
              f"SSIM: {metrics['ssim']:.4f} | "
              f"LPIPS: {metrics['lpips']:.4f} | "
              f"FID: {metrics['fid']:.2f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('PSNR', metrics['psnr'], epoch)
        writer.add_scalar('SSIM', metrics['ssim'], epoch)
        writer.add_scalar('LPIPS', metrics['lpips'], epoch)
        writer.add_scalar('FID', metrics['fid'], epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save best checkpoint
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            save_checkpoint(
                student, 
                optimizer, 
                global_step, 
                out_dir / "best.pt"
            )
            print(f"Saved best model with PSNR: {best_psnr:.2f} dB")
        
        # Save latest checkpoint and discriminator
        save_checkpoint(
            student, 
            optimizer, 
            global_step, 
            out_dir / "last.pt"
        )
        if cfg.use_gan:
            torch.save(
                discriminator.state_dict(), 
                out_dir / f"discriminator_epoch_{epoch}.pth"
            )
    
    print("Training completed!")
    writer.close()

if __name__ == "__main__":
    main()