from __future__ import annotations
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import create_dataloaders
from .model import MambaLiteSR
from .losses import DistillationLoss
from .utils import calculate_psnr_y, calculate_ssim_y, save_checkpoint, load_checkpoint

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
    # Hardcoded configuration parameters
    DATA_ROOT = "./data/DIV2K_train_HR" 
    SCALE = 4
    BATCH_SIZE = 16
    PATCH_SIZE = 64
    NUM_WORKERS = 4
    USE_KD = True
    TEACHER_CKPT = "./output/teacher/checkpoint.pt"  # Update if using KD
    EMBED_DIM = 64
    NUM_RMMB = 8
    MIXERS_PER_BLOCK = 4
    LOW_RANK = 32
    ALPHA = 0.5  # Distillation loss weight
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    MILESTONES = [200, 400, 600]
    GAMMA = 0.5
    EPOCHS = 1000
    OUT_DIR = "./output/runs/experiment1" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_root=DATA_ROOT,
        scale=SCALE,
        batch_size=BATCH_SIZE,
        lr_patch_size=PATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    
    # Create student model
    student = MambaLiteSR(
        scale=SCALE,
        embed_dim=EMBED_DIM,
        num_rmmb=NUM_RMMB,
        mixers_per_block=MIXERS_PER_BLOCK,
        low_rank=LOW_RANK,
    ).to(device)
    
    # Create teacher model (if using knowledge distillation)
    teacher = None
    if USE_KD:
        teacher = MambaLiteSR(
            scale=SCALE,
            embed_dim=EMBED_DIM * 2,  # Teacher typically larger
            num_rmmb=NUM_RMMB * 2,
            mixers_per_block=MIXERS_PER_BLOCK,
            low_rank=LOW_RANK,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if TEACHER_CKPT:
            load_checkpoint(teacher, Path(TEACHER_CKPT), map_location=device)
    
    # Loss, optimizer, and scheduler
    criterion = DistillationLoss(alpha=ALPHA)
    optimizer = optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(MILESTONES), gamma=GAMMA
    )
    
    # Output directory setup
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_psnr = 0.0
    global_step = 0
    
    for epoch in range(1, EPOCHS + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            
            with torch.set_grad_enabled(True):
                # Teacher forward (if using distillation)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_sr = teacher(lr)
                else:
                    teacher_sr = None
                
                # Student forward
                sr = student(lr)
                
                # Calculate loss
                loss = criterion(sr, hr, teacher_sr=teacher_sr)
                
                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
            
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        val_psnr, val_ssim = evaluate(student, val_loader, device)
        print(f"Validation PSNR-Y: {val_psnr:.3f} dB | SSIM-Y: {val_ssim:.4f}")
        
        # Save checkpoints
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                student, optimizer, global_step, 
                out_dir / "best.pt"
            )
        save_checkpoint(
            student, optimizer, global_step, 
            out_dir / "last.pt"
        )

if __name__ == "__main__":
    main()