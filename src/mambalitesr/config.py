from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Data
    data_root: str = "data"
    scale: int = 4
    # Training (match baseline training regime)
    epochs: int = 1000    # More epochs to match 2500 iterations
    batch_size: int = 32
    patch_size: int = 48
    num_workers: int = 4
    # Student model (target ~370K parameters to match baseline)
    embed_dim: int = 40   # Fine-tuned to match baseline parameter count
    num_rmmb: int = 6     # Balanced blocks
    mixers_per_block: int = 2
    low_rank: int = 6     # Reduced rank
    # Optimization
    lr: float = 1e-4     # Lower learning rate for stability
    weight_decay: float = 1e-4  # Add weight decay for regularization
    warmup_epochs: int = 10  # Learning rate warmup
    milestones: tuple[int, int] = (200, 400)
    scheduler_gamma: float = 0.5
    # Knowledge Distillation (enable based on baseline results)
    use_kd: bool = True   # Enable KD as baseline shows it's crucial
    alpha: float = 0.2    # Best alpha from baseline (28.71 PSNR)
    alpha_dynamic: bool = False
    # Loss weights
    beta: float = 0.0  # perceptual weight
    adv_gamma: float = 0.0  # adversarial loss weight
    temperature: float = 1.0
    # GAN options
    use_gan: bool = False
    d_lr: float = 1e-4
    # Training improvements
    early_stop_patience: int = 50  # Early stopping
    save_freq: int = 10  # Save checkpoint every N epochs
    log_freq: int = 100  # Log every N steps
    # Output (will be auto-versioned with timestamp)
    out_dir: str = "runs/latest"
    experiment_name: str = "mambalitesr"  # Base name for experiments


@dataclass(frozen=True)
class TeacherConfig:
    # Teacher model hyperparameters (larger than student for effective KD)
    embed_dim: int = 128   # 2x student size
    num_rmmb: int = 12     # 1.5x student blocks
    mixers_per_block: int = 2
    low_rank: int = 16     # Higher rank
    # Optional checkpoint to load a pretrained teacher (auto-finds latest if this path doesn't exist)
    ckpt: str | None = "runs/teacher/best.pt"  # Fallback path - will auto-find latest teacher


@dataclass(frozen=True)
class EvalConfig:
    ckpt: str = "runs/latest/best.pt"
    eval_set: str = "Set5"  # One of: Set5, Set14, BSD100, Urban100


CONFIG = TrainConfig()
TEACHER = TeacherConfig()
EVAL = EvalConfig()