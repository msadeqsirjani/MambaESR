from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Data
    data_root: str = "data"
    scale: int = 4
    # Training
    epochs: int = 500
    batch_size: int = 128
    patch_size: int = 64
    num_workers: int = 4
    # Student model
    embed_dim: int = 32
    num_rmmb: int = 4
    mixers_per_block: int = 2  # 4 RMMBs x 2 mixers = 8 mixers total
    low_rank: int = 4
    # Optimization
    lr: float = 2e-4
    weight_decay: float = 0.0
    milestones: tuple[int, int] = (200, 400)
    gamma: float = 0.5
    # Knowledge Distillation
    use_kd: bool = False            # set True to enable KD with teacher
    alpha: float = 0.8              # fixed KD alpha (teacher vs GT)
    alpha_dynamic: bool = False     # placeholder for adaptive alpha
    # Distillation/Perceptual/Adversarial loss weights
    beta: float = 0.1               # perceptual weight
    temperature: float = 1.0        # KD temperature
    # GAN options
    use_gan: bool = False
    d_lr: float = 1e-4
    # Output
    out_dir: str = "runs/latest"


@dataclass(frozen=True)
class TeacherConfig:
    # Teacher model hyperparameters (can differ from student)
    embed_dim: int = 64
    num_rmmb: int = 8
    mixers_per_block: int = 2
    low_rank: int = 4
    # Optional checkpoint to load a pretrained teacher
    ckpt: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    ckpt: str = "runs/latest/best.pt"
    eval_set: str = "Set5"  # One of: Set5, Set14, BSD100, Urban100


CONFIG = TrainConfig()
TEACHER = TeacherConfig()
EVAL = EvalConfig()
