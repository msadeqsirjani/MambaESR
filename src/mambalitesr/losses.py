from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DistillationConfig:
    alpha: float = 0.8  # weight on teacher; (1-alpha) on ground truth


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = float(alpha)
        self.l1 = nn.L1Loss()

    def forward(self, student_sr: torch.Tensor, gt_hr: torch.Tensor, teacher_sr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if teacher_sr is None:
            return self.l1(student_sr, gt_hr)
        kd = self.l1(student_sr, teacher_sr.detach())
        sup = self.l1(student_sr, gt_hr)
        return self.alpha * kd + (1.0 - self.alpha) * sup
