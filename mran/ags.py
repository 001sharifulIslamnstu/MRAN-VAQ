from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def compute_ags(attn: Tensor, mask: Tensor, lam: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Compute AGS and AGS+.

    Args:
        attn: (B, P) attention distribution over P patches/regions (sum=1 along P).
        mask: (B, P) binary mask, 1 for relevant, 0 for irrelevant.
        lam: trade-off between relevant/irrelevant regions for AGS+.

    Returns:
        ags_raw: (B,)  relevance coverage
        ags_plus: (B,) AGS+ penalizing attention on irrelevant areas
    """
    eps = 1e-8
    attn = F.normalize(attn, p=1, dim=-1)  # ensure it's a distribution

    relevant_mass = (attn * mask).sum(dim=-1)  # (B,)
    irrelevant_mass = (attn * (1.0 - mask)).sum(dim=-1)

    ags_raw = relevant_mass
    ags_plus = relevant_mass - lam * irrelevant_mass
    return ags_raw, ags_plus


class AGSLoss(nn.Module):
    """
    AGS loss: encourages attention to focus on relevant regions.

    Loss = -mean(AGS+)  (maximize AGS+).
    """

    def __init__(self, lam: float = 0.5):
        super().__init__()
        self.lam = lam

    def forward(self, attn: Tensor, mask: Tensor) -> Tensor:
        ags_raw, ags_plus = compute_ags(attn, mask, lam=self.lam)
        loss = -ags_plus.mean()
        return loss
