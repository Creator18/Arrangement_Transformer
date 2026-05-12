"""
Pattern classifier model.

A small MLP (~10 K params) that consumes per-note symbolic features and
emits a single logit for P(pattern).  Deliberately tiny — the features
already encode the heavy lifting (recurrence count, pitch rank, etc.);
the model just learns to combine them with thresholds tuned on real
piano data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_DIM


class PatternClassifier(nn.Module):
    """
    Input:  (B, FEATURE_DIM)
    Output: (B,) logits — apply sigmoid for P(pattern)
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
