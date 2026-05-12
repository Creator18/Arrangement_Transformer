"""
δ — Sparsity penalty.

Without this, the generator quickly learns to activate many keys
simultaneously to approximate the target spectrogram by brute force.  The
output becomes muddy and physically unplayable.

We penalise the *expected* simultaneous key count exceeding MAX_SIMULTANEOUS
(≈ two human hands).  Using a soft hinge (ReLU on excess) gives gradient
only above the threshold, so the model is free to play 2 or 9 notes at once
without penalty but pays a quadratic cost above the limit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from app.ml.config import MAX_SIMULTANEOUS_KEYS


def sparsity_loss(
    roll_output: torch.Tensor,
    max_simultaneous: int = MAX_SIMULTANEOUS_KEYS,
) -> torch.Tensor:
    """
    roll_output: (B, T, K) — soft activations in [0, 1]

    Returns scalar mean-squared excess simultaneous activation.
    """
    # Soft count = sum of activations per frame.  For binary output this is
    # literally the number of pressed keys.  For sigmoid output it's the
    # expected number.
    soft_count = roll_output.sum(dim=-1)                                # (B, T)
    excess = F.relu(soft_count - float(max_simultaneous))               # (B, T)
    return (excess * excess).mean()
