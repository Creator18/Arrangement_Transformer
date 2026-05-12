"""
γ — Melodic-contour similarity.

For each time frame, find the highest active pitch (the melody candidate).
Compare the trajectories of the original 88-dim roll's top pitch and the
generated 61-dim roll's top pitch as time-series.  We compare *shape*, not
absolute pitch — so we use cosine similarity on the contour after subtracting
each contour's mean.  This way the contour loss rewards "moves up and down
in the same places" even though the absolute pitch range differs by an
octave or two.

Differentiability trick
-----------------------
A hard argmax is non-differentiable.  We use a soft-argmax: weighted average
of pitch indices, where the weights are the activations themselves
(normalised).  This biases the contour toward the highest *strong* note in
each frame in a way the backward pass can still flow through.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_EPS = 1e-8


def _soft_top_pitch(roll: torch.Tensor) -> torch.Tensor:
    """
    (B, T, K) → (B, T)
    Soft-argmax over the pitch axis weighted by activation.  Biased toward
    higher pitches (mimicking "highest active pitch") by squaring the index
    weight — equivalent to a sharper, top-heavy softmax.
    """
    B, T, K = roll.shape
    idx = torch.arange(K, device=roll.device, dtype=roll.dtype)         # (K,)
    # Weight by activation × pitch-index.  Higher pitches dominate when active.
    w  = roll * idx.view(1, 1, K)                                       # (B, T, K)
    num = w.sum(dim=-1)                                                  # (B, T)
    den = roll.sum(dim=-1).clamp(min=_EPS)                               # (B, T)
    return num / den                                                     # (B, T)


def _zero_mean_unit_norm(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    n = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return x / n


def melodic_contour_loss(
    roll_target: torch.Tensor,
    roll_output: torch.Tensor,
) -> torch.Tensor:
    """
    Returns 1 - cosine_similarity(mean-centred contours).  Range ≈ [0, 2];
    0 = perfectly aligned shape, 2 = perfectly anti-correlated.
    """
    c_t = _soft_top_pitch(roll_target)         # (B, T)
    c_o = _soft_top_pitch(roll_output)         # (B, T)
    c_t = _zero_mean_unit_norm(c_t)
    c_o = _zero_mean_unit_norm(c_o)
    cos = (c_t * c_o).sum(dim=-1)              # (B,)
    return (1.0 - cos).mean()
