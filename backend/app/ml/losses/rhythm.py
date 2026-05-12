"""
β — Rhythmic-envelope similarity.

We don't care that the same exact pitches play at each beat — we care that
the *density of onsets per beat* matches.  Strong beats should fall in the
same places.  Compute a per-frame onset map (positive activation derivative),
downsample to beat resolution, and compare via MSE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _onset_curve(roll: torch.Tensor) -> torch.Tensor:
    """
    (B, T, K) → (B, T) onset intensity per frame = sum over keys of the
    positive part of the temporal derivative.  Differentiable via ReLU.
    """
    # Prepend a zero frame so the first frame can have onsets too.
    prev = F.pad(roll, (0, 0, 1, 0))[:, :-1, :]            # (B, T, K)
    rising = F.relu(roll - prev)                            # positive deltas
    return rising.sum(dim=-1)                               # (B, T)


def _downsample(curve: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Average-pool a (B, T) curve down to (B, n_bins).  Uses adaptive_avg_pool1d
    so any T → any n_bins works without manual stride math.
    """
    B, T = curve.shape
    if n_bins <= 0 or T <= 0:
        return curve.new_zeros(B, max(1, n_bins))
    x = curve.unsqueeze(1)                                  # (B, 1, T)
    return F.adaptive_avg_pool1d(x, n_bins).squeeze(1)      # (B, n_bins)


def rhythmic_envelope_loss(
    roll_target: torch.Tensor,
    roll_output: torch.Tensor,
    beats_per_segment: int = 16,
) -> torch.Tensor:
    """
    roll_target: (B, T, K_a) — original piano roll
    roll_output: (B, T, K_b) — generated piano roll (K differs from target,
                               but onsets are summed over keys so shape
                               mismatch is fine)

    Returns a scalar MSE between beat-resolution onset envelopes.
    """
    c_t = _onset_curve(roll_target)
    c_o = _onset_curve(roll_output)
    d_t = _downsample(c_t, beats_per_segment)
    d_o = _downsample(c_o, beats_per_segment)
    return F.mse_loss(d_o, d_t)
