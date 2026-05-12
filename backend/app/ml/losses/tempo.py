"""
ε — Tempo consistency penalty.

The output arrangement's implied tempo must match the original.  We infer
the implied period from each onset curve via autocorrelation: the lag with
the highest autocorrelation peak (excluding lag 0) is the dominant period.
Penalise the squared difference between target-period and output-period
fractions of the total segment.

Why this matters
----------------
Even though we set the MIDI tempo parameter explicitly at render time
(Level 2 enforcement in the spec) and snap to the beat grid at post-process
(Level 3), the *generator itself* should also internalise tempo coherence
so its raw output respects the original groove without needing as much
post-processing fixup.  That's what this loss does — Level 1.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _onset_curve(roll: torch.Tensor) -> torch.Tensor:
    """
    (B, T, K) → (B, T).  Same as in rhythm.py — repeated here to keep the
    loss files independently understandable.
    """
    prev = F.pad(roll, (0, 0, 1, 0))[:, :-1, :]
    return F.relu(roll - prev).sum(dim=-1)


def _dominant_period(curve: torch.Tensor, min_lag: int = 4, max_lag: int | None = None) -> torch.Tensor:
    """
    Autocorrelation-based period detection.  Returns a soft scalar period
    (in frames) per batch element via expectation over normalised positive
    autocorrelation peaks — fully differentiable.

    min_lag avoids the lag-0 self-correlation peak.
    """
    B, T = curve.shape
    if max_lag is None:
        max_lag = T // 2

    # Mean-centre so DC doesn't dominate the autocorr.
    c = curve - curve.mean(dim=-1, keepdim=True)            # (B, T)

    # Compute autocorr by sliding inner-products.  For each candidate lag k:
    #   r[k] = sum_t c[t] * c[t+k]
    # Vectorised by shifting once per k.  T is small (~400) so a Python loop
    # over lags is fine.
    lags = torch.arange(min_lag, max_lag, device=curve.device)
    r = torch.zeros(B, len(lags), device=curve.device, dtype=curve.dtype)
    for i, k in enumerate(lags.tolist()):
        r[:, i] = (c[:, :T - k] * c[:, k:]).sum(dim=-1)

    # Soft expectation over positive autocorrelation: weights = relu(r).
    w = F.relu(r)                                            # (B, n_lags)
    den = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p = w / den
    lag_f = lags.to(curve.dtype).unsqueeze(0)                # (1, n_lags)
    return (p * lag_f).sum(dim=-1)                           # (B,)


def tempo_consistency_loss(
    roll_target: torch.Tensor,
    roll_output: torch.Tensor,
) -> torch.Tensor:
    """
    Compare implied periods of the two rolls.  Scale by 1/T_target so the
    loss is dimensionless and similar in magnitude to the other loss terms.
    """
    c_t = _onset_curve(roll_target)
    c_o = _onset_curve(roll_output)

    T = c_t.shape[-1]
    p_t = _dominant_period(c_t)
    p_o = _dominant_period(c_o)

    # Normalise so a period error of half-the-segment is a unit-magnitude loss.
    err = (p_o - p_t) / max(1.0, float(T) / 2.0)             # (B,)
    return (err * err).mean()
