"""
α — Spectral similarity via Jensen-Shannon divergence on frequency-cropped
mel spectrograms.

This is the primary "waveform mould matching" objective.  Both spectrograms
are normalised to probability distributions per frame, then we compute the
average JS divergence across the time axis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_EPS = 1e-8


def _normalise_per_frame(spec: torch.Tensor) -> torch.Tensor:
    """
    (B, M, T) → (B, M, T) where each (B, :, t) sums to 1.
    Adds a tiny floor before normalising so completely-silent frames don't
    produce NaNs in the log; their distribution becomes uniform which has
    zero gradient anyway.
    """
    spec = spec.clamp(min=_EPS)
    return spec / spec.sum(dim=1, keepdim=True)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Sum over the frequency axis; mean over time and batch.
    return (p * (torch.log(p + _EPS) - torch.log(q + _EPS))).sum(dim=1)


def spectral_js_loss(
    mel_target: torch.Tensor,
    mel_output: torch.Tensor,
) -> torch.Tensor:
    """
    mel_target: (B, M, T) — cropped mel of the ORIGINAL roll (frequency band
                            that the 5-octave keyboard can produce)
    mel_output: (B, M, T) — cropped mel of the GENERATED roll (same band)

    Both must already be cropped to the same frequency range.  Returns a
    scalar mean JS divergence.
    """
    if mel_target.shape != mel_output.shape:
        raise ValueError(
            f"spectral_js_loss: shape mismatch — target {mel_target.shape}, output {mel_output.shape}"
        )

    p = _normalise_per_frame(mel_target)
    q = _normalise_per_frame(mel_output)
    m = 0.5 * (p + q)
    js = 0.5 * (_kl(p, m) + _kl(q, m))     # (B, T)
    return js.mean()
