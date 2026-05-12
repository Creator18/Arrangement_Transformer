"""
Differentiable piano-roll → mel-spectrogram.

Why this exists
---------------
The training loop needs to push the generator's 61-dim output toward the
*frequency-cropped* mel spectrogram of the original 88-dim piano roll.  The
straightforward approach — render piano-roll → audio (via pretty-midi) →
mel-spec — is not differentiable: backprop dies at the audio synthesis step.

Solution: skip audio entirely.  Compute a mel-spectrogram directly from the
piano-roll activations by placing each active note as a Gaussian energy
contribution at its fundamental + first few harmonics on the mel grid,
modulated by a per-note attack-decay envelope.  Every operation is a tensor
op so gradients flow cleanly from spectrogram loss back to roll activations.

What this is NOT
----------------
A realistic piano synthesizer.  Inharmonicity, hammer noise, sympathetic
resonance — none of it is modelled.  But for a *training proxy* whose only
job is to give a meaningful gradient signal for "this set of pitches at this
moment should produce this energy pattern in mel space", the harmonic-stack
approach is more than sufficient and runs roughly an order of magnitude
faster than rendering through pretty-midi.

API
---
    spec = PianoRollSpectrogram(...)            # construct once, reuse
    mel  = spec(piano_roll)                     # (B, T, P) → (B, M, T)
    mel_band = crop_mel_to_band(mel, lo_hz, hi_hz)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.config import (
    N_MELS, F_MIN_HZ, F_MAX_HZ, SAMPLE_RATE,
    PIANO_LO_MIDI, PIANO_DIM, FRAME_RATE_HZ,
)

# Number of harmonic partials we synthesize per note.  More partials = a more
# realistic spectrogram, more compute.  Six is a good compromise — captures
# everything below the human-perception roll-off without doubling cost.
N_HARMONICS = 6

# Amplitude falloff per harmonic (1/k^β).  β=2 ≈ piano-like.
HARMONIC_FALLOFF = 2.0

# σ (in mel-bin units) of the Gaussian we paint around each harmonic centre.
# Slightly wider than one bin so the gradient isn't a hard step.
HARMONIC_SIGMA = 0.75

# Attack / decay (in frames) of the per-note temporal envelope.  10 frames at
# 100 Hz = 100 ms, matches a real piano onset/decay shape closely enough that
# the rhythmic envelope loss has something to grip onto.
ATTACK_FRAMES = 3
DECAY_FRAMES  = 12


def _midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # HTK mel — matches torchaudio / librosa default
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _build_harmonic_to_mel_matrix(
    keyboard_lo: int,
    keyboard_dim: int,
    n_mels: int,
    n_harmonics: int,
    f_min_hz: float,
    f_max_hz: float,
    sigma_bins: float,
) -> torch.Tensor:
    """
    Build a (n_mels, keyboard_dim) precomputed projection matrix.  Entry
    [m, k] is the total Gaussian contribution of key k's first n_harmonics
    fundamentals to mel-bin m.  Each note's full spectrum then becomes a
    single matrix-vector multiply at runtime — exactly what we need for fast
    differentiable rendering.

    Out-of-band harmonics (above f_max_hz) are silently dropped.
    """
    # Mel-bin centre frequencies, evenly spaced in mel space.
    mel_min = _hz_to_mel(torch.tensor(f_min_hz))
    mel_max = _hz_to_mel(torch.tensor(f_max_hz))
    mel_pts = torch.linspace(mel_min.item(), mel_max.item(), n_mels)

    matrix = torch.zeros(n_mels, keyboard_dim, dtype=torch.float32)

    for k in range(keyboard_dim):
        f0 = _midi_to_hz(torch.tensor(float(keyboard_lo + k)))
        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh > f_max_hz:
                break
            amp = 1.0 / (h ** HARMONIC_FALLOFF)
            mel_h = _hz_to_mel(fh)
            # Gaussian over mel bins, centred at this harmonic's mel position
            d = (mel_pts - mel_h) / sigma_bins
            matrix[:, k] += amp * torch.exp(-0.5 * d * d)

    # Normalise so the loudest column has L1 ≈ 1 — keeps spectrogram magnitudes
    # bounded regardless of n_harmonics / falloff choices.
    matrix /= matrix.sum(dim=0, keepdim=True).clamp(min=1e-8).max()
    return matrix


def _build_temporal_envelope(attack_frames: int, decay_frames: int) -> torch.Tensor:
    """
    A short (attack + decay)-frame impulse-response kernel.  Convolving the
    binary onset map with this gives the per-frame amplitude envelope of each
    note — sharp attack, soft exponential decay.
    """
    attack = torch.linspace(0.0, 1.0, attack_frames + 1)[1:]  # skip zero
    decay  = torch.exp(torch.linspace(0.0, -3.0, decay_frames))  # 1 → ~0.05
    env    = torch.cat([attack, decay], dim=0)
    return env / env.max()


class PianoRollSpectrogram(nn.Module):
    """
    Module wrapping the precomputed projection matrix + envelope kernel.

    Forward
    -------
    Input:  piano_roll   (B, T, K)  — float in [0, 1], K = keyboard_dim
    Output: mel_spec     (B, n_mels, T)
    """

    def __init__(
        self,
        keyboard_lo: int = PIANO_LO_MIDI,
        keyboard_dim: int = PIANO_DIM,
        n_mels: int = N_MELS,
        f_min_hz: float = F_MIN_HZ,
        f_max_hz: float = F_MAX_HZ,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz

        # Mel-bin centre frequencies (Hz) — saved as buffer so crop_mel_to_band
        # can use the same grid without recomputing.
        mel_min = _hz_to_mel(torch.tensor(f_min_hz))
        mel_max = _hz_to_mel(torch.tensor(f_max_hz))
        mel_pts = torch.linspace(mel_min.item(), mel_max.item(), n_mels)
        # Inverse mel → Hz
        mel_centre_hz = 700.0 * (torch.pow(10.0, mel_pts / 2595.0) - 1.0)
        self.register_buffer("mel_centre_hz", mel_centre_hz)

        # Harmonic projection: (n_mels, keyboard_dim)
        proj = _build_harmonic_to_mel_matrix(
            keyboard_lo, keyboard_dim, n_mels,
            N_HARMONICS, f_min_hz, f_max_hz, HARMONIC_SIGMA,
        )
        self.register_buffer("harmonic_proj", proj)

        # Temporal envelope kernel: 1-D conv along time
        env = _build_temporal_envelope(ATTACK_FRAMES, DECAY_FRAMES)
        # Reshape for F.conv1d: (out_ch=1, in_ch=1, kernel_len)
        self.register_buffer("envelope", env.view(1, 1, -1))

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        """
        piano_roll: (B, T, K) — soft activations in [0, 1]
        returns:    (B, n_mels, T) — non-negative mel magnitudes
        """
        B, T, K = piano_roll.shape

        # Step 1 — temporal envelope per key.  Treat each key as its own
        # 1-D signal and convolve with the attack-decay impulse response.
        # Reshape (B, T, K) → (B*K, 1, T) for grouped conv.
        sig = piano_roll.transpose(1, 2).reshape(B * K, 1, T)
        # Pad on the LEFT so the envelope only extends into the future of an
        # onset — a real piano hit decays forward, not backward.
        pad = self.envelope.shape[-1] - 1
        sig = F.pad(sig, (pad, 0))
        env_sig = F.conv1d(sig, self.envelope)        # (B*K, 1, T)
        env_sig = env_sig.view(B, K, T)               # (B, K, T)

        # Step 2 — project each frame's K-dim activation through the harmonic
        # matrix to get its mel-bin energies.
        #   (n_mels, K) @ (B, K, T) → (B, n_mels, T)
        mel = torch.einsum("mk,bkt->bmt", self.harmonic_proj, env_sig)

        # Magnitude can't be negative.  Tiny eps avoids log-of-zero downstream.
        return mel.clamp(min=1e-7)


def crop_mel_to_band(
    mel: torch.Tensor,
    mel_centre_hz: torch.Tensor,
    f_lo_hz: float,
    f_hi_hz: float,
) -> torch.Tensor:
    """
    Slice the mel-spectrogram to the frequency band that a 5-octave keyboard
    can physically produce.  This is the discriminator's "fair comparison"
    crop — we don't penalise the generator for missing bass it can't render.

    The slice is a hard boundary in bin index space (cheap, lossless).
    """
    lo_idx = int(torch.searchsorted(mel_centre_hz, torch.tensor(f_lo_hz)).item())
    hi_idx = int(torch.searchsorted(mel_centre_hz, torch.tensor(f_hi_hz)).item())
    lo_idx = max(0, min(lo_idx, mel.shape[1] - 1))
    hi_idx = max(lo_idx + 1, min(hi_idx, mel.shape[1]))
    return mel[:, lo_idx:hi_idx, :]
