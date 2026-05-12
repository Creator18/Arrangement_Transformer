"""
Conditioning extraction — build a (T, cond_dim) matrix the generator can use
to anchor its output to the local harmony of the piece.

Two approaches were considered:
    1. Run a chord-fitting template match (like services/segmentation.py) per
       measure, then broadcast.
    2. Compute a smoothed chroma vector + root pitch class directly from the
       piano roll.

We use (2).  Reasons:
    • The generator can learn to interpret raw chroma — we don't need to
      bake in a chord vocabulary.
    • Template matching introduces its own errors that propagate as noisy
      labels into training.  Raw chroma is a cleaner signal.
    • At inference time the same function works whether chord information
      came from a MAESTRO MIDI or from audio chroma — one less code path.

Output layout (matches GeneratorConfig.cond_dim = 24):
    dims  0–11 = smoothed pitch-class chroma in [0, 1]
    dims 12–23 = one-hot of the dominant pitch class (estimated root)
"""
from __future__ import annotations

import numpy as np

from app.ml.config import PIANO_LO_MIDI, PIANO_DIM, GENERATOR, FRAME_RATE_HZ


# Half-window in frames for chroma smoothing — the chroma at frame t is the
# average over [t-W, t+W].  W ≈ a quarter-note at 120 BPM at 100 Hz = 12 frames.
_SMOOTH_HALF_WINDOW = 12

# Bass bias: weight low piano keys more heavily when estimating the root.
# Real chord roots almost always live in the bass register.
_BASS_BIAS_CUTOFF = 60   # MIDI C4
_BASS_BIAS_WEIGHT = 2.0


def _roll_to_chroma(roll: np.ndarray) -> np.ndarray:
    """
    (T, 88) → (T, 12) raw chroma — sum each frame's activations into the 12
    pitch-class bins, mod 12.
    """
    T = roll.shape[0]
    chroma = np.zeros((T, 12), dtype=np.float32)
    for k in range(PIANO_DIM):
        pc = (PIANO_LO_MIDI + k) % 12
        chroma[:, pc] += roll[:, k]
    return chroma


def _smooth_chroma(chroma: np.ndarray, half_window: int) -> np.ndarray:
    """
    Box-filter smoothing along time.  Implemented with cumulative sums so
    cost is O(T) regardless of window size.
    """
    T = chroma.shape[0]
    if half_window <= 0 or T <= 1:
        return chroma
    # Cumulative sum trick: smooth[t] = (csum[t+W+1] - csum[t-W]) / (W*2+1)
    csum = np.concatenate([
        np.zeros((1, 12), dtype=chroma.dtype),
        np.cumsum(chroma, axis=0),
    ], axis=0)                                                # (T+1, 12)
    out = np.zeros_like(chroma)
    for t in range(T):
        lo = max(0,     t - half_window)
        hi = min(T,     t + half_window + 1)
        out[t] = (csum[hi] - csum[lo]) / max(1, hi - lo)
    return out


def _bass_biased_root(roll: np.ndarray) -> np.ndarray:
    """
    Estimate the bass-biased root pitch class per frame.  Same as chroma but
    bass keys count more.  Returns (T, 12).
    """
    T = roll.shape[0]
    chroma = np.zeros((T, 12), dtype=np.float32)
    for k in range(PIANO_DIM):
        midi = PIANO_LO_MIDI + k
        pc = midi % 12
        w = _BASS_BIAS_WEIGHT if midi < _BASS_BIAS_CUTOFF else 1.0
        chroma[:, pc] += roll[:, k] * w
    return chroma


def build_conditioning_from_roll(
    roll: np.ndarray,
    cond_dim: int = GENERATOR.cond_dim,
    smooth_half_window: int = _SMOOTH_HALF_WINDOW,
) -> np.ndarray:
    """
    roll: (T, 88) float
    returns (T, cond_dim) float32

    First 12 dims = smoothed chroma normalised to max 1 per frame; pitch
    classes that ARE in the local harmony stay bright, others go dark.
    Next 12 dims = one-hot of the bass-biased root per frame.

    If cond_dim is something other than 24 (someone tweaked GeneratorConfig),
    we pad/truncate gracefully and warn the caller via an explicit zero-fill.
    """
    chroma = _roll_to_chroma(roll)
    chroma = _smooth_chroma(chroma, smooth_half_window)
    # Per-frame max-normalise to [0, 1] (silent frames stay all-zero)
    maxes = chroma.max(axis=-1, keepdims=True)
    chroma_n = np.where(maxes > 1e-6, chroma / maxes, 0.0)

    root_chroma = _bass_biased_root(roll)
    root_chroma = _smooth_chroma(root_chroma, smooth_half_window)
    # One-hot of the strongest bass-biased pitch class
    root_idx = root_chroma.argmax(axis=-1)                      # (T,)
    root_oh  = np.zeros_like(chroma_n)                          # (T, 12)
    root_oh[np.arange(len(root_idx)), root_idx] = 1.0
    # Suppress one-hot on silent frames
    root_oh = root_oh * (maxes > 1e-6).astype(np.float32)

    out = np.zeros((roll.shape[0], cond_dim), dtype=np.float32)
    out[:, : min(12, cond_dim)]                        = chroma_n[:, : min(12, cond_dim)]
    if cond_dim > 12:
        out[:, 12 : min(24, cond_dim)]                 = root_oh[:, : min(12, cond_dim - 12)]
    return out
