"""
Scale/chord-conditioned noise generator.

This is the regularising prior described in the architecture spec: instead
of zero-initialising the generator's stochastic input channel, we feed it
Gaussian noise that is BIASED toward pitches inside the current
chord/scale.  At training time this nudges the model toward exploring
musically-plausible candidate notes instead of arbitrary pitches; at
inference time we typically zero the noise out for determinism.

Inputs
------
    cond:   (T, cond_dim) — output of conditioning.build_conditioning_from_roll
    sigma:  noise scale  (default from GeneratorConfig.noise_scale)
    in_scale_boost: multiplier for pitches IN the local chord chroma

Output
------
    noise:  (T, 88) float32
"""
from __future__ import annotations

import numpy as np

from app.ml.config import PIANO_LO_MIDI, PIANO_DIM, GENERATOR


# Pitches outside the local chord get noise scaled DOWN by this factor.
_OUT_OF_CHORD_DAMP = 0.25


def conditioned_noise(
    cond:           np.ndarray,
    sigma:          float = GENERATOR.noise_scale,
    in_scale_boost: float = 1.0,
    out_damp:       float = _OUT_OF_CHORD_DAMP,
    rng:            np.random.Generator | None = None,
) -> np.ndarray:
    """
    Build (T, 88) noise damped toward pitches in cond[:, :12] (chord chroma).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = cond.shape[0]
    base = rng.standard_normal((T, PIANO_DIM)).astype(np.float32) * sigma

    # Precompute per-pitch pitch-class index (0–11) for each of the 88 keys
    pc_per_key = np.array([(PIANO_LO_MIDI + k) % 12 for k in range(PIANO_DIM)],
                          dtype=np.int64)                          # (88,)

    chord_mask = cond[:, :12]                                       # (T, 12)
    # For each key, look up the chord-strength at its pitch class:
    key_chord = chord_mask[:, pc_per_key]                           # (T, 88)

    # Scale: in_scale_boost where in chord, out_damp where not.  Smooth
    # interpolation by linear blend on the chroma value.
    scale = out_damp + (in_scale_boost - out_damp) * key_chord
    return base * scale


def zero_noise(T: int) -> np.ndarray:
    """Convenience: deterministic inference noise."""
    return np.zeros((T, PIANO_DIM), dtype=np.float32)
