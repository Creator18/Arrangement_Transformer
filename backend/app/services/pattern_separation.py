"""
REPET-style pattern / melody separation.

After Demucs has stripped drums and vocals and we've bandpass-split the
harmonic content into a bass register and an upper register, we still want
to know which notes in the upper register belong to a *repeating
accompaniment pattern* and which belong to a *non-repeating melodic line*.
The original NMF pipeline did this via per-component periodicity scoring
on activation rows of H.  That's gone now.

REPET (Rafii & Pardo, ICASSP 2012) operationalises the same idea without
needing component decomposition: things that repeat at a known period
average out to a stable template in the spectrogram; things that don't
repeat average toward zero.  Subtracting the template from the full
spectrogram yields the non-repeating melody; the template itself is the
repeating accompaniment.

Pipeline (for the upper-register audio only):
    1. STFT magnitude V (F, T)
    2. Choose a repeating period — we use 4 beats (one measure) at madmom's
       median inter-beat interval when beat_times is available, otherwise
       fall back to the strongest autocorrelation peak in the energy
       envelope.
    3. Reshape V into blocks of `period` frames, take the per-frequency-bin
       median across blocks → repeating template.
    4. Tile the template back to (F, T), clamp by V (template can't exceed
       the actual signal), Wiener-mask, ISTFT back to audio.

The output preserves total energy (`pattern + melody ≈ upper`) and works
on any sample rate.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import librosa


_DEFAULT_N_FFT = 2048
_DEFAULT_HOP   = 512
_MIN_BLOCKS    = 2          # need at least 2 repetitions to define a template


def _estimate_period_frames(
    audio:       np.ndarray,
    sr:          int,
    hop_length:  int,
    beat_times:  Optional[np.ndarray],
    n_fft:       int,
) -> int:
    """
    Period of the repeating pattern in STFT frames.

    Preference order:
      1. madmom's inter-beat intervals × 4 (one measure of 4 beats)
      2. autocorrelation peak in the spectral energy envelope
      3. fixed fallback (~2 s, the typical measure length at 120 BPM)
    """
    # 1. Beat-driven period
    if beat_times is not None and len(beat_times) >= 5:
        intervals = np.diff(np.asarray(beat_times, dtype=float))
        if intervals.size:
            median_beat_s = float(np.median(intervals))
            period_s = 4.0 * median_beat_s
            return max(1, int(round(period_s * sr / hop_length)))

    # 2. Autocorrelation fallback
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    V = np.abs(D)
    env = V.sum(axis=0)
    if env.size > 50:
        e = env - env.mean()
        ac = np.correlate(e, e, mode="full")
        ac = ac[ac.size // 2:]
        # Search lag range corresponding to 0.5–8 seconds
        lo = max(1, int(0.5 * sr / hop_length))
        hi = min(ac.size, int(8.0 * sr / hop_length))
        if hi > lo:
            peak = int(np.argmax(ac[lo:hi])) + lo
            return max(1, peak)

    # 3. Fixed fallback — 2 seconds
    return max(1, int(round(2.0 * sr / hop_length)))


def repet_separate(
    audio:      np.ndarray,
    sr:         int,
    beat_times: Optional[np.ndarray] = None,
    n_fft:      int = _DEFAULT_N_FFT,
    hop_length: int = _DEFAULT_HOP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split `audio` into (pattern_audio, melody_audio) via REPET.

    Both outputs are 1-D float32 arrays of the same length as the input.
    On very short inputs (less than two periods) we degenerate gracefully —
    pattern is silent, melody = input — so downstream transcription still
    runs without special-casing.
    """
    if audio.ndim != 1:
        audio = np.mean(audio, axis=0) if audio.ndim == 2 else audio.flatten()
    audio = audio.astype(np.float32, copy=False)

    # STFT (complex) and magnitude
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    V = np.abs(D)
    F, T = V.shape

    period = _estimate_period_frames(audio, sr, hop_length, beat_times, n_fft)
    period = min(period, max(1, T // _MIN_BLOCKS))    # need at least 2 blocks

    n_blocks = T // period
    if n_blocks < _MIN_BLOCKS:
        # Not enough material to detect a repeating template; output everything as melody
        empty = np.zeros_like(audio, dtype=np.float32)
        return empty, audio.copy()

    # Build the repeating template: per-frequency-bin median across blocks
    blocks = V[:, : n_blocks * period].reshape(F, n_blocks, period)
    blocks = blocks.transpose(1, 0, 2)                # (n_blocks, F, period)
    template = np.median(blocks, axis=0)              # (F, period)

    # Tile template across the full T
    n_tiles = int(np.ceil(T / period))
    rep_full = np.tile(template, (1, n_tiles))[:, :T] # (F, T)

    # Template can't be louder than the actual signal in any (f, t) cell
    rep_full = np.minimum(rep_full, V)

    # Soft Wiener masks — preserves energy: pattern + melody ≈ original
    eps = 1e-8
    nonrep = V - rep_full
    denom  = rep_full + nonrep + eps
    mask_rep   = rep_full / denom
    mask_melody = nonrep   / denom

    # Apply masks to the complex STFT and ISTFT back to audio
    audio_pattern = librosa.istft(D * mask_rep,
                                  hop_length=hop_length,
                                  length=audio.size)
    audio_melody  = librosa.istft(D * mask_melody,
                                  hop_length=hop_length,
                                  length=audio.size)

    return audio_pattern.astype(np.float32), audio_melody.astype(np.float32)
