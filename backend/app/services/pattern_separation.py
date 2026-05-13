"""
REPET-SIM pattern / melody separation.

The original REPET (Rafii 2012) used a single repeating period plus a per-bin
median template across the whole piece.  That broke on real piano music
because:
  • there's rarely one dominant period — figurations + harmonic loops + form
    are all repeating at different timescales
  • the median assumed exact repetitions, blurring music that VARIES
    its pattern across repeats (e.g. a left-hand ostinato with a slightly
    different chord each measure)

REPET-SIM (Rafii 2013) replaces both choices with a non-parametric one:
for each frame t, find its K most spectrally-similar OTHER frames anywhere
in the spectrogram, then take their median as t's "repeating template".
Frames that are part of a pattern have many similar neighbours; frames
that are unique (the melody) don't, so their template is weak.

Subtract the template from the magnitude spectrogram, apply soft Wiener
masks to the original complex STFT, ISTFT back to audio.

Pipeline (called from separation.py on the upper-register bandpass mix):
    1. STFT magnitude V              (F, T)
    2. Cosine similarity between every pair of frames → S (T, T)
    3. For each t, pick the K=20 most-similar frames (excluding self and a
       small temporal exclusion window around t — otherwise immediate
       neighbours dominate the median)
    4. Template[:, t] = median over those K frames' magnitudes
    5. Clamp template by V (template can't exceed actual signal)
    6. masks = template / (template + nonrepeating)
    7. Apply masks to complex STFT, ISTFT back → (pattern_audio, melody_audio)

The pair-similarity matrix is O(T²) memory but practical for our scale —
at hop_length=512 and sr=22050, a 3-minute song is ~7700 frames, so the
similarity matrix is ~225 MB of float32.  We compute it in chunks if T is
too large to materialise at once.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import librosa


_DEFAULT_N_FFT       = 2048
_DEFAULT_HOP         = 512
_DEFAULT_NEIGHBOURS  = 20      # K — number of similar frames to median
_DEFAULT_EXCL_FRAMES = 30      # temporal exclusion window around self (~0.7 s)
_MAX_T_IN_MEMORY     = 12_000  # ~5 minutes at default hop; chunks above this


def _frame_similarities(V: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every pair of frames.  V is (F, T); result is
    (T, T) with values in [0, 1] (V is non-negative magnitude → no negatives).
    """
    F, T = V.shape
    norms = np.linalg.norm(V, axis=0).reshape(1, T) + 1e-8
    Vn = V / norms                                 # column-normalised
    return (Vn.T @ Vn).astype(np.float32)           # (T, T)


def _build_template_dense(
    V: np.ndarray,
    n_neighbours: int,
    excl_frames: int,
) -> np.ndarray:
    """
    Build the repeating template using the dense similarity matrix.  Works
    for T up to ~12k frames before memory becomes a concern.
    """
    F, T = V.shape
    S = _frame_similarities(V)        # (T, T)

    # Mask out a temporal exclusion window around each frame (and the self-cell)
    # so the "similar" frames aren't just the immediate neighbours, which would
    # collapse the template to a local average.
    idx = np.arange(T)
    for d in range(-excl_frames, excl_frames + 1):
        rows = idx[max(0, -d):T - max(0, d)]
        cols = rows + d
        S[rows, cols] = -1.0

    template = np.zeros_like(V)
    k = min(n_neighbours, T - 2 * excl_frames - 1)
    if k <= 0:
        return template

    # For each frame t, pick the top-k most-similar frames and median them.
    # argpartition is faster than full sort and gives the same top-k set.
    for t in range(T):
        top_idx = np.argpartition(-S[t], k - 1)[:k]
        template[:, t] = np.median(V[:, top_idx], axis=1)
    return template


def _build_template_chunked(
    V: np.ndarray,
    n_neighbours: int,
    excl_frames: int,
    chunk: int = 2_000,
) -> np.ndarray:
    """
    Chunked variant for very long pieces — compute similarities for one row-
    chunk at a time so we never need the full (T, T) matrix in memory.
    """
    F, T = V.shape
    norms = np.linalg.norm(V, axis=0) + 1e-8
    Vn = V / norms.reshape(1, T)
    template = np.zeros_like(V)
    k = min(n_neighbours, T - 2 * excl_frames - 1)
    if k <= 0:
        return template

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        # Similarities between every frame in [start, end) and ALL T frames
        S = (Vn[:, start:end].T @ Vn).astype(np.float32)    # (end-start, T)
        for local_t, global_t in enumerate(range(start, end)):
            row = S[local_t]
            lo = max(0, global_t - excl_frames)
            hi = min(T, global_t + excl_frames + 1)
            row[lo:hi] = -1.0
            top_idx = np.argpartition(-row, k - 1)[:k]
            template[:, global_t] = np.median(V[:, top_idx], axis=1)
    return template


def repet_sim_separate(
    audio:        np.ndarray,
    sr:           int,
    beat_times:   Optional[np.ndarray] = None,    # accepted for API compat
    n_fft:        int = _DEFAULT_N_FFT,
    hop_length:   int = _DEFAULT_HOP,
    n_neighbours: int = _DEFAULT_NEIGHBOURS,
    excl_frames:  int = _DEFAULT_EXCL_FRAMES,
    return_mask:  bool = False,
):
    """
    Split `audio` into (pattern_audio, melody_audio) via REPET-SIM.

    Both outputs are 1-D float32 of the same length as the input.  Energy is
    approximately preserved (pattern + melody ≈ original) by the Wiener-style
    masking.  Falls through to (silent, input) when the audio is shorter than
    the temporal exclusion window — there's no useful similarity structure to
    extract from a couple of seconds.

    When `return_mask=True`, also returns a per-frame `pattern_strength` array
    in [0, 1] — the mean of mask_pat across frequency bins for each frame.
    Downstream callers (transcription.py) use this to label notes as
    'bass_pattern' vs 'bass_melody' without re-transcribing the bass.

    Returns:
        (pattern_audio, melody_audio)                                if not return_mask
        (pattern_audio, melody_audio, pattern_strength, frame_times) if return_mask
    """
    if audio.ndim != 1:
        audio = np.mean(audio, axis=0) if audio.ndim == 2 else audio.flatten()
    audio = audio.astype(np.float32, copy=False)

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    V = np.abs(D).astype(np.float32)
    F, T = V.shape

    if T < 2 * excl_frames + n_neighbours + 1:
        # Too short to find meaningful similar frames — pass everything through
        # as melody.
        if return_mask:
            ft = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)
            return np.zeros_like(audio), audio.copy(), np.zeros(T, dtype=np.float32), ft
        return np.zeros_like(audio), audio.copy()

    if T <= _MAX_T_IN_MEMORY:
        template = _build_template_dense(V, n_neighbours, excl_frames)
    else:
        template = _build_template_chunked(V, n_neighbours, excl_frames)

    # Template can never exceed the actual signal at any (f, t) cell.
    template = np.minimum(template, V)

    # Soft Wiener masks.  We use clipping rather than full normalisation to
    # avoid divide-by-zero on near-silent frames; small eps in denominator.
    eps = 1e-8
    nonrep   = np.maximum(V - template, 0.0)
    denom    = template + nonrep + eps
    mask_pat = template / denom
    mask_mel = nonrep    / denom

    audio_pattern = librosa.istft(D * mask_pat, hop_length=hop_length, length=audio.size)
    audio_melody  = librosa.istft(D * mask_mel, hop_length=hop_length, length=audio.size)

    if return_mask:
        # Energy-weighted average of the pattern mask across frequency.  Plain
        # mean-over-F would over-weight silent HF bins; weighting by V puts
        # the "pattern strength" where the energy actually lives.
        weights = V.sum(axis=0) + 1e-8                       # (T,)
        pattern_strength = (mask_pat * V).sum(axis=0) / weights
        pattern_strength = pattern_strength.astype(np.float32)
        frame_times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)
        return (
            audio_pattern.astype(np.float32),
            audio_melody.astype(np.float32),
            pattern_strength,
            frame_times.astype(np.float32),
        )

    return audio_pattern.astype(np.float32), audio_melody.astype(np.float32)


# ── Public alias kept for backwards-compat with the old REPET callsite ──────
def repet_separate(*args, **kwargs):
    """Alias — old code may import this name.  Forwards to REPET-SIM."""
    return repet_sim_separate(*args, **kwargs)
