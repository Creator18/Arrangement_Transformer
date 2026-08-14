"""
Inference adapter — the single function that the FastAPI /transform route
will call once a trained checkpoint exists.

Signature is deliberately note-list-in / note-list-out (the same shape the
rest of the backend deals with), so swapping the ML path in for the
existing rule-based path is a single import change in routers/transform.py.

Contract
--------
    notes_in  = [{pitch, onset, offset, velocity, ...}, ...]   (88-key range)
    notes_out = [{pitch, onset, offset, velocity, ...}, ...]   (61-key range)

The model itself works in piano-roll space.  This adapter handles the two
conversions (notes ↔ roll), conditioning extraction, and post-processing
hand-off back to the existing rule layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.ml.config import (
    PIANO_LO_MIDI, PIANO_DIM, TARGET_LO_MIDI, TARGET_DIM,
    FRAME_RATE_HZ, SPARSITY_BINARIZATION_THRESHOLD,
    GENERATOR, PATHS, USE_ML_PIPELINE,
)


# ── Notes ↔ piano roll ────────────────────────────────────────────────────────

def notes_to_roll(
    notes:     List[Dict],
    duration:  float,
    midi_lo:   int = PIANO_LO_MIDI,
    n_keys:    int = PIANO_DIM,
    frame_hz:  int = FRAME_RATE_HZ,
) -> np.ndarray:
    """(T, K) float32 in {0, 1}."""
    T = max(1, int(np.ceil(duration * frame_hz)))
    roll = np.zeros((T, n_keys), dtype=np.float32)
    for n in notes:
        p = int(n["pitch"]) - midi_lo
        if p < 0 or p >= n_keys:
            continue
        a = max(0, int(float(n["onset"])  * frame_hz))
        b = min(T, int(float(n.get("offset", n["onset"] + 0.25)) * frame_hz))
        if b > a:
            roll[a:b, p] = 1.0
    return roll


def roll_to_notes(
    roll:      np.ndarray,
    midi_lo:   int = TARGET_LO_MIDI,
    frame_hz:  int = FRAME_RATE_HZ,
    velocity:  int = 80,
) -> List[Dict]:
    """
    Run-length-encode the boolean roll back into a note list.  Adjacent
    frames of the same pitch coalesce into a single note event.
    """
    out: List[Dict] = []
    T, K = roll.shape
    for k in range(K):
        col = roll[:, k] > 0.5
        if not col.any():
            continue
        # Find runs of True
        diff = np.diff(col.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            out.append({
                "pitch":    midi_lo + k,
                "onset":    round(s / frame_hz, 4),
                "offset":   round(e / frame_hz, 4),
                "velocity": int(velocity),
            })
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    return out


# ── Conditioning ──────────────────────────────────────────────────────────────

def build_conditioning(
    chord_map:  Dict[int, Tuple[int, set]],
    tempo:      float,
    duration:   float,
    frame_hz:   int = FRAME_RATE_HZ,
    cond_dim:   int = GENERATOR.cond_dim,
) -> np.ndarray:
    """
    Build a (T, cond_dim) conditioning matrix from the chord map produced
    by segmentation.

    Layout: first 12 dims = active-chord chroma (binary), next 12 dims =
    chord-quality one-hot (placeholder = root pitch class for now).  This
    matches GeneratorConfig.cond_dim = 24.
    """
    T = max(1, int(np.ceil(duration * frame_hz)))
    cond = np.zeros((T, cond_dim), dtype=np.float32)

    if not chord_map:
        return cond

    measure_dur = (60.0 / max(tempo, 1.0)) * 4.0          # 4/4 assumed
    frames_per_measure = max(1, int(round(measure_dur * frame_hz)))

    for m_idx, (root_pc, chord_pcs) in chord_map.items():
        f0 = m_idx * frames_per_measure
        f1 = min(T, f0 + frames_per_measure)
        if f1 <= f0:
            continue
        # First 12 dims: which pitch classes are in the chord
        for pc in chord_pcs:
            cond[f0:f1, pc % 12] = 1.0
        # Next 12 dims: one-hot of the root
        cond[f0:f1, 12 + (root_pc % 12)] = 1.0

    return cond


# ── Scale/chord-conditioned noise (training prior; zeros at inference) ────────

def conditioned_noise(
    cond: np.ndarray,
    scale_mask: Optional[np.ndarray] = None,
    scale_sigma: float = GENERATOR.noise_scale,
) -> np.ndarray:
    """
    Generate (T, 88) noise biased toward pitches in the active chord/scale.
    At inference we typically pass zeros to make generation deterministic;
    this helper is here mainly so training and inference share the same
    interface.
    """
    T = cond.shape[0]
    noise = np.random.randn(T, PIANO_DIM).astype(np.float32) * scale_sigma
    # Damp pitches whose pitch class isn't in the active chord
    for t in range(T):
        chord_mask = cond[t, :12]                                   # (12,)
        for k in range(PIANO_DIM):
            pc = ((PIANO_LO_MIDI + k) % 12)
            if chord_mask[pc] < 0.5:
                noise[t, k] *= 0.25
    if scale_mask is not None:
        noise *= scale_mask
    return noise


# ── Public entrypoint ─────────────────────────────────────────────────────────

def model_available(ckpt_path: Path = PATHS.default_ckpt) -> bool:
    """Whether the /transform route can route to the ML pipeline."""
    return USE_ML_PIPELINE and Path(ckpt_path).exists()


def transform_with_model(
    notes:     List[Dict],
    chord_map: Dict[int, Tuple[int, set]],
    tempo:     float,
    duration:  float,
    ckpt_path: Path = PATHS.default_ckpt,
    device:    Optional[str] = None,
) -> List[Dict]:
    """
    Run a trained ArrangementTransformer over the input notes and return the
    transformed note list (61-key range).

    The lazy import of torch + the model lives inside this function so the
    server doesn't pay PyTorch import time at startup when ML is disabled.
    """
    import torch
    from app.ml.models.generator import ArrangementTransformer
    from app.ml.training.checkpoint import load_checkpoint

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. Notes → roll
    roll_np = notes_to_roll(notes, duration)                          # (T, 88)
    cond_np = build_conditioning(chord_map, tempo, duration)          # (T, C)
    noise_np = np.zeros_like(roll_np)                                 # deterministic infer

    # 2. To tensor
    roll  = torch.from_numpy(roll_np).unsqueeze(0).to(device)         # (1, T, 88)
    cond  = torch.from_numpy(cond_np).unsqueeze(0).to(device)         # (1, T, C)
    noise = torch.from_numpy(noise_np).unsqueeze(0).to(device)        # (1, T, 88)

    # 3. Load + run model
    model = ArrangementTransformer().to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    # 3b. Run in windows the positional encoding can actually address.
    #
    # pos_enc.pe is (1, max_len, d) and the model indexes it as [:, :T, :], so a
    # piece longer than max_len raised RuntimeError — measured, 28,146 steps
    # against 4,096.  At ~100 steps/s that cap is about 41 SECONDS, so this was
    # never a note-count limit: essentially every real piece hit it, the caller
    # caught the exception, and /transform silently returned the rule-based
    # fallback instead.  A solo-piano upload has been getting method="fallback"
    # rather than the ML arrangement it reports.
    #
    # Windows overlap and only their CENTRES are kept.  Attention has no
    # context beyond the window edge, so the first and last moments of a chunk
    # are the model's worst output; butt-joining chunks would stitch precisely
    # those together, once per chunk, forever.  Discarding a margin on each
    # side means every kept step had full context on both sides.
    max_len = int(getattr(getattr(model, "pos_enc", None), "pe",
                          torch.zeros(1, 1 << 30, 1)).shape[1])
    T = roll.shape[1]

    def _run(a: int, b: int):
        return model(roll[:, a:b], cond[:, a:b], noise[:, a:b])

    with torch.no_grad():
        if T <= max_len:
            out_soft = _run(0, T)                                      # unchanged
        else:
            margin = min(max_len // 8, 256)                            # ~2.5 s
            stride = max_len - 2 * margin
            parts, pos = [], 0
            while pos < T:
                a = min(pos, max(0, T - max_len))
                b = min(a + max_len, T)
                chunk = _run(a, b)
                lo = pos - a                       # skip the leading margin
                hi = min(lo + stride, b - a)       # ...and the trailing one
                parts.append(chunk[:, lo:hi])
                pos = a + hi
            out_soft = torch.cat(parts, dim=1)
            print(f"[ml] windowed inference: {T} steps in "
                  f"{len(parts)} windows of {max_len} (margin {margin})")
        out_hard = (out_soft >= SPARSITY_BINARIZATION_THRESHOLD).float()

    # 4. Roll → notes
    out_np = out_hard.squeeze(0).cpu().numpy()                         # (T, 61)
    return roll_to_notes(out_np)
