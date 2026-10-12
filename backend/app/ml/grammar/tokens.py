"""Piano notes <-> tokens: the "language" the grammar model reads and writes.

Time is in SECONDS, not beats, because every note source this model will sit
behind (the spectral ears, transcribed covers, audio-aligned arrangements) is
timed in seconds, and a beat grid would have to be inferred first — the very
step that snapped trills onto 16ths earlier.  Events, in onset order:

  SHIFT_k    advance time by k x 16 ms (k = 1..64, so up to ~1 s per token;
             longer gaps chain shifts).  16 ms = one spectral frame.
  PITCH_p    a note on key p (MIDI 21..108)
  DUR_d      its length, 32 log-spaced bins from 16 ms to 8 s
  VEL_v      its loudness, 8 bins

A note is PITCH DUR VEL; simultaneous notes follow each other with no shift,
lowest first.  ~200 tokens in all.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

STEP = 0.016
N_SHIFT, N_DUR, N_VEL = 64, 32, 8
PAD, BOS, EOS, MASK = 0, 1, 2, 3
SHIFT0 = 4
PITCH0 = SHIFT0 + N_SHIFT          # 68
DUR0 = PITCH0 + 88                 # 156
VEL0 = DUR0 + N_DUR                # 188
VOCAB = VEL0 + N_VEL               # 196
DUR_EDGES = np.geomspace(0.016, 8.0, N_DUR + 1)


def dur_bin(d: float) -> int:
    return int(np.clip(np.searchsorted(DUR_EDGES, d) - 1, 0, N_DUR - 1))


def dur_value(b: int) -> float:
    return float(np.sqrt(DUR_EDGES[b] * DUR_EDGES[b + 1]))


def encode(notes: np.ndarray) -> List[int]:
    """notes: [onset, offset, pitch, velocity] rows (seconds) -> token ids (no BOS/EOS)."""
    if len(notes) == 0:
        return []
    n = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    out, t = [], 0
    for on, off, p, v in n:
        p = int(p)
        if not 21 <= p <= 108:
            continue
        k = int(round(on / STEP))
        while k > t:
            s = min(N_SHIFT, k - t)
            out.append(SHIFT0 + s - 1); t += s
        out += [PITCH0 + p - 21, DUR0 + dur_bin(max(off - on, STEP)),
                VEL0 + int(np.clip(v, 1, 127)) * N_VEL // 128]
    return out


def decode(tokens: Sequence[int]) -> np.ndarray:
    """token ids -> [onset, offset, pitch, velocity] rows.  Tolerates malformed runs."""
    rows, t, i, tok = [], 0, 0, list(tokens)
    while i < len(tok):
        x = tok[i]
        if SHIFT0 <= x < PITCH0:
            t += x - SHIFT0 + 1; i += 1
        elif PITCH0 <= x < DUR0:
            p = x - PITCH0 + 21
            d = dur_value(tok[i + 1] - DUR0) if i + 1 < len(tok) and DUR0 <= tok[i + 1] < VEL0 else 0.25
            v = ((tok[i + 2] - VEL0) * 128 // N_VEL + 8) if i + 2 < len(tok) and VEL0 <= tok[i + 2] < VOCAB else 80
            rows.append((t * STEP, t * STEP + d, p, v)); i += 3
        else:
            i += 1
    return np.array(rows, float).reshape(-1, 4)


def corrupt(notes: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """The wrong notes a transcriber makes, applied to clean notes.

    Mirrors the error breakdown of the spectral ears: semitone and octave
    slips, extra notes, missing notes (soft ones first), timing jitter.  The
    model learns to put the correct note back.
    """
    n = notes.copy()
    if len(n) == 0:
        return n
    s = strength
    keep = rng.random(len(n)) > 0.06 * s * (1.5 - n[:, 3] / 127)        # soft notes vanish first
    n = n[keep]
    if len(n) == 0:
        return n
    r = rng.random(len(n))
    n[r < 0.05 * s, 2] += rng.choice([-1, 1], (r < 0.05 * s).sum())                       # semitone slip
    oc = (r >= 0.05 * s) & (r < 0.08 * s)
    n[oc, 2] += rng.choice([-12, 12], oc.sum())                                              # octave slip
    jit = rng.random(len(n)) < 0.15 * s
    n[jit, :2] += rng.normal(0, 0.03, (jit.sum(), 1))                                       # timing
    m = int(rng.binomial(len(n), 0.05 * s))                                                  # extra notes
    if m:
        src = n[rng.integers(0, len(n), m)].copy()
        src[:, 2] += rng.choice([-7, -5, -3, 3, 4, 7, 12, -12], m)
        src[:, :2] += rng.normal(0, 0.05, (m, 1))
        n = np.concatenate([n, src])
    n[:, 2] = np.clip(n[:, 2], 21, 108)
    n[:, 0] = np.maximum(n[:, 0], 0); n[:, 1] = np.maximum(n[:, 1], n[:, 0] + STEP)
    return n
