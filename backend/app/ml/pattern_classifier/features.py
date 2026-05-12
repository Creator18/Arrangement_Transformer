"""
Symbolic per-note feature extraction.

Every feature here is computable from a note list and the piece's tempo /
time-signature alone — no audio access required.  Designed so that
inference at /transform time uses the same code path as training.

Feature layout (FEATURE_DIM = 99)
---------------------------------
    [0:88]  pitch one-hot                         MIDI 21..108
    [88]    velocity / 127                        0..1
    [89]    log_e(duration_s) normalised          ≈ -1..1
    [90]    sin(beat phase)                       position within current beat
    [91]    cos(beat phase)
    [92]    sin(measure phase)                    position within current measure
    [93]    cos(measure phase)
    [94]    local density / 10                    # simultaneously-active notes
    [95]    recurrence count / 10                 # other notes with same
                                                    (pitch, beat-phase-in-measure)
    [96]    pitch rank in simultaneous chord       0=lowest, 1=highest
    [97]    backward step / 12                    semitone distance from prev
                                                    onset at near-same time
    [98]    forward step / 12                     same toward next onset
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


FEATURE_DIM = 99
_PIANO_LO   = 21
_PIANO_DIM  = 88


def _phase_pair(phase: float) -> tuple[float, float]:
    """(sin, cos) of a 0..1 phase scaled to one full revolution."""
    a = 2.0 * math.pi * (phase % 1.0)
    return math.sin(a), math.cos(a)


def note_features(
    notes:           List[Dict],
    tempo:           float,
    beats_per_bar:   int = 4,
) -> np.ndarray:
    """
    Compute the (N, FEATURE_DIM) feature matrix for a list of N notes.

    `notes` must be in the project's standard dict shape
    ({pitch, onset, offset, velocity, ...}).  Order is preserved — output
    row `i` corresponds to input note `i`.
    """
    if not notes:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    beat_dur     = 60.0 / max(tempo, 1.0)
    measure_dur  = beat_dur * max(beats_per_bar, 1)

    onsets   = np.array([float(n["onset"])                                 for n in notes], dtype=np.float64)
    offsets  = np.array([float(n.get("offset", float(n["onset"]) + 0.25)) for n in notes], dtype=np.float64)
    pitches  = np.array([int(n["pitch"])                                   for n in notes], dtype=np.int64)
    velocs   = np.array([int(n.get("velocity", 80))                        for n in notes], dtype=np.int64)
    durs     = np.clip(offsets - onsets, 1e-3, 8.0)

    # Beat / measure phases
    beat_phase    = (onsets % beat_dur)    / beat_dur     if beat_dur    > 0 else onsets * 0
    measure_phase = (onsets % measure_dur) / measure_dur  if measure_dur > 0 else onsets * 0

    # ── Local density: count of overlapping notes per onset ───────────────
    # For each note i, density = number of other notes j whose interval
    # [onset_j, offset_j) contains onset_i.
    density = np.zeros(len(notes), dtype=np.float32)
    # Naive O(N²) — fine for typical 88-key transcriptions (a few thousand notes).
    for i in range(len(notes)):
        t = onsets[i]
        density[i] = float(((onsets <= t) & (offsets > t)).sum() - 1)  # exclude self

    # ── Pitch rank within the chord active at this note's onset ───────────
    # 0 = lowest of the chord, 1 = highest.  Defined for chords with >1 note;
    # singletons get 0.5.
    rank = np.full(len(notes), 0.5, dtype=np.float32)
    for i in range(len(notes)):
        t = onsets[i]
        chord_pitches = pitches[(onsets <= t) & (offsets > t)]
        if chord_pitches.size <= 1:
            continue
        lo, hi = chord_pitches.min(), chord_pitches.max()
        if hi > lo:
            rank[i] = (pitches[i] - lo) / (hi - lo)

    # ── Recurrence count ──────────────────────────────────────────────────
    # How many *other* notes share this note's (pitch, beat-phase-in-measure)
    # rounded to a coarse grid.  This is the headline signal for "is this
    # part of a pattern?" — accompaniment notes recur, melodies usually don't.
    # We quantise the phase to 1/8-beat to absorb timing jitter.
    measure_slot = np.round(measure_phase * (beats_per_bar * 8)).astype(np.int64)
    recurrence   = np.zeros(len(notes), dtype=np.float32)
    # Build a {(pitch, slot): count} table once, then look up per note
    counts: Dict[tuple, int] = {}
    keys = list(zip(pitches.tolist(), measure_slot.tolist()))
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    for i, k in enumerate(keys):
        recurrence[i] = counts[k] - 1   # exclude self

    # ── Stepwise distances forward / backward in time ─────────────────────
    # For each note, find the nearest other note within ±60 ms in time.  The
    # semitone distance to that neighbour is the "stepwise" feature — small
    # values suggest melodic motion, large jumps suggest distinct voices.
    step_b = np.zeros(len(notes), dtype=np.float32)
    step_f = np.zeros(len(notes), dtype=np.float32)
    sort_idx = np.argsort(onsets)
    sorted_onsets  = onsets[sort_idx]
    sorted_pitches = pitches[sort_idx]
    inv_idx = np.argsort(sort_idx)
    for i_sorted in range(len(notes)):
        i = sort_idx[i_sorted]
        # Look at the immediate temporal neighbours in sorted order
        if i_sorted > 0:
            j = sort_idx[i_sorted - 1]
            if abs(onsets[i] - onsets[j]) < 0.5:
                step_b[i] = min(12, abs(int(pitches[i]) - int(pitches[j])))
        if i_sorted < len(notes) - 1:
            j = sort_idx[i_sorted + 1]
            if abs(onsets[j] - onsets[i]) < 0.5:
                step_f[i] = min(12, abs(int(pitches[i]) - int(pitches[j])))

    # ── Assemble feature matrix ───────────────────────────────────────────
    X = np.zeros((len(notes), FEATURE_DIM), dtype=np.float32)

    # pitch one-hot
    pitch_idx = np.clip(pitches - _PIANO_LO, 0, _PIANO_DIM - 1)
    X[np.arange(len(notes)), pitch_idx] = 1.0

    X[:, 88] = velocs / 127.0
    X[:, 89] = np.log(durs) / 3.0
    X[:, 90] = np.sin(2.0 * np.pi * beat_phase)
    X[:, 91] = np.cos(2.0 * np.pi * beat_phase)
    X[:, 92] = np.sin(2.0 * np.pi * measure_phase)
    X[:, 93] = np.cos(2.0 * np.pi * measure_phase)
    X[:, 94] = np.clip(density / 10.0, 0.0, 1.0)
    X[:, 95] = np.clip(recurrence / 10.0, 0.0, 1.0)
    X[:, 96] = rank
    X[:, 97] = step_b / 12.0
    X[:, 98] = step_f / 12.0

    return X
