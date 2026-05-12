"""
Heuristic label generation for pattern-classifier training.

Goal: produce, for each note in a MAESTRO piece, a 0/1 ground truth
("pattern" vs "melody") that we can train the small MLP to predict from
the per-note feature vector.

The labeling logic uses a combination of signals that map well onto the
musical intuition for the distinction:

    A note is most likely "pattern" if:
        • its (pitch, position-in-measure) signature recurs many times
          across the piece
        • it's NOT the highest pitch in its local chord at most onsets
        • its velocity is unremarkable

    A note is most likely "melody" if:
        • it doesn't recur (unique pitch+phase combination)
        • it's the highest pitch in its local chord
        • it stands out by velocity

We threshold a weighted combination of these signals to produce a
binary label.  Notes whose combined score is ambiguous (the middle band)
are returned with label = -1 so the dataset can skip them during
training — better to learn from confident examples than to fight against
noisy ambiguous ones.

`generate_labels(notes, tempo, beats_per_bar) -> np.ndarray[int8]`
    -1 = ambiguous (drop from training)
     0 = melody
     1 = pattern
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def generate_labels(
    notes:           List[Dict],
    tempo:           float,
    beats_per_bar:   int = 4,
    confident_only:  bool = True,
) -> np.ndarray:
    """
    Returns an int8 array of length len(notes) with values in {-1, 0, 1}.

    -1 = ambiguous (training datasets should mask these out)
     0 = melody
     1 = pattern
    """
    if not notes:
        return np.zeros(0, dtype=np.int8)

    beat_dur    = 60.0 / max(tempo, 1.0)
    measure_dur = beat_dur * max(beats_per_bar, 1)

    onsets  = np.array([float(n["onset"])                                 for n in notes], dtype=np.float64)
    offsets = np.array([float(n.get("offset", float(n["onset"]) + 0.25)) for n in notes], dtype=np.float64)
    pitches = np.array([int(n["pitch"])                                   for n in notes], dtype=np.int64)
    velocs  = np.array([int(n.get("velocity", 80))                        for n in notes], dtype=np.int64)
    N = len(notes)

    measure_phase = (onsets % measure_dur) / measure_dur if measure_dur > 0 else onsets * 0
    measure_slot  = np.round(measure_phase * (beats_per_bar * 8)).astype(np.int64)

    # ── Signal 1: recurrence count ────────────────────────────────────────
    counts: Dict[tuple, int] = {}
    keys = list(zip(pitches.tolist(), measure_slot.tolist()))
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    recurrence = np.array([counts[k] for k in keys], dtype=np.float32)

    # ── Signal 2: how often was this pitch the local-chord top? ───────────
    # For each note compute (top-fraction over all of its sustaining frames).
    # Approximation: at each note's onset, check if it's the highest in
    # simultaneously-sounding chord.  A note that's *frequently* the top of
    # local chords tends to carry melodic weight.
    is_top_at_onset = np.zeros(N, dtype=np.float32)
    for i in range(N):
        t = onsets[i]
        chord = pitches[(onsets <= t) & (offsets > t)]
        if chord.size and pitches[i] >= chord.max():
            is_top_at_onset[i] = 1.0

    # Aggregate "top-of-chord-ness" by pitch over the piece — a pitch that's
    # usually the top is more melody-like; one that's usually buried is more
    # pattern-like.
    per_pitch_top: Dict[int, list] = {}
    for i in range(N):
        per_pitch_top.setdefault(int(pitches[i]), []).append(is_top_at_onset[i])
    pitch_top_fraction = {p: float(np.mean(v)) for p, v in per_pitch_top.items()}

    # ── Signal 3: relative velocity (z-score within piece) ────────────────
    vel_mean = velocs.mean() if velocs.size else 80.0
    vel_std  = velocs.std()  if velocs.size else 1.0
    vel_z    = ((velocs - vel_mean) / max(vel_std, 1e-6)) if vel_std > 0 else np.zeros(N)

    # ── Combine into a single score, then threshold ────────────────────────
    # High score  → pattern (recurrent, buried, unremarkable velocity)
    # Low  score  → melody  (unique, top-of-chord, accented)
    rec_score   =  np.clip(recurrence / 5.0, 0.0, 2.0) - 0.5       # >=3 recurrences → positive
    top_score   = -np.array([pitch_top_fraction[int(p)] for p in pitches])
    vel_score   = -np.clip(vel_z, -2.0, 2.0) * 0.25                 # loud notes pull toward melody

    combined = rec_score + top_score + vel_score                    # higher = more pattern-like

    if confident_only:
        labels = np.full(N, -1, dtype=np.int8)
        labels[combined >  0.6] = 1     # confidently pattern
        labels[combined < -0.4] = 0     # confidently melody
        return labels
    else:
        return (combined > 0.0).astype(np.int8)
