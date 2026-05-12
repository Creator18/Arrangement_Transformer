"""
MAESTRO-driven dataset for pattern-classifier training.

Each MAESTRO MIDI yields (features, label) pairs — one per note that the
heuristic label generator classified confidently.  Ambiguous notes are
dropped (label = -1 from generate_labels) so training only sees clean
examples.

Pre-computed features for the full MAESTRO collection fit in well under
100 MB, so we just keep them in memory rather than caching to disk like
the piano-roll renderer does.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import IterableDataset

from app.ml.config       import PATHS
from app.ml.data.maestro import list_maestro

from .features import note_features, FEATURE_DIM
from .labels   import generate_labels


def _midi_notes(midi_path: Path) -> tuple[list, float]:
    """
    Load MAESTRO MIDI and return (note_list, tempo_bpm).

    MAESTRO performances are recorded human plays — they contain dense
    tempo-change events from the rubato.  `tempi[0]` is often a 0 placeholder.
    Estimating a single robust tempo via PrettyMIDI works much better.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    if pm.instruments:
        inst = pm.instruments[0]
        for n in inst.notes:
            notes.append({
                "pitch":    int(n.pitch),
                "onset":    float(n.start),
                "offset":   float(n.end),
                "velocity": int(n.velocity),
            })
    try:
        tempo = float(pm.estimate_tempo())
    except Exception:
        tempo = 120.0
    if not (30.0 <= tempo <= 250.0):     # sanity clamp
        tempo = 120.0
    return notes, tempo


def build_dataset(
    root:           Path = PATHS.maestro_root,
    split:          str  = "train",
    max_pieces:     int  = 0,   # 0 = all
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk MAESTRO, extract features + labels, return concatenated arrays.

    Returns:
        X  (N, FEATURE_DIM) float32   — all confidently-labeled notes
        y  (N,)             float32   — 0 or 1
    """
    entries = list_maestro(root, split=split)
    if max_pieces:
        entries = entries[:max_pieces]

    feats:  list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for i, entry in enumerate(entries):
        try:
            notes, tempo = _midi_notes(entry.midi_path(root))
        except Exception:
            continue
        if not notes:
            continue

        y = generate_labels(notes, tempo)
        keep = y >= 0
        if not keep.any():
            continue

        X = note_features(notes, tempo)
        feats.append(X[keep])
        labels.append(y[keep].astype(np.float32))

        if (i + 1) % 50 == 0:
            n_so_far = sum(f.shape[0] for f in feats)
            print(f"  [{i+1}/{len(entries)}] pieces processed — {n_so_far} labeled notes")

    if not feats:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


class TensorBatchedDataset(IterableDataset):
    """
    Simple wrapper around pre-built (X, y) arrays — yields random shuffles
    in batches.  Stays in memory; for our scale (a few million notes max)
    that's the cheapest option.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 256, seed: int = 1729):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        while True:
            idx = rng.permutation(len(self.X))
            for start in range(0, len(self.X), self.batch_size):
                sl = idx[start : start + self.batch_size]
                yield (torch.from_numpy(self.X[sl]),
                       torch.from_numpy(self.y[sl]))
