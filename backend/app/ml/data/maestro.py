"""
MAESTRO v3.0.0 MIDI loader.

MAESTRO ships as a CSV manifest plus ~1300 MIDI files (audio not needed for
us — training is entirely MIDI-driven).  This module exposes:

    list_maestro(root)        — read the CSV, return one MaestroEntry per row
    load_piano_roll(midi)     — pretty_midi → (T, 88) float roll
    sample_segment(roll, ...) — crop a random T-frame window with valid notes

The dataset wrapper in `dataset.py` consumes these.  Keeping the raw I/O and
the iterator logic separate makes both unit-testable and lets future work
(e.g. caching pre-rendered rolls to disk) plug in at the right seam.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from app.ml.config import PIANO_LO_MIDI, PIANO_DIM, FRAME_RATE_HZ


@dataclass(frozen=True)
class MaestroEntry:
    """One row from the MAESTRO manifest."""
    split:    str            # 'train' | 'validation' | 'test'
    composer: str
    title:    str
    midi_rel: str            # relative path under maestro_root
    audio_rel: str
    year:     int
    duration: float          # seconds

    def midi_path(self, root: Path) -> Path:
        return root / self.midi_rel


_MANIFEST_FILENAME = "maestro-v3.0.0.csv"


def list_maestro(root: Path, split: Optional[str] = "train") -> List[MaestroEntry]:
    """
    Read the MAESTRO manifest and return entries for the requested split
    ('train', 'validation', 'test', or None for all).

    Raises FileNotFoundError with a helpful message if the manifest is missing.
    """
    root = Path(root)
    manifest = root / _MANIFEST_FILENAME
    if not manifest.exists():
        raise FileNotFoundError(
            f"MAESTRO manifest not found at {manifest}. "
            f"Download MAESTRO v3.0.0 from https://magenta.tensorflow.org/datasets/maestro "
            f"and extract under {root}."
        )

    entries: List[MaestroEntry] = []
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row.get("split") != split:
                continue
            entries.append(MaestroEntry(
                split    = row["split"],
                composer = row.get("canonical_composer", ""),
                title    = row.get("canonical_title", ""),
                midi_rel = row["midi_filename"],
                audio_rel= row.get("audio_filename", ""),
                year     = int(row.get("year", 0) or 0),
                duration = float(row.get("duration", 0.0) or 0.0),
            ))
    return entries


# ── MIDI → roll ───────────────────────────────────────────────────────────────

def load_piano_roll(midi_path: Path, frame_hz: int = FRAME_RATE_HZ) -> np.ndarray:
    """
    Load a MIDI file and rasterise its notes onto an 88-key roll at
    `frame_hz` frames/second.  Velocity is collapsed to binary 1.0 (the
    Transformer is trained to predict presence; velocity restoration is a
    separate concern handled at post-processing).

    Returns (T, 88) float32.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if not pm.instruments:
        return np.zeros((1, PIANO_DIM), dtype=np.float32)

    total_dur = pm.get_end_time()
    T = max(1, int(np.ceil(total_dur * frame_hz)))
    roll = np.zeros((T, PIANO_DIM), dtype=np.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            p = int(note.pitch) - PIANO_LO_MIDI
            if p < 0 or p >= PIANO_DIM:
                continue
            a = max(0, int(round(note.start * frame_hz)))
            b = min(T, int(round(note.end   * frame_hz)))
            if b > a:
                roll[a:b, p] = 1.0
    return roll


def midi_tempo(midi_path: Path, default: float = 120.0) -> float:
    """Extract the initial tempo from a MIDI file (BPM), falling back to a default."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    tempi, times = pm.get_tempo_changes()
    if len(tempi):
        return float(tempi[0])
    return default


# ── Segment sampling ──────────────────────────────────────────────────────────

def sample_segment(
    roll:           np.ndarray,
    segment_frames: int,
    min_active_keys:int = 4,
    max_tries:      int = 20,
    rng:            random.Random | None = None,
) -> Optional[np.ndarray]:
    """
    Crop a random `segment_frames`-long window out of `roll`.  Rejects
    windows that don't contain enough activity (median frame has <
    min_active_keys keys pressed) to avoid training on long silences at
    the start/end of MAESTRO performances.

    Returns (segment_frames, 88) or None if no valid window found in
    `max_tries`.  Caller can either retry with another piece or fall back
    to padding.
    """
    rng = rng or random
    T, K = roll.shape
    if T <= segment_frames:
        # Whole piece is shorter than the segment — pad with silence at the end
        pad = np.zeros((segment_frames - T, K), dtype=np.float32)
        return np.concatenate([roll, pad], axis=0)

    for _ in range(max_tries):
        start = rng.randint(0, T - segment_frames)
        window = roll[start : start + segment_frames]
        # Acceptance test: enough activity?
        active_per_frame = (window > 0).sum(axis=-1)
        if np.median(active_per_frame) >= min_active_keys:
            return window

    return None
