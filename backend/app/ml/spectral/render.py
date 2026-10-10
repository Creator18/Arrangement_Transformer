"""MAESTRO MIDI -> rendered piano audio with frame-aligned labels.

There is no real piano audio on disk: MAESTRO here is MIDI only (the audio
bundle is ~118 GB) and the 50 files in eval/data/audio are themselves
synthesised.  So clips are rendered through the project's own sampler (sfizz +
Salamander), which makes the labels aligned BY CONSTRUCTION — with two
corrections, both measured in eval/spectral_frontend_check.py:

  LATENCY   the sampler starts notes late, median 11.7 ms (5-16 ms by pitch).
            Labels are shifted by that much.  Real recordings will not have
            this offset; a model trained here must be checked on real audio
            before its timing is trusted.
  ROUNDING  frames are CENTRED on t*HOP, so a time maps to round(t*fps), not
            floor.  With floor, onsets read two frames late and onset AUC fell
            from 0.882 to 0.648.

Pedal is deliberately NOT rendered.  MAESTRO's notes end at key-up; rendering
with sustain would keep notes ringing past their labels, teaching a model that
sustained energy means "no note".  Onsets are unaffected either way, and real
pedalled audio is a question for the real-audio anchor.

Split by PIECE using MAESTRO's own split column, so no piece is on both sides.
"""
from __future__ import annotations

import csv
import pathlib
import random
from typing import List

import numpy as np

from . import frontend as S

ROOT = pathlib.Path(__file__).resolve().parents[4]
MAESTRO = ROOT / "data" / "maestro-v3.0.0"
CACHE = ROOT / "data" / "maestro_render"
LATENCY_S = 0.0117


def frame(t: float) -> int:
    return int(round((t + LATENCY_S) * S.FPS))


def pieces(split: str) -> List[dict]:
    rows = list(csv.DictReader(open(MAESTRO / "maestro-v3.0.0.csv", encoding="utf-8")))
    return [r for r in rows if r["split"] == split]


def render_clip(row: dict, seconds: float, seed: int):
    """(audio float32 @ SR, onset roll, frame roll) for one random window."""
    import pretty_midi
    from app.services.sampler import render_notes

    pm = pretty_midi.PrettyMIDI(str(MAESTRO / row["midi_filename"]))
    end = pm.get_end_time()
    rng = random.Random(seed)
    t0 = rng.uniform(0, max(0.0, end - seconds))
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end <= t0 or n.start >= t0 + seconds:
                continue
            notes.append({"pitch": n.pitch,
                          "onset": max(0.0, n.start - t0),
                          "offset": min(seconds, n.end - t0),
                          "velocity": n.velocity,
                          "struck": n.start >= t0})   # began inside the window
    y = render_notes([{k: v for k, v in n.items() if k != "struck"} for n in notes],
                     sr=S.SR).astype(np.float32)
    n_samp = int(seconds * S.SR)
    y = np.pad(y, (0, max(0, n_samp - len(y))))[:n_samp]
    T = S.n_frames(n_samp)
    on = np.zeros((S.N_KEYS, T), np.uint8)
    fr = np.zeros((S.N_KEYS, T), np.uint8)
    for n in notes:
        k = n["pitch"] - 21
        if not 0 <= k < S.N_KEYS:
            continue
        a, b = frame(n["onset"]), frame(n["offset"])
        fr[k, max(0, a):min(T, max(b, a + 1))] = 1
        if n["struck"] and a < T:
            on[k, a] = 1
    return y, on, fr


def build(split: str, n_pieces: int, seconds: float = 20.0) -> List[pathlib.Path]:
    """Render and cache clips; idempotent."""
    out = CACHE / split
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(pieces(split)[:n_pieces]):
        p = out / f"{i:04d}.npz"
        if not p.exists():
            y, on, fr = render_clip(row, seconds, seed=i)
            np.savez_compressed(p, y=(np.clip(y, -1, 1) * 32767).astype(np.int16),
                                on=on, fr=fr)
        paths.append(p)
    return paths


def load(p: pathlib.Path):
    z = np.load(p)
    return z["y"].astype(np.float32) / 32767.0, z["on"], z["fr"]
