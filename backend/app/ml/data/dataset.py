"""
PyTorch IterableDataset wrappers that yield `Batch` tuples to the training
loop.

Two implementations:

    MaestroDataset  — real training data.  Walks the MAESTRO manifest,
                      lazily loads MIDIs, crops 4-second segments,
                      computes conditioning + noise on the fly.

    SyntheticDataset — random rolls.  Used by smoke-tests so the loss +
                      backward pass can be exercised end-to-end before
                      pulling in 120 GB of MAESTRO.

Both produce torch tensors so the training loop is agnostic to source.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from app.ml.config import (
    PIANO_DIM, GENERATOR, TRAIN, FRAME_RATE_HZ, PATHS,
)
from app.ml.training.loop import Batch

from .maestro     import MaestroEntry, list_maestro, load_piano_roll, sample_segment
from .conditioning import build_conditioning_from_roll
from .noise        import conditioned_noise


# ── Real MAESTRO dataset ──────────────────────────────────────────────────────

class MaestroDataset(IterableDataset):
    """
    Yields `Batch(roll, cond, noise)` tuples drawn from random MAESTRO
    pieces.  Designed to be wrapped by torch.utils.data.DataLoader with
    batch_size=1 — the dataset itself emits already-batched tensors so we
    can apply per-sample augmentations (segment length, key transpose)
    cheaply.

    For multi-worker loading, each worker gets a deterministic subset of
    the manifest indices based on worker_id, so workers never read the
    same MIDI simultaneously.
    """

    def __init__(
        self,
        root:           Path = PATHS.maestro_root,
        split:          str  = "train",
        segment_seconds:float = TRAIN.segment_seconds,
        frame_hz:       int   = FRAME_RATE_HZ,
        batch_size:     int   = TRAIN.batch_size,
        seed:           int   = TRAIN.seed,
        transpose_range: int  = 3,                # ± semitones random key shift
    ) -> None:
        super().__init__()
        self.root            = Path(root)
        self.split           = split
        self.segment_frames  = int(segment_seconds * frame_hz)
        self.frame_hz        = frame_hz
        self.batch_size      = batch_size
        self.seed            = seed
        self.transpose_range = transpose_range

        self.entries: List[MaestroEntry] = list_maestro(self.root, split=split)
        if not self.entries:
            raise RuntimeError(f"No MAESTRO entries found for split={split}")

    # ── per-worker shard logic ────────────────────────────────────────────
    def _worker_entries(self) -> List[MaestroEntry]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.entries
        n = info.num_workers
        wid = info.id
        return self.entries[wid::n]

    # ── augmentation: random transpose ────────────────────────────────────
    @staticmethod
    def _transpose(roll: np.ndarray, semitones: int) -> np.ndarray:
        if semitones == 0:
            return roll
        out = np.zeros_like(roll)
        if semitones > 0:
            out[:, semitones:] = roll[:, :-semitones]
        else:
            out[:, :semitones] = roll[:, -semitones:]
        return out

    # ── one batch ─────────────────────────────────────────────────────────
    def _sample_one(self, rng: random.Random, npy_rng: np.random.Generator) -> Optional[Batch]:
        rolls:  List[np.ndarray] = []
        conds:  List[np.ndarray] = []
        noises: List[np.ndarray] = []
        entries = self._worker_entries()

        for _ in range(self.batch_size):
            tries = 0
            while True:
                tries += 1
                if tries > 20:
                    return None
                entry = rng.choice(entries)
                try:
                    roll = load_piano_roll(entry.midi_path(self.root), self.frame_hz)
                except Exception:
                    continue
                seg = sample_segment(roll, self.segment_frames, rng=rng)
                if seg is None:
                    continue
                semis = rng.randint(-self.transpose_range, self.transpose_range)
                seg = self._transpose(seg, semis)
                cond = build_conditioning_from_roll(seg)
                noi  = conditioned_noise(cond, rng=npy_rng)
                rolls.append(seg)
                conds.append(cond)
                noises.append(noi)
                break

        return Batch(
            roll  = torch.from_numpy(np.stack(rolls,  0)),
            cond  = torch.from_numpy(np.stack(conds,  0)),
            noise = torch.from_numpy(np.stack(noises, 0)),
        )

    def __iter__(self) -> Iterator[Batch]:
        info = torch.utils.data.get_worker_info()
        seed = self.seed + (info.id if info else 0)
        rng = random.Random(seed)
        npy_rng = np.random.default_rng(seed)
        while True:
            batch = self._sample_one(rng, npy_rng)
            if batch is None:
                # Skipped a string of pieces — yield silence rather than crash
                T = self.segment_frames
                zeros = np.zeros((self.batch_size, T, PIANO_DIM), dtype=np.float32)
                yield Batch(
                    roll  = torch.from_numpy(zeros),
                    cond  = torch.zeros(self.batch_size, T, GENERATOR.cond_dim),
                    noise = torch.zeros(self.batch_size, T, PIANO_DIM),
                )
            else:
                yield batch


# ── Synthetic smoke-test dataset ──────────────────────────────────────────────

class SyntheticDataset(IterableDataset):
    """
    Random piano-roll generator for smoke tests.  Produces plausibly-shaped
    rolls (a handful of pitches active per frame, slow attack/release) so
    the loss terms have meaningful gradients to follow even without real
    music as input.
    """

    def __init__(
        self,
        segment_seconds: float = TRAIN.segment_seconds,
        frame_hz:        int   = FRAME_RATE_HZ,
        batch_size:      int   = TRAIN.batch_size,
        seed:            int   = TRAIN.seed,
        avg_notes_per_frame: float = 4.0,
    ) -> None:
        super().__init__()
        self.T              = int(segment_seconds * frame_hz)
        self.batch_size     = batch_size
        self.seed           = seed
        self.density        = avg_notes_per_frame / PIANO_DIM

    def __iter__(self) -> Iterator[Batch]:
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0))
        B, T = self.batch_size, self.T

        while True:
            # Sample a sparse onset map, then "hold" each onset for a random
            # 4–16 frame duration so we get notes rather than 1-frame blips.
            roll = np.zeros((B, T, PIANO_DIM), dtype=np.float32)
            for b in range(B):
                onsets = rng.random((T, PIANO_DIM)) < self.density
                durs   = rng.integers(4, 16, size=onsets.shape)
                for t, k in zip(*np.where(onsets)):
                    end = min(T, int(t + durs[t, k]))
                    roll[b, t:end, k] = 1.0
            cond  = np.zeros((B, T, GENERATOR.cond_dim), dtype=np.float32)
            # Fake chord conditioning: pick a random root each batch, fill
            # in major-chord chroma so the noise generator has something to
            # bias against.
            for b in range(B):
                root = rng.integers(0, 12)
                for pc in [root, (root + 4) % 12, (root + 7) % 12]:
                    cond[b, :, pc] = 1.0
                cond[b, :, 12 + root] = 1.0
            noise = np.stack([
                conditioned_noise(cond[b], rng=rng) for b in range(B)
            ], axis=0)
            yield Batch(
                roll  = torch.from_numpy(roll),
                cond  = torch.from_numpy(cond),
                noise = torch.from_numpy(noise),
            )
