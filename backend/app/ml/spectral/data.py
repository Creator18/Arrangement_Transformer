"""Rendered clips -> (audio, onset, frame, velocity) training examples.

Reads the multi-piano set written by eval/render_multi_full.py: one
<i>_<w>.npz per 20 s window (int16 audio + onset/frame rolls + piano name)
and, beside it, <i>_<w>.vel.npy from eval/make_velocity_rolls.py.

Audio stays raw here.  The spectral front end runs on the GPU inside the
training loop, because computing it per worker on the CPU would make data
loading the bottleneck, and caching it would be ~140 GB.

TRAIN crops are random, so each epoch sees different 8 s slices of every
window.  VALIDATION crops are fixed (from each window's start) so every
evaluation scores identical audio and numbers are comparable across a run.
"""
from __future__ import annotations

import pathlib
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import frontend as S


def clips(root: pathlib.Path, split: str) -> List[pathlib.Path]:
    d = pathlib.Path(root) / split
    return sorted(p for p in d.glob("*.npz") if not p.name.endswith(".tmp.npz"))


class RenderedClips(Dataset):
    def __init__(self, paths: List[pathlib.Path], crop_s: Optional[float] = 8.0,
                 random_crop: bool = True):
        self.paths = paths
        self.crop = int(crop_s * S.SR) if crop_s else None
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        z = np.load(p)
        y = z["y"].astype(np.float32) / 32767.0
        on, fr = z["on"], z["fr"]
        vp = p.with_name(p.stem + ".vel.npy")
        vel = np.load(vp) if vp.exists() else np.zeros_like(on)
        has_vel = vp.exists()
        if self.crop and len(y) > self.crop:
            # Module-level `random`, not a per-instance generator: DataLoader
            # workers each receive a COPY of the dataset, so an instance RNG
            # would hand every worker the same crops.  PyTorch reseeds the
            # module-level generator per worker and per epoch.
            s = random.randrange(0, len(y) - self.crop) if self.random_crop else 0
            # cut on a hop boundary so audio and label frames stay aligned
            s = (s // S.HOP) * S.HOP
            y = y[s:s + self.crop]
            f0, T = s // S.HOP, S.n_frames(self.crop)
            on, fr, vel = (a[:, f0:f0 + T] for a in (on, fr, vel))
            if on.shape[1] < T:
                pad = ((0, 0), (0, T - on.shape[1]))
                on, fr, vel = (np.pad(a, pad) for a in (on, fr, vel))
        return (torch.from_numpy(y),
                torch.from_numpy(on.astype(np.float32)),
                torch.from_numpy(fr.astype(np.float32)),
                torch.from_numpy(vel.astype(np.float32) / 127.0),
                torch.tensor(1.0 if has_vel else 0.0))
