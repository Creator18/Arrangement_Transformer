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


def augment(y: np.ndarray) -> np.ndarray:
    """Make a rendered clip sound less like a sampler in a simulated room.

    On 12 real MAESTRO recordings the model trained on clean renders lost
    soft notes (recall 0.655 vs 0.968 for loud ones) and held notes too long
    (median 1.33x true length) — real recordings ring with room and pedal
    that renders lack, and the model read ringing as a held key.  Training on
    reverberant, noisier renders whose labels still say "released here"
    teaches the difference.  Applied to rendered clips only: real recordings
    already carry a real room.
    """
    from scipy.signal import fftconvolve
    sr = S.SR
    if random.random() < 0.7:                          # room: decaying-noise impulse response
        rt60 = random.uniform(0.2, 1.4)
        n = int(rt60 * sr)
        t = np.arange(n) / sr
        ir = np.random.randn(n).astype(np.float32) * np.exp(-6.9 * t / rt60).astype(np.float32)
        ir /= np.sqrt((ir ** 2).sum()) + 1e-9
        wet = random.uniform(0.1, 0.45)
        y = (1 - wet) * y + wet * fftconvolve(y, ir)[: len(y)].astype(np.float32)
    if random.random() < 0.6:                          # spectral tilt: brighter or darker mics
        spec = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / sr)
        db = random.uniform(-4, 4) * np.log2(np.maximum(f, 50) / 1000) / 3.3
        y = np.fft.irfft(spec * 10 ** (db / 20), n=len(y)).astype(np.float32)
    if random.random() < 0.5:                          # background noise, 25-50 dB below signal
        rms = np.sqrt((y ** 2).mean()) + 1e-9
        snr = random.uniform(25, 50)
        y = y + np.random.randn(len(y)).astype(np.float32) * rms * 10 ** (-snr / 20)
    y = y * 10 ** (random.uniform(-9, 3) / 20)         # level: soft passages must stay findable
    return np.clip(y, -1, 1).astype(np.float32)


def clips(root: pathlib.Path, split: str) -> List[pathlib.Path]:
    d = pathlib.Path(root) / split
    return sorted(p for p in d.glob("*.npz") if not p.name.endswith(".tmp.npz"))


class RenderedClips(Dataset):
    """Clips -> training tuples.

    Always (wav, onset, frame, velocity/127, has_velocity).  Then, by flag:
      with_cond   + difficulty label (0-2, 3 = unknown)          [index 5]
      with_roles  + melody roll, bass roll, notes_valid, roles_valid  [6..9]
                    (with_roles implies the cond slot is present)

    Curriculum clips (eval/make_curriculum_clips.py) carry "mel"/"bass" rolls
    marking which keys at each frame belong to the melody and the bass line,
    and a "scope": "all" when every pitched note in the audio is labelled
    (Slakh: exact per-stem MIDI) or "melody" when only the melody is (MUSDB:
    vocals labelled by pitch tracking, accompaniment unlabelled) — then the
    onset/frame loss must be masked, or every unlabelled accompaniment note
    would be taught as a false positive.
    """

    def __init__(self, paths: List[pathlib.Path], crop_s: Optional[float] = 8.0,
                 random_crop: bool = True, augment: bool = False, mix_p: float = 0.0,
                 with_cond: bool = False, with_roles: bool = False):
        self.paths = paths
        self.crop = int(crop_s * S.SR) if crop_s else None
        self.random_crop = random_crop
        self.augment = augment
        self.mix_p = mix_p
        self.with_cond = with_cond or with_roles
        self.with_roles = with_roles

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        c = self._load(i)
        if self.mix_p and random.random() < self.mix_p:
            c = self._mix(c, self._load(random.randrange(len(self.paths))))
        y = c["y"]
        if self.augment and not c["real"]:
            y = augment(y)
        out = (torch.from_numpy(y),
               torch.from_numpy(c["on"].astype(np.float32)),
               torch.from_numpy(c["fr"].astype(np.float32)),
               torch.from_numpy(c["vel"].astype(np.float32) / 127.0),
               torch.tensor(1.0 if c["has_vel"] else 0.0))
        if self.with_cond:
            out += (torch.tensor(c["cond"]),)
        if self.with_roles:
            out += (torch.from_numpy(c["mel"].astype(np.float32)),
                    torch.from_numpy(c["bass"].astype(np.float32)),
                    torch.tensor(1.0 if c["scope"] == "all" else 0.0),
                    torch.tensor(1.0 if c["has_roles"] else 0.0),
                    torch.from_numpy(c["onmask"].astype(np.float32)))
        return out

    @staticmethod
    def _mix(a, b):
        """Masking practice: a second clip 12-24 dB under the first, labels unioned.

        On real recordings 61% of missed notes were soft, and missed soft notes
        sat under a loud attack (>= vel 70 within 60 ms) 2.5x as often as
        found ones.  Burying a whole clip under another puts quiet notes under
        loud attacks everywhere, with exact labels.  The buried notes'
        velocity is scaled down (x0.4) — approximate, but on the right side:
        they ARE soft in this audio.  Role rolls keep the first clip's (the
        buried clip is accompaniment, never the melody).
        """
        n = min(len(a["y"]), len(b["y"])); T = min(a["on"].shape[1], b["on"].shape[1])
        ra_ = np.sqrt((a["y"][:n] ** 2).mean()) + 1e-6
        rb_ = np.sqrt((b["y"][:n] ** 2).mean()) + 1e-6
        g = ra_ / rb_ * 10 ** (-random.uniform(12, 24) / 20)
        c = dict(a)
        c["y"] = a["y"].copy(); c["y"][:n] += g * b["y"][:n]
        c["y"] = np.clip(c["y"], -1, 1).astype(np.float32)
        for k in ("on", "fr", "vel"):
            c[k] = a[k].copy()
        c["on"][:, :T] = np.maximum(a["on"][:, :T], b["on"][:, :T])
        c["fr"][:, :T] = np.maximum(a["fr"][:, :T], b["fr"][:, :T])
        add = (b["on"][:, :T] > 0) & (a["on"][:, :T] == 0)
        c["vel"][:, :T] = np.where(add, (b["vel"][:, :T] * 0.4).astype(a["vel"].dtype), a["vel"][:, :T])
        c["has_vel"] = a["has_vel"] and b["has_vel"]
        return c

    def _load(self, i: int):
        p = self.paths[i]
        z = np.load(p)
        y = z["y"].astype(np.float32) / 32767.0
        on, fr = z["on"], z["fr"]
        # A missing or unreadable velocity file costs that clip its velocity
        # target (masked out of the loss), never the whole run.
        vp = p.with_name(p.stem + ".vel.npy")
        try:
            vel = np.load(vp)
            has_vel = vel.shape == on.shape
        except Exception:                                   # noqa: BLE001
            vel, has_vel = None, False
        if not has_vel:
            vel = np.zeros_like(on)
        has_roles = "mel" in z.files
        mel = z["mel"] if has_roles else np.zeros_like(on)
        bass = z["bass"] if has_roles else np.zeros_like(on)
        onmask = z["onmask"] if "onmask" in z.files else np.ones_like(on)
        rolls = [on, fr, vel, mel, bass, onmask]
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
            rolls = [a[:, f0:f0 + T] for a in rolls]
            if rolls[0].shape[1] < T:
                pad = ((0, 0), (0, T - rolls[0].shape[1]))
                rolls = [np.pad(a, pad) for a in rolls]
        on, fr, vel, mel, bass, onmask = rolls
        piano = str(z["piano"]) if "piano" in z.files else ""
        return dict(y=y, on=on, fr=fr, vel=vel, has_vel=has_vel, mel=mel, bass=bass, onmask=onmask,
                    has_roles=has_roles,
                    scope=str(z["scope"]) if "scope" in z.files else "all",
                    real=piano in ("real", "m2m", "slakh", "musdb"),
                    cond=int(z["difficulty"]) if "difficulty" in z.files else 3)
