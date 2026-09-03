"""Slakh -> (mel segment, onset roll, frame roll).

Slakh gives per-stem audio AND per-stem aligned MIDI, plus a metadata file
naming each stem's instrument class and whether it is a drum kit.  That is the
thing this project has never had: ground truth for which notes are real, which
instrument they belong to, and exactly when they start.  Every question the
NMF pipeline had to guess at becomes supervised here.

The target is the union of all NON-DRUM stems' notes — "the pitched notes in
this song" — which is exactly what the arrangement pipeline is trying to
recover.  Drums are excluded rather than predicted-and-discarded, so the model
never spends capacity on something every downstream stage drops.
"""
from __future__ import annotations

import pathlib
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C


# Slakh ships two encodings and they are not interchangeable by name:
# BabySlakh is .wav, slakh2100_redux_16k is .flac.  Checking only for mix.wav
# silently rejected all 1,559 real tracks and fell back to the 20-track
# harness — training ran to completion on the wrong data and said so in one
# line nobody was watching.
AUDIO_EXT = (".flac", ".wav")


def audio_path(base: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    """base/stem.{flac,wav}, whichever exists."""
    for ext in AUDIO_EXT:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_tracks(dirs: Optional[List[pathlib.Path]] = None) -> List[pathlib.Path]:
    """Every Track***** directory that has both a mix and its MIDI."""
    out: List[pathlib.Path] = []
    for split in ("train", "validation", "test", ""):
        out += find_split(split, dirs)
    return out


def find_split(split: str, dirs: Optional[List[pathlib.Path]] = None
               ) -> List[pathlib.Path]:
    """Tracks in one of Slakh's OWN splits, or loose tracks when split is ''.

    Slakh ships train/validation/test directories and every published number
    is computed against them.  Re-splitting randomly would make our results
    incomparable to the literature and, worse, silently optimistic: the
    official split separates tracks that share source MIDI from the Lakh
    dataset, which a random split would scatter across both sides.
    """
    out: List[pathlib.Path] = []
    for d in (dirs if dirs is not None else C.SLAKH_DIRS):
        base = (d / split) if split else d
        if not base.exists():
            continue
        for t in sorted(base.glob("Track*")):
            if audio_path(t, "mix") is not None and (t / "MIDI").is_dir():
                out.append(t)
    return out


def official_splits(dirs: Optional[List[pathlib.Path]] = None):
    """(train, val) using Slakh's splits, falling back to a by-track split.

    BabySlakh has no split directories — it is 20 loose tracks for exactly this
    kind of harness work — so the fallback keeps the smoke test runnable.
    """
    tr, va = find_split("train", dirs), find_split("validation", dirs)
    if tr and va:
        return tr, va, "official"
    loose = find_split("", dirs)
    a, b = split_tracks(loose)
    return a, b, "fallback (no official splits found)"


def pitched_stems(track: pathlib.Path) -> List[str]:
    """Stem ids that are not drums, per the track's own metadata."""
    import yaml
    meta_p = track / "metadata.yaml"
    if not meta_p.exists():
        return [p.stem for p in sorted((track / "MIDI").glob("S*.mid"))]
    meta = yaml.safe_load(meta_p.read_text())
    out = []
    for sid, info in (meta.get("stems") or {}).items():
        if C.EXCLUDE_DRUMS and info.get("is_drum"):
            continue
        if (track / "MIDI" / f"{sid}.mid").exists():
            out.append(sid)
    return sorted(out)


def track_rolls(track: pathlib.Path, n_frames: int):
    """(onset, frame, velocity) rolls of shape (88, n_frames) for one track.

    Velocity is carried because the arrangement needs dynamics: a flat
    velocity makes a piano render sound like a punched roll.  The NMF got this
    from activation height, which is a proxy for loudness confounded with how
    well the template happened to match.  Here it is ground truth.
    """
    import pretty_midi
    # uint8, not float32.  These are masks and a 0-127 velocity, so four bytes
    # per cell buys nothing and costs 4x the memory — at 1500 tracks that is
    # the difference between 24 GB and 6 GB on a 16 GB machine.
    onset = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    frame = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    vel = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    for sid in pitched_stems(track):
        try:
            pm = pretty_midi.PrettyMIDI(str(track / "MIDI" / f"{sid}.mid"))
        except Exception:                                   # noqa: BLE001
            continue
        for inst in pm.instruments:
            if inst.is_drum and C.EXCLUDE_DRUMS:
                continue
            for n in inst.notes:
                p = int(n.pitch) - C.MIDI_LO
                if not (0 <= p < C.N_PITCH):
                    continue
                a = int(round(n.start * C.FPS))
                b = int(round(n.end * C.FPS))
                if a >= n_frames:
                    continue
                frame[p, max(0, a):min(n_frames, max(b, a + 1))] = 1
                lo = max(0, a - C.ONSET_SPREAD)
                hi = min(n_frames, a + C.ONSET_SPREAD + 1)
                onset[p, lo:hi] = 1
                vel[p, lo:hi] = int(n.velocity)
    return onset, frame, vel


class MelFront(torch.nn.Module):
    """Log-mel front end, shared by training and inference so they cannot drift."""

    def __init__(self):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=C.SR, n_fft=C.N_FFT, hop_length=C.HOP,
            f_min=C.FMIN, f_max=C.FMAX, n_mels=C.N_MELS, power=2.0,
            center=True,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # log1p rather than dB: no floor to tune, and it keeps quiet inner
        # voices representable instead of clamping them into the noise.
        return torch.log1p(self.mel(wav) * 1000.0)


class SlakhSegments(Dataset):
    """Random fixed-length segments over Slakh, with rolls cached twice over.

    Rolls are expensive to build (several MIDI files parsed per track) and
    cheap to store, so they are computed once and written to a compressed npz
    beside the dataset.  In memory they are held in a BOUNDED cache: an
    unbounded one needed 24 GB for 1500 tracks on a machine with 5.6 GB free,
    which would have killed the first real run about an hour in.
    """

    MEM_CACHE = 120          # tracks; ~4 MB each as uint8, so ~0.5 GB

    def __init__(self, tracks: List[pathlib.Path], segments_per_track: int = 8,
                 seed: int = 0):
        self.tracks = tracks
        self.per = segments_per_track
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[pathlib.Path, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.tracks) * self.per

    def _rolls(self, t: pathlib.Path):
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]
        import soundfile as sf
        mix = audio_path(t, "mix")
        info = sf.info(str(mix))
        n_frames = int(info.frames * C.SR / info.samplerate / C.HOP) + 1
        npz = t / "_rolls.npz"
        if npz.exists():
            try:
                z = np.load(npz)
                on, fr, vl = z["on"], z["fr"], z["vl"]
            except Exception:                               # noqa: BLE001
                on, fr, vl = track_rolls(t, n_frames)
        else:
            on, fr, vl = track_rolls(t, n_frames)
            try:
                np.savez_compressed(npz, on=on, fr=fr, vl=vl)
            except Exception:                               # noqa: BLE001
                pass                                        # read-only dataset
        self._cache[t] = (on, fr, vl, info.frames)
        while len(self._cache) > self.MEM_CACHE:
            self._cache.popitem(last=False)
        return self._cache[t]

    def __getitem__(self, i: int):
        import soundfile as sf
        t = self.tracks[i // self.per]
        on, fr, vl, n_samples = self._rolls(t)
        seg = int(C.SEGMENT_S * C.SR)
        if n_samples <= seg:
            start = 0
        else:
            start = self.rng.randrange(0, n_samples - seg)
        wav, _ = sf.read(str(audio_path(t, "mix")), start=start, frames=seg,
                         dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if len(wav) < seg:
            wav = np.pad(wav, (0, seg - len(wav)))
        f0 = start // C.HOP
        f1 = f0 + C.SEGMENT_FRAMES
        def cut(x):
            s = x[:, f0:f1]
            if s.shape[1] < C.SEGMENT_FRAMES:
                s = np.pad(s, ((0, 0), (0, C.SEGMENT_FRAMES - s.shape[1])))
            return s
        return (torch.from_numpy(wav),
                torch.from_numpy(cut(on).astype(np.float32)),
                torch.from_numpy(cut(fr).astype(np.float32)),
                torch.from_numpy(cut(vl).astype(np.float32) / 127.0))


def split_tracks(tracks: List[pathlib.Path], val_frac: float = 0.15,
                 seed: int = 0) -> Tuple[List, List]:
    """Split BY TRACK, never by segment.

    Segments from one track share instrumentation, mix and playing style, so
    splitting on segments would put near-duplicates on both sides and report a
    validation score that is really a memorisation score.
    """
    ts = sorted(tracks)
    random.Random(seed).shuffle(ts)
    k = max(1, int(len(ts) * val_frac))
    return ts[k:], ts[:k]
