"""Music2MIDI -> (mel segment, arrangement rolls).

The transcriber's target is "every pitched note in this recording".  The
arranger's target is "the notes a human put in their piano cover of it" — a
deliberate, lossy reduction.  Same audio, same model, same front end; only the
answer changes, which is why this sits beside data.py rather than replacing it.

Music2MIDI supplies 2,974 such pairs: piano covers scraped from MuseScore,
beat-aligned to their source recordings by the dataset's authors.  The
alignment is the part that would normally cost weeks and it is already done —
verified here rather than trusted: across 347 songs the released note times end
at a median 0.988 of the YOUTUBE duration and only 1.014 of the SCORE duration,
matching within 5% for 90.8% of songs against 41.2%.  The times are in the
audio's timeframe.

WHY FINE-TUNE RATHER THAN TRAIN FRESH.  Transcription and arrangement share
almost all of their difficulty: both have to find pitches in a dense mix.  They
differ only in what to do once found.  Starting from the transcription
checkpoint keeps the expensive half — measured note F1 0.630 against a 0.078
chance floor — and spends the small dataset on the cheap half.  This is the
AMT-APC result (arXiv 2409.14086), which fine-tunes an AMT model into a piano
cover generator for the same reason.

AUDIO IS NOT SHIPPED.  The dataset gives YouTube ids, not recordings.  Point
`audio_dir` at 16 kHz mono WAVs named `{id}.wav` — the same format Slakh redux
uses, so both loaders read alike and neither pays a resample at training time.
"""
from __future__ import annotations

import pathlib
import random
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C

# Where the released dataset was unpacked.
DATA = C.ROOT / "data" / "music2midi" / "music2midi-dataset"
AUDIO = C.ROOT / "data" / "music2midi" / "audio"

# A song teaches the velocity head only if it HAS dynamics.
#
# Measured over 800 arrangements: 35.2% carry a single velocity for the whole
# piece and 53.4% carry four or more.  MuseScore exports a default velocity
# when the engraver wrote no dynamics, so a third of this corpus is not quiet
# playing — it is missing data.  Training on it would teach the head to emit a
# constant and destroy what Slakh, which has real per-note velocity, taught it.
# So those songs still train onsets and frames and are simply masked out of the
# velocity loss.
MIN_VELOCITIES = 2


def available() -> bool:
    return (DATA / "midi_numpy").is_dir()


def split_ids(audio_dir: Optional[pathlib.Path] = None
              ) -> Tuple[List[str], List[str], List[str]]:
    """The dataset's OWN train/val/test ids, filtered to what we have audio for.

    Using their split rather than ours keeps our numbers comparable to the
    paper's, and more importantly their split was made after filtering on
    alignment quality — re-splitting would mix well-aligned and badly-aligned
    pairs across both sides.
    """
    ad = audio_dir or AUDIO
    z = np.load(DATA / "dataset_split.npz", allow_pickle=True)

    def keep(key: str) -> List[str]:
        out = []
        for i in z[key]:
            sid = str(i)
            if (DATA / "midi_numpy" / f"{sid}.npy").exists() and \
               (ad / f"{sid}.wav").exists():
                out.append(sid)
        return out

    return keep("train_id"), keep("val_id"), keep("test_id")


def arrangement_rolls(sid: str, n_frames: int):
    """(onset, frame, velocity, has_dynamics) for one human piano cover.

    Deliberately mirrors data.track_rolls, including uint8 storage and the
    ONSET_SPREAD widening, so a segment from either dataset is indistinguishable
    to the model and the two can be mixed in one batch if we ever want to.
    """
    a = np.load(DATA / "midi_numpy" / f"{sid}.npy")     # [on, off, pitch, vel]
    onset = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    frame = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    vel = np.zeros((C.N_PITCH, n_frames), dtype=np.uint8)
    has_dyn = len(np.unique(a[:, 3])) >= MIN_VELOCITIES
    for on_s, off_s, pitch, v in a:
        p = int(pitch) - C.MIDI_LO
        if not (0 <= p < C.N_PITCH):
            continue
        s = int(round(float(on_s) * C.FPS))
        e = int(round(float(off_s) * C.FPS))
        if s >= n_frames:
            continue
        frame[p, max(0, s):min(n_frames, max(e, s + 1))] = 1
        lo = max(0, s - C.ONSET_SPREAD)
        hi = min(n_frames, s + C.ONSET_SPREAD + 1)
        onset[p, lo:hi] = 1
        vel[p, lo:hi] = int(np.clip(v, 1, 127))
    return onset, frame, vel, has_dyn


class Music2MidiSegments(Dataset):
    """Random fixed-length segments over Music2MIDI, rolls cached like Slakh's."""

    MEM_CACHE = 120

    def __init__(self, ids: List[str], audio_dir: Optional[pathlib.Path] = None,
                 segments_per_song: int = 8, seed: int = 0):
        self.ids = ids
        self.audio_dir = audio_dir or AUDIO
        self.per = segments_per_song
        self.rng = random.Random(seed)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.ids) * self.per

    def _rolls(self, sid: str):
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        import soundfile as sf
        wav = self.audio_dir / f"{sid}.wav"
        info = sf.info(str(wav))
        n_frames = int(info.frames * C.SR / info.samplerate / C.HOP) + 1
        npz = self.audio_dir / f"{sid}_rolls.npz"
        if npz.exists():
            try:
                z = np.load(npz)
                on, fr, vl, hd = z["on"], z["fr"], z["vl"], bool(z["hd"])
            except Exception:                               # noqa: BLE001
                on, fr, vl, hd = arrangement_rolls(sid, n_frames)
        else:
            on, fr, vl, hd = arrangement_rolls(sid, n_frames)
            try:
                np.savez_compressed(npz, on=on, fr=fr, vl=vl, hd=hd)
            except Exception:                               # noqa: BLE001
                pass
        self._cache[sid] = (on, fr, vl, hd, info.frames)
        while len(self._cache) > self.MEM_CACHE:
            self._cache.popitem(last=False)
        return self._cache[sid]

    def __getitem__(self, i: int):
        # One unreadable song should cost one example, not the run — the same
        # guard data.py needed when a single malformed track out of 1,289 took
        # down a multi-hour job.
        base = i // self.per
        for attempt in range(4):
            sid = self.ids[(base + attempt) % len(self.ids)]
            try:
                return self._load(sid, i)
            except Exception as e:                          # noqa: BLE001
                if attempt == 0:
                    print(f"[arranger-data] skipping {sid}: "
                          f"{type(e).__name__}: {e}", flush=True)
        raise RuntimeError("four consecutive songs failed to load")

    def _load(self, sid: str, i: int):
        import soundfile as sf
        on, fr, vl, hd, n_samples = self._rolls(sid)
        seg = int(C.SEGMENT_S * C.SR)
        start = 0 if n_samples <= seg else self.rng.randrange(0, n_samples - seg)
        wav, _ = sf.read(str(self.audio_dir / f"{sid}.wav"), start=start,
                         frames=seg, dtype="float32", always_2d=True)
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
                torch.from_numpy(cut(vl).astype(np.float32) / 127.0),
                torch.tensor(1.0 if hd else 0.0))
