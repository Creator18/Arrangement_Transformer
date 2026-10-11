"""Audio -> notes with the spectral patch transformer, for the app's piano path.

A drop-in alternative to the ByteDance transcriber: same note dicts (pitch,
pitch_name, onset, offset, velocity), so every downstream stage runs
unchanged.  Selected with ARRANGEMENT_PIANO_MODE=spectral; the default stays
ByteDance until the new model has been evaluated against it.

The model was trained ONLY on rendered audio (five sampled pianos).  Running
it on a real recording through the app is the first real-audio test it gets,
so treat what comes out as a measurement of that gap, not a finished feature.

It does not predict the sustain pedal, so notes end at key release and the
render will sound drier than the ByteDance path, which has a pedal head.

Long pieces are processed in overlapping 20 s windows (what the model was
validated on) and only each window's centre is kept: an attention model sees
nothing past a window edge, so edges are its weakest output.  All window
arithmetic is in FRAMES — counting in samples is how the old transcriber's
windowing drifted 640 ms across a track.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = pathlib.Path(os.environ.get(
    "ARRANGEMENT_SPECTRAL_CKPT",
    ROOT / "app" / "ml" / "spectral" / "checkpoints" / "run1_v100_60k" / "best.pt"))

# Onset threshold, swept on 800 validation clips by eval/spectral_evaluate.py:
# 0.5 -> F1 0.847, 0.93 -> 0.868 (best, bracketed: 0.97 -> 0.859, 0.99 ->
# 0.781).  On all 3,686 test clips that is note F1 0.851 tuned vs 0.830 at 0.5.
ONSET_THR = float(os.environ.get("ARRANGEMENT_SPECTRAL_ONSET_THR", 0.93))
FRAME_THR = 0.5
WINDOW_S, EDGE_S = 20.0, 2.0

_cache: Dict[str, Any] = {}


def available() -> bool:
    return CKPT.exists()


def _load():
    if "net" in _cache:
        return _cache["net"], _cache["fe"], _cache["dev"]
    import torch
    from app.ml.spectral import frontend as S
    from app.ml.spectral.model import from_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = from_checkpoint(ck).to(dev).eval()
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    _cache.update(net=net, fe=fe, dev=dev)
    print(f"[spectral] loaded {CKPT.name} (step {ck['step']}) on {dev}")
    return net, fe, dev


def transcribe(audio_path: pathlib.Path, onset_thr: Optional[float] = None) -> List[Dict[str, Any]]:
    import librosa
    import torch
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes

    net, fe, dev = _load()
    y, _ = librosa.load(str(audio_path), sr=S.SR, mono=True)
    if y.size == 0:
        return []
    T = S.n_frames(len(y))
    win = int(WINDOW_S * S.FPS)          # frames per window
    edge = int(EDGE_S * S.FPS)           # frames discarded at each inner edge
    keep = win - 2 * edge
    on, fr, vl = (np.zeros((S.N_KEYS, T), np.float32) for _ in range(3))
    f0 = 0                               # first frame this window contributes
    with torch.no_grad():
        while f0 < T:
            a = max(0, f0 - edge)        # window start, frames
            seg = y[a * S.HOP:(a + win - 1) * S.HOP]
            x = fe(torch.from_numpy(seg.astype(np.float32))[None].to(dev))
            po, pf, pv = (torch.sigmoid(t.float())[0].T.cpu().numpy() for t in net(x))
            lo = f0 - a
            hi = min(lo + keep, po.shape[1], T - a)
            on[:, a + lo:a + hi] = po[:, lo:hi]
            fr[:, a + lo:a + hi] = pf[:, lo:hi]
            vl[:, a + lo:a + hi] = pv[:, lo:hi]
            f0 = a + hi
    notes = decode_notes(on, fr, vl, onset_thr if onset_thr is not None else ONSET_THR,
                         FRAME_THR)
    out = []
    for n in notes:
        p = int(n["pitch"])
        if 21 <= p <= 108 and n["offset"] > n["onset"]:
            out.append({"pitch": p, "pitch_name": librosa.midi_to_note(p),
                        "onset": float(n["onset"]), "offset": float(n["offset"]),
                        "velocity": int(n["velocity"])})
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    print(f"[spectral] {len(out)} notes from {pathlib.Path(audio_path).name} "
          f"(onset_thr {onset_thr if onset_thr is not None else ONSET_THR})")
    return out
