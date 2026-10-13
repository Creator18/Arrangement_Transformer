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
    ROOT / "app" / "ml" / "spectral" / "checkpoints" / "run2_real_finetune" / "best.pt"))
# run2 = run1 fine-tuned on 50% REAL MAESTRO recordings + augmented renders.
# On 12 real test recordings (unfiltered MIDI truth): note F1 0.900 -> 0.960,
# F1 with offsets 0.451 -> 0.690, soft-note recall 0.713 -> 0.835 — level with
# ByteDance (0.966) on notes and well ahead on durations (0.417).
# run1 (rendered audio only) is still at checkpoints/run1_v100_60k/best.pt.

# Onset threshold.  Swept on 800 rendered validation clips the best was 0.93
# (F1 0.868), but rendered audio flatters confidence.  On 12 REAL MAESTRO
# recordings against the full MIDI truth (eval/real_audio_eval.py), with the
# short-note filters off: 0.85 -> F1 0.899, 0.90 -> 0.900, 0.93 -> 0.898 —
# flat at the top, and 0.85 keeps more of the soft playing (recall 0.713 vs
# 0.655 at 0.93).  Chosen on those same 12 recordings, so not an independent
# score.
ONSET_THR = float(os.environ.get("ARRANGEMENT_SPECTRAL_ONSET_THR", 0.85))
FRAME_THR = 0.5
WINDOW_S, EDGE_S = 20.0, 2.0

# Keep short notes and fast repeats (default on).  The shared decoder deletes
# notes under 80 ms and the re-struck note before a repeat; against an
# unfiltered ground truth that removed a third of this model's correct notes
# (F1 0.968 -> 0.757) and three quarters of notes re-struck within 150 ms.
# Playback gives short notes an audible minimum length instead.  Set
# ARRANGEMENT_SPECTRAL_KEEP_SHORT=0 for the old behaviour, to A/B the two.
KEEP_SHORT = os.environ.get("ARRANGEMENT_SPECTRAL_KEEP_SHORT", "1") != "0"

_cache: Dict[str, Any] = {}


def available() -> bool:
    return CKPT.exists()


def _load():
    # keyed by checkpoint, so an evaluation alternating models loads each once
    if CKPT in _cache:
        return _cache[CKPT]
    import torch
    from app.ml.spectral import frontend as S
    from app.ml.spectral.model import from_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if "K" in ck:                       # style-code model (app/ml/spectral/style_train.py)
        from app.ml.spectral.style_train import load_for_app
        net = load_for_app(ck, 0).to(dev).eval()
        print(f"[spectral] style model: {ck['K']} codes, ARRANGEMENT_STYLE_CODE="
              f"{os.environ.get('ARRANGEMENT_STYLE_CODE', '0')}")
    else:
        net = from_checkpoint(ck).to(dev).eval()
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    _cache[CKPT] = (net, fe, dev)
    print(f"[spectral] loaded {CKPT.name} (step {ck['step']}) on {dev}")
    return net, fe, dev


def rolls(audio_path: pathlib.Path, cond: Optional[int] = None):
    """(onset, frame, velocity) probabilities, each (88, T), for a whole file.

    Split out of transcribe() so an evaluation can run the model once and
    decode many ways.  Returns None for an empty file.  `cond` is the
    conditioning label for a model built with one (song-to-piano difficulty).
    """
    import librosa
    import torch
    from app.ml.spectral import frontend as S

    net, fe, dev = _load()
    if hasattr(net, "code"):            # style model: ONE code for the whole song
        want = os.environ.get("ARRANGEMENT_STYLE_CODE", "0")
        if want == "random":
            import random
            net.code = random.randrange(net.K)
        else:
            net.code = int(want)
        print(f"[spectral] style code {net.code} of {net.K} for this song")
    y, _ = librosa.load(str(audio_path), sr=S.SR, mono=True)
    if y.size == 0:
        return None
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
            c = torch.tensor([cond], device=dev) if cond is not None else None
            po, pf, pv = (torch.sigmoid(t.float())[0].T.cpu().numpy() for t in net(x, c)[:3])
            lo = f0 - a
            hi = min(lo + keep, po.shape[1], T - a)
            on[:, a + lo:a + hi] = po[:, lo:hi]
            fr[:, a + lo:a + hi] = pf[:, lo:hi]
            vl[:, a + lo:a + hi] = pv[:, lo:hi]
            f0 = a + hi
    return on, fr, vl


def transcribe(audio_path: pathlib.Path, onset_thr: Optional[float] = None,
               ckpt: Optional[pathlib.Path] = None, cond: Optional[int] = None,
               frame_thr: Optional[float] = None) -> List[Dict[str, Any]]:
    """Notes for a file.  ckpt/cond select another spectral model and its
    conditioning label — the song-to-piano arranger at a difficulty."""
    import librosa
    from app.ml.transcriber.decode import decode_notes
    global CKPT

    keep = CKPT
    if ckpt is not None:
        CKPT = pathlib.Path(ckpt)
    try:
        r = rolls(audio_path, cond=cond)
    finally:
        CKPT = keep
    if r is None:
        return []
    on, fr, vl = r
    thr = onset_thr if onset_thr is not None else ONSET_THR
    fthr = frame_thr if frame_thr is not None else FRAME_THR
    if KEEP_SHORT:
        notes = decode_notes(on, fr, vl, thr, fthr, min_frames=1,
                             min_note_s=0.0, keep_restruck=True)
    else:
        notes = decode_notes(on, fr, vl, thr, fthr)
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
