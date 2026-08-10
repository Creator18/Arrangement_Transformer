"""
Convolution reverb — the room Salamander was never recorded in.

Measured on the current render: 60.4% of the energy sits in 150-800 Hz, so
the instrument is genuinely warm, and it decays for over four seconds.  What
is nearly absent is the release tail — 0.071% of total energy after the damper
drops.  Salamander is close-miked with almost no ambience, so notes stop dead
at the edge of their sample instead of decaying into a space.  That is what
makes a piano sound like samples rather than a piano somewhere.

This adds the space without touching the instrument.  It is a post-process, so
it is reversible, and if room turns out not to be what was missing it comes
straight back out.

The impulse response
--------------------
Uses a real IR when ARRANGEMENT_IR points at a WAV — same pattern as
ARRANGEMENT_SFZ, so a measured hall can be dropped in without code changes.
Otherwise one is synthesised: exponentially decaying noise with early
reflections and progressive high-frequency damping.

Synthesised is a genuine approximation, not a stand-in for a measurement.  But
the thing being fixed here is gross — the near-total absence of a tail — and a
plausible decay answers that question honestly enough to tell us whether ROOM
is the missing ingredient before anyone downloads anything.

Why the defaults are conservative
---------------------------------
Note offsets already run 2.10x longer than ground truth, so the render is
over-sustained before any reverb is added.  Piling a long wet tail on top of
that would turn a measurable defect into audible mud.  WET_DEFAULT is
deliberately low; the goal is for notes to stop decaying into silence, not to
put the piano in a cathedral.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# Room, not hall.  See the note above about existing over-sustain.
RT60_S = 1.5          # time for the tail to fall 60 dB
PREDELAY_S = 0.018    # gap before the first reflection — reads as distance
WET_DEFAULT = 0.22    # proportion of wet in the mix
HF_DAMP = 0.55        # how fast highs die relative to lows; real rooms damp them

_IR_CACHE: dict = {}


def _synth_ir(sr: int, rt60: float = RT60_S) -> np.ndarray:
    """A plausible small-room impulse response."""
    rng = np.random.default_rng(4242)          # fixed: the room must not change
    n = int(sr * (rt60 + PREDELAY_S))
    ir = np.zeros(n, dtype=np.float32)

    body_start = int(sr * PREDELAY_S)
    tail = rng.standard_normal(n - body_start).astype(np.float32)

    # Exponential decay to -60 dB over rt60.
    t = np.arange(len(tail)) / sr
    env = np.power(10.0, -3.0 * t / rt60)

    # Highs decay faster than lows.  Implemented as a running low-pass whose
    # cutoff falls with time — cheap, and the perceptual point (a tail that
    # darkens as it dies) is what matters, not filter exactness.
    out = np.empty_like(tail)
    z = 0.0
    for i in range(len(tail)):
        a = HF_DAMP * (1.0 - 0.6 * (i / len(tail)))
        z = a * z + (1.0 - a) * tail[i]
        out[i] = z
    ir[body_start:] = out * env

    # A few early reflections — these are what actually convey room SIZE;
    # a smooth tail alone reads as effect rather than space.
    for delay_ms, gain in ((11, 0.35), (17, 0.28), (23, 0.22), (31, 0.16)):
        d = body_start + int(sr * delay_ms / 1000)
        if d < n:
            ir[d] += gain

    peak = np.abs(ir).max()
    return ir / peak if peak else ir


def get_ir(sr: int) -> np.ndarray:
    key = ("file" if os.environ.get("ARRANGEMENT_IR") else "synth", sr)
    if key in _IR_CACHE:
        return _IR_CACHE[key]

    path = os.environ.get("ARRANGEMENT_IR")
    ir: Optional[np.ndarray] = None
    if path and os.path.exists(path):
        try:
            import soundfile as sf
            data, ir_sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if ir_sr != sr:
                import librosa
                data = librosa.resample(data, orig_sr=ir_sr, target_sr=sr)
            peak = np.abs(data).max()
            ir = (data / peak) if peak else data
        except Exception as e:                            # noqa: BLE001
            print(f"[reverb] could not load {path}: {e}; using synthesised IR")
    if ir is None:
        ir = _synth_ir(sr)

    _IR_CACHE[key] = ir
    return ir


def apply_reverb(audio: np.ndarray, sr: int, wet: float = WET_DEFAULT) -> np.ndarray:
    """Mix a convolved copy of `audio` back over the dry signal."""
    if wet <= 0 or audio.size == 0:
        return audio
    from scipy.signal import fftconvolve

    y = np.asarray(audio, dtype=np.float32)
    mono = y.ndim == 1
    chans = [y] if mono else [y[:, c] for c in range(y.shape[1])]
    ir = get_ir(sr)

    out = []
    for ch in chans:
        wetsig = fftconvolve(ch, ir, mode="full")[: len(ch) + len(ir) - 1]
        dry = np.pad(ch, (0, len(wetsig) - len(ch)))
        # Equal-ish power mix, so raising `wet` does not quietly drop the level
        # of the note itself — the dry signal is what carries the pitch.
        mixed = (1.0 - wet) * dry + wet * (wetsig / (np.abs(wetsig).max() or 1.0)
                                           * (np.abs(ch).max() or 1.0))
        out.append(mixed.astype(np.float32))

    res = out[0] if mono else np.stack(out, axis=1)
    peak = np.abs(res).max()
    # Convolution adds energy; guard the ceiling without squashing dynamics.
    if peak > 0.99:
        res = res * (0.99 / peak)
    return res
