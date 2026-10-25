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
from typing import Dict, List, Optional

import numpy as np

from . import frontend as S

ROOT = pathlib.Path(__file__).resolve().parents[4]
MAESTRO = ROOT / "data" / "maestro-v3.0.0"
CACHE = ROOT / "data" / "maestro_render"          # single-piano (Salamander) clips
# The full multi-piano set is ~34 GB, so it lives OUTSIDE the project folder,
# on the internal NVMe drive: training re-reads it every epoch, and D: is an
# external USB SSD — slower, and unpluggable mid-run.  Overridable with
# ARRANGEMENT_RENDER_DIR (e.g. on a cluster).
def _multi_dir() -> pathlib.Path:
    import os
    env = os.environ.get("ARRANGEMENT_RENDER_DIR")
    if env:
        return pathlib.Path(env)
    d = pathlib.Path("C:/arrangement_data/maestro_render_multi")
    return d if pathlib.Path(d.anchor).exists() else ROOT / "data" / "maestro_render_multi"


CACHE_MULTI = _multi_dir()

# Seconds rendered BEFORE each window and then cropped away.  Without it, a
# note held across the window's start is rendered as starting at t=0 — a fresh
# hammer attack in the audio while the label correctly says no onset.  Cutting
# whole pieces into consecutive windows would put that contradiction at almost
# every boundary.  Two seconds lets held notes ring into the window naturally.
PRE_ROLL_S = 2.0
PIANO_DIR = ROOT / "data" / "pianos"
LATENCY_S = 0.0117          # Salamander only; kept so the Phase 2 cache stays valid

# ── Multiple pianos ───────────────────────────────────────────────────────────
#
# One sampler in one simulated room lets a model learn "what Salamander sounds
# like" instead of "what a piano sounds like", and then stumble on anything
# else.  Each clip is rendered with a piano drawn at random, so the only thing
# consistent across the training set is the piano-ness.
#
# NoFX variants are used where a library offers them: the sampler's own room
# processing is applied to every piano identically, so the room varies with
# the INSTRUMENT'S recording rather than with a baked-in reverb on some
# libraries and not others.  The SF2 Upright KW is left out: the SF2 backend
# loads a General MIDI soundfont before it, and caches the parsed font at
# module level, so it cannot be switched per clip.  LivingRoom covers uprights.
#
# "Intimate" (the close-mic variant of the Headroom C3) is deliberately
# excluded.  With half-peak latency correction, onset alignment averaged over
# three pieces lands at lag 0 for every other piano and at -1 frame for
# Intimate.  Keeping it would teach the model that timing shifts by 16 ms
# depending on the recording, and it adds little variety beyond Headroom
# itself, which is the same instrument.
_PIANO_FILES = {
    "headroom":   "headroom_c3/**/Headroom Piano (NoFX).sfz",
    "maestro":    "maestro_cf3/**/Maestro Concert Grand.sfz",
    "splendid":   "splendid_steinway/**/Splendid Grand Piano.sfz",
    "livingroom": "livingroom_upright/**/KeyPleezer LivingRoom Upright - Free Micro SFZ.sfz",
}


def pianos() -> Dict[str, pathlib.Path]:
    """name -> .sfz path, for every piano actually present on disk."""
    from app.services import sfz_sampler as Z
    out = {}
    if Z.SFZ_PATH is not None:
        out["salamander"] = pathlib.Path(Z.SFZ_PATH)
    for name, pat in _PIANO_FILES.items():
        hit = next(PIANO_DIR.glob(pat), None)
        if hit is not None:
            out[name] = hit
    return out


_SALAMANDER = None


def render_with(piano: str, notes: List[dict]) -> np.ndarray:
    """Render through one named piano, with the sampler's standard room."""
    global _SALAMANDER
    from app.services import sfz_sampler as Z
    from app.services.sampler import _room
    if _SALAMANDER is None:
        _SALAMANDER = Z.SFZ_PATH
    Z.SFZ_PATH = pianos()[piano]           # the synth cache is keyed on this path
    try:
        return _room(Z.render_notes_sfz(notes, sr=S.SR), S.SR).astype(np.float32)
    finally:
        Z.SFZ_PATH = _SALAMANDER


def latencies() -> Dict[str, float]:
    """Per-piano onset latency in seconds, measured once and cached.

    The 11.7 ms figure was measured on Salamander alone.  Every sample library
    trims its recordings differently, so each one starts its notes after its
    own delay — and a label shift tuned for one piano would leave every other
    piano's labels misaligned by the difference.  Measured on single notes
    across the keyboard, median over pitch and velocity.

    "Where the note starts" is the time the attack envelope reaches HALF ITS
    PEAK.  The first version used the first sample above the noise floor, and
    the libraries disagreed by up to two frames afterwards: piano samples ramp
    up over several milliseconds with library-specific shapes, and the close-
    mic "Intimate" variant catches hammer contact well before the tone builds.
    Half-peak is the same point on the energy curve whatever a library's
    trimming, which is what a model needs — one timing convention across every
    instrument, so timing never becomes a function of which piano played.
    """
    import json
    p = PIANO_DIR / "latency_halfpeak.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    names = pianos()
    if all(n in cache for n in names):
        return cache
    t0 = 1.0
    for name in names:
        if name in cache:
            continue
        lags = []
        for pitch in (33, 48, 60, 72, 84, 96):
            for vel in (40, 100):
                y = render_with(name, [{"pitch": pitch, "onset": t0,
                                        "offset": t0 + 1.0, "velocity": vel}])
                # 1 ms RMS envelope, peak within 200 ms of the nominal onset,
                # first crossing of half that peak
                env = np.sqrt(np.convolve(y ** 2, np.ones(24) / 24, mode="same"))
                lo, hi = int((t0 - 0.05) * S.SR), int((t0 + 0.2) * S.SR)
                w = env[lo:hi]
                i = int(np.argmax(w >= 0.5 * w.max())) + lo
                lags.append(i / S.SR - t0)
        cache[name] = float(np.median(lags))
    p.write_text(json.dumps(cache, indent=1))
    return cache


def frame(t: float, latency: float = LATENCY_S) -> int:
    return int(round((t + latency) * S.FPS))


def pieces(split: str) -> List[dict]:
    rows = list(csv.DictReader(open(MAESTRO / "maestro-v3.0.0.csv", encoding="utf-8")))
    return [r for r in rows if r["split"] == split]


def render_clip(row: dict, seconds: float, seed: int, piano: Optional[str] = None):
    """(audio float32 @ SR, onset roll, frame roll) for one random window.

    piano=None renders through the default sampler with the Salamander
    latency; a name renders through that piano with ITS measured latency.
    """
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
    plain = [{k: v for k, v in n.items() if k != "struck"} for n in notes]
    if piano is None:                       # legacy single-piano path (Phase 2 cache)
        y = render_notes(plain, sr=S.SR).astype(np.float32)
        lat = LATENCY_S
    else:
        y = render_with(piano, plain)
        lat = latencies()[piano]
    # Libraries are recorded at different levels; keep every clip out of the
    # int16 clipping range rather than let the loudest piano distort.
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 0.95:
        y = y * (0.95 / peak)
    n_samp = int(seconds * S.SR)
    y = np.pad(y, (0, max(0, n_samp - len(y))))[:n_samp]
    T = S.n_frames(n_samp)
    on = np.zeros((S.N_KEYS, T), np.uint8)
    fr = np.zeros((S.N_KEYS, T), np.uint8)
    for n in notes:
        k = n["pitch"] - 21
        if not 0 <= k < S.N_KEYS:
            continue
        a, b = frame(n["onset"], lat), frame(n["offset"], lat)
        fr[k, max(0, a):min(T, max(b, a + 1))] = 1
        if n["struck"] and a < T:
            on[k, a] = 1
    return y, on, fr


def build(split: str, n_pieces: int, seconds: float = 20.0,
          multi: bool = False) -> List[pathlib.Path]:
    """Render and cache clips; idempotent.

    multi=False keeps the original Salamander-only cache, so Phase 2's numbers
    stay reproducible.  multi=True writes a separate cache where each clip's
    piano is drawn at random — deterministically from (split, index), so a
    rebuild produces the same piano for the same clip — and the choice is
    stored in the file, so results can later be broken down per piano.
    """
    out = (CACHE_MULTI if multi else CACHE) / split
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(pianos()) if multi else []
    paths = []
    for i, row in enumerate(pieces(split)[:n_pieces]):
        p = out / f"{i:04d}.npz"
        if not p.exists():
            piano = random.Random(f"{split}-{i}").choice(names) if multi else None
            y, on, fr = render_clip(row, seconds, seed=i, piano=piano)
            np.savez_compressed(p, y=(np.clip(y, -1, 1) * 32767).astype(np.int16),
                                on=on, fr=fr, piano=piano or "salamander")
        paths.append(p)
    return paths


def load(p: pathlib.Path):
    z = np.load(p)
    return z["y"].astype(np.float32) / 32767.0, z["on"], z["fr"]


def render_window(pm, t0: float, seconds: float, piano: str, pedal=None, onset_shift=None):
    """One window of a parsed MIDI, rendered with pre-roll; (audio, onsets, frames).

    The full-dataset path.  Unlike render_clip, the window position is given
    (consecutive windows tile a whole piece) and audio is rendered from
    t0 - PRE_ROLL_S, then cropped, so notes held across t0 ring into the
    window instead of being re-struck at its first sample.
    """
    base = max(0.0, t0 - PRE_ROLL_S)
    crop = t0 - base
    span = crop + seconds
    notes, labelled = [], []
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end <= base or n.start >= t0 + seconds:
                continue
            end_snd = n.end
            if pedal:                       # sustain pedal: the SOUND rings on until the pedal lifts
                for a_, b_ in pedal:        # (the label keeps the written key release)
                    if a_ <= n.end < b_:
                        nxt = min((m.start for m in inst.notes if m.pitch == n.pitch and m.start > n.start), default=b_)
                        end_snd = max(n.end, min(b_, nxt))
            sh = onset_shift.get((n.pitch, round(n.start, 3)), 0.0) if onset_shift else 0.0   # rolled chord: SOUND only
            notes.append({"pitch": n.pitch, "onset": max(0.0, n.start + sh - base),
                          "offset": min(span, max(end_snd, n.start + sh + .05) - base), "velocity": n.velocity})
            if n.end > t0:
                labelled.append((n.pitch, n.start - t0, min(seconds, n.end - t0)))
    y = render_with(piano, notes)
    a = int(round(crop * S.SR))
    n_samp = int(seconds * S.SR)
    y = y[a:a + n_samp]
    y = np.pad(y, (0, max(0, n_samp - len(y))))
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak > 0.95:
        y = y * (0.95 / peak)
    lat = latencies()[piano]
    T = S.n_frames(n_samp)
    on = np.zeros((S.N_KEYS, T), np.uint8)
    fr = np.zeros((S.N_KEYS, T), np.uint8)
    for pitch, s, e in labelled:
        k = pitch - 21
        if not 0 <= k < S.N_KEYS:
            continue
        fa, fb = frame(max(0.0, s), lat), frame(e, lat)
        fr[k, max(0, fa):min(T, max(fb, fa + 1))] = 1
        if s >= 0 and fa < T:                 # struck inside this window
            on[k, fa] = 1
    return y.astype(np.float32), on, fr


def velocity_roll(pm, t0: float, seconds: float, piano: str) -> np.ndarray:
    """(88, T) uint8 velocity at each onset frame, for a window render_window made.

    The rendered clips were saved with onset and frame labels only.  Velocity
    comes straight from the MIDI rather than from re-rendering 36,401 clips,
    and uses EXACTLY render_window's mapping — the same frame() rounding and
    the same per-piano latency — so each velocity sits on its onset frame.
    eval/make_velocity_rolls.py checks that against the stored onset rolls.
    """
    lat = latencies()[piano]
    T = S.n_frames(int(seconds * S.SR))
    vel = np.zeros((S.N_KEYS, T), np.uint8)
    for inst in pm.instruments:
        for n in inst.notes:
            s = n.start - t0
            if s < 0 or s >= seconds:
                continue
            k = n.pitch - 21
            fa = frame(s, lat)
            if 0 <= k < S.N_KEYS and fa < T:
                vel[k, fa] = max(vel[k, fa], int(n.velocity))
    return vel


def piano_for(split: str, i: int) -> str:
    """Deterministic piano per piece, so a re-run renders identically."""
    return random.Random(f"{split}-{i}").choice(sorted(pianos()))


def piano_of(p: pathlib.Path) -> str:
    z = np.load(p)
    return str(z["piano"]) if "piano" in z.files else "salamander"
