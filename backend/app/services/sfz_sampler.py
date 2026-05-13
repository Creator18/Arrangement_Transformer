"""
SFZ-backed sampler.  Uses pysfizz (libsfizz Python bindings) to render note
events through a real SFZ instrument — Salamander Grand Piano V3 by default.

Why SFZ over SF2
----------------
SFZ is the modern instrument format used by Reaper, Ardour, LSP, MOD, and
the sample-library community.  Salamander natively ships as SFZ with 16
velocity layers per key, dedicated release-noise samples, hammer noise,
and a sustain-pedal sympathetic-resonance layer.  SF2 conversions drop
most of that.

Render strategy
---------------
sfizz is a real-time engine: you push note_on/note_off events at sample
delays inside a fixed-size audio block, then read the rendered output.
For our offline use case we:
    1. Build a flat event list (note_on@onset_sample, note_off@offset_sample)
       sorted by time.
    2. Step the synth one block at a time.  For each event whose sample
       index falls inside [block_start, block_end), call note_on/off with
       delay = (event_sample - block_start).
    3. Pull a stereo block out, downmix to mono, append.
    4. After the last event, keep rendering blocks until either
       get_num_active_voices() == 0 or we hit a 5-second tail cap.

That gives us proper polyphony, real release tails (the SFZ's release
samples ring out naturally), and pedal modelling all routed through one
engine — exactly what was missing from the SF2 path.

Module-level state
------------------
- The synth is lazily constructed at first call and cached.  Recreating it
  is cheap but the SFZ load is ~0.6 s, so we keep it around.
- A second cache keyed on `(sr, sfz_path)` lets us re-create the synth if
  a downstream caller requests a different sample rate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

import numpy as np


# ── locate the SFZ file ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Salamander V3 is the preferred default — drop additional SFZ paths here
# if you ever want to A/B against another instrument.
_SFZ_CANDIDATES = [
    os.environ.get("ARRANGEMENT_SFZ"),
    _REPO_ROOT / "SalamanderGrandPiano-master" / "SalamanderGrandPiano-master"
        / "Salamander Grand Piano V3.sfz",
    _REPO_ROOT / "SalamanderGrandPiano-master" / "Salamander Grand Piano V3.sfz",
]


def _find_sfz() -> Optional[Path]:
    for c in _SFZ_CANDIDATES:
        if c is None:
            continue
        p = Path(c)
        if p.exists() and p.suffix.lower() == ".sfz":
            return p
    return None


SFZ_PATH: Optional[Path] = _find_sfz()


# ── synth cache ───────────────────────────────────────────────────────────────
_SYNTH_CACHE: Dict[Tuple[int, str], object] = {}
_BLOCK_SIZE = 512


def _get_synth(sr: int):
    """
    Build (or fetch) a sfizz synth at `sr` with the configured SFZ loaded.
    Raises RuntimeError on any failure so the caller can fall back to the
    SF2 sampler.
    """
    if SFZ_PATH is None:
        raise RuntimeError("No SFZ file located.")
    import pysfizz                                       # local import — optional dep

    key = (int(sr), str(SFZ_PATH))
    if key in _SYNTH_CACHE:
        return _SYNTH_CACHE[key]

    synth = pysfizz.Synth(sample_rate=sr, block_size=_BLOCK_SIZE)
    # Generous polyphony — Salamander uses multiple voices per note
    # (key sample + release sample + sympathetic resonance) so we want
    # significant headroom over our typical 32-note simultaneous max.
    try:
        synth._synth.set_num_voices(256)
    except Exception:
        pass
    ok = synth.load_sfz_file(str(SFZ_PATH), quiet=True)
    if not ok:
        raise RuntimeError(f"sfizz failed to load {SFZ_PATH}")
    _SYNTH_CACHE[key] = synth
    return synth


# ── public render ─────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True iff sfizz + a Salamander/Salamander-like SFZ are installed."""
    if SFZ_PATH is None:
        return False
    try:
        import pysfizz  # noqa: F401
        return True
    except ImportError:
        return False


def render_notes_sfz(notes: List[Dict], sr: int = 22050) -> np.ndarray:
    """
    Render a note list through the SFZ instrument and return a mono float32
    waveform.  Same signature as `sf2_sampler.render_notes_sf2`.

    Note dict fields (matches the rest of the pipeline):
        pitch    int   MIDI pitch 21..108
        onset    float seconds
        offset   float seconds  (defaults to onset+0.5 if missing)
        velocity int   1..127   (defaults to 80)
    """
    synth = _get_synth(sr)
    raw = synth._synth                                   # libsfizz binding
    bs  = _BLOCK_SIZE

    if not notes:
        return np.zeros(sr, dtype=np.float32)

    # ── build event list ──────────────────────────────────────────────────
    events: List[Tuple[int, str, int, int]] = []         # (sample, kind, pitch, vel)
    max_off_s = 0
    for n in notes:
        pitch = int(n["pitch"])
        if not (0 <= pitch <= 127):
            continue
        vel = int(n.get("velocity", 80))
        vel = max(1, min(127, vel))
        onset_s  = max(0, int(float(n["onset"]) * sr))
        offset_s = max(onset_s + int(0.05 * sr),
                       int(float(n.get("offset", n["onset"] + 0.5)) * sr))
        events.append((onset_s,  "on",  pitch, vel))
        events.append((offset_s, "off", pitch, 0))
        max_off_s = max(max_off_s, offset_s)

    events.sort(key=lambda e: (e[0], 0 if e[1] == "on" else 1))

    # ── render block by block ─────────────────────────────────────────────
    # Tail cap (5 s) prevents an unterminated release sample from holding
    # the loop open forever if active_voices doesn't drop to zero.
    tail_cap_blocks = (5 * sr) // bs
    n_events = len(events)
    ev_i = 0
    block_idx = 0
    last_event_block = (max_off_s // bs) + 1

    left_chunks: List[np.ndarray] = []
    right_chunks: List[np.ndarray] = []

    # Pre-allocate enough output to cover [0, max_off_s + 5s tail]
    while True:
        block_start = block_idx * bs
        block_end   = block_start + bs

        # Fire any events that land in this block
        while ev_i < n_events and events[ev_i][0] < block_end:
            ev_sample, kind, pitch, vel = events[ev_i]
            delay = max(0, min(bs - 1, ev_sample - block_start))
            if kind == "on":
                raw.note_on(delay, pitch, vel)
            else:
                raw.note_off(delay, pitch, 0)
            ev_i += 1

        left, right = raw.render_block()
        left_chunks.append(np.asarray(left,  dtype=np.float32))
        right_chunks.append(np.asarray(right, dtype=np.float32))

        block_idx += 1

        # Stop after we've fired the last event AND voices have decayed,
        # or we've hit the tail cap.
        if ev_i >= n_events:
            if raw.get_num_active_voices() == 0:
                break
            if block_idx - last_event_block > tail_cap_blocks:
                break

    audio_stereo = np.stack(
        [np.concatenate(left_chunks), np.concatenate(right_chunks)],
        axis=0,
    )                                                    # (2, N)
    audio_mono = audio_stereo.mean(axis=0).astype(np.float32)

    peak = float(np.abs(audio_mono).max())
    if peak > 1e-6:
        audio_mono = (audio_mono / peak * 0.92).astype(np.float32)

    return audio_mono


# Convenience alias — matches the SF2 sampler's exported name pattern so
# the dispatcher can import either symmetrically.
render_notes = render_notes_sfz
