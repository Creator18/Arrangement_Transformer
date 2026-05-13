"""
Sampler dispatcher.

Single import point that callers use to render a note list to audio.  Picks
the best available backend at call time:

    1. sfizz  + Salamander Grand Piano SFZ   (preferred)
    2. SF2    + FluidSynth-style internal renderer  (fallback)

The fallback is the existing `sf2_sampler.render_notes_sf2`, which is
always available — it ships with a bundled SF2 via pretty_midi.  This
keeps the pipeline working on dev boxes that haven't installed pysfizz
or downloaded the Salamander SFZ.

Why a thin dispatcher
---------------------
Callers (export.py, transcription previews, debug renders) shouldn't care
which backend is in play.  They call `render_notes(notes, sr)`; we resolve
backend.  Logs the chosen backend once per process so it's obvious from
the server output which sampler is rendering.
"""
from __future__ import annotations

from typing import List, Dict
import numpy as np


_BACKEND_LOGGED = False


def _log_backend(name: str, detail: str = "") -> None:
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    suffix = f" ({detail})" if detail else ""
    print(f"[sampler] using {name}{suffix}")


def _try_sfz() -> bool:
    try:
        from app.services import sfz_sampler
        return sfz_sampler.is_available()
    except Exception:
        return False


def render_notes(notes: List[Dict], sr: int = 22050) -> np.ndarray:
    """
    Render `notes` to a mono float32 waveform at `sr`.

    Tries the SFZ (sfizz + Salamander) backend first.  Any failure —
    missing dep, missing SFZ file, render exception — falls back to the
    SF2 sampler.  The SF2 sampler itself falls back further to additive
    synthesis if no SF2 is available, so this call always returns audio.
    """
    if _try_sfz():
        try:
            from app.services.sfz_sampler import render_notes_sfz, SFZ_PATH
            _log_backend("sfizz", str(SFZ_PATH.name) if SFZ_PATH else "")
            return render_notes_sfz(notes, sr=sr)
        except Exception as e:
            print(f"[sampler] sfizz render failed ({e}); falling back to SF2")

    from app.services.sf2_sampler import render_notes_sf2, SF2_PATH
    _log_backend("sf2", str(SF2_PATH.name) if SF2_PATH else "additive-fallback")
    return render_notes_sf2(notes, sr=sr)


# Alias for legacy callers
render_notes_to_audio = render_notes
