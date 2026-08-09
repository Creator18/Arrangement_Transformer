"""
Pre-load everything the pipeline needs, before an upload arrives.

What this is worth, measured
-----------------------------
Every heavy component is constructed lazily on first use, so the first upload
after a server start pays for all of them:

    Onsets & Frames checkpoint     2.86 s
    Demucs htdemucs                0.64 s
    sfizz + Salamander SFZ         0.42 s
    madmom RNN + DBN               0.42 s
    first CUDA forward pass        0.27 s   (cuDNN autotune / kernel JIT)
    ----------------------------------------
                                  ~4.6 s

Those are warm-disk-cache numbers; a genuinely cold machine pays more.  It is
a ONE-TIME saving per server start, not per upload — the second upload in a
session already found everything loaded.  Worth being clear about that: this
does not make the pipeline faster, it moves a fixed cost off the critical
path and into the time the user spends choosing a file.

What is deliberately NOT here
------------------------------
Running separation and transcription concurrently looks like the bigger win,
and it is not available: /transcribe needs components.json, and even in
of_fullmix mode it uses the tempo and beat grid that separation's madmom pass
produces.  The stages are genuinely sequential.  Beat-tracking the raw mix
instead would break that dependency but would also change the beat grid every
downstream stage is tuned against, which is not a speed change.

Threading
---------
Warming happens on a daemon thread so the server answers /health immediately
and the UI can render while models load.  Each component is independent, so
one failing (no GPU, missing SFZ) does not block the rest — the pipeline
falls back to its own lazy path and simply pays the cost later.
"""
from __future__ import annotations

import threading
import time
from typing import Dict

_lock = threading.Lock()
_state: Dict[str, Dict] = {}
_started = False
_thread: threading.Thread | None = None

# Keys the interactive keyboard is most likely to be clicked on: the 5-octave
# target range. Rendering these up front makes the first key press instant
# instead of waiting on a Salamander render.
_NOTE_WARM_LO = 36      # C2
_NOTE_WARM_HI = 96      # C7


def _record(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        fn()
        _state[name] = {"ok": True, "seconds": round(time.perf_counter() - t0, 3)}
    except Exception as e:                                   # noqa: BLE001
        # A failed warm is not a failed server: the lazy path still runs on
        # demand. Record it so /warmup can say so rather than silently lying.
        _state[name] = {"ok": False, "seconds": round(time.perf_counter() - t0, 3),
                        "error": f"{type(e).__name__}: {e}"[:200]}


def _warm_transcriber() -> None:
    from app.services.transcription import _get_of_transcriptor
    _get_of_transcriptor()


def _warm_first_pass() -> None:
    """
    One dummy inference. Constructing the model does not compile CUDA
    kernels — the first real forward pass does, and that cost otherwise
    lands on the user's audio.
    """
    import numpy as np
    from app.services.transcription import _get_of_transcriptor
    inf = _get_of_transcriptor()
    inf.transcribe(np.zeros(16000 * 2, dtype=np.float32), None)


def _warm_separator() -> None:
    from app.services.separation import _get_separator
    _get_separator()


def _warm_madmom() -> None:
    from app.services.separation import _get_madmom
    _get_madmom()


def _warm_sampler() -> None:
    from app.services import sfz_sampler
    if sfz_sampler.is_available():
        sfz_sampler._get_synth(22050)


def _warm_note_samples() -> None:
    """Pre-render the clickable keyboard's notes so the first press is instant."""
    import soundfile as sf
    from app.routers.export import _NOTE_CACHE
    from app.services.sampler import render_notes

    _NOTE_CACHE.mkdir(parents=True, exist_ok=True)
    sr, made = 22050, 0
    for midi in range(_NOTE_WARM_LO, _NOTE_WARM_HI + 1):
        out = _NOTE_CACHE / f"n{midi}_v80.wav"
        if out.exists():
            continue
        audio = render_notes(
            [{"pitch": midi, "onset": 0.0, "offset": 0.9, "velocity": 80}], sr=sr
        )
        sf.write(str(out), audio, sr, subtype="PCM_16")
        made += 1
    _state.setdefault("_notes_rendered", made)


_STEPS = [
    ("transcriber", _warm_transcriber),
    ("separator",   _warm_separator),
    ("madmom",      _warm_madmom),
    ("sampler",     _warm_sampler),
    ("first_pass",  _warm_first_pass),
    ("note_samples", _warm_note_samples),
]


def _run() -> None:
    t0 = time.perf_counter()
    for name, fn in _STEPS:
        _record(name, fn)
    _state["total_seconds"] = round(time.perf_counter() - t0, 3)
    _state["done"] = True
    ok = sum(1 for k, v in _state.items() if isinstance(v, dict) and v.get("ok"))
    print(f"[warmup] complete in {_state['total_seconds']}s ({ok} components ready)")


def start() -> None:
    """Kick warming off once. Safe to call from anywhere, any number of times."""
    global _started, _thread
    with _lock:
        if _started:
            return
        _started = True
        _state["done"] = False
        _thread = threading.Thread(target=_run, name="warmup", daemon=True)
        _thread.start()


def status() -> Dict:
    """Per-component readiness, for the UI to show and for debugging."""
    ready = bool(_state.get("done"))
    return {
        "started": _started,
        "ready": ready,
        "components": {k: v for k, v in _state.items() if isinstance(v, dict)},
        "total_seconds": _state.get("total_seconds"),
        "notes_rendered": _state.get("_notes_rendered", 0),
    }
