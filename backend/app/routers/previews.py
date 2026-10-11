"""
Preview pre-render endpoint.

Splits the SFZ/SF2 audio render of the two main listening tracks
(original arrangement, transformed arrangement) into its own pipeline
step so the frontend's ProcessingPage can show explicit progress while
Salamander chugs through a few thousand notes.

Renders synchronously and writes:
    session_dir/transcription_preview.wav
    session_dir/arrangement_preview.wav

Cost
----
Salamander render of a 5-min arrangement takes ~60–150 s on CPU,
dominated by the number of notes (each fires release + resonance voices).
By the time this endpoint returns, both WAVs are on disk and the
Listen player's first-click is just a file download.

Idempotency
-----------
Existing WAVs are kept by default; pass ?force=1 to re-render (e.g. after
re-running /segment, which invalidates the original preview).
"""
from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException

import soundfile as sf

from app.services.session import get_session_dir
from app.services.sampler import render_notes


router = APIRouter()

# 44.1 kHz, not 22.05.  At 22 kHz the Nyquist limit is 11 kHz, which discards
# everything above roughly the top octave's third partial — precisely the
# brilliance and hammer noise that make a piano sound like a piano.  Salamander
# ships 48 kHz samples; we were rendering them into mud.
#
# Costs about 2x the render time and 2x the file size.  Render was ~38% of
# upload-to-result, so this is a real trade — made deliberately, because the
# instrument is the product here and the warmup already moved model loading
# off the critical path.
PREVIEW_SR = 44100


def _render_one(session_dir: Path, src_file: Path, notes_key: str,
                out_name: str, force: bool) -> Dict:
    """Render one preview track.  Returns a small summary."""
    out_path = session_dir / out_name
    if not force and out_path.exists():
        return {"cached": True}
    with open(src_file) as f:
        data = json.load(f)
    notes = data.get(notes_key, [])
    # Pedal is its own channel: note offsets are key releases, and the damper
    # decides how long the sound actually lasts.
    pedal = data.get("pedal", [])
    t0 = time.time()
    from app.services.sampler import audible
    audio = render_notes(audible(notes), sr=PREVIEW_SR, pedal=pedal)
    sf.write(str(out_path), audio, PREVIEW_SR, subtype="PCM_16")
    return {"notes": len(notes), "render_s": round(time.time() - t0, 2),
            "cached": False}


@router.post("/previews/{session_id}")
async def render_previews(session_id: str, force: bool = False,
                          only: Optional[str] = None):
    """
    Render `transcription_preview.wav` + `arrangement_preview.wav` for
    the given session.  Assumes /transcribe, /segment, and /transform
    have already run.  Returns a small JSON summary with per-track
    durations so the frontend can show timing.
    """
    session_dir         = get_session_dir(session_id)
    transcription_file  = session_dir / "transcription.json"
    transformed_file    = session_dir / "transformed.json"

    if not transcription_file.exists():
        raise HTTPException(404, "transcription.json missing — run /transcribe first.")
    if not transformed_file.exists():
        raise HTTPException(404, "transformed.json missing — run /transform first.")

    summary = {"session_id": session_id, "tracks": {}}

    # 'only' renders a single track and returns.  The frontend asks for the
    # transformed one first, lands the user on Results, then requests the
    # other in the background — so the visible wait is one render, not two.
    # Rendering both up front cost 38% of the total upload-to-result time
    # while the user can only listen to one of them at a time.
    jobs = [
        ("transformed",          transformed_file,   "transformed_notes",
         "arrangement_preview.wav"),
        ("original_arrangement", transcription_file, "notes",
         "transcription_preview.wav"),
    ]
    if only:
        jobs = [j for j in jobs if j[0] == only] or jobs

    for name, src, key, out_name in jobs:
        summary["tracks"][name] = _render_one(session_dir, src, key, out_name, force)

    return summary
