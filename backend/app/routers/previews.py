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

from fastapi import APIRouter, HTTPException

import soundfile as sf

from app.services.session import get_session_dir
from app.services.sampler import render_notes


router = APIRouter()

PREVIEW_SR = 22050


@router.post("/previews/{session_id}")
async def render_previews(session_id: str, force: bool = False):
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

    # ── transcription preview ──────────────────────────────────────────────
    out_path = session_dir / "transcription_preview.wav"
    if force or not out_path.exists():
        with open(transcription_file) as f:
            data = json.load(f)
        notes = data.get("notes", [])
        t0 = time.time()
        audio = render_notes(notes, sr=PREVIEW_SR)
        sf.write(str(out_path), audio, PREVIEW_SR, subtype="PCM_16")
        summary["tracks"]["original_arrangement"] = {
            "notes": len(notes),
            "render_s": round(time.time() - t0, 2),
            "cached":   False,
        }
    else:
        summary["tracks"]["original_arrangement"] = {"cached": True}

    # ── transformed preview ────────────────────────────────────────────────
    out_path = session_dir / "arrangement_preview.wav"
    if force or not out_path.exists():
        with open(transformed_file) as f:
            data = json.load(f)
        notes = data.get("transformed_notes", [])
        t0 = time.time()
        audio = render_notes(notes, sr=PREVIEW_SR)
        sf.write(str(out_path), audio, PREVIEW_SR, subtype="PCM_16")
        summary["tracks"]["transformed"] = {
            "notes": len(notes),
            "render_s": round(time.time() - t0, 2),
            "cached":   False,
        }
    else:
        summary["tracks"]["transformed"] = {"cached": True}

    return summary
