from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

from app.services.session import get_session_dir
from app.services.transcription import run_basic_pitch, refine_notes, get_octave_range
from app.services.audio_utils import detect_tempo_and_beats

router = APIRouter()


@router.post("/transcribe/{session_id}")
async def transcribe_audio(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path = session_dir / "audio.wav"

    if not wav_path.exists():
        raise HTTPException(404, "Session audio not found. Upload first.")

    notes = run_basic_pitch(wav_path)
    tempo, beat_times = detect_tempo_and_beats(wav_path)
    notes = refine_notes(notes, tempo)
    octave_range = get_octave_range(notes)

    # Invalidate cached transcription audio so next playback re-renders
    stale = session_dir / "transcription_preview.wav"
    if stale.exists():
        stale.unlink()

    result = {
        "session_id": session_id,
        "note_count": len(notes),
        "notes": notes,
        "octave_range": octave_range,
        "tempo_bpm": round(tempo, 1),
        "beat_times": beat_times[:20],
    }

    # Persist transcription for downstream steps
    with open(session_dir / "transcription.json", "w") as f:
        json.dump(result, f)

    return result
