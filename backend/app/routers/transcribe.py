from fastapi import APIRouter, HTTPException
import json

from app.services.session import get_session_dir
from app.services.transcription import transcribe_components, get_octave_range

router = APIRouter()


@router.post("/transcribe/{session_id}")
def transcribe_audio(session_id: str):
    session_dir = get_session_dir(session_id)

    if not (session_dir / "components.json").exists():
        raise HTTPException(404, "Components not found. Run /separate first.")

    notes, timing_feel = transcribe_components(session_dir)
    octave_range       = get_octave_range(notes)

    # Load tempo from components.json (detected during separation)
    with open(session_dir / "components.json") as f:
        comp_meta = json.load(f)
    tempo = float(comp_meta.get("tempo_bpm", 120.0))

    # Invalidate cached audio
    stale = session_dir / "transcription_preview.wav"
    if stale.exists():
        stale.unlink()

    result = {
        "session_id":   session_id,
        "note_count":   len(notes),
        "notes":        notes,
        "octave_range": octave_range,
        "tempo_bpm":    round(tempo, 1),
        "timing_feel":  timing_feel,
        # Damper-down intervals, kept as their own channel.  Note offsets
        # mean KEY RELEASE; how long the sound lasts is decided here and
        # applied at render time, never stored in the note.
        "pedal":        timing_feel.get("pedal", []),
    }

    with open(session_dir / "transcription.json", "w") as f:
        json.dump(result, f)

    return result
