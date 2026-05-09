from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from app.services.session import get_session_dir
from app.services.transformation import notes_to_pretty_midi
from app.services.nmf_voice_leading import transform_nmf, TARGET_LOW, TARGET_HIGH

router = APIRouter()

_OCTAVE_BOUNDS = {
    5: (36, 96),    # C2–C7
    6: (28, 103),   # E1–G7
    7: (21, 108),   # A0–C8 (full 88-key)
}


class TransformRequest(BaseModel):
    target_octave_range: int = 5
    source_octave_range: Optional[int] = None
    nmf_components: int = 16


@router.post("/transform/{session_id}")
async def transform_arrangement(session_id: str, body: TransformRequest = TransformRequest()):
    session_dir = get_session_dir(session_id)

    transcription_file = session_dir / "transcription.json"
    segments_file      = session_dir / "segments.json"

    if not transcription_file.exists():
        raise HTTPException(404, "Transcription not found. Run /transcribe first.")
    if not segments_file.exists():
        raise HTTPException(404, "Segments not found. Run /segment first.")

    with open(transcription_file) as f:
        transcription = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    notes       = transcription["notes"]
    tempo       = seg_data["tempo_bpm"]
    source_span = body.source_octave_range or transcription["octave_range"]["octave_span"]

    tgt_low, tgt_high = _OCTAVE_BOUNDS.get(body.target_octave_range, (TARGET_LOW, TARGET_HIGH))

    transformed_notes, scale_info = transform_nmf(
        notes,
        target_low=tgt_low,
        target_high=tgt_high,
        n_components=body.nmf_components,
        tempo=tempo,
    )

    for stale in ["arrangement_preview.wav", "transcription_preview.wav"]:
        p = session_dir / stale
        if p.exists():
            p.unlink()

    pm = notes_to_pretty_midi(transformed_notes, tempo)
    midi_path = session_dir / "transformed.mid"
    pm.write(str(midi_path))

    result = {
        "session_id": session_id,
        "method": "nmf_voice_leading",
        "source_octave_range": source_span,
        "target_octave_range": body.target_octave_range,
        "nmf_components": body.nmf_components,
        "original_note_count": len(notes),
        "transformed_note_count": len(transformed_notes),
        "transformed_notes": transformed_notes,
        "detected_scale": scale_info,
        "midi_path": str(midi_path),
    }

    with open(session_dir / "transformed.json", "w") as f:
        json.dump(result, f)

    return result
