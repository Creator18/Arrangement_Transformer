from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

from app.services.session import get_session_dir
from app.services.segmentation import segment_by_measures, detect_time_signature
from app.services.audio_utils import detect_tempo_and_beats

router = APIRouter()


@router.post("/segment/{session_id}")
async def segment_audio(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path = session_dir / "audio.wav"

    stems_file = session_dir / "stems.json"
    if not stems_file.exists():
        raise HTTPException(404, "Stems not found. Run /separate first.")

    with open(stems_file) as f:
        stems_data = json.load(f)

    bass_path = Path(stems_data["stems"].get("bass", ""))
    melody_path = Path(stems_data["stems"].get("melody", ""))

    if not bass_path.exists() or not melody_path.exists():
        raise HTTPException(500, "Stem files missing from disk.")

    tempo, _ = detect_tempo_and_beats(wav_path)
    time_sig_num, time_sig_den = detect_time_signature(wav_path)

    segments = segment_by_measures(wav_path, bass_path, melody_path, tempo, time_sig_num)

    result = {
        "session_id": session_id,
        "tempo_bpm": round(tempo, 1),
        "time_signature": f"{time_sig_num}/{time_sig_den}",
        "segment_count": len(segments),
        "segments": segments,
    }

    with open(session_dir / "segments.json", "w") as f:
        json.dump(result, f)

    return result
