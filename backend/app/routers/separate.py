from fastapi import APIRouter, HTTPException
import json

from app.services.session import get_session_dir
from app.services.separation import run_spleeter

router = APIRouter()


@router.post("/separate/{session_id}")
async def separate_stems(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path = session_dir / "audio.wav"

    if not wav_path.exists():
        raise HTTPException(404, "Session audio not found. Upload first.")

    stem_paths = run_spleeter(wav_path, session_dir, stems=2)

    result = {
        "session_id": session_id,
        "stems": {k: str(v) for k, v in stem_paths.items()},
    }

    with open(session_dir / "stems.json", "w") as f:
        json.dump(result, f)

    return result
