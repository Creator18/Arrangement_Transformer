from fastapi import APIRouter, HTTPException
import json

from app.services.session import get_session_dir
from app.services.separation import separate_components

router = APIRouter()


@router.post("/separate/{session_id}")
async def separate_stems(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path    = session_dir / "audio.wav"

    if not wav_path.exists():
        raise HTTPException(404, "Session audio not found. Upload first.")

    metadata = separate_components(wav_path, session_dir)

    return {"session_id": session_id, **metadata}
