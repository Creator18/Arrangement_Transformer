from fastapi import APIRouter, HTTPException

from app.services.session import get_session_dir
from app.services.segmentation import analyse_audio

router = APIRouter()


@router.post("/segment/{session_id}")
async def segment_audio(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path    = session_dir / "audio.wav"

    if not wav_path.exists():
        raise HTTPException(404, "Session audio not found. Upload first.")
    if not (session_dir / "components.json").exists():
        raise HTTPException(404, "Components not found. Run /separate first.")

    result = analyse_audio(wav_path, session_dir)

    return result
