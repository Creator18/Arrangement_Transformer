from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.services.session import create_session, get_session_dir
from app.services.audio_utils import convert_to_wav, get_audio_duration

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aiff"}


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    session_id = create_session()
    session_dir = get_session_dir(session_id)

    original_path = session_dir / f"original{ext}"
    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    wav_path = session_dir / "audio.wav"
    if ext != ".wav":
        convert_to_wav(original_path, wav_path)
    else:
        shutil.copy(original_path, wav_path)

    duration = get_audio_duration(wav_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "duration_seconds": round(duration, 2),
        "wav_path": str(wav_path),
    }
