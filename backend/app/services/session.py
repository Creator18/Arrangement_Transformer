import os
import shutil
import uuid
from pathlib import Path

TEMP_BASE = Path(__file__).parent.parent.parent / "temp"
TEMP_BASE.mkdir(exist_ok=True)


def create_session() -> str:
    session_id = str(uuid.uuid4())
    session_dir = TEMP_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id


def get_session_dir(session_id: str) -> Path:
    return TEMP_BASE / session_id


def cleanup_session(session_id: str):
    session_dir = TEMP_BASE / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
