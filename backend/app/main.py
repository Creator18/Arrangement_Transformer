import os
from pathlib import Path

# scipy 1.8+ removed scipy.signal.hann; librosa still calls the old name
import scipy.signal
import scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

# Register bundled FFmpeg on PATH so all subprocesses (Spleeter etc.) find it
_repo = Path(__file__).parent.parent.parent  # Arrangement_Transformer/
for _candidate in [
    _repo / "ffmpeg-8.1.1-essentials_build" / "ffmpeg-8.1.1-essentials_build" / "bin",
    _repo / "ffmpeg-8.1.1-full_build-shared" / "ffmpeg-8.1.1-full_build-shared" / "bin",
]:
    if _candidate.exists():
        os.environ["PATH"] = str(_candidate) + os.pathsep + os.environ.get("PATH", "")
        break

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import upload, transcribe, separate, segment, transform, score, export, previews, progress
from app.services.session import TEMP_BASE
from app.services import warmup

app = FastAPI(
    title="Arrangement Transformer API",
    version="1.0.0",
    description="Transforms piano audio into arrangements for different octave ranges",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["upload"])
app.include_router(transcribe.router, tags=["transcribe"])
app.include_router(separate.router, tags=["separate"])
app.include_router(segment.router, tags=["segment"])
app.include_router(transform.router, tags=["transform"])
app.include_router(previews.router, tags=["previews"])
app.include_router(score.router, tags=["score"])
app.include_router(export.router, tags=["export"])
app.include_router(progress.router, tags=["progress"])


app.mount("/session-data", StaticFiles(directory=str(TEMP_BASE)), name="session-data")


@app.on_event("startup")
def _startup_warm():
    """
    Begin loading models the moment the server is up, on a daemon thread so
    this does not delay the first request.  By the time a user has picked a
    file, the ~4.6s of model construction is usually already paid.
    """
    warmup.start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/warmup")
def warmup_status():
    """Readiness of each heavy component. Also (re)starts warming if idle."""
    warmup.start()
    return warmup.status()
