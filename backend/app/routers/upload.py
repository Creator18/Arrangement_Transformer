from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import re
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


def _reachable(host, timeout=4.0):
    import socket
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _use_reachable_server(info):
    """for each googlevideo server YouTube picked that can't be reached, move its stream links onto the first backup
    server (the link's mn parameter) that can.  Leaves other links (manifests, other sites) alone."""
    from urllib.parse import urlsplit, parse_qs
    ok = {}
    for f in info.get("formats") or []:
        u = urlsplit(f.get("url") or "")
        h = u.hostname or ""
        if not (h.startswith("rr") and "---" in h and h.endswith(".googlevideo.com")):
            continue
        if h not in ok:
            ok[h] = h if _reachable(h) else None
            if ok[h] is None:
                rr = h.split("---")[0]                                 # e.g. rr3
                for sn in (parse_qs(u.query).get("mn") or [""])[0].split(","):
                    b = f"{rr}---{sn}.googlevideo.com"
                    if sn and b != h and _reachable(b):
                        ok[h] = b
                        break
                print(f"[download] {h} unreachable: " + (f"using backup server {ok[h]}" if ok[h] else "no backup answered"))
        if ok[h] and ok[h] != h:
            f["url"] = u._replace(netloc=u.netloc.replace(h, ok[h])).geturl()


class UrlIn(BaseModel):
    url: str


@router.post("/upload-url")
def upload_url(body: UrlIn):
    """Fetch the audio of a YouTube (or other yt-dlp supported) link and start a session from it."""
    url = body.url.strip()
    if not re.match(r"^https?://", url):
        raise HTTPException(400, "Paste a full http(s) link.")
    import yt_dlp

    session_id = create_session()
    session_dir = get_session_dir(session_id)
    opts = {"format": "bestaudio/best", "outtmpl": str(session_dir / "original.%(ext)s"),
            "noplaylist": True, "quiet": True, "no_warnings": True,
            "max_filesize": 200 * 1024 * 1024, "match_filter": yt_dlp.utils.match_filter_func("duration < 1200")}
    # one unreachable video server must not fail the fetch (2026-10-28: a link's googlevideo cache node timed out on
    # every try, IPv4 and IPv6, whatever client asked).  YouTube's stream links name backup servers (the mn
    # parameter): when the given server can't be reached, try the same link on each backup.
    print(f"[download] {url}")
    try:
        with yt_dlp.YoutubeDL({**opts, "socket_timeout": 10}) as ydl:
            info = ydl.extract_info(url, download=False, process=False)   # formats not chosen yet
            _use_reachable_server(info)
            info = ydl.process_ie_result(info, download=True)
    except Exception as e:
        raise HTTPException(400, f"Couldn't fetch audio from that link: {str(e).splitlines()[0][:200]}")
    files = [p for p in session_dir.glob("original.*") if not p.name.endswith(".part")]
    if not files:
        raise HTTPException(400, "No audio was downloaded (videos over 20 minutes are skipped).")

    wav_path = session_dir / "audio.wav"
    convert_to_wav(files[0], wav_path)
    duration = get_audio_duration(wav_path)
    return {
        "session_id": session_id,
        "filename": info.get("title") or url,
        "duration_seconds": round(duration, 2),
        "wav_path": str(wav_path),
    }
