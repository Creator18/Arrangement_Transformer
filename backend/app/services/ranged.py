"""
HTTP Range support for audio responses.

Why this file exists
--------------------
Starlette 0.37.2's FileResponse has no Range handling at all — no
Accept-Ranges header, no 206 replies.  (It gained them in 0.45.)  Every
audio endpoint here used FileResponse, so the browser received one
non-seekable stream.

That is what made seeking "jump back to the beginning": setting
`audio.currentTime` on a resource the server will not serve ranges for
leaves the element with nowhere to seek to, so it restarts from zero.  The
bug looked like a frontend bug and lived entirely in the response headers.

Serving a range correctly needs three things, and browsers are strict about
all of them: Accept-Ranges on the full response so the element knows seeking
is possible at all, a 206 with Content-Range for a partial request, and a
Content-Length matching the bytes actually sent.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.requests import Request

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 64 * 1024


def ranged_file_response(
    request: Request,
    path: Path,
    media_type: str = "audio/wav",
    filename: Optional[str] = None,
) -> FileResponse | StreamingResponse:
    """
    Serve `path`, honouring a Range header when the client sends one.

    Falls back to a normal 200 FileResponse (with Accept-Ranges advertised)
    when there is no Range header, so nothing that worked before changes.
    """
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")

    size = os.path.getsize(path)
    headers = {"Accept-Ranges": "bytes"}
    if filename:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'

    raw = request.headers.get("range") or request.headers.get("Range")
    if not raw:
        return FileResponse(str(path), media_type=media_type, headers=headers)

    m = _RANGE_RE.match(raw.strip())
    if not m:
        return FileResponse(str(path), media_type=media_type, headers=headers)

    start_s, end_s = m.groups()
    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    else:
        # "bytes=-N" means the LAST n bytes, not "from 0 to N".  Getting this
        # backwards serves the wrong audio rather than failing loudly.
        if not end_s:
            return FileResponse(str(path), media_type=media_type, headers=headers)
        start = max(0, size - int(end_s))
        end = size - 1

    if start >= size:
        # Out of range: must be 416, with Content-Range naming the real size.
        raise HTTPException(
            416, "Requested Range Not Satisfiable",
            headers={"Content-Range": f"bytes */{size}"},
        )
    end = min(end, size - 1)
    length = end - start + 1

    def stream():
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(_CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    headers.update({
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Content-Length": str(length),
    })
    return StreamingResponse(stream(), status_code=206,
                             media_type=media_type, headers=headers)
