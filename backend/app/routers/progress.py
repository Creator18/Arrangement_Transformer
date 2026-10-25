"""Live snapshot of the cluster training pipeline for the /progress page.

The cluster writes ~/arrangement_transformer/progress/latest.json every 5 minutes (eval/progress_dump.py
--write, run by scrontab, which also emails failure alerts).  Reading that file is one quick ssh `cat`;
if it is missing or stale the snapshot is taken on the spot instead.  Cached briefly so several open tabs
don't each open an ssh connection.
"""
import json, subprocess, threading, time
from datetime import datetime

from fastapi import APIRouter, HTTPException

router = APIRouter()
SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "explorer"]
READ = "cat ~/arrangement_transformer/progress/latest.json"
LIVE = "cd ~/arrangement_transformer/backend && python3 eval/progress_dump.py"
STALE_S = 12 * 60     # two missed scheduled runs
TTL = 20
_cache = {"at": 0.0, "data": None}
_lock = threading.Lock()


def _run(cmd):
    try:
        out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "cluster did not answer within 60 s")
    if out.returncode != 0:
        return None, out.stderr.strip()[-300:]
    return json.loads(out.stdout), None


@router.get("/progress-data")
def progress_data():
    with _lock:
        if _cache["data"] is not None and time.time() - _cache["at"] < TTL:
            return _cache["data"]
        data, err = _run(READ)
        age = (datetime.now() - datetime.strptime(data["time"], "%Y-%m-%d %H:%M:%S")).total_seconds() if data else None
        if data is None or age > STALE_S:
            live, err = _run(LIVE)
            if live is None:
                raise HTTPException(502, f"ssh failed: {err}")
            # the scheduled snapshot stopped: say so on the page
            live["schedule_stale"] = data["time"] if data else "never"
            data = live
        _cache.update(at=time.time(), data=data)
        return data
