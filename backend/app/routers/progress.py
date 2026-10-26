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


# ---- voice work (solo-voice fix): fixed test results from eval/voice_tests.json + live cluster status ----
VOICE_JSON = __import__("pathlib").Path(__file__).resolve().parents[2] / "eval" / "voice_tests.json"
VOICE_LIVE = (
    "echo CURVES $(ls /scratch/basu.ang/solo_det/m2m_curves 2>/dev/null | wc -l) $(ls /scratch/basu.ang/m2m/audio | wc -l);"
    "echo JOBS; squeue -u $USER -h -o '%i|%j|%T|%M|%l|%b|%R' | grep -E 'm2m-demucs|solo' ;"
    "echo LOG; grep -E '^(VAL|step [0-9]*00 |data:|SoloNet|new best|\(finished\))' "
    "~/arrangement_transformer/checkpoints/solo_det/train.log 2>/dev/null | tail -400"
)
_vcache = {"at": 0.0, "data": None}


@router.get("/voice-data")
def voice_data():
    with _lock:
        if _vcache["data"] is not None and time.time() - _vcache["at"] < TTL:
            return _vcache["data"]
    data = json.loads(VOICE_JSON.read_text())
    live = {"curves": None, "songs": None, "jobs": [], "vals": [], "trace": [], "info": [], "finished": False}
    try:
        out = subprocess.run(SSH + [VOICE_LIVE], capture_output=True, text=True, timeout=60).stdout
        sec = None
        for ln in out.splitlines():
            if ln.startswith("CURVES"):
                _, c, s = ln.split()
                live["curves"], live["songs"] = int(c), int(s)
            elif ln in ("JOBS", "LOG"):
                sec = ln
            elif sec == "JOBS" and "|" in ln:
                i, n, st, el, lim, gres, why = ln.split("|", 6)
                live["jobs"].append({"id": i, "name": n, "state": st, "elapsed": el, "limit": lim, "reason": why})
            elif sec == "LOG":
                if ln.startswith("VAL "):
                    live["vals"].append(json.loads(ln[4:]))
                elif ln.startswith("step "):
                    p = ln.split()
                    live["trace"].append([int(p[1]), float(p[3]), float(p[7])])
                elif ln.startswith("(finished)"):
                    live["finished"] = True
                else:
                    live["info"].append(ln)
    except subprocess.TimeoutExpired:
        live["error"] = "cluster did not answer"
    data["live"] = live
    _vcache.update(at=time.time(), data=data)
    return data
