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
    "echo GPU; head -1 ~/arrangement_transformer/logs/solo_train_v3.out 2>/dev/null;"
    "echo REF v1; grep '^VAL' ~/arrangement_transformer/checkpoints/solo_det/train.log 2>/dev/null;"
    "echo REF v2; grep '^VAL' ~/arrangement_transformer/checkpoints/solo_det2/train.log 2>/dev/null;"
    "echo LOG; grep -E '^(VAL|step [0-9]*00 |data:|SoloNet|new best|\(finished\))' "
    "~/arrangement_transformer/checkpoints/solo_det3/train.log 2>/dev/null | tail -400"
)
_vcache = {"at": 0.0, "data": None}
_vgood = {"live": None, "time": None}      # last complete read from the cluster: kept when an ssh read fails


@router.get("/voice-data")
def voice_data():
    with _lock:
        if _vcache["data"] is not None and time.time() - _vcache["at"] < TTL:
            return _vcache["data"]
    data = json.loads(VOICE_JSON.read_text())
    live = {"curves": None, "songs": None, "jobs": [], "vals": [], "trace": [], "info": [], "finished": False,
            "run": "v3 (song-relative: band silent, or in the bottom 30% of the song's range)", "gpu": None,
            "ref": {"v1": [], "v2": []}}
    try:
        out = subprocess.run(SSH + [VOICE_LIVE], capture_output=True, text=True, timeout=60).stdout
        sec = None
        for ln in out.splitlines():
            if ln.startswith("CURVES"):
                _, c, s = ln.split()
                live["curves"], live["songs"] = int(c), int(s)
            elif ln in ("JOBS", "LOG", "GPU") or ln.startswith("REF "):
                sec = ln
            elif sec == "JOBS" and "|" in ln:
                i, n, st, el, lim, gres, why = ln.split("|", 6)
                live["jobs"].append({"id": i, "name": n, "state": st, "elapsed": el, "limit": lim, "reason": why})
            elif sec == "GPU" and ln.strip():
                live["gpu"] = ln.strip()
            elif sec and sec.startswith("REF ") and ln.startswith("VAL "):
                live["ref"][sec[4:]].append(json.loads(ln[4:]))
            elif sec == "LOG":
                if ln.startswith("VAL "):
                    live["vals"].append(json.loads(ln[4:]))
                elif ln.startswith("step "):
                    p = ln.split()
                    live["trace"].append([int(p[1]), float(p[3]), float(p[6])])
                elif ln.startswith("(finished)"):
                    live["finished"] = True
                else:
                    live["info"].append(ln)
    except subprocess.TimeoutExpired:
        live["error"] = "cluster did not answer"
    if live["curves"] is None or (not live["vals"] and not live["trace"] and _vgood["live"]):
        # the ssh read failed or came back cut short (Explorer's login node is sometimes slow): keep the last good one
        if _vgood["live"]:
            live = dict(_vgood["live"], stale=_vgood["time"])
    else:
        _vgood.update(live=live, time=datetime.now().strftime("%H:%M:%S"))
    data["live"] = live
    _vcache.update(at=time.time(), data=data)
    return data


# ---- clean-up model (keep / drop the base's 0.8 notes; app/ml/spectral/cleanup_train.py): live training health ----
CLEANUP_LIVE = (
    "echo DATA $(ls /scratch/basu.ang/cleanup_data 2>/dev/null | grep -c 'npz$') $(ls /scratch/basu.ang/base_render/validation | grep -c 'npz$');"
    "echo DATA2 $(ls /scratch/basu.ang/cleanup_data_tr 2>/dev/null | grep -c 'npz$') $(grep -c . /scratch/basu.ang/quant/train_pick.txt 2>/dev/null)"
    " $(ls /scratch/basu.ang/clip_beats 2>/dev/null | grep -c 'npz$');"
    "echo JOBS; squeue -u $USER -h -r -o '%i|%j|%T|%M|%l|%N|%R' | grep -E 'cudata|cutrain|cudtr|beats-' | sort -t'|' -k2,2 -u ;"
    "f=$(ls -t ~/arrangement_transformer/logs/cutrain2_[0-9]*.out ~/arrangement_transformer/logs/cutrain_[0-9]*.out 2>/dev/null | head -1); echo LOG $f;"
    "[ -n \"$f\" ] && grep -E '^(VAL|step [0-9]+ loss|[0-9]+ clips|[0-9.]+M params|resumed|done)' $f | tail -600"
)
_ccache = {"at": 0.0, "data": None}
_cgood = {"data": None, "time": None}


@router.get("/cleanup-data")
def cleanup_data():
    with _lock:
        if _ccache["data"] is not None and time.time() - _ccache["at"] < TTL:
            return _ccache["data"]
    d = {"data": None, "clips": None, "jobs": [], "vals": [], "trace": [], "info": [], "finished": False, "log": None,
         "version": 1, "total": 30000, "data2": None}
    try:
        out = subprocess.run(SSH + [CLEANUP_LIVE], capture_output=True, text=True, timeout=60).stdout
        sec, prev = None, None
        for ln in out.splitlines():
            if ln.startswith("DATA "):
                _, n, tot = ln.split()
                d["data"], d["clips"] = int(n), int(tot)
            elif ln.startswith("DATA2 "):
                v = [int(x) for x in ln.split()[1:]]
                d["data2"] = {"clips": v[0], "of": v[1], "beats": v[2], "beats_of": 34262 + v[1]} if len(v) == 3 else None
            elif ln == "JOBS":
                sec = ln
            elif ln.startswith("LOG"):
                sec, d["log"] = "LOG", ln[4:].strip() or None
                if d["log"] and "cutrain2_" in d["log"]:
                    d.update(version=2, total=60000)
            elif sec == "JOBS" and "|" in ln:
                i, n, st, el, lim, node, why = ln.split("|", 6)
                d["jobs"].append({"id": i, "name": n, "state": st, "elapsed": el, "limit": lim, "node": node, "reason": why})
            elif sec == "LOG":
                if ln.startswith("VAL step "):
                    step, js = ln[9:].split(" ", 1)
                    d["vals"].append({"step": int(step), **json.loads(js)})
                elif ln.startswith("step "):
                    p = ln.split()                                   # step N loss L [pos Q] M min
                    s, loss = int(p[1]), float(p[3])
                    pos = float(p[5]) if p[4] == "pos" else None
                    mins = float(p[6] if pos is not None else p[4])
                    sps = (s - prev[0]) / max(1e-9, (mins - prev[1]) * 60) if prev and mins > prev[1] else None
                    d["trace"].append([s, loss, sps, pos])
                    prev = (s, mins)
                elif ln.strip() == "done":
                    d["finished"] = True
                else:
                    d["info"].append(ln)
    except subprocess.TimeoutExpired:
        d["error"] = "cluster did not answer"
    if d["data"] is None:
        if _cgood["data"]:
            d = dict(_cgood["data"], stale=_cgood["time"])
    else:
        _cgood.update(data=d, time=datetime.now().strftime("%H:%M:%S"))
    _ccache.update(at=time.time(), data=d)
    return d
