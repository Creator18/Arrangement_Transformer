"""Time the app's per-song steps against a running container and record its memory (docker stats).

    python deploy/bench.py --url http://localhost:8001 --container arr-bench song1.webm song2.webm ...

Runs the same calls as the frontend: upload -> separate -> transcribe -> segment -> transform -> previews (both).
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def call(url, method="POST", body=None, headers=None, timeout=3600):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def upload(base, path):
    b = uuid.uuid4().hex
    data = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    return call(f"{base}/upload", body=data, headers={"Content-Type": f"multipart/form-data; boundary={b}"})


def mem_mib(name):
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         capture_output=True, text=True).stdout.split("/")[0].strip()
    n, unit = float(out[:-3]), out[-3:]
    return n * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]


class Peak(threading.Thread):
    def __init__(self, name):
        super().__init__(daemon=True)
        self.name_, self.peak, self.stop = name, 0.0, False

    def run(self):
        while not self.stop:
            try:
                self.peak = max(self.peak, mem_mib(self.name_))
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--container", default="arr-bench")
    ap.add_argument("songs", nargs="+")
    a = ap.parse_args()
    base = a.url
    print(f"idle memory {mem_mib(a.container):.0f} MiB")
    rows = []
    for s in map(Path, a.songs):
        p = Peak(a.container); p.start()
        t = {}
        t0 = time.perf_counter(); sid = upload(base, s)["session_id"]; t["upload"] = time.perf_counter() - t0
        for step, path in [("separate", f"/separate/{sid}"), ("transcribe", f"/transcribe/{sid}"),
                           ("segment", f"/segment/{sid}")]:
            t0 = time.perf_counter(); call(base + path); t[step] = time.perf_counter() - t0
        t0 = time.perf_counter()
        call(f"{base}/transform/{sid}", body=json.dumps({"target_octave_range": 5, "nmf_components": 16}).encode(),
             headers={"Content-Type": "application/json"})
        t["transform"] = time.perf_counter() - t0
        for only in ("original_arrangement", "transformed"):
            t0 = time.perf_counter(); call(f"{base}/previews/{sid}?only={only}"); t[f"previews:{only}"] = time.perf_counter() - t0
        p.stop = True; p.join()
        notes = len(call(f"{base}/session-data/{sid}/transcription.json", method="GET").get("notes", []))
        dur = float(subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(s)],
                                   capture_output=True, text=True).stdout.strip() or 0)
        total = sum(t.values())
        rows.append(dict(song=s.parent.name[:8], secs=round(dur), notes=notes, total=round(total, 1), peak_mib=round(p.peak),
                         **{k: round(v, 1) for k, v in t.items()}))
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
