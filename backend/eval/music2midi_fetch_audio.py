"""Fetch the source audio Music2MIDI references, as 16 kHz mono WAV.

    python eval/music2midi_fetch_audio.py --split val --limit 20
    python eval/music2midi_fetch_audio.py --split all --workers 3

The dataset ships piano covers and YouTube ids, not recordings.  This resolves
the ids.  Requires yt-dlp and ffmpeg on PATH.  The recordings are commercial
masters; what you may do with them is a question about your jurisdiction and
your use, and this script does not answer it.

THE CORRECTNESS TRAP, which matters more than the downloading.

`youtube_csv/<id>.csv` lists SEVERAL candidate videos per song — the official
video, live versions, covers, talent-show performances.  They are not
interchangeable.  The dataset's note times were warped onto ONE specific
recording, and `metadata/<id>.yaml` records which:

    youtube:
      url: https://www.youtube.com/watch?v=y0dbfkpWkUo
      duration: 219.37

Download a different video of the same song and every target is misaligned by
however much the two arrangements differ — a different intro length, a slower
tempo, an extra chorus.  Training would run to completion and converge on
nothing, which is the expensive kind of failure because it looks like the
ordinary kind.  So this reads the chosen url from the metadata and never
touches the candidate list.

It then VERIFIES the result: a download whose duration disagrees with the
metadata by more than DURATION_TOL_S is quarantined rather than kept, because
YouTube ids get reused, reuploaded and region-swapped, and a silently wrong
file is worse than a missing one.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import shutil
import subprocess
import sys
import threading

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.transcriber import config as C           # noqa: E402

DATA = C.ROOT / "data" / "music2midi" / "music2midi-dataset"
OUT = C.ROOT / "data" / "music2midi" / "audio"
BAD = C.ROOT / "data" / "music2midi" / "rejected"

# How far a fetched recording may differ from the duration the alignment was
# computed against.  The warp path absorbs small tempo drift; it cannot absorb
# a different edit.  Five seconds is generous for trailing silence and tight
# enough to catch a wrong video.
DURATION_TOL_S = 5.0

_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(*a, flush=True)


def chosen_url(sid: str):
    """(url, expected_duration) from the metadata — NOT from the candidate csv."""
    import yaml
    m = yaml.safe_load((DATA / "metadata" / f"{sid}.yaml").read_text(encoding="utf-8"))
    yt = m.get("youtube") or {}
    return yt.get("url"), float(yt.get("duration") or 0.0)


def ids_for(split: str):
    z = np.load(DATA / "dataset_split.npz", allow_pickle=True)
    keys = {"train": ["train_id"], "val": ["val_id"], "test": ["test_id"],
            "all": ["train_id", "val_id", "test_id"]}[split]
    out = []
    for k in keys:
        out += [str(i) for i in z[k]]
    return out


def duration_of(path: pathlib.Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def fetch(sid: str, overwrite: bool) -> tuple[str, str]:
    dst = OUT / f"{sid}.wav"
    if dst.exists() and not overwrite:
        return sid, "skip"
    url, want = chosen_url(sid)
    if not url:
        return sid, "no-url"

    tmp = OUT / f".{sid}.tmp.wav"
    # Straight to the training format in one pass: 16 kHz mono PCM, matching
    # Slakh redux so both datasets load through identical code and neither
    # pays a resample per batch.
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -sample_fmt s16",
        # Be a reasonable client: one retry burst, a pause between requests.
        "--retries", "3", "--sleep-requests", "1",
        "-o", str(tmp.with_suffix("")) + ".%(ext)s",
        url,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return sid, "timeout"
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        err = (r.stderr or "").strip().splitlines()
        return sid, f"yt-dlp: {err[-1][:80] if err else r.returncode}"
    if not tmp.exists():
        return sid, "no-output"

    try:
        got = duration_of(tmp)
    except Exception as e:                                  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return sid, f"unreadable: {type(e).__name__}"

    if want and abs(got - want) > DURATION_TOL_S:
        # Quarantined, not deleted: a mismatch is worth being able to inspect,
        # and worth never silently training on.
        BAD.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(BAD / f"{sid}.wav"))
        return sid, f"duration {got:.0f}s != {want:.0f}s (rejected)"

    tmp.replace(dst)
    return sid, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many songs -- use it first; 20 songs "
                         "is enough to prove the whole pipeline end to end")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
    if not (DATA / "metadata").is_dir():
        raise SystemExit(f"Music2MIDI not unpacked at {DATA}")

    OUT.mkdir(parents=True, exist_ok=True)
    ids = ids_for(args.split)
    todo = [i for i in ids
            if (DATA / "metadata" / f"{i}.yaml").exists()
            and (args.overwrite or not (OUT / f"{i}.wav").exists())]
    if args.limit:
        todo = todo[: args.limit]
    have = sum(1 for i in ids if (OUT / f"{i}.wav").exists())
    print(f"split {args.split}: {len(ids)} songs, {have} already present, "
          f"{len(todo)} to fetch, {args.workers} workers")

    tally: dict[str, int] = {}
    failures: dict[str, str] = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for sid, status in ex.map(lambda s: fetch(s, args.overwrite), todo):
            done += 1
            key = status if status in ("ok", "skip") else status.split(":")[0]
            tally[key] = tally.get(key, 0) + 1
            if status not in ("ok", "skip"):
                failures[sid] = status
            if status == "ok" or done % 25 == 0:
                log(f"[{done}/{len(todo)}] {sid}: {status}")

    print("\nresult")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28}{v:>6}")
    if failures:
        p = OUT.parent / "fetch_failures.json"
        p.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"\n{len(failures)} failures written to {p}")
    total = sum(1 for i in ids if (OUT / f"{i}.wav").exists())
    print(f"\n{total}/{len(ids)} songs in {args.split} now have audio")


if __name__ == "__main__":
    main()
