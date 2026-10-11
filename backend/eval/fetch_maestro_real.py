"""Fetch a few REAL MAESTRO recordings without downloading the 108 GB archive.

Every model in this project has been trained and scored on audio rendered
through sample libraries.  Whether it works on a real piano, in a real room,
through a real microphone, has never been measured.  MAESTRO's real
recordings ship only as one ~108 GB zip, but the host honours HTTP range
requests and a zip keeps its index at the end — so the index is read, and
then only the chosen members are pulled out, each one a contiguous byte range.

Chooses test-split pieces (never used in training) of moderate length, spread
across competition years, since each year's recordings came from a somewhat
different setup.  Writes <out>/<name>.wav plus the matching MIDI.

    python eval/fetch_maestro_real.py --out E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import random
import shutil
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MAESTRO = ROOT.parent / "data" / "maestro-v3.0.0"


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP byte ranges."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-s", type=float, default=180)
    ap.add_argument("--max-s", type=float, default=420)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(MAESTRO / "maestro-v3.0.0.csv", encoding="utf-8"))
            if r["split"] == "test" and args.min_s <= float(r["duration"]) <= args.max_s]
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)
    rng = random.Random(0)
    for v in by_year.values():
        rng.shuffle(v)
    chosen, years = [], sorted(by_year)
    while len(chosen) < args.n and any(by_year.values()):     # round-robin over years
        for y in years:
            if by_year[y] and len(chosen) < args.n:
                chosen.append(by_year[y].pop())
    print(f"{len(chosen)} test pieces, {sum(float(r['duration']) for r in chosen) / 60:.0f} min, "
          f"years {sorted({r['year'] for r in chosen})}", flush=True)

    f = io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20)
    zf = zipfile.ZipFile(f)
    names = {n.split("/", 1)[-1]: n for n in zf.namelist()}
    total = 0
    for r in chosen:
        stem = pathlib.Path(r["audio_filename"]).stem
        wav = out / f"{stem}.wav"
        if not wav.exists():
            member = names[r["audio_filename"]]
            tmp = wav.with_suffix(".part")
            with zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=4 << 20)
            tmp.replace(wav)
        shutil.copy(MAESTRO / r["midi_filename"], out / f"{stem}.midi")
        total += wav.stat().st_size
        print(f"  {r['year']}  {float(r['duration']) / 60:4.1f} min  {r['canonical_composer'][:22]:<22} "
              f"{wav.stat().st_size / 1e6:6.0f} MB", flush=True)
    print(f"done: {len(chosen)} recordings, {total / 1e9:.2f} GB in {out}")


if __name__ == "__main__":
    main()
