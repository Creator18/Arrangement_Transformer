"""Human labels for every rendered POP909 clip, frame-aligned with its on/fr rolls (CPU).

POP909 annotates chords (chord_midi.txt, MIDI seconds), key (key_audio.txt) and
has a separate MELODY track.  For each clip pop<NNN>_<w>.npz (eval/render_pop909.py)
this writes pop<NNN>_<w>.lab.npz next to it:

    root  (T,) int8   0-11 chord root (C=0), 12 = no chord
    qual  (T,) int8   0 maj, 1 min, 2 sus, 3 seventh (7 / maj7 / min7 ...), 4 other, 5 none
    mel   (88, T) u8  melody-track notes sounding (the human melody label)
    key   ()    int8  0-11 tonic of the first key, +12 if minor

Frames are shifted by the clip's piano latency exactly like its note labels
(render.frame), so a label sits on the same frame as the sound.

    python eval/pop909_labels.py --pop909 <POP909 dir> --clips /scratch/.../pop909_render
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SECONDS = 20.0
PC = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11, "Fb": 4, "E#": 5, "B#": 0})


def chord_class(label: str):
    if label in ("N", "X", ""):
        return 12, 5
    root, _, q = label.partition(":")
    q = q.split("/")[0].split("(")[0]
    r = PC.get(root, 12)
    if q in ("maj",):
        c = 0
    elif q in ("min",):
        c = 1
    elif q.startswith("sus"):
        c = 2
    elif "7" in q or q in ("9", "maj9", "min9", "11", "13"):
        c = 3
    else:
        c = 4
    return r, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True); ap.add_argument("--clips", required=True)
    a = ap.parse_args()
    import pretty_midi
    from app.ml.spectral import frontend as S, render as R
    lat = R.latencies()
    T = S.n_frames(int(SECONDS * S.SR))
    n = 0
    for clip in sorted(pathlib.Path(a.clips).glob("*/pop*_[0-9][0-9][0-9].npz")):
        dst = clip.with_name(clip.stem + ".lab.npz")
        if dst.exists():
            continue
        song, w = clip.stem[3:].rsplit("_", 1); w = int(w); t0 = w * SECONDS
        d = pathlib.Path(a.pop909) / song
        piano = str(np.load(clip)["piano"]); L = lat[piano]
        root = np.full(T, 12, np.int8); qual = np.full(T, 5, np.int8)
        for line in (d / "chord_midi.txt").read_text().splitlines():
            p = line.split()
            if len(p) < 3:
                continue
            s, e = float(p[0]) - t0, float(p[1]) - t0
            if e <= 0 or s >= SECONDS:
                continue
            r, c = chord_class(p[2])
            fa, fb = max(0, R.frame(max(0.0, s), L)), min(T, R.frame(min(SECONDS, e), L))
            root[fa:fb] = r; qual[fa:fb] = c
        mel = np.zeros((88, T), np.uint8)
        pm = pretty_midi.PrettyMIDI(str(d / f"{song}.mid"))
        for inst in pm.instruments:
            if inst.name.upper() != "MELODY":
                continue
            for nt in inst.notes:
                s, e = nt.start - t0, nt.end - t0
                k = nt.pitch - 21
                if e <= 0 or s >= SECONDS or not 0 <= k < 88:
                    continue
                fa, fb = max(0, R.frame(max(0.0, s), L)), min(T, R.frame(min(SECONDS, e), L))
                mel[k, fa:max(fb, fa + 1)] = 1
        kl = (d / "key_audio.txt").read_text().split()
        key = 0
        if len(kl) >= 3:
            kr, _, km = kl[2].partition(":")
            key = PC.get(kr, 0) + (12 if km.startswith("min") else 0)
        t = dst.with_name(dst.stem + ".tmp.npz")
        np.savez_compressed(t, root=root, qual=qual, mel=mel, key=np.int8(key)); t.replace(dst); n += 1
    print(f"{n} label files written")


if __name__ == "__main__":
    main()
