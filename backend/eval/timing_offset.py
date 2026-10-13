"""Do an arranger's notes land early or late?  Signed onset offsets vs the human arrangement.

The user heard some Slakh-only notes "off in terms of timing".  For each
model's arrangement of the 106 Music2MIDI test songs (cached by
eval/group_fit.py), match its notes to the human arrangement's (same pitch,
within 150 ms) and report the signed offset (+ = model late), its spread, and
the share beyond one 16 ms frame and beyond 50 ms.  The human arrangements are
score-to-audio aligned to ~0.25 s, so the constant part is shared by all
models; what differs between models is the comparison.

    python eval/timing_offset.py --notes /scratch/.../group_fit --m2m /scratch/.../m2m
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--m2m", required=True)
    args = ap.parse_args()
    from mir_eval.transcription import match_notes
    m2m, nd = pathlib.Path(args.m2m), pathlib.Path(args.notes)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    models = sorted({f.name.split("__")[0] for f in nd.glob("*__*.npy")})
    iv = lambda n: np.c_[n[:, 0], np.maximum(n[:, 1], n[:, 0] + 1e-3)]
    hz = lambda n: 440 * 2 ** ((n[:, 2] - 69) / 12)
    print(f"{'model':<8}{'matched':>9}{'median ms':>11}{'IQR ms':>16}{'|off|>16ms':>12}{'|off|>50ms':>12}"
          f"{'  rel. to v1 (median per song)':>32}")
    per_song = {}
    for mdl in models:
        offs, songmed = [], {}
        for sid in sp["test"]:
            f = nd / f"{mdl}__{sid}.npy"
            if not f.exists():
                continue
            est = np.load(f); ref = np.load(m2m / "midi_numpy" / f"{sid}.npy")
            if len(est) < 10:
                continue
            pairs = match_notes(iv(ref), hz(ref), iv(est), hz(est), onset_tolerance=0.15, offset_ratio=None)
            d = np.array([est[j, 0] - ref[i, 0] for i, j in pairs]) * 1000
            offs.append(d); songmed[sid] = np.median(d) if len(d) else np.nan
        per_song[mdl] = songmed
        d = np.concatenate(offs)
        rel = ""
        if mdl != "v1" and "v1" in per_song:
            common = [s for s in songmed if s in per_song["v1"]]
            diff = np.array([songmed[s] - per_song["v1"][s] for s in common])
            rel = f"{np.nanmedian(diff):+.1f} ms (later on {np.mean(diff > 0):.0%} of songs)"
        print(f"{mdl:<8}{len(d):>9}{np.median(d):>11.1f}{np.percentile(d, 25):>8.1f}..{np.percentile(d, 75):<6.1f}"
              f"{np.mean(np.abs(d) > 16):>12.1%}{np.mean(np.abs(d) > 50):>12.1%}   {rel}")


if __name__ == "__main__" and "--audio-align" not in sys.argv:
    main()


def audio_alignment(notes_dir, m2m, models, n_songs=40):
    """Share of model note onsets within 25 / 50 ms of an attack in the SONG audio, and median distance."""
    import librosa
    from scipy.signal import find_peaks
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    res = {m: [] for m in models + ["human"]}
    for sid in sp["test"][:n_songs]:
        y, sr = librosa.load(str(m2m / "audio" / f"{sid}.wav"), sr=22050, mono=True)
        e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128, lag=1, max_size=1)
        e = (e - e.mean()) / (e.std() + 1e-9)
        pk, _ = find_peaks(e, height=1.0, distance=int(0.04 * sr / 128))
        tp = pk * 128 / sr - 0.026                      # envelope peak -> attack (MAESTRO-measured bias)
        for who in models + ["human"]:
            f = notes_dir / f"{who}__{sid}.npy"
            n = np.load(m2m / "midi_numpy" / f"{sid}.npy") if who == "human" else (np.load(f) if f.exists() else None)
            if n is None or len(n) < 10:
                continue
            on = np.unique(np.round(n[:, 0], 3))
            j = np.clip(np.searchsorted(tp, on), 1, len(tp) - 1)
            d = np.minimum(np.abs(on - tp[j - 1]), np.abs(on - tp[j]))
            res[who].append(d)
    print(f"\nnote onsets vs attacks in the SONG audio ({n_songs} test songs)")
    print(f"{'model':<8}{'onsets':>8}{'<=25 ms':>9}{'<=50 ms':>9}{'median ms':>11}")
    for who, ds in res.items():
        if ds:
            d = np.concatenate(ds) * 1000
            print(f"{who:<8}{len(d):>8}{np.mean(d <= 25):>9.1%}{np.mean(d <= 50):>9.1%}{np.median(d):>11.1f}")


if __name__ == "__main__" and "--audio-align" in sys.argv:
    nd = pathlib.Path(sys.argv[sys.argv.index("--notes") + 1]); m2 = pathlib.Path(sys.argv[sys.argv.index("--m2m") + 1])
    audio_alignment(nd, m2, sorted({f.name.split("__")[0] for f in nd.glob("*__*.npy")}))
