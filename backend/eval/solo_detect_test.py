"""Can we tell which bars are solo voice from the mix alone?

Builds POP909 mixes (MELODY as the neutral "aah", PIANO as a pluck tone, eval/aah_test.py) and strips the
accompaniment from random runs of 2-4 bars (~30% of bars): those bars are the truth "solo voice" bars.
Detection: transcribe the mix with an arranger checkpoint at the app bar, split melody vs accompaniment with the
trained split (app/ml/melody_split.joblib), count accompaniment notes per bar, call a bar solo when the count is at
most k.  Bars come from POP909's beat annotations (the app would use its beat tracker).  Scored bar by bar.

    python eval/solo_detect_test.py --ckpt app/ml/spectral/checkpoints/base20m/best.pt --out <dir>
"""
import argparse
import pathlib
import sys

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.services import spectral_transcription as ST  # noqa: E402
from eval.aah_test import SR, aah, pluck  # noqa: E402
from eval.aah_fill_test import ROOT, bars_and_chords  # noqa: E402
from eval.rhythm_groups import melody_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-songs", type=int, default=50)
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--pad", type=float, default=None,
                    help="leave a quiet sustained pad (chord roots + fifths) in solo bars, dB below the voice")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = pathlib.Path(a.ckpt)
    t0, t1 = a.start, a.start + a.secs
    rng = np.random.default_rng(1)
    songs = [f"{i:03d}" for i in sorted(rng.choice(np.arange(1, 910), a.n_songs, replace=False))]
    ks = (0, 1, 2, 3)
    truth, counts, polys = [], [], []
    for s in songs:
        pm = pretty_midi.PrettyMIDI(str(ROOT / s / f"{s}.mid"))
        mel = next(i for i in pm.instruments if i.name.upper().startswith("MELODY")).notes
        pia = next(i for i in pm.instruments if i.name.upper().startswith("PIANO")).notes
        voice = [pretty_midi.Note(n.velocity, n.pitch, n.start - t0, n.end - t0) for n in mel if t0 <= n.start < t1]
        bars = [(b0 - t0, b1 - t0, ch) for b0, b1, ch in bars_and_chords(s) if b0 >= t0 and b1 <= t1]
        if not voice or len(bars) < 6:
            continue
        # truth: runs of 2-4 solo bars, ~30% of bars
        solo = np.zeros(len(bars), bool)
        r = np.random.default_rng(int(s))
        while solo.mean() < 0.3:
            i, L = r.integers(len(bars)), r.integers(2, 5)
            solo[i:i + L] = True
        acc = [pretty_midi.Note(n.velocity, n.pitch, n.start - t0, n.end - t0) for n in pia if t0 <= n.start < t1]
        keep = []
        for n in acc:
            b = next((j for j, (b0, b1, _) in enumerate(bars) if b0 <= n.start < b1), None)
            if b is not None and not solo[b]:
                keep.append(pretty_midi.Note(n.velocity, n.pitch, n.start, min(n.end, bars[b][1])))   # stop at bar end
        y = aah(voice, a.secs)
        acc_audio = pluck(keep, len(y), 1.0)
        mix = y + acc_audio / (np.abs(acc_audio).max() + 1e-9) * np.abs(y).max()
        if a.pad is not None:                       # soft pad in the solo bars: a harder, more realistic "solo"
            pad = np.zeros(len(y))
            for j, (b0, b1, ch) in enumerate(bars):
                c = next((c for c in ch if c), None)
                if solo[j] and c:
                    i0, i1 = int(b0 * SR), int(b1 * SR)
                    tt = np.arange(i1 - i0) / SR
                    for p in (48 + c[0], 55 + c[0]):
                        pad[i0:i1] += np.sin(2 * np.pi * pretty_midi.note_number_to_hz(p) * tt)
            mix = mix + 10 ** (a.pad / 20) * pad / (np.abs(pad).max() + 1e-9) * np.abs(y).max()
        mix = (0.5 * mix / np.abs(mix).max()).astype(np.float32)
        wav = out / f"{s}_mix.wav"
        sf.write(wav, mix, SR)

        got = ST.transcribe(wav, onset_thr=a.thr, ckpt=ck, frame_thr=0.5)
        if not got:
            continue
        arr = np.array([[g["onset"], g["offset"], g["pitch"]] for g in got], float)
        arr = arr[np.lexsort((-arr[:, 2], arr[:, 0]))]
        is_mel = melody_split(arr)
        accn = arr[~is_mel]
        c = np.array([((accn[:, 0] >= b0) & (accn[:, 0] < b1)).sum() for b0, b1, _ in bars])
        # polyphony: share of the bar's sounding time with 2+ notes at once (a lone voice is ~0)
        poly = []
        for b0, b1, _ in bars:
            tt = np.arange(b0, b1, 0.02)
            n_on = ((arr[:, 0][None] <= tt[:, None]) & (arr[:, 1][None] > tt[:, None])).sum(1)
            poly.append((n_on >= 2).sum() / max((n_on >= 1).sum(), 1))
        polys.append(np.array(poly))
        truth.append(solo)
        counts.append(c)
        print(f"{s}: bars {len(bars):2d} solo {solo.sum():2d}  acc notes in solo bars {c[solo].mean():.1f} / "
              f"in full bars {c[~solo].mean():.1f}", flush=True)
    truth, counts, polys = np.concatenate(truth), np.concatenate(counts), np.concatenate(polys)
    print(f"\n{len(truth)} bars, {truth.sum()} solo")
    for k in ks:
        pred = counts <= k
        tp = (pred & truth).sum()
        P, R = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        print(f"solo if <= {k} accompaniment notes: P {P:.3f}  R {R:.3f}  F1 {2 * P * R / max(P + R, 1e-9):.3f}  "
              f"bar accuracy {(pred == truth).mean():.3f}")

    for q in (0.05, 0.1, 0.2, 0.3):
        pred = polys <= q
        tp = (pred & truth).sum()
        P, R = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        print(f"solo if <= {q:.0%} of sounding time has 2+ notes: P {P:.3f}  R {R:.3f}  "
              f"F1 {2 * P * R / max(P + R, 1e-9):.3f}  bar accuracy {(pred == truth).mean():.3f}")


if __name__ == "__main__":
    main()
