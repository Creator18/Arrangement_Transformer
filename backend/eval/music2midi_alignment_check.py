"""Do the arrangement's notes actually line up with the downloaded audio?

Everything verified so far is circumstantial.  The note times end at 0.988 of
the YouTube duration, and the fetcher rejects a recording whose length
disagrees by more than 5 s.  Both say the right FILE was fetched.  Neither says
a note marked at 41.2 s happens at 41.2 s in it.

That distinction is the whole risk.  A misaligned target set trains to
completion and converges on nothing, and the loss curve looks normal the entire
time, because the model can still learn "this song is dense here" while learning
nothing about when a note starts.  It is the expensive failure and it is
invisible without a check like this one.

METHOD.  Take the audio's own onset strength envelope — a purely acoustic
signal, nothing to do with the labels — and the arrangement's note-onset
density at the same frame rate.  Cross-correlate.  If the pair is aligned, the
correlation peaks at lag zero.

THE CONTROL IS THE POINT.  A correlation peak near zero means nothing on its
own, because onset envelopes of pop songs are broadly similar: both have a beat.
So every song is also scored against a DIFFERENT song's arrangement.  If matched
pairs peak at zero and mismatched pairs scatter, the alignment is real.  If both
look alike, this test cannot see alignment and should not be believed.

    venv/Scripts/python.exe eval/music2midi_alignment_check.py --songs 60
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.ml.transcriber import config as C                    # noqa: E402
from app.ml.transcriber.arranger_data import AUDIO, DATA      # noqa: E402

FPS = 50.0
MAX_LAG_S = 8.0          # search +/- 8 s; a real misalignment is usually larger


def onset_env(sid: str):
    """Acoustic onset strength from the audio. Knows nothing about the labels."""
    import librosa
    y, sr = librosa.load(str(AUDIO / f"{sid}.wav"), sr=C.SR, mono=True)
    e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=int(C.SR / FPS))
    return (e - e.mean()) / (e.std() + 1e-9)


def note_density(sid: str, n: int):
    """Note-onset count per frame from the arrangement, smoothed to match."""
    a = np.load(DATA / "midi_numpy" / f"{sid}.npy")
    d = np.zeros(n)
    idx = np.clip((a[:, 0] * FPS).astype(int), 0, n - 1)
    np.add.at(d, idx, 1.0)
    # Smoothed over ~60 ms so a chord spread across two frames reads as one
    # event, the same way the acoustic envelope sees it.
    k = np.hanning(7); k /= k.sum()
    d = np.convolve(d, k, mode="same")
    return (d - d.mean()) / (d.std() + 1e-9)


def best_lag(env: np.ndarray, dens: np.ndarray):
    """(lag in seconds at peak correlation, peak correlation)."""
    n = min(len(env), len(dens))
    env, dens = env[:n], dens[:n]
    m = int(MAX_LAG_S * FPS)
    lags = np.arange(-m, m + 1, 2)
    best, bl = -2.0, 0
    for L in lags:
        if L >= 0:
            a, b = env[L:], dens[:n - L] if L else dens
        else:
            a, b = env[:n + L], dens[-L:]
        if len(a) < FPS * 10:
            continue
        c = float(np.dot(a, b) / len(a))
        if c > best:
            best, bl = c, L
    return bl / FPS, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=60)
    ap.add_argument("--write-manifest", action="store_true",
                    help="score every fetched song and write alignment.json, "
                         "which the loader then uses to exclude bad pairs")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="max |lag| in seconds a song may have and still train")
    args = ap.parse_args()

    ids = sorted(p.stem for p in AUDIO.glob("*.wav"))
    if args.write_manifest:
        import json
        keep, drop = {}, {}
        for i, sid in enumerate(ids):
            try:
                env = onset_env(sid)
                lag, c = best_lag(env, note_density(sid, len(env)))
            except Exception:                               # noqa: BLE001
                drop[sid] = None
                continue
            (keep if abs(lag) <= args.tol else drop)[sid] = round(lag, 3)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(ids)}  keep {len(keep)} drop {len(drop)}",
                      flush=True)
        out = AUDIO.parent / "alignment.json"
        out.write_text(json.dumps({"tol": args.tol, "keep": keep, "drop": drop},
                                  indent=1), encoding="utf-8")
        print(f"\nkept {len(keep)}/{len(ids)} "
              f"({len(keep) / len(ids) * 100:.0f}%), "
              f"dropped {len(drop)}  ->  {out}")
        return
    ids = ids[: args.songs]
    if len(ids) < 4:
        raise SystemExit("not enough audio")

    match, mism = [], []
    for i, sid in enumerate(ids):
        try:
            env = onset_env(sid)
            lag, c = best_lag(env, note_density(sid, len(env)))
            match.append((lag, c))
            # Control: this song's AUDIO against a different song's ARRANGEMENT.
            other = ids[(i + len(ids) // 2) % len(ids)]
            lag2, c2 = best_lag(env, note_density(other, len(env)))
            mism.append((lag2, c2))
        except Exception as e:                              # noqa: BLE001
            print(f"  skip {sid}: {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(ids)}", flush=True)

    m = np.array(match); x = np.array(mism)
    print(f"\nalignment check — {len(m)} songs\n")
    print(f"{'':<22}{'matched':>10}{'mismatched':>13}")
    print("-" * 45)
    print(f"{'|lag| median (s)':<22}{np.median(np.abs(m[:, 0])):>10.2f}"
          f"{np.median(np.abs(x[:, 0])):>13.2f}")
    print(f"{'lag within 0.25 s':<22}{np.mean(np.abs(m[:, 0]) <= 0.25) * 100:>9.0f}%"
          f"{np.mean(np.abs(x[:, 0]) <= 0.25) * 100:>12.0f}%")
    print(f"{'lag within 1.0 s':<22}{np.mean(np.abs(m[:, 0]) <= 1.0) * 100:>9.0f}%"
          f"{np.mean(np.abs(x[:, 0]) <= 1.0) * 100:>12.0f}%")
    print(f"{'peak correlation':<22}{m[:, 1].mean():>10.3f}{x[:, 1].mean():>13.3f}")

    ok = float(np.mean(np.abs(m[:, 0]) <= 0.25))
    ctl = float(np.mean(np.abs(x[:, 0]) <= 0.25))
    print()
    if ok > 0.7 and ok > ctl * 2:
        print(f"=> ALIGNED.  {ok * 100:.0f}% of matched pairs peak inside 0.25 s "
              f"against {ctl * 100:.0f}% of mismatched, so the test can tell them "
              f"apart and the targets line up with the audio.")
    elif ok <= ctl * 1.5:
        print("=> INCONCLUSIVE.  Matched pairs look like mismatched ones, so this "
              "test cannot see alignment here — do NOT read it as a pass.")
    else:
        print(f"=> SUSPECT.  Only {ok * 100:.0f}% of matched pairs peak near zero. "
              f"Training on this would converge on nothing while looking normal.")

    bad = [(ids[i], m[i, 0]) for i in range(len(m)) if abs(m[i, 0]) > 1.0]
    if bad:
        print(f"\n{len(bad)} songs off by more than 1 s, e.g. "
              + ", ".join(f"{s}({l:+.1f}s)" for s, l in bad[:6]))


if __name__ == "__main__":
    main()
