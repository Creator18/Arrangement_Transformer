"""Do the curriculum's note labels start where the sound starts?

The confidence trace showed the arranger's onset confidence collapsing at the
first curriculum stage.  A model taught onsets that land a few frames off the
real attacks learns to hedge — the piano renders showed it at a 2-frame offset.
Measured here on ISOLATED stems, so each label is compared with its own sound:

  Slakh   every pitched stem's MIDI onsets vs that stem's audio, per
          instrument class (each class is a different sample library, each
          with its own latency)
  MUSDB   the pYIN vocal and bass notes vs the vocal and bass stems

For each stem: the lag that best aligns the label onset train with the audio's
onset envelope (cross-correlation, +-150 ms, 5.8 ms resolution) — the
systematic offset; and, per labelled note, the distance to the nearest audio
onset peak — the jitter.  Reference: the 62.5 fps label frame is 16 ms, so a
systematic lag beyond ~8 ms or jitter of ~16 ms+ is a frame or more off.

    python eval/label_alignment_check.py --slakh <root> --musdb <root> --labels <curriculum/vocal>
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random

import numpy as np

SR, HOP = 22050, 128
DT = HOP / SR                                   # 5.8 ms


def envelope(y):
    import librosa
    e = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, lag=1, max_size=1)
    return (e - e.mean()) / (e.std() + 1e-9)


def measure(env, onsets):
    """(best lag in s: positive = sound comes AFTER the label, median |jitter| s, n)."""
    T = len(env)
    lab = np.zeros(T)
    idx = (np.asarray(onsets) / DT).astype(int)
    idx = idx[(idx >= 0) & (idx < T)]
    if len(idx) < 8:
        return None
    lab[idx] = 1
    lab = np.convolve(lab, np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2), "same")
    L = int(0.15 / DT)
    cc = [np.dot(lab[max(0, -k):T - max(0, k)], env[max(0, k):T - max(0, -k)]) for k in range(-L, L + 1)]
    lag = (np.argmax(cc) - L) * DT
    from scipy.signal import find_peaks
    pk, _ = find_peaks(env, height=1.0, distance=int(0.03 / DT))
    if len(pk) == 0:
        return None
    t_pk = pk * DT
    d = np.array([t_pk[np.argmin(np.abs(t_pk - (o + lag)))] - (o + lag) for o in np.asarray(onsets)])
    return lag, float(np.median(np.abs(d[np.abs(d) < 0.15]))) if (np.abs(d) < 0.15).any() else np.nan, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh", required=True)
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tracks", type=int, default=60)
    ap.add_argument("--maestro-clips", default="", help="real MAESTRO clips: the known-good reference")
    args = ap.parse_args()
    import librosa, yaml, pretty_midi
    rng = random.Random(0)

    ref_lag = 0.0
    if args.maestro_clips:
        # Disklavier MIDI is accurate to ~3 ms, so whatever lag this reports is
        # the MEASUREMENT's own bias (onset-envelope peak vs attack start) plus
        # a real piano's few ms of attack — the baseline the others are read against.
        rs = []
        for f in sorted((pathlib.Path(args.maestro_clips) / "validation").glob("*.npz"))[::8][:60]:
            z = np.load(f)
            y = librosa.resample(z["y"].astype(np.float32) / 32767, orig_sr=24000, target_sr=SR)
            k, fr = np.nonzero(z["on"])
            r = measure(envelope(y), fr / 62.5 - 0.0086)          # undo the +8.6 ms label shift
            if r:
                rs.append(r)
        lag = np.array([r[0] for r in rs]) * 1000; jit = np.array([r[1] for r in rs]) * 1000
        ref_lag = float(np.median(lag))
        print(f"REFERENCE real MAESTRO (Disklavier MIDI): {len(rs)} clips, lag median {ref_lag:.1f} ms "
              f"(IQR {np.percentile(lag, 25):.1f}..{np.percentile(lag, 75):.1f}), jitter median {np.nanmedian(jit):.1f} ms" + chr(10) +
              f"   -> lags below are also shown RELATIVE to this baseline" + chr(10))

    print("SLAKH: MIDI onsets vs each stem's own audio (30 s per stem)")
    by = collections.defaultdict(list)
    tracks = sorted((pathlib.Path(args.slakh) / "train").glob("Track*"))
    for tr in rng.sample(tracks, min(args.tracks, len(tracks))):
        try:
            meta = yaml.safe_load(open(tr / "metadata.yaml"))
        except Exception:                                   # noqa: BLE001
            continue
        for k, v in meta["stems"].items():
            if v.get("is_drum") or not v.get("audio_rendered"):
                continue
            wav, mid = tr / "stems" / f"{k}.flac", tr / "MIDI" / f"{k}.mid"
            if not (wav.exists() and mid.exists()):
                continue
            notes = [n.start for i in pretty_midi.PrettyMIDI(str(mid)).instruments for n in i.notes]
            if len(notes) < 20:
                continue
            t0 = float(np.percentile(notes, 30))
            y, _ = librosa.load(str(wav), sr=SR, mono=True, offset=t0, duration=30)
            r = measure(envelope(y), [n - t0 for n in notes if t0 <= n < t0 + 30])
            if r:
                by[v.get("inst_class", "?")].append(r)
    print(f"   {'class':<22}{'stems':>6}{'lag ms (median)':>17}{'IQR':>14}{'jitter ms':>11}{'vs ref':>9}")
    allr = []
    for c, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        lag = np.array([r[0] for r in rs]) * 1000; jit = np.array([r[1] for r in rs]) * 1000
        allr += rs
        print(f"   {c:<22}{len(rs):>6}{np.median(lag):>17.1f}{np.percentile(lag, 25):>7.1f}..{np.percentile(lag, 75):<5.1f}"
              f"{np.nanmedian(jit):>11.1f}{np.median(lag) - ref_lag:>+9.1f}")
    lag = np.array([r[0] for r in allr]) * 1000
    print(f"   {'ALL':<22}{len(allr):>6}{np.median(lag):>17.1f}   vs ref {np.median(lag) - ref_lag:+.1f}   "
          f"share off the reference by > 8 ms: {np.mean(np.abs(lag - ref_lag) > 8):.0%}")

    print("\nMUSDB: pYIN labels vs the isolated stems")
    for kind, stem in (("voc", "vocals"), ("bas", "bass")):
        rs = []
        for lab in sorted(pathlib.Path(args.labels).glob("labels_*.npz")):
            name = lab.stem[len("labels_"):]
            tr = next((pathlib.Path(args.musdb) / s / name for s in ("train", "test")
                       if (pathlib.Path(args.musdb) / s / name).exists()), None)
            if tr is None:
                continue
            on = np.load(lab)[kind][:, 0]
            if len(on) < 20:
                continue
            y, _ = librosa.load(str(tr / f"{stem}.wav"), sr=SR, mono=True, duration=120)
            r = measure(envelope(y), on[on < 120])
            if r:
                rs.append(r)
        lag = np.array([r[0] for r in rs]) * 1000; jit = np.array([r[1] for r in rs]) * 1000
        print(f"   {stem:<8} {len(rs):>4} songs   lag median {np.median(lag):6.1f} ms (IQR {np.percentile(lag, 25):.1f}.."
              f"{np.percentile(lag, 75):.1f})   vs ref {np.median(lag) - ref_lag:+.1f}   jitter median {np.nanmedian(jit):5.1f} ms   "
              f"share off ref by > 8 ms {np.mean(np.abs(lag - ref_lag) > 8):.0%}")
    print("\n(lag > 0: the sound starts AFTER the label; one label frame = 16 ms)")


if __name__ == "__main__":
    main()
