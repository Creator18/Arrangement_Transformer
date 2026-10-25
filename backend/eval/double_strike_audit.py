"""Where do the app's double strikes come from?  For songs already processed by the app (temp/<session>),
find every re-press of a key that was still held (same pitch, next strike within 30 ms of the previous
note's end), and look at the song audio at that moment:

  drum hit     a percussive transient right there (HPSS percussive onset strength, top 10% of the song)
  on the beat  within 40 ms of a tracked beat
  new attack   the pitch's own energy jumps (CQT bin, 60 ms after vs 60 ms before > 1.5x): a real
               re-articulation in the song (new syllable, strum, re-bowed note)
  register     melody range (>= C4) or below

and compare with every other note start as a baseline.

    python eval/double_strike_audit.py temp/<session> [temp/<session> ...]
"""
import json, pathlib, sys

import librosa
import numpy as np

SR, HOP = 22050, 256
for sess in sys.argv[1:]:
    d = pathlib.Path(sess)
    notes = json.loads((d / "transformed.json").read_text())["transformed_notes"]
    comp = json.loads((d / "components.json").read_text())
    beats = np.array(comp.get("beat_frames", []), float) * comp.get("hop_length", 512) / comp.get("sr", 22050)
    y, _ = librosa.load(str(d / "audio.wav"), sr=SR, mono=True)
    _, yp = librosa.effects.hpss(y)
    perc = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=HOP)
    p_thr = np.percentile(perc, 90)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21), n_bins=88, bins_per_octave=12))
    fr = lambda t: int(np.clip(round(t * SR / HOP), 0, C.shape[1] - 1))
    w = int(round(0.06 * SR / HOP))

    def feats(t, p):
        f = fr(t); k = int(p) - 21
        drum = perc[max(0, f - 2):f + 3].max() >= p_thr
        beat = len(beats) and np.min(np.abs(beats - t)) <= 0.04
        before = C[k, max(0, f - w):f].mean() + 1e-9; after = C[k, f:f + w].mean()
        return drum, bool(beat), after / before > 1.5, p >= 60

    bykey = {}
    for n in notes:
        bykey.setdefault(int(n["pitch"]), []).append(n)
    dbl, other = [], []
    for p, ns in bykey.items():
        ns.sort(key=lambda n: n["onset"])
        for i, n in enumerate(ns):
            prev = ns[i - 1] if i else None
            is_dbl = prev is not None and n["onset"] - prev["offset"] <= 0.03
            (dbl if is_dbl else other).append((n["onset"], p, (n["onset"] - prev["onset"]) if prev is not None else None))
    def share(rows):
        F = np.array([feats(t, p) for t, p, _ in rows], float) if rows else np.zeros((0, 4))
        return F.mean(0) if len(F) else np.zeros(4)
    sd, so = share(dbl), share(other)
    gaps = np.array([g for _, _, g in dbl if g is not None])
    print(f"\n{d.name}: {len(notes)} notes, {len(dbl)} double strikes ({1000 * len(dbl) / max(1, len(notes)):.0f} per 1000)")
    print(f"   time since the previous strike of that key: median {1000 * np.median(gaps):.0f} ms, "
          f"<150 ms {np.mean(gaps < .15):.0%}, 150-400 ms {np.mean((gaps >= .15) & (gaps < .4)):.0%}, >=400 ms {np.mean(gaps >= .4):.0%}")
    print(f"   {'':22s}{'double strikes':>16s}{'other note starts':>20s}")
    for nm, a_, b_ in zip(("drum hit there", "on a beat (+-40 ms)", "pitch re-attacks", "melody range (>=C4)"), sd, so):
        print(f"   {nm:22s}{a_:15.0%}{b_:19.0%}")
