"""Does the difficulty classifier still work on TRANSCRIBED covers?

It was trained on engraved MuseScore scores.  YouTube covers will reach it as
our transcriptions of a performance: pedal-stretched lengths, timing jitter,
a few missed or extra notes.  Here: 60 s of held-out scores (balanced across
levels) are rendered through the sampled pianos (with room), transcribed by
the app's spectral model, and classified both ways.  Agreement between the
score's prediction and the transcription's is what matters; accuracy against
the true label is reported too.

    python eval/difficulty_on_transcriptions.py --n 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from difficulty_classifier import features, LEVELS     # noqa: E402

SECONDS = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="songs per level")
    args = ap.parse_args()
    import joblib, yaml, soundfile as sf
    from app.ml.spectral import render as R, frontend as S
    from app.services import spectral_transcription as ST
    from app.ml.transcriber.arranger_data import DATA

    clf = joblib.load(ROOT / "app" / "ml" / "difficulty.joblib")["model"]
    sp = json.loads((DATA.parent / "m2m_splits.json").read_text())
    by = {l: [] for l in LEVELS}
    for s in sp["test"] + sp["validation"]:
        by[sp["difficulty"][s]].append(s)
    rng = random.Random(0)
    pool = [s for l in LEVELS for s in rng.sample(by[l], min(args.n, len(by[l])))]
    names = [p for p in R.pianos() if p != "intimate"]
    lat = R.latencies(); tmp = pathlib.Path(tempfile.mkdtemp())
    rows = []
    for k, sid in enumerate(pool):
        a = np.load(DATA / "midi_numpy" / f"{sid}.npy")
        t0 = float(a[:, 0].min()); a = a[(a[:, 0] >= t0) & (a[:, 0] < t0 + SECONDS)].copy()
        a[:, :2] -= t0 - 0.5
        notes = [{"pitch": int(p), "onset": float(s), "offset": float(max(e, s + 0.05)),
                  "velocity": int(np.clip(v, 1, 127))} for s, e, p, v in a if 21 <= p <= 108]
        piano = names[k % len(names)]
        wav = tmp / f"{sid}.wav"
        sf.write(wav, R.render_with(piano, notes), S.SR)
        est = ST.transcribe(wav)
        e = np.array([[n["onset"] - lat.get(piano, 0), n["offset"], n["pitch"], n["velocity"]] for n in est])
        f_s, f_t = features(a), features(e)
        p_s, p_t = clf.predict([f_s])[0], clf.predict([f_t])[0]
        rows.append((LEVELS.index(sp["difficulty"][sid]), p_s, p_t, f_s[0], f_t[0]))
        print(f"  {sid:<10} {sp['difficulty'][sid]:<13} score->{LEVELS[p_s]:<13} transcription->{LEVELS[p_t]:<13}"
              f" notes/s {f_s[0]:5.1f} -> {f_t[0]:5.1f}", flush=True)
    r = np.array(rows)
    print(f"\n{len(r)} covers (60 s each), rendered through {len(names)} pianos and transcribed")
    print(f"   score vs transcription agree:   {np.mean(r[:, 1] == r[:, 2]):.3f}")
    print(f"   accuracy on scores:             {np.mean(r[:, 0] == r[:, 1]):.3f}")
    print(f"   accuracy on transcriptions:     {np.mean(r[:, 0] == r[:, 2]):.3f}")
    print(f"   'intermediate or harder' agree: {np.mean((r[:, 1] >= 1) == (r[:, 2] >= 1)):.3f}")
    print(f"   notes/s transcription/score:    median {np.median(r[:, 4] / r[:, 3]):.3f}")


if __name__ == "__main__":
    main()
