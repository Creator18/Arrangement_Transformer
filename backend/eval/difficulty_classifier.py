"""What makes a piano arrangement beginner, intermediate or advanced?

Music2MIDI ships 8,255 MuseScore piano covers labelled by difficulty.  Learning
that boundary from note-level features gives (a) thresholds we can state in
plain terms and (b) a classifier to label piano covers we transcribe later,
which arrive without a difficulty.

Features use ONSETS AND PITCHES ONLY — no note lengths, no exact rhythm values
— because a transcribed performance has pedal-stretched durations and
expressive timing that an engraved score does not, and a classifier keyed on
those would misjudge every transcription.  Hands are approximated by a split
at middle C (the files carry no hand information).

  density     notes/s overall and in the busiest 2 s windows (p90)
  onsets      onset events/s (a chord = one event), p90; p10 inter-onset gap
  polyphony   notes per onset event, mean and p90
  movement    per-hand jump between consecutive events (semitones), mean, p90,
              and jump speed (semitones/s)
  stretch     per-hand span within one event, p90
  range       pitch p95 - p5

5-fold cross-validation GROUPED BY SONG (several covers of one song never
straddle train and test).

    python eval/difficulty_classifier.py [--out app/ml/difficulty.joblib]
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEVELS = ["beginner", "intermediate", "advanced"]
NAMES = ["notes_per_s", "notes_per_s_p90", "onsets_per_s", "onsets_per_s_p90", "gap_p10_ms",
         "poly_mean", "poly_p90", "rh_jump_mean", "rh_jump_p90", "lh_jump_mean", "lh_jump_p90",
         "jump_speed_p90", "rh_span_p90", "lh_span_p90", "range"]


def features(notes: np.ndarray) -> np.ndarray:
    """notes: [onset, offset, pitch, velocity] rows -> feature vector (onset/pitch only)."""
    on = notes[:, 0].astype(float); p = notes[:, 2].astype(float)
    o = np.argsort(on); on, p = on[o], p[o]
    dur = max(1.0, on[-1] - on[0])
    # onset events: notes within 30 ms are one gesture
    ev = np.concatenate([[0], np.cumsum(np.diff(on) > 0.03)])
    n_ev = ev[-1] + 1
    t_ev = np.array([on[ev == e].min() for e in range(n_ev)])
    w = np.arange(on[0], on[-1] + 2, 2.0)
    per2 = np.histogram(on, w)[0] / 2.0
    ev2 = np.histogram(t_ev, w)[0] / 2.0
    poly = np.bincount(ev)
    gaps = np.diff(t_ev) if n_ev > 1 else np.array([1.0])
    feats = {"rh": ([], []), "lh": ([], [])}
    prev = {"rh": None, "lh": None}; spans = {"rh": [], "lh": []}
    for e in range(n_ev):
        ps = p[ev == e]
        for h, sel in (("rh", ps[ps >= 60]), ("lh", ps[ps < 60])):
            if len(sel) == 0:
                continue
            m = sel.mean(); spans[h].append(sel.max() - sel.min())
            if prev[h] is not None:
                feats[h][0].append(abs(m - prev[h][0]))
                feats[h][1].append(abs(m - prev[h][0]) / max(0.03, t_ev[e] - prev[h][1]))
            prev[h] = (m, t_ev[e])
    q = lambda a, k: float(np.percentile(a, k)) if len(a) else 0.0
    mu = lambda a: float(np.mean(a)) if len(a) else 0.0
    return np.array([
        len(on) / dur, q(per2, 90), n_ev / dur, q(ev2, 90), q(gaps, 10) * 1000,
        mu(poly), q(poly, 90),
        mu(feats["rh"][0]), q(feats["rh"][0], 90), mu(feats["lh"][0]), q(feats["lh"][0], 90),
        q(feats["rh"][1] + feats["lh"][1], 90),
        q(spans["rh"], 90), q(spans["lh"], 90), q(p, 95) - q(p, 5)], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import yaml
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, accuracy_score
    from app.ml.transcriber.arranger_data import DATA

    X, y, g = [], [], []
    for f in sorted((DATA / "metadata").glob("*.yaml")):
        npy = DATA / "midi_numpy" / f"{f.stem}.npy"
        if not npy.exists():
            continue
        m = yaml.safe_load(open(f, encoding="utf-8"))
        a = np.load(npy)
        if len(a) < 50:
            continue
        X.append(features(a)); y.append(LEVELS.index(m["piano"]["difficulty"]))
        g.append(f"{m['song']['artist']}|{m['song']['title']}".lower())
    X, y = np.array(X), np.array(y)
    print(f"{len(y)} labelled arrangements, {len(set(g))} songs; "
          f"{dict(zip(LEVELS, np.bincount(y)))}\n")

    print("medians by level:")
    print(f"   {'feature':<18}" + "".join(f"{l:>14}" for l in LEVELS))
    for i, n in enumerate(NAMES):
        print(f"   {n:<18}" + "".join(f"{np.median(X[y == k, i]):>14.2f}" for k in range(3)))

    cv = GroupKFold(5)
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "boosted trees": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)}
    for name, mdl in models.items():
        pr = cross_val_predict(mdl, X, y, cv=cv, groups=g)
        cm = confusion_matrix(y, pr)
        adj = np.mean(np.abs(pr - y) <= 1)
        print(f"\n{name}: accuracy {accuracy_score(y, pr):.3f}  (chance {np.bincount(y).max() / len(y):.3f}),"
              f"  within one level {adj:.3f}")
        print("   rows true, cols predicted  " + "  ".join(f"{l[:5]:>6}" for l in LEVELS))
        for k in range(3):
            print(f"   {LEVELS[k]:<26}" + "  ".join(f"{v:>6}" for v in cm[k]))
        # the app's question: is it at least intermediate?
        print(f"   'intermediate or harder' correct: {np.mean((pr >= 1) == (y >= 1)):.3f}")

    # one-feature thresholds, for plain-language rules
    print("\nbest single-feature cut points (balanced accuracy):")
    for i, n in enumerate(NAMES):
        for lo, hi in ((0, 1), (1, 2)):
            sel = (y == lo) | (y == hi); xs, ys = X[sel, i], y[sel] == hi
            cands = np.percentile(xs, np.arange(5, 96, 1))
            ba = [(0.5 * (np.mean(xs[ys] > c) + np.mean(xs[~ys] <= c)), c) for c in cands]
            b, c = max(ba)
            if b >= 0.70:
                print(f"   {LEVELS[lo]:>12} vs {LEVELS[hi]:<12} {n:<18} > {c:7.2f}   bal.acc {b:.3f}")

    best = models["boosted trees"].fit(X, y)
    if args.out:
        import joblib
        joblib.dump({"model": best, "features": NAMES, "levels": LEVELS}, args.out)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
