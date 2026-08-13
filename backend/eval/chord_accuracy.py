"""
Is infer_chord wrong, or is it being fed the wrong notes?

Minor chords come out in 17.6% of bars against a human 33.4%, and that has two
possible causes which need different fixes: the scoring is broken, or the notes
it reads do not contain the third that distinguishes major from minor.  Until
now the only evidence was a distribution comparison on real uploads, which
cannot separate the two — and matching a distribution is not the same as being
right.

POP909 settles it, because it ships HUMAN CHORD LABELS (chord_midi.txt) next to
the notes they describe.  So chords can be inferred from the human's own notes
and scored against the human's own labels.  No audio, no projection, no
separation: this isolates the algorithm.

If accuracy is high here, infer_chord works and the problem is the signal — the
guitar stem should fix it.  If accuracy is low here, the scoring is wrong and no
amount of better separation will help.

Also answers a design question directly, with ground truth rather than
inference: WHICH notes should chord inference read?  The same bars are scored
with four different note sources, including the low-register-only view that the
pipeline used until recently.
"""
from __future__ import annotations

import argparse, collections, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_root(tok: str):
    if not tok or tok[0] not in _PC:
        return None
    pc = _PC[tok[0]]
    for ch in tok[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            break
    return pc % 12


def map_quality(q: str):
    q = q.split("/")[0]
    if q.startswith("min7"):
        return "min7"
    if q.startswith("hdim") or q.startswith("dim"):
        return "dim"
    if q.startswith("min"):
        return "min"
    if q.startswith("maj") or q.startswith("sus") or q in ("6", "9", ""):
        return "maj"
    if q.startswith("7") or q.startswith("9"):
        return "dom7"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    import pretty_midi
    from app.services import arpeggiator as AR

    folders = sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]

    SOURCES = ["all notes", "PIANO track only", "below G#3 only", "above G#3 only"]
    stats = {s: collections.Counter() for s in SOURCES}
    conf = {s: collections.Counter() for s in SOURCES}
    used = 0

    for folder in folders:
        mid = folder / f"{folder.name}.mid"
        cf = folder / "chord_midi.txt"
        bf = folder / "beat_midi.txt"
        if not (mid.exists() and cf.exists() and bf.exists()):
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
            rows = np.loadtxt(bf)
        except Exception:                                       # noqa: BLE001
            continue
        if rows.ndim != 2 or rows.shape[1] < 3:
            continue
        downs = rows[rows[:, 2] > 0.5, 0]
        if len(downs) < 3:
            continue

        # ground-truth chord spans
        spans = []
        for line in cf.read_text().splitlines():
            p = line.split()
            if len(p) < 3 or p[2] == "N" or ":" not in p[2]:
                continue
            r = parse_root(p[2].split(":")[0])
            q = map_quality(p[2].split(":", 1)[1])
            if r is None or q is None:
                continue
            try:
                spans.append((float(p[0]), float(p[1]), r, q))
            except ValueError:
                continue
        if not spans:
            continue
        used += 1

        by_track = {}
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            by_track.setdefault((inst.name or "").strip().upper(), []).extend(inst.notes)
        allnotes = [n for v in by_track.values() for n in v]
        piano = by_track.get("PIANO", [])

        def as_dicts(ns):
            return [{"pitch": int(n.pitch), "onset": float(n.start),
                     "offset": float(n.end), "velocity": int(n.velocity)}
                    for n in ns]

        for b0, b1 in zip(downs[:-1], downs[1:]):
            if b1 <= b0:
                continue
            # dominant ground-truth label in this bar, by overlap duration
            best, bestdur = None, 0.0
            for s0, s1, r, q in spans:
                ov = min(b1, s1) - max(b0, s0)
                if ov > bestdur:
                    best, bestdur = (r, q), ov
            if best is None or bestdur < 0.3 * (b1 - b0):
                continue
            key_root = best[0]        # give inference the tonic-free advantage off
            views = {
                "all notes":        [n for n in allnotes if b0 <= n.start < b1],
                "PIANO track only": [n for n in piano if b0 <= n.start < b1],
                "below G#3 only":   [n for n in allnotes if b0 <= n.start < b1 and n.pitch < 56],
                "above G#3 only":   [n for n in allnotes if b0 <= n.start < b1 and n.pitch >= 56],
            }
            for src, ns in views.items():
                if len(ns) < 2:
                    continue
                got = AR.infer_chord(as_dicts(ns), None)
                if got is None:
                    continue
                c = stats[src]
                c["n"] += 1
                c["root"] += int(got[0] == best[0])
                c["qual"] += int(got[1] == best[1])
                c["both"] += int(got[0] == best[0] and got[1] == best[1])
                if best[1] in ("min", "min7"):
                    c["true_min"] += 1
                    c["min_ok"] += int(got[1] in ("min", "min7"))
                if best[1] == "maj":
                    c["true_maj"] += 1
                    c["maj_ok"] += int(got[1] == "maj")
                conf[src][(best[1], got[1])] += 1

    print(f"POP909: {used} songs with labels + downbeats, "
          f"prior weight = {AR.CHORD_PRIOR_W}\n")
    print(f"{'note source':<20}{'bars':>7}{'root':>8}{'quality':>9}{'both':>8}"
          f"{'minor recall':>14}{'major recall':>14}")
    print("-" * 80)
    for src in SOURCES:
        c = stats[src]
        if not c["n"]:
            continue
        f = lambda k, d="n": 100.0 * c[k] / max(c[d], 1)
        print(f"{src:<20}{c['n']:>7}{f('root'):>7.1f}%{f('qual'):>8.1f}%"
              f"{f('both'):>7.1f}%{100.0*c['min_ok']/max(c['true_min'],1):>13.1f}%"
              f"{100.0*c['maj_ok']/max(c['true_maj'],1):>13.1f}%")

    print("\nwhere true MINOR bars actually go (all notes):")
    tot = sum(v for (t, _g), v in conf["all notes"].items() if t == "min")
    for (t, g), v in sorted(conf["all notes"].items(), key=lambda x: -x[1]):
        if t == "min" and v:
            print(f"  min -> {g:<6} {100.0*v/max(tot,1):5.1f}%")


if __name__ == "__main__":
    main()
