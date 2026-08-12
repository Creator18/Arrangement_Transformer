"""
What rhythm does a human piano accompaniment actually play?

The generated accompaniment still clutters the bass, and the suspect is the
rhythm vocabulary.  canonical_rhythms clusters the bar's OWN detected onsets, so
the figures it writes are as irregular as the detections: one shipped pattern is
(0,1,2,4,6,9,14) — seven attacks at positions no pianist would choose.  Every
one of those sits on a sixteenth, so it is quantised and still does not feel
quantised, because "on the grid" and "a musical rhythm" are different claims.

POP909's PIANO track is 909 human piano accompaniments.  This measures the
rhythms they actually use, so the vocabulary can be RETRIEVED from human
playing instead of clustered out of our own noise.

Patterns are recorded as a 16-bit mask over one bar: which sixteenth slots carry
an attack.  That representation is strict quantisation by construction — there
is nowhere off-grid for a retrieved pattern to land.
"""
from __future__ import annotations

import argparse, collections, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
OUT = ROOT / "backend/app/services/_accomp_patterns.npz"
SLOTS = 16


def bars_of(folder):
    f = folder / "beat_midi.txt"
    if not f.exists():
        return None
    rows = np.loadtxt(f)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return None
    downs = rows[rows[:, 2] > 0.5, 0]
    return downs if len(downs) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=909)
    ap.add_argument("--top", type=int, default=48, help="patterns to keep")
    args = ap.parse_args()

    import pretty_midi

    masks = collections.Counter()
    attacks_per_bar, nps, simult = [], [], []
    used = 0

    for folder in sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]:
        mid = folder / f"{folder.name}.mid"
        edges = bars_of(folder)
        if not mid.exists() or edges is None:
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:                                   # noqa: BLE001
            continue
        pno = next((i for i in pm.instruments
                    if (i.name or "").strip().upper() == "PIANO"), None)
        if pno is None or len(pno.notes) < 16:
            continue
        used += 1
        notes = sorted(pno.notes, key=lambda n: n.start)
        span = pm.get_end_time() or 1.0
        nps.append(len(notes) / span)

        # how many notes share an attack (a chord counts once as a rhythm event)
        by_onset = collections.Counter(round(n.start, 2) for n in notes)
        simult.append(float(np.mean(list(by_onset.values()))))

        for b0, b1 in zip(edges[:-1], edges[1:]):
            if b1 <= b0:
                continue
            slots = set()
            for n in notes:
                if b0 <= n.start < b1:
                    slots.add(int(np.clip((n.start - b0) / (b1 - b0) * SLOTS,
                                          0, SLOTS - 1)))
            if not slots:
                continue
            attacks_per_bar.append(len(slots))
            masks[sum(1 << s for s in slots)] += 1

    top = masks.most_common(args.top)
    keep = np.array([m for m, _ in top], dtype=np.int32)
    counts = np.array([c for _, c in top], dtype=np.float64)
    weights = counts / counts.sum()
    np.savez(OUT, masks=keep, weights=weights, slots=SLOTS, songs=used,
             bars=int(sum(masks.values())))

    print(f"POP909: {used} songs, {sum(masks.values())} bars -> {OUT.name}\n")
    a = np.array(attacks_per_bar)
    print(f"notes/second in PIANO track : mean {np.mean(nps):.2f}  median {np.median(nps):.2f}")
    print(f"notes per attack (chord size): mean {np.mean(simult):.2f}")
    print(f"attack SLOTS per bar         : median {np.median(a):.0f}  "
          f"p25 {np.percentile(a,25):.0f}  p75 {np.percentile(a,75):.0f}  "
          f"p95 {np.percentile(a,95):.0f}")
    print(f"distinct bar patterns        : {len(masks)}  "
          f"(top {args.top} cover {100*counts.sum()/sum(masks.values()):.1f}% of bars)\n")
    print("most common patterns  (# = attack, . = rest, bar = 16 sixteenths)")
    for m, c in top[:16]:
        s = "".join("#" if (m >> i) & 1 else "." for i in range(SLOTS))
        print(f"  {100*c/sum(masks.values()):5.2f}%  {s}  ({bin(m).count('1')} attacks)")


if __name__ == "__main__":
    main()
