"""
Arrangement eval — the 88 -> 61 half of the pipeline.

The README has long said this half "is a separate evaluation problem with
no formal metric".  This is that metric.  It exists so a change to range
compression can be argued from numbers instead of from theory, which is
the failure mode that has cost this project the most: three of the four
biggest wins so far came from DELETING rules that were written from theory
and never measured.

Arms are pluggable.  `passthrough` is the control — it is literally what
the 7-octave preset does, since compress_notes_to_range returns untouched
notes for (21, 108).  Comparing `fold` against it isolates exactly what
range compression costs, with nothing else in the way.

Usage
-----
    python backend/eval/eval_arrangement.py                 # 5 oct, all arms
    python backend/eval/eval_arrangement.py --octaves 6
    python backend/eval/eval_arrangement.py --arms fold --limit 30
"""
from __future__ import annotations
import argparse, glob, json, os, statistics as st, sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.nmf_voice_leading import compress_notes_to_range   # noqa: E402
from eval.arrangement_metrics import evaluate                        # noqa: E402

OCTAVE_BOUNDS = {5: (36, 96), 6: (28, 103), 7: (21, 108)}

CORPUS = os.path.join(os.path.dirname(__file__), "..",
                      "correction", "data", "work", "*", "transcription.json")


# ── arms ──────────────────────────────────────────────────────────────────────

def arm_passthrough(notes, lo, hi):
    """No compression.  The control, and what 7-octave actually does."""
    return notes


def arm_fold(notes, lo, hi):
    """Currently shipped: per-note octave folding."""
    return compress_notes_to_range(notes, lo, hi)


def arm_fold_nodouble(notes, lo, hi):
    """
    Candidate fix: fold, but never press a key that is already down.

    The measurement says doubled keys are the dominant cost of folding
    (0.2 -> 12.6 per piece, worst 96), and they are the exact mechanism
    behind repeated notes losing their sustain.  So when the folded
    destination is already sounding:

      1. try the next octave out, nearest-to-original first, so the note
         keeps as much of its register as possible;
      2. if every octave in range is occupied, DROP the note and extend
         the note already holding that key to cover it.

    Dropping is the musically honest choice: a second note-on for a key
    that is already down adds no pitch and no harmony.  Extending rather
    than plain-dropping keeps the sound lasting as long as the source
    intended, which matters now that key press and sounding duration are
    separate channels.
    """
    if lo <= 21 and hi >= 108:
        return notes

    held: Dict[int, List[list]] = {}     # pitch -> [[onset, offset], ...]
    out: List[Dict] = []

    for n in sorted(notes, key=lambda x: (float(x["onset"]), int(x["pitch"]))):
        on = float(n["onset"])
        off = float(n.get("offset", on))
        p = int(n["pitch"])
        while p < lo:
            p += 12
        while p > hi:
            p -= 12

        # Candidate octaves, nearest to the folded landing point first.
        cands = [p] + [p + d for k in (1, 2) for d in (12 * k, -12 * k)]
        cands = [c for c in cands if lo <= c <= hi]

        placed = None
        for c in cands:
            spans = held.get(c, [])
            if all(on >= s[1] - 1e-6 or off <= s[0] + 1e-6 for s in spans):
                placed = c
                break

        if placed is None:
            # Every octave occupied — extend whoever holds the key.
            spans = held.get(p, [])
            for s in spans:
                if on < s[1] and off > s[0]:
                    s[1] = max(s[1], off)
                    for m in out:
                        if int(m["pitch"]) == p and abs(float(m["onset"]) - s[0]) < 1e-6:
                            m["offset"] = s[1]
                            break
                    break
            continue

        held.setdefault(placed, []).append([on, off])
        out.append(n if placed == int(n["pitch"])
                   else {**n, "pitch": placed})

    return sorted(out, key=lambda x: (float(x["onset"]), int(x["pitch"])))


ARMS = {
    "passthrough":    arm_passthrough,
    "fold":           arm_fold,
    "fold_nodouble":  arm_fold_nodouble,
}


# ── runner ────────────────────────────────────────────────────────────────────

def run(arm_names: List[str], octaves: int, limit: int) -> Dict[str, Dict]:
    lo, hi = OCTAVE_BOUNDS[octaves]
    files = sorted(glob.glob(CORPUS))[:limit]
    if not files:
        raise SystemExit(f"no corpus found at {CORPUS}")

    results: Dict[str, List[Dict]] = {a: [] for a in arm_names}
    for f in files:
        try:
            src = json.load(open(f))["notes"]
        except Exception:
            continue
        if not src:
            continue
        for a in arm_names:
            results[a].append(evaluate(src, ARMS[a](src, lo, hi), lo, hi))

    agg: Dict[str, Dict] = {}
    for a, rows in results.items():
        if not rows:
            continue
        def mean(k):
            vals = [r[k] for r in rows if r[k] is not None]
            return st.mean(vals) if vals else float("nan")
        def total(k):
            return sum(r[k] for r in rows if r[k] is not None)
        def hit(k):
            return 100.0 * sum(1 for r in rows if r[k]) / len(rows)
        agg[a] = {
            "pieces":         len(rows),
            "out_of_range":   total("out_of_range"),
            "doubled_keys":   mean("doubled_keys"),
            "doubled_worst":  max(r["doubled_keys"] for r in rows),
            "unplayable":     mean("unplayable"),
            "crossings":      mean("crossings"),
            "crossings_pct":  hit("crossings"),
            "top_breaks":     mean("top_breaks"),
            "bass_breaks":    mean("bass_breaks"),
            "bass_role_lost": mean("bass_role_lost"),
        }
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--octaves", type=int, default=5, choices=sorted(OCTAVE_BOUNDS))
    p.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json", action="store_true", help="emit raw JSON")
    args = p.parse_args()

    agg = run(args.arms, args.octaves, args.limit)
    if args.json:
        print(json.dumps(agg, indent=2))
        return

    lo, hi = OCTAVE_BOUNDS[args.octaves]
    print(f"\nArrangement eval — {args.octaves} octaves  (MIDI {lo}..{hi}, "
          f"{hi - lo + 1} keys)   lower is better everywhere\n")
    hdr = (f"{'arm':<13}{'pieces':>7}{'out-of-range':>14}{'doubled keys':>14}"
           f"{'(worst)':>9}{'unplayable':>12}{'crossings':>11}{'bass role lost':>16}")
    print(hdr)
    print("-" * len(hdr))
    for a, r in agg.items():
        print(f"{a:<13}{r['pieces']:>7}{r['out_of_range']:>14}"
              f"{r['doubled_keys']:>14.1f}{r['doubled_worst']:>9}"
              f"{r['unplayable']:>12.1f}{r['crossings']:>11.1f}"
              f"{r['bass_role_lost']:>15.2f}%")
    print()
    print("out-of-range   notes the keyboard does not have (total across pieces)")
    print("doubled keys   same pitch sounding twice at once, per piece — must be 0;")
    print("               this is the mechanism behind lost sustain on repeats")
    print("unplayable     simultaneities no two hands can take, per piece")
    print("crossings      pairs whose pitch order flipped vs source, per piece")
    print("bass role lost % of onsets where the LOWEST note's pitch class changed")
    print("               — i.e. the chord got inverted and the harmony moved\n")


if __name__ == "__main__":
    main()
