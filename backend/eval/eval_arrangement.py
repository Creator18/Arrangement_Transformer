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

from app.services.nmf_voice_leading import (                        # noqa: E402
    compress_notes_to_range, fit_notes_to_range,
)
from eval.arrangement_metrics import evaluate, rule_violations        # noqa: E402

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
    The shipped policy — delegates to fit_notes_to_range.

    Deliberately a one-line call rather than a copy.  An earlier version of
    this file reimplemented the policy, and the moment production changed
    (extend-held-note became release-and-re-strike) the eval silently went
    on measuring code that no longer ran anywhere.  An eval that does not
    test the shipped function is worse than no eval, because it reports
    confidence it has not earned.
    """
    return fit_notes_to_range(notes, lo, hi)


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
            out = ARMS[a](src, lo, hi)
            row = evaluate(src, out, lo, hi)
            row.update(rule_violations(src, out))
            results[a].append(row)

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
            "onset_moved":    mean("onset_moved_pct"),
            "pitch_added":    mean("pitch_added_pct"),
            "kept":           mean("kept_pct"),
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
    print("Source-fidelity rule (R2/R3 must be 0.00; dropping notes is legal):")
    hdr2 = f"{'arm':<15}{'R2 onsets moved':>18}{'R3 pitch added':>17}{'notes kept':>13}"
    print(hdr2)
    print("-" * len(hdr2))
    for a, r in agg.items():
        print(f"{a:<15}{r['onset_moved']:>17.2f}%{r['pitch_added']:>16.2f}%"
              f"{r['kept']:>12.2f}%")
    print()
    print("bass role lost % of onsets where the LOWEST note's pitch class changed")
    print("               — i.e. the chord got inverted and the harmony moved\n")


if __name__ == "__main__":
    main()
