"""
Per-piece error breakdown: what kinds of mistakes are we making, and
where are they coming from.

Reads `transcription.json` (post-cleanup ORIGINAL output) + ground-truth
MIDI, then categorises every mismatch.  Numbers here are diagnostic, not
benchmarks — they tell us *which* fix will move the F1 needle most.

Categories of unmatched predictions (false positives)
-----------------------------------------------------
For each predicted note that doesn't match any ground-truth note within
the standard tolerance (±50 ms onset, ±50¢ pitch):

  near_pitch       Truth note exists at the same time but differs by
                   ±1 semitone — our pitch was off by one
  octave_dup       Truth note exists at the same time, ±12 semitones —
                   we're detecting a harmonic partial as a separate note
  timing_off       Truth note at same pitch within 200 ms — our onset
                   was just outside the tolerance window
  spurious         No nearby truth note at all — pure false positive
                   (Basic Pitch hallucination, separation artefact)

Categories of unmatched ground-truth (false negatives)
------------------------------------------------------
  near_pitch       We have a note at the right time, off by ±1 semitone
  octave_dup       We have a note at the right time, ±12 semitones
  timing_off       We have a same-pitch note within 200 ms but missed
                   the window
  missed           No nearby prediction at all — the model didn't see it

Per-layer breakdown shows which separation layer the FP came from, so
we can tell whether the pattern stem or melody stem is responsible.

Output
------
Per-piece error counts + percentages, plus an aggregate "where is the
F1 loss coming from" table:

  spurious      52%  ← biggest lever: model hallucinations
  octave_dup    18%  ← second: octave-up partial detection
  timing_off    20%  ← onset quantization drift
  near_pitch     9%  ← off-by-one pitch errors
  missed         1%

Once we know which category dominates, we know which fix to prioritise:
  spurious     → tighter Basic Pitch threshold (Tier-A confidence slider)
  octave_dup   → octave-merge dedupe pass in transcription.py
  timing_off   → re-anchor quantization grid, or disable for eval
  near_pitch   → scale-snap is already addressing this; check if it's firing
  missed       → loosen Basic Pitch threshold, or train better model

Usage
-----
    python backend/eval/analyze_errors.py <piece_id>
    python backend/eval/analyze_errors.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi


REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "backend" / "eval" / "data" / "manifest.json"
RUNS_DIR      = REPO_ROOT / "backend" / "eval" / "runs"

ONSET_TOL_S   = 0.05    # match window — same as compare.py
PITCH_TOL_ST  = 0       # 0 = semitone-exact (50¢ → no pitch tolerance in practice)
NEAR_PITCH_ST = 1       # ±1 semitone counts as "near pitch"
OCTAVE_ST     = 12
TIMING_OFF_S  = 0.2     # FP/FN whose nearest counterpart is within 200 ms
                        # (but outside the strict 50 ms match window)


def _load_truth(midi_path: Path) -> List[Dict]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end > n.start:
                notes.append({
                    "pitch":    int(n.pitch),
                    "onset":    float(n.start),
                    "offset":   float(n.end),
                    "velocity": int(n.velocity),
                })
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes


def _greedy_match(pred: List[Dict], truth: List[Dict]
                  ) -> Tuple[set, set, List[Tuple[int,int]]]:
    """
    Greedy 1-to-1 matching between predictions and ground truth.  A pair
    (p_i, t_j) matches iff |onset_p - onset_t| <= ONSET_TOL_S AND pitch_p
    == pitch_t.  Each pred and truth used at most once.

    Returns (matched_pred_idx, matched_truth_idx, list_of_pairs).
    """
    matched_p = set()
    matched_t = set()
    pairs = []

    truth_by_pitch: Dict[int, List[int]] = {}
    for j, t in enumerate(truth):
        truth_by_pitch.setdefault(t["pitch"], []).append(j)

    for i, p in enumerate(pred):
        pp = int(p["pitch"])
        po = float(p["onset"])
        candidates = truth_by_pitch.get(pp, [])
        best_j   = None
        best_dt  = ONSET_TOL_S + 1
        for j in candidates:
            if j in matched_t:
                continue
            dt = abs(float(truth[j]["onset"]) - po)
            if dt <= ONSET_TOL_S and dt < best_dt:
                best_dt, best_j = dt, j
        if best_j is not None:
            matched_p.add(i)
            matched_t.add(best_j)
            pairs.append((i, best_j))
    return matched_p, matched_t, pairs


def _categorise_fp(p: Dict, truth: List[Dict]) -> str:
    """
    Classify a false-positive prediction by its closest ground-truth
    neighbour.  Order of checks reflects the explanatory hierarchy:
    pitch errors and harmonic doublings are more diagnostic than
    timing-off, which is more diagnostic than fully spurious.
    """
    po = float(p["onset"])
    pp = int(p["pitch"])

    # Limit search to a small temporal window around the prediction
    window_lo = po - TIMING_OFF_S
    window_hi = po + TIMING_OFF_S

    near_pitch  = False
    octave_dup  = False
    timing_off  = False
    for t in truth:
        to = float(t["onset"])
        if to < window_lo: continue
        if to > window_hi: break
        tp = int(t["pitch"])
        dt = abs(to - po)
        if dt <= ONSET_TOL_S:
            if abs(tp - pp) == NEAR_PITCH_ST:
                near_pitch = True
            elif abs(tp - pp) == OCTAVE_ST:
                octave_dup = True
        elif tp == pp and dt <= TIMING_OFF_S:
            timing_off = True

    # Truth assumed sorted; we break on > window_hi
    # Above loop walked through whole list; small optimisation skipped for clarity.

    if near_pitch:  return "near_pitch"
    if octave_dup:  return "octave_dup"
    if timing_off:  return "timing_off"
    return "spurious"


def _categorise_fn(t: Dict, pred: List[Dict]) -> str:
    """Symmetric to _categorise_fp but for missed ground-truth notes."""
    to = float(t["onset"])
    tp = int(t["pitch"])
    window_lo = to - TIMING_OFF_S
    window_hi = to + TIMING_OFF_S

    near_pitch  = False
    octave_dup  = False
    timing_off  = False
    for p in pred:
        po = float(p["onset"])
        if po < window_lo: continue
        if po > window_hi: break
        pp = int(p["pitch"])
        dt = abs(po - to)
        if dt <= ONSET_TOL_S:
            if abs(pp - tp) == NEAR_PITCH_ST:
                near_pitch = True
            elif abs(pp - tp) == OCTAVE_ST:
                octave_dup = True
        elif pp == tp and dt <= TIMING_OFF_S:
            timing_off = True

    if near_pitch:  return "near_pitch"
    if octave_dup:  return "octave_dup"
    if timing_off:  return "timing_off"
    return "missed"


def _analyze_one(piece: Dict) -> Dict:
    pid       = piece["id"]
    run_dir   = RUNS_DIR / pid
    transcr   = run_dir / "transcription.json"
    midi_path = REPO_ROOT / piece["midi_path"]

    if not transcr.exists() or not midi_path.exists():
        return {"id": pid, "status": "skipped"}

    with open(transcr, encoding="utf-8") as f:
        pred = json.load(f).get("notes", [])
    truth = _load_truth(midi_path)
    pred = sorted(pred, key=lambda n: (float(n["onset"]), int(n["pitch"])))

    matched_p, matched_t, _ = _greedy_match(pred, truth)

    fp_categories  : Dict[str, int] = {"near_pitch": 0, "octave_dup": 0,
                                       "timing_off": 0, "spurious": 0}
    fp_by_layer    : Dict[str, int] = {}
    for i, p in enumerate(pred):
        if i in matched_p: continue
        cat = _categorise_fp(p, truth)
        fp_categories[cat] += 1
        layer = p.get("layer", "?")
        fp_by_layer[layer] = fp_by_layer.get(layer, 0) + 1

    fn_categories : Dict[str, int] = {"near_pitch": 0, "octave_dup": 0,
                                      "timing_off": 0, "missed": 0}
    for j, t in enumerate(truth):
        if j in matched_t: continue
        cat = _categorise_fn(t, pred)
        fn_categories[cat] += 1

    n_pred  = len(pred)
    n_truth = len(truth)
    n_match = len(matched_p)
    precision = n_match / n_pred  if n_pred  else 0.0
    recall    = n_match / n_truth if n_truth else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return {
        "id":          pid,
        "status":      "ok",
        "composer":    piece["composer"],
        "n_pred":      n_pred,
        "n_truth":     n_truth,
        "n_match":     n_match,
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "fp_total":    n_pred - n_match,
        "fp_breakdown":fp_categories,
        "fp_by_layer": fp_by_layer,
        "fn_total":    n_truth - n_match,
        "fn_breakdown":fn_categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("piece_id", nargs="?", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit("Manifest missing.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    pieces = manifest["pieces"]
    if args.all:
        target = pieces
    elif args.piece_id:
        target = [p for p in pieces if p["id"] == args.piece_id]
        if not target:
            raise SystemExit(f"No piece with id={args.piece_id}")
    else:
        parser.print_help()
        return

    results = []
    for p in target:
        r = _analyze_one(p)
        if r["status"] != "ok":
            continue
        results.append(r)
        print(f"  {r['id'][:50]:<50}  "
              f"F1={r['f1']:.3f}  "
              f"FP {r['fp_total']:>4} "
              f"(sp={r['fp_breakdown']['spurious']:>3} "
              f"oct={r['fp_breakdown']['octave_dup']:>3} "
              f"t={r['fp_breakdown']['timing_off']:>3} "
              f"p={r['fp_breakdown']['near_pitch']:>3})  "
              f"FN {r['fn_total']:>4} "
              f"(m={r['fn_breakdown']['missed']:>3} "
              f"oct={r['fn_breakdown']['octave_dup']:>3} "
              f"t={r['fn_breakdown']['timing_off']:>3} "
              f"p={r['fn_breakdown']['near_pitch']:>3})")

    if not results:
        return

    # Aggregate breakdown
    print()
    print("=" * 78)
    print(f"Aggregate over {len(results)} pieces:")
    print()

    total_fp = sum(r["fp_total"] for r in results)
    total_fn = sum(r["fn_total"] for r in results)
    fp_sum: Dict[str, int] = {"near_pitch":0, "octave_dup":0,
                              "timing_off":0, "spurious":0}
    fn_sum: Dict[str, int] = {"near_pitch":0, "octave_dup":0,
                              "timing_off":0, "missed":0}
    layer_sum: Dict[str, int] = {}
    for r in results:
        for k, v in r["fp_breakdown"].items():
            fp_sum[k] += v
        for k, v in r["fn_breakdown"].items():
            fn_sum[k] += v
        for k, v in r["fp_by_layer"].items():
            layer_sum[k] = layer_sum.get(k, 0) + v

    print(f"False-positive total: {total_fp}")
    for cat in ("spurious", "octave_dup", "timing_off", "near_pitch"):
        pct = 100 * fp_sum[cat] / total_fp if total_fp else 0
        print(f"  {cat:<12}  {fp_sum[cat]:>6}  ({pct:>5.1f}%)")
    print()
    print("FP by layer:")
    for layer, c in sorted(layer_sum.items(), key=lambda x: -x[1]):
        pct = 100 * c / total_fp if total_fp else 0
        print(f"  {layer:<14}  {c:>6}  ({pct:>5.1f}%)")
    print()
    print(f"False-negative total: {total_fn}")
    for cat in ("missed", "octave_dup", "timing_off", "near_pitch"):
        pct = 100 * fn_sum[cat] / total_fn if total_fn else 0
        print(f"  {cat:<12}  {fn_sum[cat]:>6}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
