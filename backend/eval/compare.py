"""
Compare a pipeline run's transcription.json against the ground-truth MIDI.

Uses mir_eval's `transcription` module, the de-facto standard for AMT
(automatic music transcription) evaluation.  Reports both the standard
COnP (Correct Onset + Pitch) metrics and the stricter COnPOff (also
requiring the offset to land in the right ballpark).

What gets reported
------------------
For each piece:
    onset_only          — P / R / F1 with ±50 ms onset, ±50¢ pitch tolerance
    onset_and_offset    — same as above plus ±50 ms offset, 0.2× ratio
                          (mir_eval default)
    notes_pred          — how many notes the pipeline emitted
    notes_truth         — how many in the ground truth

For the whole set:
    mean F1, median F1, min F1 piece, max F1 piece (both metric tiers)

Why two tiers
-------------
COnP (onset-only) is the easier benchmark — does the transcriber spot the
right note at roughly the right time?  COnPOff is the harder one — does
it also get the duration right?  We track both because they tell us
different things:
    high COnP + low COnPOff → onset detection is good, duration
                              estimation is the weak point
    low COnP                → fundamental note-detection problem,
                              duration tier is moot

Usage
-----
    python backend/eval/compare.py <piece_id>      # single piece
    python backend/eval/compare.py --all           # every run, aggregate
    python backend/eval/compare.py --all --save-baseline   # write baseline file
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
from mir_eval.transcription import precision_recall_f1_overlap


REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "backend" / "eval" / "data" / "manifest.json"
RUNS_DIR      = REPO_ROOT / "backend" / "eval" / "runs"
BASELINES_DIR = REPO_ROOT / "backend" / "eval" / "baselines"

# mir_eval defaults for piano-style AMT
ONSET_TOL_S   = 0.05    # ±50 ms onset
PITCH_TOL_C   = 50      # ±50 cents (i.e. semitone-exact)
OFFSET_TOL_S  = 0.05
OFFSET_RATIO  = 0.2     # offset must be within max(0.05s, 0.2*duration)


def _notes_to_arrays(notes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of note-dicts to mir_eval's expected (intervals, pitches)
    format.  intervals is (N, 2) seconds; pitches is (N,) Hz.

    Filters out malformed notes: missing pitch, NaN times, offset <= onset.
    Beat-adaptive quantization downstream can occasionally produce inverted
    intervals when a note's onset gets snapped past its offset; rather than
    dropping them silently inside the pipeline (which would hide the bug),
    we filter at the eval boundary and the count of dropped notes shows up
    indirectly as a precision penalty.
    """
    if not notes:
        return np.zeros((0, 2)), np.zeros((0,))
    valid = []
    for n in notes:
        try:
            on  = float(n["onset"])
            off = float(n["offset"])
            p   = int(n["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        # mir_eval rejects any negative value, not just non-positive duration
        if (on >= 0 and off > on and 0 <= p <= 127
                and np.isfinite(on) and np.isfinite(off)):
            valid.append((on, off, p))
    if not valid:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[v[0], v[1]] for v in valid], dtype=float)
    midi      = np.array([v[2] for v in valid], dtype=int)
    pitches   = 440.0 * np.power(2.0, (midi - 69) / 12.0)
    return intervals, pitches


def _load_truth(midi_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Ground-truth (intervals, pitches) from a MAESTRO MIDI."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    intervals: List[List[float]] = []
    midis:     List[int]         = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end > n.start:
                intervals.append([float(n.start), float(n.end)])
                midis.append(int(n.pitch))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.array(intervals, dtype=float)
    pp = 440.0 * np.power(2.0, (np.array(midis) - 69) / 12.0)
    return iv, pp


def _f1_or_zero(pred_iv, pred_p, truth_iv, truth_p, offset_ratio=None
                ) -> Tuple[float, float, float]:
    """Wrap mir_eval, returning (precision, recall, f1) — zero when empty."""
    if pred_iv.shape[0] == 0 or truth_iv.shape[0] == 0:
        return 0.0, 0.0, 0.0
    kwargs = dict(
        ref_intervals     = truth_iv,
        ref_pitches       = truth_p,
        est_intervals     = pred_iv,
        est_pitches       = pred_p,
        onset_tolerance   = ONSET_TOL_S,
        pitch_tolerance   = PITCH_TOL_C,
    )
    if offset_ratio is None:
        kwargs["offset_ratio"] = None
    else:
        kwargs["offset_ratio"]     = OFFSET_RATIO
        kwargs["offset_min_tolerance"] = OFFSET_TOL_S
    p, r, f1, _ = precision_recall_f1_overlap(**kwargs)
    return float(p), float(r), float(f1)


def _compare_one(piece: Dict) -> Optional[Dict]:
    """Run the comparison for one piece.  Returns None if data is missing."""
    pid       = piece["id"]
    run_dir   = RUNS_DIR / pid
    transcr   = run_dir / "transcription.json"
    midi_path = REPO_ROOT / piece["midi_path"]

    if not transcr.exists():
        print(f"  {pid}: ! no transcription on disk (run_pipeline.py first)")
        return None
    if not midi_path.exists():
        print(f"  {pid}: ! ground-truth MIDI missing")
        return None

    with open(transcr, encoding="utf-8") as f:
        pred_notes = json.load(f).get("notes", [])
    pred_iv, pred_p = _notes_to_arrays(pred_notes)
    truth_iv, truth_p = _load_truth(midi_path)

    p_on,  r_on,  f1_on  = _f1_or_zero(pred_iv, pred_p, truth_iv, truth_p,
                                        offset_ratio=None)
    p_off, r_off, f1_off = _f1_or_zero(pred_iv, pred_p, truth_iv, truth_p,
                                        offset_ratio=OFFSET_RATIO)

    return {
        "id":         pid,
        "composer":   piece["composer"],
        "title":      piece["title"],
        "notes_pred": int(pred_iv.shape[0]),
        "notes_truth":int(truth_iv.shape[0]),
        "onset_only":      {"precision": round(p_on,  4),
                            "recall":    round(r_on,  4),
                            "f1":        round(f1_on, 4)},
        "onset_and_offset":{"precision": round(p_off, 4),
                            "recall":    round(r_off, 4),
                            "f1":        round(f1_off,4)},
    }


def _aggregate(results: List[Dict]) -> Dict:
    """Mean/median/min/max F1 across the result set."""
    if not results:
        return {}
    on  = [r["onset_only"]["f1"]       for r in results]
    off = [r["onset_and_offset"]["f1"] for r in results]
    return {
        "n_pieces": len(results),
        "onset_only": {
            "mean_f1":   round(statistics.mean(on),   4),
            "median_f1": round(statistics.median(on), 4),
            "min_f1":    round(min(on), 4),
            "max_f1":    round(max(on), 4),
            "min_piece": results[on.index(min(on))]["id"],
            "max_piece": results[on.index(max(on))]["id"],
        },
        "onset_and_offset": {
            "mean_f1":   round(statistics.mean(off),   4),
            "median_f1": round(statistics.median(off), 4),
            "min_f1":    round(min(off), 4),
            "max_f1":    round(max(off), 4),
            "min_piece": results[off.index(min(off))]["id"],
            "max_piece": results[off.index(max(off))]["id"],
        },
    }


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("piece_id", nargs="?", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Compare every piece in the manifest")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Write baselines/baseline_<sha>_<ts>.json")
    parser.add_argument("--compare-to", type=str, default=None,
                        help="Path to a baseline JSON to diff against")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit("Manifest missing. Run select_test_set.py first.")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    pieces: List[Dict] = manifest["pieces"]
    if args.all:
        target = pieces
    elif args.piece_id:
        target = [p for p in pieces if p["id"] == args.piece_id]
        if not target:
            raise SystemExit(f"No piece with id={args.piece_id}")
    else:
        parser.print_help()
        return

    print(f"Comparing {len(target)} piece(s)...")
    results: List[Dict] = []
    for p in target:
        r = _compare_one(p)
        if r is None:
            continue
        results.append(r)
        on, off = r["onset_only"]["f1"], r["onset_and_offset"]["f1"]
        print(f"  {r['id'][:50]:<50}  COnP={on:.3f}  COnPOff={off:.3f}  "
              f"pred={r['notes_pred']:>5} truth={r['notes_truth']:>5}")

    if not results:
        print("\nNo results to aggregate.")
        return

    agg = _aggregate(results)
    print(f"\nAggregate over {agg['n_pieces']} pieces:")
    print(f"  COnP    (onset-only):       mean F1 = {agg['onset_only']['mean_f1']:.4f}, "
          f"median = {agg['onset_only']['median_f1']:.4f}, "
          f"min = {agg['onset_only']['min_f1']:.4f}, "
          f"max = {agg['onset_only']['max_f1']:.4f}")
    print(f"  COnPOff (onset+offset):     mean F1 = {agg['onset_and_offset']['mean_f1']:.4f}, "
          f"median = {agg['onset_and_offset']['median_f1']:.4f}, "
          f"min = {agg['onset_and_offset']['min_f1']:.4f}, "
          f"max = {agg['onset_and_offset']['max_f1']:.4f}")
    print(f"  Worst piece (COnP):  {agg['onset_only']['min_piece']}")
    print(f"  Best  piece (COnP):  {agg['onset_only']['max_piece']}")

    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        sha = _git_sha() or "nogit"
        ts  = time.strftime("%Y%m%d_%H%M%S")
        path = BASELINES_DIR / f"baseline_{sha}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "git_sha":   sha,
                "timestamp": ts,
                "aggregate": agg,
                "pieces":    results,
            }, f, indent=2)
        print(f"\nBaseline saved -> {path.relative_to(REPO_ROOT)}")

    if args.compare_to:
        with open(args.compare_to, encoding="utf-8") as f:
            base = json.load(f)
        base_pieces = {r["id"]: r for r in base.get("pieces", [])}
        regressed = []
        improved  = []
        for r in results:
            b = base_pieces.get(r["id"])
            if not b:
                continue
            d = r["onset_only"]["f1"] - b["onset_only"]["f1"]
            if d <= -0.01:
                regressed.append((r["id"], d))
            elif d >= 0.01:
                improved.append((r["id"], d))
        print(f"\nVs baseline {Path(args.compare_to).name}:")
        print(f"  improved (dF1 >= 0.01): {len(improved)} pieces")
        print(f"  regressed (dF1 <= -0.01): {len(regressed)} pieces")
        for pid, d in sorted(regressed, key=lambda x: x[1])[:5]:
            print(f"    REGRESS {pid:<50} dF1 = {d:+.4f}")


if __name__ == "__main__":
    main()
