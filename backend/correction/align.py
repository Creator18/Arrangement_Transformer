"""
align.py — put our extraction and the score into the same time base, then
pair them up.

The problem: our notes live in performance SECONDS (they come from audio),
the score lives in notated BEATS.  Rubato means the mapping between them is
non-linear and piece-specific.

The shortcut: ASAP ships beat annotations for BOTH sides —
    <performance>_annotations.txt   beat times in performance seconds
    midi_score_annotations.txt      the same beats in score time
with one line per beat in the same order, so beat i in one file IS beat i in
the other.  That gives an exact piecewise-linear seconds<->beats map for
free.  This is the whole reason ASAP was chosen over just using MAESTRO:
building a DTW aligner good enough to trust would have been the single
biggest time sink in the correction-model plan.

Pairing is greedy nearest-match in beat space, gated on pitch equality —
the same convention the eval harness uses, so "matched" means the same
thing in both places.

Output per piece: for every note our pipeline produced, either
    - the score note it corresponds to (a CORRECTION TARGET: how the
      duration, hand and spelling should have come out), or
    - nothing (a SPURIOUS note the model should learn to delete)
plus every score note we never found (a MISSED note).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── beat maps ────────────────────────────────────────────────────────────────

def load_annotations(path: Path) -> List[float]:
    """First column of an ASAP annotation file = the beat's time."""
    times: List[float] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        try:
            times.append(float(parts[0]))
        except (ValueError, IndexError):
            continue
    return times


class BeatMap:
    """Piecewise-linear performance-seconds <-> score-beats conversion."""

    def __init__(self, perf_times: Sequence[float], score_times: Sequence[float]):
        n = min(len(perf_times), len(score_times))
        if n < 2:
            raise ValueError("need at least two aligned beats")
        self.perf  = np.asarray(perf_times[:n],  dtype=float)
        self.score = np.asarray(score_times[:n], dtype=float)
        self.n = n
        # Annotation files are ordered but let's not assume monotonicity
        order = np.argsort(self.perf)
        self.perf, self.score = self.perf[order], self.score[order]

    def to_beats(self, t: float) -> float:
        return float(np.interp(t, self.perf, self.score,
                               left=self.score[0]
                                    + (t - self.perf[0]),
                               right=self.score[-1]
                                    + (t - self.perf[-1])))

    def to_seconds(self, b: float) -> float:
        return float(np.interp(b, self.score, self.perf))

    @classmethod
    def for_performance(cls, folder: Path, perf_annotations: str) -> "BeatMap":
        """
        Build the map for one performance.

        UNITS TRAP: midi_score_annotations.txt is in SECONDS of ASAP's
        rendered score MIDI (120 bpm by convention), not in quarter-note
        beats — its entries step by 0.5 where the score steps by 1.0
        quarter.  Our score targets come from music21, whose offsets ARE
        quarters.  Feeding the two together unconverted matched 14.6% of a
        performance against its own score; converted it should be near
        total.  The scale is read from the score MIDI's own tempo rather
        than hardcoded, so a piece rendered at another tempo still works.
        """
        score_secs = load_annotations(folder / "midi_score_annotations.txt")
        qps = 2.0                                   # 120 bpm fallback
        try:
            import pretty_midi
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm = pretty_midi.PrettyMIDI(str(folder / "midi_score.mid"))
            _, bpms = pm.get_tempo_changes()
            if len(bpms):
                qps = float(bpms[0]) / 60.0
        except Exception:
            pass
        return cls(load_annotations(folder / perf_annotations),
                   [s * qps for s in score_secs])


# ── pairing ──────────────────────────────────────────────────────────────────

def align_notes(
    ours:        Sequence[Dict],
    score_notes: Sequence[Dict],
    beat_map:    BeatMap,
    tol_beats:   float = 0.25,
) -> Dict:
    """
    Match our (seconds) notes against score (beats) notes.

    tol_beats defaults to a 16th note at 4/4 — wide enough to absorb our
    onset error plus annotation slack, tight enough that we do not pair a
    note with its neighbour.

    Returns matched pairs plus the two error sets, which are exactly the
    three things a correction model must learn: fix this, delete this,
    insert this.
    """
    ours_b = [(beat_map.to_beats(float(n["onset"])),
               beat_map.to_beats(float(n.get("offset", n["onset"]))), n)
              for n in ours]

    by_pitch: Dict[int, List[int]] = {}
    for j, s in enumerate(score_notes):
        by_pitch.setdefault(int(s["pitch"]), []).append(j)

    used: set = set()
    matched: List[Dict] = []
    spurious: List[Dict] = []

    for ob, oe, n in sorted(ours_b, key=lambda x: x[0]):
        cands = by_pitch.get(int(n["pitch"]), [])
        best, bestd = None, tol_beats + 1
        for j in cands:
            if j in used:
                continue
            d = abs(float(score_notes[j]["onset_b"]) - ob)
            if d <= tol_beats and d < bestd:
                best, bestd = j, d
        if best is None:
            spurious.append({"pitch": int(n["pitch"]), "onset_b": round(ob, 4),
                             "dur_b": round(max(0.0, oe - ob), 4)})
            continue
        used.add(best)
        s = score_notes[best]
        matched.append({
            "pitch":        int(n["pitch"]),
            "ours_onset_b": round(ob, 4),
            "ours_dur_b":   round(max(0.0, oe - ob), 4),
            "ours_vel":     int(n.get("velocity", 80)),
            "ours_layer":   n.get("layer"),
            "tgt_onset_b":  s["onset_b"],
            "tgt_dur_b":    s["dur_b"],
            "tgt_dur_name": s["dur_name"],
            "tgt_hand":     s["hand"],
            "tgt_spelling": s["spelling"],
            "tgt_tie":      s.get("tie"),
            "onset_err_b":  round(ob - float(s["onset_b"]), 4),
            "dur_ratio":    round(max(0.0, oe - ob) / max(1e-6, float(s["dur_b"])), 3),
        })

    missed = [{"pitch": int(s["pitch"]), "onset_b": s["onset_b"],
               "dur_b": s["dur_b"], "hand": s["hand"]}
              for j, s in enumerate(score_notes) if j not in used]

    return {
        "matched": matched, "spurious": spurious, "missed": missed,
        "stats": {
            "n_ours": len(ours), "n_score": len(score_notes),
            "n_matched": len(matched), "n_spurious": len(spurious),
            "n_missed": len(missed),
            "precision": round(len(matched) / max(1, len(ours)), 4),
            "recall":    round(len(matched) / max(1, len(score_notes)), 4),
        },
    }


def summarise_corrections(aligned: Dict) -> Dict:
    """
    What a correction model would have to learn from this piece — reported
    as the size of each systematic error, not just the counts.
    """
    m = aligned["matched"]
    if not m:
        return {}
    dur_ratio = np.array([x["dur_ratio"] for x in m], dtype=float)
    onset_err = np.array([x["onset_err_b"] for x in m], dtype=float)
    return {
        "duration_ratio_median": round(float(np.median(dur_ratio)), 3),
        "duration_too_short_pct": round(100 * float(np.mean(dur_ratio < 0.8)), 1),
        "duration_too_long_pct":  round(100 * float(np.mean(dur_ratio > 1.25)), 1),
        "onset_bias_beats":       round(float(np.median(onset_err)), 4),
        "onset_spread_beats":     round(float(np.std(onset_err)), 4),
    }
