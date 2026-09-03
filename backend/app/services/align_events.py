"""Make notes that belong to one strike agree with each other.

Measured on the baseline render: of the 616 events containing more than one
note, 82% were spread wider than 20 ms, with a median spread of 46 ms and a p90
of 93 ms.  That is the "out of sync" complaint, and it is not a lag — the
global onset sweep peaks at exactly 0 ms, so on average the timing is right.
It is SCATTER.

Why it happens: every pitch is a separate row of the activation matrix and is
detected on its own, each carrying its own error from a 371 ms analysis
window.  Nothing ties the rows together.  So a single guitar strum — five
notes made physically simultaneous by one pick stroke — comes back as five
independent estimates smeared across 46 ms.

Why it sounds so much worse than the source it came from: a guitar, a pad or a
voice has a soft attack, and a 46 ms disagreement disappears inside it.  A
piano sample has a hard percussive transient, so the same 46 ms becomes an
audible flam.  And in the original, timbre and stereo position help the ear
bind those notes into one chord; on a piano everything is one instrument in one
place, so TIMING IS THE ONLY GROUPING CUE LEFT — precisely the cue the
decomposition degrades.

This adds and removes nothing.  Every attempt at dropping notes made the
arrangement worse by ear, so the operation here is alignment: notes already
detected are moved onto a shared instant.

Not quantisation.  The target comes from the notes' own evidence and from the
broadband onset envelope, never from a metrical grid — snapping to a grid was
measured as a no-op, since the notes were already 99.3% on one.  The goal is
for notes to agree WITH EACH OTHER, which is a different property from
agreeing with a ruler.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Half a sixteenth at 136 BPM is 55 ms, so a window at 60 ms cannot swallow a
# genuine sixteenth-note figure.  Everything inside it is far more likely to be
# one strike scattered than two events that close together.
MAX_EVENT_WIDTH = 0.060
ONSET_SNAP_TOL = 0.050          # how far to look for a broadband attack


def _cluster(onsets: np.ndarray, max_width: float) -> List[List[int]]:
    """Group indices into events of bounded WIDTH, not bounded gap.

    Chaining is the trap: with a gap rule alone, notes at 0, 40 and 80 ms link
    into one 80 ms event, and aligning those would move a note further than the
    error being corrected.  Bounding the total width stops a cluster growing
    arbitrarily by small steps.
    """
    out: List[List[int]] = []
    cur: List[int] = []
    start = None
    for i in np.argsort(onsets):
        t = float(onsets[i])
        if start is None or t - start <= max_width:
            if start is None:
                start = t
            cur.append(int(i))
        else:
            out.append(cur)
            cur, start = [int(i)], t
    if cur:
        out.append(cur)
    return out


def align_events(notes: List[Dict[str, Any]],
                 onsets: Optional[np.ndarray] = None,
                 max_width: float = MAX_EVENT_WIDTH,
                 snap_tol: float = ONSET_SNAP_TOL
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collapse each cluster of near-simultaneous notes onto one instant."""
    if not notes:
        return notes, {"events": 0, "moved": 0}

    src = [dict(n) for n in notes]
    on = np.array([float(n["onset"]) for n in src])
    ons = np.sort(np.asarray(onsets, float)) if onsets is not None and len(onsets) else None

    out: List[Dict[str, Any]] = []
    moved = 0
    multi = 0
    for grp in _cluster(on, max_width):
        if len(grp) == 1:
            out.append(src[grp[0]])
            continue
        multi += 1
        times = on[grp]
        vel = np.array([float(src[i].get("velocity", 80)) for i in grp])

        # The loudest note in a chord is the one whose attack the decomposition
        # localised best, so the cluster is weighted toward it rather than
        # averaged flat — a faint harmonic ghost should not drag the strike.
        target = float(np.average(times, weights=np.maximum(vel, 1.0)))
        if ons is not None:
            j = int(np.searchsorted(ons, target))
            cands = [ons[k] for k in (j - 1, j) if 0 <= k < len(ons)]
            near = [c for c in cands if abs(c - target) <= snap_tol]
            if near:
                # a detected attack is independent evidence, at 5.8 ms against
                # the pitchmap's 23 ms, so it outranks our own average
                target = float(min(near, key=lambda c: abs(c - target)))

        # One pitch twice inside a single strike is not a chord, it is the same
        # note found twice; aligning both would stack two identical notes at
        # one instant, which is the double-press artefact.
        best_by_pitch: Dict[int, Dict[str, Any]] = {}
        for i in grp:
            n = src[i]
            p = int(n["pitch"])
            keep = best_by_pitch.get(p)
            if keep is None or float(n.get("velocity", 0)) > float(keep.get("velocity", 0)):
                best_by_pitch[p] = n
        for n in best_by_pitch.values():
            dur = float(n["offset"]) - float(n["onset"])
            if abs(float(n["onset"]) - target) > 1e-6:
                moved += 1
            n["onset"] = max(0.0, target)
            n["offset"] = max(0.0, target) + max(dur, 0.05)
            out.append(n)

    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out, {"events": multi, "moved": moved,
                 "dropped_dupes": len(src) - len(out)}


def spread_report(notes: List[Dict[str, Any]], tol: float = 0.05) -> Dict[str, float]:
    """Median / p90 spread inside multi-note events — the metric being fixed."""
    if not notes:
        return {}
    o = np.sort(np.array([float(n["onset"]) for n in notes]))
    clusters, cur = [], [o[0]]
    for t in o[1:]:
        if t - cur[-1] <= tol:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)
    sp = np.array([(c[-1] - c[0]) * 1000 for c in clusters if len(c) > 1])
    if not sp.size:
        return {"multi_events": 0}
    return {"multi_events": int(sp.size), "median_ms": float(np.median(sp)),
            "p90_ms": float(np.percentile(sp, 90)),
            "pct_over_20ms": float(100 * np.mean(sp > 20))}
