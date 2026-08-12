"""
Turn the voice register from a cloud of detections into a line.

Two filters already run before this and neither can touch the problem.  The
scale cleanup at /segment drops out-of-key notes (4622 -> 3971 on a real
upload) and snap_to_beat_grid puts every onset on a sixteenth.  The clutter
survives both, and the reason is structural rather than a matter of tuning
them harder:

A piano template IS a decay curve, so NMF cannot hold a sustained sung note —
its only way to keep the level up is to strike the note again, and again.
Those re-strikes are the SAME PITCH as the real note, so they are always in
scale, and once snapped they sit neatly on the grid.  Both filters pass every
one of them.  The result is clutter that is perfectly in key and perfectly in
time.

Vibrato is the partial exception: it smears onto neighbouring semitones, and
roughly half of those fall outside the scale, so the existing filter already
catches some.  The half that lands on an in-scale neighbour survives.

So the missing constraint is MONOPHONY.  A sung line is one note at a time,
and that single assumption removes exactly the errors the other two cannot
see.  Rhythm and scale still do real work here, but repurposed:

  * a COARSER grid (eighths, not sixteenths) collapses consecutive re-strikes
    into the same slot so they can be merged, instead of merely tidying where
    each fragment sits;
  * the scale is a TIEBREAK, not a filter — when two candidates sound at the
    same instant, the in-scale one wins.  As a hard filter it would delete
    genuinely chromatic melody notes, which changes the tune, and the tune is
    the one thing here that must not change.

Only the register above `split` is touched.  Everything below is the
accompaniment's business.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

# These three were guesses until POP909 was measured — 909 human-notated vocal
# lines, 309,423 notes (eval/pop909_melody_priors.py).  Three of the four
# guesses were wrong, and each was wrong in the direction of destroying real
# melody:
#
#   MELODY_SUBDIV was 2 (eighths).  Only 42.7% of human vocal onsets sit on an
#   eighth grid; 76.4% sit on a sixteenth.  Snapping to eighths was moving the
#   majority of notes OFF their position rather than onto the beat.  Note this
#   contradicts the "coarser grid collapses re-strikes" reasoning — collapsing
#   re-strikes is the bridging step's job, and buying it with a grid this coarse
#   costs more than it returns.
#
#   MIN_DUR was 0.12 s.  The real 5th percentile is 0.109 and 7.6% of sung
#   notes fall under 0.12, so the filter was deleting one real note in
#   thirteen.  0.10 sits just under p5 and costs ~3%.
#
#   GAP_BRIDGE was 0.10 s.  Median gap between two SAME-PITCH sung notes is
#   0.121 s and 35.8% of them are under 0.10 — so bridging at 0.10 welded more
#   than a third of genuinely repeated notes into one.  Repeated notes are not
#   an edge case in pop: 23.6% of all melodic intervals are a unison.  NMF
#   re-strikes inside one sustained note sit far tighter than a sung repeat, so
#   0.05 still catches them.
#
# The monophony assumption survived: 9.4% of consecutive human melody notes
# overlap, but as brief legato joins rather than two-note singing — which is
# why this truncates the earlier note instead of deleting either one.
# Swept against the POP909 distributions on two real uploads, once the
# late-binding bug above was fixed.  Target: 1.37 notes/s, median duration
# 0.291 s, 23.6% unisons.
#
#   subdiv   4    CONFIRMED.  At min_dur 0.12-0.15 the sixteenth grid beats the
#                 eighth on every axis; eighth-snapping displaces onsets that
#                 76.4% of human vocal notes place on sixteenths.
#   gap   0.05    CONFIRMED.  0.05 and 0.10 tie; 0.15 is clearly worse, welding
#                 repeated notes (unison share falls to 7.7%).
#   min_dur 0.15  CORRECTED, and in the OPPOSITE direction to the POP909
#                 percentile.  0.109 is the 5th percentile of HUMAN note
#                 lengths, so 0.10 looked right — but our short notes are
#                 disproportionately artefacts, not sung notes.  Measured:
#                 0.10 -> 1.65 notes/s, unison 8.6%;  0.15 -> 1.12 notes/s,
#                 unison 21.2% against a human 23.6%.  A percentile tells you
#                 what a threshold COSTS in real notes and nothing about what it
#                 REMOVES in noise, and only the second one is why it exists.
#
# The trade at 0.15 is honest and worth stating: density lands 18% below human
# (1.12 vs 1.37) in exchange for repeated-note structure that is nearly right.
# The composite that picks it weights density, duration and unison equally,
# which is my choice, not a measured one.
# OFF by default: melody_decode supersedes this entirely.
#
# Leaving it on would actively undo the decode.  _winner_timeline merges every
# contiguous same-pitch frame into one note, which is exactly the welding the
# decoder's re-articulation pass exists to prevent — it would take unison share
# back from 23.2% (human: 23.6%) to zero.  MIN_DUR = 0.15 would also delete the
# short restatements the decode just recovered.
#
# Kept for A/B: set True and DECODE_MELODY False to compare filtering against
# decoding on the same audio.
CLEAN_MELODY = False
FPS = 100.0               # resolution for the monophonic reduction
MELODY_SUBDIV = 4         # sixteenths — where 76.4% of sung onsets land
MIN_DUR = 0.15            # tuned on our own noisy notes, not on human ones
GAP_BRIDGE = 0.05         # tight enough to spare genuine repeated notes
SCALE_BONUS = 1.18        # how much being in key is worth in a tie


def _winner_timeline(melody: List[Dict[str, Any]], scale_pcs: Optional[Set[int]],
                     n_fr: int) -> np.ndarray:
    """Per frame, the single pitch that wins the register."""
    top = np.zeros(n_fr, dtype=np.int16)
    best = np.zeros(n_fr, dtype=float)
    for n in melody:
        p = int(n["pitch"])
        a = int(float(n["onset"]) * FPS)
        b = min(int(float(n["offset"]) * FPS), n_fr)
        if b <= a:
            continue
        score = float(n.get("velocity", 80))
        if scale_pcs and (p % 12) in scale_pcs:
            score *= SCALE_BONUS
        seg = best[a:b]
        win = seg < score
        seg[win] = score
        top[a:b][win] = p
    return top


def _runs_to_notes(top: np.ndarray, vel_of: Dict[int, int]) -> List[Dict[str, Any]]:
    """Contiguous same-pitch frames become one note.

    This is where re-striking dies: a note struck five times inside one sung
    syllable occupies one continuous run of frames at that pitch, and a run is
    a note.
    """
    out, i, n = [], 0, len(top)
    while i < n:
        if top[i] == 0:
            i += 1
            continue
        j = i
        while j < n and top[j] == top[i]:
            j += 1
        p = int(top[i])
        out.append({
            "pitch": p,
            "onset": i / FPS,
            "offset": j / FPS,
            "velocity": int(vel_of.get(p, 80)),
        })
        i = j
    return out


def _bridge(notes: List[Dict[str, Any]], gap: Optional[float] = None) -> List[Dict[str, Any]]:
    """Close short silences between runs of the same pitch.

    `gap` defaults to GAP_BRIDGE read AT CALL TIME, not as a default argument.
    Binding it in the signature captured the value at import, so a sweep that
    reassigned the module constant silently measured nothing — three parameter
    settings produced byte-identical results and looked like a real finding.
    """
    gap = GAP_BRIDGE if gap is None else gap
    out: List[Dict[str, Any]] = []
    for n in sorted(notes, key=lambda x: float(x["onset"])):
        if out and int(out[-1]["pitch"]) == int(n["pitch"]) \
                and float(n["onset"]) - float(out[-1]["offset"]) <= gap:
            out[-1]["offset"] = max(float(out[-1]["offset"]), float(n["offset"]))
            out[-1]["velocity"] = max(int(out[-1]["velocity"]), int(n["velocity"]))
        else:
            out.append(dict(n))
    return out


def _snap(notes: List[Dict[str, Any]], beat_times: Optional[np.ndarray],
          subdiv: Optional[int] = None) -> List[Dict[str, Any]]:
    # Read at call time — see the note in _bridge.
    subdiv = MELODY_SUBDIV if subdiv is None else subdiv
    if beat_times is None or len(beat_times) < 2:
        return notes
    bt = np.asarray(beat_times, dtype=float)
    grid = []
    for a, b in zip(bt[:-1], bt[1:]):
        for k in range(subdiv):
            grid.append(a + (b - a) * k / subdiv)
    grid.append(float(bt[-1]))
    grid = np.asarray(grid)
    out = []
    for n in notes:
        m = dict(n)
        on = float(grid[int(np.argmin(np.abs(grid - float(n["onset"]))))])
        off = float(grid[int(np.argmin(np.abs(grid - float(n["offset"]))))])
        if off - on < MIN_DUR:
            off = on + max(MIN_DUR, float(n["offset"]) - float(n["onset"]))
        m["onset"], m["offset"] = on, off
        out.append(m)
    return out


def clean_melody(notes: List[Dict[str, Any]],
                 beat_times: Optional[np.ndarray] = None,
                 split: int = 56,
                 scale_pcs: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    """Reduce the register above `split` to a single monophonic line."""
    if not CLEAN_MELODY or not notes:
        return notes

    melody = [n for n in notes if int(n["pitch"]) >= split]
    rest = [n for n in notes if int(n["pitch"]) < split]
    if len(melody) < 2:
        return notes

    end = max(float(n["offset"]) for n in melody)
    n_fr = max(1, int(end * FPS) + 1)

    # Loudest activation seen per pitch, so a merged run keeps a sensible
    # dynamic instead of inheriting whichever fragment happened to be first.
    vel_of: Dict[int, int] = {}
    for n in melody:
        p = int(n["pitch"])
        vel_of[p] = max(vel_of.get(p, 0), int(n.get("velocity", 80)))

    line = _runs_to_notes(_winner_timeline(melody, scale_pcs, n_fr), vel_of)
    line = _bridge(line)
    line = _snap(line, beat_times)
    line = _bridge(line)          # snapping can align two runs that now touch
    line = [n for n in line if float(n["offset"]) - float(n["onset"]) >= MIN_DUR]

    # Snapping can drag a note's onset back across the one before it, so the
    # frame-level monophony established above does not survive on its own —
    # measured, it left a residual max of 2 sounding.  Truncating each note at
    # the next onset makes "one note at a time" exact rather than approximate.
    line.sort(key=lambda n: float(n["onset"]))
    for a, b in zip(line, line[1:]):
        if float(a["offset"]) > float(b["onset"]):
            a["offset"] = float(b["onset"])
    line = [n for n in line if float(n["offset"]) - float(n["onset"]) >= MIN_DUR * 0.5]

    out = rest + line
    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out
