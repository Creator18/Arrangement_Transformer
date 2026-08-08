"""
Structural metrics for the 88 -> 61 arrangement half of the pipeline.

Why these are reference-free
----------------------------
The transcription harness compares against ground-truth MIDI, because there
is exactly one right answer.  Arrangement has no such target: a good
5-octave arrangement legitimately differs from the 7-octave source, so
"distance from the source" is not a quality signal — a perfect arrangement
would score badly on it.

So nothing here compares against a reference arrangement.  Two kinds of
metric instead:

  ABSOLUTE — properties the output must have no matter how it was made.
    A note outside the keyboard is wrong.  Two note-ons for one key at the
    same instant is wrong.  A chord no hand can reach is wrong.  These are
    facts about a 61-key piano, not opinions about arranging.

  RELATIVE — what changed versus the source, measured only on the two
    voices a listener actually tracks: the top line and the bass line.
    Deliberately NOT a note-by-note diff, so an arm that legitimately
    rewrites inner figuration (retrieval) is not punished for doing its job.

Every metric is "lower is better" and 0 is the ideal, so arms can be ranked
without knowing which metric is which.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence

# One hand: five fingers, and a span a normal hand can actually cover.
MAX_HAND_SPAN = 14      # semitones, ~an octave plus a second
MAX_PER_HAND = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def _sweep_simultaneous(notes: Sequence[Dict]):
    """
    Yield (time, [notes sounding at that time]) at every onset, via a sweep
    line.  The naive "filter all notes at each onset" is O(n^2) and this
    corpus has pieces with thousands of notes, so it matters.
    """
    if not notes:
        return
    ns = sorted(notes, key=lambda n: float(n["onset"]))
    active: List[Dict] = []
    for n in ns:
        t = float(n["onset"])
        active = [a for a in active if float(a.get("offset", a["onset"])) > t + 1e-9]
        active.append(n)
        yield t, list(active)


def _voice(notes: Sequence[Dict], lowest: bool) -> List[tuple]:
    """The bass line (lowest) or top line (highest) as (onset, pitch)."""
    by: Dict[float, List[int]] = {}
    for n in notes:
        by.setdefault(round(float(n["onset"]), 3), []).append(int(n["pitch"]))
    pick = min if lowest else max
    return [(t, pick(v)) for t, v in sorted(by.items())]


# ── absolute metrics ──────────────────────────────────────────────────────────

def out_of_range(notes: Sequence[Dict], lo: int, hi: int) -> int:
    """Notes the target keyboard physically does not have.  Must be 0."""
    return sum(1 for n in notes if not (lo <= int(n["pitch"]) <= hi))


def doubled_keys(notes: Sequence[Dict]) -> int:
    """
    Same pitch sounding twice at once — one physical key asked to be
    pressed while it is already down.

    This is the metric that matters most for us right now.  In the sampler
    it double-triggers, and in MIDI export it is genuinely ambiguous: the
    first note-off silences the key while the second note is still notated
    as held.  That is the mechanism behind "repeated notes lose their
    sustain", so this number has to be 0.
    """
    by: Dict[int, List[tuple]] = {}
    for n in notes:
        by.setdefault(int(n["pitch"]), []).append(
            (float(n["onset"]), float(n.get("offset", n["onset"]))))
    bad = 0
    for spans in by.values():
        spans.sort()
        for i in range(1, len(spans)):
            if spans[i][0] < spans[i - 1][1] - 1e-6:
                bad += 1
    return bad


def unplayable_chords(notes: Sequence[Dict]) -> int:
    """
    Simultaneities no two hands can take.  Tries every split of the
    sounding pitches into a low group and a high group — contiguous split
    is the realistic case, since hands do not interleave — and asks whether
    some split leaves both hands within 5 fingers and MAX_HAND_SPAN.
    """
    bad = 0
    for _, act in _sweep_simultaneous(notes):
        ps = sorted(int(a["pitch"]) for a in act)
        if len(ps) <= 1:
            continue
        if len(ps) > 2 * MAX_PER_HAND:
            bad += 1
            continue
        ok = False
        for k in range(len(ps) + 1):
            lh, rh = ps[:k], ps[k:]
            if len(lh) > MAX_PER_HAND or len(rh) > MAX_PER_HAND:
                continue
            if lh and lh[-1] - lh[0] > MAX_HAND_SPAN:
                continue
            if rh and rh[-1] - rh[0] > MAX_HAND_SPAN:
                continue
            ok = True
            break
        if not ok:
            bad += 1
    return bad


# ── relative metrics ──────────────────────────────────────────────────────────

def contour_breaks(src: Sequence[Dict], arr: Sequence[Dict], lowest: bool) -> Optional[float]:
    """
    Percentage of steps in a voice that were a step or small leap (<= 12)
    in the source and became a leap (> 12) after arranging — the signature
    of a line wrapping around a range boundary.

    Returns None when the two voices have different lengths, rather than
    guessing an alignment.
    """
    a, b = _voice(src, lowest), _voice(arr, lowest)
    if len(a) != len(b) or len(a) < 2:
        return None
    broke = sum(
        1 for i in range(1, len(a))
        if abs(a[i][1] - a[i - 1][1]) <= 12 and abs(b[i][1] - b[i - 1][1]) > 12
    )
    return 100.0 * broke / (len(a) - 1)


def bass_role_lost(src: Sequence[Dict], arr: Sequence[Dict]) -> Optional[float]:
    """
    Percentage of onsets where the PITCH CLASS of the lowest sounding note
    changed.  The bass carries the harmony's foundation: if the lowest note
    was a C and is now an E, the chord has been inverted, which is heard as
    a different harmony even though every pitch class is still present.

    This is what folding a low bass note up an octave actually costs, and
    it is invisible to a plain note-count diff.
    """
    a, b = _voice(src, True), _voice(arr, True)
    if len(a) != len(b) or not a:
        return None
    changed = sum(1 for i in range(len(a)) if a[i][1] % 12 != b[i][1] % 12)
    return 100.0 * changed / len(a)


def crossings_introduced(src: Sequence[Dict], arr: Sequence[Dict]) -> Optional[int]:
    """
    Pairs of simultaneous notes whose pitch ORDER flipped — a voice that
    was below another is now above it.  Index-aligned, so only valid for
    arms that transform notes in place (folding).  Returns None otherwise.
    """
    if len(src) != len(arr):
        return None
    groups: Dict[float, List[int]] = {}
    for i, n in enumerate(src):
        groups.setdefault(round(float(n["onset"]), 3), []).append(i)
    flips = 0
    for idxs in groups.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                x, y = idxs[i], idxs[j]
                before = int(src[x]["pitch"]) - int(src[y]["pitch"])
                after = int(arr[x]["pitch"]) - int(arr[y]["pitch"])
                if before * after < 0:
                    flips += 1
    return flips


def evaluate(src: Sequence[Dict], arr: Sequence[Dict], lo: int, hi: int) -> Dict:
    """All metrics for one piece.  Lower is better throughout."""
    return {
        "out_of_range":   out_of_range(arr, lo, hi),
        "doubled_keys":   doubled_keys(arr),
        "unplayable":     unplayable_chords(arr),
        "crossings":      crossings_introduced(src, arr),
        "top_breaks":     contour_breaks(src, arr, lowest=False),
        "bass_breaks":    contour_breaks(src, arr, lowest=True),
        "bass_role_lost": bass_role_lost(src, arr),
        "n_src":          len(src),
        "n_arr":          len(arr),
    }
