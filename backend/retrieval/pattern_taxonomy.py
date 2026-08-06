"""
pattern_taxonomy.py — classify a figuration cycle into a named
accompaniment type, and hold the register/clarity constraints that
arranging practice imposes on accompaniment.

Sources (researched 2026-08, not recalled from training):
  - Baylor "Piano Basics" open textbook, accompaniment-styles chapter
    (broken chord / Alberti / waltz bass definitions)
  - Noble Academy, "Arrangement techniques for left-hand piano
    accompaniment" (11-pattern taxonomy with style and usage guidance)
  - Wikipedia, "Alberti bass" (lowest-highest-middle-highest ordering)
  - tonebase / Kevin Class / NATS Journal of Singing on orchestral
    reduction (keep melody+harmony+bass, omit what one player cannot
    cover, adapt figuration rather than transcribe literally)
  - klang.io and Pianote arranging guides (density as tension control,
    break block chords up to avoid mud)

Why this exists
---------------
Before this, corpus units were tagged only by composer, which is a proxy
for style at best.  A named pattern type gives us three things:
  1. a retrieval filter that means something musically ("give me stride")
  2. a human-readable provenance string for the review UI
  3. a way to check the corpus is balanced across textures rather than
     being 90% one figure

Classification is heuristic and ordered most-specific-first.  It is meant
to be *useful*, not authoritative — 'mixed' is a legitimate answer and
roughly a fifth of real accompaniment writing does not fit a named box.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ── register / clarity constraints (for the selection cost function) ──────────
#
# Every source consulted independently states the same principle in different
# words: the accompaniment must not swamp the melody, and closely-spaced low
# chords turn to mud on a piano.  These constants encode that so the
# retrieval selector can score candidates on it instead of relying on the
# music-theory pass to clean up afterwards.

# Minimum gap between the top of the accompaniment and the melody's lowest
# note.  A sixth is the figure arranging guides give for "the melody has
# space to be heard".
MIN_MELODY_CLEARANCE_ST = 9

# Below this pitch, chords of 3+ notes within a narrow span read as mud.
MUD_REGISTER_MAX_PITCH = 48          # C3
MUD_MIN_SPACING_ST     = 5           # a fourth or wider between voices down there

# A reduction is allowed to move material up an octave to escape mud —
# explicitly endorsed by the orchestral-reduction sources.
MUD_ESCAPE_OCTAVE_SHIFT = 12


def mud_penalty(notes: List[Dict]) -> float:
    """
    0.0 = clean, 1.0 = maximally muddy.  Counts simultaneous groups that sit
    below MUD_REGISTER_MAX_PITCH with 3+ notes spaced tighter than a fourth.
    """
    if not notes:
        return 0.0
    by_onset: Dict[float, List[int]] = {}
    for n in notes:
        by_onset.setdefault(round(float(n["beat"]), 2), []).append(int(n["pitch"]))
    bad = 0
    for pitches in by_onset.values():
        if len(pitches) < 3:
            continue
        ps = sorted(pitches)
        if ps[-1] > MUD_REGISTER_MAX_PITCH:
            continue
        gaps = np.diff(ps)
        if gaps.size and float(np.min(gaps)) < MUD_MIN_SPACING_ST:
            bad += 1
    return bad / max(1, len(by_onset))


def melody_clearance(notes: List[Dict], melody_lowest_pitch: int) -> float:
    """
    Semitones between the accompaniment's highest note and the melody's
    lowest.  Negative means the accompaniment crosses into the melody's
    register — the thing every arranging source warns about.
    """
    if not notes:
        return float(MIN_MELODY_CLEARANCE_ST)
    top = max(int(n["pitch"]) for n in notes)
    return float(melody_lowest_pitch - top)


# ── pattern taxonomy ─────────────────────────────────────────────────────────

PATTERN_TYPES = (
    "pedal_point",       # sustained bass under changing harmony
    "block_chords",      # chords struck together, one or few per bar
    "stride",            # low bass note alternating with higher chord (boom-chick)
    "broken_octaves",    # alternating octave doublings
    "alberti",           # single line, low-high-mid-high oscillation
    "walking_bass",      # single line, ~1/beat, stepwise through chord tones
    "arpeggiated",       # single line spanning a wide range, directional runs
    "ostinato",          # short repeating figure, few distinct pitches
    "mixed",             # legitimate fallback
)

# Style affinity per type, from the researched taxonomy.  Used for optional
# genre-aware filtering later ("pop arrangement" should not retrieve Alberti).
STYLE_AFFINITY: Dict[str, Tuple[str, ...]] = {
    "pedal_point":    ("classical", "romantic", "cinematic"),
    "block_chords":   ("pop", "hymn", "gospel"),
    "stride":         ("jazz", "ragtime", "blues", "pop"),
    "broken_octaves": ("classical", "romantic", "dramatic"),
    "alberti":        ("classical",),
    "walking_bass":   ("jazz", "pop", "blues"),
    "arpeggiated":    ("romantic", "pop", "cinematic"),
    "ostinato":       ("contemporary", "minimalist", "pop"),
    "mixed":          (),
}


def _groups(notes: List[Dict], window: float = 0.18) -> List[List[Dict]]:
    """Cluster notes into simultaneous-onset groups (same window as featurizer)."""
    ns = sorted(notes, key=lambda n: float(n["beat"]))
    out: List[List[Dict]] = []
    for n in ns:
        if out and abs(float(n["beat"]) - float(out[-1][0]["beat"])) < window:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def classify_pattern_type(unit: Dict) -> str:
    """
    Heuristic classification of one figuration cycle.  Checked
    most-specific-first; 'mixed' when nothing fits.
    """
    notes = unit.get("notes", [])
    if len(notes) < 2:
        return "mixed"
    bpb = int(unit.get("beats_per_bar", 4))
    cycle_beats = max(0.5, bpb * float(unit.get("cycle_bars", 1.0)))

    grps = _groups(notes)
    n_g = len(grps)
    sizes = np.array([len(g) for g in grps], dtype=float)
    mean_size = float(sizes.mean())
    lows = np.array([min(int(x["pitch"]) for x in g) for g in grps], dtype=float)
    tops = np.array([max(int(x["pitch"]) for x in g) for g in grps], dtype=float)
    durs = np.array([float(n.get("dur", 0.25)) for n in notes], dtype=float)
    pitches = np.array([int(n["pitch"]) for n in notes], dtype=float)
    span = float(pitches.max() - pitches.min())
    groups_per_beat = n_g / cycle_beats

    # 1. Pedal point — very few attacks, held long
    if n_g <= 2 and float(durs.max()) >= 0.7 * cycle_beats:
        return "pedal_point"

    # 2. Block chords — thick simultaneities, sparse in time
    if mean_size >= 2.5 and groups_per_beat <= 1.2:
        return "block_chords"

    # 3. Stride / boom-chick — single low note alternating with higher chord
    if n_g >= 2:
        singles = sizes <= 1.5
        chordal = sizes >= 2
        if singles.any() and chordal.any():
            alternating = float(np.mean(singles[:-1] != singles[1:])) if n_g > 1 else 0.0
            low_of_singles  = float(lows[singles].mean())
            low_of_chordal  = float(lows[chordal].mean())
            if alternating >= 0.6 and (low_of_chordal - low_of_singles) >= 7:
                return "stride"

    # 4. Broken octaves — mostly single notes, intervals of exactly an octave
    if mean_size <= 1.5 and n_g >= 3:
        steps = np.abs(np.diff(lows))
        if steps.size and float(np.mean(np.isclose(steps, 12, atol=0.6))) >= 0.5:
            return "broken_octaves"

    # 5. Alberti — single line, >=3 attacks/bar, oscillating, narrow span,
    #    and the cycle's lowest note comes first (the defining shape:
    #    lowest-highest-middle-highest)
    if mean_size <= 1.4 and groups_per_beat >= 0.9 and span <= 16:
        d = np.diff(lows)
        if d.size >= 2:
            dir_changes = float(np.mean(np.sign(d[:-1]) != np.sign(d[1:])))
            if dir_changes >= 0.45 and float(lows.argmin()) <= 0.5:
                return "alberti"

    # 6. Walking bass — single notes about one per beat, mostly stepwise,
    #    progressing rather than oscillating
    if mean_size <= 1.4 and 0.6 <= groups_per_beat <= 1.6:
        d = np.diff(lows)
        if d.size:
            stepwise = float(np.mean(np.abs(d) <= 4))
            dir_changes = (float(np.mean(np.sign(d[:-1]) != np.sign(d[1:])))
                           if d.size >= 2 else 0.0)
            if stepwise >= 0.6 and dir_changes < 0.45:
                return "walking_bass"

    # 7. Arpeggiated — single line, wide span, directional
    if mean_size <= 1.6 and span > 16 and groups_per_beat >= 0.9:
        return "arpeggiated"

    # 8. Ostinato — short cycle, few distinct pitches, repeating
    if len(set(pitches.tolist())) <= 4 and n_g >= 3:
        return "ostinato"

    return "mixed"


def annotate_units(units: List[Dict]) -> List[Dict]:
    """Add `pattern_type` (and its style affinities) to each unit in place."""
    for u in units:
        t = classify_pattern_type(u)
        u["pattern_type"] = t
        u["style_affinity"] = list(STYLE_AFFINITY.get(t, ()))
    return units
