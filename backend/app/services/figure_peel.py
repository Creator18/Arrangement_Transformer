"""Keep what repeats; discard what does not.

An accompaniment instrument in a pop song plays a FIGURE — a short rhythmic
and intervallic shape restated over each chord.  A transcription of that stem
contains the figure plus a great deal of debris: harmonic smear from other
instruments, bleed the separator left behind, and notes the decomposition
invented because it must explain every scrap of energy it is handed.  The
figure recurs.  The debris does not.  That difference is the filter.

So each pitched accompaniment stem is peeled: find the most prominent
repeating figure, lift it out, subtract it, then look at what surfaces in the
remainder.  Repeat until nothing left recurs often enough to be music, and
throw that residue away.

Three things this has to get right, each of which sinks it if missed.

TRANSPOSITION.  A figure repeats at a different PITCH over each chord — on the
test song the bass plays one shape over C#m7, Bsus4, A and E.  Matching on
absolute pitch would see four unrelated events with a handful of occurrences
each instead of one figure with forty, and would peel almost nothing.  So a
figure is keyed by its rhythm and its INTERVALS above the bar's lowest note,
which makes the same shape on four roots a single pattern.

STRUCTURE.  A figure belongs to a section, not to a song.  The verse figure
may be absent from the chorus.  Searching globally would both bury a chorus
figure beneath verse counts and subtract a verse figure from bars where it
never occurred, so the peel runs INSIDE each detected section with occurrence
counts relative to that section.

VOICE AND `other`.  Neither is peeled.  A sung line is through-composed — it
is supposed not to repeat, and "keep what repeats" would delete the tune.  The
`other` stem is whatever Demucs could not name, which is exactly the material
we have least reason to trust.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SLOTS = 16              # sixteenths per 4/4 bar; the figure's rhythmic grid
MIN_OCCURRENCES = 3     # fewer than this in its own section is a straggler
MAX_ROUNDS = 8          # peels per section; a guard, not a target
MAX_INTERVAL = 36       # semitones above the bar's lowest note worth keying on
PEEL_STEMS = ("bass", "guitar", "piano")


# ── structure ────────────────────────────────────────────────────────────────

def bar_chroma(H: np.ndarray, low: int, downbeats: np.ndarray,
               fps: float) -> np.ndarray:
    """(n_bars, 12) pitch-class energy per bar, L2-normalised.

    Taken from the activation matrix rather than from decoded notes, because
    the notes are the thing being judged and using them to derive the structure
    that judges them is circular.
    """
    n = len(downbeats) - 1
    C = np.zeros((n, 12))
    for i, (a, b) in enumerate(zip(downbeats[:-1], downbeats[1:])):
        fa, fb = int(a * fps), int(b * fps)
        if fb <= fa:
            continue
        rows = H[:, fa:fb].sum(axis=1)
        for r, v in enumerate(rows):
            C[i, (low + r) % 12] += v
    nrm = np.linalg.norm(C, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return C / nrm


def cycle_length(C: np.ndarray, lo: int = 2, hi: int = 16) -> int:
    """Bars per chord cycle, by self-similarity at each candidate lag.

    Derived from bar chroma rather than from chord labels: chord inference on
    the test song reported B7 for Bsus4, Amaj7 for A, and never found E at all,
    so its sequence is not stable enough to define a cycle.  Correlation needs
    only that the bars RESEMBLE each other a cycle apart, which survives every
    one of those labelling errors.
    """
    if len(C) < 2 * lo + 2:
        return 4
    best, score = 4, -np.inf
    for lag in range(lo, min(hi, len(C) // 2) + 1):
        s = float(np.mean(np.sum(C[:-lag] * C[lag:], axis=1)))
        if s > score:
            best, score = lag, s
    return best


def sections(C: np.ndarray, cycle: int, min_bars: Optional[int] = None
             ) -> List[Tuple[int, int]]:
    """Section spans as [start_bar, end_bar), by novelty in bar self-similarity.

    Keyed on TEXTURE, not on chords: a verse and a chorus routinely share a
    progression while being different sections, so a chord-change detector
    would merge them and the peel would then mix two different figures.
    """
    n = len(C)
    min_bars = max(2 * cycle, 4) if min_bars is None else min_bars
    if n < 2 * min_bars:
        return [(0, n)]
    S = C @ C.T
    k = max(2, cycle)
    nov = np.zeros(n)
    for i in range(k, n - k):
        before, after = S[i - k:i, i - k:i], S[i:i + k, i:i + k]
        cross = S[i - k:i, i:i + k]
        nov[i] = (before.mean() + after.mean()) / 2 - cross.mean()
    cuts = [0]
    if np.any(nov > 0):
        thr = float(np.percentile(nov[nov > 0], 85))
        for i in np.argsort(nov)[::-1]:
            if nov[i] < thr:
                break
            if all(abs(int(i) - c) >= min_bars for c in cuts + [n]):
                cuts.append(int(i))
    cuts = sorted(set(cuts)) + [n]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_bars] \
        or [(0, n)]


# ── figures ──────────────────────────────────────────────────────────────────

def _bar_key(notes: List[Dict], b0: float, b1: float) -> Optional[tuple]:
    """Transposition-invariant signature: (slot, interval-above-lowest) pairs."""
    if not notes:
        return None
    span = max(b1 - b0, 1e-6)
    lo = min(int(n["pitch"]) for n in notes)
    sig = set()
    for n in notes:
        s = int(np.clip((float(n["onset"]) - b0) / span * SLOTS, 0, SLOTS - 1))
        iv = int(n["pitch"]) - lo
        if iv <= MAX_INTERVAL:
            sig.add((s, iv))
    return tuple(sorted(sig)) if sig else None


SIM_THRESH = 0.35       # Jaccard overlap for two bars to count as the same figure
CORE_FRAC = 0.5         # an element must appear in this share of them to be core


def peel_section(notes: List[Dict], edges: np.ndarray,
                 min_occ: int = MIN_OCCURRENCES) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Iteratively lift the most prominent repeating figure out of one section.

    Similarity, not identity.  The first version keyed each bar by its COMPLETE
    set of (slot, interval) pairs and grouped bars whose keys matched exactly.
    That kept nothing at all — 0 notes from 1201 on bass, and the same on every
    stem — because a bar of real playing has eight to twelve notes and no two
    such bars are ever identical.  Every bar became its own unique key and
    therefore a straggler.  "The notes that repeat majorly" is not "the bars
    that repeat exactly".

    So bars are CLUSTERED by Jaccard overlap, and what gets lifted is the CORE
    of the cluster: the elements present in at least CORE_FRAC of its bars.
    Notes matching the core come out; the ornaments and one-off variations in
    those same bars stay in the pool and get their chance in the next round.
    That is what makes the peel iterative rather than a single partition.

    Returns (kept, discarded, log).  Prominence is occurrences x energy, not
    occurrences alone: a two-note fragment recurring everywhere would otherwise
    outrank the real figure simply by being short enough to match often.
    """
    pool = [dict(n) for n in notes]
    kept: List[Dict] = []
    log: List[str] = []

    for rnd in range(MAX_ROUNDS):
        by_bar: Dict[int, List[Dict]] = {}
        for n in pool:
            i = int(np.searchsorted(edges, float(n["onset"]), side="right")) - 1
            if 0 <= i < len(edges) - 1:
                by_bar.setdefault(i, []).append(n)
        sig = {}
        for i, bn in by_bar.items():
            k = _bar_key(bn, edges[i], edges[i + 1])
            if k and len(k) >= 2:
                sig[i] = set(k)
        if len(sig) < min_occ:
            break

        # seed each candidate cluster on one bar, then take every bar that
        # overlaps it enough to be a restatement of the same shape
        best = None
        for i, si in sig.items():
            members = [j for j, sj in sig.items()
                       if len(si & sj) / max(len(si | sj), 1) >= SIM_THRESH]
            if len(members) < min_occ:
                continue
            energy = sum(float(n.get("velocity", 80))
                         for j in members for n in by_bar[j])
            score = len(members) * energy
            if best is None or score > best[0]:
                best = (score, i, members)
        if best is None:
            break

        _, seed, members = best
        tally: Dict[tuple, int] = {}
        for j in members:
            for el in sig[j]:
                tally[el] = tally.get(el, 0) + 1
        core = {el for el, c in tally.items()
                if c >= max(2, int(CORE_FRAC * len(members)))}
        if len(core) < 2:
            break

        taken = []
        for j in members:
            b0, b1 = edges[j], edges[j + 1]
            span = max(b1 - b0, 1e-6)
            lo = min(int(n["pitch"]) for n in by_bar[j])
            for n in by_bar[j]:
                s = int(np.clip((float(n["onset"]) - b0) / span * SLOTS,
                                0, SLOTS - 1))
                if (s, int(n["pitch"]) - lo) in core:
                    taken.append(n)
        if not taken:
            break
        ids = {id(n) for n in taken}
        pool = [n for n in pool if id(n) not in ids]
        for n in taken:
            n["figure"] = rnd
        kept.extend(taken)
        log.append(f"r{rnd}: {len(core)}-note core x{len(members)} bars "
                   f"-> {len(taken)} notes")

    return kept, pool, log


def peel_stem(notes: List[Dict], H: np.ndarray, low: int,
              downbeats: np.ndarray, fps: float,
              min_occ: int = MIN_OCCURRENCES) -> Tuple[List[Dict], Dict[str, Any]]:
    """Full hierarchy for one stem: sections -> cycle -> figures -> residue."""
    if len(downbeats) < 4 or not notes:
        return notes, {"sections": 0, "kept": len(notes), "dropped": 0}
    C = bar_chroma(H, low, downbeats, fps)
    cyc = cycle_length(C)
    secs = sections(C, cyc)
    kept_all, drop_all, logs = [], [], []
    for (a, b) in secs:
        t0, t1 = float(downbeats[a]), float(downbeats[min(b, len(downbeats) - 1)])
        seg = [n for n in notes if t0 <= float(n["onset"]) < t1]
        if not seg:
            continue
        k, d, lg = peel_section(seg, downbeats[a:b + 1], min_occ)
        kept_all += k
        drop_all += d
        logs.append(f"  bars {a}-{b} ({t0:.0f}-{t1:.0f}s): "
                    f"kept {len(k)}, dropped {len(d)}"
                    + ("" if not lg else " | " + "; ".join(lg)))
    kept_all.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return kept_all, {"cycle_bars": cyc, "sections": len(secs),
                      "kept": len(kept_all), "dropped": len(drop_all),
                      "log": logs}


# ── A: clean first, so the figure is the majority of the bar ─────────────────

ONSET_TOL = 0.045       # a note may sit this far from a detected attack
TOP_PER_BAR = 4         # ...and only the loudest this many survive per bar


def clean_by_onsets(notes: List[Dict], onsets: Optional[np.ndarray],
                    downbeats: np.ndarray, tol: float = ONSET_TOL,
                    top_k: int = TOP_PER_BAR) -> List[Dict]:
    """Cut each bar down to a few strong, attack-aligned notes.

    The peel could not find repetition because there was none to find: bass
    bars held 6-14 notes where a real bass figure has two to four, so any bar
    signature was mostly describing smear, and two restatements of one figure
    overlapped by a median of 0.16.  Repetition cannot be used to remove the
    noise while the noise is what hides the repetition.

    Two independent cuts, because they fail differently.  The ONSET gate keeps
    only notes that coincide with a broadband attack — a real signal at 5.8 ms
    that the decomposition never saw, so it is evidence rather than a rephrasing
    of what we already have.  The per-bar CAP then keeps the loudest few, which
    is the "produce the most prominent sound" half of the idea.  The gate alone
    leaves sustained smear that happens to start near an attack; the cap alone
    keeps the loudest smear in a bar that has no real notes at all.
    """
    out = notes
    if onsets is not None and len(onsets):
        ons = np.sort(np.asarray(onsets, float))
        keep = []
        for n in out:
            i = int(np.searchsorted(ons, float(n["onset"])))
            near = min((abs(ons[j] - float(n["onset"]))
                        for j in (i - 1, i) if 0 <= j < len(ons)), default=9e9)
            if near <= tol:
                keep.append(n)
        out = keep
    if top_k and len(downbeats) > 1:
        by_bar: Dict[int, List[Dict]] = {}
        for n in out:
            i = int(np.searchsorted(downbeats, float(n["onset"]), "right")) - 1
            by_bar.setdefault(i, []).append(n)
        out = []
        for bn in by_bar.values():
            bn.sort(key=lambda n: -float(n.get("velocity", 80)))
            out.extend(bn[:top_k])
    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out


# ── B: harmony, which needs no repetition at all ─────────────────────────────

_TEMPLATES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dom7": (0, 4, 7, 10),
    "min7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11), "sus4": (0, 5, 7),
    "sus2": (0, 2, 7), "dim": (0, 3, 6),
}


def bar_chords(notes: List[Dict], downbeats: np.ndarray,
               scale_pcs: Optional[set] = None) -> Dict[int, set]:
    """Best-fitting chord for each bar, as a set of pitch classes.

    Inferred from the bar's OWN weighted pitch content rather than read from
    chord_map, which on this song reported B7 for Bsus4, Amaj7 for A, and never
    found E at all.  Weighted by duration x velocity so a long loud root counts
    for more than a passing sixteenth.
    """
    out: Dict[int, set] = {}
    by_bar: Dict[int, List[Dict]] = {}
    for n in notes:
        i = int(np.searchsorted(downbeats, float(n["onset"]), "right")) - 1
        if 0 <= i < len(downbeats) - 1:
            by_bar.setdefault(i, []).append(n)
    for i, bn in by_bar.items():
        w = np.zeros(12)
        for n in bn:
            dur = max(float(n["offset"]) - float(n["onset"]), 0.05)
            w[int(n["pitch"]) % 12] += dur * float(n.get("velocity", 80))
        if w.sum() <= 0:
            continue
        w /= w.sum()
        best, score = None, -9e9
        for name, ivs in _TEMPLATES.items():
            for root in range(12):
                pcs = {(root + iv) % 12 for iv in ivs}
                # coverage minus leakage, so a template is not rewarded merely
                # for being large enough to cover everything
                s = sum(w[p] for p in pcs) - 0.35 * sum(w[p] for p in range(12)
                                                        if p not in pcs)
                if scale_pcs:
                    s -= 0.25 * len(pcs - set(scale_pcs)) / len(pcs)
                if s > score:
                    best, score = pcs, s
        if best:
            out[i] = best
    return out


def harmony_filter(notes: List[Dict], downbeats: np.ndarray,
                   scale_pcs: Optional[set] = None
                   ) -> Tuple[List[Dict], Dict[str, Any]]:
    """Keep notes that are chord tones of their own bar.

    The alternative to peeling: a note earns its place by being harmonically
    right where it sits, not by recurring.  That matters because the measured
    bar-to-bar similarity of this transcription tops out around 0.2, so
    repetition is not detectable — whereas harmony is a per-bar judgement and
    needs no structure above the bar at all.
    """
    ch = bar_chords(notes, downbeats, scale_pcs)
    kept = []
    for n in notes:
        i = int(np.searchsorted(downbeats, float(n["onset"]), "right")) - 1
        pcs = ch.get(i)
        if pcs is None or (int(n["pitch"]) % 12) in pcs:
            kept.append(n)
    return kept, {"bars_with_chord": len(ch),
                  "kept": len(kept), "dropped": len(notes) - len(kept)}


# ── recombination ────────────────────────────────────────────────────────────

def merge_overlaps(notes: List[Dict], tol: float = 0.05) -> List[Dict]:
    """Fuse same-pitch notes that overlap in time, keeping the loudest.

    Stems are decomposed independently, so one sounding note routinely appears
    in two of them — a guitar note bleeding into `piano`, say.  Summing the note
    lists would strike the same key twice a few milliseconds apart, which is the
    double-press artefact rather than a thicker texture.
    """
    out: List[Dict] = []
    by_pitch: Dict[int, List[Dict]] = {}
    for n in notes:
        by_pitch.setdefault(int(n["pitch"]), []).append(n)
    for p, group in by_pitch.items():
        group.sort(key=lambda n: float(n["onset"]))
        cur = dict(group[0])
        for n in group[1:]:
            if float(n["onset"]) <= float(cur["offset"]) - tol:
                cur["offset"] = max(float(cur["offset"]), float(n["offset"]))
                if float(n.get("velocity", 0)) > float(cur.get("velocity", 0)):
                    cur["velocity"] = n["velocity"]
            else:
                out.append(cur)
                cur = dict(n)
        out.append(cur)
    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out
