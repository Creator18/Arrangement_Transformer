"""
Rebuild the accompaniment as a repeating figure instead of playing back detections.

Compared side by side against a human Synthesia arrangement of the same track,
ours differed in three ways: onsets scattered off the grid, roughly ten times
the note count, and — the one that cannot be filtered away — no REPEATING
FIGURE.  The human left hand is one pattern, restated bar after bar.  Ours was
420 independent detections, because nothing in the pipeline knew that bar 5
ought to resemble bar 4.

The argument for generating rather than reporting
-------------------------------------------------
Note-level precision on this path is 0.31-0.47, so roughly half the individual
detections are wrong, and no threshold fixes that.  But the two things needed
to WRITE an accompaniment — which chord is sounding, and where the beats fall
— survive the same noise easily, because both are aggregates over a whole bar.
That is the measured result from the retrieval work too: bar-level chroma
retrieved its own unit at rank 1 with recall@1 60.8%, while onset timing from
audio scored at pure chance.

So this uses the activations to infer harmony and rhythm per bar, then writes
an idiomatic figure from them.  Individual wrong notes stop mattering, because
no individual detection is played.

What is deliberately NOT touched
--------------------------------
The melody.  Replacing the top line with generated material would substitute a
different tune for the song's own, which is the one error a listener cannot
forgive — and the top voice is the part of the detection that is most worth
keeping.  Only the accompaniment underneath is rebuilt.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

ARPEGGIATE = True
MELODY_MIN = 56           # G#3 — matches _BASS_SPLIT_PITCH in transcription
SUBDIV = 4                # sixteenths at 4/4
ACC_LOW, ACC_HIGH = 36, 67   # where the left hand actually lives

# Chord templates as pitch-class offsets from the root.  Kept small on purpose:
# more templates fit the noise better without describing the music better, and
# a wrong-but-plausible triad is far less damaging than a confidently wrong
# extended chord.
TEMPLATES = {
    "maj":  (0, 4, 7),
    "min":  (0, 3, 7),
    "dom7": (0, 4, 7, 10),
    "min7": (0, 3, 7, 10),
    "dim":  (0, 3, 6),
}

# Alberti order — root, fifth, third, fifth.  This is the figure that shows up
# in the reference arrangement and in most of the corpus: it states the harmony
# without competing with the melody, and it repeats, which is the whole point.
ALBERTI = (0, 2, 1, 2)


def _bar_edges(beat_times: np.ndarray, downbeat_times: Optional[np.ndarray],
               beats_per_bar: int) -> List[float]:
    """Bar boundaries, from downbeats when madmom found them."""
    if downbeat_times is not None and len(downbeat_times) >= 2:
        edges = list(map(float, downbeat_times))
    else:
        bt = list(map(float, beat_times))
        edges = bt[::max(1, beats_per_bar)]
    if len(edges) >= 2:
        edges.append(edges[-1] + (edges[-1] - edges[-2]))
    return edges


_CHORD_PRIORS: Optional[tuple] = None
_CHORD_PRIOR_PATH = pathlib.Path(__file__).with_name("_chord_priors.npz")
# Scale on the log-priors.  Swept on three real uploads against the human
# distribution (dim 1.3%, maj 59.5%, min 33.4%, sevenths 5.9%), crossed with
# whether harmony is read from the accompaniment alone or the whole bar:
#
#   source        w      dim%   maj%   min%  7ths%
#   accomp only  0.00    30.7   39.2    5.2   24.8
#   accomp only  0.04     0.0   87.6    3.9    8.5
#   FULL BAR     0.00     3.9   44.4   17.6   34.0
#   FULL BAR     0.04     0.0   72.5   17.6    9.8   <- ships
#   FULL BAR     0.08     0.0   83.0   15.0    2.0
#   human                 1.3   59.5   33.4    5.9
#
# Reading the full bar does most of the work on its own — diminished falls from
# 30.7% to 3.9% with no prior at all, because the missing third was the whole
# problem.  The prior then fixes the seventh over-firing (34.0% -> 9.8%), which
# is the same cheap-template effect: dom7 and min7 have four tones and fit
# noise more easily than triads.  Above 0.04 it starts flattening everything
# into major.
#   weight   root   quality   both   minor recall     (maj/min prior TIED)
#     0.00  73.8%     68.5%  62.0%          86.8%
#     0.08  74.6%     81.8%  69.7%          84.2%   <- ships, start of plateau
#     0.15  74.6%     82.0%  69.9%          84.2%
#
# Scored against POP909's own chord labels rather than against a distribution,
# which is what made the earlier 0.04 look reasonable: matching how often
# humans play each quality is not the same as naming the right chord.
#
# 0.08 is the smallest weight that reaches the plateau.  Taking the least prior
# that achieves the result leaves the most work to the evidence.
CHORD_PRIOR_W = 0.08


def chord_priors() -> Optional[tuple]:
    """(quality->logp, degree logp array) measured from POP909, or None.

    Why this has to exist: the scoring below compared 12 roots x 5 qualities
    with a FLAT prior, and the five qualities are not remotely equally likely.
    Measured over 119,891 human chord labels, duration-weighted:

        maj 0.5946   min 0.3340   min7 0.0415   dom7 0.0171   dim 0.0128

    A diminished triad is 46x rarer than a major one, and it is also the
    cheapest template to fit — (0,3,6) is a symmetric stack of minor thirds
    narrow enough to partially explain almost any pitch-class set.  Flat prior
    plus cheap template is why 31% of bars came out diminished.

    The degree prior is what keeps this from being a diatonic-only rule.  bVII
    (7.4%), bVI (6.2%) and bIII (5.4%) are all non-diatonic in major and all
    common — those are the borrowed chords that sound right.  bII, bV and VII
    are each around 1% and are the ones that do not.  Diatonic degrees carry
    78.6% of chord time, so the prior discourages chromaticism without
    forbidding it.

    CHORD_PRIOR_W is small on purpose.  Chroma cosines differ between
    candidates by perhaps 0.05-0.2, while the log-priors span ~3.8, so at
    weight 1.0 the prior would simply pick the most common chord and ignore the
    audio.  It has to be able to break a near-tie, not overrule real evidence.
    """
    global _CHORD_PRIORS
    if _CHORD_PRIORS is not None:
        return _CHORD_PRIORS
    if not _CHORD_PRIOR_PATH.exists():
        return None
    z = np.load(_CHORD_PRIOR_PATH, allow_pickle=False)
    names = [str(s) for s in z["qualities"]]
    qp = dict(zip(names, [float(v) for v in z["quality_p"]]))

    # Tie maj and min to the same prior, deliberately.
    #
    # Measured against POP909's own chord labels, the prior does two jobs that
    # pull opposite ways.  Suppressing dim (46x rarer) and the four-note
    # sevenths (cheap templates that fit noise) raised quality accuracy from
    # 68.5% to 81.3%.  But maj is only 1.8x more common than min, and letting
    # that break the tie cost minor recall 86.8% -> 77.4%.
    #
    # maj vs min is decided by ONE semitone — the third — and it is the most
    # perceptually loaded distinction in the whole vocabulary: calling a minor
    # chord major changes the emotional character of the bar, while calling a
    # maj7 a maj barely registers.  So that comparison is left to the evidence
    # and the prior keeps only the job it is good at.
    if "maj" in qp and "min" in qp:
        share = (qp["maj"] + qp["min"]) / 2.0
        qp["maj"] = qp["min"] = share

    qlog = {n: float(np.log(p + 1e-9)) for n, p in qp.items()}
    dlog = np.log(z["degree_p"] + 1e-9)
    _CHORD_PRIORS = (qlog, dlog)
    return _CHORD_PRIORS


def infer_chord(notes: List[Dict[str, Any]],
                key_root: Optional[int] = None) -> Optional[tuple]:
    """Best-fitting triad/seventh for one bar, from a weighted chroma.

    Weighted by velocity AND duration: on this path velocity carries activation
    strength, so a loud sustained note is better evidence of the harmony than a
    brief weak one, and both terms are available for free.

    `key_root` enables the POP909 degree prior.  Without it only the quality
    prior applies, which still removes most of the diminished over-firing.
    """
    if not notes:
        return None
    chroma = np.zeros(12)
    for n in notes:
        dur = max(1e-3, float(n["offset"]) - float(n["onset"]))
        chroma[int(n["pitch"]) % 12] += (int(n.get("velocity", 80)) / 127.0) * dur
    if chroma.sum() <= 0:
        return None
    chroma = chroma / np.linalg.norm(chroma)

    P = chord_priors()
    best, best_score = None, -1e9
    for root in range(12):
        for name, ivs in TEMPLATES.items():
            t = np.zeros(12)
            for i, iv in enumerate(ivs):
                # The root and fifth anchor the identity; upper extensions are
                # weaker evidence and are weighted accordingly.
                t[(root + iv) % 12] = 1.0 if i == 0 else (0.9 if iv == 7 else 0.7)
            t /= np.linalg.norm(t)
            s = float(chroma @ t)
            if P is not None:
                qlog, dlog = P
                s += CHORD_PRIOR_W * qlog.get(name, -4.0)
                if key_root is not None:
                    s += CHORD_PRIOR_W * float(dlog[(root - int(key_root)) % 12])
            if s > best_score:
                best, best_score = (root, name, tuple((root + iv) % 12 for iv in ivs)), s
    return best


def bar_rhythm_vector(notes: List[Dict[str, Any]], start: float, end: float,
                      n_slots: int) -> np.ndarray:
    """Onset weight per subdivision slot for one bar."""
    hits = np.zeros(n_slots)
    if end <= start:
        return hits
    for n in notes:
        pos = (float(n["onset"]) - start) / (end - start)
        if 0.0 <= pos < 1.0:
            hits[int(pos * n_slots)] += int(n.get("velocity", 80))
    return hits


N_RHYTHMS = 3             # how many distinct figures the piece may use

_PATTERNS: Optional[tuple] = None
_PATTERN_PATH = pathlib.Path(__file__).with_name("_accomp_patterns.npz")
PRIOR_WEIGHT = 0.25       # how much a pattern's human frequency counts
                          # against how well it fits this bar's onsets
MAX_ATTACKS = 12          # densest figure allowed; see human_patterns()


def human_patterns() -> Optional[tuple]:
    """(masks, weights) measured from POP909's PIANO track, or None."""
    global _PATTERNS
    if _PATTERNS is not None:
        return _PATTERNS
    if not _PATTERN_PATH.exists():
        return None
    z = np.load(_PATTERN_PATH)
    masks = z["masks"].astype(np.int64)
    weights = z["weights"].astype(float)
    slots = int(z["slots"])

    # Drop the very dense figures, all-sixteen included.  They are real — a
    # wall of sixteenths is 5.8% of human bars — but they do not survive this
    # pipeline: cap_polyphony and the scale drop remove notes AFTER the figure
    # is written, and trimming a 16-attack pattern yields an arbitrary subset,
    # which is precisely the irregular clutter this retrieval exists to remove.
    # Sparser patterns degrade into sparser patterns; dense ones degrade into
    # noise.  Human median is 8 attacks, so 12 keeps everything typical.
    keep = np.array([bin(int(m)).count("1") <= MAX_ATTACKS for m in masks])
    if keep.any():
        masks, weights = masks[keep], weights[keep]
        weights = weights / weights.sum()
    vecs = np.array([[1.0 if (m >> i) & 1 else 0.0 for i in range(slots)]
                     for m in masks])
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    _PATTERNS = (vecs / norms[:, None], weights, slots)
    return _PATTERNS


def retrieve_rhythms(vectors: List[np.ndarray],
                     k: int = N_RHYTHMS) -> Optional[List[List[int]]]:
    """Pick each bar's figure from rhythms humans actually play.

    Clustering the bar's own onsets (canonical_rhythms, below) produced figures
    as irregular as the detections: a shipped pattern was (0,1,2,4,6,9,14),
    which hits the odd sixteenths 1 and 9 while skipping 8 and 12.  Every attack
    was on the grid and the result still read as clutter, because being on the
    grid and being a rhythm are different things.

    Measured over 72,861 POP909 accompaniment bars, human figures are
    overwhelmingly regular and downbeat-anchored — straight eighths alone are
    17.8% of all bars.  So the vocabulary is retrieved from them rather than
    invented here.  This is also strict quantisation by construction: a pattern
    IS a set of sixteenth slots, so there is nowhere off-grid to land.

    Two-stage, because idiomatic-per-bar is not the same as coherent-per-song:
    each bar retrieves its best human pattern, then the piece is restricted to
    the `k` that won most often and every bar is reassigned among those.  That
    keeps the repetition that fixed the bass in the first place.
    """
    P = human_patterns()
    if P is None:
        return None
    vecs, weights, slots = P
    log_prior = np.log(weights + 1e-9)

    def best_of(v, allowed):
        n = np.linalg.norm(v)
        if n <= 0:
            return None
        s = vecs[allowed] @ (v / n) + PRIOR_WEIGHT * log_prior[allowed]
        return allowed[int(np.argmax(s))]

    all_idx = np.arange(len(vecs))
    first = [best_of(v, all_idx) if v.sum() > 0 else None for v in vectors]
    chosen = [i for i in first if i is not None]
    if not chosen:
        return None

    counts = np.bincount(chosen, minlength=len(vecs))
    allowed = np.argsort(-counts)[:max(1, k)]
    allowed = allowed[counts[allowed] > 0]

    out = []
    for v in vectors:
        if v.sum() <= 0:
            out.append([])
            continue
        i = best_of(v, allowed)
        out.append([s for s in range(slots) if (int(round(vecs[i][s] * 1e6)) > 0)]
                   if i is not None else [])
    return out


def canonical_rhythms(vectors: List[np.ndarray],
                      k: int = N_RHYTHMS) -> List[List[int]]:
    """Reduce per-bar rhythms to a handful the piece actually reuses.

    Inferring each bar's rhythm independently sounds principled and is not:
    measured, it produced 34 distinct figures across 35 bars, because the
    detections vary bar to bar and every wobble became a new rhythm.  That is
    relaying noise as musical variation, and it is exactly what separated our
    output from the human arrangement in the first place.

    Clustering the bars and giving every member its centroid's pattern forces
    the repetition while keeping the vocabulary the SONG's own — a piece in
    straight eighths gets straight eighths, a syncopated one keeps its
    syncopation.  Only the bar-to-bar jitter is removed.
    """
    live = [v for v in vectors if v.sum() > 0]
    if not live:
        return [[] for _ in vectors]
    k = max(1, min(k, len(live)))

    # Normalise per bar so clustering keys on WHERE the attacks fall, not on
    # how loud the bar happens to be.
    norm = [v / (np.linalg.norm(v) or 1.0) for v in live]
    if k == 1:
        centroids = [np.mean(norm, axis=0)]
        labels_live = [0] * len(norm)
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(np.stack(norm))
        centroids = [km.cluster_centers_[i] for i in range(k)]
        labels_live = list(km.labels_)

    slot_sets = []
    for c in centroids:
        pos = c[c > 0]
        thr = pos.mean() * 0.8 if pos.size else 0.0
        s = [i for i in range(len(c)) if c[i] >= thr and c[i] > 0]
        slot_sets.append(s or [0])

    out, j = [], 0
    for v in vectors:
        if v.sum() <= 0:
            out.append([])
        else:
            out.append(slot_sets[labels_live[j]])
            j += 1
    return out


def build_figure(chord_pcs: tuple, slots: List[int], start: float, end: float,
                 centre: int, n_slots: int,
                 lo: int = ACC_LOW, hi: int = ACC_HIGH,
                 scale_pcs=None) -> List[Dict[str, Any]]:
    """Write an Alberti figure for one bar over the given slots.

    `scale_pcs` filters the chord's tones to those in key BEFORE the figure is
    written.  Dropping out-of-key notes afterwards would leave gaps in a
    pattern whose whole value is that it repeats — better never to place them.
    The chord's root is kept regardless, because a figure with no root stops
    stating the harmony at all.
    """
    if not slots or end <= start:
        return []
    if scale_pcs:
        pcs = set(int(p) % 12 for p in scale_pcs)
        keep = tuple(pc for pc in chord_pcs if pc % 12 in pcs) or (chord_pcs[0],)
        chord_pcs = (chord_pcs[0],) + tuple(p for p in keep if p != chord_pcs[0])
    # Anchor the root in the octave nearest where the accompaniment was already
    # sitting, so the rebuilt part occupies the same register as the original
    # rather than jumping somewhere arbitrary.
    root_pc = chord_pcs[0]
    root = root_pc + 12 * int(round((centre - root_pc) / 12))
    root = int(np.clip(root, lo, hi - 12))

    tones = []
    for pc in chord_pcs:
        p = root + ((pc - root_pc) % 12)
        tones.append(p)
    tones.sort()

    slot_dur = (end - start) / n_slots
    out = []
    for k, s in enumerate(sorted(slots)):
        idx = ALBERTI[k % len(ALBERTI)] % len(tones)
        pitch = tones[idx]
        # The downbeat gets the root in the bass — it is what makes the figure
        # read as harmony rather than as a run.
        if s == 0:
            pitch = tones[0]
        on = start + s * slot_dur
        out.append({
            "pitch": int(np.clip(pitch, lo, hi)),
            "onset": float(on),
            "offset": float(min(on + slot_dur * 0.95, end)),
            "velocity": 72 if s == 0 else 58,
        })
    return out


def melody_mask(notes: List[Dict[str, Any]], fps: float = 100.0) -> List[bool]:
    """Mark the notes that form the top line, which must survive untouched."""
    if not notes:
        return []
    t_end = max(float(n["offset"]) for n in notes)
    n_fr = max(1, int(t_end * fps) + 1)
    top = np.zeros(n_fr, dtype=np.int16)
    for n in notes:
        p = int(n["pitch"])
        if p < MELODY_MIN:
            continue
        a, b = int(float(n["onset"]) * fps), min(int(float(n["offset"]) * fps), n_fr)
        if b > a:
            seg = top[a:b]
            seg[seg < p] = p
    keep = []
    for n in notes:
        p = int(n["pitch"])
        a, b = int(float(n["onset"]) * fps), min(int(float(n["offset"]) * fps), n_fr)
        # A note is melody if it is the top voice for a real part of its life,
        # not merely for one frame while another note crosses it.
        keep.append(p >= MELODY_MIN and b > a and float(np.mean(top[a:b] == p)) > 0.5)
    return keep


def arpeggiate(notes: List[Dict[str, Any]],
               beat_times: Optional[np.ndarray],
               downbeat_times: Optional[np.ndarray] = None,
               beats_per_bar: int = 4,
               scale_pcs=None,
               key_root: Optional[int] = None) -> List[Dict[str, Any]]:
    """Keep the melody, rebuild everything under it as a repeating figure."""
    if not ARPEGGIATE or not notes or beat_times is None or len(beat_times) < 2:
        return notes

    keep = melody_mask(notes)
    melody = [n for n, k in zip(notes, keep) if k]
    accomp = [n for n, k in zip(notes, keep) if not k]
    if not accomp:
        return notes

    edges = _bar_edges(np.asarray(beat_times, float), downbeat_times, beats_per_bar)
    if len(edges) < 2:
        return notes

    n_slots = max(1, int(round(beats_per_bar * SUBDIV)))

    # Two passes: gather every bar's rhythm first, so the figures can be
    # chosen from a vocabulary the whole piece shares rather than invented
    # independently in each bar.
    bars = []
    for start, end in zip(edges[:-1], edges[1:]):
        bar = [n for n in accomp if start <= float(n["onset"]) < end]
        # Harmony is inferred from EVERY note in the bar, melody included, while
        # rhythm still comes from the accompaniment alone.
        #
        # Major and minor differ only in the third, and the bass overwhelmingly
        # plays roots and fifths — so with accompaniment-only chroma the two
        # templates score almost identically and the prior hands every tie to
        # major.  Measured that way, minor came out at 3.9% of bars against a
        # human 33.4%.  The third is usually in the tune, so the tune has to be
        # allowed to vote on the chord.
        full = [n for n in notes if start <= float(n["onset"]) < end]
        bars.append((start, end, bar, full))
    vectors = [bar_rhythm_vector(b, s, e, n_slots) for s, e, b, _f in bars]
    # Retrieval from human playing first; the clustering path stays as a
    # fallback for when the pattern artefact is absent.
    rhythms = retrieve_rhythms(vectors)
    if rhythms is None:
        rhythms = canonical_rhythms(vectors)

    rebuilt: List[Dict[str, Any]] = []
    for (start, end, bar, full), slots in zip(bars, rhythms):
        if not bar or not slots:
            continue
        chord = infer_chord(full or bar, key_root)   # harmony follows the song
        if chord is None:
            continue
        centre = int(np.median([int(n["pitch"]) for n in bar]))

        # Stay out of the singer's way.  With a fixed ACC_HIGH of 67, 41% of
        # generated notes landed ABOVE the melody split — the figure written to
        # declutter the bass was depositing a hundred notes straight into the
        # vocal register.  The ceiling now follows the melody in this bar, so
        # the left hand cannot crowd the line it is supposed to support.
        sung = [int(n["pitch"]) for n in melody
                if start <= float(n["onset"]) < end]
        ceiling = min(ACC_HIGH, (min(sung) - 2) if sung else ACC_HIGH)
        ceiling = max(ceiling, ACC_LOW + 12)     # never collapse the register

        rebuilt.extend(build_figure(chord[2], slots, start, end, centre,
                                    n_slots, ACC_LOW, ceiling, scale_pcs))

    out = melody + rebuilt
    out.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
    return out
