"""Structure-aware tokens: the arrangement language WITH its musical reasons.

Extends tokens.py without changing it: PAD/BOS/EOS/MASK, SHIFT, PITCH, DUR and
VEL keep their ids, so the grammar model's learned note embeddings carry over
(warm start) and only the structure tokens are new.  Per bar:

  BAR  [METER m] [KEY k] [SEC s]       meter/key/section only when they change
  [GROUP g]                            the passage's accompaniment rhythm group
  POS p CHORD c                        chord changes, at their beat
  POS p ROLE r PITCH DUR VEL ...       notes: position in the bar on a 24-per-
                                       beat grid (~21 ms at 120 BPM), role
                                       (melody / bass / accompaniment), then the
                                       unchanged note tokens

Time is bars + beats, so a groove is the same pattern at any tempo; decoding
maps positions back to seconds through the song's beat grid.
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Sequence

import numpy as np

from . import tokens as T
from .structure import QUALITIES, Structure

GRID = 24                          # positions per beat
MAX_BEATS = 6                      # longest bar handled (12/8 counts 4, 9/8 counts 3)
METERS = ["4/4", "3/4", "12/8", "9/8", "2/4", "6/8"]
N_SEC, N_GROUP = 8, 64

BAR = T.VOCAB                      # 196
METER0 = BAR + 1
KEY0 = METER0 + len(METERS)
SEC0 = KEY0 + 24
GROUP0 = SEC0 + N_SEC
GROUP_NONE = GROUP0 + N_GROUP - 1
POS0 = GROUP0 + N_GROUP
CHORD0 = POS0 + GRID * MAX_BEATS
CHORD_NONE = CHORD0 + 12 * len(QUALITIES)
ROLE0 = CHORD_NONE + 1
VOCAB = ROLE0 + 3


def group_table(path) -> dict:
    """Stable group-name -> index map, from group_centroids.json (sorted names)."""
    names = sorted(json.loads(pathlib.Path(path).read_text()))
    return {n: i for i, n in enumerate(names[: N_GROUP - 1])}


def bar_grid(st: Structure) -> List[int]:
    """Beat index of every bar start: a pickup becomes bar 0, and bars longer
    than MAX_BEATS (irregular meter, tracking slips) are split so positions fit."""
    bs = list(st.bar_starts)
    starts = bs if bs and bs[0] == 0 else [0] + bs
    out = []
    for a, b in zip(starts, starts[1:]):
        out += list(range(a, b, MAX_BEATS)) if b - a > MAX_BEATS else [a]
    return out + [starts[-1]]


def encode(st: Structure, gmap: dict | None = None) -> List[int]:
    out, cur = [], {"meter": None, "key": None, "sec": None}
    beats, bs = st.beats, st.bar_starts
    on = st.notes[:, 0]
    # a pickup before the first downbeat becomes bar 0
    starts = bar_grid(st)
    for b in range(len(starts) - 1):
        i0, i1 = starts[b], starts[b + 1]
        out.append(BAR)                                 # every bar, so bars never shift
        if st.meter != cur["meter"]:
            out.append(METER0 + METERS.index(st.meter) if st.meter in METERS else METER0); cur["meter"] = st.meter
        if st.key != cur["key"]:
            out.append(KEY0 + st.key); cur["key"] = st.key
        bi = b if bs[0] == 0 else b - 1                 # index into st.sections / st.groups
        sec = st.sections[bi] if 0 <= bi < len(st.sections) else None
        if sec is not None and sec != cur["sec"]:
            out.append(SEC0 + min(sec, N_SEC - 1)); cur["sec"] = sec
        if bi in st.groups and gmap is not None:
            out.append(GROUP0 + gmap.get(st.groups[bi], N_GROUP - 1))
        events = []                                     # (pos, order, tokens)
        prev = None
        for j in range(i0, i1):
            c = st.chords[j] if j < len(st.chords) else None
            if c != prev:
                tok = CHORD_NONE if c is None else CHORD0 + c[0] * len(QUALITIES) + QUALITIES.index(c[1])
                events.append(((j - i0) * GRID, 0, [tok]))
                prev = c
        t0, t1 = beats[i0], beats[i1]
        sel = np.flatnonzero((on >= t0) & (on < t1))
        for k in sel:
            j = min(int(np.searchsorted(beats, on[k], "right")) - 1, i1 - 1)
            frac = (on[k] - beats[j]) / max(1e-3, beats[j + 1] - beats[j])
            pos = (j - i0) * GRID + min(GRID - 1, int(round(frac * GRID)))
            s, e, p, v = st.notes[k]
            if not 21 <= p <= 108:
                continue
            events.append((pos, 1 + int(st.roles[k]), [ROLE0 + int(st.roles[k]), T.PITCH0 + int(p) - 21,
                                                        T.DUR0 + T.dur_bin(max(e - s, T.STEP)),
                                                        T.VEL0 + int(np.clip(v, 1, 127)) * T.N_VEL // 128]))
        last = None
        for pos, _, toks in sorted(events, key=lambda x: (x[0], x[1])):
            if pos != last:
                out.append(POS0 + pos); last = pos
            out += toks
    return out


def decode(tokens: Sequence[int], beats: np.ndarray, bar_starts: Sequence[int]):
    """tokens -> (notes [on,off,pitch,vel], roles, chords per beat).

    The song's BAR GRID (beat index of each bar start, from bar_grid() or the
    song's own beat tracking at inference) places each BAR; positions inside a
    bar map to seconds through `beats`.  Given the grid, a bar can never drift."""
    notes, roles, chords = [], [], {}
    bar, bar_beat, pos, role = -1, 0, 0, 2
    tok, i = list(tokens), 0
    while i < len(tok):
        x = tok[i]
        if x == BAR:
            bar += 1
            bar_beat = bar_starts[bar] if bar < len(bar_starts) else bar_starts[-1] + 4 * (bar - len(bar_starts) + 1)
            pos = 0; i += 1; continue
        if POS0 <= x < CHORD0:
            pos = x - POS0
        elif CHORD0 <= x <= CHORD_NONE:
            chords[bar_beat + pos // GRID] = None if x == CHORD_NONE else \
                ((x - CHORD0) // len(QUALITIES), QUALITIES[(x - CHORD0) % len(QUALITIES)])
        elif ROLE0 <= x < VOCAB:
            role = x - ROLE0
        elif T.PITCH0 <= x < T.DUR0:
            j = bar_beat + pos // GRID
            if 0 <= j < len(beats) - 1:
                t = beats[j] + (pos % GRID) / GRID * (beats[j + 1] - beats[j])
                d = T.dur_value(tok[i + 1] - T.DUR0) if i + 1 < len(tok) and T.DUR0 <= tok[i + 1] < T.VEL0 else 0.25
                v = ((tok[i + 2] - T.VEL0) * 128 // T.N_VEL + 8) if i + 2 < len(tok) and T.VEL0 <= tok[i + 2] < T.VOCAB else 80
                notes.append((t, t + d, x - T.PITCH0 + 21, v)); roles.append(role)
            i += 3; continue
        i += 1
    return np.array(notes, float).reshape(-1, 4), np.array(roles, int), chords


def encode_plain(notes: np.ndarray, beats: np.ndarray, bar_starts: Sequence[int]) -> List[int]:
    """Notes on the bar/beat grid with NO structure: BAR, POS, PITCH, DUR, VEL only.

    The encoder's input.  The generation check showed the decoder losing notes
    to beat placement (POP909 per-bar pitch F1 0.870 -> 0.893, yet timed F1
    0.857 -> 0.804) because plain-seconds input made it GUESS the beat grid.
    Given the grid here, it only decides what to play at each position.
    `bar_starts`: beat index of each bar start, one more than the bars to encode.
    """
    out = []
    on = notes[:, 0] if len(notes) else np.zeros(0)
    for b in range(len(bar_starts) - 1):
        i0, i1 = int(bar_starts[b]), int(bar_starts[b + 1])
        if i1 >= len(beats):
            break
        out.append(BAR)
        t0, t1 = beats[i0], beats[i1]
        sel = np.flatnonzero((on >= t0) & (on < t1))
        ev = []
        for k in sel:
            j = min(int(np.searchsorted(beats, on[k], "right")) - 1, i1 - 1)
            frac = (on[k] - beats[j]) / max(1e-3, beats[j + 1] - beats[j])
            pos = min((j - i0) * GRID + min(GRID - 1, int(round(frac * GRID))), GRID * MAX_BEATS - 1)
            s, e, p, v = notes[k]
            if 21 <= p <= 108:
                ev.append((pos, int(p), [T.PITCH0 + int(p) - 21, T.DUR0 + T.dur_bin(max(e - s, T.STEP)),
                                         T.VEL0 + int(np.clip(v, 1, 127)) * T.N_VEL // 128]))
        last = None
        for pos, _, toks in sorted(ev, key=lambda x: (x[0], x[1])):
            if pos != last:
                out.append(POS0 + pos); last = pos
            out += toks
    return out
