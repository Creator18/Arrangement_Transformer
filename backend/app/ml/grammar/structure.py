"""Structure labels for an arrangement: meter, bars, beats, key, chords, roles, sections, groups.

The structure-aware grammar (tokens2.py) learns piano writing IN TERMS OF its
musical reasons, so every arrangement is labelled with:

  beats / bars / meter   POP909: its own beat file; Music2MIDI: counted from the
                         notes (eval/rhythm_groups.py: beat F ~0.81-0.86 on POP909)
  key                    POP909: key_audio.txt; otherwise Krumhansl-Schmuckler on
                         duration-weighted pitch classes
  chords, per beat       POP909: chord_midi.txt; otherwise arpeggiator.infer_chord
                         on the accompaniment sounding in the beat (scored against
                         POP909: root 76%, quality 78% with accompaniment notes)
  role, per note         melody / bass / accompaniment: POP909's MELODY track or the
                         trained melody split; bass = lowest accompaniment note of
                         each onset event below C4
  sections               repeat structure: bars clustered by harmony + rhythm,
                         smoothed into runs (A, B, C … in order of appearance)
  rhythm group           per passage, from eval/rhythm_groups.py's group_labels.json

Everything is keyed to BARS and BEATS, so a pattern is the same at any tempo.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "eval"))

QUALITIES = ["maj", "min", "dom7", "min7", "dim"]
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
ROLE_MEL, ROLE_BASS, ROLE_ACC = 0, 1, 2


@dataclass
class Structure:
    notes: np.ndarray                 # [on, off, pitch, vel] sorted by onset
    roles: np.ndarray                 # per note
    beats: np.ndarray                 # beat times (s), one past the last bar
    bar_starts: np.ndarray            # indices into beats
    beats_per_bar: int
    subdivision: int
    meter: str
    key: int                          # 0-11 major, 12-23 minor
    chords: List[Optional[tuple]]     # per beat: (root, quality) or None
    sections: List[int] = field(default_factory=list)          # per bar
    groups: Dict[int, str] = field(default_factory=dict)       # bar index -> group name


def estimate_key(notes) -> int:
    c = np.zeros(12)
    np.add.at(c, notes[:, 2].astype(int) % 12, np.maximum(notes[:, 1] - notes[:, 0], 0.05))
    best, k = -9, 0
    for r in range(12):
        for m, prof in ((0, KS_MAJ), (12, KS_MIN)):
            s = np.corrcoef(c, np.roll(prof, r))[0, 1]
            if s > best:
                best, k = s, r + m
    return k


def parse_key(txt: str) -> Optional[int]:
    """'Gb:maj' -> 6, 'A:min' -> 21."""
    try:
        name, mode = txt.split(":")
        flat = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}.get(name, name)
        return NAMES.index(flat) + (12 if mode.startswith("min") else 0)
    except (ValueError, IndexError):
        return None


def parse_harte(label: str) -> Optional[tuple]:
    """POP909 Harte chord -> (root, quality in QUALITIES) or None."""
    if label in ("N", "X", ""):
        return None
    root, _, q = label.partition(":")
    root = root.split("/")[0]
    flat = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb", "Cb": "B", "Fb": "E"}.get(root, root)
    if flat not in NAMES:
        return None
    q = q.split("/")[0] or "maj"
    qual = ("min7" if q.startswith("min7") else "min" if q.startswith("min") else
            "dim" if q.startswith("dim") or q.startswith("hdim") else
            "dom7" if q in ("7", "9", "11", "13") else "maj")
    return NAMES.index(flat), qual


def assign_roles(notes, melody_mask) -> np.ndarray:
    roles = np.where(melody_mask, ROLE_MEL, ROLE_ACC)
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    for e in np.unique(ev):
        ix = np.flatnonzero((ev == e) & (roles == ROLE_ACC))
        if len(ix):
            j = ix[np.argmin(notes[ix, 2])]
            if notes[j, 2] < 60:
                roles[j] = ROLE_BASS
    return roles


def beat_chords(notes, roles, beats, key) -> List[Optional[tuple]]:
    from app.services.arpeggiator import infer_chord
    acc = notes[roles != ROLE_MEL]
    out = []
    for i in range(len(beats) - 1):
        t0, t1 = beats[i], beats[i + 1]
        sel = acc[(acc[:, 0] < t1) & (acc[:, 1] > t0)]
        ns = [{"pitch": int(p), "onset": max(on, t0), "offset": min(off, t1), "velocity": int(v)}
              for on, off, p, v in sel]
        c = infer_chord(ns, key_root=key % 12) if ns else None
        out.append((c[0], c[1]) if c and c[1] in QUALITIES else None)
    return out


def sections(notes, beats, bar_starts, k_max=6) -> List[int]:
    """Repeat structure: bars clustered by harmony + accompaniment rhythm, runs smoothed."""
    from sklearn.cluster import AgglomerativeClustering
    B = len(bar_starts) - 1
    if B < 4:
        return [0] * max(B, 0)
    F = np.zeros((B, 12 + 16))
    for b in range(B):
        t0, t1 = beats[bar_starts[b]], beats[bar_starts[b + 1]]
        sel = notes[(notes[:, 0] >= t0) & (notes[:, 0] < t1)]
        if len(sel):
            np.add.at(F[b, :12], sel[:, 2].astype(int) % 12, 1.0)
            pos = np.clip(((sel[:, 0] - t0) / max(1e-3, t1 - t0) * 16).astype(int), 0, 15)
            np.add.at(F[b, 12:], pos, 1.0)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    # 4-bar context: a section is a phrase, not a single bar
    G = np.stack([F[max(0, b - 1):b + 3].mean(0) for b in range(B)])
    k = int(np.clip(round(B / 16), 2, k_max))
    lab = AgglomerativeClustering(k).fit_predict(G)
    # smooth: runs shorter than 4 bars join their neighbour
    for _ in range(2):
        b = 0
        while b < B:
            e = b
            while e < B and lab[e] == lab[b]:
                e += 1
            if e - b < 4 and b > 0:
                lab[b:e] = lab[b - 1]
            b = e
    order = {}
    return [order.setdefault(int(x), len(order)) for x in lab]


def label(notes, melody_mask=None, beats=None, downbeats=None, key=None, chords_harte=None,
          group_labels=None) -> Structure:
    """notes [on,off,pitch,vel]; optional truths (POP909) else estimated."""
    import rhythm_groups as RG
    o = np.argsort(notes[:, 0], kind="stable")
    notes = notes[o]
    melody_mask = melody_mask[o] if melody_mask is not None else RG.melody_split(notes)
    if beats is None:
        beats, tempo = RG.track_beats(notes)
        beats, tempo = RG.settle_tempo(notes, beats, tempo)
    beats = np.asarray(beats, float)
    m, ph, sub, _ = RG.meter_from_beats(notes, beats)
    if downbeats is not None and len(downbeats) > 2:
        ph = int(np.argmin(np.abs(beats - downbeats[0])))
        m = int(round(np.median(np.diff(np.searchsorted(beats, downbeats)))))
        m = m if m in (2, 3, 4) else 4
    bar_starts = np.arange(ph, len(beats), m)
    roles = assign_roles(notes, melody_mask)
    key = estimate_key(notes) if key is None else key
    if chords_harte is not None:
        ch = []
        for i in range(len(beats) - 1):
            mid = (beats[i] + beats[i + 1]) / 2
            hit = [parse_harte(l) for s, e, l in chords_harte if s <= mid < e]
            ch.append(hit[0] if hit else None)
    else:
        ch = beat_chords(notes, roles, beats, key)
    st = Structure(notes, roles, beats, bar_starts, m, sub, RG.MERGE.get(RG.meter_name(m, sub), RG.meter_name(m, sub)),
                   key, ch)
    st.sections = sections(notes, beats, bar_starts)
    if group_labels:
        bt = beats[bar_starts]
        for t, g in group_labels:
            if g:
                st.groups[int(np.argmin(np.abs(bt - t)))] = g
    return st
