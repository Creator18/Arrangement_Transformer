"""Drop the passing notes of singer / instrument slides (the user's "flickers", 2026-10-29).

A scoop, slide or wide vibrato passes through the keys between two sung notes for a moment, and the base hears it
exactly: a quick out-of-key note right next to the real one.  Measured on 925 Music2MIDI songs (eval/never_kept.py):
such notes are 1.7% of ours, and 84.5% of them are in NO human arrangement of the song (a human's own: 46%).

Rule (drop only, never moves or writes a note): a note is dropped when it is
    short (< 150 ms), out of the local key, and next to a longer note (>= 1.5x) 1-2 keys away that it overlaps or
    runs into (that note starts before this one ends + 100 ms, and ends after this one starts - 50 ms).
The local key: Krumhansl profiles over our own notes (length-weighted) within +-15 s.

    ARRANGEMENT_SLIDE_DROP   1 = on (off when unset)
"""
import os

import numpy as np

ON = os.environ.get("ARRANGEMENT_SLIDE_DROP", "0") == "1"
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
STEPS = (np.array([0, 2, 4, 5, 7, 9, 11]), np.array([0, 2, 3, 5, 7, 8, 10]))
SPAN, BLOCK = 15.0, 5.0


def enabled():
    return ON


def local_scales(s, e, p):
    """per BLOCK-second block: the set of pitch classes in the key heard within +-SPAN s."""
    L = np.clip(e - s, 0.05, 2.0)
    out = []
    for b in range(int(s.max() // BLOCK) + 1 if len(s) else 0):
        c = (b + 0.5) * BLOCK
        m = np.abs(s - c) <= SPAN
        h = np.bincount(p[m] % 12, weights=L[m], minlength=12)
        if h.sum() == 0:
            out.append(set(range(12)))
            continue
        _, k, mode = max((np.corrcoef(np.roll(prof, k), h)[0, 1], k, mode)
                         for k in range(12) for mode, prof in enumerate((MAJ, MIN)))
        out.append(set(((STEPS[mode] + k) % 12).tolist()))
    return out


def slide_notes(notes):
    """boolean mask of the slide notes (see the module docstring)."""
    if not notes:
        return np.zeros(0, bool)
    s = np.array([g["onset"] for g in notes]); e = np.array([g["offset"] for g in notes])
    p = np.array([int(g["pitch"]) for g in notes])
    L = e - s
    sc = local_scales(s, e, p)
    drop = np.zeros(len(notes), bool)
    for i in np.where(L < 0.15)[0]:
        if p[i] % 12 in sc[int(s[i] // BLOCK)]:
            continue
        d = np.abs(p - p[i])
        drop[i] = np.any((d >= 1) & (d <= 2) & (s < e[i] + 0.1) & (e > s[i] - 0.05) & (L >= 1.5 * L[i]))
    return drop


def apply(notes):
    drop = slide_notes(notes)
    print(f"[slides] dropped {int(drop.sum())} of {len(notes)} notes (short, out of key, next to a longer note 1-2 keys away)")
    return [g for g, d in zip(notes, drop) if not d]
