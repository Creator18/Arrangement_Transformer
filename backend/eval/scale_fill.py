"""Drop out-of-scale notes with a WINDOWED key, then fill the gaps by extending earlier notes forward.

The user's rule for discretising a hum into piano:
  0. held by default: consecutive strikes of one key less than 80 ms apart
     are merged into one held note, unless the song's own energy at that
     pitch jumps by >= 6 dB at the second strike (a re-articulation loud
     enough to notice) — only then is it re-struck.
  1. key per window, not one global key — songs modulate.  An 8 s window
     SLIDES in 0.5 s steps; each position gets a key from its notes'
     duration-weighted pitch classes (Krumhansl-Kessler profiles, 24 keys),
     smoothed with a switch penalty (Viterbi) so the key changes only when the
     music does.  Each change is then pinned to the exact note where it starts:
     the first note around the change that belongs to the new key and not the
     old one.
  2. a note outside its window's scale is dropped.  The scale is the major
     scale, or for a minor key the natural minor plus its raised 7th (the
     leading tone of the harmonic minor), so V chords in minor keys survive.
  3. each gap a dropped note leaves is filled by extending a note ALREADY
     PLAYED: the nearest-pitch in-scale note (within an octave) that started
     before the dropped note and is still sounding or ended in the last 0.5 s.
     It is held to the dropped note's end, never past the next strike of its
     own key.  Nothing starts earlier: a note not yet played cannot be extended.

    python eval/scale_fill.py <notes.json> <out.json>    (or import scale_fill)
"""
from __future__ import annotations

import json
import sys

import numpy as np

MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SCALE = {"maj": {0, 2, 4, 5, 7, 9, 11}, "min": {0, 2, 3, 5, 7, 8, 10, 11}}     # minor: + raised 7th


def window_keys(notes, win=8.0, hop=0.5, switch=4.0):
    """(window starts, key index 0-23 per window); 0-11 major on C.., 12-23 minor."""
    end = max(n["offset"] for n in notes)
    starts = np.arange(0, max(end - win, 0) + hop, hop)
    prof = [np.roll(MAJ, k) for k in range(12)] + [np.roll(MIN, k) for k in range(12)]
    prof = np.array([(p - p.mean()) / p.std() for p in prof])
    S = np.zeros((len(starts), 24))
    for i, s in enumerate(starts):
        h = np.zeros(12)
        for n in notes:
            ov = min(n["offset"], s + win) - max(n["onset"], s)
            if ov > 0:
                h[n["pitch"] % 12] += ov
        if h.sum() > 0:
            h = (h - h.mean()) / (h.std() + 1e-9)
            S[i] = prof @ h / 12                                           # correlation per key
    acc, back = S[0].copy(), []
    for i in range(1, len(S)):
        m = acc[:, None] - switch * (1 - np.eye(24))
        b = m.argmax(0); back.append(b); acc = m[b, np.arange(24)] + S[i]
    path = [int(acc.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return starts, np.array(path[::-1]), win


def key_name(k):
    return f"{NAMES[k % 12]} {'major' if k < 12 else 'minor'}"


def in_scale(pitch, k):
    return (pitch - k % 12) % 12 in SCALE["maj" if k < 12 else "min"]


def hold(notes, energy=None, gap=0.08, jump_db=6.0):
    """Merge near-touching re-strikes of one key unless the song's energy there jumps by jump_db.

    energy: optional f(pitch, t) -> song energy (linear) at that pitch and time."""
    out, last = [], {}
    merged = 0
    for n in sorted(notes, key=lambda n: (n["onset"], n["pitch"])):
        p = last.get(n["pitch"])
        if p is not None and n["onset"] - p["offset"] < gap:
            loud = False
            if energy is not None:
                before, after = energy(n["pitch"], n["onset"] - 0.06), energy(n["pitch"], n["onset"] + 0.03)
                loud = 20 * np.log10((after + 1e-9) / (before + 1e-9)) >= jump_db
            if not loud:
                p["offset"] = max(p["offset"], n["offset"]); merged += 1
                continue
        n = dict(n); out.append(n); last[n["pitch"]] = n
    return out, merged


def key_changes(notes, starts, keys, win):
    """[(time, key)]: each change pinned to the first note fitting the new key but not the old."""
    centres = starts + win / 2
    seg = [(0.0, int(keys[0]))]
    for i in range(1, len(keys)):
        if keys[i] == keys[i - 1]:
            continue
        old, new = int(keys[i - 1]), int(keys[i])
        lo, hi = centres[i - 1] - win / 2, centres[i] + win / 2
        hit = [n["onset"] for n in notes if lo <= n["onset"] <= hi
               and in_scale(n["pitch"], new) and not in_scale(n["pitch"], old)]
        t = min(hit) if hit else (centres[i - 1] + centres[i]) / 2
        seg.append((max(t, seg[-1][0]), new))
    return seg


def scale_fill(notes, win=8.0, hop=0.5, reach=0.5, energy=None):
    notes, merged = hold(notes, energy)
    notes = sorted(notes, key=lambda n: (n["onset"], n["pitch"]))
    starts, keys, win = window_keys(notes, win, hop)
    seg = key_changes(notes, starts, keys, win)
    seg_t = np.array([t for t, _ in seg])
    key_at = lambda t: seg[max(0, int(np.searchsorted(seg_t, t, side="right")) - 1)][1]
    kept, dropped = [], []
    for n in notes:
        (kept if in_scale(n["pitch"], key_at(n["onset"])) else dropped).append(n)
    by_pitch = {}
    for n in kept:
        by_pitch.setdefault(n["pitch"], []).append(n["onset"])
    extended = 0
    for d in dropped:
        cand = [n for n in kept if n["onset"] < d["onset"] and n["offset"] >= d["onset"] - reach
                and abs(n["pitch"] - d["pitch"]) <= 12]
        if not cand:
            continue
        n = min(cand, key=lambda n: (abs(n["pitch"] - d["pitch"]), d["onset"] - n["onset"]))
        nxt = [t for t in by_pitch[n["pitch"]] if t > n["onset"]]
        new_off = min(d["offset"], min(nxt) if nxt else d["offset"])
        if new_off > n["offset"]:
            n["offset"] = new_off; extended += 1
    changes = [(round(float(t), 2), key_name(k)) for t, k in seg]
    return kept, dict(held=merged, kept=len(kept), dropped=len(dropped), extended=extended, keys=changes)


if __name__ == "__main__":
    notes = json.load(open(sys.argv[1]))
    kept, info = scale_fill(notes)
    json.dump(kept, open(sys.argv[2], "w"))
    print(info)
