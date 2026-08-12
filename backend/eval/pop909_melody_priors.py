"""
What does a human-notated vocal line actually look like?

melody_cleanup.py enforces three things about the voice register — one note at
a time, nothing shorter than MIN_DUR, onsets snapped to eighths — and all three
numbers were my guesses.  POP909 ships 909 pop songs whose MELODY track is a
human transcription of the sung line, with beat annotations alongside, so the
guesses can be replaced with measurements.

What this settles, and why each one matters
-------------------------------------------
  monophony     The whole cleanup rests on "a sung line is one note at a
                time".  If human-notated vocal tracks contain overlaps, that
                assumption is deleting real material and needs weakening.
  duration      MIN_DUR = 0.12 s was a guess at "shorter than this is an
                artefact".  Set it above the real 5th percentile and the filter
                removes notes singers actually sing.
  onset grid    MELODY_SUBDIV = 2 assumes vocal onsets fall on eighths.  If
                they mostly land on sixteenths, snapping to eighths is moving
                notes off the beat rather than onto it.
  gaps          GAP_BRIDGE = 0.10 s decides when two same-pitch runs are one
                re-struck note versus two sung notes.  The real distribution of
                same-pitch gaps says where that line belongs.
  semitone move How often a sung line genuinely moves by one semitone and
                straight back — the pattern the cleanup treats as vibrato
                smear.  If humans do it often, that filter is too aggressive.

Read the numbers as an upper bound on strictness: POP909 is quantised human
transcription, so it is tidier than a real performance and tidier still than
our detections.  Where it disagrees with a guess, the guess is wrong.

RESULT (909 songs, 309,423 melody notes, 2026-08-11)
----------------------------------------------------
    monophony     9.40% of consecutive notes overlap (median song 5.81%),
                  54.1% of songs above 5%
    duration      p1 0.082  p5 0.109  p10 0.130  p50 0.291  p95 1.269
                  7.6% shorter than the guessed MIN_DUR of 0.12
    onset grid    beat 24.3%  eighth 18.4%  sixteenth 33.7%  triplet 1.9%
                  none 21.8%   -> eighth-or-coarser only 42.7%,
                  sixteenth-or-coarser 76.4%
    gaps          same-pitch p25 0.068  p50 0.121; 35.8% under the guessed
                  GAP_BRIDGE of 0.10
    movement      1.37 notes/s mean (median 1.34); unison 23.6% of intervals,
                  +-1 semitone 8.2%; 29.9% of semitone steps return

These figures were then swept against two real uploads, and the outcome split:
the GRID and GAP corrections held (sixteenths, 0.05), while the DURATION one
was wrong in the opposite direction — 0.15 beats both 0.12 and the
percentile-derived 0.10.

That is the standing lesson from this file.  A percentile of the human
distribution tells you what a threshold COSTS in real notes; it says nothing
about what that threshold REMOVES in artefacts, and removing artefacts is the
only reason the threshold exists.  Our sub-0.15 s notes are overwhelmingly
NMF re-strikes, not sung notes, so the filter should sit well above the human
5th percentile.  Use POP909 to price a decision, never to make it.

One validation worth recording: after cleanup our melody runs 1.32 notes/s
against a human 1.37, from 5.1 before it.  The cleanup landed on human density
without being told what it was.

The monophony assumption HELD, with a caveat: the 9.4% overlap is brief legato
joins, not two-note singing, which is exactly why the implementation truncates
the earlier note rather than dropping either.  And 29.9% of human semitone
steps step back — the pattern treated elsewhere as vibrato smear — so some
genuine ornament is being flattened.  In absolute terms that is only ~2.5% of
all transitions, since semitone steps are just 8.2% of intervals.
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
TOL = 0.03            # +-30 ms counts as "on" a grid position


def beats_of(folder: pathlib.Path):
    """Beat and downbeat times from beat_midi.txt.

    Columns are (time, beat_flag, downbeat_flag); rows are laid out at
    sub-beat resolution, so the beat grid is the subset where beat_flag is set.
    """
    f = folder / "beat_midi.txt"
    if not f.exists():
        return None, None
    rows = np.loadtxt(f)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return None, None
    beats = rows[rows[:, 1] > 0.5, 0]
    downs = rows[rows[:, 2] > 0.5, 0]
    return (beats if len(beats) >= 4 else rows[:, 0]), downs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=909)
    args = ap.parse_args()

    import pretty_midi

    folders = sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]
    print(f"POP909: {len(folders)} songs\n")

    durs, gaps_same, gaps_any, nps = [], [], [], []
    overlap_frac, n_songs = [], 0
    grid_hits = {"beat": 0, "eighth": 0, "sixteenth": 0, "triplet": 0, "none": 0}
    total_onsets = 0
    semitone_wobbles, melodic_moves = 0, 0
    interval_hist = np.zeros(25)          # -12..+12

    for folder in folders:
        mid = folder / f"{folder.name}.mid"
        if not mid.exists():
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:                                   # noqa: BLE001
            continue
        mel = next((i for i in pm.instruments
                    if (i.name or "").strip().upper() == "MELODY"), None)
        if mel is None or len(mel.notes) < 8:
            continue
        n_songs += 1
        notes = sorted(mel.notes, key=lambda n: n.start)

        # monophony: how much of the line has something else sounding over it
        ov = sum(1 for a, b in zip(notes[:-1], notes[1:]) if b.start < a.end - 1e-4)
        overlap_frac.append(ov / max(len(notes) - 1, 1))

        span = pm.get_end_time() or 1.0
        nps.append(len(notes) / span)
        for n in notes:
            durs.append(n.end - n.start)
        for a, b in zip(notes[:-1], notes[1:]):
            g = b.start - a.end
            if g >= 0:
                gaps_any.append(g)
                if a.pitch == b.pitch:
                    gaps_same.append(g)

        # melodic movement
        for a, b in zip(notes[:-1], notes[1:]):
            iv = int(np.clip(b.pitch - a.pitch, -12, 12))
            interval_hist[iv + 12] += 1
        for x, y, z in zip(notes[:-2], notes[1:-1], notes[2:]):
            if abs(y.pitch - x.pitch) == 1:
                melodic_moves += 1
                if z.pitch == x.pitch:
                    semitone_wobbles += 1

        # onset placement against the song's own beat grid
        beats, _ = beats_of(folder)
        if beats is None or len(beats) < 4:
            continue
        beats = np.asarray(beats, dtype=float)
        for n in notes:
            t = n.start
            k = int(np.searchsorted(beats, t)) - 1
            if k < 0 or k + 1 >= len(beats):
                continue
            b0, b1 = beats[k], beats[k + 1]
            if b1 <= b0:
                continue
            pos = (t - b0) / (b1 - b0)          # 0..1 within the beat
            total_onsets += 1
            near = lambda frac: abs(pos - frac) * (b1 - b0) <= TOL
            if near(0.0) or near(1.0):
                grid_hits["beat"] += 1
            elif near(0.5):
                grid_hits["eighth"] += 1
            elif any(near(f) for f in (0.25, 0.75)):
                grid_hits["sixteenth"] += 1
            elif any(near(f) for f in (1 / 3, 2 / 3)):
                grid_hits["triplet"] += 1
            else:
                grid_hits["none"] += 1

    d = np.array(durs)
    print(f"songs used: {n_songs}   melody notes: {len(d)}\n")

    print("MONOPHONY")
    print(f"  overlapping consecutive notes: {100*np.mean(overlap_frac):.2f}% "
          f"(median song {100*np.median(overlap_frac):.2f}%)")
    print(f"  songs with >5% overlap: {100*np.mean(np.array(overlap_frac) > 0.05):.1f}%\n")

    print("NOTE DURATION (seconds)")
    for q in (1, 5, 10, 25, 50, 75, 95):
        print(f"  p{q:<3} {np.percentile(d, q):.3f}")
    print(f"  mean {d.mean():.3f}   shorter than our MIN_DUR=0.12: "
          f"{100*np.mean(d < 0.12):.1f}%\n")

    print("ONSET PLACEMENT (share of onsets within +-30 ms)")
    for k in ("beat", "eighth", "sixteenth", "triplet", "none"):
        print(f"  {k:<10} {100*grid_hits[k]/max(total_onsets,1):5.1f}%")
    cum8 = grid_hits["beat"] + grid_hits["eighth"]
    print(f"  -> on an EIGHTH grid or coarser: {100*cum8/max(total_onsets,1):.1f}%")
    print(f"  -> on a SIXTEENTH grid or coarser: "
          f"{100*(cum8+grid_hits['sixteenth'])/max(total_onsets,1):.1f}%\n")

    ga, gs = np.array(gaps_any), np.array(gaps_same)
    print("GAPS between consecutive melody notes (seconds)")
    print(f"  any pitch  p25 {np.percentile(ga,25):.3f}  p50 {np.percentile(ga,50):.3f}")
    if gs.size:
        print(f"  SAME pitch p25 {np.percentile(gs,25):.3f}  p50 {np.percentile(gs,50):.3f}"
              f"   under our GAP_BRIDGE=0.10: {100*np.mean(gs < 0.10):.1f}%")
    print()

    print("MELODIC MOVEMENT")
    print(f"  notes/second: mean {np.mean(nps):.2f}  median {np.median(nps):.2f}")
    print(f"  1-semitone step then straight back: "
          f"{100*semitone_wobbles/max(melodic_moves,1):.1f}% of semitone steps")
    tot = interval_hist.sum() or 1
    print(f"  intervals: unison {100*interval_hist[12]/tot:.1f}%  "
          f"+-1st {100*(interval_hist[11]+interval_hist[13])/tot:.1f}%  "
          f"+-2nd {100*(interval_hist[10]+interval_hist[14])/tot:.1f}%  "
          f"leap>4st {100*(interval_hist[:8].sum()+interval_hist[17:].sum())/tot:.1f}%")


if __name__ == "__main__":
    main()
