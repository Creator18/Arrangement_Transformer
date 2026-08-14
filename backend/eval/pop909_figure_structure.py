"""
Step 0 — do human accompaniment figures actually follow song structure?

The plan is to retrieve one accompaniment figure per SECTION instead of one per
bar, so that figures repeat where the song repeats and change where it changes.
That only makes sense if humans behave that way.  If a real arranger's figure
changes scatter randomly across bars, section-matched retrieval solves a problem
that does not exist.

POP909's base distribution ships no phrase or section annotations — only beats,
chords and key — so this uses two tests, neither of which needs them.

  RUN LENGTH (needs nothing)
      How many consecutive bars share the same figure?  If figures are held over
      spans, mean run length is well above 1 and the number of distinct figures
      per song is small.  If they are re-chosen every bar, runs are ~1 and the
      distinct count approaches the bar count.  This is the premise itself,
      tested directly, with a shuffled control for what chance looks like.

  BOUNDARY ALIGNMENT (needs a structural proxy)
      Sections are derived from the MELODY track's self-similarity — a different
      track from the PIANO one being measured, so this is not circular.  Then:
      do figure changes land on melody boundaries more often than chance?

Both are reported per song and pooled.  The premise needs the first to hold;
the second tells us whether the boundaries we can detect are the right ones.
"""
from __future__ import annotations

import argparse, collections, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

POP909 = ROOT / "data/POP909"
SLOTS = 16


def bars_of(folder):
    f = folder / "beat_midi.txt"
    if not f.exists():
        return None
    rows = np.loadtxt(f)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return None
    downs = rows[rows[:, 2] > 0.5, 0]
    return downs if len(downs) >= 8 else None


def figure_per_bar(notes, edges):
    """16-slot attack mask per bar, as an int; 0 means the bar is empty."""
    out = []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        if b1 <= b0:
            out.append(0)
            continue
        slots = set()
        for n in notes:
            if b0 <= n.start < b1:
                slots.add(int(np.clip((n.start - b0) / (b1 - b0) * SLOTS, 0, SLOTS - 1)))
        out.append(sum(1 << s for s in slots))
    return out


def melody_boundaries(mel_notes, edges, k=4):
    """Bars where the melody's content changes most — a structural proxy."""
    n_bars = len(edges) - 1
    chroma = np.zeros((n_bars, 12))
    for i, (b0, b1) in enumerate(zip(edges[:-1], edges[1:])):
        for n in mel_notes:
            if b0 <= n.start < b1:
                chroma[i, n.pitch % 12] += max(0.05, min(n.end, b1) - n.start)
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    C = chroma / norm
    S = C @ C.T                                   # bar-by-bar self-similarity
    # novelty: how different is the block after bar i from the block before it
    nov = np.zeros(n_bars)
    for i in range(k, n_bars - k):
        before, after = S[i - k:i, i - k:i], S[i:i + k, i:i + k]
        cross = S[i - k:i, i:i + k]
        nov[i] = (before.mean() + after.mean()) / 2 - cross.mean()
    if nov.max() <= 0:
        return set()
    thr = np.percentile(nov[nov > 0], 80)
    return {i for i in range(n_bars) if nov[i] >= thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    import pretty_midi

    runs_all, distinct_frac, bars_all = [], [], []
    shuf_runs = []
    align_hit, align_tot, chance_sum, n_align = 0, 0, 0.0, 0
    used = 0
    rng = np.random.default_rng(0)

    for folder in sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]:
        mid = folder / f"{folder.name}.mid"
        edges = bars_of(folder)
        if not mid.exists() or edges is None:
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:                                       # noqa: BLE001
            continue
        tracks = {(i.name or "").strip().upper(): i.notes for i in pm.instruments
                  if not i.is_drum}
        pno, mel = tracks.get("PIANO"), tracks.get("MELODY")
        if not pno or len(pno) < 32:
            continue
        used += 1

        figs = figure_per_bar(pno, edges)
        live = [f for f in figs if f]
        if len(live) < 8:
            continue
        bars_all.append(len(live))
        distinct_frac.append(len(set(live)) / len(live))

        def mean_run(seq):
            runs, cur = [], 1
            for a, b in zip(seq[:-1], seq[1:]):
                if a == b:
                    cur += 1
                else:
                    runs.append(cur)
                    cur = 1
            runs.append(cur)
            return float(np.mean(runs))

        runs_all.append(mean_run(live))
        sh = list(live)
        rng.shuffle(sh)
        shuf_runs.append(mean_run(sh))

        # boundary alignment
        if mel and len(mel) >= 16:
            bnds = melody_boundaries(mel, edges)
            changes = {i for i in range(1, len(figs))
                       if figs[i] and figs[i - 1] and figs[i] != figs[i - 1]}
            if bnds and changes:
                # a change counts as aligned if it is within one bar of a boundary
                near = {b + d for b in bnds for d in (-1, 0, 1)}
                hit = len(changes & near)
                align_hit += hit
                align_tot += len(changes)
                chance_sum += len(near) / max(len(figs), 1) * len(changes)
                n_align += 1

    print(f"POP909: {used} songs\n")
    print("RUN LENGTH — are figures held over spans?")
    print(f"  mean consecutive bars sharing a figure : {np.mean(runs_all):.2f}")
    print(f"  same, with bar order SHUFFLED         : {np.mean(shuf_runs):.2f}  <- chance")
    print(f"  median bars per song                  : {np.median(bars_all):.0f}")
    print(f"  distinct figures as a share of bars   : {100*np.mean(distinct_frac):.1f}%")
    print()
    print("BOUNDARY ALIGNMENT — do changes land on melody boundaries?")
    if n_align:
        obs = 100.0 * align_hit / max(align_tot, 1)
        exp = 100.0 * chance_sum / max(align_tot, 1)
        print(f"  figure changes within 1 bar of a boundary : {obs:.1f}%")
        print(f"  expected if changes were placed at random : {exp:.1f}%")
        print(f"  lift over chance                          : {obs/max(exp,1e-9):.2f}x")
    else:
        print("  not enough songs with both tracks")
    print()
    print("Premise holds if run length clearly exceeds the shuffled control and")
    print("distinct figures are a small share of bars.  If figures turn over every")
    print("bar, section-matched retrieval has nothing to retrieve.")


if __name__ == "__main__":
    main()
