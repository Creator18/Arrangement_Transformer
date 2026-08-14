"""
Does the piano arrangement FIT INSIDE the song?

The arrangement is a reduction, so the honest relationship is containment, not
similarity: the song legitimately holds vocals, drums and inner parts we chose
to drop, and none of that should count against us.  But there is no legitimate
reason for PIANO energy to sit where the song has none.  So recall is ignored
on purpose and only precision is scored — where we put a note, was anything
there?

The number alone is not the point, and taken alone it is a bad metric: an
arrangement of ten notes would score nearly perfectly, so containment rewards
silence.  Two things fix that.

    A CHANCE BASELINE.  The same note set, circularly shifted by a large
    offset, keeps the density and the pitch distribution exactly and destroys
    only the alignment.  Containment must beat that or it is measuring how
    dense the song is rather than whether we are in time.

    A SHIFT SWEEP.  Slide the piano map +-k and see where containment peaks.
    The peak location is the timing answer and is robust to almost every bias
    here, because anything that inflates the score inflates it at every shift
    roughly equally.  Peak at 0 means the timing is right; a peak elsewhere is
    a systematic offset, and its width is the precision.

Two containment tests, because they fail differently:

    LEVEL   is there energy at that pitch at that time at all?  Forgiving; a
            sustained chord covers a lot of frames.
    ONSET   is energy RISING there?  This is the sharp one for timing — a note
            placed on the wrong sixteenth can still sit inside a sustained
            chord and pass the level test while failing this.

Finally the sweep is repeated PER BAR.  A per-bar optimum that drifts across
the song is a beat-grid problem; per-bar optima that scatter randomly is a
slot-choice problem.  timing_alignment_test could not separate those two — its
grid control was too weak to convict or clear the grid — and this does, with
no external annotation.

RESULT (session 23f6a3dc, 1032-note two-voice arrangement, 2026-08-14)
----------------------------------------------------------------------
    threshold   ONSET peak            at 0      chance
    q=60        86.0% at   +0 ms      86.0%      19.8%
    q=75        80.0% at   +0 ms      80.0%      12.5%
    q=40        91.3% at   +0 ms      91.3%      30.2%

    per bar (177 bars): median -23 ms, IQR -46..+0, r vs bar index -0.093

The onset sweep peaks at EXACTLY zero at every threshold, at 4-6x chance.
There is no systematic timing offset.  And r = -0.09 over 177 bars means the
beat grid does not drift — which is the question timing_alignment_test's grid
control was too weak to answer, now settled without POP909.

The LEVEL rows peak instead at +70..+116 ms, and that is an artefact worth
naming: a note's energy PERSISTS after its attack, so sliding the piano map
later keeps it inside the sustain and the level score keeps rising.  Level
containment is therefore biased late by construction and cannot time anything.
Only the onset row is a timing measurement.  Having both is what exposed it.

This supersedes timing_alignment_test's "35.2% more than 50 ms from evidence".
That searched a +-250 ms window for the largest activation rise at the note's
pitch, which on dense material finds a NEIGHBOURING note's attack and scores
the distance to that.  It had no chance baseline, so there was nothing to
catch it.  Here 86% of onsets land on a rising edge against a 19.8% chance
floor, and the residual is ~14%, not ~35%.

So the remaining error is neither a lag nor a drift: it is a scattered ~14% of
notes on the wrong slot, plus a per-bar pull of about -23 ms with an IQR of
roughly half a subdivision (81 ms here).
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

TEMP = ROOT / "backend/temp"


def piano_onset_map(notes, low, n_pitch, n_fr, fps, mark_s=0.03):
    """Binary (88, T): a short mark at each note ONSET, not the whole note.

    Sustain would smear the shift curve — a long note overlaps its neighbours'
    slots and the peak flattens.  Timing lives in the attack.
    """
    P = np.zeros((n_pitch, n_fr), dtype=bool)
    w = max(1, int(mark_s * fps))
    for n in notes:
        r = int(n["pitch"]) - low
        f = int(float(n["onset"]) * fps)
        if 0 <= r < n_pitch and 0 <= f < n_fr:
            P[r, f: min(n_fr, f + w)] = True
    return P


def song_masks(H, q_level, q_onset):
    """Boolean masks for 'energy present' and 'energy rising', per pitch."""
    lev = np.zeros_like(H, dtype=bool)
    ons = np.zeros_like(H, dtype=bool)
    d = np.diff(H, axis=1, prepend=H[:, :1])
    for r in range(H.shape[0]):
        row = H[r]
        if not np.any(row > 0):
            continue
        lev[r] = row >= np.percentile(row[row > 0], q_level)
        dr = d[r]
        pos = dr[dr > 0]
        if pos.size:
            ons[r] = dr >= np.percentile(pos, q_onset)
    return lev, ons


def contain(P, M, shift):
    """Fraction of P's marks that land on M, with P shifted by `shift` frames."""
    Ps = np.roll(P, shift, axis=1)
    tot = int(Ps.sum())
    return (float((Ps & M).sum()) / tot) if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--max-shift-ms", type=float, default=160.0)
    args = ap.parse_args()

    sess = (TEMP / args.session if args.session else
            max((d for d in TEMP.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime))
    print(f"session {sess.name}")

    from app.services.piano_projection import _activations, FPS, LOW
    from app.services.melody_decode import VOCAL_HI as MHI, BASS_LO as BLO

    H = _activations(sess / "components/harmonic.wav", BLO, MHI)
    if H is None:
        print("harmonic.wav missing")
        return
    n_pitch, n_fr = H.shape
    tf = json.loads((sess / "transformed.json").read_text())
    notes = tf["transformed_notes"] if isinstance(tf, dict) else tf
    P = piano_onset_map(notes, LOW, n_pitch, n_fr, FPS)
    print(f"{len(notes)} arrangement notes, {int(P.sum())} onset frames, "
          f"song map {n_pitch}x{n_fr} @ {FPS:g} fps\n")

    step = max(1, int(0.01 * FPS))
    shifts = np.arange(-int(args.max_shift_ms / 1000 * FPS),
                       int(args.max_shift_ms / 1000 * FPS) + 1, step)

    for qlev, qons in ((60, 60), (75, 75), (40, 40)):
        lev, ons = song_masks(H, qlev, qons)
        cl = np.array([contain(P, lev, int(s)) for s in shifts])
        co = np.array([contain(P, ons, int(s)) for s in shifts])
        # chance: same notes, alignment destroyed
        rng = np.random.default_rng(0)
        ch_l, ch_o = [], []
        for _ in range(5):
            k = int(rng.integers(int(2 * FPS), n_fr - int(2 * FPS)))
            ch_l.append(contain(P, lev, k))
            ch_o.append(contain(P, ons, k))
        bl, bo = float(np.mean(ch_l)), float(np.mean(ch_o))
        pl = shifts[int(np.nanargmax(cl))] / FPS * 1000
        po = shifts[int(np.nanargmax(co))] / FPS * 1000
        print(f"threshold q={qlev}  (song mask covers "
              f"{100*lev.mean():.1f}% level / {100*ons.mean():.1f}% onset cells)")
        print(f"   LEVEL  peak {np.nanmax(cl)*100:5.1f}% at {pl:+.0f} ms   "
              f"at 0: {cl[len(cl)//2]*100:5.1f}%   chance {bl*100:5.1f}%")
        print(f"   ONSET  peak {np.nanmax(co)*100:5.1f}% at {po:+.0f} ms   "
              f"at 0: {co[len(co)//2]*100:5.1f}%   chance {bo*100:5.1f}%\n")

    # ── per ROLE ────────────────────────────────────────────────────────────
    #
    # Pooling the two voices can hide the failure that matters most here.  If
    # melody peaks at +20 ms and bass at -20 ms, the pooled curve still peaks
    # at 0 while the parts sit 40 ms APART FROM EACH OTHER — which is what
    # "treble and bass not in sync" actually is.  Their difference, not their
    # individual offsets, is the number to read.
    lev, ons = song_masks(H, 60, 60)
    peaks = {}
    print("PER ROLE (onset containment)")
    for role in ("melody", "bass"):
        sub = [n for n in notes if n.get("role") == role]
        if not sub:
            continue
        Pr = piano_onset_map(sub, LOW, n_pitch, n_fr, FPS)
        sc = np.array([contain(Pr, ons, int(s)) for s in shifts])
        pk = shifts[int(np.nanargmax(sc))] / FPS * 1000
        peaks[role] = pk
        # width: how far you can slide before losing a tenth of the peak
        thr = np.nanmax(sc) * 0.9
        wide = shifts[np.where(sc >= thr)[0]] / FPS * 1000
        print(f"   {role:<7} n={len(sub):<5} peak {np.nanmax(sc)*100:5.1f}% at "
              f"{pk:+.0f} ms   >=90% of peak over {wide.min():+.0f}..{wide.max():+.0f} ms")
    if len(peaks) == 2:
        print(f"   melody - bass = {peaks['melody'] - peaks['bass']:+.0f} ms "
              f"(0 means the two voices agree)\n")

    # Per BAR, per role: does the disagreement wander within the piece?
    downs_meta = json.loads((sess / "components.json").read_text())
    _hop = float(downs_meta.get("hop_length", 512))
    _msr = float(downs_meta.get("sr", 22050))
    _downs = np.asarray(downs_meta.get("downbeat_frames", []), float) * _hop / _msr
    if len(_downs) >= 8:
        per = {}
        for role in ("melody", "bass"):
            sub = [n for n in notes if n.get("role") == role]
            Pr = piano_onset_map(sub, LOW, n_pitch, n_fr, FPS)
            vals = []
            for a, b in zip(_downs[:-1], _downs[1:]):
                fa, fb = int(a * FPS), int(b * FPS)
                if fb - fa < 8 or not Pr[:, fa:fb].any():
                    vals.append(np.nan)
                    continue
                sc = [contain(Pr[:, fa:fb], ons[:, fa:fb], int(s)) for s in shifts]
                vals.append(np.nan if np.all(np.isnan(sc))
                            else shifts[int(np.nanargmax(sc))] / FPS * 1000)
            per[role] = np.array(vals, dtype=float)
        d = per["melody"] - per["bass"]
        d = d[~np.isnan(d)]
        if len(d):
            print(f"PER BAR, melody minus bass ({len(d)} bars)")
            print(f"   median {np.median(d):+.0f} ms   IQR {np.percentile(d,25):+.0f}"
                  f"..{np.percentile(d,75):+.0f}   |diff|>40ms in "
                  f"{100*np.mean(np.abs(d) > 40):.1f}% of bars")
            print("   a median near 0 with a WIDE spread is the two parts drifting")
            print("   against each other bar to bar — inconsistent, not offset.\n")

    # ── per bar ─────────────────────────────────────────────────────────────
    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    downs = np.asarray(meta.get("downbeat_frames", []), float) * hop / msr
    if len(downs) < 8:
        print("too few downbeats for the per-bar sweep")
        return
    lev, ons = song_masks(H, 60, 60)
    best = []
    for a, b in zip(downs[:-1], downs[1:]):
        fa, fb = int(a * FPS), int(b * FPS)
        if fb - fa < 8:
            continue
        sub = P[:, fa:fb]
        if not sub.any():
            continue
        sc = [contain(sub, ons[:, fa:fb], int(s)) for s in shifts]
        if np.all(np.isnan(sc)):
            continue
        best.append(shifts[int(np.nanargmax(sc))] / FPS * 1000)
    best = np.array(best)
    if not len(best):
        print("no usable bars")
        return
    idx = np.arange(len(best))
    r = float(np.corrcoef(idx, best)[0, 1]) if len(best) > 2 else float("nan")
    print(f"PER BAR ({len(best)} bars): best shift median {np.median(best):+.0f} ms, "
          f"IQR {np.percentile(best,25):+.0f}..{np.percentile(best,75):+.0f}")
    print(f"   correlation with bar index r = {r:+.3f}")
    print("   drift (|r| high) = the beat grid; scatter (|r| ~ 0, wide IQR) =")
    print("   slot choice.  Both together means both.")


if __name__ == "__main__":
    main()
