"""
Does penalising out-of-scale keys beat filtering them afterwards — or masking?

Three ways to use the key, in increasing severity:

    filter    decompose freely, drop out-of-scale NOTES afterwards.  What ships
              today.  Cannot undo a misattribution: the energy already went to
              the wrong key, and removing the note leaves a hole.
    penalty   keep all 88 templates but make out-of-scale ones expensive, so a
              chromatic note needs real evidence to appear.
    mask      remove the out-of-scale templates entirely.  Energy that would
              have gone there MUST land on a legal key instead.

Masking is the strongest and the most dangerous: measured in key_source_test,
even the best estimator gets the key wrong on about one song in seven, and a
wrong mask is unrecoverable because the correct pitches are not in the
dictionary at all.  The penalty is the same idea without the cliff.

Implementation: KL-NMF's multiplicative update divides by (W^T 1).  Adding a
per-pitch constant to that denominator is exactly a weighted L1 prior on the
activations, so the penalty costs nothing and needs no new solver.  Masking is
the same thing with the weight at infinity.

Both an ORACLE key (from ground-truth notes) and an ESTIMATED one (chroma) are
run, so "does the penalty help" is separated from "does key error hurt".

MAESTRO is deliberately a hard case here — classical piano modulates and is
full of chromaticism, so any scale constraint is fighting the material.  A
result that survives on this should be safe on pop.

RESULT (6 MAESTRO pieces, 30 s each, 2026-08-13) — both ideas are dead
---------------------------------------------------------------------
    key        lambda  correct   decay  octave   other  accounted
    oracle       0.00    41.3%   47.4%    2.0%    8.2%      88.7%
    oracle       0.25    41.3%   47.2%    2.2%    8.0%      88.5%
    oracle       0.50    40.7%   46.7%    2.4%    8.8%      87.4%
    oracle       1.00    39.8%   46.0%    2.5%   10.0%      85.8%
    oracle       2.00    38.6%   45.2%    2.6%   11.4%      83.8%
    oracle       5.00    37.5%   44.6%    2.7%   12.8%      82.1%
    oracle       mask    37.1%   44.4%    2.8%   13.3%      81.5%

Monotone decline, no sweet spot: even lambda 0.25 is fractionally negative.
The chroma-key rows (omitted, see git history) sit within half a point of the
oracle rows at every lambda.

The prediction was that out-of-scale templates STEAL energy from in-scale
pitches, so taxing them would push it home.  The `other` column says otherwise
— it GROWS, 8.2% -> 13.3%, as the constraint tightens.  The energy was not
misattributed to out-of-scale neighbours; it was genuinely explained there, and
removing that explanation scatters it somewhere worse.  Chromatic content is
real, and the scale is not a property of the signal.

The corollary matters more than the result: because oracle and chroma track
each other the whole way down, KEY ACCURACY IS IRRELEVANT HERE.  The risk
trade this test was framed around — 85% key accuracy meaning one song in seven
gets an unrecoverable mask — is moot, since the constraint fails with a perfect
key too.

So the scale stays where it is: a POST-HOC filter on decoded notes, where a
wrong call deletes a note instead of corrupting the decomposition.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from octave_ghost_test import (                                  # noqa: E402
    SR, LOW, stft_mag, categorise, ground_truth, load_dictionary)
from key_source_test import key_from_chroma                      # noqa: E402


def nmf_penalised(V, W, penalty=None, iters=150):
    """KL multiplicative updates with a per-pitch L1 weight.

    `penalty` is a length-88 vector added to the update's denominator.  Zero
    reproduces the plain solver exactly.
    """
    eps = 1e-9
    H = np.full((W.shape[1], V.shape[1]), V.mean() / W.shape[1] + eps)
    Wt = W.T
    ones = Wt @ np.ones_like(V)
    if penalty is not None:
        ones = ones + penalty[:, None] * float(ones.mean())
    for _ in range(iters):
        H *= (Wt @ (V / (W @ H + eps))) / (ones + eps)
    return H


def scale_mask(root, mode):
    ivs = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    pcs = {(root + i) % 12 for i in ivs}
    return np.array([(LOW + i) % 12 in pcs for i in range(88)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    import librosa
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W = load_dictionary()

    LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    rows = {("oracle", l): [] for l in LAMBDAS}
    rows.update({("chroma", l): [] for l in LAMBDAS})
    rows[("oracle", "mask")] = []
    rows[("chroma", "mask")] = []
    key_ok = 0
    n = 0

    for piece in man["pieces"][: args.pieces]:
        wav = ROOT / piece["audio_synth_path"]
        roll_p = ROOT / piece["midi_path"].replace(".midi", ".roll.100.npy")
        if not wav.exists() or not roll_p.exists():
            continue
        y, _ = librosa.load(str(wav), sr=SR, mono=True, duration=args.seconds)
        V = stft_mag(y)
        GT = ground_truth(roll_p, V.shape[1])
        n += 1

        # oracle key: the pitch-class histogram of the ground-truth notes
        pcv = np.zeros(12)
        for i in range(GT.shape[0]):
            pcv[(LOW + i) % 12] += GT[i].sum()
        ok = key_from_chroma(pcv)
        # estimated key: chroma off the audio, no NMF
        ck = key_from_chroma(librosa.feature.chroma_cqt(
            y=y, sr=SR, hop_length=512).mean(axis=1))
        if ok and ck and ok == ck:
            key_ok += 1

        for tag, k in (("oracle", ok), ("chroma", ck)):
            if k is None:
                continue
            inscale = scale_mask(*k)
            pen_base = (~inscale).astype(float)
            for l in LAMBDAS:
                rows[(tag, l)].append(
                    categorise(nmf_penalised(V, W, pen_base * l), GT))
            keep = np.where(inscale)[0]
            Hm = np.zeros((88, V.shape[1]))
            Hm[keep] = nmf_penalised(V, W[:, keep])
            rows[(tag, "mask")].append(categorise(Hm, GT))

    print(f"MAESTRO, {n} pieces, {args.seconds:.0f}s each")
    print(f"chroma key agreed with oracle on {key_ok}/{n} pieces\n")
    print(f"{'key':<9}{'lambda':>8}{'correct':>9}{'decay':>8}{'octave':>8}"
          f"{'other':>8}{'accounted':>11}")
    print("-" * 52)
    for tag in ("oracle", "chroma"):
        for l in LAMBDAS + ["mask"]:
            v = rows[(tag, l)]
            if not v:
                continue
            m = np.array(v).mean(axis=0) * 100
            lab = "mask" if l == "mask" else f"{l:.2f}"
            star = "  <- ships" if (tag == "oracle" and l == 0.0) else ""
            print(f"{tag:<9}{lab:>8}{m[0]:>8.1f}%{m[1]:>7.1f}%{m[2]:>7.1f}%"
                  f"{m[4]:>7.1f}%{m[0]+m[1]:>10.1f}%{star}")
        print()
    print("lambda 0 is the current unconstrained solver — the two 0.00 rows are")
    print("identical by construction and act as the control.")


if __name__ == "__main__":
    main()
