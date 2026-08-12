"""
Test 1 — can an AUDIO query find a SYMBOLIC corpus unit?

The retrieval corpus is 5,490 symbolic units: notes and chords, no audio.
Queries from songs will be audio.  So the index key has to be computable
BOTH ways — analytically from notes on the corpus side, from a signal on the
query side — and the two have to land in the same space.

featurizer.py already warns about this: "If the two sides ever compute
features [differently] ... that symmetry is load-bearing."  Here the two
sides are not merely different code, they are different DOMAINS.  Analytic
chroma from clean notes is pristine; audio chroma carries harmonics,
reverb tails and noise.  If that gap is wide, retrieval returns nonsense no
matter how good everything downstream is.

The experiment, which needs no new data
----------------------------------------
  1. compute chroma + onset ANALYTICALLY from each unit's notes  -> index
  2. render those same units to audio through our own sampler
  3. compute chroma + onset from the AUDIO                        -> query
  4. ask: does each rendered unit retrieve ITSELF?

Self-retrieval rank near 1 means the two sides are compatible.  Rank near
random means the domain gap kills the design before a song is ever involved.

This is deliberately the easiest possible version of the real task — same
instrument, same notes, no other instruments, no drums.  If it fails HERE it
cannot work on a pop mix, so it is a lower bound and a fast one.

Note the render goes through the shipped chain, so it carries our reverb and
tone shaping.  That is kept on purpose: real queries are real recordings with
real ambience, and a test on dry audio would flatter the design.

RESULT (n=120, 4/4, 2026-08-11) — harmony crosses, rhythm does not
------------------------------------------------------------------
As proposed (chroma 16/bar + onset envelope):  median rank 36.5, r@1 8.3%
against a random median of 60.  Above chance, useless in practice.

Ablated, the two blocks behave completely differently:

    onset envelope only              median 66.0    r@1  0.0%   <- CHANCE
    chroma, 16 frames/bar            median 13.0    r@1 12.5%
    chroma,  8 frames/bar            median 12.5    r@1 12.5%
    chroma,  4 frames/bar            median 13.5    r@1 14.2%
    chroma,  2 frames/bar            median  7.5    r@1 31.7%
    chroma,  1 frame /bar            median  1.0    r@1 60.8%   <- WORKS

The onset envelope carries literally zero transferable information — 0/120
correct, median rank worse than random.  And every added unit of time
resolution in the chroma makes retrieval worse, monotonically, from rank 1 to
rank 13.  Both point at the same cause: sustain and room smear energy across
the bar, so WHEN a note is struck does not survive the trip to audio, while
WHICH pitch classes are present survives almost perfectly.

The consequence for the design is specific.  Audio can supply the harmonic
key and (via madmom) the bar grid; it cannot supply the figuration.  So the
rhythmic half of any retrieval key has to come from the symbolic side we
already produce — the transcription — not from the signal.  A retrieval
system keyed on audio rhythm was never going to work, and this is the
cheapest place to have found that out.

Caveat on the 60.8%: this is the easiest possible case — our own renders of
the corpus's own notes, one instrument, no drums, no other parts, and only
120 candidates.  A real pop mix is strictly harder on every axis.  Read it as
an upper bound that clears the gate, not as expected performance.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

CORPUS = ROOT / "backend/retrieval/corpus/units.jsonl"
SR = 22050
BPM = 100.0
FRAMES = 16          # 16th-note grid across one bar — fine enough for
                     # figuration, coarse enough to survive audio smearing
N_PC = 12


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def analytic_features(unit) -> np.ndarray:
    """chroma (12 x FRAMES) + onset (FRAMES) straight from the notes."""
    bpb = float(unit.get("beats_per_bar", 4))
    chroma = np.zeros((N_PC, FRAMES), dtype=float)
    onset = np.zeros(FRAMES, dtype=float)
    for n in unit["notes"]:
        b = float(n["beat"]) / bpb          # 0..1 through the bar
        d = float(n["dur"]) / bpb
        v = float(n.get("vel", 80)) / 127.0
        pc = int(n["pitch"]) % 12
        f0 = int(np.clip(b * FRAMES, 0, FRAMES - 1))
        f1 = int(np.clip(np.ceil((b + d) * FRAMES), f0 + 1, FRAMES))
        chroma[pc, f0:f1] += v
        onset[f0] += v
    return np.concatenate([_l2(chroma.ravel()), _l2(onset)])


def audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """The same two blocks, from a signal."""
    import librosa
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    O = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    # Resample both onto the same FRAMES grid the analytic side uses.
    def grid(x, n_out):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        T = x.shape[1]
        if T == 0:
            return np.zeros((x.shape[0], n_out))
        idx = np.linspace(0, T, n_out + 1).astype(int)
        return np.stack([x[:, idx[i]:max(idx[i] + 1, idx[i + 1])].mean(axis=1)
                         for i in range(n_out)], axis=1)
    chroma = grid(C, FRAMES)
    onset = grid(O, FRAMES).ravel()
    return np.concatenate([_l2(chroma.ravel()), _l2(onset)])


def render_unit(unit) -> np.ndarray:
    from app.services.sampler import render_notes
    spb = 60.0 / BPM
    notes = [{"pitch": int(n["pitch"]),
              "onset": float(n["beat"]) * spb,
              "offset": (float(n["beat"]) + max(0.05, float(n["dur"]))) * spb,
              "velocity": int(n.get("vel", 80))}
             for n in unit["notes"]]
    if not notes:
        return np.zeros(int(SR * 0.5), dtype=np.float32)
    return np.asarray(render_notes(notes, sr=SR), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="units to test")
    ap.add_argument("--meter", default="4/4")
    args = ap.parse_args()

    units = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            # One meter keeps the frame grid meaningful across units.
            if u.get("meter") == args.meter and len(u.get("notes", [])) >= 3:
                units.append(u)
    rng = np.random.default_rng(0)
    if len(units) > args.n:
        units = [units[i] for i in rng.choice(len(units), args.n, replace=False)]
    print(f"units: {len(units)}  meter {args.meter}  grid {FRAMES}/bar  @{BPM:.0f}bpm")

    index = np.stack([analytic_features(u) for u in units])
    print(f"index: {index.shape}  (analytic, from notes)")

    queries, kept = [], []
    for i, u in enumerate(units):
        try:
            y = render_unit(u)
            if y.size < SR * 0.2:
                continue
            queries.append(audio_features(y, SR))
            kept.append(i)
        except Exception as e:                                  # noqa: BLE001
            print(f"  render failed on {u['id']}: {type(e).__name__}")
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{len(units)}", flush=True)
    Q = np.stack(queries)
    print(f"query: {Q.shape}  (audio, through the shipped render chain)\n")

    # cosine similarity — both sides are already block-L2-normalised
    sim = Q @ index.T
    ranks = []
    for row, true_i in zip(sim, kept):
        order = np.argsort(-row)
        ranks.append(int(np.where(order == true_i)[0][0]) + 1)
    ranks = np.array(ranks)
    N = len(units)

    print(f"{'self-retrieval rank':<26} median {np.median(ranks):6.1f}   mean {ranks.mean():7.1f}")
    print(f"{'recall@1':<26} {100 * np.mean(ranks == 1):6.1f}%")
    print(f"{'recall@5':<26} {100 * np.mean(ranks <= 5):6.1f}%")
    print(f"{'recall@10':<26} {100 * np.mean(ranks <= 10):6.1f}%")
    print(f"\nrandom baseline: median rank would be ~{N // 2}, recall@1 ~{100 / N:.1f}%")
    print("A design that works should sit at recall@1 well above chance and a")
    print("median rank in the low single digits.  Anything near the random line")
    print("means the audio and symbolic sides are not in the same space.")


if __name__ == "__main__":
    main()
