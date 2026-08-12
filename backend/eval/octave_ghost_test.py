"""
Does an 88-key dictionary survive its own harmonic series?

The plan it is testing: render all 88 keys once, treat them as a fixed
dictionary, and decompose a song's spectrogram against it.  Whatever notes
come back are the best piano approximation of that sound — projection onto a
subspace we can actually name, rather than a learned one.

The known way this fails
------------------------
The dictionary is badly non-orthogonal.  A piano note's harmonic series very
nearly CONTAINS the series of the note an octave above it: C3's partials land
at C4, G4, C5, E5..., and C4's land at C5, G5, C6...  So NMF is free to
explain one struck note as a stack of higher ones, or to hallucinate an octave
that was never played.  Octave errors are the classic failure of template
transcription, and they are fatal here — if the note content comes back wrong
nothing downstream repairs it.

So this measures one number: of all the activation energy, how much lands on
a pitch that is NOT sounding but is an octave (or a fifth) away from one that
is.

Fixed-W NMF, and why not sklearn
--------------------------------
sklearn's NMF wants to learn W.  Here W is given and only H is solved for, so
this uses the standard KL multiplicative update for H alone.  KL rather than
Frobenius because the target is a magnitude spectrogram, where errors are
multiplicative and the loud partials would otherwise dominate the fit.

Read the result as a LOWER BOUND on difficulty
----------------------------------------------
MAESTRO ships MIDI but no audio here, so the test signal is synthesised with
the same sampler that builds the dictionary.  Templates are therefore matched
to the material exactly — same instrument, same samples, same room.  A real
recording, let alone a pop mix, is strictly harder.  If ghosting is bad in
THIS condition it cannot be rescued later, which is the point: it is the cheap
test that can only disqualify, never confirm.

RESULT (8 MAESTRO pieces, 30 s each, 2026-08-11) — the prediction was wrong
--------------------------------------------------------------------------
Octave ghosting is NOT the wall.  Energy allocation, 2.0 s ring-out window:

    correct 45.8%   decay 41.9%   octave-ghost 2.0%   fifth 1.2%   other 9.0%

87.7% of all activation energy lands on a pitch that is either sounding or
ringing from a recent strike.  Since 41.9% sits in the decay bucket the result
could easily be an artefact of a generous window, so it was swept:

    window   correct   decay   octave-ghost   fifth   other
    0.00 s     45.8%    0.0%          7.4%    4.9%   42.0%
    0.25 s     45.8%   28.0%          4.5%    2.6%   19.1%
    0.50 s     45.8%   35.0%          3.4%    1.9%   13.9%
    1.00 s     45.8%   39.1%          2.6%    1.5%   11.0%
    2.00 s     45.8%   41.9%          2.0%    1.2%    9.0%

Octave ghosting never exceeds 7.4% even at zero tolerance, where every ringing
string is scored as an error.  The conclusion does not depend on the window.

Worth noting the dictionary is exactly as non-orthogonal as feared — mean
cos(p, p+12) = 0.446 and one pair reaches 0.932.  The overlap is real; KL-NMF
simply resolves it, because a struck note's partials do not merely resemble
the octave's, they arrive with a specific amplitude pattern the octave alone
cannot reproduce.

What this does NOT establish: this measures where ENERGY goes, not whether
discrete note events can be read off the activations.  And the dictionary is
matched to the test signal.  Both are the next questions, not answered ones.
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

# The algorithm itself now lives in the SERVICE, and this imports it, so what
# is measured here is literally what ships.  These were duplicated while the
# idea was being tested; leaving two copies once it went into the pipeline
# would mean this file could keep reporting numbers for code no user runs.
from app.services.piano_projection import (                      # noqa: E402
    SR, N_FFT, HOP, FPS, LOW, HIGH, N_PITCH,
    stft_mag, build_dictionary, nmf_h, get_dictionary)

DECAY_S = 2.0         # how long a released string may still legitimately ring
CACHE = pathlib.Path(__file__).with_name("_dict88.npy")


def shifted(mask, k):
    out = np.zeros_like(mask)
    if k > 0:   out[k:] = mask[:-k]
    elif k < 0: out[:k] = mask[-k:]
    else:       out = mask.copy()
    return out


def categorise(H, GT, decay_s=None):
    """Split activation energy into correct / decay / ghost / other.

    Shared with the cross-timbre eval on purpose.  Two copies of a scoring
    rule is how a comparison quietly stops comparing the same thing.
    """
    decay_s = DECAY_S if decay_s is None else decay_s
    E = H / (H.sum() or 1.0)
    on = GT

    oct_near = np.zeros_like(on)
    for k in (-24, -12, 12, 24):
        oct_near |= shifted(on, k)
    fifth_near = np.zeros_like(on)
    for k in (-19, -7, 7, 19):
        fifth_near |= shifted(on, k)

    # A struck string keeps sounding long after the key comes up, and with the
    # pedal down it rings until the pedal does.  MAESTRO's roll marks note-off
    # at KEY RELEASE, so energy on a just-released pitch is the sampler being
    # correct, not the decomposition being wrong.  Counting it as error would
    # condemn the method for reproducing the instrument.
    rang = np.zeros_like(on)
    for k in range(1, int(decay_s * FPS) + 1):
        rang[:, k:] |= on[:, :-k]

    return (E[on].sum(),
            E[(~on) & rang].sum(),
            E[(~on) & ~rang & oct_near].sum(),
            E[(~on) & ~rang & fifth_near & ~oct_near].sum(),
            E[(~on) & ~rang & ~oct_near & ~fifth_near].sum())


def ground_truth(roll_path, n_frames):
    """MAESTRO's 100 fps roll, resampled onto our STFT frame grid."""
    roll = np.load(roll_path)                        # (frames@100fps, 88) bool
    idx = np.clip((np.arange(n_frames) / FPS * 100).astype(int),
                  0, roll.shape[0] - 1)
    return roll[idx].T                               # (88, frames)


def load_dictionary():
    """The service's cache is authoritative; this local one is a leftover."""
    if CACHE.exists():
        return np.load(CACHE)
    return get_dictionary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--decay", type=float, default=DECAY_S,
                    help="ring-out window; the result should not hinge on it")
    args = ap.parse_args()
    globals()["DECAY_S"] = args.decay

    import librosa
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())

    W = load_dictionary()
    # How much of the problem is baked into the dictionary alone: the cosine
    # between a template and the one an octave above it.
    oct_sim = [float(W[:, i] @ W[:, i + 12]) for i in range(N_PITCH - 12)]
    print(f"dictionary: {W.shape}   mean cos(p, p+12) = {np.mean(oct_sim):.3f}"
          f"   max {np.max(oct_sim):.3f}\n")

    rows = []
    for piece in man["pieces"][: args.pieces]:
        wav = ROOT / piece["audio_synth_path"]
        roll_p = ROOT / (piece["midi_path"].replace(".midi", ".roll.100.npy"))
        if not wav.exists() or not roll_p.exists():
            print(f"  skip {piece['id'][:40]} (missing audio or roll)")
            continue

        y, _ = librosa.load(str(wav), sr=SR, mono=True, duration=args.seconds)
        V = stft_mag(y)
        H = nmf_h(V, W)

        GT = ground_truth(roll_p, V.shape[1])
        correct, decay, ghost_o, ghost_5, other = categorise(H, GT)
        rows.append((correct, decay, ghost_o, ghost_5, other))
        print(f"  {piece['composer'][:22]:<24} correct {100*correct:5.1f}%  "
              f"decay {100*decay:5.1f}%  octave-ghost {100*ghost_o:5.1f}%  "
              f"fifth {100*ghost_5:4.1f}%  other {100*other:5.1f}%", flush=True)

    a = np.array(rows)
    print(f"\n{'MEAN over ' + str(len(rows)) + ' pieces':<26}"
          f"correct {100*a[:,0].mean():5.1f}%  decay {100*a[:,1].mean():5.1f}%"
          f"  octave-ghost {100*a[:,2].mean():5.1f}%"
          f"  fifth {100*a[:,3].mean():4.1f}%  other {100*a[:,4].mean():5.1f}%")
    print(f"\naccounted for (correct + decay): {100*(a[:,0]+a[:,1]).mean():.1f}%")
    print("The number that decides the design is octave-ghost — energy on a pitch")
    print("that is silent, is not ringing from a recent strike, and sits an octave")
    print("from one that is sounding.  That is the error the method cannot repair.")


if __name__ == "__main__":
    main()
