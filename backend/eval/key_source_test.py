"""
Can a chroma profile establish the key as well as our decoded notes?

The constrained-dictionary idea needs the key BEFORE the decomposition runs —
remove the out-of-scale templates and the energy that would have been
misattributed to them has to land on a legal key instead.  But we currently
derive the key AFTER, from decoded notes, which is a chicken-and-egg problem.
Chroma is the cheap way out: it needs no NMF at all.

The question is whether it is good enough, and it has to be, because a
constrained dictionary is unforgiving.  Filtering with a wrong key merely
deletes some notes — yesterday it deleted the tune's own leading tone.
CONSTRAINING with a wrong key makes those pitches unreachable, with no recovery
downstream.

POP909 ships human key labels, so both methods are scored against truth rather
than against each other.  Three are compared:

    chroma      Krumhansl-Schmuckler correlation on audio chroma.  No NMF.
    pitchmap    the same profile, but summed from the 88-row activation matrix
                by pitch class.  Free once the NMF has run, so it cannot solve
                the ordering problem — included to separate "chroma is weak"
                from "audio is weak".
    decoded     what we do now: project, decode melody and bass, detect_scale.

Relative-key confusion (C major vs A minor) is counted separately.  Those share
all seven pitch classes, so for the purpose of masking a dictionary the mistake
costs nothing — the same templates survive either way.

RESULT (40 POP909 songs, 30 s excerpts, 2026-08-13)
---------------------------------------------------
    method       songs    exact   root only   +relative
    chroma          40    65.0%       67.5%       80.0%
    pitchmap        40    75.0%       77.5%       85.0%
    decoded (was)   40    27.5%       27.5%       72.5%

Two things were wrong with `decoded`, and only one was the note source.  It
also scored by dot product against a BINARY in-scale mask, and the diatonic
modes in _SCALES are rotations of one seven-note set — so every rotation scored
identically on the true collection and the root fell out arbitrarily.  That is
the 27.5% / 72.5% gap: it could find the notes of the key but not its tonic.
Both are fixed; the shipped path now uses key_from_activations + the graded K-S
profiles, and this file imports that code rather than keeping a copy.

The premise this test was built to serve turned out not to hold.  It was
written to decide whether the key could be known BEFORE the NMF, so the
dictionary could be masked.  scale_penalty_test then showed masking and
penalising both HURT, even given a perfect key — so nothing needs the key
early, and the ordering constraint that made chroma attractive (no NMF
required) is void.  Hence pitchmap, which is better and is free once the
decomposition has run.  The key now only picks among notes we already have.
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

POP909 = ROOT / "data/POP909"
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# The K-S profiles and the correlation now live in the SERVICE, because the
# pitchmap method won this comparison and shipped.  Keeping a second copy here
# is how an eval quietly stops measuring the code that runs.
from app.services.nmf_voice_leading import (                     # noqa: E402
    key_from_pitch_classes as key_from_chroma)


def parse_key(tok: str):
    if not tok or tok[0] not in _PC:
        return None
    pc = _PC[tok[0]]
    i = 1
    while i < len(tok) and tok[i] in "#b":
        pc += 1 if tok[i] == "#" else -1
        i += 1
    mode = "minor" if "min" in tok.lower() else "major"
    return pc % 12, mode


def relative_of(root, mode):
    return ((root + 9) % 12, "minor") if mode == "major" else ((root + 3) % 12, "major")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    import librosa, pretty_midi
    from app.services.sampler import render_notes
    from app.services.piano_projection import (SR, stft_mag, nmf_h,
                                               get_dictionary, LOW,
                                               key_from_activations)
    from app.services import melody_decode as MD
    from app.services.nmf_voice_leading import detect_scale

    W = get_dictionary()
    res = {k: {"exact": 0, "root": 0, "rel": 0, "n": 0}
           for k in ("chroma", "pitchmap", "decoded")}

    for folder in sorted([d for d in POP909.iterdir() if d.is_dir()])[: args.n]:
        mid, kf, bf = (folder / f"{folder.name}.mid",
                       folder / "key_audio.txt", folder / "beat_midi.txt")
        if not (mid.exists() and kf.exists() and bf.exists()):
            continue
        truth = None
        for line in kf.read_text().splitlines():
            p = line.split()
            if len(p) >= 3 and ":" in p[2]:
                truth = parse_key(p[2].split(":")[0] + ":" + p[2].split(":")[1])
                truth = parse_key(p[2].replace(":", ""))
                break
        if truth is None:
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
            rows = np.loadtxt(bf)
        except Exception:                                       # noqa: BLE001
            continue

        notes = [{"pitch": int(n.pitch), "onset": float(n.start),
                  "offset": float(min(n.end, args.seconds)),
                  "velocity": int(n.velocity)}
                 for i in pm.instruments if not i.is_drum
                 for n in i.notes if n.start < args.seconds]
        if len(notes) < 32:
            continue
        y = np.asarray(render_notes(notes, sr=SR), dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        pk = float(np.abs(y).max())
        if pk < 1e-5:
            continue
        y = y / pk

        beats = rows[rows[:, 1] > 0.5, 0] if rows.ndim == 2 else None
        beats = beats[beats < args.seconds] if beats is not None else None

        cands = {}
        # 1. chroma straight off the audio — no NMF
        C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=512)
        cands["chroma"] = key_from_chroma(C.mean(axis=1))

        # 2. same profile, from the activation matrix by pitch class
        H = nmf_h(stft_mag(y), W)
        cands["pitchmap"] = key_from_activations(H)

        # 3. what we do now
        dec = []
        if beats is not None and len(beats) > 4:
            for lo, hi in ((MD.VOCAL_LO, MD.VOCAL_HI), (MD.BASS_LO, MD.BASS_HI)):
                dec += MD.decode(H, LOW, beats, None, 4, lo, hi)
        if len(dec) >= 8:
            r, m, _ = detect_scale(dec)
            cands["decoded"] = (r, m)
        else:
            cands["decoded"] = None

        for k, got in cands.items():
            if got is None:
                continue
            res[k]["n"] += 1
            res[k]["exact"] += int(got == truth)
            res[k]["root"] += int(got[0] == truth[0])
            res[k]["rel"] += int(got == truth or got == relative_of(*truth))

    print(f"POP909, {args.seconds:.0f}s excerpts, scored against human key labels\n")
    print(f"{'method':<12}{'songs':>7}{'exact':>9}{'root only':>12}{'+relative':>12}")
    print("-" * 52)
    for k in ("chroma", "pitchmap", "decoded"):
        d = res[k]
        if not d["n"]:
            continue
        f = lambda x: 100.0 * d[x] / d["n"]
        print(f"{k:<12}{d['n']:>7}{f('exact'):>8.1f}%{f('root'):>11.1f}%{f('rel'):>11.1f}%")
    print("\n'+relative' forgives C major vs A minor.  For masking a dictionary")
    print("that confusion is harmless — both keep the same seven pitch classes.")


if __name__ == "__main__":
    main()
