"""Are the missed soft notes learnable, or nearly silent?

The error breakdown put 61.5% of missed notes in "soft (velocity < 50)", and
ByteDance misses soft notes at the same rate (recall 0.838 vs 0.835).  The
Disklavier records every key press, including ones that barely sound.  Before
training harder on soft notes, compare MISSED soft notes with FOUND ones on:

  velocity      how soft, exactly
  masking       loudest other note struck within 60 ms, and notes already sounding
  pedal         sustain pedal down at the onset
  model belief  peak onset probability on that key within +-2 frames:
                0.5-0.85 = heard but below threshold (cheap fix),
                < 0.1    = not heard at all

and write a few audio snippets of missed soft notes to listen to.

    python eval/soft_miss_check.py --dir E:/arrangement_transformer/maestro_real
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from real_audio_eval import midi_truth, score       # noqa: E402


def pedal_at(pm):
    ev = sorted((c.time, c.value) for i in pm.instruments for c in i.control_changes if c.number == 64)
    t = np.array([e[0] for e in ev]); v = np.array([e[1] for e in ev])
    return lambda x: bool(len(t)) and (v[np.searchsorted(t, x, "right") - 1] >= 64
                                       if np.searchsorted(t, x, "right") else False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--clips", default="E:/arrangement_transformer/soft_miss_clips")
    ap.add_argument("--n-clips", type=int, default=12)
    args = ap.parse_args()
    import pretty_midi, soundfile as sf, librosa
    from app.services import spectral_transcription as ST
    from app.ml.spectral import frontend as S
    from app.ml.transcriber.decode import decode_notes

    rows = []           # (found, velocity, masker_vel, n_sounding, pedal, peak_prob, wav, onset, pitch)
    for wav in sorted(pathlib.Path(args.dir).glob("*.wav")):
        ref = midi_truth(wav.with_suffix(".midi"))
        ped = pedal_at(pretty_midi.PrettyMIDI(str(wav.with_suffix(".midi"))))
        on, fr, vl = ST.rolls(wav)
        onk = on if on.shape[0] == 88 else on.T            # (88, T)
        est = decode_notes(on, fr, vl, args.thr, 0.5, min_frames=1, min_note_s=0.0,
                           keep_restruck=True)
        *_, pairs = score(ref, est)
        found = {i for i, _ in pairs}
        ons = np.array([r["onset"] for r in ref])
        for i, r in enumerate(ref):
            if r["velocity"] >= 50:
                continue
            t = r["onset"]
            lo, hi = np.searchsorted(ons, t - 0.06), np.searchsorted(ons, t + 0.06)
            mask = max([ref[j]["velocity"] for j in range(lo, hi) if j != i] or [0])
            sounding = sum(1 for j in range(max(0, lo - 400), lo) if ref[j]["offset"] > t)
            f = int(round(t * S.FPS)); k = r["pitch"] - 21
            peak = float(onk[k, max(0, f - 2):f + 3].max()) if f < onk.shape[1] else 0.0
            rows.append((i in found, r["velocity"], mask, sounding, ped(t), peak, wav, t, r["pitch"]))
        print(f"  {wav.stem[:50]}", flush=True)

    F = [r for r in rows if r[0]]; M = [r for r in rows if not r[0]]
    print(f"\nsoft notes (vel < 50): {len(rows)}   found {len(F)}   missed {len(M)}")
    def dist(sel, lab):
        v = np.array([r[1] for r in sel]); m = np.array([r[2] for r in sel])
        s = np.array([r[3] for r in sel]); p = np.array([r[4] for r in sel])
        print(f"  {lab:<7} vel<20 {np.mean(v < 20):5.1%}  20-34 {np.mean((v >= 20) & (v < 35)):5.1%}"
              f"  35-49 {np.mean(v >= 35):5.1%} | median vel {np.median(v):.0f}"
              f" | masker >=70 {np.mean(m >= 70):5.1%}  median masker {np.median(m):.0f}"
              f" | sounding median {np.median(s):.0f} | pedal {np.mean(p):5.1%}")
    dist(F, "found"); dist(M, "missed")

    print("\nrecall by velocity band:")
    for lo, hi in ((0, 20), (20, 30), (30, 40), (40, 50)):
        b = [r for r in rows if lo <= r[1] < hi]
        print(f"   vel {lo:>2}-{hi - 1:<2}  {len(b):>5} notes   recall {np.mean([r[0] for r in b]):.3f}")

    pk = np.array([r[5] for r in M])
    print("\nmodel's peak onset probability on MISSED soft notes:")
    for lo, hi, lab in ((0, .1, "not heard (<0.1)"), (.1, .5, "faint (0.1-0.5)"),
                        (.5, args.thr, f"below thr (0.5-{args.thr})"), (args.thr, 1.01, "above thr*")):
        print(f"   {lab:<22} {np.mean((pk >= lo) & (pk < hi)):6.1%}")
    print("   * peak above threshold but no matched note: decoded at wrong time or merged")
    for t in (0.3, 0.5, 0.7):
        print(f"   recoverable at a thr of {t}: {np.mean(pk >= t):.1%} of missed soft notes")

    out = pathlib.Path(args.clips); out.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    for n, r in enumerate(random.sample(M, min(args.n_clips, len(M)))):
        _, v, mask, snd, pd, peak, wav, t, p = r
        y, sr = librosa.load(str(wav), sr=None, offset=max(0, t - 1.0), duration=2.5, mono=True)
        name = (f"{n:02d}_{pretty_midi.note_number_to_name(p)}_vel{v}_at1.0s_masker{mask}"
                f"_{'pedal' if pd else 'nopedal'}_p{peak:.2f}.wav")
        sf.write(out / name, y, sr)
    print(f"\nwrote {min(args.n_clips, len(M))} snippets to {out} "
          f"(each 2.5 s, the missed note starts at 1.0 s)")


if __name__ == "__main__":
    main()
