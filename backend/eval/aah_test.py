"""Can the arranger read a melody sung as a neutral "aah"?

Synthesises POP909 MELODY lines as one fixed vowel voice (no words, no consonants, one timbre) with a short
re-articulation dip at every note start, transcribes them with an arranger checkpoint at the app's bars, and scores
against the written notes.  Writes, per song, aah.wav and a stereo A/B (left = aah, right = transcription as sines).

    python eval/aah_test.py --ckpt app/ml/spectral/checkpoints/base20m/best.pt --out <dir> [--songs 001,002 ...]
"""
import argparse
import pathlib
import sys

import mir_eval
import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.services import spectral_transcription as ST  # noqa: E402

SR = 24000
FLAT = None   # --flat: every note at one loudness
VIB, SCOOP = 0.2, 0.3   # vibrato depth and start scoop, semitones (eval/aah_fill_test.py varies them)
FORMANTS = [(730, 90), (1090, 110), (2440, 160), (3400, 250)]   # /a/: centre Hz, bandwidth Hz
FORMANT_GAIN = [1.0, 0.5, 0.25, 0.1]


def envelope(f):
    """Vowel spectral envelope at frequencies f (Hz)."""
    g = np.zeros_like(f)
    for (c, b), a in zip(FORMANTS, FORMANT_GAIN):
        g += a / (1 + ((f - c) / b) ** 2)
    return g + 0.02


def aah(notes, dur, seed=0):
    """Neutral sung vowel: harmonic source shaped by /a/ formants, gentle vibrato, re-articulation dips."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR) + SR
    t = np.arange(n) / SR
    f0 = np.zeros(n)
    amp = np.zeros(n)
    for nt in notes:
        a, b = int(nt.start * SR), min(n, int(nt.end * SR))
        if b <= a:
            continue
        seg = np.arange(b - a) / SR
        hz = pretty_midi.note_number_to_hz(nt.pitch)
        scoop = -SCOOP * np.exp(-seg / 0.03)                              # starts ~30 cents flat, settles in ~50 ms
        vib = VIB * np.sin(2 * np.pi * 5.5 * seg) * np.clip((seg - 0.2) / 0.2, 0, 1)   # vibrato after 200 ms
        f0[a:b] = hz * 2 ** ((scoop + vib) / 12)
        lvl = (nt.velocity / 127) ** 1.5 if FLAT is None else FLAT
        att = np.clip(seg / 0.025, 0, 1)                                # 25 ms attack from silence-ish
        rel = np.clip((seg[-1] - seg) / 0.03, 0, 1)                     # 30 ms release
        amp[a:b] = np.maximum(amp[a:b], lvl * (0.25 + 0.75 * att) * np.maximum(rel, 0.25 * (b - a > 0)))
    # touching notes: the dip happens because each note restarts its attack from 0.25
    on = f0 > 0
    f0s = np.where(on, f0, 0)
    phase = 2 * np.pi * np.cumsum(f0s) / SR
    y = np.zeros(n)
    for k in range(1, 40):
        fk = k * f0s
        g = np.where((fk > 0) & (fk < SR / 2 - 500), envelope(fk) / k ** 0.6, 0)
        y += g * np.sin(k * phase)
    y += 0.003 * rng.standard_normal(n) * on                           # a little breath
    y *= amp
    return (0.5 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def pluck(notes, n, gain):
    """Known added part: a soft plucked harmonic tone (decaying partials), level `gain` relative to the voice."""
    y = np.zeros(n)
    for nt in notes:
        a = int(nt.start * SR)
        b = min(n, a + int(min(nt.end - nt.start + 0.3, 2.0) * SR))
        if b <= a:
            continue
        s = np.arange(b - a) / SR
        hz = pretty_midi.note_number_to_hz(nt.pitch)
        w = sum(np.sin(2 * np.pi * k * hz * s) / k ** 1.2 * np.exp(-s * (2 + k)) for k in range(1, 12) if k * hz < SR / 2)
        y[a:b] += (nt.velocity / 127) * w * np.clip(s / 0.004, 0, 1)
    return gain * y


def sines(notes, n):
    y = np.zeros(n)
    for nt in notes:
        a, b = int(nt["onset"] * SR), min(n, int(nt["offset"] * SR))
        s = np.arange(b - a) / SR
        y[a:b] += 0.2 * np.sin(2 * np.pi * pretty_midi.note_number_to_hz(nt["pitch"]) * s) * np.exp(-s * 2)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--songs", default="001,014,052,128,247,333,410,505,688,812")
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--flat", type=float, default=None)
    ap.add_argument("--accomp", type=float, default=0.0,
                    help="add the song's PIANO part (voiced below the melody) at this level, then remove its notes")
    a = ap.parse_args()
    global FLAT
    FLAT = a.flat
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(__file__).resolve().parents[2] / "data" / "POP909"
    rows = []
    miss = {k: [0, 0] for k in ('repeat', 'short', 'other')}
    for s in a.songs.split(","):
        pm = pretty_midi.PrettyMIDI(str(root / s / f"{s}.mid"))
        mel = next(i for i in pm.instruments if i.name.upper().startswith("MELODY"))
        notes = [pretty_midi.Note(n.velocity, n.pitch, n.start - a.start, n.end - a.start)
                 for n in mel.notes if a.start <= n.start < a.start + a.secs]
        if not notes:
            continue
        y = aah(notes, a.secs)
        added = []
        if a.accomp:
            pia = next(i for i in pm.instruments if i.name.upper().startswith("PIANO"))
            lo = min(n.pitch for n in notes)
            for n in pia.notes:
                if a.start <= n.start < a.start + a.secs:
                    p = n.pitch
                    while p > lo - 2:                    # keep the added part below the melody's range
                        p -= 12
                    if p >= 28:
                        added.append(pretty_midi.Note(n.velocity, p, n.start - a.start, n.end - a.start))
            acc = pluck(added, len(y), 1.0)
            y = y + a.accomp * acc / (np.abs(acc).max() + 1e-9) * np.abs(y).max() / 0.5 * 0.5
            y = (0.5 * y / np.abs(y).max()).astype(np.float32)
        wav = out / f"{s}_aah.wav"
        sf.write(wav, y, SR)
        got = ST.transcribe(wav, onset_thr=a.thr, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
        if added:                                        # remove exactly what we added: same key, start within 50 ms
            got = [g for g in got if not any(g["pitch"] == x.pitch and abs(g["onset"] - x.start) < 0.05 for x in added)]
            got = [g for g in got if g["pitch"] >= min(n.pitch for n in notes) - 2]   # and anything in the added range
        ref_i = np.array([[n.start, n.end] for n in notes])
        ref_p = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
        est_i = np.array([[n["onset"], n["offset"]] for n in got]).reshape(-1, 2)
        est_p = np.array([pretty_midi.note_number_to_hz(n["pitch"]) for n in got])
        P, R, F, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, offset_ratio=None)
        # pitch class (octave errors forgiven) and the octave of what was found
        pc = lambda p: 440 * 2 ** (((12 * np.log2(p / 440)) % 12) / 12)
        _, _, Fpc, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, pc(ref_p), est_i, pc(est_p) if len(est_p) else est_p, offset_ratio=None)
        # which written notes were missed: a repeat of the same pitch right after the last note, or a short one
        m = mir_eval.transcription.match_notes(ref_i, ref_p, est_i, est_p, offset_ratio=None)
        hit = {i for i, _ in m}
        for i, nt in enumerate(notes):
            prev = notes[i - 1] if i else None
            kind = ("repeat" if prev and prev.pitch == nt.pitch and nt.start - prev.end < 0.05 else
                    "short" if nt.end - nt.start < 0.12 else "other")
            miss[kind][0] += 1
            miss[kind][1] += i not in hit
        rows.append((s, len(notes), len(got), P, R, F, Fpc))
        print(f"{s}: ref {len(notes):3d} got {len(got):3d}  P {P:.3f} R {R:.3f} F1 {F:.3f}  pitch-class F1 {Fpc:.3f}",
              flush=True)
        ab = np.stack([y, sines(got, len(y)).astype(np.float32)], 1)
        sf.write(out / f"{s}_ab.wav", ab, SR)
    r = np.array([x[1:] for x in rows], float)
    print(f"ALL: ref {int(r[:, 0].sum())} got {int(r[:, 1].sum())}  mean P {r[:, 2].mean():.3f} "
          f"R {r[:, 3].mean():.3f} F1 {r[:, 4].mean():.3f}  pitch-class F1 {r[:, 5].mean():.3f}")
    for k, (n, mi) in miss.items():
        print(f"  {k:7s} notes {n:4d}  missed {mi:4d} ({mi / max(n, 1):.0%})")


if __name__ == "__main__":
    main()
