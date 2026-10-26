"""Fill a solo voice with the song's own accompaniment, transcribe, then remove what was added.

The user's idea (2026-10-26): the base reads a voice well inside a mix but drops notes when the voice is alone.
So under a solo passage, loop bars of the song's OWN accompaniment taken from elsewhere in the song (beat-cut,
chord-matched), transcribe the filled audio, and delete exactly the notes that were added.

Test on POP909 (no real singer yet): the solo voice is the MELODY of a 40 s span sung as the neutral "aah"
(eval/aah_test.py).  The accompaniment comes only from bars OUTSIDE that span:
  1. melody vs accompaniment is decided by the trained split (app/ml/melody_split.joblib), not POP909's labels
  2. each target bar gets the source bar whose per-beat chords match best (transposed if that matches better)
  3. the source bar's accompaniment notes are time-warped onto the target bar and synthesised (pluck tone)
Removal: a transcribed note is dropped if it matches an added note (same key, start within 50 ms).
"+range" also drops notes below the voice's range as estimated from a solo pass (no oracle).

    python eval/aah_fill_test.py --ckpt app/ml/spectral/checkpoints/base20m/best.pt --out <dir>
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
import eval.aah_test as AT  # noqa: E402
from eval.aah_test import SR, aah, pluck, sines  # noqa: E402
from eval.rhythm_groups import melody_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2] / "data" / "POP909"


def parse_chord(lab):
    if lab == "N" or ":" not in lab:
        return None
    r, q = lab.split(":", 1)
    return pretty_midi.note_name_to_number(r + "0") % 12, q


def bars_and_chords(song):
    beats = np.loadtxt(ROOT / song / "beat_midi.txt")
    t, down = beats[:, 0], beats[:, 2] > 0
    ch = [ln.split("\t") for ln in (ROOT / song / "chord_midi.txt").read_text().splitlines() if ln.strip()]
    ch = [(float(a), float(b), parse_chord(c.strip())) for a, b, c in ch]

    def at(x):
        for a, b, c in ch:
            if a <= x + 1e-3 < b:
                return c
        return None
    d = np.flatnonzero(down)
    bars = []
    for i, j in zip(d[:-1], d[1:]):
        bars.append((t[i], t[j], [at(x) for x in t[i:j]]))
    return bars


def match_score(src, tgt, shift):
    if len(src) != len(tgt):
        return -1
    s = 0
    for a, b in zip(src, tgt):
        if a is None or b is None:
            s += (a is None) == (b is None)
        else:
            s += (a[0] + shift) % 12 == b[0] and a[1] == b[1]
    return s


def mono(notes, jump=0.06, overlap=0.1):
    """One voice at a time: the best chain of notes that don't overlap (by more than `overlap` s), favouring
    loud, high notes and small pitch steps.  Bleed under a held voice note loses to the voice."""
    if not notes:
        return []
    ns = sorted(notes, key=lambda g: g["onset"])
    w = [1 + g["velocity"] / 127 + 0.02 * (g["pitch"] - 60) for g in ns]
    best, prev = list(w), [-1] * len(ns)
    for j in range(len(ns)):
        for i in range(j):
            if ns[j]["onset"] >= ns[i]["offset"] - overlap and ns[j]["onset"] > ns[i]["onset"] + 0.03:
                v = best[i] + w[j] - jump * abs(ns[j]["pitch"] - ns[i]["pitch"])
                if v > best[j]:
                    best[j], prev[j] = v, i
    j = int(np.argmax(best))
    out = []
    while j >= 0:
        out.append(ns[j])
        j = prev[j]
    return out[::-1]


def voice_track(y):
    """Pitch track of the dominant (sung) line in the stem: pYIN at 16 kHz, MIDI per 16 ms frame, nan = unvoiced."""
    import librosa
    z = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000)
    f0, voiced, _ = librosa.pyin(z, fmin=80, fmax=1100, sr=16000, frame_length=1024, hop_length=256)
    return np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)


def voice_like(notes, track, fps=16000 / 256, tol=1.0, need=0.4):
    """Keep notes the sung line actually follows: the track sits within `tol` semitones of the note for at least
    `need` of its first 150 ms."""
    out = []
    for g in notes:
        a = int(g["onset"] * fps)
        b = max(a + 1, int(min(g["offset"], g["onset"] + 0.15) * fps))
        seg = track[a:b]
        if len(seg) and np.mean(np.abs(seg - g["pitch"]) <= tol) >= need:     # nan compares False
            out.append(g)
    return out


def score(ref, got):
    ri = np.array([[n.start, n.end] for n in ref])
    rp = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in ref])
    ei = np.array([[g["onset"], g["offset"]] for g in got]).reshape(-1, 2)
    ep = np.array([pretty_midi.note_number_to_hz(g["pitch"]) for g in got])
    P, R, F, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    return P, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--songs", default="001,014,052,128,247,333,410,505,688,812")
    ap.add_argument("--secs", type=float, default=40.0)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--level", type=float, default=1.0, help="added part's peak level relative to the voice")
    ap.add_argument("--n-songs", type=int, default=0, help="random POP909 songs instead of --songs")
    ap.add_argument("--vib", type=float, default=0.2, help="vibrato depth, semitones")
    ap.add_argument("--scoop", type=float, default=0.3, help="start scoop, semitones")
    ap.add_argument("--jitter", type=float, default=0.0, help="sung start offset sd, s (targets stay written)")
    ap.add_argument("--bleed", type=float, default=None, help="true accompaniment left in the 'solo' stem, dB")
    ap.add_argument("--mono-guide", action="store_true", help="estimate range / protected notes from mono(solo)")
    ap.add_argument("--voice-filter", action="store_true", help="keep only notes the stem's pitch track follows")
    ap.add_argument("--random-bars", action="store_true", help="ignore chords when choosing source bars")
    a = ap.parse_args()
    AT.VIB, AT.SCOOP = a.vib, a.scoop
    if a.n_songs:
        rng = np.random.default_rng(1)
        a.songs = ",".join(f"{i:03d}" for i in sorted(rng.choice(np.arange(1, 910), a.n_songs, replace=False)))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = pathlib.Path(a.ckpt)
    t0, t1 = a.start, a.start + a.secs
    res = {k: [] for k in ("solo", "fill", "fill+range", "below+protect", "solo+mono", "below+protect+mono", "solo+mono+voice", "below+protect+mono+voice")}
    split_acc, exact, collide = [], [], 0
    for s in a.songs.split(","):
        pm = pretty_midi.PrettyMIDI(str(ROOT / s / f"{s}.mid"))
        mel = next(i for i in pm.instruments if i.name.upper().startswith("MELODY")).notes
        pia = next(i for i in pm.instruments if i.name.upper().startswith("PIANO")).notes
        voice = [pretty_midi.Note(n.velocity, n.pitch, n.start - t0, n.end - t0) for n in mel if t0 <= n.start < t1]
        if not voice:
            continue

        # 1. melody / accompaniment split on the whole song outside the span (the trained split, not the labels)
        allnotes = sorted([(n.start, n.end, n.pitch, n.velocity, 1) for n in mel] +
                          [(n.start, n.end, n.pitch, n.velocity, 0) for n in pia], key=lambda x: (x[0], -x[2]))
        arr = np.array(allnotes, float)
        is_mel = melody_split(arr[:, :3])
        split_acc.append((is_mel == arr[:, 4].astype(bool)).mean())
        acc = arr[~is_mel]

        # 2. chord-matched source bar for every target bar
        bars = bars_and_chords(s)
        src = [b for b in bars if b[1] <= t0 or b[0] >= t1]
        fill = []
        n_exact = n_bars = 0
        for b0, b1, chords in bars:
            if b1 <= t0 or b0 >= t1:
                continue
            if a.random_bars:
                rs = np.random.default_rng(int(b0 * 1000))
                sb0, sb1, c = src[rs.integers(len(src))] if src else (0, 0, [])
                best = (len(chords), 0, 0, sb0, sb1, 0) if src else None
            else:
                best = max(((match_score(c, chords, sh), -abs(sh), -abs(sb0 - b0), sb0, sb1, sh)
                            for sb0, sb1, c in src for sh in range(-5, 7)), default=None)
            if best is None or best[0] < 0:
                continue
            n_bars += 1
            n_exact += best[0] == len(chords) and best[5] == 0
            _, _, _, sb0, sb1, sh = best
            k = (b1 - b0) / (sb1 - sb0)
            for on, off, p, v, _ in acc[(acc[:, 0] >= sb0) & (acc[:, 0] < sb1)]:
                ns, ne = b0 + (on - sb0) * k, b0 + (off - sb0) * k
                fill.append(pretty_midi.Note(int(v), int(p + sh), ns - t0, min(ne, b1 + 0.3) - t0))
        exact.append(n_exact / max(n_bars, 1))
        fill = [f for f in fill if f.start >= 0 and f.start < a.secs]
        collide += sum(any(f.pitch == n.pitch and abs(f.start - n.start) < 0.05 for f in fill) for n in voice)

        # 3. audio: solo voice, and voice + added accompaniment
        jr = np.random.default_rng(int(s))
        sung = [pretty_midi.Note(n.velocity, n.pitch, *(np.array([n.start, n.end]) + jr.normal(0, a.jitter)))
                for n in voice] if a.jitter else voice
        y = aah(sung, a.secs)
        if a.bleed is not None:                   # separation leftovers: the span's real accompaniment, quiet
            real = [pretty_midi.Note(n.velocity, n.pitch, n.start - t0, n.end - t0) for n in pia if t0 <= n.start < t1]
            b = pluck(real, len(y), 1.0)
            y = (y + 10 ** (a.bleed / 20) * b / (np.abs(b).max() + 1e-9) * np.abs(y).max()).astype(np.float32)
        sf.write(out / f"{s}_solo.wav", y, SR)
        solo = ST.transcribe(out / f"{s}_solo.wav", onset_thr=a.thr, ckpt=ck, frame_thr=0.5)
        solo_m = mono(solo) if a.mono_guide else solo        # range and protection from one voice, not the bleed
        lo = np.percentile([g["pitch"] for g in solo_m], 5) - 2 if solo_m else 0     # voice range from the solo pass
        added = pluck(fill, len(y), 1.0)
        mix = y + a.level * added / (np.abs(added).max() + 1e-9) * np.abs(y).max()
        mix = (0.5 * mix / np.abs(mix).max()).astype(np.float32)
        sf.write(out / f"{s}_fill.wav", mix, SR)

        got = ST.transcribe(out / f"{s}_fill.wav", onset_thr=a.thr, ckpt=ck, frame_thr=0.5)
        got = [g for g in got if not any(g["pitch"] == f.pitch and abs(g["onset"] - f.start) < 0.05 for f in fill)]
        ranged = [g for g in got if g["pitch"] >= lo]

        # variant: added part kept below the voice range (moved down by octaves), and notes the solo pass
        # already found are never removed
        low = []
        for f in fill:
            p = f.pitch
            while p > lo - 1:
                p -= 12
            if p >= 28:
                low.append(pretty_midi.Note(f.velocity, p, f.start, f.end))
        added2 = pluck(low, len(y), 1.0)
        mix2 = y + a.level * added2 / (np.abs(added2).max() + 1e-9) * np.abs(y).max()
        mix2 = (0.5 * mix2 / np.abs(mix2).max()).astype(np.float32)
        sf.write(out / f"{s}_fill_low.wav", mix2, SR)
        g2 = ST.transcribe(out / f"{s}_fill_low.wav", onset_thr=a.thr, ckpt=ck, frame_thr=0.5)
        seen = lambda g: any(g["pitch"] == x["pitch"] and abs(g["onset"] - x["onset"]) < 0.05 for x in solo_m)
        g2 = [g for g in g2 if seen(g) or not any(g["pitch"] == f.pitch and abs(g["onset"] - f.start) < 0.05
                                                  for f in low)]
        g2 = [g for g in g2 if g["pitch"] >= lo]
        g3 = mono(g2)
        trk = voice_track(y) if a.voice_filter else None
        g4 = voice_like(g3, trk) if a.voice_filter else g3
        s4 = voice_like(mono(solo), trk) if a.voice_filter else mono(solo)
        for k, g in (("solo", solo), ("fill", got), ("fill+range", ranged), ("below+protect", g2),
                     ("solo+mono", mono(solo)), ("below+protect+mono", g3),
                     ("solo+mono+voice", s4), ("below+protect+mono+voice", g4)):
            res[k].append(score(voice, g))
        print(f"{s}: voice {len(voice):3d} added {len(fill):4d}  exact-chord bars {exact[-1]:.0%}  "
              + "  ".join(f"{k} F1 {res[k][-1][2]:.3f}" for k in res), flush=True)
        sf.write(out / f"{s}_ab.wav", np.stack([mix2, sines(g2, len(mix)).astype(np.float32)], 1), SR)

    print(f"\nmelody/accompaniment split accuracy (whole songs): {np.mean(split_acc):.3f}")
    print(f"bars with an exact chord match (no transposition): {np.mean(exact):.0%}")
    print(f"voice notes colliding with an added note (lost by construction): {collide}")
    for k, v in res.items():
        v = np.array(v)
        print(f"{k:24s} P {v[:, 0].mean():.3f}  R {v[:, 1].mean():.3f}  F1 {v[:, 2].mean():.3f}")


if __name__ == "__main__":
    main()
