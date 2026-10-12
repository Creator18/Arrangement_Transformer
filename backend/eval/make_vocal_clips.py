"""MUSDB18-HQ -> curriculum clips with REAL vocals as the melody.

Slakh has no singing, and in songs the voice is usually the tune.  MUSDB18-HQ
gives real recordings split into vocals / bass / drums / other, but no notes.
Labels come from pitch tracking (pYIN) on the ISOLATED stems: the vocals stem
becomes the melody, the bass stem the bass line.  "other" (guitars, keys,
synths, all summed) cannot be labelled note by note, so any mix containing it
is scope "melody": only the melody/bass outputs are trained there, and the
onset/frame loss is masked rather than teaching every unlabelled chord as a
false positive.

Mixes per track (drums are unlabelled and harmless — no pitched notes):
  vocals                       scope all
  vocals + drums               scope all
  vocals + bass (+ drums)      scope all   (both lines labelled)
  vocals + other               scope melody
  full mix                     scope melody

pYIN on a sung line is the usual choice but not ground truth (vibrato, slides,
breathy onsets): eval/vocal_label_check.py listens to a sample before training
relies on these labels.

    python eval/make_vocal_clips.py --musdb /scratch/.../musdb18hq --out /scratch/.../curriculum
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from make_curriculum_clips import rolls          # noqa: E402

SECONDS = 20.0
MIXES = [("vocals", ["vocals"], "all"), ("vocals_drums", ["vocals", "drums"], "all"),
         ("vocals_bass", ["vocals", "bass", "drums"], "all"),
         ("vocals_other", ["vocals", "other"], "melody"),
         ("full", ["vocals", "bass", "drums", "other"], "melody")]


def pyin_notes(y, sr, lo, hi):
    """Pitch-track a monophonic stem -> [onset, offset, pitch, velocity] rows."""
    from app.services.transcription import _pyin_to_notes
    ns = _pyin_to_notes(y, sr, hop_length=256, fmin_note=lo, fmax_note=hi, min_duration=0.06)
    return np.array([(n["onset"], n["offset"], n["pitch"], n.get("velocity", 80)) for n in ns],
                    float).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=4, help="per mix per track")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import soundfile as sf, librosa
    from scipy.signal import resample_poly
    from app.ml.spectral import frontend as S

    root = pathlib.Path(args.musdb)
    seg = int(SECONDS * S.SR); T = S.n_frames(seg)
    si, sn = (int(v) for v in args.shard.split("/"))
    # MUSDB ships train/test; hold out 10 train tracks as validation, by track
    for split, src in (("train", "train"), ("validation", "train"), ("test", "test")):
        tracks = sorted((root / src).iterdir())
        if src == "train":
            held = set(t.name for t in random.Random(20261012).sample(tracks, 10))
            tracks = [t for t in tracks if (t.name in held) == (split == "validation")]
        out = pathlib.Path(args.out) / "vocal" / split
        out.mkdir(parents=True, exist_ok=True)
        n_out = 0
        for tr in tracks[si::sn][: args.limit or None]:
            rng = random.Random(tr.name)
            stem = {}
            for k in ("vocals", "bass", "drums", "other"):
                y, sr = sf.read(str(tr / f"{k}.wav"), dtype="float32", always_2d=True)
                g = math.gcd(S.SR, sr)
                stem[k] = resample_poly(y.mean(1), S.SR // g, sr // g).astype(np.float32)
            cache = out.parent / f"labels_{tr.name}.npz"
            if cache.exists():
                z = np.load(cache); voc, bas = z["voc"], z["bas"]
            else:
                y16 = librosa.resample(stem["vocals"], orig_sr=S.SR, target_sr=16000)
                voc = pyin_notes(y16, 16000, "E2", "C6")
                y16 = librosa.resample(stem["bass"], orig_sr=S.SR, target_sr=16000)
                bas = pyin_notes(y16, 16000, "E1", "G3")
                np.savez(cache, voc=voc, bas=bas)
            dur = len(stem["vocals"]) / S.SR
            for name, parts, scope in MIXES:
                for w in range(args.windows):
                    dst = out / f"{tr.name.replace(' ', '_')}_{name}_{w}.npz"
                    if dst.exists():
                        continue
                    for _ in range(30):                          # a window where the voice sings
                        t0 = rng.uniform(0, max(0.0, dur - SECONDS))
                        if ((voc[:, 0] >= t0) & (voc[:, 0] < t0 + SECONDS)).sum() >= 8:
                            break
                    else:
                        continue
                    a = int(t0 * S.SR)
                    mix = sum(np.pad(stem[k][a:a + seg], (0, max(0, seg - len(stem[k][a:a + seg]))))
                              for k in parts)
                    mix = mix * 0.5 / (np.abs(mix).max() + 1e-6)
                    v_on, v_fr, v_vel = rolls(voc, t0, T, S.FPS)
                    b_on, b_fr, b_vel = rolls(bas, t0, T, S.FPS) if "bass" in parts else (np.zeros_like(v_on),) * 3
                    tmp = dst.with_name(dst.stem + ".tmp.npz")
                    np.savez_compressed(tmp, y=(np.clip(mix, -1, 1) * 32767).astype(np.int16),
                                        on=np.maximum(v_on, b_on), fr=np.maximum(v_fr, b_fr),
                                        mel=v_fr, bass=b_fr, piano="musdb", scope=scope,
                                        stage=len([p for p in parts if p != "drums"]), mix=name)
                    tmp.replace(dst)
                    n_out += 1
        print(f"  vocal {split}: {n_out} clips", flush=True)


if __name__ == "__main__":
    main()
