"""Slakh2100 -> curriculum clips: k instruments mixed, every note labelled, roles marked.

Song-to-piano v2, stage k.  The v1 arranger hears well but cannot tell which
sound is the tune (melody fit 0.60 vs human 0.66).  Training on mixes that
grow from 2 instruments upward teaches it to follow individual parts inside a
mix without separating them, with exact labels at every stage because Slakh
ships audio AND MIDI per stem.

For each track and stage k, combinations of k PITCHED stems are drawn (drums
added to half of them, unlabelled — the model must learn to ignore them), and
20 s windows are cut where every chosen stem actually plays.  Per window:

  on / fr    every pitched note of the chosen stems, folded onto the 88 keys
  mel        notes of the MELODY stem: among the chosen non-bass stems, the
             mostly-monophonic one that holds the top voice longest (Slakh
             has no melody labels; this is the usual heuristic and is stated
             as one).  Zeros when no stem qualifies.
  bass       notes of stems whose instrument class is Bass
  scope      "all" — every pitched note in this audio is labelled

Output matches the other clip builders (24 kHz int16, rolls at 62.5 fps,
<split>/<track>_<k>_<c>_<w>.npz + .vel.npy) with piano="slakh", so
data.RenderedClips reads it with with_roles=True.

    python eval/make_curriculum_clips.py --slakh /scratch/.../slakh2100_flac_redux \\
        --out /scratch/.../curriculum --stage 2 --shard 0/32
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

SECONDS = 20.0
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def stem_notes(mid: pathlib.Path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(mid))
    return np.array([(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments
                     if not i.is_drum for n in i.notes], float).reshape(-1, 4)


def rolls(notes, t0, T, fps):
    on = np.zeros((88, T), np.uint8); fr = np.zeros((88, T), np.uint8)
    vel = np.zeros((88, T), np.uint8)
    for s, e, p, v in notes:
        k = int(p) - 21
        if not 0 <= k < 88 or e <= t0 or s >= t0 + T / fps:
            continue
        a, b = int(round((s - t0) * fps)), int(round((e - t0) * fps))
        fr[k, max(0, a):min(T, max(b, a + 1))] = 1
        if 0 <= a < T:
            on[k, a] = 1; vel[k, a] = max(vel[k, a], int(v))
    return on, fr, vel


def pick_melody(stem_frames: dict, bass_ids: set):
    """Mostly-monophonic non-bass stem holding the top voice longest, or None."""
    cand = {s: f for s, f in stem_frames.items() if s not in bass_ids}
    if not cand:
        return None
    T = next(iter(cand.values())).shape[1]
    top = np.full(T, -1); owner = np.full(T, None, object)
    for s, f in cand.items():
        hi = np.where(f.any(0), 87 - np.argmax(f[::-1], 0), -1)      # highest active key
        win = hi > top
        top[win], owner[win] = hi[win], s
    best, share = None, 0.0
    for s, f in cand.items():
        act = f.any(0)
        if act.mean() < 0.25 or f.sum(0)[act].mean() > 1.6:           # too sparse / chordal
            continue
        sh = float((owner == s).mean())
        if sh > share:
            best, share = s, sh
    return best if share >= 0.3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", type=int, required=True, help="pitched instruments per mix")
    ap.add_argument("--combos", type=int, default=2, help="per track")
    ap.add_argument("--windows", type=int, default=2, help="per combination")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import yaml, soundfile as sf
    from scipy.signal import resample_poly
    from app.ml.spectral import frontend as S

    root = pathlib.Path(args.slakh)
    seg = int(SECONDS * S.SR); T = S.n_frames(seg)
    si, sn = (int(v) for v in args.shard.split("/"))
    for split in SPLITS:
        tracks = sorted((root / split).glob("Track*"))[si::sn][: args.limit or None]
        out = pathlib.Path(args.out) / f"stage{args.stage}" / split
        out.mkdir(parents=True, exist_ok=True)
        n_out = 0
        for tr in tracks:
            rng = random.Random(f"{tr.name}-{args.stage}")
            try:
                meta = yaml.safe_load(open(tr / "metadata.yaml"))
            except Exception:                                   # noqa: BLE001
                continue
            stems = {k: v for k, v in meta["stems"].items()
                     if v.get("audio_rendered") and (tr / "stems" / f"{k}.flac").exists()}
            pitched = [k for k, v in stems.items() if not v.get("is_drum")
                       and (tr / "MIDI" / f"{k}.mid").exists()]
            drums = [k for k, v in stems.items() if v.get("is_drum")]
            if len(pitched) < args.stage:
                continue
            notes = {k: stem_notes(tr / "MIDI" / f"{k}.mid") for k in pitched}
            dur = sf.info(str(tr / "stems" / f"{pitched[0]}.flac")).duration
            for c in range(args.combos):
                # half the combinations keep a bass part when the track has one,
                # so the bass role is learned at every stage, not by luck
                basses = [k for k in pitched if stems[k].get("inst_class") == "Bass"]
                if basses and rng.random() < 0.5:
                    b0 = rng.choice(basses)
                    combo = [b0] + rng.sample([k for k in pitched if k != b0], args.stage - 1)
                else:
                    combo = rng.sample(pitched, args.stage)
                use_drums = bool(drums) and rng.random() < 0.5
                bass_ids = {k for k in combo if stems[k].get("inst_class") == "Bass"}
                for w in range(args.windows):
                    dst = out / f"{tr.name}_{args.stage}_{c}_{w}.npz"
                    if dst.exists():
                        continue
                    for _ in range(20):                         # a window where all play
                        t0 = rng.uniform(0, max(0.0, dur - SECONDS))
                        if all(((notes[k][:, 0] >= t0) & (notes[k][:, 0] < t0 + SECONDS)).sum() >= 5
                               for k in combo):
                            break
                    else:
                        continue
                    mix = None
                    for k in combo + (drums[:1] if use_drums else []):
                        info = sf.info(str(tr / "stems" / f"{k}.flac"))
                        a = int(t0 * info.samplerate)
                        y, sr = sf.read(str(tr / "stems" / f"{k}.flac"), start=a,
                                        frames=int(SECONDS * info.samplerate), dtype="float32",
                                        always_2d=True)
                        y = y.mean(1); g = math.gcd(S.SR, sr)
                        y = resample_poly(y, S.SR // g, sr // g).astype(np.float32)
                        y = np.pad(y, (0, max(0, seg - len(y))))[:seg]
                        mix = y if mix is None else mix + y
                    mix *= 0.5 / (np.abs(mix).max() + 1e-6)     # peak -6 dBFS
                    per = {k: rolls(notes[k], t0, T, S.FPS) for k in combo}
                    on = np.max([per[k][0] for k in combo], 0)
                    fr = np.max([per[k][1] for k in combo], 0)
                    vel = np.max([per[k][2] for k in combo], 0)
                    mel_id = pick_melody({k: per[k][1] for k in combo}, bass_ids)
                    mel = per[mel_id][1] if mel_id else np.zeros_like(fr)
                    bass = np.max([per[k][1] for k in bass_ids], 0) if bass_ids else np.zeros_like(fr)
                    tmp = dst.with_name(dst.stem + ".tmp.npz")
                    np.savez_compressed(tmp, y=(np.clip(mix, -1, 1) * 32767).astype(np.int16),
                                        on=on, fr=fr, mel=mel, bass=bass, piano="slakh",
                                        scope="all", stage=args.stage, drums=use_drums,
                                        stems=np.array([stems[k].get("inst_class", "") for k in combo]),
                                        melody=stems[mel_id].get("inst_class", "") if mel_id else "")
                    tmp.replace(dst)
                    vt = dst.with_name(dst.stem + ".vel.tmp.npy")
                    np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
                    n_out += 1
        print(f"  stage {args.stage} {split}: {n_out} clips", flush=True)


if __name__ == "__main__":
    main()
