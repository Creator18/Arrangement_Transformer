"""Group fit: does an arranger keep each passage's accompaniment rhythm group?

Every passage of the Music2MIDI test songs has a rhythm group from the HUMAN
arrangement (eval/rhythm_groups.py).  For a model's arrangement of the same
song, on the same beat grid and passages:

  fit     cosine similarity of the model's accompaniment rhythm fingerprint
          (rotated to the bass-on-beat-1 anchor) with the assigned group's
          centroid — how close its rhythm is to the part's group
  match   share of passages whose nearest group IS the assigned one
  silent  share of passages where the model plays no accompaniment at all

The human arrangement's own fit/match are reported as the ceiling.  Two
stages so the GPU never waits on scoring:

  python eval/group_fit.py --stage notes --ckpt v1=<best.pt>=0.6 ... --out <dir>   (GPU)
  python eval/group_fit.py --stage score --out <dir>                                (CPU)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import rhythm_groups as RG                          # noqa: E402

LEVELS = ["beginner", "intermediate", "advanced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["notes", "score"])
    ap.add_argument("--m2m", default="/scratch/basu.ang/m2m")
    ap.add_argument("--groups", default="/scratch/basu.ang/rhythm_groups")
    ap.add_argument("--ckpt", action="append", default=[], help="label=path=onset_thr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m2m, out = pathlib.Path(args.m2m), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    test = sp["test"]

    if args.stage == "notes":
        from app.services import spectral_transcription as ST
        from app.ml.transcriber.decode import decode_notes
        for spec in args.ckpt:
            label, path, thr = spec.split("=")
            ST.CKPT = pathlib.Path(path)
            for sid in test:
                on, fr, vl = ST.rolls(m2m / "audio" / f"{sid}.wav", cond=LEVELS.index(sp["difficulty"][sid]))
                ns = decode_notes(on, fr, vl, float(thr), 0.3, min_frames=1, min_note_s=0.0, keep_restruck=True)
                np.save(out / f"{label}__{sid}.npy",
                        np.array([[n["onset"], n["offset"], n["pitch"], n["velocity"]] for n in ns]).reshape(-1, 4))
            print(f"  notes for {label}", flush=True)
        return

    cent = json.loads((pathlib.Path(args.groups) / "group_centroids.json").read_text())
    labels = json.loads((pathlib.Path(args.groups) / "group_labels.json").read_text())
    models = sorted({f.name.split("__")[0] for f in out.glob("*__*.npy")})
    npy = m2m / "midi_numpy"
    acc = {m: {"fit": [], "match": [], "silent": [], "level": []} for m in models + ["human"]}
    for sid in test:
        lab = {round(t, 3): g for t, g in labels.get(f"m2m:{sid}", []) if g}
        if not lab:
            continue
        human = np.load(npy / f"{sid}.npy"); human = human[np.argsort(human[:, 0], kind="stable")]
        beats, tempo = RG.track_beats(human)
        beats, tempo = RG.settle_tempo(human, beats, tempo)
        m, ph, sub, _ = RG.meter_from_beats(human, beats)
        meter = RG.MERGE.get(RG.meter_name(m, sub), RG.meter_name(m, sub))
        same = {g: np.array(c["centroid"]) for g, c in cent.items() if c["meter"] == meter}
        if not same:
            continue
        names = list(same); C = np.stack([same[g] for g in names])
        for who in models + ["human"]:
            if who == "human":
                notes = human
            else:
                f = out / f"{who}__{sid}.npy"
                if not f.exists():
                    continue
                notes = np.load(f)
                notes = notes[np.argsort(notes[:, 0], kind="stable")] if len(notes) else notes
            if len(notes) < 10:
                continue
            fps = RG.fingerprints(notes, RG.melody_split(notes), beats, m, ph, sub)
            for t, v in fps:
                g = lab.get(round(t, 3))
                if g is None or g not in same:
                    continue
                bpb = len(v) // (2 * RG.SLOTS)
                silent = v.sum() == 0
                x = RG.canonical(v.astype(float), bpb)
                x = x / (np.linalg.norm(x) + 1e-9)
                sims = C @ x
                a = acc[who]
                a["fit"].append(0.0 if silent else float(sims[names.index(g)]))
                a["match"].append(0.0 if silent else float(names[int(np.argmax(sims))] == g))
                a["silent"].append(float(silent)); a["level"].append(sp["difficulty"][sid])
        print(f"  {sid} {meter}", flush=True)

    lines = [f"group fit on {len(test)} Music2MIDI test songs (passages labelled from the human arrangement)", ""]
    for lv in LEVELS + ["all"]:
        lines.append(f"-- {lv} --")
        lines.append(f"   {'system':<12}{'passages':>9}{'fit':>8}{'match':>8}{'silent':>8}")
        for who in models + ["human"]:
            a = acc[who]; sel = [i for i, L in enumerate(a["level"]) if lv == "all" or L == lv]
            if not sel:
                continue
            g = lambda k: np.mean([a[k][i] for i in sel])
            lines.append(f"   {who:<12}{len(sel):>9}{g('fit'):>8.3f}{g('match'):>8.3f}{g('silent'):>8.3f}")
    (out / "group_fit.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
