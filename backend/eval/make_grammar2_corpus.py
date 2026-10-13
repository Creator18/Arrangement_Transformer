"""Structure-labelled, structure-tokenized corpus for the structure-aware grammar.

Every arrangement -> app.ml.grammar.structure.label (meter, bars, beats, key,
chords per beat, roles, sections, rhythm groups) -> tokens2.encode.  Per song it
also keeps the beat grid and bar grid, so the audio stage can line each bar of
the arrangement up with the same bar of the song.

  pop909  truths where POP909 has them (beats/downbeats, key, chords, MELODY track)
  m2m     everything estimated from the notes; Music2MIDI TEST songs are labelled
          into the test split and never used for training

Output per source and split: <out>/<source>_<split>.npz with
  tokens (concatenated int16), starts (song offsets), ids, beats (object), bars (object)

    python eval/make_grammar2_corpus.py --source pop909 --out <dir> [--shard i/n]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["pop909", "m2m"])
    ap.add_argument("--data", default=str(ROOT.parent / "data"))
    ap.add_argument("--groups", default=str(ROOT.parent / "data" / "rhythm_groups"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    import pretty_midi
    from app.ml.grammar import structure as STc, tokens2 as T2
    data, out = pathlib.Path(args.data), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gm = T2.group_table(pathlib.Path(args.groups) / "group_centroids.json")
    labels = json.loads((pathlib.Path(args.groups) / "group_labels.json").read_text())
    si, sn = (int(v) for v in args.shard.split("/"))
    acc = {"train": [], "validation": [], "test": []}

    if args.source == "pop909":
        for d in sorted((data / "POP909").glob("[0-9][0-9][0-9]"))[si::sn]:
            sid = d.name
            pm = pretty_midi.PrettyMIDI(str(d / f"{sid}.mid"))
            rows, mel = [], []
            for inst in pm.instruments:
                for n in inst.notes:
                    rows.append((n.start, n.end, n.pitch, n.velocity)); mel.append(inst.name == "MELODY")
            b = np.loadtxt(d / "beat_midi.txt")
            chords = [(float(s), float(e), l) for s, e, l in
                      (ln.split("\t") for ln in open(d / "chord_midi.txt").read().strip().split("\n"))]
            try:
                key = STc.parse_key(open(d / "key_audio.txt").read().strip().split("\n")[0].split("\t")[-1].strip())
            except Exception:                                   # noqa: BLE001
                key = None
            st = STc.label(np.array(rows), np.array(mel), beats=b[:, 0], downbeats=b[b[:, 2] > 0, 0],
                           key=key, chords_harte=chords, group_labels=labels.get(f"pop909:{sid}"))
            h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 100
            acc["validation" if h < 5 else "train"].append((sid, st))
            print(f"  pop909 {sid}", flush=True)
    else:
        m2m = data / "music2midi"
        sp = json.loads((m2m / "m2m_splits.json").read_text())
        # audio-aligned arrangements (eval/realign_m2m.py) when present
        npy = next(d for d in (m2m / "midi_numpy_aligned", m2m / "midi_numpy",
                               m2m / "music2midi-dataset" / "midi_numpy") if d.is_dir())
        split_of = {s: k for k in ("train", "validation", "test") for s in sp[k]}
        for sid in sorted(split_of)[si::sn]:
            notes = np.load(npy / f"{sid}.npy")
            if len(notes) < 50:
                continue
            st = STc.label(notes, group_labels=labels.get(f"m2m:{sid}"))
            acc[split_of[sid]].append((sid, st))
            print(f"  m2m {sid} {st.meter}", flush=True)

    tag = args.source + ("" if args.shard == "0/1" else f"_{si:02d}")
    for split, items in acc.items():
        if not items:
            continue
        toks = [np.array(T2.encode(st, gm), np.int16) for _, st in items]
        np.savez_compressed(out / f"{tag}_{split}.npz", tokens=np.concatenate(toks),
                            starts=np.cumsum([0] + [len(t) for t in toks]), ids=np.array([i for i, _ in items]),
                            beats=np.array([st.beats for _, st in items], dtype=object),
                            bars=np.array([np.array(T2.bar_grid(st)) for _, st in items], dtype=object),
                            meter=np.array([st.meter for _, st in items]), key=np.array([st.key for _, st in items]))
        print(f"  {tag} {split}: {len(items)} songs, {sum(len(t) for t in toks)} tokens", flush=True)


if __name__ == "__main__":
    main()
