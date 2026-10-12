"""Symbolic corpora -> one note archive per source, for the grammar model.

Broad on purpose (pop, rock, classical, film, …), each source kept separate so
training can balance them rather than let the largest dominate:

  m2m       Music2MIDI piano covers (pop, human arrangements, seconds aligned
            to audio).  Songs in the song-to-piano TEST split are excluded
            entirely, by artist+title (several covers of one song share it);
            VALIDATION songs go to this corpus's validation split.
  pop909    POP909 piano arrangements (melody + bridge + piano tracks merged)
  maestro   MAESTRO performances (classical, expressive timing), its train split
  asap      ASAP score MIDIs (classical, engraved)
  lakh      Lakh MIDI (pop/rock/many genres, multi-instrument) folded onto the
            piano: every non-drum note, pieces under 20 s, over 15 min, or with
            more than 10 notes sounding on average dropped.

Output: <out>/<source>_{train,validation}.npz with `notes` (N, 4 float32:
onset, offset, pitch, velocity, seconds) and `starts` (piece offsets).

    python eval/make_grammar_corpus.py --source m2m --out <dir>
    python eval/make_grammar_corpus.py --source lakh --lakh /scratch/.../lmd_full --out <dir> --shard 3/16
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
DATA = ROOT.parent / "data"


def midi_notes(path, drop_drums=True):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    rows = [(n.start, n.end, n.pitch, n.velocity) for i in pm.instruments
            if not (drop_drums and i.is_drum) for n in i.notes if 21 <= n.pitch <= 108]
    return np.array(rows, np.float32).reshape(-1, 4)


def heldout(key: str, frac=0.02) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000 < frac * 10000


def pieces(source, args):
    """yield (key, split, notes)."""
    if source == "m2m":
        import yaml
        m2m = DATA / "music2midi"
        sp = json.loads((m2m / "m2m_splits.json").read_text())
        meta = lambda s: yaml.safe_load(open(m2m / "music2midi-dataset" / "metadata" / f"{s}.yaml", encoding="utf-8"))
        song = lambda m: f"{m['song']['artist']}|{m['song']['title']}".lower()
        test_songs = {song(meta(s)) for s in sp["test"]}
        val_songs = {song(meta(s)) for s in sp["validation"]}
        for f in sorted((m2m / "music2midi-dataset" / "midi_numpy").glob("*.npy")):
            try:
                k = song(meta(f.stem))
            except Exception:                                   # noqa: BLE001
                continue
            if k in test_songs:
                continue
            yield f.stem, "validation" if k in val_songs else "train", np.load(f).astype(np.float32)
    elif source == "pop909":
        for d in sorted((DATA / "POP909").glob("[0-9][0-9][0-9]")):
            yield d.name, "validation" if heldout(d.name, 0.05) else "train", midi_notes(d / f"{d.name}.mid")
    elif source == "maestro":
        import csv
        root = DATA / "maestro-v3.0.0"
        for r in csv.DictReader(open(root / "maestro-v3.0.0.csv", encoding="utf-8")):
            if r["split"] in ("train", "validation"):
                yield r["midi_filename"], r["split"], midi_notes(root / r["midi_filename"])
    elif source == "asap":
        for f in sorted((DATA / "asap-dataset").rglob("midi_score.mid")):
            k = str(f.parent.relative_to(DATA / "asap-dataset"))
            yield k, "validation" if heldout(k, 0.05) else "train", midi_notes(f)
    elif source == "lakh":
        files = sorted(pathlib.Path(args.lakh).rglob("*.mid"))
        si, sn = (int(v) for v in args.shard.split("/"))
        for f in files[si::sn]:
            try:
                n = midi_notes(f)
            except Exception:                                   # noqa: BLE001
                continue
            if len(n) < 50:
                continue
            dur = n[:, 1].max() - n[:, 0].min()
            if not 20 <= dur <= 900:
                continue
            if (n[:, 1] - n[:, 0]).sum() / dur > 10:            # wall of sustained parts
                continue
            yield f.stem, "validation" if heldout(f.stem) else "train", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["m2m", "pop909", "maestro", "asap", "lakh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lakh", default="")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    acc = {"train": [], "validation": []}
    for key, split, n in pieces(args.source, args):
        if len(n) >= 20:
            n = n[np.argsort(n[:, 0], kind="stable")]
            n[:, :2] -= n[:, 0].min()
            acc[split].append(n)
    tag = args.source + ("" if args.shard == "0/1" else f"_{args.shard.split('/')[0]:0>2}")
    for split, lst in acc.items():
        if not lst:
            continue
        starts = np.cumsum([0] + [len(x) for x in lst])
        np.savez_compressed(out / f"{tag}_{split}.npz", notes=np.concatenate(lst), starts=starts)
        print(f"  {tag} {split}: {len(lst)} pieces, {starts[-1]} notes", flush=True)


if __name__ == "__main__":
    main()
