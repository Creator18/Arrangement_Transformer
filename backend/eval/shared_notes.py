"""Shared notes: the same pitch wanted by two voices at once (e.g. melody + arpeggio).

Part 1 (arrangements, CPU): in POP909 and Music2MIDI, how often two notes of the SAME pitch
  - OVERLAP (the second starts before the first ends)              -> a re-press on a held key
  - TOUCH  (the second starts within 30 ms after the first ends)   -> release, then press again at once
and, for POP909 (whose parts are labelled), whether the two notes come from DIFFERENT parts.

Part 2 (model, GPU, --ckpt): on held-out renders, the model's presses on a key it is already
holding; how many land on a written re-press (shared or repeated note) vs. where the arrangement
just holds the note (a true double press).

    python eval/shared_notes.py --pop909 <dir> --m2m <dir> --units m2m_units.json [--ckpt best.pt --clips <dir>]
"""
import argparse, json, pathlib, random, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from render_arrangements import load_pop909, load_m2m        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pop909", required=True); ap.add_argument("--m2m", required=True); ap.add_argument("--units", required=True)
ap.add_argument("--ckpt", default=""); ap.add_argument("--clips", default=""); ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()


def pairs(notes, parts=None):
    """Same-pitch neighbours: counts of overlapping / touching pairs (and cross-part ones)."""
    out = dict(notes=len(notes), overlap=0, touch=0, overlap_cross=0, touch_cross=0, overlap_ms=[])
    for p in np.unique(notes[:, 2]):
        ix = np.flatnonzero(notes[:, 2] == p); ix = ix[np.argsort(notes[ix, 0], kind="stable")]
        for i, j in zip(ix[:-1], ix[1:]):
            s1, e1 = notes[i, 0], notes[i, 1]; s2 = notes[j, 0]
            cross = parts is not None and parts[i] != parts[j]
            if s2 < e1 - 0.02:
                out["overlap"] += 1; out["overlap_cross"] += cross; out["overlap_ms"].append(1000 * (e1 - s2))
            elif s2 - e1 <= 0.03:
                out["touch"] += 1; out["touch_cross"] += cross
    return out


def summary(name, rows):
    n = sum(r["notes"] for r in rows); ov = sum(r["overlap"] for r in rows); to = sum(r["touch"] for r in rows)
    ovc = sum(r["overlap_cross"] for r in rows); toc = sum(r["touch_cross"] for r in rows)
    ms = np.concatenate([r["overlap_ms"] for r in rows if r["overlap_ms"]] or [np.zeros(0)])
    songs = sum(1 for r in rows if r["overlap"] + r["touch"])
    print(f"\n{name}: {len(rows)} arrangements, {n} notes; songs with any: {songs} ({songs / max(1, len(rows)):.0%})")
    print(f"   OVERLAP (re-press on a held key): {ov} pairs = {1000 * ov / n:.1f} per 1000 notes"
          + (f"; from DIFFERENT parts: {ovc} ({ovc / max(1, ov):.0%})" if ovc else ""))
    if len(ms):
        print(f"      how much they overlap: median {np.median(ms):.0f} ms, 90th pct {np.percentile(ms, 90):.0f} ms")
    print(f"   TOUCH (release then immediate re-press, gap <= 30 ms): {to} pairs = {1000 * to / n:.1f} per 1000 notes"
          + (f"; from DIFFERENT parts: {toc} ({toc / max(1, to):.0%})" if toc else ""))


pop = []
for d in sorted(pathlib.Path(a.pop909).iterdir()):
    if d.is_dir() and d.name.isdigit():
        notes, roles, _b, _f, parts = load_pop909(d, with_parts=True)
        pop.append(pairs(notes, parts))
summary("POP909 (parts: melody / bridge / piano accompaniment)", pop)
U = json.loads(pathlib.Path(a.units).read_text())
m2m = []
for i in sorted({i for ids in U["splits"].values() for i in ids}):
    f = pathlib.Path(a.m2m) / "midi_numpy" / f"{i}.npy"
    if f.exists():
        notes, roles, _b, _f = load_m2m(f)
        m2m.append(pairs(notes))
summary("Music2MIDI (one piano track)", m2m)

if a.ckpt:
    import torch
    from app.services import spectral_transcription as st
    from app.ml.transcriber.decode import decode_notes
    st.CKPT = pathlib.Path(a.ckpt); net, fe, dev = st._load()
    files = sorted(p for p in pathlib.Path(a.clips).glob("*.npz") if ".tmp" not in p.name)
    files = random.Random(0).sample(files, min(a.n, len(files)))
    c = dict(gt_notes=0, gt_repress=0, pred_repress=0, on_written=0, true_double=0)
    for f in files:
        z = np.load(f); on, fr = z["on"] > 0, z["fr"] > 0
        y = z["y"].astype(np.float32) / 32767
        with torch.no_grad():
            o = net(fe(torch.from_numpy(y)[None].to(dev)), None)
            po, pf, pv = (torch.sigmoid(t.float())[0].T.cpu().numpy() for t in o[:3])
            pof = torch.sigmoid(o[-1].float())[0].T.cpu().numpy() if getattr(net, "offset_head", None) is not None else None
        T = min(po.shape[1], on.shape[1]); on, fr = on[:, :T], fr[:, :T]
        notes = decode_notes(po[:, :T], pf[:, :T], pv[:, :T], 0.9, 0.5, min_frames=1, min_note_s=0.0, keep_restruck=True,
                             merge_fragment_s=0.06, offset=None if pof is None else pof[:, :T], offset_thr=0.7)
        c["gt_notes"] += int(on.sum())
        c["gt_repress"] += int((on[:, 1:] & fr[:, :-1]).sum())            # written press on a key already sounding
        byk = {}
        for n in notes:
            byk.setdefault(int(n["pitch"]) - 21, []).append(n)
        for k, ns in byk.items():
            ns.sort(key=lambda n: n["onset"])
            for x, y_ in zip(ns[:-1], ns[1:]):
                f2 = int(round(y_["onset"] * 62.5))
                if y_["onset"] - x["offset"] > 0.03 or not (0 < f2 < T):
                    continue                                              # not a re-press on a held key
                c["pred_repress"] += 1
                if on[k, max(0, f2 - 3):f2 + 4].any():
                    c["on_written"] += 1                                  # the arrangement re-presses here too
                elif fr[k, max(0, f2 - 2):f2 + 3].all():
                    c["true_double"] += 1                                 # the arrangement just HOLDS the note
    print(f"\nmodel re-presses on a held key ({len(files)} held-out render clips, {c['gt_notes']} written notes):")
    print(f"   the arrangements themselves re-press a sounding key {c['gt_repress']} times ({1000 * c['gt_repress'] / max(1, c['gt_notes']):.1f} per 1000 notes)")
    print(f"   the model re-presses a held key {c['pred_repress']} times ({1000 * c['pred_repress'] / max(1, c['gt_notes']):.1f} per 1000)")
    print(f"      where the arrangement re-presses too:  {c['on_written']} ({c['on_written'] / max(1, c['pred_repress']):.0%})")
    print(f"      where the arrangement HOLDS the note:  {c['true_double']} ({c['true_double'] / max(1, c['pred_repress']):.0%})")
