"""Generate the audio arranger's arrangement of every Music2MIDI test song (GPU stage).

As the app would: nothing from the human arrangement is used.  Per song:
  beat grid   counted from v1's OWN notes (structure.label on v1's output)
  windows     consecutive 8-bar windows; each gets v1's notes and v1's audio
              features for its bars, and the decoder writes the structured
              arrangement after [BOS, LEVEL] (the song's difficulty level)
  notes       generated tokens -> tokens2.decode on the song's beat grid

Windows from many songs are generated together (greedy, batched).  Output:
<out>/aa__<id>.npy ([on, off, pitch, vel]) — the same layout as the other
models' cached notes, so eval/group_fit.py, eval/timing_offset.py and
eval/score_notes.py score it beside v1 / v3 / Slakh-only.

    python eval/audio_arranger_eval.py --ckpt <best.pt> --cache <v1 cache> --m2m <dir> --out <dir>
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
BARS = 8
LEVELS = ["beginner", "intermediate", "advanced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="aa")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch
    from app.ml.grammar import tokens as T, tokens2 as T2, structure as STc
    from app.ml.grammar.audio_arranger import AudioArranger, LEVEL0, PATCH_S, MAX_AUDIO, MAX_SRC, collate

    dev = torch.device("cuda")
    net = AudioArranger()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    net = net.to(dev).eval()
    m2m, out = pathlib.Path(args.m2m), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    test = sp["test"][: args.limit or None]

    jobs = []                                   # (sid, window index, src, feat, prefix, beats, grid slice)
    meta = {}
    for sid in test:
        z = np.load(pathlib.Path(args.cache) / f"{sid}.npz")
        v1 = z["notes"].astype(np.float64)
        if len(v1) < 20:
            continue
        st = STc.label(v1)                       # beat grid from v1's own notes
        grid = T2.bar_grid(st); beats = st.beats
        meta[sid] = []
        for w, b0 in enumerate(range(0, len(grid) - 1, BARS)):
            b1 = min(b0 + BARS, len(grid) - 1)
            t_a, t_b = beats[min(grid[b0], len(beats) - 1)], beats[min(grid[b1], len(beats) - 1)]
            t_b = min(t_b, t_a + MAX_AUDIO * PATCH_S)
            feat = z["feat"][int(t_a / PATCH_S):int(t_b / PATCH_S) + 1][:MAX_AUDIO].astype(np.float32)
            n = v1[(v1[:, 0] >= t_a) & (v1[:, 0] < t_b)].copy(); n[:, :2] -= t_a
            src = T.encode(n)[:MAX_SRC] if len(n) else []
            if len(feat) == 0:
                continue
            jobs.append((sid, w, src, feat, [LEVEL0 + LEVELS.index(sp["difficulty"][sid])], beats, list(grid[b0:])))
        print(f"  {sid}: {len(grid) - 1} bars from v1's notes ({st.meter}), "
              f"{sum(1 for j in jobs if j[0] == sid)} windows", flush=True)

    print(f"\ngenerating {len(jobs)} windows in batches of {args.batch}", flush=True)
    res = {}
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][3]))      # similar lengths together
    for k in range(0, len(order), args.batch):
        idx = order[k:k + args.batch]
        batch = [(jobs[i][2], jobs[i][3], []) for i in idx]
        src, aud, alen, _, _ = collate(batch)
        prefix = torch.tensor([[T.BOS] + jobs[i][4] for i in idx], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            gen = net.generate(src.to(dev), aud.to(dev), alen.to(dev), prefix).cpu().numpy()
        for i, g in zip(idx, gen):
            toks = [int(x) for x in g if x not in (T.PAD,)]
            toks = toks[:toks.index(T.EOS)] if T.EOS in toks else toks
            sid, w, _, _, _, beats, gslice = jobs[i]
            notes, _, _ = T2.decode(toks, beats, gslice)
            res.setdefault(sid, []).append(notes)
        print(f"   {min(k + args.batch, len(order))}/{len(order)}", flush=True)

    for sid, parts in res.items():
        n = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.zeros((0, 4))
        n = n[np.argsort(n[:, 0], kind="stable")] if len(n) else n
        np.save(out / f"{args.label}__{sid}.npy", n.astype(np.float32))
    print(f"wrote {len(res)} songs to {out}")


if __name__ == "__main__":
    main()
