"""Generation check for the structure-aware grammar: let it WRITE, then score what it wrote.

Next-token accuracy is the easy test (the correct sequence so far is given).
Here the model gets only CORRUPTED, structure-free notes for a passage of
held-out validation songs and writes the whole structured passage itself,
greedily.  Scored against the true passage:

  notes   note F1 (50 ms) of what it wrote vs the clean notes, beside the F1
          of the corrupted input it was given — correction > input means it
          fixes more than it breaks (grammar v1 broke even: 0.863 -> 0.864)
  chords  share of beats whose written chord equals the true chord
  roles   share of matched notes (same key, onset within 50 ms) whose role
          (melody / bass / accompaniment) is right
  groups  share of bars whose rhythm-group token is right
  key     whether the passage's key is right
  bars    whether it wrote the right number of bars

Per source (POP909 / Music2MIDI).  GPU generation, then light CPU scoring.

    python eval/grammar2_check.py --ckpt <grammar2 best.pt> --corpus <grammar2 dir> --n 192
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def f1(ref, est, tol=0.05):
    from mir_eval.transcription import precision_recall_f1_overlap as prf
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    iv = lambda n: np.c_[n[:, 0], np.maximum(n[:, 1], n[:, 0] + 1e-3)]
    hz = lambda n: 440 * 2 ** ((n[:, 2] - 69) / 12)
    return prf(iv(ref), hz(ref), iv(est), hz(est), onset_tolerance=tol, offset_ratio=None)[2]


def bar_pitch_f1(ref, est, beats, gs, nb):
    """Timing-free per bar: F1 of the multiset of pitches each bar contains."""
    from collections import Counter
    edges = [beats[min(gs[i], len(beats) - 1)] for i in range(min(nb, len(gs) - 1) + 1)]
    tp = fp = fn = 0
    for a, b in zip(edges, edges[1:]):
        r = Counter(int(p) for s, _, p, _ in ref if a <= s < b)
        e = Counter(int(p) for s, _, p, _ in est if a <= s < b)
        hit = sum((r & e).values()); tp += hit; fp += sum(e.values()) - hit; fn += sum(r.values()) - hit
    return 2 * tp / max(1, 2 * tp + fp + fn)


def bar_tokens(toks, lo, hi):
    """per bar: the first token in [lo, hi) after that BAR, or None."""
    from app.ml.grammar import tokens2 as T2
    out, cur = [], None
    for t in toks:
        if t == T2.BAR:
            out.append(None)
        elif lo <= t < hi and out and out[-1] is None:
            out[-1] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=192, help="passages per source")
    ap.add_argument("--free", action="store_true", help="unconstrained greedy (the first check)")
    ap.add_argument("--grid-src", action="store_true", help="model trained with beat-grid input")
    args = ap.parse_args()
    import torch
    from app.ml.grammar import tokens as T, tokens2 as T2
    from app.ml.grammar.model import GrammarModel
    from app.ml.grammar.train2 import collate
    from app.ml.grammar import constrained as CD
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = GrammarModel(**ck["arch"]).to(dev).eval(); net.load_state_dict(ck["model"])

    for src_name in ("pop909", "m2m"):
        songs = []
        for f in sorted(pathlib.Path(args.corpus).glob(f"{src_name}*_validation.npz")):
            z = np.load(f, allow_pickle=True)
            for i in range(len(z["ids"])):
                songs.append((z["tokens"][z["starts"][i]:z["starts"][i + 1]].astype(np.int64),
                              np.asarray(z["beats"][i], float), np.asarray(z["bars"][i], int)))
        if not songs:
            continue
        rng = np.random.default_rng(3)
        items = []
        while len(items) < args.n:
            tok, beats, grid = songs[rng.integers(len(songs))]
            bar_ix = np.flatnonzero(tok == T2.BAR)
            if len(bar_ix) < 10:
                continue
            nb = int(rng.integers(4, 9)); b0 = int(rng.integers(0, len(bar_ix) - nb))
            a, e = bar_ix[b0], bar_ix[b0 + nb]
            head = [t for t in tok[:a] if T2.METER0 <= t < T2.KEY0][-1:] + \
                   [t for t in tok[:a] if T2.KEY0 <= t < T2.SEC0][-1:]
            clean = [T2.BAR] + head + list(tok[a + 1:e])
            if len(clean) > 900:
                continue
            gs = list(grid[b0:])
            notes, roles, chords = T2.decode(clean, beats, gs)
            if len(notes) < 10:
                continue
            t0 = notes[:, 0].min()
            noisy = notes.copy(); noisy[:, :2] -= t0
            noisy = T.corrupt(noisy, rng)
            if args.grid_src:
                nabs = noisy.copy(); nabs[:, :2] += t0
                srct = T2.encode_plain(nabs, beats, list(grid[b0:b0 + nb + 1]))[:1000]
            else:
                srct = T.encode(noisy)[:1000]
            items.append(dict(src=srct, clean=clean, beats=beats, gs=gs, t0=t0, nb=nb,
                              notes=notes, roles=roles, chords=chords, noisy=noisy))
        gens = []
        for k in range(0, len(items), 24):
            chunk = items[k:k + 24]
            src, _, _ = collate([(it["src"], it["clean"]) for it in chunk])
            src = src.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                if args.free:
                    g = net.generate(src, max_len=950).cpu().numpy()
                    rows = [[int(x) for x in r if x != T.PAD] for r in g]
                    rows = [r[:r.index(T.EOS)] if T.EOS in r else r for r in rows]
                else:
                    mem, pad = net.encode(src), src == T.PAD
                    step = lambda y: net.decode(y, mem, pad)[:, -1]
                    pre = torch.full((len(chunk), 1), T.BOS, dtype=torch.long, device=dev)
                    rows = CD.generate(step, pre, [CD.bar_lengths(it["gs"], it["nb"]) for it in chunk])
            gens += rows
        R = {k: [] for k in ("f_in", "f_out", "chord", "role", "group", "key", "bars",
                             "f_in150", "f_out150", "bar_in", "bar_out")}
        for it, g in zip(items, gens):
            n2, r2, c2 = T2.decode(g, it["beats"], it["gs"])
            noisy_abs = it["noisy"].copy(); noisy_abs[:, :2] += it["t0"]
            R["f_in"].append(f1(it["notes"], noisy_abs)); R["f_out"].append(f1(it["notes"], n2))
            R["f_in150"].append(f1(it["notes"], noisy_abs, 0.15)); R["f_out150"].append(f1(it["notes"], n2, 0.15))
            R["bar_in"].append(bar_pitch_f1(it["notes"], noisy_abs, it["beats"], it["gs"], it["nb"]))
            R["bar_out"].append(bar_pitch_f1(it["notes"], n2, it["beats"], it["gs"], it["nb"]))
            beats_with = [b for b, c in it["chords"].items() if c is not None]
            # chords are written at changes: carry each forward to every beat
            def per_beat(ch, n):
                out, cur, ks = {}, None, sorted(ch)
                for b in range(min(ks, default=0), min(ks, default=0) + n):
                    cur = ch.get(b, cur); out[b] = cur
                return out
            nbeats = max(1, max(it["chords"], default=0) - min(it["chords"], default=0) + 1)
            pt, pg = per_beat(it["chords"], nbeats), per_beat(c2, nbeats)
            R["chord"].append(np.mean([pg.get(b) == pt[b] for b in pt if pt[b] is not None]) if beats_with else np.nan)
            if len(n2):
                hits = []
                for (s, e_, p, v), rr in zip(it["notes"], it["roles"]):
                    j = np.flatnonzero((n2[:, 2] == p) & (np.abs(n2[:, 0] - s) <= 0.05))
                    if len(j):
                        hits.append(r2[j[0]] == rr)
                R["role"].append(np.mean(hits) if hits else np.nan)
            gt = bar_tokens(it["clean"], T2.GROUP0, T2.POS0); gg = bar_tokens(g, T2.GROUP0, T2.POS0)
            pairs = [(x, y) for x, y in zip(gt, gg) if x is not None]
            R["group"].append(np.mean([x == y for x, y in pairs]) if pairs else np.nan)
            kt = [t for t in it["clean"] if T2.KEY0 <= t < T2.SEC0][:1]; kg = [t for t in g if T2.KEY0 <= t < T2.SEC0][:1]
            R["key"].append(float(kt == kg) if kt else np.nan)
            R["bars"].append(float(sum(t == T2.BAR for t in g) == sum(t == T2.BAR for t in it["clean"])))
        m = {k: np.nanmean(v) for k, v in R.items()}
        print(f"{src_name}: {len(items)} held-out passages written from corrupted notes\n"
              f"   note F1 (50 ms):  corrupted input {m['f_in']:.3f} -> model {m['f_out']:.3f}\n"
              f"   note F1 (150 ms): corrupted input {m['f_in150']:.3f} -> model {m['f_out150']:.3f}\n"
              f"   per-bar pitch F1 (no timing): corrupted input {m['bar_in']:.3f} -> model {m['bar_out']:.3f}\n"
              f"   chords right per beat {m['chord']:.3f} | roles right {m['role']:.3f} | rhythm group right per bar "
              f"{m['group']:.3f} | key right {m['key']:.3f} | right number of bars {m['bars']:.3f}", flush=True)


if __name__ == "__main__":
    main()
