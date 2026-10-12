"""What does the grammar model fix, and what does it break?

Correction F1 alone (0.846 model vs 0.870 corrupted input) hides the trade:
precision says whether wrong notes were REMOVED, recall whether right notes
were KEPT.  Also split by source, since the corpora differ in style.

    python eval/grammar_diagnose.py --ckpt <grammar best.pt> --corpus <dir>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prf(ref, est):
    from mir_eval.transcription import precision_recall_f1_overlap as f
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    iv = lambda n: np.c_[n[:, 0], np.maximum(n[:, 1], n[:, 0] + 1e-3)]
    hz = lambda n: 440 * 2 ** ((n[:, 2] - 69) / 12)
    return f(iv(ref), hz(ref), iv(est), hz(est), onset_tolerance=0.05, offset_ratio=None)[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=96, help="windows per source")
    args = ap.parse_args()
    from app.ml.grammar import tokens as T
    from app.ml.grammar.model import GrammarModel
    from app.ml.grammar.train import Windows, collate, load
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = GrammarModel(**ck["arch"]).to(dev).eval(); net.load_state_dict(ck["model"])
    print(f"grammar checkpoint step {ck['step']}\n")
    print(f"{'source':<10}{'input P':>9}{'R':>7}{'F1':>7}   {'model P':>9}{'R':>7}{'F1':>7}")
    tot = []
    for name, a in load(args.corpus, "validation", ["m2m", "lakh", "pop909", "maestro", "asap"]):
        ds = Windows([a], [1.0], seed=99); rng = np.random.default_rng(3)
        ex = [ds.sample(rng) for _ in range(args.n)]
        rows = []
        for i in range(0, len(ex), 32):
            src, _, _ = collate(ex[i:i + 32])
            with torch.autocast("cuda", dtype=torch.float16):
                gen = net.generate(src.to(dev)).cpu().numpy()
            for (noisy, clean), g in zip(ex[i:i + 32], gen):
                ref = T.decode(clean)
                rows.append(prf(ref, T.decode(noisy)) + prf(ref, T.decode([x for x in g if x != T.PAD])))
        r = np.array(rows).mean(0); tot.append(r)
        print(f"{name:<10}{r[0]:>9.3f}{r[1]:>7.3f}{r[2]:>7.3f}   {r[3]:>9.3f}{r[4]:>7.3f}{r[5]:>7.3f}", flush=True)
    r = np.mean(tot, 0)
    print(f"{'mean':<10}{r[0]:>9.3f}{r[1]:>7.3f}{r[2]:>7.3f}   {r[3]:>9.3f}{r[4]:>7.3f}{r[5]:>7.3f}")


if __name__ == "__main__":
    main()
