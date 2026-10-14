"""What did each piano grammar learn, and how alike are the two?  Probes + CKA, per OCTAVE WINDOW (GPU, short).

On frozen encoders from app/ml/grammar/piano_grammar.py (notes grammar, sound
grammar) and same-size RANDOM encoders (the floor).  Every token is one octave
window x 64 ms, so every probe reads ONE small band of the track:

  melody in band     does the human MELODY track sound in this octave window     (binary, F1)
  density in band    note onsets in this window and 64 ms                          (regression, R^2)
  mud (low bands)    two keys within 4 semitones sounding together, windows below C3 (binary, F1)
  chord root, band   the chord root (13 classes) from this ONE window's token      (accuracy)
  chord root, all    the same from all 14 windows of the moment, averaged          (accuracy)
  chord quality, all maj / min / sus / seventh / other / none                       (accuracy)

Each probe is one linear layer fitted on training clips, scored on validation.
CKA between the two encoders is taken over the same octave-window x time cells,
after removing each cell position's mean (shared position embeddings alone make
two random encoders look alike); two random encoders give the floor — how alike
notes and sound already are before any learning.

    python eval/piano_grammar_probe.py --clips /scratch/.../pop909_render --fe <spectral_real/best.pt> --grammar <out dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.grammar.piano_grammar import Encoder, tokens_notes, tokens_sound, TOK, CROP, BANDS, KEY0  # noqa: E402
from app.ml.spectral import frontend as S  # noqa: E402

N = CROP // TOK
LOW = [r for r in range(BANDS) if KEY0[r] + 12 <= 27]        # windows entirely below C3


def load_clip(p):
    z = np.load(p); lab = np.load(p.with_name(p.stem + ".lab.npz"))
    n = int(round(CROP / S.FPS * S.SR))
    y = z["y"].astype(np.float32)[:n] / 32767; y = np.pad(y, (0, n - len(y)))
    on, fr = z["on"][:, :CROP], z["fr"][:, :CROP]
    roll = np.stack([on, fr]).astype(np.float32)
    mode = lambda v: np.array([np.bincount(v[i * TOK:(i + 1) * TOK]).argmax() for i in range(N)])
    root, qual = mode(lab["root"][:CROP].astype(np.int64)), mode(lab["qual"][:CROP].astype(np.int64))
    tok = lambda a: a.reshape(a.shape[0], N, TOK)
    mel, ons, frs = tok(lab["mel"][:, :CROP]).max(-1), tok(on).sum(-1), tok(fr).max(-1)       # (88, N)
    melb = np.stack([mel[k:k + 12].max(0) for k in KEY0]).astype(np.float32)               # (R, N)
    densb = np.stack([ons[k:k + 12].sum(0) for k in KEY0]).astype(np.float32)
    clus = np.stack([[float(any(frs[k + i, t] and frs[k + i + 1:k + min(12, i + 5), t].any() for i in range(11)))
                      for t in range(N)] for k in KEY0]).astype(np.float32)
    return y, roll, dict(melb=melb, densb=densb, mudb=clus[LOW], root=root, qual=qual)


@torch.no_grad()
def features(enc, kind, clips, fe, dev):
    out = []
    for y, roll, _ in clips:
        y, roll = torch.from_numpy(y)[None].to(dev), torch.from_numpy(roll)[None].to(dev)
        x = tokens_notes(roll) if kind == "notes" else tokens_sound(fe(y)[..., :CROP])
        out.append(enc(x)[0].float().cpu())                                              # (R, N, d)
    return torch.stack(out)                                                               # (C, R, N, d)


def fit(Xtr, ytr, Xva, yva, task, steps=500):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    nout = 13 if task == "root" else 6 if task == "qual" else 1
    lin = nn.Linear(Xtr.shape[1], nout); opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        z = lin(Xtr)
        if task in ("root", "qual"):
            loss = F.cross_entropy(z, ytr.long())
        elif task == "reg":
            loss = F.mse_loss(z[:, 0], ytr)
        else:
            loss = F.binary_cross_entropy_with_logits(z[:, 0], ytr, pos_weight=torch.tensor(3.0))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = lin(Xva)
        if task in ("root", "qual"):
            return float((z.argmax(1) == yva.long()).float().mean())
        if task == "reg":
            return float(1 - F.mse_loss(z[:, 0], yva) / yva.var().clamp(min=1e-6))
        p, t = torch.sigmoid(z[:, 0]) > 0.5, yva > 0.5
        return float(2 * (p & t).sum() / max(1, int(p.sum() + t.sum())))


def lab(clips, k):
    return torch.from_numpy(np.stack([c[2][k] for c in clips]))


def run_probes(Ftr, Fva, tr, va):
    d = Ftr.shape[-1]
    band = lambda Fx: Fx.reshape(-1, d)                                                  # every (clip, window, t)
    r = {}
    r["melody in band F1"] = fit(band(Ftr), lab(tr, "melb").reshape(-1), band(Fva), lab(va, "melb").reshape(-1), "bin")
    r["density in band R2"] = fit(band(Ftr), lab(tr, "densb").reshape(-1), band(Fva), lab(va, "densb").reshape(-1), "reg")
    r["mud low bands F1"] = fit(Ftr[:, LOW].reshape(-1, d), lab(tr, "mudb").reshape(-1),
                                Fva[:, LOW].reshape(-1, d), lab(va, "mudb").reshape(-1), "bin")
    rb = lambda c: lab(c, "root")[:, None, :].expand(-1, BANDS, -1).reshape(-1)
    r["chord root, one band"] = fit(band(Ftr), rb(tr), band(Fva), rb(va), "root")
    allw = lambda Fx: Fx.mean(1).reshape(-1, d)                                          # average the 14 windows
    r["chord root, all bands"] = fit(allw(Ftr), lab(tr, "root").reshape(-1), allw(Fva), lab(va, "root").reshape(-1), "root")
    r["chord quality, all bands"] = fit(allw(Ftr), lab(tr, "qual").reshape(-1), allw(Fva), lab(va, "qual").reshape(-1), "qual")
    return r


def cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)                     # per cell position
    X = X.reshape(-1, X.shape[-1]); Y = Y.reshape(-1, Y.shape[-1])
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--grammar", required=True); ap.add_argument("--n-train", type=int, default=600)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    sel = lambda s: sorted(p for p in (pathlib.Path(a.clips) / s).glob("pop*_[0-9][0-9][0-9].npz")
                           if p.with_name(p.stem + ".lab.npz").exists())
    rng = np.random.default_rng(0)
    trp = sel("train"); trp = [trp[i] for i in sorted(rng.choice(len(trp), min(a.n_train, len(trp)), replace=False))]
    vap = sel("validation")[:300]
    tr, va = [load_clip(p) for p in trp], [load_clip(p) for p in vap]
    g = pathlib.Path(a.grammar); encs = {}
    for kind in ("notes", "sound"):
        c = torch.load(g / f"{kind}_grammar.pt", map_location=dev, weights_only=False)
        e = Encoder(c["din"]).to(dev).eval(); e.load_state_dict(c["enc"]); encs[f"{kind} grammar"] = (e, kind)
        encs[f"{kind} random"] = (Encoder(c["din"]).to(dev).eval(), kind)
    Ftr = {k: features(e, kind, tr, fe, dev) for k, (e, kind) in encs.items()}
    Fva = {k: features(e, kind, va, fe, dev) for k, (e, kind) in encs.items()}
    res = {}
    print(f"{len(tr)} train / {len(va)} validation clips; tokens = {BANDS} octave windows x {N} x 64 ms per clip\n")
    cols = ["melody in band F1", "density in band R2", "mud low bands F1", "chord root, one band",
            "chord root, all bands", "chord quality, all bands"]
    print(f"{'encoder':<15}" + "".join(f"{c:>26}" for c in cols))
    for k in encs:
        r = run_probes(Ftr[k], Fva[k], tr, va); res[k] = r
        print(f"{k:<15}" + "".join(f"{r[c]:>26.3f}" for c in cols))
    res["cka notes vs sound grammar"] = cka(Fva["notes grammar"], Fva["sound grammar"])
    res["cka notes vs sound random"] = cka(Fva["notes random"], Fva["sound random"])
    print(f"\nCKA over octave-window cells, notes grammar vs sound grammar: {res['cka notes vs sound grammar']:.3f}"
          f"   (two random encoders: {res['cka notes vs sound random']:.3f})")
    (g / "probes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
