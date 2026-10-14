"""What did each piano grammar learn, and how alike are the two?  Probes + CKA (GPU, short).

On frozen encoders from app/ml/grammar/piano_grammar.py (notes grammar, sound
grammar) and same-size RANDOM encoders (the floor), per 64 ms token:

  theory   chord root (13 classes), chord quality (6)      POP909 human labels
           melody: which keys the MELODY track is playing   (88 binary, F1)
  sound    density: number of note onsets in the token      (regression, R^2)
           low-register mud: two keys below C3 within 4 semitones sounding together (binary, F1)

Each probe is one linear layer fitted on training clips, scored on validation.
Then linear CKA between the notes and sound encoders' features on the same
validation tokens (1 = the same picture up to rotation), against the CKA of two
random encoders.

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
from app.ml.grammar.piano_grammar import Encoder, tokens_notes, tokens_sound, TOK, CROP  # noqa: E402
from app.ml.spectral import frontend as S  # noqa: E402


def load_clip(p):
    z = np.load(p); lab = np.load(p.with_name(p.stem + ".lab.npz"))
    n = int(round(CROP / S.FPS * S.SR))
    y = np.pad(z["y"].astype(np.float32)[:n] / 32767, (0, max(0, n - len(z["y"][:n]))))
    roll = np.stack([z["on"][:, :CROP], z["fr"][:, :CROP]]).astype(np.float32)
    N = CROP // TOK
    mode = lambda v: np.array([np.bincount(v[i * TOK:(i + 1) * TOK]).argmax() for i in range(N)])
    root, qual = mode(lab["root"][:CROP].astype(np.int64)), mode(lab["qual"][:CROP].astype(np.int64))
    mel = lab["mel"][:, :CROP].reshape(88, N, TOK).max(-1).T                         # (N, 88)
    dens = z["on"][:, :CROP].reshape(88, N, TOK).sum((0, 2)).astype(np.float32)
    low = z["fr"][:27, :CROP].reshape(27, N, TOK).max(-1).T                          # keys below C3
    mud = np.array([any(low[i, k] and low[i, k + 1:k + 5].any() for k in range(27)) for i in range(N)])
    return y, roll, dict(root=root, qual=qual, mel=mel, dens=dens, mud=mud.astype(np.float32))


@torch.no_grad()
def features(enc, kind, clips, fe, dev):
    out = []
    for y, roll, _ in clips:
        y, roll = torch.from_numpy(y)[None].to(dev), torch.from_numpy(roll)[None].to(dev)
        x = tokens_notes(roll) if kind == "notes" else tokens_sound(fe(y)[..., :CROP])
        out.append(enc(x)[0].float().cpu())
    return torch.cat(out)                                                              # (sum N, 256)


def stack(clips, k):
    v = torch.from_numpy(np.concatenate([c[2][k] for c in clips]))
    return v if k in ("root", "qual") else v.float()


def probe(Xtr, ytr, Xva, yva, kind, steps=600):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    nout = {"root": 13, "qual": 6}.get(kind, ytr.shape[1] if ytr.ndim == 2 else 1)
    lin = nn.Linear(Xtr.shape[1], nout); opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        z = lin(Xtr)
        if kind in ("root", "qual"):
            loss = F.cross_entropy(z, ytr.long())
        elif kind == "dens":
            loss = F.mse_loss(z[:, 0], ytr)
        else:
            t = ytr if ytr.ndim == 2 else ytr[:, None]
            loss = F.binary_cross_entropy_with_logits(z, t, pos_weight=torch.tensor(5.0))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = lin(Xva)
        if kind in ("root", "qual"):
            return float((z.argmax(1) == yva.long()).float().mean())
        if kind == "dens":
            return float(1 - F.mse_loss(z[:, 0], yva) / yva.var().clamp(min=1e-6))
        t = (yva if yva.ndim == 2 else yva[:, None]) > 0.5; p = torch.sigmoid(z) > 0.5
        return float(2 * (p & t).sum() / max(1, int(p.sum() + t.sum())))


def cka(X, Y, n=CROP // TOK):
    """Linear CKA after removing each token POSITION's mean: every encoder carries the same
    position embedding, which alone makes two random encoders look alike (0.94 before this)."""
    X = X.view(-1, n, X.shape[-1]); Y = Y.view(-1, n, Y.shape[-1])
    X = (X - X.mean(0, keepdim=True)).reshape(-1, X.shape[-1]); Y = (Y - Y.mean(0, keepdim=True)).reshape(-1, Y.shape[-1])
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--grammar", required=True); ap.add_argument("--n-train", type=int, default=1200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    sel = lambda s, n: sorted(p for p in (pathlib.Path(a.clips) / s).glob("pop*_[0-9][0-9][0-9].npz")
                              if p.with_name(p.stem + ".lab.npz").exists())[::1][:n]
    rng = np.random.default_rng(0)
    trp = sel("train", 100000); trp = [trp[i] for i in sorted(rng.choice(len(trp), min(a.n_train, len(trp)), replace=False))]
    tr, va = [load_clip(p) for p in trp], [load_clip(p) for p in sel("validation", 300)]
    g = pathlib.Path(a.grammar)
    encs = {}
    for kind in ("notes", "sound"):
        c = torch.load(g / f"{kind}_grammar.pt", map_location=dev, weights_only=False)
        e = Encoder(c["din"]).to(dev).eval(); e.load_state_dict(c["enc"]); encs[f"{kind} grammar"] = (e, kind)
        encs[f"{kind} random"] = (Encoder(c["din"]).to(dev).eval(), kind)
    F_tr = {k: features(e, kind, tr, fe, dev) for k, (e, kind) in encs.items()}
    F_va = {k: features(e, kind, va, fe, dev) for k, (e, kind) in encs.items()}
    res = {}
    print(f"{len(tr)} train / {len(va)} validation clips, {len(F_va['notes grammar'])} validation tokens\n")
    print(f"{'encoder':<16}{'root acc':>10}{'quality acc':>13}{'melody F1':>11}{'density R2':>12}{'low mud F1':>12}")
    for k in encs:
        r = {lab: probe(F_tr[k], stack(tr, lab), F_va[k], stack(va, lab), lab)
             for lab in ("root", "qual", "mel", "dens", "mud")}
        res[k] = r
        print(f"{k:<16}{r['root']:>10.3f}{r['qual']:>13.3f}{r['mel']:>11.3f}{r['dens']:>12.3f}{r['mud']:>12.3f}")
    res["cka notes vs sound grammar"] = cka(F_va["notes grammar"], F_va["sound grammar"])
    res["cka notes vs sound random"] = cka(F_va["notes random"], F_va["sound random"])
    print(f"\nCKA notes grammar vs sound grammar: {res['cka notes vs sound grammar']:.3f}   "
          f"(two random encoders: {res['cka notes vs sound random']:.3f})")
    (g / "probes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
