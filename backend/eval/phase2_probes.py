"""Phase 2 probe gate: does the pitch-aligned patch design make notes easier to read?

Small probes trained on FROZEN front-end features — no big model yet.  If the
representation does not carry the information more accessibly than the
standard alternative, a model built on it will not either, and a week of
training would be spent finding that out.

FRONT ENDS, each adding one design choice so its contribution is isolated:

    mel AST 16x16   standard Audio Spectrogram Transformer patches (the baseline)
    mel tall-thin   same mel, but our patch SHAPE  -> isolates patch geometry
    CQT             pitch-aligned axis, fundamental only  -> isolates the axis
    HCQT            + harmonic channels  -> isolates harmonic stacking
    HCQT+onset      + short-window onset stream  -> isolates the timing channel

PROBES

  P1/P2 PITCH + ONSET  per 32 ms column: which keys are sounding, which were
        struck.  Threshold-free average precision, so no front end is helped
        or hurt by where a threshold lands.  Linear and 1-hidden-layer probes.

  P3 TRANSPOSITION  chord quality (7 classes) trained on roots C2-B3, tested
        on roots C5-B5, two octaves the probe never saw.  The probe shares one
        MLP across patch ROWS and max-pools them, so it can generalise across
        register only if the representation turns a transposition into a
        vertical SHIFT — the central claim of the CQT design.  Controls:
        a non-shared (flat) probe on HCQT, and 12-bin chroma, which is octave-
        invariant by construction and so should pass; matching it shows the
        CQT keeps chord shape across register WITHOUT folding register away,
        which the arrangement needs and chroma cannot give.

Every probe runs over several seeds; differences inside the seed spread are
not differences.

    venv/Scripts/python.exe eval/phase2_probes.py [--seeds 3]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.ml.spectral import frontend as S        # noqa: E402
from app.ml.spectral import render as R          # noqa: E402

DEV = torch.device("cuda")
H1 = S.HARMONICS.index(1)

# name -> (feature source, patch h, h step, patch w, w step)
FRONTENDS = {
    "mel AST 16x16": ("mel", 16, 16, 16, 16),
    "mel tall-thin": ("mel", 16, 8, 2, 2),
    "CQT":           ("cqt", 36, 18, 2, 2),
    "HCQT":          ("hcqt", 36, 18, 2, 2),
    "HCQT+onset":    ("hcqton", 36, 18, 2, 2),
}

QUALITIES = {"maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
             "dom7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10]}


# ── features ──────────────────────────────────────────────────────────────────
class Feats(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        self.fe = S.SpectralFrontend()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=S.SR, n_fft=2048, hop_length=S.HOP, f_min=27.5,
            f_max=S.SR / 2, n_mels=128, power=1.0, center=True)
        self.stats = {}

    @torch.no_grad()
    def raw(self, w):
        r = self.fe.raw(w)
        return {"mel": torch.log1p(S.LOG_GAIN * self.mel(w)).unsqueeze(1),
                "cqt": r[:, H1:H1 + 1], "hcqt": r[:, :len(S.HARMONICS)], "hcqton": r}

    @torch.no_grad()
    def calibrate(self, wavs):
        acc = {}
        for w in wavs:
            for k, x in self.raw(w).items():
                s, s2, n = acc.get(k, (0, 0, 0))
                acc[k] = (s + x.sum((0, 3), keepdim=True),
                          s2 + (x ** 2).sum((0, 3), keepdim=True),
                          n + x.shape[0] * x.shape[-1])
        for k, (s, s2, n) in acc.items():
            mu = s / n
            self.stats[k] = (mu, torch.sqrt((s2 / n - mu ** 2).clamp(min=1e-6)))

    @torch.no_grad()
    def __call__(self, w):
        out = {}
        for k, x in self.raw(w).items():
            mu, sd = self.stats[k]
            out[k] = (x - mu) / sd
        return out


def patch(x, ph, phs, pw, pws):
    """(B, C, F, T) -> (B, rows, cols, C*ph*pw)."""
    B, C, Fb, T = x.shape
    rows = math.ceil(max(Fb - ph, 0) / phs) + 1
    Fp = (rows - 1) * phs + ph
    Tp = math.ceil(T / pw) * pw
    x = F.pad(x, (0, Tp - T, 0, Fp - Fb))
    p = x.unfold(2, ph, phs).unfold(3, pw, pws)
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, rows, p.shape[3], C * ph * pw)


def columns(tok, pw, n_cols):
    """Per 2-frame column, all rows concatenated: (B*n_cols, rows*D).

    A patch wider than one column (AST, 16 frames) is shared by every column
    it covers — exactly the time resolution that geometry gives a model.
    """
    B, rows, cols, D = tok.shape
    rep = pw // 2
    tok = tok.repeat_interleave(rep, dim=2)[:, :, :n_cols]
    return tok.permute(0, 2, 1, 3).reshape(B * n_cols, rows * D)


def col_labels(r):
    """(B, 88, T) uint8 -> (B, ceil(T/2), 88) union over each column's 2 frames."""
    B, K, T = r.shape
    r = F.pad(r, (0, T % 2))
    return r.view(B, K, -1, 2).amax(-1).transpose(1, 2)


# ── P1/P2: pitch + onset ──────────────────────────────────────────────────────
def micro_ap(score, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y.ravel(), score.ravel()))


def run_frame_probe(feats, name, kind, data, seed, epochs=10):
    src, ph, phs, pw, pws = FRONTENDS[name]
    torch.manual_seed(seed); random.seed(seed)
    (trw, tron, trfr), (tew, teon, tefr) = data
    with torch.no_grad():
        f = feats(trw[:2].to(DEV))[src]
        D = columns(patch(f, ph, phs, pw, pws), pw, math.ceil(f.shape[-1] / 2)).shape[1]
    if kind == "linear":
        net = nn.Linear(D, 2 * S.N_KEYS)
    else:
        net = nn.Sequential(nn.Linear(D, 512), nn.ReLU(), nn.Linear(512, 2 * S.N_KEYS))
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    pw_fr = torch.tensor(2.0, device=DEV); pw_on = torch.tensor(10.0, device=DEV)
    idx = list(range(len(trw)))

    def batch_xy(w, on, fr):
        f = feats(w.to(DEV))[src]
        nc = math.ceil(f.shape[-1] / 2)
        x = columns(patch(f, ph, phs, pw, pws), pw, nc)
        yf = col_labels(fr.to(DEV)).reshape(-1, S.N_KEYS).float()
        yo = col_labels(on.to(DEV)).reshape(-1, S.N_KEYS).float()
        return x, yf, yo

    for _ in range(epochs):
        random.shuffle(idx)
        for i in range(0, len(idx), 8):
            b = idx[i:i + 8]
            x, yf, yo = batch_xy(trw[b], tron[b], trfr[b])
            out = net(x)
            loss = (F.binary_cross_entropy_with_logits(out[:, :88], yf, pos_weight=pw_fr)
                    + F.binary_cross_entropy_with_logits(out[:, 88:], yo, pos_weight=pw_on))
            opt.zero_grad(); loss.backward(); opt.step()

    sf, so, lf, lo = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(tew), 8):
            x, yf, yo = batch_xy(tew[i:i + 8], teon[i:i + 8], tefr[i:i + 8])
            out = torch.sigmoid(net(x)).cpu().numpy()
            sf.append(out[:, :88]); so.append(out[:, 88:])
            lf.append(yf.cpu().numpy()); lo.append(yo.cpu().numpy())
    return (micro_ap(np.concatenate(sf), np.concatenate(lf)),
            micro_ap(np.concatenate(so), np.concatenate(lo)))


# ── P3: transposition ─────────────────────────────────────────────────────────
def chord_set(roots, n_per, seed):
    from app.services.sampler import render_notes
    rng = random.Random(seed)
    ys, labs = [], []
    names = list(QUALITIES)
    for qi, q in enumerate(names):
        for _ in range(n_per):
            root = rng.choice(roots)
            ps = [root + iv for iv in QUALITIES[q]]
            for _ in range(rng.randrange(len(ps))):        # random inversion
                ps = sorted(ps)[1:] + [sorted(ps)[0] + 12]
            notes = [{"pitch": p, "onset": 0.0, "offset": 0.9, "velocity": 80} for p in ps]
            y = render_notes(notes, sr=S.SR).astype(np.float32)[:S.SR]
            ys.append(np.pad(y, (0, S.SR - len(y)))); labs.append(qi)
    return np.stack(ys), np.array(labs)


def chords_cached():
    p = R.CACHE / "chords.npz"
    if p.exists():
        z = np.load(p)
        return {k: (z[f"{k}_y"], z[f"{k}_l"]) for k in ("train", "in", "out")}
    low, high = list(range(36, 60)), list(range(72, 84))
    sets = {"train": chord_set(low, 120, 1), "in": chord_set(low, 30, 2),
            "out": chord_set(high, 30, 3)}
    np.savez_compressed(p, **{f"{k}_y": v[0] for k, v in sets.items()},
                        **{f"{k}_l": v[1] for k, v in sets.items()})
    return sets


class SharedRows(nn.Module):
    """One MLP applied to every patch row, max over rows: shift-equivariant in pitch."""
    def __init__(self, D, n_cls):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.out = nn.Linear(256, n_cls)

    def forward(self, rows):                       # (B, rows, D)
        return self.out(self.f(rows).amax(1))


def chroma(x_cqt):
    """(B, 1, 264, T) -> (B, 12) pitch-class energy; A0 is pitch class 9."""
    keys = x_cqt[:, 0].view(x_cqt.shape[0], S.N_KEYS, 3, -1).mean(2).mean(-1)   # (B, 88)
    pc = (torch.arange(S.N_KEYS, device=keys.device) + 9) % 12
    return torch.zeros(keys.shape[0], 12, device=keys.device).index_add_(1, pc, keys)


def run_chord_probe(feats, variant, sets, seed, epochs=40):
    torch.manual_seed(seed)
    n_cls = len(QUALITIES)

    def featurize(y):
        out = []
        for i in range(0, len(y), 64):
            f = feats(torch.from_numpy(y[i:i + 64]).to(DEV))
            if variant == "chroma":
                out.append(chroma(f["cqt"]))
            else:
                src, ph, phs, pw, pws = FRONTENDS[variant.replace(" (flat)", "")]
                out.append(patch(f[src], ph, phs, pw, pws).mean(2))   # (B, rows, D)
        return torch.cat(out)

    Xtr = featurize(sets["train"][0]); ytr = torch.from_numpy(sets["train"][1]).long().to(DEV)
    if variant == "chroma":
        net = nn.Sequential(nn.Linear(12, 256), nn.ReLU(), nn.Linear(256, n_cls))
        fwd = net
    elif variant.endswith("(flat)"):
        net = nn.Sequential(nn.Flatten(), nn.Linear(Xtr.shape[1] * Xtr.shape[2], 256),
                            nn.ReLU(), nn.Linear(256, n_cls))
        fwd = net
    else:
        net = SharedRows(Xtr.shape[2], n_cls)
        fwd = net
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = F.cross_entropy(fwd(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    acc = {}
    with torch.no_grad():
        for k in ("in", "out"):
            X = featurize(sets[k][0])
            acc[k] = float((fwd(X).argmax(1).cpu().numpy() == sets[k][1]).mean())
    return acc["in"], acc["out"]


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--only-p3", action="store_true")
    args = ap.parse_args()

    def stack(split, n):
        ys, ons, frs = zip(*(R.load(p) for p in R.build(split, n)))
        return (torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ons)),
                torch.from_numpy(np.stack(frs)))

    tr, te = stack("train", args.train), stack("test", args.test)
    print(f"rendered MAESTRO: {len(tr[0])} train / {len(te[0])} test clips of 20 s, "
          f"split by piece; {int(te[1].sum())} test onsets\n")

    feats = Feats().to(DEV)
    feats.calibrate([tr[0][i:i + 8].to(DEV) for i in range(0, len(tr[0]), 8)])

    seeds = list(range(args.seeds))
    res = {}
    if args.only_p3:
        return run_p3(feats, seeds, res)
    print("P1/P2  PITCH + ONSET  — micro average precision on held-out pieces "
          f"(mean ± sd over {len(seeds)} seeds)\n")
    print(f"{'front end':<16}{'probe':<8}{'pitch AP':>16}{'onset AP':>16}")
    print("-" * 56)
    for name in FRONTENDS:
        for kind in ("linear", "mlp"):
            r = np.array([run_frame_probe(feats, name, kind, (tr, te), s) for s in seeds])
            res[(name, kind)] = r
            m, sd = r.mean(0), r.std(0)
            print(f"{name:<16}{kind:<8}{m[0]:>9.3f} ±{sd[0]:.3f}{m[1]:>9.3f} ±{sd[1]:.3f}",
                  flush=True)
    run_p3(feats, seeds, res)


def run_p3(feats, seeds, res):
    print("\nP3  TRANSPOSITION — chord quality, 7 classes, chance 0.143")
    print("    trained on roots C2-B3; 'unseen' = roots C5-B5, never in training\n")
    sets = chords_cached()
    print(f"{'representation':<22}{'same register':>16}{'unseen register':>18}")
    print("-" * 56)
    variants = ["chroma", "mel tall-thin", "CQT", "HCQT", "HCQT (flat)"]
    cres = {}
    # Normalisation stays calibrated on the full-register MAESTRO corpus.
    # Recalibrating on the chord TRAINING set (roots C2-B3 only) gave every
    # high band a near-zero std, so unseen high-register chords were blown up
    # into extreme z-scores: per-band normalisation is position-specific, and
    # calibrating it on half the register breaks the very shift-invariance this
    # probe measures.  Chroma, octave-invariant by construction, dropping from
    # 0.848 to 0.537 was the tell.
    for v in variants:
        r = np.array([run_chord_probe(feats, v, sets, s) for s in seeds])
        cres[v] = r
        m, sd = r.mean(0), r.std(0)
        print(f"{v:<22}{m[0]:>9.3f} ±{sd[0]:.3f}{m[1]:>11.3f} ±{sd[1]:.3f}", flush=True)

    if not res:
        out_hcqt, out_mel = cres["HCQT"][:, 1].mean(), cres["mel tall-thin"][:, 1].mean()
        print(f"\nGATE  transposition: HCQT unseen {out_hcqt:.3f} vs mel unseen "
              f"{out_mel:.3f} -> {'PASS' if out_hcqt > out_mel + 0.1 else 'FAIL'}")
        return
    # ── gate ──
    def best(names, col):
        return max(res[(n, k)][:, col].mean() for n in names for k in ("linear", "mlp"))
    mel_p, cqt_p = best(["mel AST 16x16", "mel tall-thin"], 0), best(["HCQT", "HCQT+onset"], 0)
    hc_on, hcon_on = best(["HCQT"], 1), best(["HCQT+onset"], 1)
    out_hcqt, out_mel = cres["HCQT"][:, 1].mean(), cres["mel tall-thin"][:, 1].mean()
    print("\nGATE")
    print(f"  pitch:          HCQT family {cqt_p:.3f}  vs best mel {mel_p:.3f}  "
          f"-> {'PASS' if cqt_p > mel_p else 'FAIL'}")
    print(f"  onset channel:  HCQT+onset {hcon_on:.3f}  vs HCQT {hc_on:.3f}  "
          f"-> {'PASS' if hcon_on > hc_on else 'FAIL'}")
    print(f"  transposition:  HCQT unseen {out_hcqt:.3f}  vs mel unseen {out_mel:.3f}  "
          f"-> {'PASS' if out_hcqt > out_mel + 0.1 else 'FAIL'}")


if __name__ == "__main__":
    main()
