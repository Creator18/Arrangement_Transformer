"""Same-prompt test of the two piano grammars: same music, same blanks, compare what each fills in (GPU).

The user's test.  The notes grammar and the sound grammar (app/ml/grammar/piano_grammar.py),
trained separately, get IDENTICAL prompts: the same held-out POP909 clips with the
same octave-window x 64 ms cells blanked out.  Each fills the blanks in its own form:

  notes grammar   fills in notes directly
  sound grammar   fills in the sound; its filled-in sound is spliced into the clip's
                  real sound and read by the frozen solo-piano transcriber -> notes

Both are then scored on the SAME ruler, only inside the blanked region: which keys
sound at each frame (frame F1) and where notes start (onset F1, +-2 frames), at each
method's best threshold — against the true notes, and against EACH OTHER (agreement).

Baselines: the transcriber on the sound with the blanks left empty (what it guesses
without any grammar), and "nothing" for the notes (all silent).

Prompts:
  random cells   30% of cells, in runs of 4 tokens (0.26 s) within a window
  one band, 2 s  one octave window blanked for 2 s (rotates through the middle register)
  gap, 0.5 s     every window blanked for 0.5 s (a hole in time)
  bass, 4 s      the four lowest windows blanked for 4 s

    python eval/grammar_prompt_test.py --clips /scratch/.../pop909_render --fe <spectral_real/best.pt> --grammar <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.grammar.piano_grammar import (Encoder, tokens_notes, tokens_sound, TOK, CROP, BANDS, KEY0,  # noqa: E402
                                          DN, DS)
from app.ml.spectral import frontend as S  # noqa: E402
from app.ml.spectral.model import from_checkpoint  # noqa: E402

N = CROP // TOK
THRS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def prompt_mask(kind, i, rng):
    m = np.zeros((BANDS, N), bool)
    if kind == "random cells":
        for _ in range(int(BANDS * N * 0.3 / 4)):
            r, t = rng.integers(0, BANDS), rng.integers(0, N - 4); m[r, t:t + 4] = True
    elif kind == "one band, 2 s":
        r = 4 + i % 6; t = rng.integers(0, N - 32); m[r, t:t + 32] = True
    elif kind == "gap, 0.5 s":
        t = rng.integers(0, N - 8); m[:, t:t + 8] = True
    elif kind == "bass, 4 s":
        t = rng.integers(0, N - 63); m[:4, t:t + 63] = True
    return m


def untok_notes(tok):                         # (R, N, DN) -> (2, 88, CROP), averaging overlapping windows
    out = torch.zeros(2, 88, CROP, device=tok.device); cnt = torch.zeros_like(out)
    t = tok.view(BANDS, N, 2, 12, TOK).permute(0, 2, 3, 1, 4).reshape(BANDS, 2, 12, CROP)
    for r, k in enumerate(KEY0):
        out[:, k:k + 12] += t[r]; cnt[:, k:k + 12] += 1
    return out / cnt.clamp(min=1)


def untok_sound(tok, K=S.N_BINS):             # (R, N, DS) -> (7, 264, CROP)
    C = len(S.HARMONICS) + 1
    out = torch.zeros(C, K, CROP, device=tok.device); cnt = torch.zeros_like(out)
    t = tok.view(BANDS, N, C, 36, TOK).permute(0, 2, 3, 1, 4).reshape(BANDS, C, 36, CROP)
    for r, k in enumerate(KEY0):
        out[:, 3 * k:3 * k + 36] += t[r]; cnt[:, 3 * k:3 * k + 36] += 1
    return out / cnt.clamp(min=1)


def cells_to_region(m):                       # (R, N) cells -> (88, CROP) keys x frames
    reg = np.zeros((88, CROP), bool)
    for r, k in enumerate(KEY0):
        for t in np.flatnonzero(m[r]):
            reg[k:k + 12, t * TOK:(t + 1) * TOK] = True
    return reg


def cells_to_bins(m):                         # (R, N) -> (264, CROP)
    reg = np.zeros((S.N_BINS, CROP), bool)
    for r, k in enumerate(KEY0):
        for t in np.flatnonzero(m[r]):
            reg[3 * k:3 * k + 36, t * TOK:(t + 1) * TOK] = True
    return reg


class Acc:
    """Counts inside the blanked region, per threshold: frame (sounding) and onset (+-2 frames)."""

    def __init__(self):
        self.f = np.zeros((len(THRS), 3)); self.o = np.zeros((len(THRS), 3))

    def add(self, p_fr, p_on, g_fr, g_on, reg):
        g_on_d = np.maximum.reduce([np.roll(g_on, s, 1) for s in (-2, -1, 0, 1, 2)])     # +-2 frames of a true onset
        for i, t in enumerate(THRS):
            pf, po = (p_fr > t) & reg, (p_on > t) & reg
            self.f[i] += [(pf & g_fr).sum(), pf.sum(), (g_fr & reg).sum()]
            self.o[i] += [(po & g_on_d).sum(), po.sum(), (g_on & reg).sum()]

    def best(self, c):
        f1 = 2 * c[:, 0] / np.maximum(1, c[:, 1] + c[:, 2]); i = int(f1.argmax())
        return float(f1[i]), THRS[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True); ap.add_argument("--fe", required=True)
    ap.add_argument("--grammar", required=True); ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.fe, map_location=dev, weights_only=False)
    fe = S.SpectralFrontend().to(dev).eval(); fe.mu.copy_(ck["fe_mu"]); fe.sd.copy_(ck["fe_sd"])
    tx = from_checkpoint(ck).to(dev).eval()                                     # the solo-piano transcriber
    g = pathlib.Path(a.grammar); M = {}
    for kind in ("notes", "sound"):
        c = torch.load(g / f"{kind}_grammar.pt", map_location=dev, weights_only=False)
        e = Encoder(c["din"]).to(dev).eval(); e.load_state_dict(c["enc"])
        h = torch.nn.Linear(256, c["din"]).to(dev); h.load_state_dict(c["head"]); M[kind] = (e, h)
    clips = sorted(pathlib.Path(a.clips, "validation").glob("pop*_[0-9][0-9][0-9].npz"))[: a.n]
    prompts = ("random cells", "one band, 2 s", "gap, 0.5 s", "bass, 4 s")
    acc = {p: {k: Acc() for k in ("notes grammar", "sound grammar", "transcriber, blanks empty",
                                   "agreement notes vs sound")} for p in prompts}
    with torch.no_grad():
        for i, path in enumerate(clips):
            z = np.load(path)
            n = int(round(CROP / S.FPS * S.SR))
            y = np.pad(z["y"].astype(np.float32)[:n] / 32767, (0, max(0, n - len(z["y"][:n]))))
            g_on, g_fr = z["on"][:, :CROP] > 0, z["fr"][:, :CROP] > 0
            roll = torch.from_numpy(np.stack([g_on, g_fr]).astype(np.float32))[None].to(dev)
            feat = fe(torch.from_numpy(y)[None].to(dev))[..., :CROP]
            tn, ts = tokens_notes(roll), tokens_sound(feat)
            for p in prompts:
                m = prompt_mask(p, i, np.random.default_rng(1000 + i))
                mt = torch.from_numpy(m)[None].to(dev); reg = cells_to_region(m)
                en, hn = M["notes"]; pn = torch.sigmoid(untok_notes(hn(en(tn, mt))[0]).float()).cpu().numpy()
                es, hs = M["sound"]; ss = untok_sound(hs(es(ts, mt))[0]).float()
                binreg = torch.from_numpy(cells_to_bins(m)).to(dev)[None]
                spliced = torch.where(binreg, ss, feat[0])[None]
                emptied = torch.where(binreg, torch.zeros_like(feat[0]), feat[0])[None]
                read = lambda f: [torch.sigmoid(o[0].float()).T.cpu().numpy() for o in tx(f)[:2]]   # (88, T) onset, frame
                so, sf = read(spliced); eo, ef = read(emptied)
                A = acc[p]
                A["notes grammar"].add(pn[1], pn[0], g_fr, g_on, reg)
                A["sound grammar"].add(sf, so, g_fr, g_on, reg)
                A["transcriber, blanks empty"].add(ef, eo, g_fr, g_on, reg)
                A["agreement notes vs sound"].add(pn[1], pn[0], sf > 0.5, so > 0.5, reg)
    res = {}
    print(f"{len(clips)} held-out POP909 clips (8 s); every prompt identical for both grammars; scored inside the blanks only\n")
    for p in prompts:
        print(f"== prompt: {p}")
        print(f"   {'':<30}{'frame F1 (thr)':>18}{'onset F1 (thr)':>18}")
        for k, A in acc[p].items():
            fb, ft = A.best(A.f); ob, ot = A.best(A.o)
            res.setdefault(p, {})[k] = dict(frame=fb, frame_thr=ft, onset=ob, onset_thr=ot)
            print(f"   {k:<30}{fb:>12.3f} ({ft}){ob:>12.3f} ({ot})")
        print()
    (g / "prompt_test.json").write_text(json.dumps(res, indent=1))
    print("agreement = the notes grammar's fill-in scored against the sound grammar's (via the transcriber, at 0.5)")


if __name__ == "__main__":
    main()
