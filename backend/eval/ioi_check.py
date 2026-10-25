"""Are quick notes missed because of time resolution?  Recall of written onsets binned by the gap to
the PREVIOUS onset (any key, and the same key), on held-out renders whose labels are exact, at the
app's decoding (onset 0.9, frame 0.5, short notes and re-strikes kept) and at lower onset
thresholds.  If misses pile up under ~30-60 ms, resolution is the limit (the transformer's tokens
are 2 frames = 32 ms); if they are spread out, it is the threshold or the training data.

    python eval/ioi_check.py --ckpt <best.pt> --clips <dir with *.npz> [--n 300]
"""
import argparse, os, pathlib, random, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral import frontend as S                     # noqa: E402
from app.ml.transcriber.decode import decode_notes            # noqa: E402
from app.services import spectral_transcription as st         # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from context_decode import context_decode                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--clips", required=True)
ap.add_argument("--fine", action="store_true", help="finer gap bins (16 ms steps up to 64 ms) + the 16 vs 8 ms grid ceiling")
ap.add_argument("--no-variants", action="store_true", help="only the plain thresholds")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--tol", type=int, default=3, help="match window, frames (+-)")
ap.add_argument("--bandlimit", type=int, default=0,
                help="resample each clip to this rate and back first (16000 = how the Music2MIDI songs are stored: nothing above 8 kHz)")
ap.add_argument("--switches", default="oracle", choices=["oracle", "off", "all"], help="joint checkpoints: which adapters are on")
ap.add_argument("--joint-init", action="store_true", help="joint checkpoints: the adapters as trained alone (stage 5 step 0)")
a = ap.parse_args()
import joint_load as JL                                         # stage-5 joint checkpoints need per-clip switches
JOINT = JL.is_joint(a.ckpt)
if JOINT:
    net, fe, dev = JL.load(a.ckpt, a.switches, a.joint_init)
else:
    st.CKPT = pathlib.Path(a.ckpt); net, fe, dev = st._load()
files = sorted(p for p in pathlib.Path(a.clips).glob("*.npz") if ".tmp" not in p.name)
files = random.Random(0).sample(files, min(a.n, len(files)))
# decoders: the app's single threshold, and the context-aware variants (eval/context_decode.py)
KW = dict(min_frames=1, min_note_s=0.0, keep_restruck=True)
DEC = {"0.5": lambda o, f, v: decode_notes(o, f, v, 0.5, 0.5, **KW),
       "0.7": lambda o, f, v: decode_notes(o, f, v, 0.7, 0.5, **KW),
       "app 0.9": lambda o, f, v: decode_notes(o, f, v, 0.9, 0.5, **KW)}
for mode in ("same", "any"):
    for low in (0.6, 0.7):
        for win in (9, 15):                                       # 144 / 240 ms after a strike
            DEC[f"{mode} {low}/{win * 16}ms"] = (lambda m, l, w: lambda o, f, v: context_decode(o, f, v, 0.9, l, w, m))(mode, low, win)
if a.no_variants:
    DEC = {k: v for k, v in DEC.items() if k in ("0.5", "0.7", "app 0.9")}
    DEC["0.9 + merge 60ms"] = lambda o, f, v: decode_notes(o, f, v, 0.9, 0.5, merge_fragment_s=0.06, **KW)
    DEC["0.9 + merge 100ms"] = lambda o, f, v: decode_notes(o, f, v, 0.9, 0.5, merge_fragment_s=0.10, **KW)
    DEC["0.95"] = lambda o, f, v: decode_notes(o, f, v, 0.95, 0.5, **KW)
    DEC["0.97"] = lambda o, f, v: decode_notes(o, f, v, 0.97, 0.5, **KW)
THR = tuple(DEC)
GAP_EDGES = [0, 2, 4, 6, 9, 1000]            # frames: <32, 32-64, 64-96, 96-144, >=144 ms
names = ["<32 ms", "32-64", "64-96", "96-144", ">=144"]
if a.fine:
    GAP_EDGES = [0, 1, 2, 3, 4, 6, 9, 1000]
    names = ["<16 ms", "16-32", "32-48", "48-64", "64-96", "96-144", ">=144"]
NB = len(names)
grid = {"notes": 0, "16": 0, "8": 0}      # notes a grid of that step cannot separate (same key, < 2 steps apart)
KINDS = ("any", "same", "sung", "other", "piano", "octave", "no_octave")
dbl = {}                                     # decoder -> extra presses on a key the song is still holding
hit = {t: {k: np.zeros(NB) for k in KINDS} for t in THR}; tot = {k: np.zeros(NB) for k in KINDS}
fp = {t: 0 for t in THR}; npred = {t: 0 for t in THR}
for f in files:
    z = np.load(f); y = z["y"].astype(np.float32) / 32767; on = z["on"]
    if a.bandlimit:
        import librosa
        y = librosa.resample(librosa.resample(y, orig_sr=S.SR, target_sr=a.bandlimit), orig_sr=a.bandlimit, target_sr=S.SR)[:len(y)].astype(np.float32)
    if JOINT:
        JL.set_clip(z)
    with torch.no_grad():
        x = fe(torch.from_numpy(y)[None].to(dev))
        o = net(x, None)
        po, pf, pv = (torch.sigmoid(t.float())[0].T.cpu().numpy() for t in o[:3])
    T = min(on.shape[1], po.shape[1]); on = on[:, :T]
    gt = sorted(zip(*np.nonzero(on)), key=lambda kf: (kf[1], kf[0]))      # (key, frame)
    meta = z["note_meta"] if "note_meta" in z.files else np.zeros((0, 9))
    sung_at = {(int(p_) - 21, int(round(o_ * S.FPS))) for o_, p_, *r_ in meta if len(r_) >= 7 and r_[6] > 0.5}
    is_multi = str(z["piano"]) == "multi"
    frl = z["fr"][:, :on.shape[1]] > 0
    snd = np.maximum(frl, on > 0)
    if a.fine and len(meta):                                       # exact onsets -> what a 16 / 8 ms grid can resolve
        for kk in np.unique(meta[:, 1]):
            o_ = np.sort(meta[meta[:, 1] == kk, 0]); grid["notes"] += len(o_)
            for step, key in ((0.016, "16"), (0.008, "8")):
                q = np.round(o_ / step); grid[key] += int(np.sum(np.diff(q) < 2))
    frames_all = np.array(sorted({f_ for _, f_ in gt}))
    last_same = {}
    rows = []
    for k, fr_ in gt:
        prev = frames_all[frames_all < fr_]
        g_any = fr_ - prev[-1] if len(prev) else 999
        g_same = fr_ - last_same[k] if k in last_same else 999
        last_same[k] = fr_
        kd = ("sung" if {(k, fr_ - 1), (k, fr_), (k, fr_ + 1)} & sung_at else "other") if is_multi else "piano"
        rows.append((k, fr_, np.searchsorted(GAP_EDGES, g_any, "right") - 1, np.searchsorted(GAP_EDGES, g_same, "right") - 1, kd))
    rows = [r + (("octave" if any(r[0] - o >= 0 and snd[r[0] - o, r[1]] for o in (12, 24, 36)) else "no_octave"),) for r in rows]
    for _, _, ba, bs, kd, ko in rows:
        tot["any"][min(ba, NB - 1)] += 1; tot["same"][min(bs, NB - 1)] += 1; tot[kd][min(ba, NB - 1)] += 1
        tot[ko][min(ba, NB - 1)] += 1
    for t in THR:
        notes = DEC[t](po[:, :T], pf[:, :T], pv[:, :T])
        pred = {}
        for n in notes:
            pred.setdefault(int(n["pitch"]) - 21, []).append(int(round(n["onset"] * S.FPS)))
        used = set()
        for k, fr_, ba, bs, kd, ko in rows:
            c = [q for q in pred.get(k, []) if abs(q - fr_) <= a.tol and (k, q) not in used]
            if c:
                q = min(c, key=lambda q: abs(q - fr_)); used.add((k, q))
                hit[t]["any"][min(ba, NB - 1)] += 1; hit[t]["same"][min(bs, NB - 1)] += 1; hit[t][kd][min(ba, NB - 1)] += 1
                hit[t][ko][min(ba, NB - 1)] += 1
        n_p = sum(len(v) for v in pred.values()); npred[t] += n_p; fp[t] += n_p - len(used)
        T_ = on.shape[1]                                             # double presses: unmatched strikes on a held key
        dbl[t] = dbl.get(t, 0) + sum(1 for k, qs in pred.items() for q in qs if (k, q) not in used and 0 <= q < T_
                                     and frl[k, q] and not on[k, max(0, q - a.tol):q + a.tol + 1].any())
print(f"{len(files)} clips, {int(tot['any'].sum())} written onsets; match = same key within +-{a.tol} frames ({a.tol * 16} ms)")
TITLE = {"any": "ALL notes, by gap to the previous onset (any key)",
         "same": "ALL notes, by gap to the previous strike of the SAME key",
         "sung": "SUNG notes (multi renders), by gap to the previous onset",
         "other": "instrument-played notes (multi renders), by gap to the previous onset",
         "piano": "pure piano renders, by gap to the previous onset",
         "octave": "notes with the SAME note sounding 1-3 octaves below (bass masking)",
         "no_octave": "notes without a lower octave sounding"}
for kind in KINDS:
    print(f"\nrecall: {TITLE[kind]}:")
    print("   decoder              " + "  ".join(f"{nm:>7s}" for nm in names) + "    all")
    for t in THR:
        print(f"   {t:20s} " + "  ".join(f"{hit[t][kind][i] / max(1, tot[kind][i]):7.3f}" for i in range(NB))
              + f"  {hit[t][kind].sum() / max(1, tot[kind].sum()):6.3f}")
    print("   (count)              " + "  ".join(f"{int(tot[kind][i]):7d}" for i in range(NB)))
print("\noverall:   decoder              recall  precision  F1     double presses / 1000 notes")
for t in THR:
    r_ = hit[t]["any"].sum() / max(1, tot["any"].sum()); p_ = 1 - fp[t] / max(1, npred[t])
    print(f"           {t:20s} {r_:.3f}   {p_:.3f}     {2 * r_ * p_ / max(1e-9, r_ + p_):.3f}  {1000 * dbl[t] / max(1, tot['any'].sum()):6.1f}")
if a.fine and grid["notes"]:
    print(f"\ngrid ceiling ({grid['notes']} notes with exact onsets): a re-strike of the same key less than 2 grid steps"
          f" after the last one cannot come out as its own note.")
    print(f"   16 ms grid: {grid['16']} notes ({100 * grid['16'] / grid['notes']:.2f}%)   8 ms grid: {grid['8']} notes"
          f" ({100 * grid['8'] / grid['notes']:.2f}%)   -> the most 8 ms could ever add: {100 * (grid['16'] - grid['8']) / grid['notes']:.2f}% of notes")
