"""Where does note timing go wrong?  For every correctly found note on held-out renders (exact
onsets from note_meta), how far off its START and END are -- signed (early < 0 < late) -- by kind of
sound: pure piano / plucked-or-struck instrument / sustained instrument by attack speed / sung, at the
app's decoding (onset 0.9, frame 0.5, short notes + re-strikes kept, sliver merge 60 ms).  Also: the
start taken at the first frame over the bar (the app) vs the PEAK frame of the onset run vs a
sub-frame CENTROID of the probabilities around the peak.

    python eval/timing_check.py --ckpt <best.pt> --clips <dir> [--n 600]
"""
import argparse, pathlib, random, sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.ml.spectral import frontend as S                     # noqa: E402
from app.ml.transcriber.decode import decode_notes            # noqa: E402
from app.services import spectral_transcription as st         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--clips", required=True)
ap.add_argument("--n", type=int, default=600); ap.add_argument("--thr", type=float, default=0.9)
ap.add_argument("--offset-thr", type=float, default=0.5)
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
files = random.Random(1).sample(files, min(a.n, len(files)))
FPS = S.FPS
ERR = {}                                    # group -> {"first": [], "peak": [], "centroid": [], "end": []}


def add(g, k, v):
    ERR.setdefault(g, {"first": [], "peak": [], "centroid": [], "end": []})[k].append(v)


def group_of(row, is_multi):
    if not is_multi:
        return "piano (pure piano renders)"
    if len(row) >= 9 and row[8] > 0.5:
        return "sung"
    if len(row) >= 11 and row[10] < 0.5:
        return "multi, sound unknown"
    att = row[2]
    return ("multi: sharp attack (<=5 ms)" if att <= 0.005 else "multi: quick attack (5-50 ms)" if att <= 0.05
            else "multi: slow attack (50-150 ms)" if att <= 0.15 else "multi: very slow attack (>150 ms)")


n_gt = n_found = 0
for f in files:
    z = np.load(f)
    if JOINT:
        JL.set_clip(z)
    meta = z["note_meta"] if "note_meta" in z.files else np.zeros((0, 9))
    if not len(meta):
        continue
    y = z["y"].astype(np.float32) / 32767; fr_lab = z["fr"] > 0
    with torch.no_grad():
        o = net(fe(torch.from_numpy(y)[None].to(dev)), None)
        po, pf, pv = (torch.sigmoid(t.float())[0].T.cpu().numpy() for t in o[:3])
        pof = torch.sigmoid(o[-1].float())[0].T.cpu().numpy() if getattr(net, "offset_head", None) is not None else None
    T = min(po.shape[1], fr_lab.shape[1])
    notes = decode_notes(po[:, :T], pf[:, :T], pv[:, :T], a.thr, 0.5, min_frames=1, min_note_s=0.0,
                         keep_restruck=True, merge_fragment_s=0.06,
                         offset=None if pof is None else pof[:, :T], offset_thr=a.offset_thr)
    pred = {}
    for nt in notes:
        k = int(nt["pitch"]) - 21; f0 = int(round(nt["onset"] * FPS))
        run_end = f0
        while run_end + 1 < T and po[k, run_end + 1] >= a.thr:
            run_end += 1
        pk = f0 + int(np.argmax(po[k, f0:run_end + 1]))
        lo, hi = max(0, pk - 1), min(T, pk + 2)
        w = po[k, lo:hi]; cen = float((np.arange(lo, hi) * w).sum() / max(w.sum(), 1e-9))
        pred.setdefault(k, []).append(dict(first=f0 / FPS, peak=pk / FPS, centroid=cen / FPS, end=nt["offset"]))
    is_multi = str(z["piano"]) == "multi"
    used = set()
    for row in meta:
        on_s, p = float(row[0]), int(row[1]); k = p - 21
        if not (0 <= k < 88) or not (0 <= on_s * FPS < T - 2):
            continue
        n_gt += 1
        c = [(abs(q["first"] - on_s), i) for i, q in enumerate(pred.get(k, [])) if (k, i) not in used and abs(q["first"] - on_s) <= 0.05]
        if not c:
            continue
        _, i = min(c); used.add((k, i)); q = pred[k][i]; n_found += 1
        g = group_of(row, is_multi)
        dg = None
        if is_multi and len(row) >= 16:                                  # how late this note's SOUND came in
            dl = row[15]
            dg = ("sound late 0-30 ms" if dl < 30 else "sound late 30-60 ms" if dl < 60 else "sound late 60-90 ms"
                  if dl < 90 else "sound late 90-120 ms") if dl >= 0 else "sound early"
        for kk in ("first", "peak", "centroid"):
            add(g, kk, 1000 * (q[kk] - on_s)); add("ALL", kk, 1000 * (q[kk] - on_s))
            if dg:
                add(dg, kk, 1000 * (q[kk] - on_s))
        t0 = int(round(on_s * FPS)); t1 = t0                         # true end: the labelled sustain run
        while t1 + 1 < fr_lab.shape[1] and fr_lab[k, t1 + 1]:
            t1 += 1
        if t1 + 1 < fr_lab.shape[1]:                                 # not cut by the clip edge
            add(g, "end", 1000 * (q["end"] - (t1 + 1) / FPS)); add("ALL", "end", 1000 * (q["end"] - (t1 + 1) / FPS))

print(f"{len(files)} clips; {n_gt} notes with exact onsets, {n_found} found ({n_found / max(1, n_gt):.1%}) and timed")
print("signed error in ms (minus = early, plus = late); |err| = absolute")


def line(v):
    v = np.asarray(v)
    if not len(v):
        return "-"
    av = np.abs(v)
    return (f"median {np.median(v):+6.1f}  |err| median {np.median(av):5.1f}  within 8 ms {np.mean(av <= 8):5.1%}  "
            f"16 ms {np.mean(av <= 16):5.1%}  32 ms {np.mean(av <= 32):5.1%}  early>16 {np.mean(v < -16):5.1%}  late>16 {np.mean(v > 16):5.1%}")


order = ["ALL", "piano (pure piano renders)", "multi: sharp attack (<=5 ms)", "multi: quick attack (5-50 ms)",
         "multi: slow attack (50-150 ms)", "multi: very slow attack (>150 ms)", "sung", "multi, sound unknown",
         "sound early", "sound late 0-30 ms", "sound late 30-60 ms", "sound late 60-90 ms", "sound late 90-120 ms"]
for g in order:
    if g not in ERR:
        continue
    e = ERR[g]
    print(f"\n{g}  ({len(e['first'])} notes)")
    for kk, nm in (("first", "start, first frame over bar (app)"), ("peak", "start, peak frame"), ("centroid", "start, sub-frame centroid")):
        print(f"   {nm:36s} {line(e[kk])}")
    v = np.asarray(e["end"])
    if len(v):
        av = np.abs(v)
        print(f"   {'END (note release)':36s} median {np.median(v):+6.1f}  |err| median {np.median(av):5.1f}  "
              f"within 32 ms {np.mean(av <= 32):5.1%}  100 ms {np.mean(av <= 100):5.1%}  early>100 {np.mean(v < -100):5.1%}  late>100 {np.mean(v > 100):5.1%}")
