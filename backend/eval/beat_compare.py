"""madmom vs OUR beat model on the same held-out clips (2026-10-28).  The clips are the beat model's validation
subset (beat_train.py: 600 clips of the 90 held-out POP909 songs, the same seed), with the same metrics: mir_eval beat
F-measure (+-70 ms), downbeat F, decoded tempo within 4%.

    python eval/beat_compare.py --data /scratch/basu.ang/beat_data --clips /scratch/basu.ang/base_render/validation \
        --out /scratch/basu.ang/beat_compare/madmom_0.json --shard 0 --nshards 10            # madmom
    python eval/beat_compare.py ... --ckpt checkpoints/beat1/best.pt --out .../ours.json      # ours (GPU or CPU)
"""
import argparse
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.spectral import beat_train as BT  # noqa: E402


def score(ref, refd, est, estd):
    import mir_eval
    fb = mir_eval.beat.f_measure(mir_eval.beat.trim_beats(ref, 0), mir_eval.beat.trim_beats(est, 0)) if len(est) > 1 else 0.0
    fd = mir_eval.beat.f_measure(refd, estd) if len(refd) > 1 and len(estd) > 0 else (np.nan if len(refd) < 2 else 0.0)
    bpm = 60 / np.median(np.diff(ref))
    eb = 60 / np.median(np.diff(est)) if len(est) > 2 else 0
    r = eb / bpm
    return dict(beat_f=float(fb), downbeat_f=float(fd), tempo_right=bool(abs(r - 1) < .04),
                tempo_double=bool(abs(r / 2 - 1) < .04), tempo_half=bool(abs(r * 2 - 1) < .04))


def summary(rows):
    k = rows[0].keys()
    return {x: round(float(np.nanmean([r[x] for r in rows])), 4) for x in k if x != "clip"} | dict(clips=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clips", default="/scratch/basu.ang/base_render/validation")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", nargs="*")
    a = ap.parse_args()
    if a.merge:
        rows = sum((json.loads(pathlib.Path(p).read_text())["rows"] for p in a.merge), [])
        print(json.dumps(summary(rows), indent=1))
        pathlib.Path(a.out).write_text(json.dumps(dict(summary=summary(rows), rows=rows)))
        return
    vm, vr = BT.load(a.data, "validation")
    idx = random.Random(1).sample(range(len(vr)), min(a.n, len(vr)))[a.shard::a.nshards]
    rows = []
    if a.ckpt:
        import torch
        from app.ml.spectral import beat_decode as D
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        s = torch.load(a.ckpt, map_location=dev, weights_only=False)
        net = BT.BeatNet(**s["cfg"]).to(dev).eval(); net.load_state_dict(s["model"])
        with torch.no_grad():
            for i in idx:
                x, _, _, _ = BT.make_batch(vm, vr, [i], dev, train=False)
                p = torch.sigmoid(net(x)[0].float())[0].cpu().numpy()
                est = D.beats(p[0]); estd, _ = D.downbeats(est, p[1])
                rows.append(dict(clip=vr[i]["clip"], **score(np.asarray(vr[i]["beats"]), np.asarray(vr[i]["downbeats"]), est, estd)))
    else:
        import librosa
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
        rnn, dbn = RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)
        drnn, ddbn = RNNDownBeatProcessor(), DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        for i in idx:
            y = np.load(pathlib.Path(a.clips) / f"{vr[i]['clip']}.npz")["y"].astype(np.float32) / 32767
            y44 = librosa.resample(y, orig_sr=24000, target_sr=44100)
            est = dbn(rnn(y44))
            db = ddbn(drnn(y44))
            estd = db[db[:, 1] == 1, 0] if len(db) else np.zeros(0)
            rows.append(dict(clip=vr[i]["clip"], **score(np.asarray(vr[i]["beats"]), np.asarray(vr[i]["downbeats"]), est, estd)))
            print(json.dumps(rows[-1]), flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(dict(summary=summary(rows), rows=rows)))
    print(json.dumps(summary(rows), indent=1))


if __name__ == "__main__":
    main()
