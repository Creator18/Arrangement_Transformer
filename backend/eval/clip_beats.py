"""madmom beats for every clip the clean-up model trains on (clean-up v2: beat position per note, learned quantizing).

Per clip, /scratch/basu.ang/clip_beats/<clip>.npz:
    beats     madmom RNN + DBN beat tracker (s)
    dbeats    madmom RNN + DBN downbeat tracker: beat times (s) and their number in the bar (dbnum, 1 = downbeat)
    act, dact beat / downbeat activations at 100 fps (float16)
The app runs the same trackers on the whole song (app/services/cleanup.py).

    python eval/clip_beats.py --clips <dir> [--list names.txt] --out /scratch/basu.ang/clip_beats --task k --ntasks n
"""
import argparse
import pathlib

import numpy as np

SR = 24000


def _compat():
    """madmom 0.16 (pip, the laptop's venv) on Python 3.11 / numpy >= 1.24; a no-op for the git build on the cluster."""
    import collections
    import collections.abc
    for n in ("MutableSequence", "MutableMapping", "Sequence", "Mapping", "Iterable"):
        if not hasattr(collections, n):
            setattr(collections, n, getattr(collections.abc, n))
    for n, t in (("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)):
        if n not in np.__dict__:
            setattr(np, n, t)
    import madmom.features.downbeats as D

    class _NP:                                             # results = [(path, logprob), ...] is ragged for new numpy
        def __getattr__(self, n):
            return getattr(np, n)

        def asarray(self, a, *k, **kw):
            try:
                return np.asarray(a, *k, **kw)
            except ValueError:
                return np.asarray(a, dtype=object)
    if D.np is np:
        D.np = _NP()


def track(y24):
    """the trackers on 24 kHz mono audio: dict of arrays (also used by the app)."""
    _compat()
    import librosa
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    P = track.__dict__
    if "rnn" not in P:
        P.update(rnn=RNNBeatProcessor(), dbn=DBNBeatTrackingProcessor(fps=100), drnn=RNNDownBeatProcessor(),
                 ddbn=DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100))
    y = librosa.resample(y24.astype(np.float32), orig_sr=SR, target_sr=44100)
    act = P["rnn"](y)
    dact = P["drnn"](y)
    beats = P["dbn"](act)
    db = P["ddbn"](dact) if len(dact) else np.zeros((0, 2))
    return dict(beats=np.asarray(beats, np.float32), dbeats=db[:, 0].astype(np.float32) if len(db) else np.zeros(0, np.float32),
                dbnum=db[:, 1].astype(np.int8) if len(db) else np.zeros(0, np.int8), act=act.astype(np.float16),
                dact=dact[:, 1].astype(np.float16) if dact.ndim == 2 else np.zeros(0, np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--list", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in open(a.list) if n.strip()] if a.list else \
        sorted(p.name for p in pathlib.Path(a.clips).glob("*.npz") if ".tmp" not in p.name)
    names = names[a.task::a.ntasks]
    for i, n in enumerate(names):
        dst = out / n
        if dst.exists():
            continue
        z = np.load(pathlib.Path(a.clips) / n)
        r = track(z["y"].astype(np.float32) / 32767)
        tmp = out / (pathlib.Path(n).stem + f".tmp{a.task}.npz")
        np.savez_compressed(tmp, **r)
        tmp.rename(dst)
        if i % 100 == 0:
            print(f"{i + 1}/{len(names)} {n}: {len(r['beats'])} beats", flush=True)
    print("done")


if __name__ == "__main__":
    main()
