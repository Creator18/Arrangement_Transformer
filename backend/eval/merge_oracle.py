"""Ceiling of a clean-up that can MERGE same-key strikes (the user's plan, 2026-10-27): which merges bring the base's
0.8 output closest to the clean written arrangement?

Per clip and per key (keys never interact), an exact search over that key's strikes in time order: split them into
runs of neighbouring strikes; each run is either dropped or played as ONE note that starts at the run's first base
start and ends at its last base end (only the gaps inside the run are filled: never a new start, never past the base's
own sound).  A run of one strike is plain keep / drop.  The plan picked is the one closest to the key's written notes:
  +1 per kept note whose start matches a written start (50 ms, one-to-one), +1 more if its end matches too
  (mir_eval's rule: within max(50 ms, 20% of the written length)), -1 per kept note matching nothing.
The searched plans are the training labels for a merge output later; their score is the ceiling.

Compared on the held-out clips (checkpoints/cleanup1/split.json), mir_eval note F1 without and with note ends:
  base 0.8 as is | v2 keep 0.3 (the user's pick) | best keep / drop (the clean-up's own labels) | best keep / drop / merge
plus how often the base splits one written note, and how many written same-key repeats sit closer than 150 ms.

    python eval/merge_oracle.py --data /scratch/basu.ang/cleanup_data --beats /scratch/basu.ang/clip_beats \
        --ckpt checkpoints/cleanup2/best.pt --split checkpoints/cleanup1/split.json --out part.json --task k --ntasks n
    python eval/merge_oracle.py --merge part*.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOL, OFF_RATIO, OFF_MIN = 0.05, 0.2, 0.05
MAX_GAP = 2.0                      # s: runs never bridge a silence longer than this (search size only)
CONDS = ("base 0.8", "v2 keep 0.3", "best keep / drop", "best keep / drop / merge")


def off_ok(o_pred, w):
    return abs(o_pred - w[1]) <= max(OFF_MIN, OFF_RATIO * (w[1] - w[0]))


def best_plan(S, W):
    """S: (n, 2) this key's base strikes (on, off) in time order; W: (m, 2) its written notes.
    returns runs [(a, b, keep)] covering 0..n-1, maximising the score above."""
    n, m = len(S), len(W)
    NEG = -1e9
    # f[i][k]: best score for strikes[:i] with written notes[:k] available-or-used (k = first unusable index)
    f = np.full((n + 1, m + 1), NEG)
    back = {}
    f[0, 0] = 0
    for i in range(n):
        for k in range(m + 1):
            if f[i, k] <= NEG / 2:
                continue
            # carry: written notes may be skipped (left unmatched)
            if k < m and f[i, k] > f[i, k + 1]:
                f[i, k + 1] = f[i, k]
                back[(i, k + 1)] = ("skip", i, k)
            end = S[i, 1]
            for b in range(i + 1, n + 1):
                if b > i + 1:
                    if S[b - 1, 0] - end > MAX_GAP:
                        break
                    end = max(end, S[b - 1, 1])
                on = S[i, 0]
                # drop the run [i, b)
                if f[i, k] > f[b, k]:
                    f[b, k] = f[i, k]
                    back[(b, k)] = ("run", i, k, 0, -1)
                # keep it as one note, matched to written note j >= k, or to nothing
                cand = [(f[i, k] - 1, -1)]
                for j in range(k, m):
                    if W[j, 0] > on + TOL:
                        break
                    if abs(W[j, 0] - on) <= TOL:
                        cand.append((f[i, k] + 1 + off_ok(end, W[j]), j))
                for sc, j in cand:
                    kk = k if j < 0 else j + 1
                    if sc > f[b, kk]:
                        f[b, kk] = sc
                        back[(b, kk)] = ("run", i, k, 1, j)
    k = int(np.argmax(f[n]))
    runs, i = [], n
    while i > 0 or k > 0:
        st = back.get((i, k))
        if st is None:
            break
        if st[0] == "skip":
            i, k = st[1], st[2]
            continue
        _, a, ka, keep, _ = st
        runs.append((a, i, keep))
        i, k = a, ka
    return runs[::-1]


def apply_plan(notes, plan_by_key):
    out = []
    for p, (idx, runs) in plan_by_key.items():
        for a, b, keep in runs:
            if keep:
                sel = idx[a:b]
                out.append((notes[sel[0], 0], notes[sel, 1].max(), p))
    return np.array(out, float).reshape(-1, 3)


def f1s(pred, gt):
    import mir_eval
    if not len(pred) or not len(gt):
        return dict(n_pred=len(pred), n_gt=len(gt), on=(0, 0), onoff=(0, 0))
    hz = lambda p: 440.0 * 2 ** ((p - 69) / 12)
    ri, rp = gt[:, :2], hz(gt[:, 2])
    ei, ep = pred[:, :2].copy(), hz(pred[:, 2])
    ei[:, 1] = np.maximum(ei[:, 1], ei[:, 0] + 1e-3)
    ri = ri.copy(); ri[:, 1] = np.maximum(ri[:, 1], ri[:, 0] + 1e-3)
    m_on = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=TOL, offset_ratio=None))
    m_of = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=TOL, offset_ratio=OFF_RATIO,
                                                  offset_min_tolerance=OFF_MIN))
    return dict(n_pred=len(pred), n_gt=len(gt), on=(m_on, m_on), onoff=(m_of, m_of))


def run(a):
    import torch
    from app.ml.spectral import cleanup2_train as C2
    C2.BEATS_DIR[0] = a.beats
    s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(s["cfg"]); cfg.pop("version", None)
    net = C2.Cleanup2(**cfg); net.load_state_dict(s["model"]); net.eval()
    torch.set_num_threads(2)
    val = set(json.loads(pathlib.Path(a.split).read_text()))
    files = sorted(p for p in pathlib.Path(a.data).glob("*.npz") if ".tmp" not in p.name and p.name.split("_")[0] in val)
    files = files[a.task::a.ntasks]
    R = {c: dict(pred=0, gt=0, on=0, onoff=0) for c in CONDS}
    st = dict(written=0, split2=0, split3=0, fast_rep=0, merges=0, merged_pieces=0, dropped=0, clips=0)
    for f in files:
        z = np.load(f)
        n, gt = z["notes"].astype(float), z["gt"].astype(float)
        if not len(n) or not len(gt):
            continue
        st["clips"] += 1
        c = C2.featurize(str(f))
        N = min(C2.MAXN, len(n))
        with torch.no_grad():
            P = torch.from_numpy(c["pitch"][:N].astype(np.int64))[None]
            T = torch.from_numpy(c["t"][:N])[None]
            X = torch.from_numpy(c["x"][:N].astype(np.float32))[None]
            lk, _ = net(P, T, X, torch.zeros(1, N, dtype=torch.bool))
        keep = np.ones(len(n), bool)
        keep[:N] = torch.sigmoid(lk)[0].numpy() >= 0.3
        plans = {}
        for p in np.unique(n[:, 2]).astype(int):
            idx = np.nonzero(n[:, 2] == p)[0]
            idx = idx[np.argsort(n[idx, 0], kind="stable")]
            W = gt[gt[:, 2] == p][:, :2]
            W = W[np.argsort(W[:, 0])]
            plans[p] = (idx, best_plan(n[idx, :2], W))
            for a_, b_, kp in plans[p][1]:
                if kp and b_ - a_ > 1:
                    st["merges"] += 1; st["merged_pieces"] += b_ - a_
                if not kp:
                    st["dropped"] += b_ - a_
            for w in W:                                    # how many base strikes start inside each written note
                k_in = int(((n[idx, 0] >= w[0] - TOL) & (n[idx, 0] < w[1])).sum())
                st["split2"] += k_in >= 2; st["split3"] += k_in >= 3
            st["fast_rep"] += int((np.diff(W[:, 0]) < 0.15).sum()) if len(W) > 1 else 0
        st["written"] += len(gt)
        outs = {"base 0.8": n[:, [0, 1, 2]], "v2 keep 0.3": n[keep][:, [0, 1, 2]],
                "best keep / drop": n[n[:, 5] > 0][:, [0, 1, 2]], "best keep / drop / merge": apply_plan(n, plans)}
        for cn, pred in outs.items():
            r = f1s(pred, gt[:, :3])
            R[cn]["pred"] += r["n_pred"]; R[cn]["gt"] += r["n_gt"]; R[cn]["on"] += r["on"][0]; R[cn]["onoff"] += r["onoff"][0]
    return dict(R=R, stats=st)


def labels(a):
    """the searched plan per note, for training the merge output: <out>/<clip>.npy int8 in the clip's note order,
    0 = drop, 1 = keep (a run's first strike), 2 = merge into the previous strike of the same key."""
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in a.data.split(","):
        files += sorted(p for p in pathlib.Path(d).glob("*.npz") if ".tmp" not in p.name)
    files = files[a.task::a.ntasks]
    for i, f in enumerate(files):
        dst = out / (f.stem + ".npy")
        if dst.exists():
            continue
        z = np.load(f)
        n, gt = z["notes"].astype(float), z["gt"].astype(float).reshape(-1, 3)
        act = np.zeros(len(n), np.int8)
        for p in np.unique(n[:, 2]).astype(int) if len(n) else []:
            idx = np.nonzero(n[:, 2] == p)[0]
            idx = idx[np.argsort(n[idx, 0], kind="stable")]
            W = gt[gt[:, 2] == p][:, :2]
            W = W[np.argsort(W[:, 0])]
            for a_, b_, kp in best_plan(n[idx, :2], W):
                if kp:
                    act[idx[a_]] = 1
                    act[idx[a_ + 1:b_]] = 2
        np.save(out / (f.stem + ".tmp.npy"), act)
        (out / (f.stem + ".tmp.npy")).rename(dst)
        if i % 500 == 0:
            print(f"{i + 1}/{len(files)} {f.name}: keep {int((act == 1).sum())} merge {int((act == 2).sum())} "
                  f"drop {int((act == 0).sum())}", flush=True)
    print("done")


def report(D):
    R, st = D["R"], D["stats"]

    def f(h, r):
        rc, pr = h / max(1, r["gt"]), h / max(1, r["pred"])
        return f"{rc:.3f} / {pr:.3f} / {2 * rc * pr / max(1e-9, rc + pr):.3f}"
    print(f"\n=== {st['clips']} held-out clips, {st['written']} written notes")
    print(f"{'':<27}{'notes':>8}{'starts R / P / F1':>24}{'starts + ends R / P / F1':>28}")
    for c in CONDS:
        r = R[c]
        print(f"{c:<27}{r['pred']:>8}{f(r['on'], r):>24}{f(r['onoff'], r):>28}")
    w = max(1, st["written"])
    print(f"\nwritten notes the base strikes 2+ times: {st['split2'] / w:.1%} (3+: {st['split3'] / w:.1%});  "
          f"written same-key repeats < 150 ms apart: {st['fast_rep'] / w:.2%}")
    print(f"best plan: {st['merges']} merges of {st['merged_pieces']} strikes "
          f"({st['merged_pieces'] / max(1, st['merges']):.2f} per merge), {st['dropped']} strikes dropped")


def main():
    ap = argparse.ArgumentParser()
    for k in ("data", "beats", "ckpt", "split", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--labels", action="store_true", help="write the searched plan per clip (--data may list dirs)")
    a = ap.parse_args()
    if a.labels:
        labels(a)
        return
    if a.merge:
        ds = [json.loads(pathlib.Path(p).read_text()) for p in a.merge]
        D = dict(R={c: {k: sum(d["R"][c][k] for d in ds) for k in ds[0]["R"][c]} for c in CONDS},
                 stats={k: sum(d["stats"][k] for d in ds) for k in ds[0]["stats"]})
        report(D)
        return
    D = run(a)
    pathlib.Path(a.out).write_text(json.dumps(D))
    report(D)


if __name__ == "__main__":
    main()
