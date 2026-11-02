"""Two questions from listening (the user, 2026-11-01), measured on the 840 real Music2MIDI songs of eval/human_data.py
with the hosted chain (base at 0.8 + clean-up v4 without beats: keep 0.3, merge 0.5, never-kept 0.7):

1. POSITION: do we drop more notes near the start of a song?  Share of the arranger's notes removed (dropped by
   keep, merged, dropped as never kept) by time from the song's start and end, our density vs the human
   arrangements' density in the same stretch, and an EDGE test: the clean-up's 20 s windows (every 10 s,
   centre-weighted) shifted back 10 s, so a note in the first 10 s sits mid-window instead of at a window edge.
2. REPEATS: when a part of the song repeats, do we write it the same way both times?  Repeats come from the audio
   (chroma self-similarity, lag >= 15 s, 10 s pieces, the lag refined to the ms on the onset envelope).  Per pair:
   note F1 between the two occurrences (same pitch, onset within 50 / 100 ms) for the raw arranger, after keep /
   merge, and final; the same for each human arrangement; and where each final mismatch comes from (the arranger
   never wrote the counterpart, or the clean-up removed it).

    --probs  (GPU) clean-up probabilities per note, normal and shifted windows -> OUT/probs/<song>.npz
    --score  (CPU) per song: position bins + repeat pairs -> OUT/rows_<shard>.json
    --report
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
HUMAN = pathlib.Path("/scratch/basu.ang/human_data")
SONGS = pathlib.Path("/scratch/basu.ang/never_kept/songs.json")
NKDIR = pathlib.Path("/scratch/basu.ang/never_kept")
AUDIO = pathlib.Path("/scratch/basu.ang/solo_det/m2m")
OUT = pathlib.Path("/scratch/basu.ang/position_repeat")
CKPT = pathlib.Path.home() / "arrangement_transformer/checkpoints/cleanup4_nobeat/best.pt"
SR, KEEP, MERGE, NKBAR = 24000, 0.3, 0.5, 0.7
START_BINS = [0, 5, 10, 20, 30, 60, 120, 1e9]
END_BINS = [0, 5, 10, 20, 30, 1e9]
PIECE, MIN_LAG, SIM_THR = 10.0, 15.0, 0.9


def probs(a):
    import torch
    from app.ml.spectral import cleanup4_train as C4
    s = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(s["cfg"]); cfg.pop("version", None)
    net = C4.Cleanup4(**cfg); net.load_state_dict(s["model"]); net = net.cuda().eval()
    songs, _ = C4.load_human(str(HUMAN), str(SONGS))
    songs = songs[:a.limit] if a.limit else songs
    (OUT / "probs").mkdir(parents=True, exist_ok=True)
    for s in songs:
        f = OUT / "probs" / f"{s['name']}.npz"
        if f.exists():
            continue
        end = float(s["notes"][:, 0].max())
        p1 = C4.song_probs(net, s, "cuda")
        s2 = dict(s, win=np.arange(-10.0, max(end - 10, 0) + 1e-6, 10.0))
        p2 = C4.song_probs(net, s2, "cuda")
        np.savez(f, p=p1.astype(np.float32), p_shift=p2.astype(np.float32))
        print(s["name"], len(s["notes"]), flush=True)
    print("done", flush=True)


def stages(n, p):
    """per raw note: 0 final, 1 dropped by keep, 2 merged into the strike before, 3 dropped as never kept."""
    pk, pm, pn = p[:, 0], p[:, 1], p[:, 2]
    st = np.full(len(n), 1, np.int8)
    for q in np.unique(n[:, 2]):
        idx = np.nonzero(n[:, 2] == q)[0]
        idx = idx[np.argsort(n[idx, 0], kind="stable")]
        alive = False
        for i in idx:
            if alive and pm[i] >= MERGE:
                st[i] = 2
            elif pk[i] >= KEEP:
                st[i] = 3 if pn[i] >= NKBAR else 0
                alive = True
            else:
                alive = False
    return st


def f1(a, b, tol):
    """greedy note matching (same pitch, onsets within tol): (matched, len a, len b) and the matched mask of a."""
    hit_a = np.zeros(len(a), bool)
    m = 0
    for q in np.unique(np.concatenate([a[:, 1], b[:, 1]])) if len(a) and len(b) else []:
        ia = np.nonzero(a[:, 1] == q)[0]; ib = np.nonzero(b[:, 1] == q)[0]
        ta, tb = a[ia, 0], b[ib, 0]
        used = np.zeros(len(ib), bool)
        for k in np.argsort(ta):
            d = np.abs(tb - ta[k]); d[used] = 9
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= tol:
                used[j] = True; hit_a[ia[k]] = True; m += 1
    return m, len(a), len(b), hit_a


def repeats(y):
    """10 s piece pairs (t1, t2) whose chroma repeats; t2 - t1 refined on the onset envelope; with audio-sameness."""
    import librosa
    hop = 2048
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=hop, tuning=0.0)   # tuning estimate crashes (numba) on some CPU nodes
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    fps = SR / hop
    T = C.shape[1]
    P, Lmin, k = int(PIECE * fps), int(MIN_LAG * fps), int(4 * fps)
    cand = []
    for L in range(Lmin, T - P):
        s = np.sum(C[:, :T - L] * C[:, L:], 0)
        sm = np.convolve(s, np.ones(k) / k, mode="same")
        good = np.concatenate([[False], sm >= SIM_THR, [False]])
        edges = np.nonzero(np.diff(good.astype(np.int8)))[0]
        for r0, r1 in zip(edges[::2], edges[1::2]):           # runs of frames above the bar
            for t in range(r0, r1 - P + 1, P):
                cand.append((float(s[t:t + P].mean()), t, L))
    cand.sort(reverse=True)
    taken = np.zeros(T + 1, bool)
    pairs = []
    for sim, t, L in cand:
        if taken[t:t + P].any() or taken[t + L:t + L + P].any():
            continue
        taken[t:t + P] = True; taken[t + L:t + L + P] = True
        pairs.append((t / fps, (t + L) / fps, sim))
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=240)          # 100 fps
    rms = librosa.feature.rms(y=y, hop_length=240)[0]
    out = []
    for t1, t2, sim in pairs:
        a0, b0 = int(t1 * 100), int(t2 * 100)
        a = env[a0:a0 + int(PIECE * 100)]
        best, bl = -2.0, 0
        for d in range(-30, 31):
            b = env[b0 + d:b0 + d + len(a)]
            if len(b) < len(a) or b0 + d < 0:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best:
                best, bl = r, d
        ra = rms[a0:a0 + len(a)].mean(); rb = rms[b0 + bl:b0 + bl + len(a)].mean()
        out.append(dict(t1=t1, t2=t2 + bl / 100, chroma=sim, env_r=best, db=float(20 * np.log10((rb + 1e-9) / (ra + 1e-9)))))
    return out


def human_notes(sid, info, raw):
    import human_merge as HM
    s = info.get(sid)
    if not s or not s["used"]:
        return []
    H = []
    for aid in s["used"]:
        h = np.load(NKDIR / "midi" / f"{aid}.npy").astype(float)
        lag = s["lags"][aid][0]
        c = [h.copy() for _ in range(3)]
        c[1][:, :2] += lag; c[2][:, :2] -= lag
        H.append(max(c, key=lambda x: HM.match_rate(raw, x)))
    return H


def score(a):
    from app.ml.spectral import cleanup4_train as C4
    info = {}
    for p in NKDIR.glob("info_*.json"):
        for s in json.loads(p.read_text())["songs"]:
            info[s["song"]] = s
    files = sorted((OUT / "probs").glob("*.npz"))[a.shard::a.nshards]
    files = files[:a.limit] if a.limit else files
    rows = []
    for f in files:
        sid = f.stem
        z = np.load(HUMAN / f"{sid}.npz"); pz = np.load(f)
        n = z["notes"].astype(float); kept = z["kept"]
        y = np.load(AUDIO / f"{sid}.npy").astype(np.float32)
        dur = len(y) / SR
        st, st2 = stages(n, pz["p"]), stages(n, pz["p_shift"])
        H = human_notes(sid, info, n)
        on = n[:, 0]
        pos = []
        for name, bins, t in (("start", START_BINS, on), ("end", END_BINS, dur - on)):
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (t >= lo) & (t < hi)
                span = max(0.0, min(hi, dur) - lo)
                if span <= 0:
                    continue
                hum = [float(((hh[:, 0] if name == "start" else dur - hh[:, 0]) >= lo).sum()
                             - ((hh[:, 0] if name == "start" else dur - hh[:, 0]) >= hi).sum()) for hh in H]
                pos.append(dict(side=name, lo=lo, hi=hi, span=span, raw=int(m.sum()),
                                st=[int((st[m] == k).sum()) for k in range(4)],
                                st_shift=[int((st2[m] == k).sum()) for k in range(4)],
                                never=int((m & (kept == 0)).sum()), human=float(np.mean(hum)) if hum else -1.0))
        reps = []
        try:
            pairs = repeats(y)
        except Exception as e:                                   # noqa: BLE001
            print(f"{sid}: repeats failed {e}", flush=True); pairs = []
        for r in pairs:
            L = r["t2"] - r["t1"]
            sel = lambda t0, mask: n[mask & (on >= t0) & (on < t0 + PIECE)][:, [0, 2]] - [t0, 0]
            res = dict(r)
            for stage, mask in (("raw", np.ones(len(n), bool)), ("keepmerge", (st == 0) | (st == 3)), ("final", st == 0)):
                A, B = sel(r["t1"], mask), sel(r["t2"], mask)
                for tol in (0.05, 0.1):
                    m_, na, nb, _ = f1(A, B, tol)
                    res[f"{stage}_{int(tol * 1000)}"] = [m_, na, nb]
            # where final mismatches come from: counterpart absent in the other occurrence's raw notes, or removed
            src = [0, 0, 0, 0]       # arranger never wrote it, dropped, merged, never-kept drop
            for x0, y0 in ((r["t1"], r["t2"]), (r["t2"], r["t1"])):
                Af = sel(x0, st == 0)
                yi = np.nonzero((on >= y0) & (on < y0 + PIECE))[0]
                Br = n[yi][:, [0, 2]] - [y0, 0]
                _, _, _, hit_f = f1(Af, sel(y0, st == 0), 0.05)
                for k in np.nonzero(~hit_f)[0]:
                    d = np.abs(Br[:, 0] - Af[k, 0]); d[Br[:, 1] != Af[k, 1]] = 9
                    j = int(np.argmin(d)) if len(d) else -1
                    src[0 if j < 0 or d[j] > 0.05 else int(st[yi[j]])] += 1
            res["src"] = src
            hum = []
            for hh in H:
                hs = lambda t0: hh[(hh[:, 0] >= t0) & (hh[:, 0] < t0 + PIECE)][:, [0, 2]] - [t0, 0]
                hum.append(list(f1(hs(r["t1"]), hs(r["t2"]), 0.1)[:3]))
            res["human_100"] = hum
            reps.append(res)
        rows.append(dict(song=sid, dur=dur, pos=pos, reps=reps))
        print(f"{sid}: {len(n)} notes, {len(reps)} repeat pairs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rows_{a.shard}.json").write_text(json.dumps(rows))


def report(a):
    rows = sum((json.loads(f.read_text()) for f in sorted(OUT.glob("rows_*.json"))), [])
    print(f"songs {len(rows)}")
    print("\n1. POSITION: share of the arranger's notes removed, and density vs humans")
    print(f"{'stretch':>16} {'raw/s':>6} {'final/s':>8} {'human/s':>8} {'removed':>8} {'dropped':>8} {'merged':>7} "
          f"{'nk':>5} {'never kept':>10} | {'shifted windows: removed':>24}")
    for side, bins in (("start", START_BINS), ("end", END_BINS)):
        for lo, hi in zip(bins[:-1], bins[1:]):
            P = [p for r in rows for p in r["pos"] if p["side"] == side and p["lo"] == lo]
            if not P:
                continue
            span = sum(p["span"] for p in P); raw = sum(p["raw"] for p in P)
            st = np.sum([p["st"] for p in P], 0); st2 = np.sum([p["st_shift"] for p in P], 0)
            hp = [p for p in P if p["human"] >= 0]
            hum = sum(p["human"] for p in hp) / max(sum(p["span"] for p in hp), 1e-9)
            lab = f"{side} {lo:g}-{hi:g}s" if hi < 1e8 else f"{side} {lo:g}s+"
            print(f"{lab:>16} {raw / span:6.1f} {st[0] / span:8.1f} {hum:8.1f} {1 - st[0] / max(raw, 1):8.1%} "
                  f"{st[1] / max(raw, 1):8.1%} {st[2] / max(raw, 1):7.1%} {st[3] / max(raw, 1):5.1%} "
                  f"{sum(p['never'] for p in P) / max(raw, 1):10.1%} | {1 - st2[0] / max(raw, 1):24.1%}")
    R = [x for r in rows for x in r["reps"]]
    print(f"\n2. REPEATS: {len(R)} pairs of 10 s pieces in {sum(1 for r in rows if r['reps'])} songs")

    def agg(sel, key):
        m = np.sum([x[key] for x in sel], 0) if sel else np.zeros(3)
        return 2 * m[0] / max(m[1] + m[2], 1)

    def hum(sel):
        H = [h for x in sel for h in x["human_100"]]
        if not H:
            return float("nan")
        m = np.sum(H, 0)
        return 2 * m[0] / max(m[1] + m[2], 1)
    groups = [("all", R)]
    for lo, hi in ((0.9, 0.95), (0.95, 0.98), (0.98, 1.01)):
        groups.append((f"chroma {lo}-{min(hi, 1):.2f}", [x for x in R if lo <= x["chroma"] < hi]))
    for lo, hi in ((-1, 0.5), (0.5, 0.8), (0.8, 1.01)):
        groups.append((f"onset-env r {lo}-{min(hi, 1):.1f}", [x for x in R if lo <= x["env_r"] < hi]))
    for lo, hi in ((0, 1), (1, 3), (3, 99)):
        groups.append((f"|loudness diff| {lo}-{hi} dB", [x for x in R if lo <= abs(x["db"]) < hi]))
    print(f"{'pairs':>34} {'n':>6} | same-notes F1 50 ms: raw / keep+merge / final | 100 ms final | humans 100 ms")
    for name, sel in groups:
        if not sel:
            continue
        print(f"{name:>34} {len(sel):6d} | {agg(sel, 'raw_50'):.3f} / {agg(sel, 'keepmerge_50'):.3f} / "
              f"{agg(sel, 'final_50'):.3f}            | {agg(sel, 'final_100'):.3f}        | {hum(sel):.3f}")
    src = np.sum([x["src"] for x in R], 0)
    tot = max(src.sum(), 1)
    print(f"\nfinal notes with no counterpart in the other occurrence ({src.sum():,}): the arranger never wrote it "
          f"{src[0] / tot:.1%}; the clean-up dropped it {src[1] / tot:.1%}, merged it {src[2] / tot:.1%}, "
          f"never-kept drop {src[3] / tot:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", action="store_true"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    probs(a) if a.probs else score(a) if a.score else report(a)


if __name__ == "__main__":
    main()
