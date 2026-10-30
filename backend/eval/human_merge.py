"""Which of our repeated strikes do humans play as ONE press, and how many notes do humans keep where?
(the user's idea, 2026-10-30: learn from the varying human arrangements, as for "never kept")

Measurement only, nothing is trained.  Inputs are what eval/never_kept.py left in /scratch/basu.ang/never_kept: per
song our raw 0.8 notes + clean-up v3's keep / merge probabilities + its output (run/<audio>.npz), the arrangements
(midi/<id>.npy) and which passed the lag gate with their lags (info_*.json).  Each arrangement is moved by its lag
(the sign that matches more of our notes is used, checked per arrangement).

1. Repeats.  A pair = two strikes A, B of ours on the same key, B starting <= GAP after A ends (the merge candidates).
   For every arrangement that has A (same pitch class within 150 ms of A's start):
       repeat  it also strikes that pitch class within 150 ms of B's start (a DIFFERENT note from the one matching A)
       single  it does not
       held    (of single) its A note lasts until B's start - 50 ms (else it let go; pedal may hide that)
   Label of a pair = share of the matching arrangements that repeat.  Control: the same test on each arrangement's
   OWN same-key pairs against the other arrangements (how much humans agree).
2. Density.  Notes / s per 10 s window: ours raw, ours after clean-up, every arrangement.  Do humans agree on where
   to play more or fewer notes (window-to-window shape within a song), and does our shape follow theirs?

    python eval/human_merge.py --score --shard k --nshards n     (courses, CPU)
    python eval/human_merge.py --report
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
NK = pathlib.Path("/scratch/basu.ang/never_kept")
OUT = pathlib.Path("/scratch/basu.ang/human_merge")
VOCAL = pathlib.Path("/scratch/basu.ang/solo_det/m2m_sim")
TOL, GAP, HOLD, WIN = 0.15, 0.15, 0.05, 10.0
SAMEKEY = '--samekey' in sys.argv          # a re-strike must be the SAME key as the human's A note (not any octave)
if SAMEKEY:
    OUT = OUT / 'samekey'


def onsets_by_pc(h):
    """pitch class -> (sorted onsets, matching rows)"""
    d = {}
    for q in range(12):
        r = h[h[:, 2].astype(int) % 12 == q]
        r = r[np.argsort(r[:, 0])]
        d[q] = r
    return d


def match_rate(n, h):
    d = onsets_by_pc(h)
    hit = 0
    for s, p in zip(n[:, 0], n[:, 2].astype(int)):
        t = d[p % 12][:, 0]
        j = np.searchsorted(t, s - TOL)
        hit += j < len(t) and t[j] <= s + TOL
    return hit / max(len(n), 1)


def pairs(n):
    """indices (a, b) of same-key consecutive strikes with b starting <= GAP after a ends."""
    out = []
    for p in np.unique(n[:, 2]):
        idx = np.nonzero(n[:, 2] == p)[0]
        idx = idx[np.argsort(n[idx, 0], kind="stable")]
        for a, b in zip(idx[:-1], idx[1:]):
            if n[b, 0] - n[a, 1] <= GAP:
                out.append((a, b))
    return np.array(out, int).reshape(-1, 2)


def judge(n, P, d, shift=0.0):
    """per pair against one arrangement (pc dict d): -1 no A match, 0 single let go, 1 single held, 2 repeat.
    shift > 0: the chance baseline (looks for the re-strike at B's start + shift, where nothing of ours is)"""
    r = np.full(len(P), -1, np.int8)
    for k, (a, b) in enumerate(P):
        q = int(n[a, 2]) % 12
        x = d[q]
        if not len(x):
            continue
        t = x[:, 0]
        ia = np.nonzero(np.abs(t - n[a, 0]) <= TOL)[0]
        if not len(ia):
            continue
        ja = ia[np.argmin(np.abs(t[ia] - n[a, 0]))]
        ib = np.nonzero((np.abs(t - n[b, 0] - shift) <= TOL) & (x[:, 2] == x[ja, 2] if SAMEKEY else True) & (np.arange(len(t)) != ja) & (t > t[ja] + 0.03))[0]
        if len(ib):
            r[k] = 2
        else:
            r[k] = 1 if x[ja, 1] >= n[b, 0] - HOLD else 0
    return r


def voiced(sid, t):
    f = VOCAL / f"{sid}.npz"
    if not f.exists():
        return np.full(len(t), -1, np.int8)
    z = np.load(f)
    hop, vdb = float(z["hop"]), z["vocals_db"].astype(float)
    v = (vdb > np.percentile(vdb, 99.5) - 35) & (vdb > -60)
    return v[np.clip((t / hop).astype(int), 0, len(v) - 1)].astype(np.int8)


def density(n, secs):
    w = np.zeros(int(np.ceil(secs / WIN)))
    np.add.at(w, np.clip((n[:, 0] // WIN).astype(int), 0, len(w) - 1), 1)
    return w / WIN


def score(a):
    info = {}
    for p in NK.glob("info_*.json"):
        for s in json.loads(p.read_text())["songs"]:
            info[s["song"]] = s
    songs = sorted(info)[a.shard::a.nshards]
    R = {k: [] for k in ("song", "who", "ioi", "gap", "la", "lb", "pitch", "pm", "vel", "voice", "nref", "nmatch",
                         "nrep", "nheld", "nnull", "v3merged")}
    D = []
    for sid in songs:
        s = info[sid]
        if not s["used"] or not (NK / "run" / f"{sid}.npz").exists():
            continue
        z = np.load(NK / "run" / f"{sid}.npz")
        raw, pm, out, secs = z["raw"], z["pm"], z["out"], float(z["secs"])
        H = []
        for aid in s["used"]:
            h = np.load(NK / "midi" / f"{aid}.npy").astype(float)
            lag = s["lags"][aid][0]
            cand = [h.copy() for _ in range(3)]
            cand[1][:, :2] += lag; cand[2][:, :2] -= lag
            H.append(max(cand, key=lambda c: match_rate(raw, c)))
        dicts = [onsets_by_pc(h) for h in H]
        # 1. our pairs (raw), and whether v3 merged them (B gone from the output as its own strike)
        P = pairs(raw)
        if len(P):
            J = np.stack([judge(raw, P, d) for d in dicts])
            J0 = np.stack([judge(raw, P, d, 0.37) for d in dicts])
            outs = {(round(o[0], 3), int(o[2])) for o in out}
            m = J >= 0
            A, B = P[:, 0], P[:, 1]
            R["song"] += [sid] * len(P); R["who"] += [0] * len(P)
            R["ioi"].append(raw[B, 0] - raw[A, 0]); R["gap"].append(raw[B, 0] - raw[A, 1])
            R["la"].append(raw[A, 1] - raw[A, 0]); R["lb"].append(raw[B, 1] - raw[B, 0]); R["pitch"].append(raw[A, 2])
            R["pm"].append(pm[B]); R["vel"].append(raw[B, 3] - raw[A, 3]); R["voice"].append(voiced(sid, raw[B, 0]))
            R["nref"].append(np.full(len(P), len(H))); R["nmatch"].append(m.sum(0)); R["nrep"].append((J == 2).sum(0))
            R["nheld"].append((J == 1).sum(0)); R["nnull"].append((J0 == 2).sum(0))
            R["v3merged"].append(np.array([(round(raw[b, 0], 3), int(raw[b, 2])) not in outs and
                                           (round(raw[a_, 0], 3), int(raw[a_, 2])) in outs for a_, b in P], np.int8))
        # control: each arrangement's own pairs against the others
        if len(H) >= 2:
            for i, h in enumerate(H):
                Ph = pairs(h)
                if not len(Ph):
                    continue
                J = np.stack([judge(h, Ph, d) for j, d in enumerate(dicts) if j != i])
                J0 = np.stack([judge(h, Ph, d, 0.37) for j, d in enumerate(dicts) if j != i])
                A, B = Ph[:, 0], Ph[:, 1]
                R["song"] += [sid] * len(Ph); R["who"] += [1] * len(Ph)
                R["ioi"].append(h[B, 0] - h[A, 0]); R["gap"].append(h[B, 0] - h[A, 1])
                R["la"].append(h[A, 1] - h[A, 0]); R["lb"].append(h[B, 1] - h[B, 0]); R["pitch"].append(h[A, 2])
                R["pm"].append(np.full(len(Ph), np.nan)); R["vel"].append(h[B, 3] - h[A, 3])
                R["voice"].append(voiced(sid, h[B, 0])); R["nref"].append(np.full(len(Ph), len(H) - 1))
                R["nmatch"].append((J >= 0).sum(0)); R["nrep"].append((J == 2).sum(0)); R["nheld"].append((J == 1).sum(0))
                R["nnull"].append((J0 == 2).sum(0)); R["v3merged"].append(np.full(len(Ph), -1, np.int8))
        # 2. density per window
        D.append(dict(song=sid, raw=density(raw, secs).tolist(), out=density(out, secs).tolist(),
                      hum=[density(h[(h[:, 0] >= 0) & (h[:, 0] < secs)], secs).tolist() for h in H],
                      voice=[float(np.mean(v)) if (v := voiced(sid, np.arange(i * WIN, (i + 1) * WIN, 0.5)))[0] >= 0
                             else -1.0 for i in range(int(np.ceil(secs / WIN)))]))
        print(f"{sid}: {len(P)} pairs, {len(H)} arrangements", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"rows_{a.shard}.npz", **{k: (np.array(v) if k in ("song", "who") else
                                                  (np.concatenate(v) if v else np.zeros(0))) for k, v in R.items()})
    (OUT / f"dens_{a.shard}.json").write_text(json.dumps(D))


def report(a):
    Z = [np.load(f) for f in sorted(OUT.glob("rows_*.npz"))]
    r = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files}
    D = sum((json.loads(f.read_text()) for f in sorted(OUT.glob("dens_*.json"))), [])
    ours, hum = r["who"] == 0, r["who"] == 1
    lab = r["nmatch"] > 0
    single = lab & (r["nrep"] == 0)
    rep_share = np.where(lab, r["nrep"] / np.maximum(r["nmatch"], 1), np.nan)
    print(f"songs {len(D)};  our pairs {ours.sum():,} (A matched by a human: {(ours & lab).mean() / ours.mean():.1%});"
          f"  human control pairs {hum.sum():,}")

    def row(name, m):
        m = m & lab
        if m.sum() < 30:
            return
        sm = single[m]
        held = (r["nheld"][m] > 0) & sm
        null = (r["nnull"][m] == 0).mean()
        print(f"  {name:34s} {m.sum():>8,}   single {sm.mean():6.1%}   (held {held.mean():6.1%})   "
              f"chance-single {null:6.1%}   repeat-share {np.nanmean(rep_share[m]):5.2f}")

    for who, wm in (("OURS (raw 0.8 pairs)", ours), ("HUMAN control (own pairs vs the others)", hum)):
        print(f"\n{who}: pairs where a human plays A;  'single' = NO matching arrangement re-strikes at B;"
              f" chance-single = the same test 370 ms after B (a re-strike there is coincidence)")
        row("all", wm)
        for lo, hi in ((0, .15), (.15, .3), (.3, .6), (.6, 9)):
            row(f"start-to-start {lo * 1000:.0f}-{hi * 1000:.0f} ms", wm & (r["ioi"] >= lo) & (r["ioi"] < hi))
        for lo, hi in ((-9, 0), (0, .05), (.05, .15)):
            row(f"gap {lo * 1000:.0f}-{hi * 1000:.0f} ms", wm & (r["gap"] >= lo) & (r["gap"] < hi))
        for lo, hi in ((0, .1), (.1, .25), (.25, 9)):
            row(f"B length {lo * 1000:.0f}-{hi * 1000:.0f} ms", wm & (r["lb"] >= lo) & (r["lb"] < hi))
        for nm, lo, hi in (("bass < C3", 0, 48), ("middle C3-C5", 48, 72), ("high >= C5", 72, 200)):
            row(nm, wm & (r["pitch"] >= lo) & (r["pitch"] < hi))
        row("B softer than A", wm & (r["vel"] < 0)); row("B as loud or louder", wm & (r["vel"] >= 0))
        row("voice sounding", wm & (r["voice"] == 1)); row("voice silent", wm & (r["voice"] == 0))
        for k in (1, 2, 3):
            row(f"{k}{'+' if k == 3 else ''} arrangements match A", wm & ((r["nmatch"] >= 3) if k == 3 else (r["nmatch"] == k)))
    m = ours & lab
    print("\nclean-up v3's merge output vs the humans (our pairs):")
    y = single[m].astype(float)
    pmv = r["pm"][m]
    o = np.argsort(-pmv); tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o])
    auc = float(np.trapz(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    print(f"  AUC of v3's merge probability for 'humans play it single': {auc:.3f}")
    for b in (0.3, 0.5, 0.7):
        sel = pmv >= b
        print(f"  pm >= {b}: {sel.mean():6.1%} of pairs, of which single {y[sel].mean() if sel.any() else 0:6.1%};"
              f"  below: single {y[~sel].mean():6.1%}")
    vm = r["v3merged"][m] == 1
    print(f"  merged in v3's output (keep 0.3 / merge 0.5): {vm.mean():.1%};  single among merged {y[vm].mean():.1%},"
          f" among not merged {y[~vm].mean():.1%}")

    print("\nDENSITY per 10 s window (notes/s; windows where every arrangement has notes)")
    rr, oo, hh, cc_h, cc_o, cc_r = [], [], [], [], [], []
    for s in D:
        raw, out, H = np.array(s["raw"]), np.array(s["out"]), np.array(s["hum"])
        ok = (H > 0).all(0)
        if ok.sum() < 6:
            continue
        rr += raw[ok].tolist(); oo += out[ok].tolist(); hh += np.median(H[:, ok], 0).tolist()
        lg = lambda x: np.log(x + 0.5)
        if len(H) >= 2:
            for i in range(len(H)):
                rest = lg(np.delete(H, i, 0)[:, ok]).mean(0)
                cc_h.append(np.corrcoef(lg(H[i, ok]), rest)[0, 1])
                cc_o.append(np.corrcoef(lg(out[ok]), rest)[0, 1]); cc_r.append(np.corrcoef(lg(raw[ok]), rest)[0, 1])
    rr, oo, hh = map(np.array, (rr, oo, hh))
    print(f"  windows {len(rr):,}: median notes/s ours raw {np.median(rr):.1f}, after clean-up {np.median(oo):.1f},"
          f" human {np.median(hh):.1f}")
    for nm, x in (("raw", rr), ("after clean-up", oo)):
        q = np.percentile(x, [0, 25, 50, 75, 100])
        print(f"  human / ours {nm} by our density quartile: " + "  ".join(
            f"{np.median(hh[(x >= q[i]) & (x <= q[i + 1])] / np.maximum(x[(x >= q[i]) & (x <= q[i + 1])], .1)):.2f}"
            for i in range(4)))
    print(f"  shape across a song (corr. of log notes/s per window with the OTHER arrangements' mean, median over "
          f"{len(cc_h)} arrangement-songs):  human {np.nanmedian(cc_h):.2f}   ours after clean-up "
          f"{np.nanmedian(cc_o):.2f}   ours raw {np.nanmedian(cc_r):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--samekey", action="store_true"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    score(a) if a.score else report(a)


if __name__ == "__main__":
    main()
