"""Our transcription vs ALL the human arrangements of a song, in detail (the user, 2026-11-02).

Held-out Music2MIDI songs with >= 3 arrangements that pass the lag gate.  Ours = the hosted chain (the base at 0.8
+ clean-up v4 without beats: keep 0.3, merge 0.5, never-kept 0.8), plus the arranger's raw 0.8 notes.  Inputs are
already on the cluster: /scratch/basu.ang/human_data (raw notes), /scratch/basu.ang/position_repeat/probs (the
clean-up's probabilities), never_kept (arrangements + lags), solo_det/m2m_sim (vocal curves).

A note "matches" another source if that source has the same PITCH CLASS within 150 ms (arrangers move lines by
octaves); exact-pitch matches are reported too.  Human arrangements are moved by their lag to the audio.

    python eval/human_compare.py [--songs id,id,...] [--min-arr 3]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import position_repeat as PR          # noqa: E402  (stages, human_notes)
import human_merge as HM              # noqa: E402  (voiced)

NKDIR, HUMAN, PROBS = PR.NKDIR, PR.HUMAN, PR.OUT / "probs"
TOL = 0.15
PR.NKBAR = 0.8                        # the hosted never-kept bar


CHANCE = 0.37                          # the chance level: the same test with the reference moved by 370 ms


def has(X, R, tol=TOL, exact=False, shift=0.0):
    """per note of X: does R have the pitch class (or exact pitch) within tol?  shift moves R in time (chance)."""
    out = np.zeros(len(X), bool)
    if not len(R) or not len(X):
        return out
    o = np.argsort(R[:, 0]); t, p = R[o, 0] + shift, R[o, 2].astype(int)
    for i, (s, q) in enumerate(zip(X[:, 0], X[:, 2].astype(int))):
        j0, j1 = np.searchsorted(t, s - tol), np.searchsorted(t, s + tol, "right")
        out[i] = np.any((p[j0:j1] == q) if exact else (p[j0:j1] % 12 == q % 12))
    return out


def nearest_octave(X, R, tol=TOL):
    """per note of X matched by pitch class: R's pitch minus X's pitch for the nearest such note (in semitones)."""
    d = np.full(len(X), np.nan)
    if not len(R):
        return d
    o = np.argsort(R[:, 0]); t, p = R[o, 0], R[o, 2].astype(int)
    for i, (s, q) in enumerate(zip(X[:, 0], X[:, 2].astype(int))):
        j0, j1 = np.searchsorted(t, s - tol), np.searchsorted(t, s + tol, "right")
        c = p[j0:j1][p[j0:j1] % 12 == q % 12]
        if len(c):
            d[i] = (c - q)[np.argmin(np.abs(c - q))]
    return d


def roles(X):
    """per note: 'top' (highest sounding at its start), 'bass' (lowest), 'inner', from the same source."""
    r = np.array(["inner"] * len(X), dtype=object)
    for i, (s, e, p) in enumerate(X[:, :3]):
        m = (X[:, 0] <= s + 0.03) & (X[:, 1] > s + 0.03)
        if not m.any():
            continue
        ps = X[m, 2]
        if p >= ps.max():
            r[i] = "top"
        elif p <= ps.min() and m.sum() > 1:
            r[i] = "bass"
    return r


def semitone_neighbour(X):
    """per note: shorter than 150 ms and next to a longer note 1-2 semitones away (the slide shape)."""
    s, e, p = X[:, 0], X[:, 1], X[:, 2]
    L = e - s
    out = np.zeros(len(X), bool)
    for i in np.nonzero(L < 0.15)[0]:
        d = np.abs(p - p[i])
        out[i] = np.any((d >= 1) & (d <= 2) & (s < e[i] + 0.1) & (e > s[i] - 0.05) & (L >= 1.5 * L[i]))
    return out


def describe(X, dur):
    """overall style of one source."""
    if not len(X):
        return {}
    s = np.sort(X[:, 0]); ioi = np.diff(s); ioi = ioi[ioi > 0.03]
    on_groups = np.split(X[np.argsort(X[:, 0]), 2], np.nonzero(np.diff(s) > 0.03)[0] + 1)
    L = X[:, 1] - X[:, 0]; p = X[:, 2]
    r = roles(X)
    return dict(nps=len(X) / dur, chord=float(np.mean([len(g) for g in on_groups])),
                p5=float(np.percentile(p, 5)), p50=float(np.median(p)), p95=float(np.percentile(p, 95)),
                bass=float(np.mean(p < 48)), mid=float(np.mean((p >= 48) & (p < 72))), high=float(np.mean(p >= 72)),
                len50=float(np.median(L)), short=float(np.mean(L < 0.1)), ioi50=float(np.median(ioi)) if len(ioi) else 0.0,
                top=float(np.mean(r == "top")), pch=np.bincount(p.astype(int) % 12, weights=np.clip(L, 0.05, 2), minlength=12))


def f1(A, B, tol=TOL, exact=False):
    """ONE-TO-ONE note F1 (each note matched at most once, nearest first): pitch class or exact pitch within tol."""
    if not len(A) or not len(B):
        return 0.0
    ka = A[:, 2].astype(int) if exact else A[:, 2].astype(int) % 12
    kb = B[:, 2].astype(int) if exact else B[:, 2].astype(int) % 12
    m = 0
    for q in np.unique(np.r_[ka, kb]):
        a, b = np.sort(A[ka == q, 0]), np.sort(B[kb == q, 0])
        used = np.zeros(len(b), bool)
        for t in a:
            d = np.abs(b - t); d[used] = 9
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= tol:
                used[j] = True; m += 1
    return 2 * m / (len(A) + len(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default=""); ap.add_argument("--min-arr", type=int, default=3)
    a = ap.parse_args()
    S = {s["audio"]: s for s in json.loads((NKDIR / "songs.json").read_text())["songs"]}
    info = {}
    for p in NKDIR.glob("info_*.json"):
        for s in json.loads(p.read_text())["songs"]:
            info[s["song"]] = s
    ids = a.songs.split(",") if a.songs else sorted(
        (sid for sid, s in info.items() if S.get(sid, {}).get("split") == "test" and len(s["used"]) >= a.min_arr
         and (HUMAN / f"{sid}.npz").exists() and (PROBS / f"{sid}.npz").exists()),
        key=lambda sid: -len(info[sid]["used"]))
    T = {k: [] for k in ("ours", "hum", "miss", "desc", "pair", "oct")}
    for sid in ids:
        z = np.load(HUMAN / f"{sid}.npz"); raw = z["notes"].astype(float)
        st = PR.stages(raw, np.load(PROBS / f"{sid}.npz")["p"])
        fin = raw[st == 0]
        H = PR.human_notes(sid, info, raw)
        dur = float(raw[:, 0].max())
        H = [h[(h[:, 0] >= 0) & (h[:, 0] <= dur + 1)] for h in H]
        nH = len(H)
        vo_f = HM.voiced(sid, fin[:, 0]); r_f = roles(fin); sn_f = semitone_neighbour(fin)
        # 1. our final notes: in how many arrangements?
        cnt_f = np.sum([has(fin, h) for h in H], 0); cnt_r = np.sum([has(raw, h) for h in H], 0)
        cnt_c = np.sum([has(fin, h, shift=CHANCE) for h in H], 0)
        exact_any = np.any([has(fin, h, exact=True) for h in H], 0)
        T["ours"].append(dict(song=sid, nH=nH, n=len(fin), nraw=len(raw), all=int((cnt_f == nH).sum()),
                              some=int(((cnt_f > 0) & (cnt_f < nH)).sum()), none=int((cnt_f == 0).sum()),
                              exact=int(exact_any.sum()), c_all=int((cnt_c == nH).sum()), c_none=int((cnt_c == 0).sum()),
                              raw_all=int((cnt_r == nH).sum()), raw_none=int((cnt_r == 0).sum()),
                              none_by={k: int(((cnt_f == 0) & m).sum()) for k, m in (
                                  ("bass<C3", fin[:, 2] < 48), ("mid", (fin[:, 2] >= 48) & (fin[:, 2] < 72)), ("high>=C5", fin[:, 2] >= 72),
                                  ("top", r_f == "top"), ("inner", r_f == "inner"), ("bass role", r_f == "bass"),
                                  ("short<100ms", fin[:, 1] - fin[:, 0] < 0.1), ("slide-shaped", sn_f),
                                  ("voice sounding", vo_f == 1), ("voice silent", vo_f == 0))},
                              all_by={k: int(((cnt_f == nH) & m).sum()) for k, m in (
                                  ("bass<C3", fin[:, 2] < 48), ("mid", (fin[:, 2] >= 48) & (fin[:, 2] < 72)), ("high>=C5", fin[:, 2] >= 72),
                                  ("top", r_f == "top"), ("inner", r_f == "inner"), ("bass role", r_f == "bass"))},
                              tot_by={k: int(m.sum()) for k, m in (
                                  ("bass<C3", fin[:, 2] < 48), ("mid", (fin[:, 2] >= 48) & (fin[:, 2] < 72)), ("high>=C5", fin[:, 2] >= 72),
                                  ("top", r_f == "top"), ("inner", r_f == "inner"), ("bass role", r_f == "bass"),
                                  ("short<100ms", fin[:, 1] - fin[:, 0] < 0.1), ("slide-shaped", sn_f),
                                  ("voice sounding", vo_f == 1), ("voice silent", vo_f == 0))}))
        # 2. human notes by agreement, and whether we have them
        for i, h in enumerate(H):
            others = [has(h, H[j]) for j in range(nH) if j != i]
            k = np.sum(others, 0)
            lvl = np.where(k == nH - 1, "all", np.where(k == 0, "only this one", "some"))
            in_f, in_r = has(h, fin), has(h, raw)
            in_c = has(h, fin, shift=-CHANCE)          # ours moved 370 ms: chance
            rh = roles(h); vo_h = HM.voiced(sid, h[:, 0])
            for L in ("all", "some", "only this one"):
                m = lvl == L
                T["hum"].append(dict(song=sid, lvl=L, n=int(m.sum()), fin=int((m & in_f).sum()), raw=int((m & in_r).sum()),
                                     chance=int((m & in_c).sum())))
            # 3. notes every human plays that our final output misses
            mm = (lvl == "all") & ~in_f
            T["miss"].append(dict(song=sid, n_all=int((lvl == "all").sum()), miss=int(mm.sum()),
                                  arranger_never=int((mm & ~in_r).sum()), cleanup_removed=int((mm & in_r).sum()),
                                  by={kk: int((mm & m).sum()) for kk, m in (
                                      ("bass<C3", h[:, 2] < 48), ("mid", (h[:, 2] >= 48) & (h[:, 2] < 72)), ("high>=C5", h[:, 2] >= 72),
                                      ("top", rh == "top"), ("inner", rh == "inner"), ("bass role", rh == "bass"),
                                      ("long>=500ms", h[:, 1] - h[:, 0] >= 0.5), ("voice sounding", vo_h == 1))},
                                  tot={kk: int(((lvl == "all") & m).sum()) for kk, m in (
                                      ("bass<C3", h[:, 2] < 48), ("mid", (h[:, 2] >= 48) & (h[:, 2] < 72)), ("high>=C5", h[:, 2] >= 72),
                                      ("top", rh == "top"), ("inner", rh == "inner"), ("bass role", rh == "bass"),
                                      ("long>=500ms", h[:, 1] - h[:, 0] >= 0.5), ("voice sounding", vo_h == 1))}))
            # 4. octave choices: human pitch - our pitch, for our top-line notes matched by pitch class
            d = nearest_octave(fin[r_f == "top"], h)
            T["oct"] += [x for x in d[~np.isnan(d)].tolist()]
        # 5. pairwise agreement
        for i in range(nH):
            T["pair"].append(dict(song=sid, kind="ours vs human", f1=f1(fin, H[i]), strict=f1(fin, H[i], 0.05, True)))
            T["pair"].append(dict(song=sid, kind="raw arranger vs human", f1=f1(raw, H[i]), strict=f1(raw, H[i], 0.05, True)))
            for j in range(i + 1, nH):
                T["pair"].append(dict(song=sid, kind="human vs human", f1=f1(H[i], H[j]), strict=f1(H[i], H[j], 0.05, True)))
        # 6. overall style
        T["desc"].append(dict(song=sid, src="ours final", **describe(fin, dur)))
        T["desc"].append(dict(song=sid, src="ours raw", **describe(raw, dur)))
        for h in H:
            T["desc"].append(dict(song=sid, src="human", **describe(h, dur)))
        print(f"{sid}: {nH} arrangements, ours {len(fin)} notes (raw {len(raw)})", flush=True)
    report(T)


def pct(a, b):
    return f"{a / b:.0%}" if b else "-"


def report(T):
    O = T["ours"]
    print("\n=== 1. OUR NOTES (final): in how many of the song's human arrangements? (pitch class, 150 ms)")
    print(f"{'song':>8} {'arr':>3} {'ours':>6} {'in ALL':>7} {'in SOME':>8} {'in NONE':>8} {'exact pitch in any':>19} | raw: {'in ALL':>7} {'in NONE':>8} | chance: {'in ALL':>6} {'in NONE':>8}")
    for r in O:
        print(f"{r['song']:>8} {r['nH']:3d} {r['n']:6d} {pct(r['all'], r['n']):>7} {pct(r['some'], r['n']):>8} {pct(r['none'], r['n']):>8} "
              f"{pct(r['exact'], r['n']):>19} | {pct(r['raw_all'], r['nraw']):>12} {pct(r['raw_none'], r['nraw']):>8} | "
              f"{pct(r['c_all'], r['n']):>14} {pct(r['c_none'], r['n']):>8}")
    tot = lambda k: sum(r[k] for r in O)
    print(f"{'ALL':>8} {'':3} {tot('n'):6d} {pct(tot('all'), tot('n')):>7} {pct(tot('some'), tot('n')):>8} {pct(tot('none'), tot('n')):>8} "
          f"{pct(tot('exact'), tot('n')):>19} | {pct(tot('raw_all'), tot('nraw')):>12} {pct(tot('raw_none'), tot('nraw')):>8} | "
          f"{pct(tot('c_all'), tot('n')):>14} {pct(tot('c_none'), tot('n')):>8}")
    print("\n  our notes by kind: share in ALL arrangements / share in NONE")
    for k in O[0]["tot_by"]:
        n = sum(r["tot_by"][k] for r in O)
        a_ = sum(r["all_by"].get(k, 0) for r in O) if k in O[0]["all_by"] else None
        print(f"  {k:>16}: {n:6d} notes ({pct(n, tot('n')):>4} of ours) | in ALL {pct(a_, n) if a_ is not None else '  -':>5} | in NONE {pct(sum(r['none_by'][k] for r in O), n):>5}")
    print("\n=== 2. HUMAN NOTES by how many arrangers agree, and how many of them we have")
    print(f"{'agreement':>15} {'human notes':>12} {'share':>6} {'ours final has':>15} {'raw arranger has':>17} {'chance':>7}")
    Hh = T["hum"]; N = sum(r["n"] for r in Hh)
    for L in ("all", "some", "only this one"):
        n = sum(r["n"] for r in Hh if r["lvl"] == L)
        print(f"{L:>15} {n:12d} {pct(n, N):>6} {pct(sum(r['fin'] for r in Hh if r['lvl'] == L), n):>15} "
              f"{pct(sum(r['raw'] for r in Hh if r['lvl'] == L), n):>17} {pct(sum(r['chance'] for r in Hh if r['lvl'] == L), n):>7}")
    M = T["miss"]
    nall, miss = sum(r["n_all"] for r in M), sum(r["miss"] for r in M)
    print(f"\n=== 3. NOTES EVERY ARRANGER PLAYS that our final output misses: {miss:,} of {nall:,} ({pct(miss, nall)})")
    print(f"  arranger never wrote it: {pct(sum(r['arranger_never'] for r in M), miss)} | clean-up removed it: "
          f"{pct(sum(r['cleanup_removed'] for r in M), miss)}")
    for k in M[0]["by"]:
        n = sum(r["tot"][k] for r in M); m = sum(r["by"][k] for r in M)
        print(f"  {k:>16}: missed {m:6d} of {n:6d} consensus notes of this kind ({pct(m, n)})")
    d = np.array(T["oct"])
    if len(d):
        print("\n=== 4. OCTAVE CHOICES for our melody (top-line) notes matched by pitch class: human pitch - ours")
        for v, lab in ((-24, "2 oct down"), (-12, "1 oct down"), (0, "same octave"), (12, "1 oct up"), (24, "2 oct up")):
            print(f"  {lab:>12}: {np.mean(d == v):.0%}")
    print("\n=== 5. PAIRWISE AGREEMENT (one-to-one note F1)")
    P = T["pair"]
    for kind in ("human vs human", "ours vs human", "raw arranger vs human"):
        v = [r["f1"] for r in P if r["kind"] == kind]; w = [r["strict"] for r in P if r["kind"] == kind]
        print(f"  {kind:>22}: pitch class 150 ms median {np.median(v):.2f} (range {min(v):.2f}-{max(v):.2f}) | "
              f"exact pitch 50 ms median {np.median(w):.2f} (range {min(w):.2f}-{max(w):.2f}) | {len(v)} pairs")
    print(f"  per song: " + "  ".join(f"{s}: humans {np.median([r['f1'] for r in P if r['song'] == s and r['kind'] == 'human vs human']):.2f} / "
                                     f"ours {np.median([r['f1'] for r in P if r['song'] == s and r['kind'] == 'ours vs human']):.2f}"
                                     for s in dict.fromkeys(r["song"] for r in P)))
    print("\n=== 6. OVERALL STYLE (median over songs; humans = median over arrangements)")
    D = T["desc"]
    keys = [("nps", "notes / s", "{:.1f}"), ("chord", "notes per onset", "{:.2f}"), ("p5", "lowest (p5, MIDI)", "{:.0f}"),
            ("p50", "middle (MIDI)", "{:.0f}"), ("p95", "highest (p95, MIDI)", "{:.0f}"), ("bass", "share below C3", "{:.0%}"),
            ("mid", "share C3-C5", "{:.0%}"), ("high", "share C5+", "{:.0%}"), ("len50", "median length (s)", "{:.2f}"),
            ("short", "share < 100 ms", "{:.0%}"), ("ioi50", "median gap between onsets (s)", "{:.3f}"), ("top", "share on the top line", "{:.0%}")]
    print(f"  {'':>30} {'ours final':>11} {'ours raw':>9} {'humans':>8}")
    for k, lab, fmt in keys:
        vals = [np.median([r[k] for r in D if r["src"] == s and k in r]) for s in ("ours final", "ours raw", "human")]
        print(f"  {lab:>30} " + " ".join(f"{fmt.format(v):>{w}}" for v, w in zip(vals, (11, 9, 8))))
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    pc_o = [cos(r["pch"], np.mean([q["pch"] for q in D if q["song"] == r["song"] and q["src"] == "human"], 0))
            for r in D if r["src"] == "ours final"]
    print(f"  {'pitch-class profile vs humans':>30} {np.median(pc_o):>11.3f}   (cosine; 1 = same use of the 12 notes)")


if __name__ == "__main__":
    main()
