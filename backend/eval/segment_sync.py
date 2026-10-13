"""Segment-anchor sync, scored on held-out notes.

Every W seconds of arrangement gets its own offset (one anchor per segment);
offsets are solved jointly by dynamic programming with a penalty on jumps
between neighbouring segments, so the warp is smooth and piecewise.  Anchors
are fitted on the EVEN-indexed onsets only and scored on the ODD ones, so a
method cannot score well just by moving the notes it is graded on.

Versions (on-attack share of held-out onsets, 25 / 50 ms):
  original            as shipped
  global              one offset for the whole song (+-1.5 s)
  seg4 / seg8 / seg16 segment anchors every 4 / 8 / 16 s (+-0.5 s)
  realign_warp        realign_m2m's 4 s local shift, fitted on even notes
  realign_snap        realign_m2m in full (self-scored: snapping grades itself)

    python eval/segment_sync.py --m2m <dir> --out <dir> --shard i/n   (then --report)
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from realign_m2m import attacks, realign, on_attack, DT, CONV          # noqa: E402

LAM = 0.01      # jump penalty per frame of offset change (env is z-scored)


def seg_offsets(fit, env, win, max_shift, lam=LAM):
    """Viterbi over per-segment offsets; returns (segment centres, offsets)."""
    T = len(env); L = int(round(max_shift / DT)); ks = np.arange(-L, L + 1)
    starts = np.arange(fit[0], fit[-1] + 1e-6, win) if win else np.array([fit[0]])
    w = win or (fit[-1] - fit[0] + 1)
    S = np.zeros((len(starts), len(ks)))
    for i, s in enumerate(starts):
        sel = fit[(fit >= s) & (fit < s + w)]
        if len(sel) == 0:
            continue
        idx = np.round((sel[None, :] + ks[:, None] * DT + CONV) / DT).astype(int)
        S[i] = np.where((idx >= 0) & (idx < T), env[np.clip(idx, 0, T - 1)], 0).sum(1)
    pen = lam * np.abs(ks[:, None] - ks[None, :])
    acc, back = S[0].copy(), []
    for i in range(1, len(starts)):
        m = acc[:, None] - pen
        b = m.argmax(0); back.append(b); acc = m[b, np.arange(len(ks))] + S[i]
    path = [int(acc.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return starts + w / 2, ks[path[::-1]] * DT


def realign_warp(fit_notes, env, tp, held):
    """realign_m2m's smoothed local shift curve, read off by fitting on fit_notes."""
    moved, _ = realign(fit_notes, env, np.array([-1e9, 1e9]))     # tp far away -> no snap
    o = fit_notes[np.argsort(fit_notes[:, 0], kind="stable"), 0]
    return held + np.interp(held, o, moved[:, 0] - o)


def score(on, tp):
    d = on_attack(np.c_[on, on], tp)
    return float(np.mean(d <= .025)), float(np.mean(d <= .05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    V = ["original", "global", "seg4", "seg8", "seg16", "realign_warp", "realign_snap"]
    if a.report:
        r = [json.loads(l) for f in sorted(out.glob("s_*.jsonl")) for l in open(f)]
        print(f"{len(r)} songs, scored on held-out (odd) onsets\n\n{'version':<14}{'<=25 ms':>9}{'<=50 ms':>9}  beats original")
        for v in V:
            print(f"{v:<14}{np.mean([x[v][0] for x in r]):>9.1%}{np.mean([x[v][1] for x in r]):>9.1%}"
                  f"{np.mean([x[v][0] > x['original'][0] for x in r]):>10.0%}")
        return
    sp = json.loads((pathlib.Path(a.m2m) / "m2m_splits.json").read_text())
    ids = sorted(sp["train"] + sp["validation"] + sp["test"])
    si, sn = (int(v) for v in a.shard.split("/"))
    with open(out / f"s_{si:02d}.jsonl", "w") as fo:
        for sid in ids[si::sn]:
            n = np.load(pathlib.Path(a.m2m) / "midi_numpy" / f"{sid}.npy").astype(np.float64)
            on = np.unique(np.round(n[:, 0], 3))
            if len(on) < 40:
                continue
            env, tp = attacks(pathlib.Path(a.m2m) / "audio" / f"{sid}.wav")
            fit, held = on[0::2], on[1::2]
            rec = dict(id=sid, original=score(held, tp))
            for name, win, ms in (("global", 0, 1.5), ("seg4", 4, .5), ("seg8", 8, .5), ("seg16", 16, .5)):
                c, o = seg_offsets(fit, env, win, ms)
                rec[name] = score(held + (np.interp(held, c, o) if len(c) > 1 else o[0]), tp)
            rec["realign_warp"] = score(realign_warp(np.c_[fit, fit + .1], env, tp, held), tp)
            rec["realign_snap"] = score(realign(n, env, tp)[0][:, 0], tp)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"  {sid}: " + " ".join(f"{k} {rec[k][0]:.0%}" for k in V), flush=True)


if __name__ == "__main__":
    main()
