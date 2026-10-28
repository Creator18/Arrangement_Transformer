"""Our own beat / downbeat decoding (2026-10-28; replaces madmom's DBN): frame activations from beat_train.py ->
beat and downbeat times.

Beats: a bar-pointer HMM (Krebs et al. 2015, the idea behind madmom's DBN), solved with Viterbi.  States are
(beat interval in frames, position inside the beat); position steps by one per frame; at the end of a beat the
interval may change to a neighbouring one (tempo drift), with a cost that grows with the change.  A frame near a beat
(the first frames of the interval) is scored by the beat activation, the others by its complement.

Downbeats: the beats found above, each scored by the downbeat activation around it; a small Viterbi over (meter 3 or
4, position in the bar) with a rare switch, so the bar phase is stable but can recover.
"""
import numpy as np

FPS = 100


def beats(act, min_bpm=55.0, max_bpm=215.0, transition_lambda=100.0, obs_lambda=16, tempo_prior=None):
    """act: (T,) beat activation in [0, 1] at FPS.  tempo_prior: optional (n_intervals,) log weights.  -> beat times (s)."""
    T = len(act)
    if T < 2:
        return np.zeros(0)
    lo, hi = int(np.floor(60.0 * FPS / max_bpm)), int(np.ceil(60.0 * FPS / min_bpm))
    iv = np.arange(lo, hi + 1)                                  # beat intervals (frames)
    start = np.concatenate([[0], np.cumsum(iv)[:-1]])           # first state of each interval
    S = int(iv.sum())
    pos = np.concatenate([np.arange(n) for n in iv])            # position in the beat of each state
    ivs = np.repeat(iv, iv)
    beat_state = pos < np.maximum(1, ivs // obs_lambda)         # the "near a beat" states
    # tempo change at the end of a beat: last state of interval i -> first state of interval j
    r = iv[None, :] / iv[:, None]
    tr = -transition_lambda * np.abs(r - 1.0)
    tr = tr - np.logaddexp.reduce(tr, axis=1, keepdims=True)    # (i -> j) log probs
    if tempo_prior is not None:
        tr = tr + np.asarray(tempo_prior)[None, :]
    a = np.clip(act.astype(np.float64), 1e-6, 1 - 1e-6)
    lb, lnb = np.log(a), np.log((1 - a) / (obs_lambda - 1))
    delta = np.full(S, -np.log(S)) + np.where(beat_state, lb[0], lnb[0])
    nxt = np.arange(S) + 1                                      # inside an interval: one step on
    last = start + iv - 1
    back_to = np.zeros((T, len(iv)), np.int32)                  # which interval each beat start came from
    for t in range(1, T):
        new = np.empty(S)
        inner = np.ones(S, bool); inner[last] = False
        new[nxt[inner]] = delta[inner]
        cand = delta[last][:, None] + tr                        # (from i, to j)
        bi = cand.argmax(0)
        new[start] = cand[bi, np.arange(len(iv))]
        back_to[t] = bi
        delta = new + np.where(beat_state, lb[t], lnb[t])
    # backtrack: walk the interval / position path
    s = int(delta.argmax())
    j = int(np.searchsorted(start, s, side="right") - 1)
    p = s - start[j]
    out = []
    for t in range(T - 1, -1, -1):
        if p == 0:
            out.append(t)
            if t > 0:
                j = int(back_to[t, j]); p = iv[j] - 1
        else:
            p -= 1
    b = np.array(sorted(out), float)
    # place each beat at the activation peak of its "near a beat" window (sub-frame: parabola)
    res = []
    for f in b.astype(int):
        w = act[max(0, f - 3):f + 4]
        k = max(0, f - 3) + int(np.argmax(w))
        if 0 < k < T - 1:
            y0, y1, y2 = act[k - 1], act[k], act[k + 1]
            d = 0.5 * (y0 - y2) / max(1e-9, y0 - 2 * y1 + y2)
            k = k + float(np.clip(d, -0.5, 0.5))
        res.append(k / FPS)
    res = np.array(res)
    if len(res) > 1:                                            # the first / last beat need some evidence
        keep = np.array([act[int(round(x * FPS))] > 0.05 for x in res])
        keep[1:-1] = True
        res = res[keep]
    return res


def downbeats(beat_times, dact, meters=(3, 4), switch=1e-3):
    """beat_times (s), dact: (T,) downbeat activation.  -> (downbeat times, meter per beat)."""
    n = len(beat_times)
    if n < 2:
        return np.zeros(0), np.zeros(n, int)
    T = len(dact)
    f = np.clip(np.round(np.asarray(beat_times) * FPS).astype(int), 0, T - 1)
    o = np.array([dact[max(0, x - 2):x + 3].max() for x in f])
    o = np.clip(o, 1e-6, 1 - 1e-6)
    states = [(m, k) for m in meters for k in range(m)]          # k = position in the bar (0 = downbeat)
    S = len(states)
    lo = np.array([[np.log(o[i]) if k == 0 else np.log(1 - o[i]) for (m, k) in states] for i in range(n)])
    stay, jump = np.log(1 - switch), np.log(switch / S)
    delta = lo[0] - np.log(S)
    back = np.zeros((n, S), int)
    for i in range(1, n):
        new = np.full(S, -np.inf)
        for s, (m, k) in enumerate(states):
            # natural successor of each state; plus a rare jump from anywhere
            for q, (m2, k2) in enumerate(states):
                v = delta[q] + (stay if (m2 == m and (k2 + 1) % m2 == k) else jump)
                if v > new[s]:
                    new[s], back[i, s] = v, q
        delta = new + lo[i]
    s = int(delta.argmax())
    path = [s]
    for i in range(n - 1, 0, -1):
        s = back[i, s]; path.append(s)
    path = path[::-1]
    pos = np.array([states[s][1] for s in path]); met = np.array([states[s][0] for s in path])
    return np.asarray(beat_times)[pos == 0], met
