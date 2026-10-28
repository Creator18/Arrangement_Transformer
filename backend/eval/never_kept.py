"""Which of our notes do humans NEVER keep?  (the user's idea, 2026-10-28)

The hosted chain (the base at 0.8 + clean-up v3 keep 0.3 / merge 0.5) on every Music2MIDI song we have audio for (925
songs, one audio each), scored against ALL the song's human arrangements (1,520; 271 songs have 2-17).  Not "what
do humans play", only "which of our notes does no human keep": a note counts as kept if ANY arrangement has the same
pitch class (any octave: arrangers move the sung line) starting within 150 ms.  Measurement only, nothing is trained.

    --beats  (CPU) per song: madmom beats for the clean-up, cached in <out>/beats (madmom is ~80 of the ~80 s a song
             takes; left inside --run it kept the GPU idle and the cluster cancelled the jobs for idling)
    --run    (GPU) per song: the base's raw 0.8 notes, the clean-up's keep / merge probabilities and its output;
             --cached-only skips songs whose beats are not cached yet
    --score  (CPU) per song: every arrangement's lag against the audio (gate |lag| <= 0.5 s, as the training
             manifest did), then one row per note of ours (raw and after clean-up) and per human note (the control:
             each arrangement against the OTHER arrangements of its song)
    --report merges the rows and prints the tables

    python eval/never_kept.py --beats --songs songs.json --out /scratch/basu.ang/never_kept --shard k --nshards n
    python eval/never_kept.py --run   --songs songs.json --out /scratch/basu.ang/never_kept --shard k --nshards n
    python eval/never_kept.py --score --songs songs.json --out /scratch/basu.ang/never_kept --shard k --nshards n
    python eval/never_kept.py --report --out /scratch/basu.ang/never_kept
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
SR = 24000
AUDIO = pathlib.Path("/scratch/basu.ang/solo_det/m2m")          # 24 kHz float arrays, one per arrangement id
VOCAL = pathlib.Path("/scratch/basu.ang/solo_det/m2m_sim")      # Demucs vocal curves (where we have them)
TOL_PC, TOL_EXACT, LAG_TOL = 0.15, 0.05, 0.5
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def beats(a, songs):
    """madmom on the CPU, one cached record per song still to run (what cleanup.song_beats returns)."""
    import clip_beats
    run_dir, out = pathlib.Path(a.out) / "run", pathlib.Path(a.out) / "beats"
    out.mkdir(parents=True, exist_ok=True)
    for s in songs:
        f = out / f"{s['audio']}.npz"
        if f.exists() or (run_dir / f"{s['audio']}.npz").exists():
            continue
        z = clip_beats.track(np.load(AUDIO / f"{s['audio']}.npy").astype(np.float32))
        np.savez(out / f"{s['audio']}.tmp.npz", **z)
        os.replace(out / f"{s['audio']}.tmp.npz", f)
        print(f"{s['audio']}: {len(z['beats'])} beats", flush=True)


def run(a, songs):
    os.environ["ARRANGEMENT_CLEANUP"] = a.cleanup
    import soundfile as sf
    from app.services import cleanup
    from app.services import spectral_transcription as ST
    cleanup.CKPT, cleanup.KEEP, cleanup.MERGE, cleanup.QUANT = a.cleanup, 0.3, 0.5, False
    cap = {}
    probs0, apply0 = cleanup.probs, cleanup.apply

    def probs(notes, *x, **k):
        r = probs0(notes, *x, **k)
        cap["pk"], cap["pm"] = r[0], r[2]
        return r

    def apply(notes, *x, **k):
        cap["raw"] = [dict(g) for g in notes]
        return apply0(notes, *x, **k)
    cleanup.probs, cleanup.apply = probs, apply
    cached = pathlib.Path(a.out) / "beats"
    beats0 = cleanup.song_beats
    cleanup.song_beats = lambda p: (dict(np.load(cached / f"{cur}.npz")) if (cached / f"{cur}.npz").exists()
                                    else beats0(p))
    out = pathlib.Path(a.out) / "run"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"tmp_{a.shard}.wav"
    arr = lambda ns: np.array([[g["onset"], g["offset"], g["pitch"], g["velocity"]] for g in ns], float).reshape(-1, 4)
    for s in songs:
        f = out / f"{s['audio']}.npz"
        if f.exists() or (a.cached_only and not (cached / f"{s['audio']}.npz").exists()):
            continue
        cur = s["audio"]
        y = np.load(AUDIO / f"{s['audio']}.npy").astype(np.float32)
        sf.write(tmp, y, SR)
        cap.clear()
        notes = ST.transcribe(tmp, onset_thr=0.8, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
        np.savez(f, raw=arr(cap["raw"]), pk=np.asarray(cap["pk"], np.float32), pm=np.asarray(cap["pm"], np.float32),
                 out=arr(notes), secs=len(y) / SR)
        print(f"{s['audio']}: {len(cap['raw'])} raw -> {len(notes)} notes, {len(y) / SR:.0f} s", flush=True)


FPS, MAX_LAG_S = 50.0, 8.0          # as eval/music2midi_alignment_check.py (not on the cluster), which made the manifest


def best_lag(env, dens):
    """(lag in seconds at peak correlation, peak correlation): music2midi_alignment_check.best_lag."""
    n = min(len(env), len(dens))
    env, dens = env[:n], dens[:n]
    m = int(MAX_LAG_S * FPS)
    best, bl = -2.0, 0
    for L in np.arange(-m, m + 1, 2):
        a, b = (env[L:], dens[:n - L] if L else dens) if L >= 0 else (env[:n + L], dens[-L:])
        if len(a) < FPS * 10:
            continue
        c = float(np.dot(a, b) / len(a))
        if c > best:
            best, bl = c, L
    return bl / FPS, best


def lag_of(env, hum):
    d = np.zeros(len(env))
    np.add.at(d, np.clip((hum[:, 0] * FPS).astype(int), 0, len(env) - 1), 1.0)
    k = np.hanning(7); k /= k.sum()
    d = np.convolve(d, k, mode="same")
    return best_lag(env, (d - d.mean()) / (d.std() + 1e-9))


def song_key(hums):
    h = np.zeros(12)
    for x in hums:
        np.add.at(h, x[:, 2].astype(int) % 12, np.clip(x[:, 1] - x[:, 0], 0.05, 2.0))
    best = max((np.corrcoef(np.roll(p, k), h)[0, 1], k, m) for k in range(12) for p, m in ((MAJ, 0), (MIN, 1)))
    steps = np.array([0, 2, 4, 5, 7, 9, 11]) if best[2] == 0 else np.array([0, 2, 3, 5, 7, 8, 10])
    return set(((steps + best[1]) % 12).tolist()), best


def kept_by(n, refs, tol, pc):
    """per note of n: does any reference arrangement in refs have it (pitch class or exact key) within tol?"""
    hit = np.zeros(len(n), bool)
    for r in refs:
        if not len(r):
            continue
        o = np.argsort(r[:, 0]); t, p = r[o, 0], r[o, 2].astype(int)
        for i, (s, q) in enumerate(zip(n[:, 0], n[:, 2].astype(int))):
            if hit[i]:
                continue
            j0, j1 = np.searchsorted(t, s - tol), np.searchsorted(t, s + tol, "right")
            hit[i] = np.any((p[j0:j1] % 12 == q % 12) if pc else (p[j0:j1] == q))
    return hit


def shape(n, scale):
    """per note: length, out of key, slide-shaped, double press, quick re-strike."""
    s, e, p = n[:, 0], n[:, 1], n[:, 2].astype(int)
    L = e - s
    ook = np.array([q % 12 not in scale for q in p])
    slide = np.zeros(len(n), bool)
    dbl = np.zeros(len(n), bool)
    rst = np.zeros(len(n), bool)
    o = np.argsort(s)
    for i in range(len(n)):         # short, next to a longer note 1-2 keys away that it overlaps or runs into
        if L[i] >= 0.15:
            continue
        j = np.where((np.abs(p - p[i]) >= 1) & (np.abs(p - p[i]) <= 2) & (s < e[i] + 0.1) & (e > s[i] - 0.05)
                     & (L >= 1.5 * L[i]))[0]
        slide[i] = len(j) > 0
    for k in np.unique(p):
        i = o[p[o] == k]
        if len(i) > 1:
            dbl[i[1:]] = (s[i[1:]] - e[i[:-1]] < 0.06) & (s[i[1:]] - s[i[:-1]] < 0.3)    # not a legato repeat
            rst[i[1:]] = s[i[1:]] - s[i[:-1]] < 0.15
    return L, ook, slide, dbl, rst


def voiced_at(sid, t):
    f = VOCAL / f"{sid}.npz"
    if not f.exists():
        return np.full(len(t), -1, np.int8)
    z = np.load(f)
    hop, vdb = float(z["hop"]), z["vocals_db"].astype(float)
    v = (vdb > np.percentile(vdb, 99.5) - 35) & (vdb > -60)
    return v[np.clip((t / hop).astype(int), 0, len(v) - 1)].astype(np.int8)


def score(a, songs):
    import librosa
    base = pathlib.Path(a.out)
    rows, info = [], []
    for si, s in enumerate(songs):
        f = base / "run" / f"{s['audio']}.npz"
        if not f.exists():
            print(f"{s['audio']}: no run yet", flush=True)
            continue
        z = np.load(f)
        y = np.load(AUDIO / f"{s['audio']}.npy").astype(np.float32)
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=int(SR / FPS))
        env = (env - env.mean()) / (env.std() + 1e-9)
        hums, lags = [], {}
        for aid in s["arrs"]:
            h = np.load(base / "midi" / f"{aid}.npy").astype(float)
            lag, c = lag_of(env, h)
            lags[aid] = (lag, c)
            if abs(lag) <= LAG_TOL:
                hums.append((aid, h))
        info.append(dict(song=s["audio"], url=s["url"], split=s["split"], secs=float(z["secs"]), arrs=len(s["arrs"]),
                         used=[x for x, _ in hums], lags=lags, n_raw=len(z["raw"]), n_out=len(z["out"]),
                         hum_notes={x: len(h) for x, h in hums}))
        if not hums:
            print(f"{s['audio']}: no arrangement within {LAG_TOL} s", flush=True)
            continue
        H = [h for _, h in hums]
        scale, key = song_key(H)
        raw, out = z["raw"], z["out"]
        # the clean-up's verdict on each raw note: out notes keep the raw start + key
        ko = {(round(x[0], 4), int(x[2])) for x in out}
        verdict = np.array([1 if (round(x[0], 4), int(x[2])) in ko else 0 for x in raw], np.int8)
        sets = [("raw", raw, verdict, z["pk"], z["pm"]), ("out", out, np.ones(len(out), np.int8), None, None)]
        for name, n, v, pk, pm in sets:
            if not len(n):
                continue
            L, ook, slide, dbl, rst = shape(n, scale)
            r = dict(song=np.full(len(n), si, np.int32), set=np.full(len(n), name == "out", np.int8),
                     pitch=n[:, 2].astype(np.int16), onset=n[:, 0].astype(np.float32), len=L.astype(np.float32),
                     vel=n[:, 3].astype(np.int16), ook=ook, slide=slide, dbl=dbl, rst=rst,
                     voiced=voiced_at(s["audio"], n[:, 0]), kept_cleanup=v,
                     pk=(pk if pk is not None else np.full(len(n), np.nan)).astype(np.float32),
                     pm=(pm if pm is not None else np.full(len(n), np.nan)).astype(np.float32),
                     pc_any=kept_by(n, H, TOL_PC, True), exact_any=kept_by(n, H, TOL_EXACT, False),
                     nref=np.full(len(n), len(H), np.int8), who=np.zeros(len(n), np.int8))
            if len(H) >= 2:     # the fair comparison with humans: against all but the first arrangement
                r["pc_loo"] = kept_by(n, H[1:], TOL_PC, True)
            else:
                r["pc_loo"] = np.zeros(len(n), bool)
            rows.append(r)
        if len(H) >= 2:         # control: the first arrangement's notes against the others
            h = H[0]
            L, ook, slide, dbl, rst = shape(h, scale)
            k = kept_by(h, H[1:], TOL_PC, True)
            rows.append(dict(song=np.full(len(h), si, np.int32), set=np.full(len(h), 2, np.int8),
                             pitch=h[:, 2].astype(np.int16), onset=h[:, 0].astype(np.float32), len=L.astype(np.float32),
                             vel=h[:, 3].astype(np.int16), ook=ook, slide=slide, dbl=dbl, rst=rst,
                             voiced=voiced_at(s["audio"], h[:, 0]), kept_cleanup=np.ones(len(h), np.int8),
                             pk=np.full(len(h), np.nan, np.float32), pm=np.full(len(h), np.nan, np.float32),
                             pc_any=k, exact_any=kept_by(h, H[1:], TOL_EXACT, False),
                             nref=np.full(len(h), len(H) - 1, np.int8), who=np.ones(len(h), np.int8), pc_loo=k))
        print(f"{s['audio']}: {len(H)}/{len(s['arrs'])} arrangements used, key {key[1]}{'m' if key[2] else ''}, "
              f"{len(raw)} raw / {len(out)} out", flush=True)
    R = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else {}
    np.savez(base / f"rows_{a.shard}.npz", **R)
    (base / f"info_{a.shard}.json").write_text(json.dumps(dict(songs=info, index=[s["audio"] for s in songs])))


def report(a):
    base = pathlib.Path(a.out)
    parts = sorted(base.glob("rows_*.npz"))
    R = {}
    offs = []
    for p in parts:
        z = np.load(p)
        for k in z.files:
            R.setdefault(k, []).append(z[k])
    R = {k: np.concatenate(v) for k, v in R.items()}
    infos = [json.loads(p.read_text()) for p in sorted(base.glob("info_*.json"))]
    songs = [s for i in infos for s in i["songs"]]
    print(f"{len(songs)} songs scored, {sum(len(s['used']) for s in songs)} of {sum(s['arrs'] for s in songs)} "
          f"arrangements within {LAG_TOL} s of the audio")
    secs = sum(s["secs"] for s in songs if s["used"])
    hum = [n / s["secs"] for s in songs for n in s["hum_notes"].values()]
    print(f"notes/s: ours raw {sum(s['n_raw'] for s in songs if s['used']) / secs:.1f}, "
          f"after clean-up {sum(s['n_out'] for s in songs if s['used']) / secs:.1f}, "
          f"humans median {np.median(hum):.1f} (p10 {np.percentile(hum, 10):.1f} / p90 {np.percentile(hum, 90):.1f})")

    def table(title, m0, label):
        print(f"\n{title}")
        print(f"{'':<44}{'notes':>9}{'share':>8}{'never kept':>12}")
        tot = m0.sum()
        groups = [("all", np.ones_like(m0)), ("out of key", R["ook"]), ("in key", ~R["ook"]),
                  ("slide-shaped", R["slide"]), ("slide-shaped + out of key", R["slide"] & R["ook"]),
                  ("double press (< 60 ms gap, < 300 ms apart)", R["dbl"]), ("re-strike (< 150 ms)", R["rst"]),
                  ("short < 100 ms", R["len"] < 0.1), ("100-250 ms", (R["len"] >= 0.1) & (R["len"] < 0.25)),
                  ("250 ms+", R["len"] >= 0.25),
                  ("below C3", R["pitch"] < 48), ("C3-B3", (R["pitch"] >= 48) & (R["pitch"] < 60)),
                  ("C4-B4", (R["pitch"] >= 60) & (R["pitch"] < 72)), ("C5+", R["pitch"] >= 72),
                  ("voice sounding", R["voiced"] == 1), ("voice silent", R["voiced"] == 0)]
        for nm, g in groups:
            m = m0 & g.astype(bool)
            if m.sum():
                print(f"{nm:<44}{int(m.sum()):>9}{m.sum() / tot:>8.1%}{1 - label[m].mean():>12.1%}")

    raw, out, hu = R["set"] == 0, R["set"] == 1, R["set"] == 2
    table("OUR notes after clean-up (hosted), vs ALL arrangements of the song", out, R["pc_any"])
    table("OUR raw 0.8 notes (before clean-up), vs ALL arrangements", raw, R["pc_any"])
    multi = R["nref"] >= 2
    table("FAIR control, songs with 2+ arrangements: OUR notes after clean-up vs all but one arrangement", out & multi,
          R["pc_loo"])
    table("... the left-out HUMAN arrangement vs the same others", hu, R["pc_loo"])
    print("\nclean-up's verdict on raw notes vs the humans (all arrangements):")
    for nm, m in (("kept / merged-into by clean-up", raw & (R["kept_cleanup"] == 1)),
                  ("dropped or merged away", raw & (R["kept_cleanup"] == 0))):
        print(f"  {nm:<44}{int(m.sum()):>9} notes, never kept {1 - R['pc_any'][m].mean():.1%}")
    print("\nnever-kept rate by how many arrangements the song has (ours after clean-up):")
    for k in (1, 2, 3, 4, 6, 9):
        m = out & (R["nref"] >= k) & (R["nref"] < {1: 2, 2: 3, 3: 4, 4: 6, 6: 9, 9: 99}[k])
        if m.sum():
            print(f"  {k}+ arrangements: {int(m.sum()):>8} notes, never kept {1 - R['pc_any'][m].mean():.1%}, "
                  f"exact key never kept {1 - R['exact_any'][m].mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--cached-only", action="store_true"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--songs"); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.expanduser("~/arrangement_transformer/checkpoints/base20m/best.pt"))
    ap.add_argument("--cleanup", default=os.path.expanduser("~/arrangement_transformer/checkpoints/cleanup3/best.pt"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.report:
        return report(a)
    songs = json.loads(pathlib.Path(a.songs).read_text())["songs"][a.shard::a.nshards]
    if a.beats:
        beats(a, songs)
    if a.run:
        run(a, songs)
    if a.score:
        score(a, songs)


if __name__ == "__main__":
    main()
