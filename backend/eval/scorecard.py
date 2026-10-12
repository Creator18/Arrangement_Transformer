"""Arrangement scorecard: is a piano arrangement musically acceptable for its level?

Note F1 against one human cover undersells any arranger — two good arrangers
write different notes.  The scorecard instead measures the PRINCIPLES a good
arrangement follows, each against the song's own audio, and calibrates what
"acceptable" means from the human arrangements themselves, per difficulty: a
metric's acceptable range for a level is where the middle 80% of human
arrangers at that level land.

Metrics (arrangement notes + song audio; onsets and pitches only):
  difficulty   notes/s, chord size, hand span, jumps   (difficulty_classifier)
  harmonic     mean song-chroma energy at each note's pitch class, at its onset
               — are the notes the song's notes?
  bass         same, lowest left-hand note per event vs LOW-register chroma
  melody       same, highest right-hand note per event vs UPPER-register chroma
  groove       correlation of the arrangement's onset rate with the song's
               onset envelope (50 ms bins, 200 ms smoothing)
  mud          share of left-hand events with two notes below C3 within a 3rd

Chroma is max-normalised per frame, so 1.0 = the strongest pitch class sounding.

    python eval/scorecard.py --shard 3/16 --out /scratch/.../scorecard   # one shard
    python eval/scorecard.py --calibrate --out /scratch/.../scorecard    # merge + ranges
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from difficulty_classifier import features as diff_features, NAMES as DIFF_NAMES   # noqa: E402

SR, HOP = 16000, 512                       # 32 ms frames
LEVELS = ["beginner", "intermediate", "advanced"]
KEEP_DIFF = ["notes_per_s", "poly_mean", "rh_span_p90", "lh_span_p90", "jump_speed_p90"]


def song_analysis(wav: pathlib.Path):
    """Per-frame chroma (full, low register, upper register) and onset envelope."""
    import librosa
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                           n_bins=84, bins_per_octave=12))          # C1..B7

    def chroma(lo, hi):                    # octave slice -> (12, T), max-normalised
        c = C[(lo - 1) * 12:(hi - 1) * 12].reshape(hi - lo, 12, -1).sum(0)
        return c / (c.max(0, keepdims=True) + 1e-9)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    return {"full": chroma(2, 8), "low": chroma(1, 4), "high": chroma(4, 8), "env": env}


def score_arrangement(notes: np.ndarray, song) -> dict:
    """notes: [onset, offset, pitch, velocity] rows."""
    on = notes[:, 0]; p = notes[:, 2].astype(int)
    T = song["full"].shape[1]
    fr = lambda t: np.clip((t * SR / HOP).astype(int), 0, T - 1)
    out = {}
    out["harmonic"] = float(song["full"][p % 12, fr(on)].mean())
    ev = np.concatenate([[0], np.cumsum(np.diff(on) > 0.03)])
    lows, highs, mud, n_lh = [], [], 0, 0
    for e in np.unique(ev):
        ps, t = p[ev == e], on[ev == e].min()
        lh, rh = ps[ps < 60], ps[ps >= 60]
        if len(lh):
            lows.append((lh.min(), t)); n_lh += 1
            deep = np.sort(lh[lh < 48])
            mud += int(len(deep) > 1 and np.diff(deep).min() <= 4)
        if len(rh):
            highs.append((rh.max(), t))
    q = lambda arr, key: float(song[key][[a % 12 for a, _ in arr], fr(np.array([t for _, t in arr]))].mean()) \
        if arr else float("nan")
    out["bass"], out["melody"] = q(lows, "low"), q(highs, "high")
    out["mud"] = mud / max(1, n_lh)
    # groove: onset rate vs song onset envelope, both on 50 ms bins smoothed to 200 ms
    dur = T * HOP / SR
    b = np.arange(0, dur + 0.05, 0.05)
    a_rate = np.histogram(on, b)[0].astype(float)
    env_t = np.arange(len(song["env"])) * HOP / SR
    s_rate = np.interp(b[:-1], env_t, song["env"])
    k = np.ones(4) / 4
    a_rate, s_rate = np.convolve(a_rate, k, "same"), np.convolve(s_rate, k, "same")
    out["groove"] = float(np.corrcoef(a_rate, s_rate)[0, 1]) if a_rate.std() > 0 else 0.0
    f = diff_features(notes)
    for n in KEEP_DIFF:
        out[n] = float(f[DIFF_NAMES.index(n)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--m2m", default="", help="dir with audio/, midi_numpy/, m2m_splits.json")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        rows = [json.loads(l) for f in sorted(out.glob("human_*.jsonl")) for l in open(f)]
        tr = [r for r in rows if r["split"] == "train"]
        keys = [k for k in tr[0] if k not in ("id", "split", "difficulty", "mismatch")]
        ranges = {}
        print(f"{len(rows)} songs scored ({len(tr)} train used for calibration)\n")
        print(f"{'metric':<16}" + "".join(f"{l:>26}" for l in LEVELS))
        for k in keys:
            ranges[k] = {}
            cells = []
            for l in LEVELS:
                v = np.array([r[k] for r in tr if r["difficulty"] == l and np.isfinite(r[k])])
                lo, med, hi = np.percentile(v, [10, 50, 90]) if len(v) else (np.nan,) * 3
                ranges[k][l] = {"p10": lo, "p50": med, "p90": hi}
                cells.append(f"{lo:7.2f} {med:7.2f} {hi:7.2f}")
            print(f"{k:<16}" + "".join(f"{c:>26}" for c in cells))
        print("\n(each cell: p10  median  p90 of human arrangements at that level)")
        print("\nreal pairing vs chance (another song's arrangement on this audio), train:")
        for k in ("harmonic", "melody", "bass", "groove"):
            a = np.nanmedian([r[k] for r in tr]); c = np.nanmedian([r["mismatch"][k] for r in tr])
            win = np.nanmean([r[k] > r["mismatch"][k] for r in tr])
            print(f"   {k:<10} real {a:.3f}   chance {c:.3f}   real > chance on {win:.1%} of songs")
        (out / "ranges.json").write_text(json.dumps(ranges, indent=1))
        print(f"wrote {out / 'ranges.json'}")
        return

    m2m = pathlib.Path(args.m2m) if args.m2m else ROOT.parent / "data" / "music2midi"
    audio = m2m / "audio"
    npy = m2m / "midi_numpy" if (m2m / "midi_numpy").is_dir() else m2m / "music2midi-dataset" / "midi_numpy"
    sp = json.loads((m2m / "m2m_splits.json").read_text())
    ids = [(s, k) for k in ("train", "validation", "test") for s in sp[k]]
    all_ids = [s for s, _ in ids]
    si, sn = (int(v) for v in args.shard.split("/"))
    ids = ids[si::sn][: args.limit or None]
    dst = out / f"human_{si:02d}.jsonl"
    done = {json.loads(l)["id"] for l in open(dst)} if dst.exists() else set()
    with open(dst, "a") as fo:
        for sid, split in ids:
            if sid in done:
                continue
            song = song_analysis(audio / f"{sid}.wav")
            np.savez_compressed(out / f"song_{sid}.npz", **song)     # reused to score model outputs
            r = score_arrangement(np.load(npy / f"{sid}.npy"), song)
            r.update(id=sid, split=split, difficulty=sp["difficulty"][sid])
            # chance floor: ANOTHER song's arrangement scored against this audio,
            # so every metric shows how far a real pairing sits above a wrong one
            other = all_ids[(all_ids.index(sid) + 1) % len(all_ids)]
            r["mismatch"] = score_arrangement(np.load(npy / f"{other}.npy"), song)
            fo.write(json.dumps(r) + "\n"); fo.flush()
            print(f"  {sid} {split} {r['difficulty']}: harmonic {r['harmonic']:.2f} "
                  f"melody {r['melody']:.2f} bass {r['bass']:.2f} groove {r['groove']:.2f}", flush=True)


if __name__ == "__main__":
    main()
