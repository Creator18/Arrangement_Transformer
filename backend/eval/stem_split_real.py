"""The user's stem split vs the hosted chain on the 106 held-out Music2MIDI songs (2026-10-27).

    mix 0.9            the base at 0.9 on the mix (the old app)
    mix 0.8 + v2       the base at 0.8 + clean-up v2 keep 0.4 + quantizing (hosted before the stem split)
    stems              Demucs voice at 0.8 + clean-up keep 0.5, the rest at 0.9, joined, quantized (app/services/stem_split.py)
    stems, keep 0.4    the same with the hosted keep bar on the voice

Scored as eval/thr_review.py: strict F1 (exact key, 50 ms), pitch-class F1 (150 ms), notes/s, the sung line (the human
arrangement's top line where Demucs' vocal sounds, pitch class within 150 ms), plus re-strikes: output notes that strike
a key again within 150 ms of its last strike (the double presses the user hears; no human labels needed).

    python eval/stem_split_real.py --ckpt <base> --cleanup <cleanup2 best.pt> --out <json> --shard k --nshards n
    python eval/stem_split_real.py --merge <jsons>
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
SR = 24000
CONDS = ("mix 0.9", "mix 0.8 + v2", "stems", "stems, keep 0.4")
KEYS = ("s_hit", "s_hp", "pc_hit", "pc_hp", "true", "pred", "secs", "sung_hit", "sung", "restrike")


def restrikes(notes):
    last, n = {}, 0
    for g in sorted(notes, key=lambda g: g["onset"]):
        t = last.get(g["pitch"])
        n += t is not None and g["onset"] - t < 0.15
        last[g["pitch"]] = g["onset"]
    return n


def run(a):
    os.environ["ARRANGEMENT_CLEANUP"] = a.cleanup
    from app.services import cleanup, stem_split
    from app.services import spectral_transcription as ST
    from rhythm_groups import top_voice
    from thr_review import match
    ck = pathlib.Path(a.ckpt)
    m2m = pathlib.Path("/scratch/basu.ang/m2m")
    U = json.loads((m2m / "m2m_units.json").read_text())
    old = set(json.loads((m2m / "m2m_splits.json").read_text())["train"])
    test = [i for i in U["splits"]["test"] if i not in old][a.shard::a.nshards]
    R = {c: dict.fromkeys(KEYS, 0.0) for c in CONDS}
    tmp = pathlib.Path(a.out).with_suffix(".tmp.wav")
    for sid in test:
        y = np.load(f"/scratch/basu.ang/solo_det/m2m/{sid}.npy").astype(np.float32)
        sf.write(tmp, y, SR)
        out = {}
        cleanup.CKPT = ""
        stem_split.ON = False
        out["mix 0.9"] = ST.transcribe(tmp, onset_thr=0.9, ckpt=ck, frame_thr=0.5)
        cleanup.CKPT, cleanup.QUANT, cleanup.KEEP = a.cleanup, True, 0.4
        out["mix 0.8 + v2"] = ST.transcribe(tmp, onset_thr=0.8, ckpt=ck, frame_thr=0.5)
        stem_split.ON = True
        for name, keep in (("stems", 0.5), ("stems, keep 0.4", 0.4)):
            stem_split.VOCAL_KEEP = keep
            out[name] = ST.transcribe(tmp, onset_thr=0.8, ckpt=ck, frame_thr=0.5)
        hum = np.load(m2m / "midi_numpy" / f"{sid}.npy")
        hum = hum[np.lexsort((-hum[:, 2], hum[:, 0]))]
        gt = [(float(t), int(p)) for t, _, p, *_ in hum]
        sung = []
        zf = pathlib.Path(f"/scratch/basu.ang/solo_det/m2m_sim/{sid}.npz")
        if zf.exists():
            z = np.load(zf)
            hop, vdb = float(z["hop"]), z["vocals_db"].astype(float)
            voiced = (vdb > np.percentile(vdb, 99.5) - 35) & (vdb > -60)
            for t, _, p in hum[top_voice(hum)][:, :3]:
                i0, i1 = max(0, int((t - 0.1) / hop)), int((t + 0.1) / hop) + 1
                if i0 < len(voiced) and voiced[i0:i1].any():
                    sung.append((float(t), int(p)))
        for c in CONDS:
            pred = [(g["onset"], g["pitch"]) for g in out[c]]
            h, hp = match(pred, gt, 0.05)
            hpc, hppc = match(pred, gt, 0.15, pc=True)
            r = R[c]
            r["s_hit"] += h; r["s_hp"] += hp; r["pc_hit"] += hpc; r["pc_hp"] += hppc; r["true"] += len(gt)
            r["pred"] += len(pred); r["secs"] += len(y) / SR; r["restrike"] += restrikes(out[c])
            r["sung_hit"] += match(pred, sung, 0.15, pc=True)[0]; r["sung"] += len(sung)
        print(sid, "  ".join(f"{c}: {len(out[c])}" for c in CONDS), flush=True)
    tmp.unlink(missing_ok=True)
    return dict(n=len(test), R=R)


def prf(h, n, hp, m):
    r, p = h / max(1, n), hp / max(1, m)
    return r, p, 2 * r * p / max(1e-9, r + p)


def report(D):
    print(f"\n=== {D['n']} real songs (human: 8.0 notes/s)")
    print(f"{'':<17}{'strict F1':>10}{'pc R / P / F1':>22}{'notes/s':>9}{'sung line':>11}{'re-strikes/1000':>17}")
    for c in CONDS:
        r = D["R"][c]
        s = prf(r["s_hit"], r["true"], r["s_hp"], r["pred"])
        p = prf(r["pc_hit"], r["true"], r["pc_hp"], r["pred"])
        print(f"{c:<17}{s[2]:>10.3f}{p[0]:>10.3f} {p[1]:.3f} {p[2]:.3f}{r['pred'] / max(1e-9, r['secs']):>9.1f}"
              f"{r['sung_hit'] / max(1, r['sung']):>11.3f}{1000 * r['restrike'] / max(1, r['pred']):>17.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--cleanup")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", nargs="*")
    a = ap.parse_args()
    if a.merge:
        ds = [json.loads(pathlib.Path(p).read_text()) for p in a.merge]
        report(dict(n=sum(d["n"] for d in ds), R={c: {k: sum(d["R"][c][k] for d in ds) for k in KEYS} for c in CONDS}))
        return
    D = run(a)
    pathlib.Path(a.out).write_text(json.dumps(D))
    report(D)


if __name__ == "__main__":
    main()
