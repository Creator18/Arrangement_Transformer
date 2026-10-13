"""Does a start/end tempo match sync the arrangements with their songs?

The user's test: pad both with silence, find where each actually STARTS and
ENDS, and if the arrangement's span differs from the song's, stretch the
arrangement (one global tempo change + offset) so the two line up end to end.
Any misalignment left after that may be the arranger's own choice, and the
model can learn it.

Per song:
  song span        first / last moment the audio is above -40 dB of its peak
                   (RMS, 23 ms), after padding 2 s of silence both sides
  arrangement span first note onset / last note offset
  stretch          map [arr_start, arr_end] -> [song_start, song_end] linearly

Compared on every song, by the share of arrangement onsets within 25 / 50 ms
of an attack in the song:
  original      as shipped
  endpoint      the user's start/end stretch
  realigned     eval/realign_m2m.py (local shifts + snap), for reference
  both          endpoint stretch, then the realignment's local steps

    python eval/endpoint_sync.py --m2m /scratch/.../m2m --shard i/n --out <dir>
    python eval/endpoint_sync.py --m2m /scratch/.../m2m --out <dir> --report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from realign_m2m import attacks, realign, on_attack          # noqa: E402

PAD = 2.0


def song_span(wav):
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    y = np.concatenate([np.zeros(int(PAD * sr)), y, np.zeros(int(PAD * sr))])
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    act = np.flatnonzero(rms > rms.max() * 10 ** (-40 / 20))
    t = librosa.frames_to_time(act, sr=sr, hop_length=512) - PAD
    return float(t[0]), float(t[-1]), len(y) / sr - 2 * PAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.report:
        rows = [json.loads(l) for f in sorted(out.glob("sync_*.jsonl")) for l in open(f)]
        k = ["original", "endpoint", "realigned", "both"]
        print(f"{len(rows)} songs\n")
        print(f"start offset (arrangement - song): median {np.median([r['d_start'] for r in rows]):+.2f} s, "
              f"|.| > 0.5 s on {np.mean([abs(r['d_start']) > .5 for r in rows]):.0%}")
        print(f"end offset   (arrangement - song): median {np.median([r['d_end'] for r in rows]):+.2f} s, "
              f"|.| > 0.5 s on {np.mean([abs(r['d_end']) > .5 for r in rows]):.0%}")
        print(f"span ratio (song / arrangement):   median {np.median([r['ratio'] for r in rows]):.4f}, "
              f"outside 0.99-1.01 on {np.mean([abs(r['ratio'] - 1) > .01 for r in rows]):.0%}\n")
        print(f"{'version':<11}{'<=25 ms':>9}{'<=50 ms':>9}")
        for v in k:
            print(f"{v:<11}{np.mean([r[v + '_25'] for r in rows]):>9.1%}{np.mean([r[v + '_50'] for r in rows]):>9.1%}")
        better = np.mean([r["endpoint_25"] > r["original_25"] for r in rows])
        print(f"\nendpoint stretch improves on-attack share for {better:.0%} of songs")
        return
    sp = json.loads((pathlib.Path(args.m2m) / "m2m_splits.json").read_text())
    ids = sorted(sp["train"] + sp["validation"] + sp["test"])
    si, sn = (int(v) for v in args.shard.split("/"))
    with open(out / f"sync_{si:02d}.jsonl", "w") as fo:
        for sid in ids[si::sn]:
            n = np.load(pathlib.Path(args.m2m) / "midi_numpy" / f"{sid}.npy").astype(np.float64)
            if len(n) < 20:
                continue
            wav = pathlib.Path(args.m2m) / "audio" / f"{sid}.wav"
            s0, s1, dur = song_span(wav)
            a0, a1 = float(n[:, 0].min()), float(n[:, 1].max())
            ratio = (s1 - s0) / max(1e-3, a1 - a0)
            e = n.copy(); e[:, :2] = (e[:, :2] - a0) * ratio + s0
            env, tp = attacks(wav)
            r_new, _ = realign(n, env, tp)
            b_new, _ = realign(e, env, tp)
            rec = dict(id=sid, d_start=a0 - s0, d_end=a1 - s1, ratio=ratio)
            for name, x in (("original", n), ("endpoint", e), ("realigned", r_new), ("both", b_new)):
                d = on_attack(x, tp)
                rec[name + "_25"] = float(np.mean(d <= .025)); rec[name + "_50"] = float(np.mean(d <= .05))
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"  {sid}: start {a0 - s0:+.2f}s end {a1 - s1:+.2f}s ratio {ratio:.4f} | on-attack 25ms "
                  f"orig {rec['original_25']:.0%} endpoint {rec['endpoint_25']:.0%} both {rec['both_25']:.0%}", flush=True)


if __name__ == "__main__":
    main()
