"""Data for clean-up v4's "no human keeps this" output (the user's idea, 2026-10-29): our raw notes on real songs,
labelled by ALL the song's human arrangements (eval/never_kept.py measured it: 925 Music2MIDI songs).

Per song, the clean-up's inputs exactly as the app builds them (app/services/cleanup.py: features on the whole song,
madmom on the whole song, then 20 s windows every 10 s), and per note:
    kept   1 if any arrangement that passed the lag gate (never_kept's info_*.json "used") has the note's pitch
           class within 150 ms (any octave), else 0
    nref   how many arrangements that is (labels are cleaner with more: never kept 37% with 1, 9% with 9+)

One npz per song in --out:
    notes  (N, 4) onset, offset, pitch, velocity (the base at 0.8, the app's decoding, before the clean-up)
    x      (N, NFEAT) float16 the clean-up v2/v3 features;  kept (N,) uint8;  nref ()
    win    (W,) window starts (s);  zb_*  the madmom record

    python eval/human_data.py --songs songs.json --nk /scratch/basu.ang/never_kept --out /scratch/basu.ang/human_data \
        --shard k --nshards n
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
SR = 24000
AUDIO = pathlib.Path("/scratch/basu.ang/solo_det/m2m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True); ap.add_argument("--nk", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.expanduser("~/arrangement_transformer/checkpoints/base20m/best.pt"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    import soundfile as sf
    from app.ml.spectral import cleanup_train as C1, cleanup2_train as C2
    from app.services import cleanup, slides, spectral_transcription as ST
    import clip_beats
    sys.path.insert(0, a.nk)
    import frozen as NK                                         # never_kept.py as it ran
    cleanup.CKPT, slides.ON = "", False                         # the base's raw notes: what the clean-up sees
    used = {}
    for p in pathlib.Path(a.nk).glob("info_*.json"):
        for s in json.loads(p.read_text())["songs"]:
            used[s["song"]] = s["used"]
    songs = json.loads(pathlib.Path(a.songs).read_text())["songs"][a.shard::a.nshards]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"_tmp{a.shard}.wav"
    real, cache = ST.rolls, {}
    ST.rolls = lambda path, cond=None, with_offset=False: cache.setdefault("r", real(path, cond=cond, with_offset=with_offset))
    for s in songs:
        dst = out / f"{s['audio']}.npz"
        if dst.exists() or not used.get(s["audio"]):
            continue
        y = np.load(AUDIO / f"{s['audio']}.npy").astype(np.float32)
        sf.write(tmp, y, SR)
        cache.clear()
        notes = ST.transcribe(tmp, onset_thr=0.8, ckpt=pathlib.Path(a.ckpt), frame_thr=0.5)
        if not notes:
            continue
        on, fr = cache["r"][0], cache["r"][1]
        zb = clip_beats.track(y)
        n = np.array([[g["onset"], g["offset"], g["pitch"], g["velocity"], 0.0] for g in notes], np.float32)
        T = on.shape[1]
        k = n[:, 2].astype(int) - 21
        f = np.clip(np.round(n[:, 0] * C1.FPS).astype(int), 0, T - 1)
        n[:, 4] = [on[kk, max(0, ff - 2):ff + 3].max() for kk, ff in zip(k, f)]
        x = np.concatenate([C1.note_features(n, on, fr), C2.key_features(n), C2.beat_features(n[:, 0], zb)], 1)
        H = [np.load(pathlib.Path(a.nk) / "midi" / f"{aid}.npy").astype(float) for aid in used[s["audio"]]]
        kept = NK.kept_by(n[:, :4].astype(float), H, NK.TOL_PC, True)
        end = float(n[:, 0].max())
        win = np.arange(0, max(end - 10, 0) + 1e-6, 10.0) if end > 20 else np.zeros(1)
        t = out / f"{s['audio']}.tmp.npz"
        np.savez_compressed(t, notes=n[:, :4], x=x.astype(np.float16), kept=kept.astype(np.uint8), nref=len(H),
                            win=win.astype(np.float32), **{f"zb_{q}": v for q, v in zb.items()})
        t.rename(dst)
        print(f"{s['audio']}: {len(n)} notes, {len(H)} arrangements, never kept {1 - kept.mean():.1%}, "
              f"{len(win)} windows", flush=True)
    tmp.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
