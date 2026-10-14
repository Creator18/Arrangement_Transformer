"""Stage-2 data: both hums for every Music2MIDI window — the SONG's hum and its PIANO arrangement's hum (CPU).

Reads the paired windows of eval/make_m2m_pairs.py (y = song audio, yp = the
human arrangement rendered on a piano, on/fr = the arrangement's labels) and
makes the same hum as eval/make_piano_hums.py of each side:

    <out>/song/<split>/<id>_<w>.npz    y = song hum,  on, fr, difficulty
    <out>/piano/<split>/<id>_<w>.npz   y = piano hum, on, fr, difficulty

Same format as the clips the spectral code reads, so stage 1 can be checked
on the piano side directly.  Splits are the folders the pairs were written
in; stage 2 re-sorts by song (eval/m2m_bysong.py units).

    python eval/make_pair_hums.py --pairs /scratch/.../m2m_pairs --out <dir> --shard i/n
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="validation,test,train")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from basic_pitch import inference, ICASSP_2022_MODEL_PATH
    from app.ml.spectral import frontend as S
    from make_piano_hums import hum_audio
    model = inference.Model(ICASSP_2022_MODEL_PATH)
    si, sn = (int(v) for v in a.shard.split("/"))
    tmp = pathlib.Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / f"phum_{os.getpid()}.wav"
    done = 0
    for split in a.splits.split(","):
        paths = sorted(p for p in (pathlib.Path(a.pairs) / split).glob("*.npz")
                       if not p.name.endswith(".tmp.npz") and ".ears." not in p.name)
        for side in ("song", "piano"):
            (pathlib.Path(a.out) / side / split).mkdir(parents=True, exist_ok=True)
        for p in paths[si::sn][: a.limit or None]:
            z = None
            for side, key in (("song", "y"), ("piano", "yp")):
                dst = pathlib.Path(a.out) / side / split / p.name
                if dst.exists():
                    continue
                z = z if z is not None else np.load(p)
                y = z[key].astype(np.float32) / 32767
                y22 = resample_poly(y, 147, 160).astype(np.float32)
                sf.write(tmp, y22, 22050)
                sal = inference.run_inference(tmp, model)["contour"]
                hum = hum_audio(y22, sal, S.SR, len(y))
                t = dst.with_name(dst.stem + ".tmp.npz")
                np.savez_compressed(t, y=(np.clip(hum, -1, 1) * 32767).astype(np.int16), on=z["on"], fr=z["fr"],
                                    difficulty=z["difficulty"], piano=z["piano"] if side == "piano" else "song")
                t.replace(dst); done += 1
            if done and done % 100 == 0:
                print(f"  {done} hums", flush=True)
    tmp.unlink(missing_ok=True)
    print(f"{done} hums written", flush=True)


if __name__ == "__main__":
    main()
