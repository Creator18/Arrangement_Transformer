"""Stage A1 of the realistic-singer batch (CPU): for every POP909 song, V sung versions of its melody
as RVC inputs -- content audio (TTS phonemes, eval/sing_tts.py) + the exact pitch track -- and the
plan for the clone (which VocalSet singer).  Stage A2 (eval/rvc_sing.py, GPU) turns each into a
real singer's voice; stage B (render_arrangements.py --voices) mixes them into the renders.

Each version is one singer for the whole song (a voice never changes inside a phrase), with its
own articulation per phrase (both ways), slides, and the approved few-cent drift.

    python eval/sing_batch.py --pop909 <POP909 dir> --out /scratch/.../voices --versions 3 --shard i/n
writes <out>/pop<id>_v<k>.content.wav, .f0.npy, .json  and  <out>/jobs_<shard>.json (for rvc_sing)
"""
import argparse, json, pathlib, sys

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from render_arrangements import load_pop909, split_of_pop      # noqa: E402
from sing_tts import Singer, SR                                 # noqa: E402

FEMALE, MALE = list(range(0, 9)), list(range(9, 20))           # VocalSet speaker ids in the voice model
TTS_F, TTS_M = ["af_heart", "af_bella", "af_nicole", "af_sarah"], ["am_michael", "am_adam", "am_fenrir"]

ap = argparse.ArgumentParser()
ap.add_argument("--pop909", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--versions", type=int, default=3); ap.add_argument("--shard", default="0/1")
ap.add_argument("--slide-p", type=float, default=0.6)
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
si, sn = (int(x) for x in a.shard.split("/"))
songs = [d for d in sorted(pathlib.Path(a.pop909).iterdir()) if d.is_dir() and d.name.isdigit()]
singers, jobs = {}, []
for d in songs[si::sn]:
    if split_of_pop(int(d.name)) == "test":
        continue
    notes, roles, _b, _f = load_pop909(d)
    mel = notes[roles == 0]
    if len(mel) < 8:
        continue
    high = float(np.median(mel[:, 2])) >= 60                    # register decides the voice family
    for k in range(a.versions):
      try:
        stem = out / f"pop{d.name}_v{k}"
        if (stem.with_suffix(".json")).exists():
            jobs.append(json.loads(stem.with_suffix(".json").read_text())["job"]); continue
        rng = np.random.default_rng(int(d.name) * 10 + k)
        voice = str(rng.choice(TTS_F if high else TTS_M))
        s = singers.setdefault(voice, Singer(voice=voice))
        y, slid, f0 = s.sing(mel, rng, slide_p=a.slide_p, art_both=True,
                             content_hz=210.0 if voice.startswith("af") else 120.0)
        sf.write(f"{stem}.content.wav", y.astype(np.float32), SR); np.save(f"{stem}.f0.npy", f0.astype(np.float32))
        spk = int(rng.choice(FEMALE if high else MALE))
        job = dict(content=f"{stem}.content.wav", f0=f"{stem}.f0.npy", out=f"{stem}.voice.wav", speaker=spk)
        stem.with_suffix(".json").write_text(json.dumps(dict(
            job=job, tts_voice=voice, speaker=spk, slid=sorted(int(x) for x in slid),
            length_class=s.length_class.astype(int).tolist(), n_melody=int(len(mel)))))
        jobs.append(job)
      except Exception as e:                                       # one song never stops the rest
        print(f"  pop{d.name} v{k} FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"  pop{d.name}: {a.versions} versions", flush=True)
(out / f"jobs_{si}.json").write_text(json.dumps(jobs))
print(len(jobs), "voice jobs", flush=True)
