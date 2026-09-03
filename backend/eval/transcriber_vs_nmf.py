"""Head to head: the trained transcriber against the NMF, same audio, same score.

There has never been a number comparing them.  The case for replacing the NMF
has rested on an argument — that a dictionary of 88 piano templates must
explain every scrap of energy it is handed and therefore cannot decline to
call something a note — plus a week of ear tests in which every attempt to
remove the resulting surplus made the arrangement worse.  That is suggestive.
It is not a measurement.

Slakh supplies the ground truth neither approach has had: per-stem MIDI
aligned to the mix, so "which notes are actually there" is known rather than
inferred.  Both systems get the same audio and are scored the same way, with
mir_eval note-level precision/recall/F1 at a 50 ms onset tolerance.

Four rows, because two of them exist to stop the other two flattering
themselves:

    random        notes at the ground truth's own density and pitch
                  distribution, placed at random times.  Everything must beat
                  this or the metric is measuring density, not accuracy.  Three
                  metrics this week looked decisive until a chance baseline was
                  added; this one has it from the start.
    NMF (mix)     the projection on the same full mix the model sees.  The
                  honest like-for-like.
    NMF (drumless) the projection on the non-drum stems summed — its best
                  case, and what the production pipeline actually feeds it
                  after Demucs.  Included so the NMF is not handicapped by an
                  input it never gets in practice.
    transcriber   the trained model.

Run:  python eval/transcriber_vs_nmf.py --tracks 3
"""
from __future__ import annotations

import argparse, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann


def ground_truth(track: pathlib.Path):
    """Notes from the per-stem MIDI, drums excluded — the training target."""
    import pretty_midi
    from app.ml.transcriber.data import pitched_stems
    out = []
    for sid in pitched_stems(track):
        try:
            pm = pretty_midi.PrettyMIDI(str(track / "MIDI" / f"{sid}.mid"))
        except Exception:                                       # noqa: BLE001
            continue
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                if 21 <= n.pitch <= 108:
                    out.append({"pitch": int(n.pitch), "onset": float(n.start),
                                "offset": float(n.end), "velocity": int(n.velocity)})
    out.sort(key=lambda n: (n["onset"], n["pitch"]))
    return out


def random_like(ref, seed=0):
    """Same count, same pitch histogram, same duration spread — random times."""
    if not ref:
        return []
    rng = np.random.default_rng(seed)
    span = max(n["offset"] for n in ref)
    pitches = rng.choice([n["pitch"] for n in ref], size=len(ref))
    durs = rng.choice([n["offset"] - n["onset"] for n in ref], size=len(ref))
    ons = rng.uniform(0, max(span - 0.1, 0.1), size=len(ref))
    return sorted(({"pitch": int(p), "onset": float(o), "offset": float(o + d),
                    "velocity": 80} for p, o, d in zip(pitches, ons, durs)),
                  key=lambda n: n["onset"])


def nmf_notes(wav_path: pathlib.Path):
    import librosa
    from app.services.piano_projection import (
        SR, stft_mag, get_dictionary, nmf_h, extract_notes, velocities, LOW)
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    pk = float(np.abs(y).max())
    if pk < 1e-6:
        return []
    H = nmf_h(stft_mag((y / pk).astype(np.float32)), get_dictionary())
    est = sorted(extract_notes(H), key=lambda n: (n[0], n[2]))
    vel = velocities(H, est)
    return [{"pitch": int(p) , "onset": float(a), "offset": float(b),
             "velocity": int(v)} for (a, b, p), v in zip(est, vel)]


def drumless_mix(track: pathlib.Path, out: pathlib.Path):
    """Sum the non-drum stems — the NMF's best case, and its production input."""
    import soundfile as sf, librosa, yaml
    from app.services.piano_projection import SR
    meta = yaml.safe_load((track / "metadata.yaml").read_text())
    keep = [sid for sid, i in (meta.get("stems") or {}).items()
            if not i.get("is_drum") and (track / "stems" / f"{sid}.wav").exists()]
    sigs = []
    for sid in keep:
        y, _ = librosa.load(str(track / "stems" / f"{sid}.wav"), sr=SR, mono=True)
        sigs.append(y)
    if not sigs:
        return None
    n = max(len(s) for s in sigs)
    mix = np.zeros(n, dtype=np.float32)
    for s in sigs:
        mix[: len(s)] += s
    pk = float(np.abs(mix).max())
    sf.write(str(out), (mix / pk * 0.89 if pk > 0 else mix), SR)
    return out


def model_notes(wav_path: pathlib.Path, ckpt: pathlib.Path):
    """Run the transcriber over a whole track in segments, then decode once."""
    import torch, soundfile as sf, librosa
    from app.ml.transcriber import config as C
    from app.ml.transcriber.data import MelFront
    from app.ml.transcriber.model import Transcriber
    from app.ml.transcriber.decode import decode_notes

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Transcriber().to(dev).eval()
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"])
    front = MelFront().to(dev)

    y, _ = librosa.load(str(wav_path), sr=C.SR, mono=True)
    seg = int(C.SEGMENT_S * C.SR)
    rolls = {"on": [], "fr": [], "vl": []}
    with torch.no_grad():
        for a in range(0, len(y), seg):
            chunk = y[a:a + seg]
            if len(chunk) < seg:
                chunk = np.pad(chunk, (0, seg - len(chunk)))
            mel = front(torch.from_numpy(chunk).float().unsqueeze(0).to(dev))
            po, pf, pv = net(mel)
            for k, t in (("on", po), ("fr", pf), ("vl", pv)):
                rolls[k].append(torch.sigmoid(t)[0].cpu().numpy().T)
    on = np.concatenate(rolls["on"], axis=1)
    fr = np.concatenate(rolls["fr"], axis=1)
    vl = np.concatenate(rolls["vl"], axis=1)
    return decode_notes(on, fr, vl), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    from app.ml.transcriber import config as C
    from app.ml.transcriber.data import find_split, official_splits
    from app.ml.transcriber.decode import note_scores

    tracks = find_split(args.split)
    if not tracks:
        _, tracks, how = official_splits()
        print(f"[note] no '{args.split}' split; using {how}")
    tracks = tracks[: args.tracks]
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else C.CKPT_DIR / "transcriber.pt"
    tmp = ROOT / "backend" / "eval" / "_drumless_tmp.wav"

    rows = {k: [] for k in ("random", "NMF (mix)", "NMF (drumless)", "transcriber")}
    counts = {k: [] for k in rows}
    for t in tracks:
        ref = ground_truth(t)
        est = {"random": random_like(ref)}
        est["NMF (mix)"] = nmf_notes(t / "mix.wav")
        d = drumless_mix(t, tmp)
        est["NMF (drumless)"] = nmf_notes(d) if d else []
        if ckpt.exists():
            est["transcriber"], ck = model_notes(t / "mix.wav", ckpt)
        else:
            est["transcriber"] = []
        print(f"  {t.name}: {len(ref)} reference notes")
        for k, v in est.items():
            s = note_scores(ref, v)
            rows[k].append(s)
            counts[k].append(len(v))

    print(f"\n{len(tracks)} tracks from '{args.split}'"
          + (f", checkpoint epoch {ck.get('epoch')}" if ckpt.exists() else
             "  [NO CHECKPOINT - transcriber row is empty]"))
    print(f"\n{'system':<17}{'notes':>8}{'F1':>8}{'P':>8}{'R':>8}{'+offsets':>10}")
    print("-" * 59)
    for k in ("random", "NMF (mix)", "NMF (drumless)", "transcriber"):
        r = rows[k]
        if not r:
            continue
        m = lambda f: float(np.mean([x[f] for x in r]))
        print(f"{k:<17}{int(np.mean(counts[k])):>8}{m('conp_f1'):>8.3f}"
              f"{m('conp_p'):>8.3f}{m('conp_r'):>8.3f}{m('conpoff_f1'):>10.3f}")
    print("\nF1/P/R are onset+pitch at 50 ms tolerance; +offsets also requires")
    print("the note length to match.  Anything at or below the random row is")
    print("measuring note density rather than accuracy.")
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
