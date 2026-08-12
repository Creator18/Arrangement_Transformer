"""
Listen to it.  Writes three WAVs per piece so the numbers can be checked by ear.

    <id>_1_input.wav      the band mixture that goes in (bass, guitar, drums)
    <id>_2_recovered.wav  what the 88-key projection recovers, rendered as piano
    <id>_3_reference.wav  the original notes rendered as piano — the target

The middle file is the whole pipeline end to end: mixture -> Demucs -> KL-NMF
against the 88-key dictionary -> peak-picked note events -> our own sampler.
Nothing is cleaned up afterwards.  No hand model, no playability constraints,
no quantisation — this is the raw extractor output, because the point is to
hear where it actually stands rather than where post-processing could carry it.

Expect it to sound thin and to miss inner voices: measured, the extractor runs
at COnP 0.375 on this exact condition, with precision 0.44 and recall 0.33.
The reference file is there so that is audible as a comparison rather than an
abstraction.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend/eval"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

from octave_ghost_test import SR, FPS, LOW, load_dictionary       # noqa: E402
from note_extraction_test import extract_notes, activations, ref_notes  # noqa: E402

OUT_SR = 44100        # render the audible files at full rate, not analysis rate


def velocities(H, notes):
    """Peak activation -> MIDI velocity.

    extract_notes returns timing and pitch only, and a flat velocity makes a
    real arrangement sound like a player piano roll punched by a machine.  The
    activation height is the only dynamic information the decomposition has, so
    it is what the dynamics come from.
    """
    peaks = []
    for on, off, p in notes:
        a, b = int(on * FPS), max(int(off * FPS), int(on * FPS) + 1)
        row = H[p - LOW, a:b]
        peaks.append(float(row.max()) if row.size else 0.0)
    if not peaks:
        return []
    lo, hi = np.percentile(peaks, 5), np.percentile(peaks, 95)
    out = []
    for v in peaks:
        t = 0.0 if hi <= lo else np.clip((v - lo) / (hi - lo), 0, 1)
        out.append(int(round(40 + 70 * t)))          # 40..110, a musical range
    return out


def to_note_dicts(notes, vels):
    return [{"pitch": int(p), "onset": float(on), "offset": float(off),
             "velocity": int(v)}
            for (on, off, p), v in zip(notes, vels)]


def write(path, y, sr):
    import soundfile as sf
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    peak = float(np.abs(y).max())
    if peak > 1e-9:
        y = y / peak * 0.95
    sf.write(str(path), y, sr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--q", type=float, default=0.90,
                    help="threshold quantile; 0.90 is what tuning chose")
    ap.add_argument("--out", default=str(ROOT / "backend/eval/data/demo"))
    args = ap.parse_args()

    import librosa
    from app.services.sampler import render_notes
    from mixture_test import (build_parts, render_part, mix, demucs_strip_drums,
                              BASS, GUITAR, DRUMS, RENDER_SR)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / "backend/eval/data/manifest.json").read_text())
    W = load_dictionary()

    picked = []
    for p in man["pieces"]:
        midi = ROOT / p["midi_path"]
        if midi.exists():
            picked.append((p, midi))
        if len(picked) >= args.pieces:
            break

    for p, midi in picked:
        pid = p["id"][:28]
        print(f"\n{p['composer']} — {p['title'][:44]}")

        low, high, drums = build_parts(midi, args.seconds)
        mixed = mix((render_part(low, *BASS[:2]), BASS[2]),
                    (render_part(high, *GUITAR[:2]), GUITAR[2]),
                    (render_part(drums, DRUMS[0], DRUMS[1]), DRUMS[2]))
        write(outdir / f"{pid}_1_input.wav", mixed, RENDER_SR)

        stripped = demucs_strip_drums(mixed)
        y = librosa.resample(stripped, orig_sr=RENDER_SR, target_sr=SR)
        H = activations(y, W)
        est = extract_notes(H, args.q)
        est.sort(key=lambda n: n[0])
        vels = velocities(H, est)
        ref = ref_notes(midi, args.seconds)
        print(f"  recovered {len(est)} notes vs {len(ref)} in the original "
              f"({len(est)/max(len(ref),1):.2f}x)")

        write(outdir / f"{pid}_2_recovered.wav",
              render_notes(to_note_dicts(est, vels), sr=OUT_SR), OUT_SR)
        write(outdir / f"{pid}_3_reference.wav",
              render_notes([{"pitch": p_, "onset": a, "offset": b,
                             "velocity": 80} for a, b, p_ in ref], sr=OUT_SR),
              OUT_SR)

    print(f"\nwrote {len(list(outdir.glob('*.wav')))} files to {outdir}")


if __name__ == "__main__":
    main()
