"""
Which signal should the beat tracker actually get, on a SONG?

separation.py feeds madmom the DRUMLESS mix, with the stated reason that
"drumless input keeps drum transients from confusing onset detection".  That
reasoning is about pitch onsets for transcription, where drum transients really
do pollute the estimate.  It was then applied to BEAT TRACKING, where it is
backwards: the drums are the beat, and we remove them before asking where the
beats are.

It has never been tested on a song.  beat_source_test.py asked whether a cheaper
source AGREES with the Demucs grid — it treated that grid as the reference
rather than questioning it — and it ran on solo piano, where there are no drums
to remove so the question could not arise.

This matters more than a tempo readout, because the grid is load-bearing in four
places at once: beat snapping, the arpeggiator's bar edges, the metrical prior
inside the melody decode, and the 16-slot rhythm retrieval.  A wrong grid makes
all four wrong together, which is the hardest kind of error to recognise as a
grid problem.

No external ground truth here, so two things are measured:

  agreement   pairwise beat F-measure between candidate grids.  Disagreement
              alone proves one of them is wrong without saying which.
  drum fit    what fraction of DRUM-STEM onsets land within tolerance of each
              grid's beats.  This is the closest thing to truth available: a
              kick and snare pattern is the most reliable rhythmic evidence in
              a pop mix, so a grid that fails to explain it is the wrong grid.
              Note the drum stem is used only to SCORE here, which is why it is
              fair even for the candidate that was tracked from drums.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")

TOL = 0.070          # mir_eval's standard beat tolerance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--sessions", type=int, default=2)
    args = ap.parse_args()

    import librosa, mir_eval, torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from app.services import separation as S

    model = get_model("htdemucs_6s")
    temp = ROOT / "backend/temp"
    picked = []
    for d in sorted(temp.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        mp = d / "components.json"
        if mp.exists() and (d / "audio.wav").exists():
            if json.loads(mp.read_text()).get("method", "").startswith("demucs"):
                picked.append(d)
        if len(picked) >= args.sessions:
            break

    for sess in picked:
        y, _ = librosa.load(str(sess / "audio.wav"), sr=44100, mono=True,
                            duration=args.seconds)
        mix = torch.from_numpy(np.stack([y, y]).astype(np.float32))
        with torch.no_grad():
            out = apply_model(model, mix.unsqueeze(0))[0].cpu().numpy()
        stems = {s: out[i].mean(axis=0) for i, s in enumerate(model.sources)}

        def to_sr(sig):
            return librosa.resample(sig.astype(np.float32), orig_sr=44100,
                                    target_sr=S.SR)

        drums = to_sr(stems["drums"])
        drumless = to_sr(np.sum([v for k, v in stems.items() if k != "drums"],
                                axis=0))
        full = to_sr(y)
        drums_bass = to_sr(stems["drums"] + stems["bass"])

        candidates = {
            "drumless (current)": drumless,
            "drums only": drums,
            "full mix": full,
            "drums + bass": drums_bass,
        }

        # Reference onsets for scoring: the drum stem's own attacks.
        d_on = librosa.onset.onset_detect(y=drums, sr=S.SR, units="time",
                                          backtrack=False)

        print(f"\n=== session {sess.name[:8]} — {len(y)/44100:.0f}s, "
              f"{len(d_on)} drum onsets ===")
        grids = {}
        print(f"{'source':<20}{'tempo':>8}{'b/bar':>7}{'beats':>7}{'downb':>7}"
              f"{'drum fit':>10}{'median dev':>12}{'sec':>7}")
        for name, sig in candidates.items():
            t0 = time.perf_counter()
            try:
                bf, dbf, tempo, bpb = S._madmom_beats(sig, S.SR)
            except Exception as e:                              # noqa: BLE001
                print(f"{name:<20}  failed: {type(e).__name__}: {e}")
                continue
            el = time.perf_counter() - t0
            hop = 512
            beats = np.asarray(bf, dtype=float) * hop / S.SR
            downs = np.asarray(dbf, dtype=float) * hop / S.SR
            grids[name] = beats
            if len(beats) and len(d_on):
                dev = np.abs(d_on[:, None] - beats[None, :]).min(axis=1)
                fit = 100.0 * float(np.mean(dev <= TOL))
                med = 1000.0 * float(np.median(dev))
            else:
                fit, med = 0.0, float("nan")
            print(f"{name:<20}{tempo:>8.1f}{bpb:>7}{len(beats):>7}{len(downs):>7}"
                  f"{fit:>9.0f}%{med:>11.1f}ms{el:>7.1f}")

        names = list(grids)
        print(f"\n  pairwise beat F-measure (tolerance {int(TOL*1000)} ms)")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                f = mir_eval.beat.f_measure(grids[a], grids[b], f_measure_threshold=TOL)
                print(f"    {a:<20} vs {b:<20} F = {f:.3f}")

    print("\n'drum fit' is the closest thing to truth: the share of drum-stem")
    print("attacks a grid explains. A grid that cannot explain the kick and")
    print("snare is the wrong grid, whatever its tempo readout says.")


if __name__ == "__main__":
    main()
