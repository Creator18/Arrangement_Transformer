"""
Onset/frame threshold sweep on the FASTEST pieces.

Why these pieces: accuracy is strongly density-dependent.  Stratifying the
50-piece baseline by notes/sec gave COnP 0.857 / 0.782 / 0.716 across slow,
middle and fast thirds, driven almost entirely by RECALL (0.807 -> 0.587).
On the fastest pieces we emit ~63% as many notes as exist, so the failure is
under-detection, not mis-timing — the quantiser loses only 0.02% of distinct
attacks, which rules it out.

Under-emission means we hold precision headroom we can spend on recall.  This
measures whether that trade actually pays, rather than assuming it.
"""
import sys, os, json, warnings, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import librosa, pretty_midi, mir_eval                       # noqa: E402
from app.services.transcription import _get_of_transcriptor  # noqa: E402

SR = 16000
ONSET_TOL = 0.05
PITCH_TOL = 50.0


def truth_of(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ns = [n for i in pm.instruments for n in i.notes]
    ns.sort(key=lambda n: n.start)
    iv = np.array([[n.start, n.end] for n in ns])
    pitch = np.array([librosa.midi_to_hz(n.pitch) for n in ns])
    return iv, pitch, len(ns)


def score(est_notes, ref_iv, ref_hz):
    if not est_notes:
        return 0.0, 0.0, 0.0
    iv = np.array([[n["onset"], n["offset"]] for n in est_notes])
    hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in est_notes])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, iv, hz,
        onset_tolerance=ONSET_TOL, pitch_tolerance=PITCH_TOL, offset_ratio=None,
    )
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--configs", default="0.30/0.10,0.20/0.10,0.15/0.08,0.10/0.05")
    args = ap.parse_args()

    man = json.load(open("backend/eval/data/manifest.json"))
    pieces = man["pieces"] if isinstance(man, dict) and "pieces" in man else man

    # Rank by note density from ground truth, take the fastest.
    scored = []
    for p in pieces:
        try:
            pm = pretty_midi.PrettyMIDI(p["midi_path"])
        except Exception:
            continue
        ns = [n for i in pm.instruments for n in i.notes]
        if len(ns) < 50 or not os.path.exists(p["audio_synth_path"]):
            continue
        on = np.array([n.start for n in ns])
        scored.append((len(ns) / max(1e-6, on.max() - on.min()), p))
    scored.sort(key=lambda x: -x[0])
    sel = scored[: args.pieces]
    if not sel:
        raise SystemExit("no usable pieces (missing audio or midi)")

    inf = _get_of_transcriptor()
    configs = [tuple(float(x) for x in c.split("/")) for c in args.configs.split(",")]

    print(f"\n{len(sel)} fastest pieces, {sel[-1][0]:.1f}-{sel[0][0]:.1f} notes/s\n")
    print(f"{'onset/frame':<14}{'COnP F1':>10}{'precision':>11}{'recall':>9}{'pred/truth':>12}")
    print("-" * 56)

    for onset_t, frame_t in configs:
        inf.onset_threshold = onset_t
        inf.frame_threshold = frame_t
        fs, ps, rs, ratio = [], [], [], []
        for dens, p in sel:
            audio, _ = librosa.load(p["audio_synth_path"], sr=SR, mono=True)
            out = inf.transcribe(audio, None)
            est = [
                {"pitch": int(e["midi_note"]),
                 "onset": float(e["onset_time"]),
                 "offset": float(e["offset_time"])}
                for e in out.get("est_note_events", [])
                if 21 <= int(e["midi_note"]) <= 108
                and float(e["offset_time"]) > float(e["onset_time"])
            ]
            ref_iv, ref_hz, n_truth = truth_of(p["midi_path"])
            pr, rc, f1 = score(est, ref_iv, ref_hz)
            ps.append(pr); rs.append(rc); fs.append(f1)
            ratio.append(len(est) / max(1, n_truth))
        print(f"{onset_t:.2f}/{frame_t:.2f}      "
              f"{np.mean(fs):>10.4f}{np.mean(ps):>11.4f}{np.mean(rs):>9.4f}{np.mean(ratio):>12.2f}")


if __name__ == "__main__":
    main()


# ── Results, 2026-08-09 ───────────────────────────────────────────────────────
#
# Sweep, 4 fastest pieces (19.5-20.6 notes/s):
#
#   onset/frame   COnP F1   precision   recall   pred/truth
#   0.30/0.10      0.9486      0.9348   0.9630         1.03   <- shipped
#   0.20/0.10      0.9437      0.9212   0.9673         1.05
#   0.15/0.08      0.9382      0.9091   0.9692         1.07
#   0.10/0.05      0.9293      0.8907   0.9715         1.09
#
# Control, same thresholds, slowest vs fastest:
#
#   SLOWEST 4 (3.6-5.7 n/s)    COnP 0.9388   recall 0.9881   pred/truth 1.11
#   FASTEST 4 (19.5-20.6 n/s)  COnP 0.9486   recall 0.9630   pred/truth 1.03
#
# CONCLUSION, and it inverts the hypothesis this file was written to test.
#
# The model does not struggle with fast passages.  Raw O&F is FLAT across
# density — 0.949 on Gnomenreigen-class material versus 0.939 on the slowest
# pieces, recall 0.96 either way.  Lowering thresholds buys a little recall
# and loses more precision, so the shipped 0.30/0.10 is already the best of
# the four.  There is nothing to win here.
#
# But the 50-piece baseline puts those same fast pieces at COnP 0.60-0.77 with
# recall 0.45-0.65 and pred/truth 0.49-0.66.  Raw model 0.95, shipped output
# 0.66, same audio, same ground truth.
#
# So the density-dependent collapse is entirely in OUR post-processing, not in
# the model or its thresholds.  Next step is an ablation of the cleanup chain
# (dedupe_octave_partials, refine_notes, remove_beat_outliers,
# enforce_sounding_limits, the 10-key cap) on these pieces, stage by stage.
#
# Caveat: the baseline is from an older commit and ran the full pipeline
# including separation, whereas this runs O&F directly on the synthesised
# audio.  The gap is far too large to be explained by that alone, but the
# ablation should re-measure rather than inherit the number.
