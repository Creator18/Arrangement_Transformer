"""
Velocity fidelity — the axis COnP and COnPOff cannot see.

Both transcription metrics score onset, pitch and offset.  Neither looks at
velocity at all, so any change to dynamics is invisible to the existing
harness and would be tuned blind.  This is the missing gauge.

Scored on notes MATCHED by onset+pitch, because comparing raw distributions
conflates two different failures: predicting the wrong dynamics, and
predicting the wrong notes.  Only matched pairs isolate the first.

Four numbers, each answering a different question:

  spread_ratio   std(ours) / std(truth).  <1 means we hedge toward the mean,
                 which is the characteristic failure of a regression head
                 trained with a squared-error objective.
  pearson        do we get the SHAPE right — loud where it is loud — even if
                 the scale is compressed?  A high correlation with a low
                 spread ratio means a calibration curve can fix it outright.
  mae            average absolute error in MIDI velocity units (0-127).
  soft_bias      mean error on the quietest fifth of true notes.  Reported
                 separately because compression hurts asymmetrically: the
                 soft end is where an expressive performance lives, and
                 averaging hides it.
"""
import sys, os, json, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa, pretty_midi, mir_eval                        # noqa: E402


def velocity_scores(ours, truth):
    """ours/truth: [{pitch, onset, offset, velocity}].  Returns a dict."""
    if not ours or not truth:
        return None
    ref_iv = np.array([[n["onset"], n["offset"]] for n in truth])
    ref_hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in truth])
    est_iv = np.array([[n["onset"], n["offset"]] for n in ours])
    est_hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in ours])
    m = mir_eval.transcription.match_notes(
        ref_iv, ref_hz, est_iv, est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
    if len(m) < 20:
        return None
    t = np.array([float(truth[ri]["velocity"]) for ri, _ in m])
    o = np.array([float(ours[ei]["velocity"]) for _, ei in m])
    soft = t <= np.percentile(t, 20)
    return {
        "n": len(m),
        "spread_ratio": float(o.std() / max(1e-6, t.std())),
        "pearson": float(np.corrcoef(t, o)[0, 1]),
        "mae": float(np.mean(np.abs(o - t))),
        "soft_bias": float(np.mean(o[soft] - t[soft])),
        "mean_ours": float(o.mean()), "mean_truth": float(t.mean()),
        "std_ours": float(o.std()), "std_truth": float(t.std()),
    }


def main():
    from app.services.transcription import _get_of_transcriptor
    man = json.load(open("backend/eval/data/manifest.json"))
    pieces = [p for p in (man["pieces"] if isinstance(man, dict) and "pieces" in man else man)
              if os.path.exists(p["audio_synth_path"])][:6]
    inf = _get_of_transcriptor()
    rows = []
    for p in pieces:
        audio, _ = librosa.load(p["audio_synth_path"], sr=16000, mono=True)
        out = inf.transcribe(audio, None)
        ours = [{"pitch": int(e["midi_note"]), "onset": float(e["onset_time"]),
                 "offset": float(e["offset_time"]), "velocity": int(e["velocity"])}
                for e in out.get("est_note_events", [])
                if 21 <= int(e["midi_note"]) <= 108
                and float(e["offset_time"]) > float(e["onset_time"])]
        pm = pretty_midi.PrettyMIDI(p["midi_path"])
        truth = [{"pitch": n.pitch, "onset": float(n.start),
                  "offset": float(n.end), "velocity": int(n.velocity)}
                 for i in pm.instruments for n in i.notes]
        s = velocity_scores(ours, truth)
        if s:
            rows.append(s)
    if not rows:
        raise SystemExit("no scorable pieces")
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("spread_ratio", "pearson", "mae", "soft_bias",
                     "mean_ours", "mean_truth", "std_ours", "std_truth")}
    n = sum(r["n"] for r in rows)
    print(f"\nvelocity fidelity — {len(rows)} pieces, {n} matched notes\n")
    print(f"  spread ratio (ours/truth) : {agg['spread_ratio']:.3f}   (1.0 = faithful)")
    print(f"  pearson r                 : {agg['pearson']:.3f}   (shape, scale-free)")
    print(f"  mean abs error            : {agg['mae']:.1f} velocity units")
    print(f"  bias on softest 20%       : {agg['soft_bias']:+.1f}   (+ = we play it too loud)")
    print(f"\n  ours  mean {agg['mean_ours']:.1f}  std {agg['std_ours']:.1f}")
    print(f"  truth mean {agg['mean_truth']:.1f}  std {agg['std_truth']:.1f}")


if __name__ == "__main__":
    main()


# ── Baseline, 2026-08-09 (6 pieces, 12,805 matched notes) ────────────────────
#
#   spread ratio (ours/truth) : 0.733
#   pearson r                 : 0.835
#   mean abs error            : 11.1 velocity units
#   bias on softest 20%       : +14.8
#
#   ours  mean 68.6  std 11.2
#   truth mean 59.9  std 15.2
#
# READ THESE TWO TOGETHER: r = 0.835 with a spread ratio of 0.733 says the
# ORDERING is largely right — we know which notes are louder — and only the
# SCALE is wrong.  That is the good case, because a monotonic calibration can
# fix a compressed scale outright, whereas a low correlation would mean the
# model does not hear dynamics at all and no post-hoc mapping could help.
#
# The +14.8 bias on the softest fifth is where it is worst and where it is
# most audible: we cannot play quietly.  Expressive performance lives at the
# soft end, and the mean error of 11.1 hides this because the loud end is
# comparatively accurate.
#
# This is the signature of a regression head trained on squared error, which
# is minimised by hedging toward the conditional mean.
#
# Next step is a calibration fitted on this corpus — quantile mapping from our
# velocity distribution onto the true one, or the simpler expansion
#     v' = mean_truth + (v - mean_ours) * (std_truth / std_ours)
# which this baseline already gives every constant for.  Re-run this file to
# score any candidate: spread ratio toward 1.0 and soft bias toward 0, with
# pearson NOT dropping (a mapping that improved spread by scrambling the
# ordering would be worse, and only r would reveal it).
