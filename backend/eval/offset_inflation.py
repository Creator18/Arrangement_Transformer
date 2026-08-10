"""
Is the sounding-limit crowding real, or an artefact of long offsets?

enforce_sounding_limits shortens 3.52% of notes, removing a mean 243 ms of
ring, and 81.4% of those releases are not covered by the pedal — so they cut
an audibly sounding note.  That only makes sense if the hands were genuinely
full.

But O&F is known to over-sustain offsets (it is why the same-pitch merge was
destroying re-strikes: a true 150 ms gap became 20 ms in our data).  If our
offsets run long, notes look HELD when the finger has already left, the limit
sees a full hand that is not full, and it cuts a real note to make room.

Two measurements:

  1. duration ratio, ours / truth, over notes matched by onset+pitch.
  2. at each instant where the limit forces a release, how many notes were
     sounding by OUR offsets versus by GROUND TRUTH.  If truth is
     comfortably under the limit while ours is over it, the crowding is
     manufactured and the cut was unnecessary.
"""
import sys, os, json, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa, pretty_midi, mir_eval                        # noqa: E402
from app.services.transcription import _get_of_transcriptor  # noqa: E402
from app.services.music_theory import enforce_sounding_limits, HAND_SPLIT  # noqa: E402

SR = 16000
N_PIECES = 4


def sounding_at(notes, t, key_on="onset", key_off="offset"):
    return [n for n in notes
            if float(n[key_on]) <= t < float(n[key_off])]


def main():
    man = json.load(open("backend/eval/data/manifest.json"))
    pieces = [p for p in (man["pieces"] if isinstance(man, dict) and "pieces" in man else man)
              if os.path.exists(p["audio_synth_path"])][:N_PIECES]
    inf = _get_of_transcriptor()

    ratios, our_cnt, truth_cnt, unnecessary, total_rel = [], [], [], 0, 0

    for p in pieces:
        audio, _ = librosa.load(p["audio_synth_path"], sr=SR, mono=True)
        out = inf.transcribe(audio, None)
        ours = [{"pitch": int(e["midi_note"]), "onset": float(e["onset_time"]),
                 "offset": float(e["offset_time"]), "velocity": int(e["velocity"])}
                for e in out.get("est_note_events", [])
                if 21 <= int(e["midi_note"]) <= 108
                and float(e["offset_time"]) > float(e["onset_time"])]
        ours.sort(key=lambda n: (n["onset"], n["pitch"]))

        pm = pretty_midi.PrettyMIDI(p["midi_path"])
        truth = sorted(({"pitch": n.pitch, "onset": float(n.start), "offset": float(n.end)}
                        for i in pm.instruments for n in i.notes),
                       key=lambda n: (n["onset"], n["pitch"]))

        # ── 1. duration ratio on matched notes ──────────────────────────────
        ref_iv = np.array([[n["onset"], n["offset"]] for n in truth])
        ref_hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in truth])
        est_iv = np.array([[n["onset"], n["offset"]] for n in ours])
        est_hz = np.array([librosa.midi_to_hz(n["pitch"]) for n in ours])
        matches = mir_eval.transcription.match_notes(
            ref_iv, ref_hz, est_iv, est_hz,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
        for ri, ei in matches:
            td = truth[ri]["offset"] - truth[ri]["onset"]
            od = ours[ei]["offset"] - ours[ei]["onset"]
            if td > 1e-3:
                ratios.append(od / td)

        # ── 2. crowding at forced-release instants ─────────────────────────
        limited = enforce_sounding_limits(ours)
        for a, b in zip(ours, limited):
            if a["offset"] - b["offset"] > 1e-6:
                t = float(b["offset"])          # the moment of the early release
                total_rel += 1
                o = len(sounding_at(ours, t))
                r = len(sounding_at(truth, t))
                our_cnt.append(o)
                truth_cnt.append(r)
                if r <= 10:
                    unnecessary += 1            # truth was never over the limit

    ratios = np.array(ratios)
    print(f"\nmatched notes: {len(ratios)}")
    print(f"duration ratio ours/truth   mean {ratios.mean():.2f}   "
          f"median {np.median(ratios):.2f}   "
          f"p90 {np.percentile(ratios, 90):.2f}")
    print(f"  notes we hold >1.5x too long: {100 * np.mean(ratios > 1.5):.1f}%")
    print(f"  notes we hold >3x   too long: {100 * np.mean(ratios > 3.0):.1f}%")

    print(f"\nforced releases: {total_rel}")
    print(f"  notes sounding at that instant, BY OUR OFFSETS   mean {np.mean(our_cnt):.1f}")
    print(f"  notes sounding at that instant, BY GROUND TRUTH  mean {np.mean(truth_cnt):.1f}")
    print(f"  releases where TRUTH was within the 10-key limit: "
          f"{unnecessary} ({100 * unnecessary / max(1, total_rel):.1f}%)")


if __name__ == "__main__":
    main()


# ── Result, 2026-08-09 (4 pieces, 9,679 matched notes) ───────────────────────
#
#   duration ratio ours/truth   mean 2.10   median 1.18   p90 4.06
#     held >1.5x too long  39.8%
#     held >3x   too long  16.9%
#
#   forced releases: 365
#     sounding by OUR offsets    mean 5.4
#     sounding by GROUND TRUTH   mean 2.2
#     releases where truth was within the limit:  365 (100.0%)
#
# The crowding is manufactured.  Ground truth averaged 2.2 notes sounding at
# the exact instants we decided a hand was full — nowhere near five fingers.
# Every one of the 365 early releases, each removing a mean 243 ms of ring and
# 81.4% of them not covered by the pedal, was caused by our own offsets, not
# by the music.
#
# Root cause is the offset head reporting SOUND, not key release.  O&F is
# trained on sustained piano where the damper decides when a note stops, so
# its offsets are pedal-blurred.  enforce_sounding_limits counts FINGERS, so
# feeding it sound durations is a category error: a pianist lifts the finger
# as soon as the pedal can hold the note, and those notes stop consuming a
# finger while our data says they still do.
#
# The median ratio of 1.18 against a mean of 2.10 says this is not a uniform
# stretch — most notes are close, and a long tail is enormously inflated.
# That tail is what accumulates into phantom chords.
#
# Fix direction: count a note as occupying a finger only while it is NOT
# pedal-sustained, rather than for its whole sounding duration.  Note this is
# the opposite use of the pedal from the earlier failed attempt, which tried
# to choose WHICH note to release — that could never work, since pedal state
# at the release instant is identical for every candidate.  Here the pedal
# discriminates per-note, which is the part that carries information.
