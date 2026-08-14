"""
Are the notes in time — and if not, which stage moved them?

"The timing is off" has at least four causes, and they take different fixes, so
measuring them apart matters more than getting one number.

    1. GLOBAL LAG      everything is uniformly early or late.  A constant, and
                       therefore the cheapest possible fix.  Measured by
                       cross-correlating the onset envelope of the rendered
                       arrangement against the original audio — literally the
                       "compare the produced track with the original" check.

    2. vs THE EVIDENCE decoded onsets against where the activation for that
                       pitch actually rises in H.  This needs no ground truth
                       at all: H is what the notes were derived FROM, so any
                       disagreement is our own error.  A systematic bias here
                       is a decoder problem; a wide spread is a precision one.

    3. SNAP DAMAGE     how far snap_to_beat_grid moves each note.  Snapping is
                       supposed to remove sub-beat scatter, not relocate notes.
                       A note pushed more than half a subdivision landed on the
                       WRONG sixteenth, which is audible as a note in the wrong
                       place rather than as tightening.

    4. GRID QUALITY    are madmom's beats on the music?  Distance from each
                       grid line to the nearest spectral-flux peak, against a
                       deliberately mis-phased control.  If the grid itself is
                       wrong then 3 is guaranteed and fixing the decoder is
                       pointless.

Only 1 and 4 involve the original audio; 2 and 3 are internal consistency and
are the ones that can be fixed without re-running separation.

Run against a session that has already been through the pipeline:
    python eval/timing_alignment_test.py --session <id>

RESULT (session 23f6a3dc, two-voice path, 2026-08-14)
-----------------------------------------------------
    1. global lag                    -11.6 ms   (negligible, and slightly EARLY)
    2. vs evidence  median +0.0 ms, p25 -23.2, p75 +46.4, |.|>50ms 35.2%
    3. snap damage  median 0.0 ms, p90 0.0 ms   (a no-op)
    4. grid quality 34.8 ms vs 46.4 ms mis-phased control

Not a lag, and not the snap.  Test 3 reads as a no-op because the notes are
ALREADY hard-quantised when they reach /transform: 99.3% of them sit within
5 ms of a grid line, done upstream by _beat_adaptive_quantize in
transcription.py.  snap_to_beat_grid then has nothing left to move.

So put 2 and 3 together.  The notes are locked to the sixteenth grid, a
sixteenth here is 81 ms, and 35.2% of them are more than 50 ms from their own
activation evidence.  A note cannot be both on the grid and 50+ ms from its
evidence unless it was quantised onto the WRONG SIXTEENTH — the error is
discrete, not a matter of precision, which is why it reads as notes in the
wrong place rather than as looseness.

That points the fix at slot CHOICE, not at quantisation strength.  Loosening
the quantiser would only bring back the out-of-sync complaint that caused it.
Picking each note's subdivision by where its activation actually rises keeps
everything perfectly quantised — so the parts still agree — while landing it
on the right one.

Caveat on 4: the grid beats the mis-phased control by only 34.8 vs 46.4 ms,
which is a thin margin, and 34.8 ms is itself nearly half a subdivision.  The
control is weak (a half-beat shift still lands on eighth-note content), so
this does not prove the grid is bad — but it does not clear it either, and if
the grid is loose then the right slot is still in the wrong place.  Worth
settling separately against POP909's beat annotations.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
warnings.filterwarnings("ignore")
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann

TEMP = ROOT / "backend/temp"


def onset_env(y, sr, hop=256):
    import librosa
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), sr / hop


def global_lag(notes, orig_y, sr, max_lag_s=0.5):
    """Cross-correlate rendered vs original onset envelopes.  Returns ms."""
    from app.services.audio_utils import render_notes_to_audio
    rend = np.asarray(render_notes_to_audio(notes, sr=sr), dtype=np.float64)
    if rend.ndim > 1:
        rend = rend.mean(axis=1)
    n = min(len(rend), len(orig_y))
    a, fps = onset_env(rend[:n], sr)
    b, _ = onset_env(np.asarray(orig_y[:n], dtype=np.float64), sr)
    m = min(len(a), len(b))
    a, b = a[:m] - a[:m].mean(), b[:m] - b[:m].mean()
    lags = int(max_lag_s * fps)
    c = np.correlate(a, b, mode="full")
    mid = len(c) // 2
    w = c[mid - lags: mid + lags + 1]
    return (int(np.argmax(w)) - lags) / fps * 1000.0


def vs_evidence(notes, H, low, proj_fps, win_s=0.25):
    """Signed ms from each note's onset to the activation rise that supports it."""
    win = max(1, int(win_s * proj_fps))
    d = np.diff(H, axis=1, prepend=H[:, :1])          # positive = rising
    offs = []
    for n in notes:
        r = int(n["pitch"]) - low
        if not (0 <= r < H.shape[0]):
            continue
        f = int(float(n["onset"]) * proj_fps)
        a, b = max(0, f - win), min(H.shape[1], f + win + 1)
        if b <= a:
            continue
        seg = d[r, a:b]
        if not np.any(seg > 0):
            continue
        offs.append((a + int(np.argmax(seg)) - f) / proj_fps * 1000.0)
    return np.array(offs)


def snap_damage(pre, post, beat_times, subdivisions=4):
    """Match notes by pitch+nearest onset, report how far the snap moved them."""
    by_pitch = {}
    for n in pre:
        by_pitch.setdefault(int(n["pitch"]), []).append(float(n["onset"]))
    for k in by_pitch:
        by_pitch[k] = np.sort(np.array(by_pitch[k]))
    moved = []
    for n in post:
        arr = by_pitch.get(int(n["pitch"]))
        if arr is None or not len(arr):
            continue
        o = float(n["onset"])
        j = int(np.argmin(np.abs(arr - o)))
        moved.append((o - arr[j]) * 1000.0)
    bt = np.asarray(beat_times, float)
    sub_ms = float(np.median(np.diff(bt))) / subdivisions * 1000.0
    return np.array(moved), sub_ms


def grid_quality(beat_times, y, sr):
    """Median ms from a grid line to the nearest onset peak, vs a shifted control."""
    import librosa
    env, fps = onset_env(y, sr)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                post_avg=5, delta=0.2, wait=5) / fps
    if len(pk) < 8:
        return None, None
    bt = np.asarray(beat_times, float)
    bt = bt[(bt >= pk[0]) & (bt <= pk[-1])]
    if len(bt) < 8:
        return None, None

    def med(g):
        i = np.clip(np.searchsorted(pk, g), 1, len(pk) - 1)
        return float(np.median(np.minimum(np.abs(pk[i] - g),
                                          np.abs(pk[i - 1] - g))) * 1000.0)
    half_beat = float(np.median(np.diff(bt))) / 2.0
    return med(bt), med(bt + half_beat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--skip-nmf", action="store_true",
                    help="skip test 2, which needs an 80 s decomposition")
    args = ap.parse_args()

    import librosa
    sess = (TEMP / args.session if args.session else
            max((d for d in TEMP.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime))
    print(f"session {sess.name}\n")

    tr = json.loads((sess / "transcription.json").read_text())
    tf = json.loads((sess / "transformed.json").read_text())
    pre = tr.get("notes", [])
    post = tf["transformed_notes"] if isinstance(tf, dict) else tf
    meta = json.loads((sess / "components.json").read_text())
    hop, msr = float(meta.get("hop_length", 512)), float(meta.get("sr", 22050))
    beats = np.asarray(meta.get("beat_frames", []), float) * hop / msr
    print(f"{len(pre)} decoded notes, {len(post)} after transform, "
          f"{len(beats)} beats\n")

    from app.services.piano_projection import SR, FPS as PROJ_FPS, LOW
    y, _ = librosa.load(str(sess / "audio.wav"), sr=SR, mono=True)

    print("1. GLOBAL LAG (rendered arrangement vs original audio)")
    try:
        print(f"   {global_lag(post, y, SR):+.1f} ms   "
              "(positive = our notes are LATE)\n")
    except Exception as e:                                      # noqa: BLE001
        print(f"   failed: {type(e).__name__}: {e}\n")

    if not args.skip_nmf:
        print("2. vs THE EVIDENCE (decoded onset -> activation rise in H)")
        from app.services.piano_projection import _activations
        from app.services.melody_decode import (VOCAL_HI as MHI,
                                                BASS_LO as BLO)
        H = _activations(sess / "components/harmonic.wav", BLO, MHI)
        if H is None:
            print("   harmonic.wav missing\n")
        else:
            o = vs_evidence(pre, H, LOW, PROJ_FPS)
            if len(o):
                print(f"   n={len(o)}  median {np.median(o):+.1f} ms   "
                      f"p25 {np.percentile(o,25):+.1f}  "
                      f"p75 {np.percentile(o,75):+.1f}  "
                      f"|.|>50ms {100*np.mean(np.abs(o)>50):.1f}%\n")

    print("3. SNAP DAMAGE (how far the beat-grid snap moved each note)")
    moved, sub_ms = snap_damage(pre, post, beats)
    if len(moved):
        am = np.abs(moved)
        print(f"   subdivision = {sub_ms:.0f} ms; a move past {sub_ms/2:.0f} ms "
              "is the WRONG sixteenth")
        print(f"   median |move| {np.median(am):.1f} ms   p90 {np.percentile(am,90):.1f}"
              f"   past half a subdivision: {100*np.mean(am > sub_ms/2):.1f}%\n")

    print("4. GRID QUALITY (beat lines vs onset peaks in the original)")
    g, ctrl = grid_quality(beats, y, SR)
    if g is None:
        print("   too few onset peaks to judge\n")
    else:
        print(f"   median distance {g:.1f} ms   mis-phased control {ctrl:.1f} ms")
        print(f"   the grid is on the music if this is well under the control.\n")

    print("Read 4 first: if the grid is wrong, 3 is guaranteed and fixing the")
    print("decoder cannot help.  Then 2 — a systematic median there is a")
    print("constant to subtract; a wide spread is a precision problem instead.")


if __name__ == "__main__":
    main()
