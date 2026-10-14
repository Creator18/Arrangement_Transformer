"""
Per-component transcription.

After separation.py (Demucs) has split the audio into a bass-register and
upper-register stem:

  bass.wav  → pYIN monophonic pitch tracking → bass notes (layer='bass')
  upper.wav → Basic Pitch polyphonic model  → upper notes (no layer label)

Upper notes carry no `layer` field — pattern_extraction.py classifies them
via cycle-based repetition mining instead of the old NMF-component-based
labeling, which is no longer available after the Demucs swap.
"""
from __future__ import annotations

import json
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.services.nmf_voice_leading import quantize_notes, apply_feel_offset


# ── pYIN bass transcription ───────────────────────────────────────────────────

def _pyin_to_notes(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin_note: str = 'A0',
    fmax_note: str = 'G3',
    min_duration: float = 0.06,
) -> List[Dict[str, Any]]:
    """
    Convert a pYIN pitch track on a bass-register audio signal into note events.

    Consecutive voiced frames within 2 semitones of each other are merged into
    a single note.  Gaps shorter than 80 ms are bridged.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin_note),
        fmax=librosa.note_to_hz(fmax_note),
        sr=sr,
        hop_length=hop_length,
        frame_length=4096,
        fill_na=None,
    )

    with np.errstate(invalid='ignore'):
        midi_track = np.where(
            voiced_flag & (f0 > 0),
            librosa.hz_to_midi(np.where(f0 > 0, f0, 1.0)),
            np.nan,
        )

    frame_times = librosa.frames_to_time(np.arange(len(midi_track)), sr=sr, hop_length=hop_length)
    frame_dur   = float(frame_times[1] - frame_times[0]) if len(frame_times) > 1 else hop_length / sr

    notes: List[Dict[str, Any]] = []
    in_note       = False
    note_onset    = 0.0
    note_pitches: List[float] = []

    def _flush(onset: float, pitches: List[float], offset: float) -> Optional[Dict]:
        if offset - onset < min_duration or not pitches:
            return None
        pitch = int(round(float(np.median(pitches))))
        pitch = max(21, min(108, pitch))
        return {
            'pitch':      pitch,
            'pitch_name': librosa.midi_to_note(pitch),
            'onset':      round(onset, 4),
            'offset':     round(offset, 4),
            'velocity':   80,
            # Provisional layer — relabelled to bass_pattern / bass_melody
            # downstream using the REPET-SIM mask when one is available.
            'layer':      'bass',
        }

    for i, midi in enumerate(midi_track):
        t = float(frame_times[i])
        if not np.isnan(midi):
            if not in_note:
                in_note    = True
                note_onset = t
                note_pitches = [float(midi)]
            else:
                # Bridge small gaps and tolerate ±2 semitone pitch variation
                note_pitches.append(float(midi))
        else:
            if in_note:
                n = _flush(note_onset, note_pitches, t)
                if n:
                    notes.append(n)
                in_note = False
                note_pitches = []

    if in_note and note_pitches:
        n = _flush(note_onset, note_pitches, float(frame_times[-1]))
        if n:
            notes.append(n)

    return notes


# ── Basic Pitch upper transcription ───────────────────────────────────────────
# (NMF-component-based layer labeling was removed when separation switched
# to Demucs.  pattern_extraction.py now classifies unlabeled upper notes via
# cycle mining, which works on register-mixed input without needing a
# precomputed NMF activation matrix.)

def _basic_pitch_notes(wav_path: Path) -> List[Dict[str, Any]]:
    """Run Basic Pitch on a wav file, return raw note list."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    _, midi_data, _ = predict(
        str(wav_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.55,
        frame_threshold=0.35,
        minimum_note_length=60,
        minimum_frequency=float(librosa.note_to_hz('C2')),
        maximum_frequency=float(librosa.note_to_hz('C8')),
        melodia_trick=True,
    )

    notes: List[Dict[str, Any]] = []
    if not midi_data.instruments:
        return notes
    for note in midi_data.instruments[0].notes:
        notes.append({
            'pitch':      int(note.pitch),
            'pitch_name': librosa.midi_to_note(int(note.pitch)),
            'onset':      float(note.start),
            'offset':     float(note.end),
            'velocity':   int(note.velocity),
        })
    return notes


# ── Onsets & Frames upper transcription (PyTorch port of Magenta) ────────────
# Higher recall + sharper onsets + meaningful velocity vs Basic Pitch.  Phase 7
# is what makes this safe — its trained pattern classifier labels every note,
# so the density jump doesn't overwhelm the cycle-mining classifier that
# pattern_extraction.py would otherwise fall back to.

_of_transcriptor = None
_OF_CKPT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
_OF_CKPT_FILENAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
_OF_CKPT_MIN_BYTES = int(1.6e8)


def _ensure_of_checkpoint() -> Path:
    """
    Download the Onsets & Frames checkpoint into the package's expected cache
    path using urllib (so this works on Windows — the upstream package
    auto-downloader shells out to `wget`, which isn't on PATH there).
    Idempotent: an existing file of the expected size is left alone.
    """
    import urllib.request
    from pathlib import Path as _P
    ckpt_dir = _P.home() / "piano_transcription_inference_data"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / _OF_CKPT_FILENAME

    if ckpt_path.exists() and ckpt_path.stat().st_size >= _OF_CKPT_MIN_BYTES:
        return ckpt_path

    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    print(f"[onsets_frames] downloading checkpoint to {ckpt_path} (~165 MB)…")
    with urllib.request.urlopen(_OF_CKPT_URL) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total:
                print(f"\r[onsets_frames] {downloaded // (1<<20)} / {total // (1<<20)} MiB",
                      end="", flush=True)
        print()
    tmp_path.replace(ckpt_path)
    return ckpt_path


# The upstream package hardcodes batch_size=1 in transcribe(), so a piece is
# pushed through the GPU as a series of separate 10-second calls with no
# parallelism.  On an 8 GB card with ~7.4 GB free that is pure waste.
# Measured on a 2.3-minute piece: 23.0s at batch 1, 12.8s at 4, 10.1s at 8 —
# 2.3x faster with byte-identical output (833 notes in every case).
OF_BATCH_SIZE = 8


def _patch_of_batch_size(batch_size: int = OF_BATCH_SIZE) -> None:
    """
    Force a larger inference batch.  Done by wrapping the module-level
    `forward` rather than reimplementing transcribe(), so we inherit any
    upstream fixes to the segmenting and post-processing logic.  Idempotent.
    """
    import piano_transcription_inference.inference as INF
    if getattr(INF.forward, "_batched", False):
        return
    _orig = INF.forward

    def _batched_forward(model, x, batch_size=1, _orig=_orig, _bs=batch_size):
        return _orig(model, x, batch_size=_bs)

    _batched_forward._batched = True
    INF.forward = _batched_forward


def _get_of_transcriptor():
    """Construct (or fetch) the Onsets & Frames transcriber. Lazy + cached."""
    global _of_transcriptor
    if _of_transcriptor is not None:
        return _of_transcriptor
    import torch
    from piano_transcription_inference import PianoTranscription
    ckpt_path = _ensure_of_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        _patch_of_batch_size()
    _of_transcriptor = PianoTranscription(device=device, checkpoint_path=str(ckpt_path))
    return _of_transcriptor


# Damper-down intervals from the last _onsets_frames_notes call.  The model
# predicts pedal on a dedicated head (the checkpoint is named
# note_F1=0.9677_pedal_F1=0.9186) and we used to discard it entirely, which
# forced sustain to be faked by lengthening notes — and made a repeated note
# impossible, since you must release a key to strike it again and releasing
# was the only way to stop the sound.
_last_pedal: List[Dict[str, float]] = []

# Which model produced the notes now moving through the pipeline.  Kept at
# module level like _last_pedal, because per-note tags get dropped by any stage
# that rebuilds note dicts (measured: a role tag survived on ~200 of 2,851
# notes through one such stage).
_last_source: str = "bytedance"


def _onsets_frames_notes(wav_path: Path) -> List[Dict[str, Any]]:
    """
    Run Onsets & Frames on a WAV file, return raw note list (88-key range).

    Note offsets are KEY RELEASES, which is what the model predicts and what
    the ground truth records.  Sustain is not baked in here; the damper
    intervals are captured separately into `_last_pedal`.
    """
    global _last_pedal
    from piano_transcription_inference import sample_rate as OF_SR

    audio, _ = librosa.load(str(wav_path), sr=OF_SR, mono=True)
    if audio.size == 0:
        _last_pedal = []
        return []

    transcriptor = _get_of_transcriptor()
    result = transcriptor.transcribe(audio.astype(np.float32), midi_path=None)

    _last_pedal = [
        {"onset": round(float(e["onset_time"]), 4),
         "offset": round(float(e["offset_time"]), 4)}
        for e in result.get("est_pedal_events", [])
        if float(e["offset_time"]) > float(e["onset_time"])
    ]

    notes: List[Dict[str, Any]] = []
    for ev in result.get("est_note_events", []):
        pitch = int(ev["midi_note"])
        if pitch < 21 or pitch > 108:
            continue
        onset  = float(ev["onset_time"])
        offset = float(ev["offset_time"])
        if offset <= onset:
            continue
        notes.append({
            "pitch":      pitch,
            "pitch_name": librosa.midi_to_note(pitch),
            "onset":      onset,
            "offset":     offset,
            "velocity":   calibrate_velocity(ev["velocity"]),
        })
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes


# ── velocity calibration ──────────────────────────────────────────────────────
# O&F's velocity head is a regression trained on squared error, which is
# minimised by hedging toward the conditional mean.  Measured over 6 pieces /
# 12,805 notes matched by onset+pitch against ground truth:
#
#     spread ratio (ours/truth)  0.733      ours  mean 68.6  std 11.2
#     pearson r                  0.835      truth mean 59.9  std 15.2
#     bias on the softest 20%   +14.8
#
# Those first two numbers together are why a fix is possible at all: r = 0.835
# means the ORDERING is right — we know which notes are louder — and only the
# scale is compressed.  A low correlation would have meant the model does not
# hear dynamics and no post-hoc mapping could help.
#
# So this is a plain linear expansion about the mean, mapping our distribution
# onto the true one.  Deliberately NOT a quantile mapping: that would fit this
# corpus more tightly and generalise worse, and these constants already come
# from synthesised MAESTRO — one instrument, one set of performers.  A linear
# rescale is the weakest assumption that fixes the measured defect.
#
# The audible target is the soft end.  We currently cannot play quietly, and
# that is where expressive performance lives; the mean error hides it because
# the loud end is comparatively accurate.
_VEL_SRC_MEAN, _VEL_SRC_STD = 68.6, 11.2
_VEL_TGT_MEAN, _VEL_TGT_STD = 59.9, 15.2
CALIBRATE_VELOCITY = True


def calibrate_velocity(v: float) -> int:
    """Expand a predicted velocity onto the true dynamic range."""
    if not CALIBRATE_VELOCITY:
        return max(1, min(127, int(round(float(v)))))
    out = _VEL_TGT_MEAN + (float(v) - _VEL_SRC_MEAN) * (_VEL_TGT_STD / _VEL_SRC_STD)
    return max(1, min(127, int(round(out))))


# ── post-processing: refine note list ─────────────────────────────────────────

# Rejoin same-pitch notes separated by <50 ms.  See the call site: this was
# written for Basic Pitch's fragmentation and, with O&F, destroys 40% of
# repeated-note articulation instead.
MERGE_SAME_PITCH = False


# Shortest note we keep.  Was 60 ms, and that single number was the largest
# accuracy loss in the pipeline.
#
# Measured over all 50 eval pieces, O&F output plus this filter and nothing
# else:
#
#     floor    COnP F1   precision   recall
#      60 ms    0.8632      0.9281   0.8135
#      25 ms    0.9529      0.9335   0.9736
#
# +0.0896 F1 mean, 50/50 pieces improved, none worse; worst case +0.0011,
# best +0.2402.  Precision rose too, so this is not a recall-for-precision
# trade — the filter was deleting true positives outright.
#
# The damage is density-dependent, which is why it showed up as "the model
# cannot handle fast passages".  Fast playing means short notes: 28.3% of
# corpus notes are under 60 ms, but 64.3% of Gnomenreigen's are.  A fixed
# floor therefore takes a disproportionate bite out of exactly the virtuoso
# writing where it was noticed.  Raw O&F scores 0.9486 on the fastest pieces
# and 0.9388 on the slowest — the model was never the problem.
#
# Like the same-pitch merge above, this is a Basic Pitch holdover.  That
# model fragmented notes and sprayed short spurious fragments; O&F has a
# dedicated offset head and does neither.  Below ~25 ms the floor does still
# earn its keep (0 ms scores slightly worse), so it is lowered, not removed.
MIN_NOTE_DURATION_S = 0.025


def refine_notes(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Drop notes shorter than MIN_NOTE_DURATION_S, optionally merge same-pitch
    notes with gap < 50 ms (off by default), remove exact duplicates within
    10 ms.
    """
    if not notes:
        return notes

    notes = [n for n in notes
             if float(n['offset']) - float(n['onset']) >= MIN_NOTE_DURATION_S]

    notes.sort(key=lambda n: (n['pitch'], n['onset']))
    merged: List[Dict] = []
    for n in notes:
        # Same-pitch merging is OFF by default — measured, it was destroying
        # 40% of all repeated-note articulation.
        #
        # Traced stage by stage on a test piece: ground truth has 142
        # re-strikes, raw O&F output has 192 (it over-detects, so the
        # evidence is abundant), and after this merge only 116 survive.
        #
        # My earlier exoneration of this rule measured the wrong thing.  Real
        # re-strike GAPS are 50-350 ms, comfortably outside the 50 ms window
        # — but the window is applied to OUR note boundaries, and O&F
        # over-sustains offsets, so a true 150 ms gap becomes a 20 ms gap in
        # our data and gets merged.
        #
        # The rule is a holdover from the Basic Pitch era, which fragmented
        # notes into pieces that genuinely needed rejoining.  O&F has a
        # dedicated offset head and rarely fragments, so the rule now costs
        # far more than it saves.
        if (MERGE_SAME_PITCH
                and merged
                and merged[-1]['pitch'] == n['pitch']
                and not n.get('restrike')
                and float(n['onset']) - float(merged[-1]['offset']) < 0.05):
            merged[-1]['offset']   = max(merged[-1]['offset'],   float(n['offset']))
            merged[-1]['velocity'] = max(merged[-1]['velocity'],  int(n.get('velocity', 80)))
        else:
            merged.append(dict(n))
    notes = merged

    notes.sort(key=lambda n: (n['onset'], n['pitch']))
    deduped: List[Dict] = []
    for n in notes:
        if (deduped
                and deduped[-1]['pitch'] == n['pitch']
                and abs(float(n['onset']) - float(deduped[-1]['onset'])) < 0.01):
            continue
        deduped.append(n)

    deduped.sort(key=lambda n: (n['onset'], n['pitch']))
    return deduped


def dedupe_octave_partials(
    notes:     List[Dict],
    onset_tol: float = 0.04,
    vel_ratio: float = 0.55,
) -> List[Dict]:
    """
    Drop probable harmonic-partial hallucinations: a note exactly 12 or 24
    semitones ABOVE another note at (nearly) the same onset, but clearly
    weaker.  Piano partials at 2x/4x the fundamental frequency routinely
    trip the transcriber into emitting a phantom note an octave (or two) up.

    The velocity-ratio gate is what protects real octave doublings — an
    intentional LH octave or octave-doubled melody line is struck with
    comparable force on both pitches, so both notes carry similar velocity
    and neither is dropped.  A transcribed partial inherits the (much
    lower) energy of the harmonic, so it shows up well below the
    fundamental's velocity.

    Eval evidence (N=50 MAESTRO): at vel_ratio=0.8 this removed 2.6k octave
    FPs but ALSO deleted ~1.3k real doublings (Basic Pitch's velocities
    aren't calibrated enough to trust a loose gate) — net F1 flat.  The
    default is therefore conservative (0.55): only drop notes at less than
    ~half the fundamental's velocity, i.e. unambiguous partials.
    """
    if not notes:
        return notes
    notes_sorted = sorted(notes, key=lambda n: float(n['onset']))
    onsets = np.array([float(n['onset']) for n in notes_sorted])
    drop: set = set()

    for i, n in enumerate(notes_sorted):
        if i in drop:
            continue
        lo = int(np.searchsorted(onsets, onsets[i] - onset_tol, side='left'))
        hi = int(np.searchsorted(onsets, onsets[i] + onset_tol, side='right'))
        for j in range(lo, hi):
            if j == i or j in drop:
                continue
            m = notes_sorted[j]
            dp = int(m['pitch']) - int(n['pitch'])
            if dp not in (12, 24):
                continue
            v_low = int(n.get('velocity', 80))
            v_up  = int(m.get('velocity', 80))
            if v_up <= vel_ratio * v_low:
                drop.add(j)

    kept = [n for k, n in enumerate(notes_sorted) if k not in drop]
    if drop:
        print(f"[transcription] octave-dedupe dropped {len(drop)} probable "
              f"harmonic partials ({len(notes)} -> {len(kept)})")
    return kept


# Musical note values in beats (quarter-note = 1 beat): 16th, 8th-triplet,
# dotted-16th, 8th, quarter-triplet, dotted-8th, quarter, dotted-quarter,
# half, dotted-half, whole.
_NOTE_VALUES_BEATS = (0.25, 1/3, 0.375, 0.5, 2/3, 0.75,
                      1.0, 1.5, 2.0, 3.0, 4.0)


def quantize_durations_to_note_values(
    notes:      List[Dict],
    beat_times: np.ndarray,
) -> List[Dict]:
    """
    Snap each note's DURATION to the nearest standard musical value, using
    the local beat period at the note's onset (so rubato/tempo drift is
    handled the same way as the onset grid).

    Onset quantization alone leaves durations at whatever the transcriber
    emitted — 0.187 s of raw model output that corresponds to no note value.
    That's the main reason sheet music renders with bizarre tied fragments,
    and a contributor to the COnP >> COnPOff gap on the eval.

    Durations beyond ~5 beats are left untouched: those are long pedal
    tones where "snap to whole note" would be a bigger distortion than the
    raw estimate.
    """
    if len(beat_times) < 2 or not notes:
        return notes
    bt = np.sort(np.asarray(beat_times, dtype=float))
    out: List[Dict] = []
    for n in notes:
        onset  = float(n['onset'])
        offset = float(n.get('offset', onset + 0.25))
        i = min(max(int(np.searchsorted(bt, onset, side='right')) - 1, 0),
                len(bt) - 2)
        period = float(bt[i + 1] - bt[i])
        if period <= 1e-3:
            out.append(n)
            continue
        dur_beats = (offset - onset) / period
        if dur_beats > _NOTE_VALUES_BEATS[-1] * 1.25:
            out.append(n)
            continue
        snapped = min(_NOTE_VALUES_BEATS, key=lambda v: abs(v - dur_beats))
        out.append({**n, 'offset': round(onset + snapped * period, 4)})
    return out


def get_octave_range(notes: List[Dict]) -> Dict[str, Any]:
    if not notes:
        return {'lowest': None, 'highest': None, 'octave_span': 0}
    pitches    = [int(n['pitch']) for n in notes]
    lowest_oct  = min(pitches) // 12 - 1
    highest_oct = max(pitches) // 12 - 1
    return {
        'lowest_midi':    min(pitches),
        'highest_midi':   max(pitches),
        'lowest_note':    librosa.midi_to_note(min(pitches)),
        'highest_note':   librosa.midi_to_note(max(pitches)),
        'lowest_octave':  lowest_oct,
        'highest_octave': highest_oct,
        'octave_span':    highest_oct - lowest_oct + 1,
    }


# ── beat-adaptive timing helpers ──────────────────────────────────────────────

def _compute_feel(notes: List[Dict], beat_times: np.ndarray) -> float:
    """
    Measure how far ahead or behind the beat this set of notes tends to land.

    For each note onset, find the nearest detected beat time and compute the
    offset (onset - beat_time).  The median of all such offsets is the "feel":
      positive → notes tend to land behind the beat
      negative → notes tend to land ahead of the beat

    Offsets beyond 500 ms are discarded (note is closer to the wrong beat).
    Returns 0.0 when there are too few notes or no beat times.
    """
    if not notes or len(beat_times) < 2:
        return 0.0

    offsets = []
    for n in notes:
        onset = float(n['onset'])
        nearest = float(beat_times[int(np.argmin(np.abs(beat_times - onset)))])
        delta = onset - nearest
        if abs(delta) < 0.5:
            offsets.append(delta)

    if not offsets:
        return 0.0
    return float(np.median(offsets))


# Gentle-quantization gate.  A snap is only applied when it moves the note
# by less than this many seconds; larger corrections are assumed to be the
# grid disagreeing with the performance (or madmom's beat being off) rather
# than the note needing fixing, and the original time is kept.
#
# Eval evidence: after the of_fullmix swap, timing_off became the #1 FP
# category (40%) and #2 FN (29%) — ~38k events that are real notes landing
# just outside the +-50 ms match window.  Unconditional snapping was a prime
# suspect for *creating* some of that.  45 ms keeps corrections inside the
# metric's own tolerance so a snap can never push a correct note out of it.
_MAX_SNAP_SHIFT_S = 0.045


def _beat_adaptive_quantize(
    notes: List[Dict],
    beat_times: np.ndarray,
    subdivisions: int = 4,
    max_shift_s: Optional[float] = _MAX_SNAP_SHIFT_S,
) -> List[Dict]:
    """
    Snap note onsets/offsets to the nearest sub-beat grid position.

    Unlike constant-tempo quantization, this uses the *actual* detected beat
    positions, so it handles tempo variation and rubato correctly.  Between
    consecutive beats the interval is divided into `subdivisions` equal slots
    (subdivisions=4 → 16th-note grid within each beat pair).

    `max_shift_s` gates each snap: onsets/offsets further than that from the
    nearest grid line are left where the transcriber put them.  Pass None to
    restore unconditional snapping (the pre-2026-08 behaviour).

    Extrapolates using the last beat interval for notes beyond the final beat.
    Falls back to leaving times unchanged if fewer than 2 beats are detected.
    """
    if len(beat_times) < 2:
        return notes

    beat_times = np.sort(beat_times).astype(float)

    def _snap(t: float) -> float:
        t = float(t)

        # Before first beat — extrapolate using first interval
        if t < beat_times[0]:
            dt   = beat_times[1] - beat_times[0]
            sub  = dt / subdivisions
            diff = t - beat_times[0]
            idx  = round(diff / sub)
            return float(beat_times[0] + idx * sub)

        # After last beat — extrapolate using last interval
        if t >= beat_times[-1]:
            dt   = beat_times[-1] - beat_times[-2]
            sub  = dt / subdivisions
            diff = t - beat_times[-1]
            idx  = round(diff / sub)
            return float(beat_times[-1] + idx * sub)

        # Within the beat grid — find containing interval
        i    = int(np.searchsorted(beat_times, t, side='right')) - 1
        i    = min(i, len(beat_times) - 2)
        dt   = beat_times[i + 1] - beat_times[i]
        sub  = dt / subdivisions
        frac = (t - beat_times[i]) / dt
        idx  = round(frac * subdivisions)
        return float(beat_times[i] + idx * sub)

    def _gated_snap(t: float) -> float:
        """Snap only if the correction is small enough to be plausible."""
        snapped = _snap(t)
        if max_shift_s is not None and abs(snapped - t) > max_shift_s:
            return t
        return snapped

    result = []
    for n in notes:
        # A note whose time was set from the SONG'S OWN activation is already
        # where the evidence says it belongs; the grid is only ever a proxy for
        # that, so re-snapping here would overwrite the better answer with the
        # worse one.  melody_decode.refine_onsets stamps these.
        if n.get('_timed'):
            result.append(dict(n))
            continue
        # Clamp to zero — extrapolation before the first beat can snap an
        # early onset negative, which downstream (mir_eval, MIDI export)
        # rejects as a malformed interval.
        onset_s  = max(0.0, _gated_snap(float(n['onset'])))
        offset_s = max(0.0, _gated_snap(float(n.get('offset', float(n['onset']) + 0.25))))

        if offset_s <= onset_s:
            # Extend by one sub-beat using the local interval
            i  = min(int(np.searchsorted(beat_times, onset_s, side='right')) - 1,
                     len(beat_times) - 2)
            dt = (beat_times[i + 1] - beat_times[i]) if i < len(beat_times) - 1 \
                 else (beat_times[-1] - beat_times[-2])
            offset_s = onset_s + dt / subdivisions

        result.append({
            **n,
            'onset':  round(onset_s,  4),
            'offset': round(offset_s, 4),
        })

    result.sort(key=lambda n: (float(n['onset']), int(n['pitch'])))
    return result


# ── layer relabelling helpers ─────────────────────────────────────────────────

def _relabel_bass_via_mask(
    notes:    List[Dict],
    mask_npz: Path,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Use the REPET-SIM bass pattern_strength array to relabel each pYIN note
    as either 'bass_pattern' (recurring figure) or 'bass_melody' (free line).

    For each note, average the pattern_strength across the frames inside its
    [onset, offset) window.  If mean >= threshold → bass_pattern, else
    bass_melody.  Notes outside the mask's coverage (or with no overlapping
    frames) default to bass_melody.
    """
    try:
        npz = np.load(str(mask_npz))
        strength = np.asarray(npz["strength"], dtype=np.float32)
        times    = np.asarray(npz["times"],    dtype=np.float32)
    except Exception as e:
        print(f"[transcription] bass mask load failed ({e}); keeping layer='bass'")
        return notes

    if strength.size == 0 or times.size == 0:
        return notes

    out: List[Dict] = []
    for n in notes:
        onset  = float(n.get('onset',  0.0))
        offset = float(n.get('offset', onset + 0.05))
        i0 = int(np.searchsorted(times, onset,  side='left'))
        i1 = int(np.searchsorted(times, offset, side='right'))
        i0 = max(0, min(i0, strength.size - 1))
        i1 = max(i0 + 1, min(i1, strength.size))
        slab = strength[i0:i1]
        m = float(slab.mean()) if slab.size else 0.0
        relabelled = dict(n)
        relabelled['layer'] = 'bass_pattern' if m >= threshold else 'bass_melody'
        relabelled['_pattern_strength'] = round(m, 3)
        out.append(relabelled)
    return out


def _stamp_pattern_idx(
    notes:         List[Dict],
    bars:          List[Dict],
    layer_filter:  str,
) -> List[Dict]:
    """
    Annotate each note in `layer_filter` with `pattern_idx` = the cluster
    index of the bar it falls in (matched by onset time).  Notes in other
    layers are passed through untouched.
    """
    if not bars:
        return notes
    # Pre-sort bars by start time for binary search
    starts = np.array([float(b['start']) for b in bars], dtype=np.float32)
    out: List[Dict] = []
    for n in notes:
        if n.get('layer') != layer_filter:
            out.append(n); continue
        onset = float(n.get('onset', 0.0))
        idx = int(np.searchsorted(starts, onset, side='right')) - 1
        if 0 <= idx < len(bars) and float(bars[idx]['end']) > onset:
            nn = dict(n)
            nn['pattern_idx'] = int(bars[idx]['cluster'])
            out.append(nn)
        else:
            out.append(n)
    return out


# ── full-mix O&F path ─────────────────────────────────────────────────────────

_BASS_SPLIT_PITCH = 56    # G#3 — matches music_theory.BASS_SPLIT

# Score-informed corrector.  Trained on 140 ASAP scores / 180k matched notes;
# on held-out pieces it cut mean |log duration error| by 39%.
#
# MEASURED ON THE REAL HARNESS (eval run 6) IT MADE THINGS MUCH WORSE:
#     COnP    0.7821 -> 0.7821   (unchanged; it touches no onsets or pitches)
#     COnPOff 0.4594 -> 0.2784   (-39% relative)
#
# The model is not broken — it was trained toward a different target than the
# metric measures.  It learned NOTATED durations from scores; the eval's
# ground truth is PERFORMANCE MIDI, and a performer detaches notes, so a
# written quarter is played shorter than a quarter.  We sit below the notated
# value on 52% of notes, so the model dutifully lengthens — correct against a
# score, wrong against a performance.
#
# Consequences, kept separate on purpose:
#   - the DURATION head is off for the transcription path, which is judged
#     against performances.  It is the right correction for sheet-music
#     export, where notated values are what should be printed — that is a
#     future use, not this one.
#   - the HAND head is unaffected by this: hand assignment is notation
#     information that has no performance counterpart to disagree with, and
#     it does not touch timing.  It beat the pitch>=C4 heuristic 0.883 to
#     0.788 and is applied.
USE_CORRECTOR = True
CORRECTOR_APPLY_DURATION = False   # see above: harmful against performance truth
CORRECTOR_APPLY_HAND     = True

# Recover repeated-note articulation — a key released and struck again reads
# as one held note.  The PROBLEM is real and measured: we emit only 46% of
# the re-strikes present in the ground truth.
#
# This DETECTOR is off because it measured worse (eval run 8):
#     COnP    0.7821 -> 0.7495
#     COnPOff 0.4594 -> 0.4282
#     0 pieces improved, 45 of 50 regressed, 12,431 attacks added
#
# It passed all five synthetic tests and still failed on real audio, which
# is the instructive part.  Every test was a single note in isolation —
# "does this held note split where it should".  Real music has dozens of
# notes sounding at once, any of which can inject energy into any other's
# harmonic bins, plus pedal resonance and room reflections doing the same.
# Precision that is perfect on one note in silence is not precision.
#
# The right approach is supervised, not hand-tuned: backend/correction/
# already contains 180k aligned (our extraction, real score) pairs, and the
# score says exactly where re-strikes we merged should have been.  That is
# labelled training data for this precise question, and it does not require
# me to guess spectral thresholds.
USE_RESTRIKE_DETECTION = False


# 'ml' | 'projection' | 'of'.  Override with ARRANGEMENT_SONG_MODE for an A/B.
#
# 'ml' is the trained transcriber and is the default because it is measured
# better on ground truth: on 8 held-out Slakh tracks, note F1 0.630 against the
# projection's 0.247, with a chance floor of 0.078.  The projection must explain
# EVERY scrap of energy it is handed — 88 piano templates are its whole
# vocabulary — so a voice or a cymbal comes back as piano notes and no
# downstream filter undoes that.  A trained model is shown examples where the
# right answer is mostly nothing, so it can decline.
#
# It also needs no source separation: Slakh's mixes contain drums while its
# targets exclude them, so ignoring percussion is what it was trained to do.
# Demucs still runs, because the beat grid comes from that step.
SONG_MODE = 'ml'

# Solo-piano note source: 'bytedance' (default, checkpoint note_F1=0.9677) or
# 'spectral' (the pitch-aligned patch transformer, app/ml/spectral).
import os as _os
PIANO_MODE = _os.environ.get("ARRANGEMENT_PIANO_MODE", "bytedance")
PIANO_GRID_SNAP = _os.environ.get("ARRANGEMENT_PIANO_GRID_SNAP") == "1"
# Song-to-piano arranger (spectral, difficulty-conditioned).  Set the checkpoint
# to use it for songs; with ARRANGEMENT_PIANO_ONLY the song skips Demucs and
# reaches this branch on the original mix.  Threshold provisional until the v1
# evaluation tunes it on validation songs.
SONG_ARRANGER = _os.environ.get("ARRANGEMENT_SONG_ARRANGER") or None
SONG_ARRANGER_LEVEL = int(_os.environ.get("ARRANGEMENT_SONG_LEVEL", "2"))
SONG_ARRANGER_THR = float(_os.environ.get("ARRANGEMENT_SONG_ARRANGER_THR", "0.6"))
# Note length: at the transcriber's 0.5 frame threshold v1 arrangements came out
# about half the length of the human ones (median 0.08-0.17 s vs 0.19-0.24 s on
# three songs); 0.3 lands near the human lengths.
SONG_ARRANGER_FRAME_THR = float(_os.environ.get("ARRANGEMENT_SONG_ARRANGER_FRAME_THR", "0.3"))


def _song_method(session_dir: Path, meta: Dict) -> str:
    """Which note source to use: 'ml', 'projection' or 'of'.

    Solo piano always goes to O&F regardless of mode.  It is trained on
    MAESTRO, scores note_F1 0.9677 on exactly that material, and neither the
    projection nor the Slakh-trained transcriber comes close on solo piano —
    the transcriber has never seen an isolated piano recording in training.

    The piano/song decision is separation's, not a second opinion:
    content_detect routes solo piano past Demucs and stamps 'raw+madmom', so
    anything that ran Demucs is a song.  One detector that can be wrong beats
    two that can disagree.  Measured on real uploads, flatness 0.000184 for
    solo piano against 0.026551 for a full mix, a 144x separation either side
    of the 0.005 threshold.
    """
    import os
    mode = os.environ.get('ARRANGEMENT_SONG_MODE', SONG_MODE).lower()
    if meta.get('separation_skipped') or meta.get('method') == 'raw+madmom':
        return 'of'         # solo piano
    # Experiment switch: every upload, songs included, goes through the solo
    # piano transcriber (the spectral model when ARRANGEMENT_PIANO_MODE is
    # 'spectral'), with no projection and no song-path rules.
    if os.environ.get('ARRANGEMENT_PIANO_ONLY') == '1':
        return 'of'
    if mode == 'ml':
        from app.services import ml_transcription
        if ml_transcription.available():
            return 'ml'
        print("[transcription] no transcriber checkpoint; using the projection")
        return 'projection'
    return mode if mode in ('projection', 'of') else 'projection'


def _transcribe_fullmix_of(
    session_dir: Path,
    comp_dir:    Path,
    tempo:       float,
    beat_times:  np.ndarray,
    pyin_bass_notes: List[Dict],     # unused by design — see docstring
    meta:        Dict | None = None,
) -> tuple:
    """
    'of_fullmix' architecture: ONE polyphonic transcriber (O&F) over the
    original audio, hands/layers assigned symbolically afterwards.

    Why not merge pYIN's bass in: the whole point of this mode is a single
    timing character across the full range — mixing a second transcriber's
    onsets back in would reintroduce the LH/RH inconsistency this mode
    exists to fix.  pYIN's notes are accepted as a parameter only so the
    caller doesn't need to know which mode consumes them.

    Layer assignment (symbolic, replaces the audio-stem routing):
      pitch < 56  → bass; relabelled bass_pattern / bass_melody via the
                    REPET-SIM mask when separation produced one
      pitch >= 56 → pattern / melody via the phase-7 classifier when its
                    checkpoint exists, else 'melody'
    """
    audio_path = session_dir / 'audio.wav'
    if not audio_path.exists():
        raise FileNotFoundError('audio.wav missing for full-mix transcription.')

    # Songs and solo piano get different transcribers, because they are
    # different questions.  O&F is trained on MAESTRO and scores note F1
    # 0.9677 there — nothing beats it on piano.  Pointed at a band it is still
    # asking "what piano notes are IN this audio", which for a guitar chord or
    # a vocal line has no right answer.  piano_projection asks "what piano
    # notes best APPROXIMATE this audio" instead, which does.
    #
    # The gate is separation's own verdict: content_detect routes solo piano
    # past Demucs and stamps method 'raw+madmom', so anything that actually ran
    # Demucs is a song.  Reusing that decision means there is one detector to
    # be wrong rather than two that can disagree.
    method = _song_method(session_dir, meta or {})
    print(f"[transcription] note source: {method}")
    use_projection = method != 'of'

    if method == 'ml':
        try:
            from app.services import ml_transcription
            # The RAW mix, drums and all.
            #
            # The model was trained on mixes CONTAINING drums with drums
            # excluded from the target, so ignoring percussion is the task it
            # was taught rather than something to arrange for it in advance.
            # On ground truth the two agree: 6 held-out Slakh tracks give note
            # F1 0.625 on the raw mix against 0.629 drumless, indistinguishable
            # from noise.
            #
            # This briefly ran on Demucs' drumless mix instead, on the strength
            # of one track sounding mildly better by ear.  That was the right
            # kind of evidence to take seriously — Slakh's drums are
            # SYNTHESISED from MIDI, so it cannot test whether "ignore drums"
            # transfers to a real kit with cymbal wash and room bleed — but one
            # track is one track, and it did not reproduce.  The two outputs
            # share only 62% of their notes while differing with no drum
            # signature at all (raw median pitch 57, drumless 59, the bulk of
            # the disagreement mid-register either way), which is the shape of
            # Demucs artefacts rather than of drums removed.
            #
            # Going back to the raw mix also stops Demucs being load-bearing
            # here: on the song path separation now feeds only the projection
            # fallback, so skipping it is once again available as a speed win.
            src = session_dir / "audio.wav"
            notes = ml_transcription.transcribe(src)
            print(f"[transcription] transcriber: {len(notes)} notes "
                  f"from {src.name}")
        except Exception as e:                              # noqa: BLE001
            print(f"[transcription] transcriber failed ({type(e).__name__}: "
                  f"{e}); falling back to the projection")
            method = 'projection'

    if method == 'projection':
        try:
            from app.services import piano_projection
            notes = piano_projection.project_from_stems(comp_dir)
            print(f"[transcription] projection: {len(notes)} notes "
                  f"from the drumless stems")
        except Exception as e:                              # noqa: BLE001
            # A song transcribed by O&F is mediocre; a song that 500s is
            # useless.  Fall back rather than fail the request.
            print(f"[transcription] projection failed ({type(e).__name__}: {e}); "
                  f"falling back to O&F")
            use_projection = False

    if not use_projection:
        notes = None
        # Opt-in: the spectral patch transformer on the solo-piano path.
        # ByteDance stays the default until the new model has been measured
        # against it; the new one was trained on rendered audio only, so a
        # real upload here is its first real-audio test.
        if method == 'of' and PIANO_MODE == 'spectral':
            try:
                from app.services import spectral_transcription
                global _last_pedal, _last_source
                content = (meta or {}).get("content") or {}
                # ARRANGEMENT_FORCE_SONG=1: every upload is a song (the solo-piano
                # detector misreads piano-heavy songs — flatness 0.0025 on one)
                force_song = _os.environ.get("ARRANGEMENT_FORCE_SONG") == "1"
                if SONG_ARRANGER and (force_song or content.get("piano_only") is False):
                    # Tagged spectral too: the corrector's duration head was
                    # trained on ByteDance output and, untagged, shortened
                    # every arranged note (2,356 of 2,356 on the first song).
                    _last_pedal, _last_source = [], "spectral"
                    # A song, heard by the song-to-piano arranger at the
                    # requested difficulty (0 beginner, 1 intermediate,
                    # 2 advanced).  Solo piano stays on the transcriber.
                    if _os.environ.get("ARRANGEMENT_SONG_ARRANGER_KIND") == "hum":
                        # song -> song hum -> imagined piano hum -> stage-1 reader
                        from app.services import hum_pipeline
                        notes = hum_pipeline.transcribe(audio_path)
                    elif _os.environ.get("ARRANGEMENT_SONG_ARRANGER_KIND") == "audio_arranger":
                        # structure-aware: frozen v1 ears + grammar decoder
                        from app.services import audio_arranger_service
                        notes = audio_arranger_service.transcribe(audio_path, level=SONG_ARRANGER_LEVEL)
                    else:
                        notes = spectral_transcription.transcribe(
                            audio_path, onset_thr=SONG_ARRANGER_THR,
                            ckpt=SONG_ARRANGER, cond=SONG_ARRANGER_LEVEL,
                            frame_thr=SONG_ARRANGER_FRAME_THR)
                    print(f"[transcription] song-to-piano arranger "
                          f"(level {SONG_ARRANGER_LEVEL}, thr {SONG_ARRANGER_THR})")
                elif spectral_transcription.available():
                    notes = spectral_transcription.transcribe(audio_path)
                    if _os.environ.get("ARRANGEMENT_SCALE_FILL") == "1" and notes:
                        # hold + sliding-window key drop + forward fill (app/services/scale_fill.py)
                        from app.services import scale_fill as _sf
                        notes = _sf.apply_to_file(notes, audio_path)
                    _last_pedal = []        # no pedal head: notes end at key-up
                    _last_source = "spectral"
                    print(f"[transcription] spectral: {len(notes)} notes")
            except Exception as e:                          # noqa: BLE001
                print(f"[transcription] spectral failed ({type(e).__name__}: {e}); "
                      f"falling back to ByteDance")
                notes = None
        if notes is None:
            _last_source = "bytedance"
            notes = _onsets_frames_notes(audio_path)
            print(f"[transcription] of_fullmix: {len(notes)} notes from full mix")

    # Repeated-note recovery.  Measured against ground truth: we produce only
    # 46% of the re-strikes real performances contain, because a key struck
    # again while still ringing gives a small transient against a large
    # background and the onset head misses it.  restrike.py asks the much
    # narrower question — is energy being re-injected into THIS pitch's own
    # harmonics, part-way through a note we already found — which is
    # answerable where general polyphonic onset detection is not.
    # Restrike detection asks whether energy is being re-injected into a note
    # O&F already found.  The projection produces its own re-strikes by
    # splitting activation runs, so running this on top would double-count.
    if USE_RESTRIKE_DETECTION and not use_projection:
        try:
            from app.services.restrike import detect_restrikes
            y, sr_r = librosa.load(str(audio_path), sr=22050, mono=True)
            notes = detect_restrikes(y, sr_r, notes)
        except Exception as e:
            print(f"[transcription] restrike detection skipped: {e}")

    bass_notes  = [dict(n, layer='bass') for n in notes
                   if int(n['pitch']) < _BASS_SPLIT_PITCH]
    upper_notes = [dict(n) for n in notes
                   if int(n['pitch']) >= _BASS_SPLIT_PITCH]

    bass_mask_path = comp_dir / "bass_pattern_mask.npz"
    if bass_mask_path.exists():
        bass_notes = _relabel_bass_via_mask(bass_notes, bass_mask_path)

    try:
        from app.ml.pattern_classifier import classify_notes, model_available
        if model_available():
            upper_notes = classify_notes(upper_notes, tempo=tempo)
    except Exception as e:
        print(f"[transcription] pattern_classifier skipped: {e}")
    for n in upper_notes:
        n.setdefault('layer', 'melody')

    # pattern_idx stamping — same contract as the stem path
    clusters_path = session_dir / "pattern_clusters.json"
    if clusters_path.exists():
        try:
            with open(clusters_path) as f:
                cluster_data = json.load(f)
            upper_notes = _stamp_pattern_idx(
                upper_notes, cluster_data.get("pattern", {}).get("bars", []),
                layer_filter="pattern")
            bass_notes = _stamp_pattern_idx(
                bass_notes, cluster_data.get("bass_pattern", {}).get("bars", []),
                layer_filter="bass_pattern")
        except Exception as e:
            print(f"[transcription] pattern_idx stamping skipped: {e}")

    bass_feel  = _compute_feel(bass_notes,  beat_times)
    upper_feel = _compute_feel(upper_notes, beat_times)

    # Same post-processing tail as the stem path.  The cross-layer dedupe is
    # unnecessary here (single transcriber can't double-emit across stems)
    # but harmless, and keeping the tail identical makes mode comparisons
    # apples-to-apples.
    all_notes = sorted(bass_notes + upper_notes,
                       key=lambda n: (float(n['onset']), int(n['pitch'])))
    refined = refine_notes(all_notes, tempo)
    refined = dedupe_octave_partials(refined)

    # Solo piano keeps the transcriber's timing.  The 16th-note snap moved 67%
    # of notes (a third by >30 ms) on 12 real MAESTRO recordings and took F1
    # at a 20 ms window from 0.936 to 0.485: trills and runs faster than 16ths
    # were pulled onto grid lines and drifted audibly against the original
    # (eval/post_timing_check.py).  Notation export quantizes on its own.
    if method == 'of' and not PIANO_GRID_SNAP:
        quantized = refined
    elif len(beat_times) >= 2:
        quantized = _beat_adaptive_quantize(refined, beat_times, subdivisions=4)
    else:
        quantized = quantize_notes(refined, tempo, subdivisions=16)

    # Score-informed correction.  Runs last, on final timings, because it was
    # trained on the output of exactly this pipeline — feeding it anything
    # earlier would show it a distribution it never saw.
    if USE_CORRECTOR:
        try:
            from app.ml.corrector import correct_notes, available
            if available():
                # The duration head was trained on THIS pipeline's ByteDance
                # output and pulls lengths toward what that model gets wrong.
                # On the spectral model's notes, measured against true key
                # releases (60 rendered clips), it took offset-aware F1 from
                # 0.811 to 0.397 and median length from 1.00x to 1.57x.  Hand
                # assignment is unaffected and stays on.
                dur_ok = CORRECTOR_APPLY_DURATION and (
                    _last_source != "spectral"
                    or _os.environ.get("ARRANGEMENT_SPECTRAL_CORRECTOR") == "1")
                quantized = correct_notes(
                    quantized, beat_times,
                    apply_duration=dur_ok,
                    apply_hand=CORRECTOR_APPLY_HAND)
        except Exception as e:
            print(f"[transcription] corrector skipped: {e}")

    return quantized, {'bass': round(bass_feel, 4), 'upper': round(upper_feel, 4),
                       'pedal': list(_last_pedal)}


# ── main entry point ──────────────────────────────────────────────────────────

def transcribe_components(
    session_dir: Path,
) -> tuple:
    """
    Full per-component transcription pipeline.

    Reads:
      session_dir/components/bass.wav
      session_dir/components/upper.wav
      session_dir/components/H.npy
      session_dir/components.json   (component metadata from separation)

    Returns (notes, timing_feel) where:
      notes        — merged, labeled list sorted by (onset, pitch);
                     each note has a 'layer' field: 'bass' | 'pattern' | 'melody'
      timing_feel  — {'bass': float, 'upper': float} median offset in seconds
                     from the nearest beat for each register (positive = behind beat)
    """
    comp_dir     = session_dir / 'components'
    bass_path    = comp_dir / 'bass.wav'
    upper_path   = comp_dir / 'upper.wav'
    pattern_path = comp_dir / 'pattern.wav'
    melody_path  = comp_dir / 'melody.wav'
    meta_path    = session_dir / 'components.json'

    if not bass_path.exists() or not upper_path.exists():
        raise FileNotFoundError('Component audio files missing. Run /separate first.')

    with open(meta_path) as f:
        meta = json.load(f)

    tempo      = float(meta.get('tempo_bpm', 120.0))
    hop_length = int(meta.get('hop_length', 512))
    sr_comp    = int(meta.get('sr', 22050))

    # Actual beat positions saved during separation for beat-adaptive quantization
    raw_beat_frames = meta.get('beat_frames', [])
    beat_times = (
        librosa.frames_to_time(np.array(raw_beat_frames), sr=sr_comp, hop_length=hop_length)
        if raw_beat_frames else np.array([])
    )

    # ── bass: pYIN ────────────────────────────────────────────────────────────
    y_bass, sr_bass = librosa.load(str(bass_path), sr=sr_comp, mono=True)
    bass_notes      = _pyin_to_notes(y_bass, sr_bass, hop_length=hop_length)

    # If separation.py produced a REPET-SIM bass pattern_strength mask,
    # relabel pYIN notes as bass_pattern / bass_melody instead of plain
    # 'bass'.  pYIN remains the transcriber (rock-solid mono F0); REPET
    # only decides layer.
    bass_mask_path = comp_dir / "bass_pattern_mask.npz"
    if bass_mask_path.exists():
        bass_notes = _relabel_bass_via_mask(bass_notes, bass_mask_path)
        n_p = sum(1 for n in bass_notes if n.get('layer') == 'bass_pattern')
        n_m = sum(1 for n in bass_notes if n.get('layer') == 'bass_melody')
        print(f"[transcription] bass-layer split via REPET mask: "
              f"bass_pattern={n_p}, bass_melody={n_m}")

    # ── upper transcription ───────────────────────────────────────────────────
    # Transcriber selection.  The earlier "O&F didn't pay off" verdict was
    # made by ear, before the eval harness existed; we're now re-testing it
    # with measurement (see backend/eval/).  O&F has a dedicated offset head
    # (Basic Pitch doesn't) and was trained on MAESTRO — the eval's exact
    # domain — so it's the prime candidate for fixing our recall (24%
    # under-emission) and duration (COnPOff 0.10) weaknesses.
    # Transcriber architecture (measured on the N=50 eval harness):
    #   'basic_pitch_stems' — Basic Pitch on the REPET pattern/melody stems +
    #                         pYIN bass.  COnP 0.4085 (run-1 baseline family).
    #   'of_stems'          — O&F on the same stems.  COnP 0.380 — REGRESSION.
    #                         O&F is trained on full-mix piano; Wiener-masked
    #                         stems are out-of-domain for it and it double-
    #                         transcribes stem bleed (+8.6% over-emission).
    #   'of_fullmix'        — O&F once over the original audio (its native
    #                         domain), hands/layers assigned SYMBOLICALLY
    #                         afterwards.  One transcriber = one timing
    #                         character = coherent hands; also the fix for
    #                         the audible LH/RH inconsistency.
    TRANSCRIBER_MODE = 'of_fullmix'

    if TRANSCRIBER_MODE == 'of_fullmix':
        return _transcribe_fullmix_of(
            session_dir, comp_dir, tempo, beat_times, bass_notes, meta,
        )

    _upper_transcriber = (_onsets_frames_notes if TRANSCRIBER_MODE == 'of_stems'
                          else _basic_pitch_notes)

    # If separation.py wrote REPET-SIM pattern + melody stems, transcribe each
    # independently and pre-label their notes.  pattern_extraction.py then
    # routes by these audio-level labels rather than running the pattern
    # classifier or cycle-mining on the merged upper.wav.  Falls back to the
    # combined upper.wav + classifier path when the stems aren't present.
    if pattern_path.exists() and melody_path.exists():
        pattern_notes = _upper_transcriber(pattern_path)
        for n in pattern_notes:
            n['layer'] = 'pattern'
        melody_notes_audio = _upper_transcriber(melody_path)
        for n in melody_notes_audio:
            n['layer'] = 'melody'
        upper_notes = pattern_notes + melody_notes_audio
        print(f"[transcription] audio-level layer split: "
              f"pattern={len(pattern_notes)}, melody={len(melody_notes_audio)}")
    else:
        upper_notes = _upper_transcriber(upper_path)
        try:
            from app.ml.pattern_classifier import classify_notes, model_available
            if model_available():
                upper_notes = classify_notes(upper_notes, tempo=tempo)
        except Exception as e:
            print(f"[transcription] pattern_classifier skipped: {e}")

    # ── stamp pattern_idx (cluster) on pattern + bass_pattern notes ───────────
    # pattern_clusters.json was written by separation.py and maps each bar
    # window to a cluster index (Pattern 1..N).  The frontend Layers tab
    # uses pattern_idx to split each pattern layer into per-cluster rows.
    clusters_path = session_dir / "pattern_clusters.json"
    if clusters_path.exists():
        try:
            with open(clusters_path) as f:
                cluster_data = json.load(f)
            upper_notes = _stamp_pattern_idx(
                upper_notes,
                cluster_data.get("pattern", {}).get("bars", []),
                layer_filter="pattern",
            )
            bass_notes = _stamp_pattern_idx(
                bass_notes,
                cluster_data.get("bass_pattern", {}).get("bars", []),
                layer_filter="bass_pattern",
            )
        except Exception as e:
            print(f"[transcription] pattern_idx stamping skipped: {e}")

    # ── measure per-layer feel before any quantization ────────────────────────
    # Raw pYIN and Basic Pitch onsets reveal how far ahead/behind the beat each
    # register tends to land.  This is applied to generated notes later so that
    # walking bass and pYIN fill inherit the same timing character.
    bass_feel  = _compute_feel(bass_notes,  beat_times)
    upper_feel = _compute_feel(upper_notes, beat_times)

    # ── merge + deduplicate ───────────────────────────────────────────────────
    all_notes = sorted(
        bass_notes + upper_notes,
        key=lambda n: (float(n['onset']), int(n['pitch'])),
    )
    deduped: List[Dict] = []
    for n in all_notes:
        if any(
            abs(float(n['onset']) - float(x['onset'])) < 0.03
            and abs(int(n['pitch']) - int(x['pitch'])) <= 2
            for x in deduped[-40:]
        ):
            continue
        deduped.append(n)

    refined = refine_notes(deduped, tempo)

    # ── octave-partial dedupe ─────────────────────────────────────────────────
    # Runs after cross-layer dedupe so all layers' notes are visible to the
    # +12/+24 semitone comparison.  Velocity-gated: real octave doublings
    # survive, weak harmonic phantoms get dropped.
    refined = dedupe_octave_partials(refined)

    # ── beat-adaptive quantization ────────────────────────────────────────────
    # Use actual beat frame positions (not a computed constant-tempo lattice) so
    # that tempo variation and rubato in the performance are handled correctly.
    if len(beat_times) >= 2:
        quantized = _beat_adaptive_quantize(refined, beat_times, subdivisions=4)
        # Duration snap to musical note values: measured on the N=50 eval as
        # F1-neutral-to-slightly-negative (COnPOff 0.100 -> 0.099) — the
        # transcriber's durations are systematically short, so snapping a
        # wrong duration to a clean note value keeps it wrong.  OFF by
        # default; flip for sheet-music-focused rendering where visual note
        # values matter more than offset accuracy.
        QUANTIZE_DURATIONS = False
        if QUANTIZE_DURATIONS:
            quantized = quantize_durations_to_note_values(quantized, beat_times)
    else:
        quantized = quantize_notes(refined, tempo, subdivisions=16)

    timing_feel = {
        'bass':  round(bass_feel,  4),
        'upper': round(upper_feel, 4),
    }

    return quantized, timing_feel
