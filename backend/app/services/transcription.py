from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile
import numpy as np
import librosa
import soundfile as sf


def _midi_data_to_notes(midi_data) -> List[Dict[str, Any]]:
    """Convert a pretty_midi MidiFile (from Basic Pitch) to note dicts."""
    notes = []
    if not midi_data.instruments:
        return notes
    for note in midi_data.instruments[0].notes:
        notes.append({
            "pitch":      int(note.pitch),
            "pitch_name": librosa.midi_to_note(int(note.pitch)),
            "onset":      float(note.start),
            "offset":     float(note.end),
            "velocity":   int(note.velocity),
        })
    return notes


def _separate_stems(wav_path: Path):
    """
    Split audio into bass (< 260 Hz) and melody (> 260 Hz) stems using
    STFT masking with a soft sigmoid transition. Returns (bass_audio, melody_audio, sr).
    """
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    D = librosa.stft(y, n_fft=4096, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    cutoff_hz  = 260.0
    cutoff_bin = int(np.searchsorted(freqs, cutoff_hz))
    trans_bins = 30  # soft transition over ~30 bins

    mask_bass = np.zeros(len(freqs), dtype=np.float32)
    mask_bass[:max(0, cutoff_bin - trans_bins)] = 1.0
    for i in range(trans_bins):
        idx = cutoff_bin - trans_bins + i
        if 0 <= idx < len(freqs):
            mask_bass[idx] = 1.0 - (i / trans_bins)

    mask_melody = 1.0 - mask_bass

    bass_audio   = librosa.istft(D * mask_bass[:, None],   hop_length=512).astype(np.float32)
    melody_audio = librosa.istft(D * mask_melody[:, None], hop_length=512).astype(np.float32)

    for stem in [bass_audio, melody_audio]:
        peak = np.abs(stem).max()
        if peak > 0:
            stem /= peak

    return bass_audio, melody_audio, sr


_CHORD_TEMPLATES = [
    [0, 4, 7],        # major
    [0, 3, 7],        # minor
    [0, 4, 7, 10],    # dom7
    [0, 3, 7, 10],    # min7
    [0, 4, 7, 11],    # maj7
    [0, 3, 6],        # dim
    [0, 5, 7],        # sus4
    [0, 2, 7],        # sus2
    [0, 3, 6, 9],     # dim7
    [0, 4, 8],        # aug
]

# Interval from chord root → bass priority (lower = more preferred)
# Bass most naturally plays root → fifth → octave → third
_BASS_INTERVAL_PRIORITY = {0: 0, 7: 1, 12: 0, 5: 2, 4: 3, 3: 3, 10: 4, 11: 4}


def _infer_measure_chords(notes: List[Dict], tempo: float) -> Dict[int, tuple]:
    """
    Per-measure chord inference (4-beat window) from a note list.
    Returns {measure_idx: (root_pc, chord_pcs_set)}.

    Using a full measure window instead of single beats gives much more
    pitch-class evidence, leading to more reliable chord detection — which
    in turn makes bass snapping significantly more accurate.
    """
    if not notes or tempo <= 0:
        return {}

    measure_dur = (60.0 / tempo) * 4   # assume 4/4
    measure_pcs: Dict[int, set] = {}

    for n in notes:
        mi = int(float(n["onset"]) / measure_dur)
        measure_pcs.setdefault(mi, set()).add(int(n["pitch"]) % 12)

    result: Dict[int, tuple] = {}
    for mi, pcs in measure_pcs.items():
        best_score, best_root, best_cpcs = -1.0, 0, pcs
        for ivs in _CHORD_TEMPLATES:
            for root in range(12):
                cpcs = {(root + i) % 12 for i in ivs}
                hits = len(pcs & cpcs)
                score = 0.6 * (hits / len(ivs)) + 0.4 * (hits / max(len(pcs), 1))
                if score > best_score:
                    best_score, best_root, best_cpcs = score, root, cpcs
        result[mi] = (best_root, best_cpcs)
    return result


def _chroma_per_measure_from_notes(notes: List[Dict], tempo: float) -> np.ndarray:
    """
    Build a velocity-weighted (n_measures, 12) chroma matrix from a note list.
    Each row is L1-normalised so rows are comparable across different densities.
    """
    if not notes or tempo <= 0:
        return np.zeros((0, 12), dtype=np.float32)

    measure_dur = (60.0 / tempo) * 4
    max_time    = max(float(n["onset"]) for n in notes)
    n_measures  = max(1, int(np.ceil((max_time + measure_dur) / measure_dur)))

    chroma = np.zeros((n_measures, 12), dtype=np.float32)
    for n in notes:
        mi = int(float(n["onset"]) / measure_dur)
        if 0 <= mi < n_measures:
            pc  = int(n["pitch"]) % 12
            vel = float(n.get("velocity", 80)) / 127.0
            chroma[mi, pc] += vel

    row_sums = chroma.sum(axis=1, keepdims=True)
    chroma  /= np.where(row_sums > 0, row_sums, 1.0)
    return chroma


def _detect_sections(chroma_measures: np.ndarray, min_section: int = 4) -> List[Tuple[int, int]]:
    """
    Novelty-curve section detection on a (n_measures, 12) chroma matrix.
    Uses a kernel-based novelty function: cosine distance between context windows
    before and after each measure boundary. Peaks become section boundaries.
    Returns list of (start_measure, end_measure) pairs.
    """
    n = len(chroma_measures)
    if n <= min_section * 2:
        return [(0, n)]

    k = min(4, n // 4)
    novelty = np.zeros(n, dtype=np.float32)
    for i in range(k, n - k):
        before = chroma_measures[max(0, i - k):i].mean(axis=0)
        after  = chroma_measures[i:min(n, i + k)].mean(axis=0)
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb > 0 and na > 0:
            novelty[i] = 1.0 - float(np.dot(before, after) / (nb * na))

    threshold  = max(0.12, float(np.percentile(novelty[k:n - k], 65)))
    boundaries = [0]
    i = k
    while i < n - k:
        window = novelty[max(0, i - 2): min(n, i + 3)]
        is_peak = (novelty[i] >= threshold and novelty[i] == window.max())
        if is_peak and (i - boundaries[-1]) >= min_section:
            boundaries.append(i)
            i += min_section
        else:
            i += 1
    boundaries.append(n)

    return [(boundaries[j], boundaries[j + 1]) for j in range(len(boundaries) - 1)]


def _chroma_to_chord(chroma: np.ndarray) -> Tuple[int, set]:
    """Fit chord templates to a chroma vector. Returns (root_pc, chord_pcs_set)."""
    best_score, best_root, best_pcs = -1.0, 0, {0, 4, 7}
    total = max(float(chroma.sum()), 1e-8)
    for ivs in _CHORD_TEMPLATES:
        for root in range(12):
            cpcs  = {(root + i) % 12 for i in ivs}
            hits  = sum(float(chroma[(root + i) % 12]) for i in ivs)
            score = hits / total
            if score > best_score:
                best_score, best_root, best_pcs = score, root, cpcs
    return best_root, best_pcs


def _vote_section_progression(
    sec_chroma: np.ndarray,
    note_chords: Dict[int, tuple],
    sec_start: int,
) -> List[tuple]:
    """
    For a section with per-measure chroma (M, 12), try repeating period lengths
    2, 4, 8 and pick the one whose averaged positions best reconstruct each
    repetition (highest mean cosine similarity).

    Within the winning period each position's chord is resolved by fitting
    templates to the averaged chroma, then cross-checked against note-derived
    chords from the same position across all repetitions (majority root vote).

    Returns a list of (root_pc, chord_pcs_set) of length M, one per measure.
    """
    M = len(sec_chroma)
    if M == 0:
        return []
    if M == 1:
        root, pcs = _chroma_to_chord(sec_chroma[0])
        if sec_start in note_chords:
            nr, np_ = note_chords[sec_start]
            if nr == root:
                pcs = pcs | np_
        return [(root, pcs)]

    best_score, best_period, best_voted = -1.0, 1, [sec_chroma.mean(axis=0)]

    for period in [2, 4, 8]:
        if period > M:
            continue
        n_full  = (M // period) * period
        stacked = sec_chroma[:n_full].reshape(-1, period, 12)   # (reps, period, 12)
        voted   = stacked.mean(axis=0)                           # (period, 12)

        sims = []
        for r in range(stacked.shape[0]):
            for p in range(period):
                a, b = stacked[r, p], voted[p]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0 and nb > 0:
                    sims.append(float(np.dot(a, b) / (na * nb)))

        score = float(np.mean(sims)) if sims else 0.0
        score += 0.005 * float(np.log2(max(period, 1)))   # small bonus for longer periods

        if score > best_score:
            best_score, best_period, best_voted = score, period, [voted[p] for p in range(period)]

    # Resolve each position to a chord, blending audio-derived and note-derived evidence
    base: List[tuple] = []
    for pos, chroma_vec in enumerate(best_voted):
        audio_root, audio_pcs = _chroma_to_chord(chroma_vec)

        # Collect note-based root votes across all repetitions of this position
        note_root_votes: Dict[int, int] = {}
        note_pcs_union: set = set()
        for rep in range(M // best_period):
            mi = sec_start + rep * best_period + pos
            if mi in note_chords:
                nr, np_ = note_chords[mi]
                note_root_votes[nr] = note_root_votes.get(nr, 0) + 1
                note_pcs_union |= np_

        if note_root_votes:
            top_nr  = max(note_root_votes, key=note_root_votes.__getitem__)
            top_cnt = note_root_votes[top_nr]
            n_reps  = max(1, M // best_period)
            if top_nr == audio_root:
                # Both agree — merge pitch-class sets for richer snapping
                base.append((audio_root, audio_pcs | note_pcs_union))
            elif top_cnt >= max(1, n_reps // 2):
                # Note-based majority overrides (usually more harmonically detailed)
                note_pcs = {(top_nr + i) % 12 for i in _CHORD_TEMPLATES[0]}
                base.append((top_nr, note_pcs | note_pcs_union))
            else:
                base.append((audio_root, audio_pcs))
        else:
            base.append((audio_root, audio_pcs))

    return [base[i % best_period] for i in range(M)]


def _infer_section_progressions(
    melody_notes: List[Dict],
    tempo: float,
) -> Dict[int, tuple]:
    """
    Two-layer chord inference for robust bass snapping:

    Layer 1 — structural section detection
        Build a per-measure chroma matrix from melody notes and run a
        kernel-based novelty curve to find section boundaries (Part 1,
        Part 2, …).  Each section is treated independently so a verse
        chord progression doesn't bleed into the chorus.

    Layer 2 — progression pattern voting
        Within each section try repeating period lengths 2, 4, 8 measures.
        Stack all repetitions and average per position, then fit chord
        templates to the voted average.  Cross-check against single-measure
        note-based chords (from _infer_measure_chords) and merge or override
        by majority root vote.

    Result is far more stable than per-measure inference because a single
    noisy measure is just one vote in many.

    Returns {measure_idx: (root_pc, chord_pcs_set)}.
    """
    if not melody_notes or tempo <= 0:
        return {}

    chroma_measures = _chroma_per_measure_from_notes(melody_notes, tempo)
    if len(chroma_measures) == 0:
        return _infer_measure_chords(melody_notes, tempo)

    note_chords = _infer_measure_chords(melody_notes, tempo)
    sections    = _detect_sections(chroma_measures)

    result: Dict[int, tuple] = {}
    for sec_start, sec_end in sections:
        sec_chroma  = chroma_measures[sec_start:sec_end]
        progression = _vote_section_progression(sec_chroma, note_chords, sec_start)
        for offset, chord_info in enumerate(progression):
            result[sec_start + offset] = chord_info

    return result


def _generate_bass_from_percussion(
    bass_audio: np.ndarray, sr: int, tempo: float,
    melody_notes: List[Dict]
) -> List[Dict[str, Any]]:
    """
    Generate bass notes from the percussion/onset rhythm in the bass stem.

    Algorithm:
      1. Detect onsets via librosa onset detection on the bass stem
      2. Run librosa.pyin() over the full stem for frame-by-frame monophonic
         pitch tracking (pYIN is probabilistic YIN — far more accurate for
         bass frequencies than CQT peak-picking).
      3. At each onset, take the median voiced pYIN pitch from a short window
         (~6 frames ≈ 140 ms) right after the attack.
      4. Map energy → preferred register (heavy hit → lower root, light → upper)
      5. Snap to nearest chord tone inferred from the melody part.
    """
    if len(bass_audio) == 0 or tempo <= 0:
        return []

    HOP  = 512
    BASS_LOW, BASS_HIGH = 28, 52   # E1 – E3

    # ── onset detection ───────────────────────────────────────────────────────
    onset_frames = librosa.onset.onset_detect(
        y=bass_audio, sr=sr, units='frames', hop_length=HOP,
        backtrack=True, pre_max=3, post_max=3,
        pre_avg=3, post_avg=5, delta=0.07, wait=10,
    )
    if len(onset_frames) == 0:
        return []
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)

    # ── pYIN pitch tracking ───────────────────────────────────────────────────
    # pYIN returns (f0_hz, voiced_flag, voiced_probs) per frame.
    # fmin/fmax constrain it to the bass register (E1=41.2 Hz, E3=164.8 Hz).
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            bass_audio,
            fmin=librosa.note_to_hz('E1'),
            fmax=librosa.note_to_hz('E3'),
            sr=sr,
            hop_length=HOP,
            frame_length=4096,
            fill_na=None,      # NaN for unvoiced frames
        )
    except Exception:
        f0 = None

    # Convert voiced f0 frames to MIDI (NaN stays NaN)
    if f0 is not None:
        with np.errstate(invalid='ignore'):
            midi_f0 = np.where(
                voiced_flag & (f0 > 0),
                librosa.hz_to_midi(np.where(f0 > 0, f0, 1.0)),
                np.nan,
            )
    else:
        midi_f0 = None

    # ── energy envelope for register preference ───────────────────────────────
    rms = librosa.feature.rms(y=bass_audio, hop_length=HOP)[0]
    rms_max = rms.max() + 1e-8

    # ── chord map from melody (section-aware, progression-voted) ─────────────
    chord_map   = _infer_section_progressions(melody_notes, tempo)
    beat_dur    = 60.0 / tempo
    measure_dur = beat_dur * 4

    notes: List[Dict[str, Any]] = []
    for i, t in enumerate(onset_times):
        dur = float(onset_times[i + 1] - t) if i + 1 < len(onset_times) else beat_dur
        dur = max(0.08, min(dur, beat_dur * 2))

        onset_frame = min(int(t * sr / HOP), len(rms) - 1)

        # ── pitch estimate via pYIN window ────────────────────────────────────
        estimated_pitch = None
        if midi_f0 is not None:
            WINDOW = 6   # frames ≈ 140 ms post-attack
            frame_end = min(onset_frame + WINDOW, len(midi_f0))
            window    = midi_f0[onset_frame:frame_end]
            voiced    = window[~np.isnan(window)]
            if len(voiced) > 0:
                estimated_pitch = int(round(float(np.median(voiced))))
                estimated_pitch = max(BASS_LOW, min(BASS_HIGH, estimated_pitch))

        # Fallback: if pYIN found nothing voiced, skip this onset
        if estimated_pitch is None:
            continue

        # ── energy → preferred register ───────────────────────────────────────
        norm_energy = float(rms[onset_frame]) / rms_max
        if norm_energy > 0.55:
            r_lo, r_hi = BASS_LOW, 45          # heavy hit → root/lower
        elif norm_energy > 0.25:
            r_lo, r_hi = 33, 48               # medium
        else:
            r_lo, r_hi = 36, BASS_HIGH        # light hit → upper bass

        # ── chord snapping (measure-level, root-priority) ─────────────────────
        mi         = int(t / measure_dur)
        chord_info = chord_map.get(mi) or chord_map.get(mi - 1) or chord_map.get(mi + 1)

        pitch = estimated_pitch
        if chord_info:
            root_pc, chord_pcs = chord_info
            # Score by (bass-interval priority, proximity to pYIN estimate)
            best_p   = pitch
            best_key = (999, 999)
            for p in range(r_lo, r_hi + 1):
                if p % 12 in chord_pcs:
                    interval = (p - root_pc) % 12
                    priority = _BASS_INTERVAL_PRIORITY.get(interval, 5)
                    key      = (priority, abs(p - estimated_pitch))
                    if key < best_key:
                        best_key, best_p = key, p
            if best_key[0] == 999:   # nothing found in preferred range → expand
                for p in range(BASS_LOW, BASS_HIGH + 1):
                    if p % 12 in chord_pcs:
                        interval = (p - root_pc) % 12
                        priority = _BASS_INTERVAL_PRIORITY.get(interval, 5)
                        key      = (priority, abs(p - estimated_pitch))
                        if key < best_key:
                            best_key, best_p = key, p
            pitch = best_p

        pitch = max(BASS_LOW, min(BASS_HIGH, pitch))
        vel   = int(min(127, max(40, norm_energy * 90 + 40)))

        notes.append({
            "pitch":      pitch,
            "pitch_name": librosa.midi_to_note(pitch),
            "onset":      round(float(t), 4),
            "offset":     round(float(t + dur), 4),
            "velocity":   vel,
        })

    return notes


def run_basic_pitch(wav_path: Path) -> List[Dict[str, Any]]:
    """
    Hybrid transcription:
      • Melody/harmony — Basic Pitch on the high-frequency stem (> 260 Hz)
      • Bass           — onset detection + CQT pitch estimation on the
                         low-frequency stem; pitches snapped to chord tones
                         inferred from the melody pass.
    This avoids Basic Pitch's known weakness with low-frequency polyphony.
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    bass_audio, melody_audio, sr = _separate_stems(wav_path)

    # ── melody / harmony pass ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        melody_path = Path(tmpdir) / "melody.wav"
        sf.write(str(melody_path), melody_audio, sr)

        _, melody_midi, _ = predict(
            str(melody_path),
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.6,
            frame_threshold=0.4,
            minimum_note_length=80,
            minimum_frequency=250.0,
            maximum_frequency=4200.0,
            melodia_trick=True,
        )

    melody_notes = _midi_data_to_notes(melody_midi)

    # ── bass pass via percussion detection ───────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=bass_audio, sr=sr)
    tempo     = float(tempo) if float(tempo) > 0 else 120.0
    bass_notes = _generate_bass_from_percussion(bass_audio, sr, tempo, melody_notes)

    # ── merge & deduplicate ───────────────────────────────────────────────────
    all_notes = sorted(bass_notes + melody_notes, key=lambda n: (n["onset"], n["pitch"]))
    deduped: List[Dict[str, Any]] = []
    for n in all_notes:
        if any(
            abs(n["onset"] - x["onset"]) < 0.05 and abs(n["pitch"] - x["pitch"]) <= 2
            for x in deduped[-30:]
        ):
            continue
        deduped.append(n)

    return deduped


def refine_notes(notes: List[Dict], tempo: float) -> List[Dict]:
    """
    Post-process transcription:
    1. Drop notes shorter than 60 ms
    2. Merge same-pitch notes with gap < 50 ms
    3. Remove exact duplicates within 10 ms
    """
    if not notes:
        return notes

    MIN_DURATION = 0.06
    MERGE_GAP    = 0.05
    DEDUP_TOL    = 0.01

    notes = [n for n in notes if (n["offset"] - n["onset"]) >= MIN_DURATION]

    notes.sort(key=lambda n: (n["pitch"], n["onset"]))
    merged = []
    for n in notes:
        if (merged
                and merged[-1]["pitch"] == n["pitch"]
                and n["onset"] - merged[-1]["offset"] < MERGE_GAP):
            merged[-1]["offset"]   = max(merged[-1]["offset"], n["offset"])
            merged[-1]["velocity"] = max(merged[-1]["velocity"], n["velocity"])
        else:
            merged.append(dict(n))
    notes = merged

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    deduped = []
    for n in notes:
        if (deduped
                and deduped[-1]["pitch"] == n["pitch"]
                and abs(n["onset"] - deduped[-1]["onset"]) < DEDUP_TOL):
            continue
        deduped.append(n)

    deduped.sort(key=lambda n: (n["onset"], n["pitch"]))
    return deduped


# ── pitch correction ──────────────────────────────────────────────────────────

def _chroma_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Return mean chroma vector (12,) from audio array."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
    return chroma.mean(axis=1)


def _chroma_from_notes(notes: List[Dict], sr: int = 22050) -> np.ndarray:
    """Synthesise notes to audio with simple sine waves and get mean chroma."""
    if not notes:
        return np.zeros(12)
    duration = max(float(n.get("offset", n["onset"] + 0.5)) for n in notes) + 1.0
    total = int(duration * sr)
    audio = np.zeros(total, dtype=np.float32)
    for n in notes:
        freq  = 440.0 * (2.0 ** ((int(n["pitch"]) - 69) / 12.0))
        onset = int(float(n["onset"]) * sr)
        end   = min(int(float(n.get("offset", n["onset"] + 0.25)) * sr), total)
        length = end - onset
        if length <= 0:
            continue
        t = np.arange(length) / sr
        audio[onset:end] += (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return _chroma_from_audio(audio, sr)


def _best_semitone_shift(chroma_orig: np.ndarray, chroma_trans: np.ndarray) -> int:
    """
    Try all 12 cyclic shifts of chroma_trans and return the one with the
    highest dot-product similarity to chroma_orig.
    Prefers smaller shifts: ties broken by |shift|.
    """
    best_shift = 0
    best_score = -np.inf
    for shift in range(12):
        score = float(np.dot(chroma_orig, np.roll(chroma_trans, shift)))
        # Prefer smaller absolute shift (wrap > 6 to negative)
        eff = shift if shift <= 6 else shift - 12
        if score > best_score or (score == best_score and abs(eff) < abs(best_shift)):
            best_score = score
            best_shift = eff
    return best_shift


def _apply_semitone_shift(notes: List[Dict], shift: int) -> List[Dict]:
    if shift == 0:
        return notes
    corrected = []
    for n in notes:
        new_pitch = int(n["pitch"]) + shift
        if 21 <= new_pitch <= 108:
            corrected.append({
                **n,
                "pitch":      new_pitch,
                "pitch_name": librosa.midi_to_note(new_pitch),
            })
        else:
            corrected.append(n)   # keep original if out of piano range
    return corrected


def _per_note_octave_snap(
    notes: List[Dict], y_orig: np.ndarray, sr: int
) -> List[Dict]:
    """
    For each note, test ±1 octave shift and pick the version whose
    frequency region has stronger energy in the original audio.
    Uses a simple constant-Q energy check in the note's time window.
    """
    cqt = np.abs(librosa.cqt(y_orig, sr=sr, bins_per_octave=12, n_bins=88))
    # cqt rows correspond to MIDI notes starting at C1 (MIDI 24) approximately
    CQT_BASE = 24

    corrected = []
    for n in notes:
        pitch = int(n["pitch"])
        onset_frame = int(float(n["onset"]) * sr / 512)
        offset_frame = max(onset_frame + 1,
                           int(float(n.get("offset", n["onset"] + 0.25)) * sr / 512))
        offset_frame = min(offset_frame, cqt.shape[1])

        def energy_at(p: int) -> float:
            idx = p - CQT_BASE
            if not (0 <= idx < cqt.shape[0]):
                return 0.0
            return float(cqt[idx, onset_frame:offset_frame].mean())

        base_energy  = energy_at(pitch)
        up_energy    = energy_at(pitch + 12)
        down_energy  = energy_at(pitch - 12)

        best_pitch = pitch
        best_energy = base_energy
        for candidate, e in [(pitch + 12, up_energy), (pitch - 12, down_energy)]:
            if e > best_energy * 1.4 and 21 <= candidate <= 108:
                best_energy = e
                best_pitch = candidate

        corrected.append({
            **n,
            "pitch":      best_pitch,
            "pitch_name": librosa.midi_to_note(best_pitch),
        })
    return corrected


def pitch_correct(notes: List[Dict], wav_path: Path) -> Tuple[List[Dict], int]:
    """
    Compare the chroma fingerprint of the original audio against the
    transcription and correct any systematic semitone offset.
    Then snap individual notes to their best octave based on CQT energy.

    Returns (corrected_notes, semitone_shift_applied).
    """
    sr = 22050
    y_orig, _ = librosa.load(str(wav_path), sr=sr)

    chroma_orig  = _chroma_from_audio(y_orig, sr)
    chroma_trans = _chroma_from_notes(notes, sr)

    shift = _best_semitone_shift(chroma_orig, chroma_trans)
    notes = _apply_semitone_shift(notes, shift)
    notes = _per_note_octave_snap(notes, y_orig, sr)

    return notes, shift


def get_octave_range(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not notes:
        return {"lowest": None, "highest": None, "octave_span": 0}
    pitches     = [n["pitch"] for n in notes]
    lowest      = min(pitches)
    highest     = max(pitches)
    lowest_oct  = lowest  // 12 - 1
    highest_oct = highest // 12 - 1
    return {
        "lowest_midi":    lowest,
        "highest_midi":   highest,
        "lowest_note":    librosa.midi_to_note(lowest),
        "highest_note":   librosa.midi_to_note(highest),
        "lowest_octave":  lowest_oct,
        "highest_octave": highest_oct,
        "octave_span":    highest_oct - lowest_oct + 1,
    }
