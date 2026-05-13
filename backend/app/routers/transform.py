from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import numpy as np

from app.services.session import get_session_dir
from app.services.transformation import notes_to_pretty_midi, enforce_uniform_tempo
from app.services.nmf_voice_leading import (
    BASS_MAX, MELODY_MIN,
    compress_notes_to_range, post_process_notes,
    detect_scale, _NOTE_NAMES,
    TARGET_LOW, TARGET_HIGH,
    apply_feel_offset,
    enforce_hand_constraints,
)
from app.services.segmentation import _chord_map_from_json
from app.services.pattern_extraction import extract_layers
from app.services.percussion_pattern import (
    extract_percussion_onsets,
    build_measure_patterns,
    find_section_kick_patterns,
    generate_walking_bass,
)

# ML inference path — disabled by default via config.USE_ML_PIPELINE.  Once a
# checkpoint exists at PATHS.default_ckpt AND the flag is flipped, the
# Transformer generator takes over from extract_layers().  Everything else
# (compression, walking bass, voice leading, tempo enforcement) still runs
# on top as the deterministic safety layer.
from app.ml.config       import USE_ML_PIPELINE
from app.ml.inference    import model_available, transform_with_model

router = APIRouter()

_OCTAVE_BOUNDS = {
    5: (36, 96),    # C2–C7
    6: (28, 103),   # E1–G7
    7: (21, 108),   # A0–C8 (full 88-key)
}


class TransformRequest(BaseModel):
    target_octave_range: int = 5
    source_octave_range: Optional[int] = None
    nmf_components: int = 16


@router.post("/transform/{session_id}")
async def transform_arrangement(session_id: str, body: TransformRequest = TransformRequest()):
    session_dir = get_session_dir(session_id)

    for fname, step in [
        ("transcription.json", "/transcribe"),
        ("segments.json",      "/segment"),
    ]:
        if not (session_dir / fname).exists():
            raise HTTPException(404, f"{fname} not found. Run {step} first.")

    with open(session_dir / "transcription.json") as f:
        transcription = json.load(f)
    with open(session_dir / "segments.json") as f:
        seg_data = json.load(f)

    notes       = transcription["notes"]
    tempo       = float(seg_data["tempo_bpm"])
    source_span = body.source_octave_range or transcription["octave_range"]["octave_span"]
    tgt_low, tgt_high = _OCTAVE_BOUNDS.get(body.target_octave_range, (TARGET_LOW, TARGET_HIGH))
    audio_path  = session_dir / "audio.wav"

    # Per-layer timing feel from transcription (median onset offset vs beat)
    timing_feel = transcription.get("timing_feel", {})
    bass_feel   = float(timing_feel.get("bass",  0.0))
    upper_feel  = float(timing_feel.get("upper", 0.0))

    # Chord map from audio chroma — loaded directly, no re-inference from notes
    chord_map = _chord_map_from_json(seg_data.get("chord_map", {}))

    # Per-measure dominance map: 'bass' or 'melody' based on JS divergence from segmentation
    dominance_map = {
        seg["measure_index"]: seg["star"]
        for seg in seg_data.get("segments", [])
        if "star" in seg and "measure_index" in seg
    }

    # Actual beat / downbeat positions (seconds) from separation.  Used by
    # walking-bass placement, enforce_uniform_tempo, Phase 6b's importance
    # scoring, and Phase 6c's measure-boundary lookups (for overlap
    # resolution).
    beat_times_arr:     Optional[np.ndarray] = None
    downbeat_times_arr: Optional[np.ndarray] = None
    beats_per_bar:      int                  = 4
    try:
        with open(session_dir / "components.json") as _f:
            _comp = json.load(_f)
        _frames = _comp.get("beat_frames", [])
        _dbf    = _comp.get("downbeat_frames", [])
        _sr     = float(_comp.get("sr", 22050))
        _hop    = float(_comp.get("hop_length", 512))
        beats_per_bar = int(_comp.get("beats_per_bar", 4))
        if _frames:
            beat_times_arr = np.array(_frames, dtype=float) * _hop / _sr
        if _dbf:
            downbeat_times_arr = np.array(_dbf, dtype=float) * _hop / _sr
    except Exception:
        pass

    # ── ML or rule-based transformation ────────────────────────────────────
    # Both paths land at the same `merged` shape so downstream post-processing
    # (compress, walking-bass, voice-leading, tempo enforcement) is shared.
    layers       = None
    export_notes = None
    scale_info   = {}
    method       = "pattern_extraction"

    try:
        if USE_ML_PIPELINE and model_available():
            # ── ML path ───────────────────────────────────────────────────
            method = "ml"
            # Estimate piece duration from the latest note offset (+1s tail).
            duration_s = (
                max(float(n.get("offset", n["onset"] + 0.5)) for n in notes) + 1.0
                if notes else 1.0
            )
            merged = transform_with_model(
                notes     = notes,
                chord_map = chord_map,
                tempo     = tempo,
                duration  = duration_s,
            )
            # The ML adapter returns notes already in the 5-octave target
            # range, so we skip extract_layers entirely.  We still run
            # compress_notes_to_range below as a clamp, which is a no-op
            # when notes are already in-range.
        else:
            # ── Rule-based path (current default) ─────────────────────────
            layers = extract_layers(
                notes,
                chord_map,
                tempo,
                audio_path if audio_path.exists() else None,
                dominance_map=dominance_map,
                upper_feel=upper_feel,
            )

            merged = layers[0].pop("_merged_notes", None)
            if merged is None:
                merged = sorted(
                    (n for layer in layers for n in layer["notes"]),
                    key=lambda n: (float(n["onset"]), int(n["pitch"])),
                )

        # Compress into target octave range
        merged = compress_notes_to_range(merged, tgt_low, tgt_high)

        # Inject walking bass from kick-pattern grid
        bass_lo = tgt_low
        bass_hi = min(BASS_MAX, tgt_high)
        if audio_path.exists() and bass_lo < bass_hi:
            try:
                duration_s  = max(float(n.get("offset", n["onset"] + 0.5)) for n in merged) + 1.0
                measure_dur = (60.0 / tempo) * 4
                n_measures  = max(1, int(np.ceil(duration_s / measure_dur)))

                perc_onsets  = extract_percussion_onsets(audio_path)
                patterns     = build_measure_patterns(perc_onsets, n_measures, tempo)
                kick_map     = find_section_kick_patterns(patterns["kick"])
                walking_bass = generate_walking_bass(
                    chord_map             = chord_map,
                    measure_kick_patterns = kick_map,
                    tempo                 = tempo,
                    lo                    = bass_lo,
                    hi                    = bass_hi,
                    beat_times            = beat_times_arr,
                )
                if walking_bass:
                    # Shift generated walking bass by the bass register's feel so
                    # it lands with the same timing character as the transcribed bass
                    walking_bass = apply_feel_offset(walking_bass, bass_feel)
                    merged = [n for n in merged if int(n["pitch"]) > bass_hi] + walking_bass
                    merged.sort(key=lambda n: (float(n["onset"]), int(n["pitch"])))
            except Exception:
                pass

        # Voice leading, quantise, chord snap, music theory
        root, scale_name, _ = detect_scale(merged)
        scale_info = {
            "root":  _NOTE_NAMES[root],
            "scale": scale_name,
            "notes": sorted(_NOTE_NAMES[(root + iv) % 12] for iv in [0,2,4,5,7,9,11]),
        }
        merged = post_process_notes(
            merged,
            chord_map  = chord_map,
            tempo      = tempo,
            key_root   = root,
            scale_name = scale_name,
            bass_lo    = bass_lo,
            bass_hi    = bass_hi,
        )

        # Final pass: smooth the entire piece onto a strict uniform-tempo grid
        # using the original beat_times.  All notes (transcribed + walking bass)
        # are now on the same original-audio timeline, so this remaps them
        # together with no relative drift.
        merged = enforce_uniform_tempo(merged, beat_times_arr, tempo)

        # Phase 6a/6b/6c — physical playability:
        #   6a: hard constraints (≤5/hand, ≤1-octave span/hand, ≤10 total)
        #   6b: comfort-aware assignment (movement vs spread, phrase coherence,
        #       weak-beat-first drop using downbeat data)
        #   6c: cross-hand overlap resolution using per-measure dominance data
        #       (overlap zone routes to the segment's dominant voice's hand)
        merged = enforce_hand_constraints(
            merged,
            beat_times     = beat_times_arr.tolist()     if beat_times_arr     is not None else None,
            downbeat_times = downbeat_times_arr.tolist() if downbeat_times_arr is not None else None,
            dominance_map  = dominance_map,
            tempo          = tempo,
            beats_per_bar  = beats_per_bar,
        )

        export_notes = merged

    except Exception:
        # Hard fallback: return original notes post-processed only
        from app.services.nmf_voice_leading import (
            quantize_notes, apply_voice_leading,
            remove_beat_outliers, snap_to_chord_tones,
        )
        method = "fallback"
        try:
            export_notes = compress_notes_to_range(notes, tgt_low, tgt_high)
            export_notes = apply_voice_leading(export_notes)
            export_notes = quantize_notes(export_notes, tempo)
            export_notes = remove_beat_outliers(export_notes, tempo)
            export_notes = snap_to_chord_tones(export_notes, tempo)
            export_notes = enforce_uniform_tempo(export_notes, beat_times_arr, tempo)
            export_notes = enforce_hand_constraints(
                export_notes,
                beat_times     = beat_times_arr.tolist()     if beat_times_arr     is not None else None,
                downbeat_times = downbeat_times_arr.tolist() if downbeat_times_arr is not None else None,
                dominance_map  = dominance_map,
                tempo          = tempo,
                beats_per_bar  = beats_per_bar,
            )
            root, scale_name, _ = detect_scale(export_notes)
            scale_info = {"root": _NOTE_NAMES[root], "scale": scale_name, "notes": []}
        except Exception:
            export_notes = notes

    for stale in ["arrangement_preview.wav", "transcription_preview.wav"]:
        p = session_dir / stale
        if p.exists():
            p.unlink()

    # Notes are already on a uniform-tempo timeline after enforce_uniform_tempo,
    # so MIDI export just does a direct seconds→tick conversion (no beat grid).
    pm        = notes_to_pretty_midi(export_notes, tempo)
    midi_path = session_dir / "transformed.mid"
    pm.write(str(midi_path))

    result = {
        "session_id":             session_id,
        "method":                 method,
        "source_octave_range":    source_span,
        "target_octave_range":    body.target_octave_range,
        "nmf_components":         body.nmf_components,
        "original_note_count":    len(notes),
        "transformed_note_count": len(export_notes),
        "transformed_notes":      export_notes,
        "detected_scale":         scale_info,
        "midi_path":              str(midi_path),
        "layers":                 layers,
    }

    with open(session_dir / "transformed.json", "w") as f:
        json.dump(result, f)

    # Note: preview pre-rendering moved to /previews so the frontend can
    # show a separate progress step.  /transform stays a JSON-only call.

    return result
