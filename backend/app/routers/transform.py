from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import numpy as np

# How the accompaniment is produced — see the block in transform_arrangement.
#   'transformer' : 88->61 model + rule stack (default, unchanged behaviour)
#   'retrieval'   : borrow figuration from real scores, re-pitched onto this
#                   song's harmony
ARRANGEMENT_MODE = "transformer"

# ── source-fidelity rule ─────────────────────────────────────────────────────
# The arrangement may only be a SUBSET of the original, never a superset:
#
#   R1  same total duration
#   R2  onsets unchanged — the melody stays locked to the original timestamps
#   R3  at any timestamp, the pitch classes sounding must be a subset of the
#       pitch classes the original sounded at that timestamp (octave-free)
#
# Notes may be dropped or released early to fit the smaller keyboard.  No note
# may move in time, change pitch class, or be invented.
#
# Audited over 60 pieces, the stages this switches off and why:
#
#   quantize_notes         98.3% of onsets moved   -> breaks R2
#   enforce_uniform_tempo  95.1% of onsets moved   -> breaks R2
#   snap_to_chord_tones     6.9% pitch classes added -> breaks R3
#   generate_walking_bass  invents a bass line wholesale, and replaces the
#                          transcribed bass with it -> breaks R3 by design
#
# The two timing stages score ~43% on R3 as well, but that is a SYMPTOM: they
# do not invent pitches, they move notes to timestamps where those pitches
# were not sounding.  Which is also the audible "rushing" — those two
# functions flatten performed rubato onto a metronomic grid.  The fidelity
# rule and the rushing complaint point at exactly the same code.
#
# Quantisation is not gone, it moved: it is a NOTATION concern and now happens
# only when building sheet music, where a readable rhythm is the whole point.
SOURCE_FIDELITY = True

from app.services.session import get_session_dir
from app.services.transformation import notes_to_pretty_midi, enforce_uniform_tempo
from app.services.nmf_voice_leading import (
    BASS_MAX, MELODY_MIN,
    compress_notes_to_range, fit_notes_to_range, resolve_doubled_keys,
    post_process_notes,
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
        merged = fit_notes_to_range(merged, tgt_low, tgt_high)

        # Inject walking bass from kick-pattern grid
        bass_lo = tgt_low
        bass_hi = min(BASS_MAX, tgt_high)
        if (not SOURCE_FIDELITY) and audio_path.exists() and bass_lo < bass_hi:
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
        if SOURCE_FIDELITY:
            # Only the stages that drop or re-voice within the original
            # pitch-class set at the original timestamps.
            from app.services.music_theory import enforce_sounding_limits
            from app.services.nmf_voice_leading import remove_beat_outliers
            # apply_voice_leading is NOT here on purpose.  It was the last
            # remaining R3 violator at 0.25% — small, but the rule's whole
            # value is being absolute: an invariant that holds 99.75% of the
            # time cannot be used as a gate, because every future regression
            # hides inside the exception.  Voice leading reassigns pitches,
            # and "no note can change" leaves no room for that.
            merged = remove_beat_outliers(merged, tempo)
            merged = enforce_sounding_limits(merged)
        else:
            merged = post_process_notes(
            merged,
            chord_map  = chord_map,
            tempo      = tempo,
            key_root   = root,
            scale_name = scale_name,
            bass_lo    = bass_lo,
            bass_hi    = bass_hi,
        )

        # Uniform-tempo remapping is a rule violation (R2: 95% of onsets
        # move) and is the direct cause of the rushing, so under fidelity the
        # notes stay on the original performance timeline.
        if not SOURCE_FIDELITY:
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
            export_notes = fit_notes_to_range(notes, tgt_low, tgt_high)
            export_notes = remove_beat_outliers(export_notes, tempo)
            if not SOURCE_FIDELITY:
                export_notes = apply_voice_leading(export_notes)
                export_notes = quantize_notes(export_notes, tempo)
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

    # ── retrieval-augmented arrangement (opt-in) ──────────────────────────
    # ARRANGEMENT_MODE selects how the accompaniment is produced:
    #   'transformer' — the 88->61 model plus the rule stack (current
    #                   default; what every arrangement has used so far)
    #   'retrieval'   — keep the source melody and harmony, but borrow the
    #                   accompaniment FIGURATION from real scores and
    #                   re-pitch it onto this song's own chords
    # Both write the same transformed.json shape, so the whole downstream
    # (previews, MIDI, MusicXML, CDEF, the UI) is unchanged either way.
    # This is what makes the three-arm A/B possible from the app instead of
    # from a script.
    retrieval_provenance = None
    if ARRANGEMENT_MODE == "retrieval":
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from retrieval.arrange import arrange_session
            arranged = arrange_session(session_dir, verbose=False)
            if arranged["notes"]:
                retrieved = arranged["notes"]

                # Retrieval used to replace export_notes outright, which
                # meant its output skipped the entire rule stack — including
                # the range compression and the playability constraints.
                # Retrieved patterns are individually playable because they
                # come from real scores, but nothing checked them AFTER
                # re-pitching onto this song's chords, or against the melody
                # sitting on top of them.
                #
                # Only the constraints are applied, not the full stack:
                # voice-leading and chord-tone snapping would rewrite the
                # borrowed figuration, which is the one thing retrieval
                # exists to preserve.
                from app.services.music_theory import enforce_sounding_limits
                retrieved = fit_notes_to_range(retrieved, tgt_low, tgt_high)
                retrieved = enforce_hand_constraints(
                    retrieved,
                    beat_times     = beat_times_arr.tolist()
                                     if beat_times_arr is not None else None,
                    downbeat_times = downbeat_times_arr.tolist()
                                     if downbeat_times_arr is not None else None,
                    dominance_map  = dominance_map,
                    tempo          = tempo,
                    beats_per_bar  = beats_per_bar,
                )
                retrieved = enforce_sounding_limits(retrieved)

                export_notes = retrieved
                retrieval_provenance = arranged["provenance"]
                method = f"{method}+retrieval"
                print(f"[transform] retrieval arrangement: "
                      f"{arranged['counts']} over "
                      f"{len(retrieval_provenance)} spans -> "
                      f"{len(export_notes)} notes after constraints")
        except Exception as e:
            print(f"[transform] retrieval failed, keeping transformer output: "
                  f"{type(e).__name__}: {e}")

    # Every arm converges here.  Later stages (walking bass, hand-constraint
    # re-routing, chord-tone snapping) can land a note on a key that is
    # already held, so the "one key, one press" invariant is enforced last.
    # Applies at every octave setting: the 7-octave source carries ~0.2
    # collisions per piece of its own, and those cost sustain too.
    export_notes = resolve_doubled_keys(export_notes)

    for stale in ["arrangement_preview.wav", "transcription_preview.wav"]:
        p = session_dir / stale
        if p.exists():
            p.unlink()

    # Under SOURCE_FIDELITY the notes are on the ORIGINAL performance
    # timeline, which is what we want in the MIDI: an expressive performance,
    # not a metronomic one.  Either way this is a direct seconds->tick
    # conversion, so the written tempo is only a reference for notation.
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
        "arrangement_mode":       ARRANGEMENT_MODE,
        # Carried through from transcription.  Note offsets are key releases
        # on both views; the damper decides how long the sound lasts, and it
        # is applied at render time rather than baked into durations.
        "pedal":                  transcription.get("pedal", []),
        # Per-span record of which corpus pattern each passage borrowed and
        # why it won — the review surface for retrieval, in the same spirit
        # as the Layers tab for separation.
        "retrieval_provenance":   retrieval_provenance,
    }

    with open(session_dir / "transformed.json", "w") as f:
        json.dump(result, f)

    # Note: preview pre-rendering moved to /previews so the frontend can
    # show a separate progress step.  /transform stays a JSON-only call.

    return result
