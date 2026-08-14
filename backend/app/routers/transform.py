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


def _scale_info(root: int, scale_name: str) -> dict:
    """The key as reported to the client — one builder, called by every branch.

    There were two, and they disagreed.  The main path hardcoded the MAJOR
    interval set regardless of scale_name, so a minor or dorian detection
    reported the major scale on that root.  The FALLBACK path — which is what
    solo piano actually takes — hardcoded an empty note list, so it reported no
    scale notes at all.  Neither matched what drop_non_scale was enforcing,
    which reads the real set, so in both cases the notes were right and only
    the label was wrong.  That presents as "the key says X but I can see a note
    that is not in X", which is a real bug report about an unreal bug.

    Fixing one copy and not the other is what happened the first time, so now
    there is one copy.
    """
    from app.services.music_theory import _scale_pitch_classes
    try:
        pcs = _scale_pitch_classes(root, scale_name)
    except Exception:                                       # noqa: BLE001
        pcs = {(root + iv) % 12 for iv in (0, 2, 4, 5, 7, 9, 11)}
    return {
        "root":  _NOTE_NAMES[root],
        "scale": scale_name,
        "notes": sorted(_NOTE_NAMES[p] for p in pcs),
    }


def _apply_song_stages(notes, key_root, scale_name, beat_times_arr,
                       downbeat_times_arr, beats_per_bar, label="main"):
    """Every song-only arrangement stage, in one place.

    Factored out because it lived only in the main branch: when an exception
    upstream sent execution to the fallback arm, beat snapping, the arpeggiated
    accompaniment, the polyphony cap and scale enforcement all vanished at once
    while /transform still returned 200.  Both arms now call this, so they
    cannot drift apart again.
    """
    from app.services.transformation import (
        snap_to_beat_grid, cap_polyphony, drop_non_scale, enforce_melody_on_top,
        MAX_SOUNDING)
    from app.services.arpeggiator import arpeggiate
    from app.services.melody_cleanup import clean_melody
    from app.services.melody_decode import MEL_LO, DECODE_MELODY
    from app.services.music_theory import _scale_pitch_classes

    before = len(notes)
    try:
        scale_pcs = _scale_pitch_classes(key_root, scale_name)
    except Exception:                                       # noqa: BLE001
        scale_pcs = None

    # Snap EVERY part to the same grid.
    #
    # This snapped by PITCH — below MIDI 56 — on the reasoning that the melody
    # decode already places onsets by evidence (73.9% on sixteenths against a
    # human 76.4%) and forcing 100% moves it away from how people sing.  Sound
    # in isolation, wrong for an ensemble.  Two problems, measured:
    #
    #   The bass came out 100% on-grid and the melody 87%, drifting up to 48 ms
    #   at p90 — nearly half a sixteenth at 110 ms — so the parts pulled against
    #   each other.  Only 15 of 161 bars had melody and bass both landing on the
    #   downbeat; snapping both takes that to 39.
    #
    #   And a pitch threshold cuts ACROSS a role: 108 melody notes sat below
    #   MIDI 56 and were snapped while the rest were not, so one melodic line
    #   got two different timing treatments.
    #
    # Expressive timing only reads as expressive when the pulse underneath it is
    # steady.  With only melody and bass there is no such anchor, so the drift
    # reads as the parts being out of sync rather than as rubato.  One grid for
    # everything; if the melody later needs to breathe against a steady
    # accompaniment, that should be a deliberate offset, not an unsnapped
    # residue.
    notes = snap_to_beat_grid(notes, beat_times_arr)

    # Reduce the voice register before the arpeggiator, so its
    # melody/accompaniment split reads a clean top voice.  (No-op while
    # melody_decode owns the melody; kept for the A/B switch.)
    notes = clean_melody(notes, beat_times_arr, scale_pcs=scale_pcs)

    # The figure is generated at the PROJECTION now, so the full-range output is
    # already an arrangement.  Running it twice would arpeggiate an arpeggio.
    from app.services.piano_projection import REDUCE_HARMONY
    if not REDUCE_HARMONY:
        notes = arpeggiate(
            notes, beat_times_arr,
            downbeat_times_arr if downbeat_times_arr is not None else None,
            beats_per_bar, scale_pcs, key_root,
        )
    # Melody on top BEFORE capping, so the cap thins a texture that is already
    # correctly voiced rather than one where harmony is masking the tune.
    notes = enforce_melody_on_top(notes)
    notes = cap_polyphony(notes, MAX_SOUNDING)

    n_pre = len(notes)
    notes = drop_non_scale(notes, scale_pcs)
    print(f"[transform:{label}] song reduction: beat snap + arpeggiated "
          f"accompaniment + cap {MAX_SOUNDING} + scale enforcement — "
          f"{before} -> {n_pre} -> {len(notes)} notes")
    return notes


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
    # Damper-down intervals from the model's pedal head.  Without these,
    # enforce_sounding_limits counts sustained notes as occupied fingers and
    # cuts real ones to make room; measured, 100% of those cuts were spurious.
    pedal_events = transcription.get("pedal", []) or []
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
    is_song:            bool                 = False
    try:
        with open(session_dir / "components.json") as _f:
            _comp = json.load(_f)
        _frames = _comp.get("beat_frames", [])
        _dbf    = _comp.get("downbeat_frames", [])
        _sr     = float(_comp.get("sr", 22050))
        _hop    = float(_comp.get("hop_length", 512))
        beats_per_bar = int(_comp.get("beats_per_bar", 4))
        # Songs went through piano_projection, not Onsets & Frames.  Same gate
        # separation and transcription use, so there is one detector to be
        # wrong rather than three that can disagree.
        is_song = not (_comp.get("separation_skipped")
                       or _comp.get("method") == "raw+madmom")
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
        # The ML generator is NOT used for songs.  It was trained for the
        # piano -> 5-octave compression task, and the projection path feeds it
        # far longer input than it has ever seen: its positional encoding stops
        # at 4096 and this song produced 28,423, so it raised RuntimeError,
        # the handler below caught it silently, and every song stage was
        # skipped while /transform still returned 200.
        if USE_ML_PIPELINE and model_available() and not is_song:
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
        elif is_song:
            # ── Song path: skip extract_layers entirely ───────────────────
            # That stage finds repeating figures in a PIANO TRANSCRIPTION and
            # splits it into layers.  On the song path we already do that job
            # explicitly and better: the melody is decoded from the vocals
            # stem, the bass from the bass stem, and the accompaniment is
            # generated from inferred harmony onto a retrieved rhythm.
            #
            # Running it anyway was not merely redundant, it was destructive.
            # It SYNTHESISES representative notes for its layers rather than
            # passing the originals through, so the role tags the projection
            # attached did not survive: measured, 2,851 notes reached the
            # reduction with roles on about 200 of them, while
            # transcription.json had them on all 3,534.  Every note copy in
            # that module uses {**n} and preserves keys — the notes simply were
            # not the same notes.
            #
            # Same shape of mistake as handing songs to the ML generator: the
            # wrong tool for the task, quietly doing damage while returning 200.
            method = "stem_roles"
            merged = [dict(n) for n in notes]
            print(f"[transform] song path: extract_layers bypassed, "
                  f"{len(merged)} notes with roles carried through")
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
        # Trust the key the projection already established from DETECTED notes.
        # Re-deriving it here would poll the generated harmony, which is 52% of
        # the note set — that is how a song in E major came to be reported as A
        # major, with the accompaniment generated in G.
        _stamped = next((n for n in merged if n.get("key_root") is not None), None)
        if _stamped is not None:
            root, scale_name = int(_stamped["key_root"]), str(_stamped["key_scale"])
        else:
            # Nothing stamped means the projection did not run (it fell back to
            # Onsets & Frames, or failed).  Re-deriving here is the very poll
            # the comment above warns about, so at least exclude the generated
            # material: prefer notes that carry a DETECTED role.
            _det = [n for n in merged if n.get("role") in ("melody", "bass")]
            root, scale_name, _ = detect_scale(_det or merged)
        scale_info = _scale_info(root, scale_name)
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
            merged = enforce_sounding_limits(merged, pedal=pedal_events)
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

        # Song-arrangement reduction.  SOURCE_FIDELITY protects Onsets & Frames
        # onsets, which deserve it; it should not protect projection onsets,
        # which come from peak-picking at precision 0.31-0.47.  So the rule is
        # SCOPED to the path it was written for rather than dropped.
        if is_song:
            merged = _apply_song_stages(
                merged, root, scale_name, beat_times_arr,
                downbeat_times_arr, beats_per_bar, label="main")

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

    except Exception as _main_exc:
        # SAY SO.  This handler swallowed the reason silently, and the fallback
        # arm skips every song-path stage — beat snap, arpeggiated
        # accompaniment, polyphony cap, scale enforcement.  So a single
        # exception anywhere above turned all of them off while /transform
        # still returned 200 and the log looked clean.  That is exactly how the
        # scale drop appeared to be running when it never had.
        import traceback
        print(f"[transform] MAIN PATH FAILED ({type(_main_exc).__name__}: "
              f"{_main_exc}) — falling back, song stages SKIPPED")
        traceback.print_exc()
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
            scale_info = _scale_info(root, scale_name)
            # Song stages belong here as well.  Their absence is what made a
            # single upstream exception silently undo beat snapping, the
            # arpeggiated accompaniment, the polyphony cap and scale
            # enforcement all at once.
            if is_song:
                export_notes = _apply_song_stages(
                    export_notes, root, scale_name, beat_times_arr,
                    downbeat_times_arr, beats_per_bar, label="fallback")
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
                retrieved = enforce_sounding_limits(retrieved, pedal=pedal_events)

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
    # Songs skipped extract_layers, so give the UI role-based layers instead of
    # nothing.  These are more useful than the pattern-cluster layers anyway —
    # melody / bass / accompaniment is what a player actually wants to isolate.
    if is_song and not layers:
        _ROLE_META = [("melody", "Melody", "#f87171"),
                      ("bass", "Bass", "#3b82f6"),
                      ("harmony", "Accompaniment", "#a78bfa")]
        layers = []
        for key, label, colour in _ROLE_META:
            got = [n for n in export_notes if n.get("role") == key]
            if got:
                layers.append({"id": key, "name": label, "color": colour,
                               "notes": got})

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
