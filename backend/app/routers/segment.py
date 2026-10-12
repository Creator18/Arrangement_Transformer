import json

from fastapi import APIRouter, HTTPException

from app.services.session import get_session_dir
from app.services.segmentation import analyse_audio, _chord_map_from_json

router = APIRouter()


# Map detect_scale's pitch-class int → flat-spelled name.  Kept here rather
# than imported so we don't drag in the heavier nmf_voice_leading symbols.
_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F",
               "Gb", "G", "Ab", "A", "Bb", "B"]


def _apply_original_cleanup(session_dir, segment_result):
    """
    Apply chord-snap + scale-snap + hand-rule pruning to the ORIGINAL
    transcription, in-place on transcription.json.

    This mirrors the harmonic guardrails the TRANSFORMED arrangement gets
    via apply_music_theory, so the two piano-roll views stay consistent.
    Runs after analyse_audio so we have chord_map + tempo available.

    Failure here is non-fatal: a print + skip leaves the raw transcription
    intact, and /segment's own return value is unaffected.
    """
    transcription_file = session_dir / "transcription.json"
    if not transcription_file.exists():
        return None

    from app.services.music_theory import apply_chord_and_scale_snap
    from app.services.nmf_voice_leading import detect_scale

    # Solo piano is a transcription of what was played: keep every note.  On
    # Flight of the Bumblebee (chromatic, no single scale) this cleanup took
    # 137 of 1,569 correctly heard notes out of the Original track.  The
    # guardrails stay for songs, whose notes are approximations.
    import os
    if os.environ.get("ARRANGEMENT_PIANO_ONLY") == "1":
        print("[segment] ORIGINAL cleanup skipped: piano-only experiment")
        return None
    try:
        with open(session_dir / "components.json") as f:
            if json.load(f).get("method") == "raw+madmom":
                print("[segment] ORIGINAL cleanup skipped: solo piano")
                return None
    except (OSError, ValueError):
        pass

    with open(transcription_file) as f:
        trans = json.load(f)

    notes = trans.get("notes", [])
    if not notes:
        return None

    tempo     = float(segment_result.get("tempo_bpm", 120.0))
    chord_map = _chord_map_from_json(segment_result.get("chord_map", {}))
    root, scale_name, _ = detect_scale(notes)

    n_before = len(notes)
    cleaned  = apply_chord_and_scale_snap(
        notes, chord_map, tempo, root, scale_name,
    )
    n_after = len(cleaned)

    trans["notes"]            = cleaned
    trans["detected_scale"]   = {
        "root":  _NOTE_NAMES[root],
        "scale": scale_name,
    }
    trans["original_cleanup"] = {
        "applied":      True,
        "notes_before": n_before,
        "notes_after":  n_after,
        "scale":        f"{_NOTE_NAMES[root]} {scale_name}",
    }

    with open(transcription_file, "w") as f:
        json.dump(trans, f)

    # Invalidate any pre-rendered preview WAV; it was rendered from the
    # uncleaned note list.  Will be re-rendered lazily (or pre-rendered
    # by /transform later in the pipeline).
    stale = session_dir / "transcription_preview.wav"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass

    print(f"[segment] ORIGINAL cleanup: scale={_NOTE_NAMES[root]} {scale_name}, "
          f"{n_before} -> {n_after} notes")
    return {"notes_before": n_before, "notes_after": n_after,
            "scale": f"{_NOTE_NAMES[root]} {scale_name}"}


@router.post("/segment/{session_id}")
async def segment_audio(session_id: str):
    session_dir = get_session_dir(session_id)
    wav_path    = session_dir / "audio.wav"

    if not wav_path.exists():
        raise HTTPException(404, "Session audio not found. Upload first.")
    if not (session_dir / "components.json").exists():
        raise HTTPException(404, "Components not found. Run /separate first.")

    result = analyse_audio(wav_path, session_dir)

    try:
        cleanup_info = _apply_original_cleanup(session_dir, result)
        if cleanup_info is not None:
            result["original_cleanup"] = cleanup_info
    except Exception as e:
        print(f"[segment] original-side cleanup skipped: {e}")

    return result
