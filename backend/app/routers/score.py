from fastapi import APIRouter, HTTPException
import json
import numpy as np

from app.services.session import get_session_dir
from app.services.audio_utils import compute_mel_from_array, js_divergence, render_notes_to_audio

router = APIRouter()


def _render_notes_to_audio(notes, tempo, sr=22050):
    return render_notes_to_audio(notes, sr=sr), sr


@router.post("/score/{session_id}")
def score_fidelity(session_id: str):
    session_dir = get_session_dir(session_id)

    transcription_file = session_dir / "transcription.json"
    transformed_file = session_dir / "transformed.json"
    segments_file = session_dir / "segments.json"

    for f in [transcription_file, transformed_file, segments_file]:
        if not f.exists():
            raise HTTPException(404, f"{f.name} not found. Complete prior pipeline steps.")

    with open(transcription_file) as f:
        transcription = json.load(f)
    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    tempo = seg_data["tempo_bpm"]
    original_notes = transcription["notes"]
    transformed_notes = transformed["transformed_notes"]
    segments = seg_data["segments"]

    # Render both to audio and compare mel spectrograms
    orig_audio, sr = _render_notes_to_audio(original_notes, tempo)
    trans_audio, _ = _render_notes_to_audio(transformed_notes, tempo)

    mel_orig = compute_mel_from_array(orig_audio, sr)
    mel_trans = compute_mel_from_array(trans_audio, sr)

    js_score = js_divergence(mel_orig, mel_trans)
    fidelity = max(0.0, 1.0 - js_score)

    if fidelity >= 0.80:
        tier = "green"
    elif fidelity >= 0.60:
        tier = "yellow"
    else:
        tier = "red"

    # Per-segment fidelity analysis
    flagged_segments = []
    for seg in segments:
        start, end = seg["start_time"], seg["end_time"]
        seg_orig = [n for n in original_notes if start <= n["onset"] < end]
        seg_trans = [n for n in transformed_notes if start <= n["onset"] < end]

        if not seg_orig or not seg_trans:
            continue

        seg_orig_audio, _ = _render_notes_to_audio(seg_orig, tempo)
        seg_trans_audio, _ = _render_notes_to_audio(seg_trans, tempo)

        if len(seg_orig_audio) < 512 or len(seg_trans_audio) < 512:
            continue

        seg_mel_orig = compute_mel_from_array(seg_orig_audio, sr)
        seg_mel_trans = compute_mel_from_array(seg_trans_audio, sr)
        seg_js = js_divergence(seg_mel_orig, seg_mel_trans)
        seg_fidelity = max(0.0, 1.0 - seg_js)

        if seg_fidelity < 0.80:
            flagged_segments.append({
                **seg,
                "segment_fidelity": round(seg_fidelity, 4),
                "flag": "yellow" if seg_fidelity >= 0.60 else "red",
            })

    result = {
        "session_id": session_id,
        "fidelity_score": round(fidelity, 4),
        "js_divergence": round(js_score, 4),
        "tier": tier,
        "flagged_segments": flagged_segments,
        "manual_review_required": tier in ("yellow", "red"),
    }

    with open(session_dir / "score.json", "w") as f:
        json.dump(result, f)

    return result
