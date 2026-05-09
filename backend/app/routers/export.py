from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any

from app.services.session import get_session_dir, cleanup_session
from app.services.cdef_format import notes_to_cdef
from app.services.sf2_sampler import render_notes_sf2 as render_notes_to_audio

router = APIRouter()

# ── part pitch ranges (MIDI) ─────────────────────────────────────────────────
PART_RANGES: Dict[str, tuple] = {
    "bass":   (0,  47),
    "melody": (48, 60),
    "voice":  (61, 127),
    "all":    (0,  127),
}


def _filter_notes_by_part(notes: List[Dict], part: str) -> List[Dict]:
    lo, hi = PART_RANGES.get(part, (0, 127))
    return [n for n in notes if lo <= int(n["pitch"]) <= hi]


# ── original recording ───────────────────────────────────────────────────────

@router.get("/export/{session_id}/audio/original")
async def export_original_recording(session_id: str):
    """Serve the original uploaded WAV."""
    session_dir = get_session_dir(session_id)
    wav_path = session_dir / "audio.wav"
    if not wav_path.exists():
        raise HTTPException(404, "Original audio not found.")
    return FileResponse(str(wav_path), media_type="audio/wav", filename="original.wav")


# ── full-arrangement previews ────────────────────────────────────────────────

@router.get("/export/{session_id}/audio/transcription")
async def export_transcription_audio(session_id: str):
    """Render the original transcribed notes to WAV (pre-transform)."""
    session_dir = get_session_dir(session_id)
    transcription_file = session_dir / "transcription.json"
    if not transcription_file.exists():
        raise HTTPException(404, "Transcription not found.")

    with open(transcription_file) as f:
        transcription = json.load(f)

    audio_path = session_dir / "transcription_preview.wav"
    if not audio_path.exists():
        sr = 22050
        audio = render_notes_to_audio(transcription["notes"], sr=sr)
        sf.write(str(audio_path), audio, sr, subtype="PCM_16")

    return FileResponse(str(audio_path), media_type="audio/wav",
                        filename="transcription_preview.wav")


@router.get("/export/{session_id}/audio/transformed")
async def export_transformed_audio(session_id: str):
    """Render the transformed arrangement to WAV."""
    session_dir = get_session_dir(session_id)
    transformed_file = session_dir / "transformed.json"
    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found. Run /transform first.")

    with open(transformed_file) as f:
        transformed = json.load(f)

    audio_path = session_dir / "arrangement_preview.wav"
    if not audio_path.exists():
        sr = 22050
        audio = render_notes_to_audio(transformed["transformed_notes"], sr=sr)
        sf.write(str(audio_path), audio, sr, subtype="PCM_16")

    return FileResponse(str(audio_path), media_type="audio/wav",
                        filename="arrangement_preview.wav")


# ── per-part audio (bass / melody / voice / all) ─────────────────────────────

@router.get("/export/{session_id}/audio/{arrangement}/{part}")
async def export_part_audio(session_id: str, arrangement: str, part: str):
    """
    Render a single register of an arrangement to WAV on demand.
    arrangement: 'transcription' | 'transformed'
    part:        'bass' | 'melody' | 'voice' | 'all'
    """
    if arrangement not in ("transcription", "transformed"):
        raise HTTPException(400, "arrangement must be 'transcription' or 'transformed'.")
    if part not in PART_RANGES:
        raise HTTPException(400, f"part must be one of {list(PART_RANGES)}")

    session_dir = get_session_dir(session_id)

    if arrangement == "transcription":
        src_file = session_dir / "transcription.json"
        notes_key = "notes"
    else:
        src_file = session_dir / "transformed.json"
        notes_key = "transformed_notes"

    if not src_file.exists():
        raise HTTPException(404, f"{arrangement} data not found.")

    with open(src_file) as f:
        data = json.load(f)

    notes = _filter_notes_by_part(data[notes_key], part)

    cache_name = f"{arrangement}_{part}.wav"
    audio_path = session_dir / cache_name

    if not audio_path.exists():
        sr = 22050
        audio = render_notes_to_audio(notes, sr=sr)
        sf.write(str(audio_path), audio, sr, subtype="PCM_16")

    return FileResponse(str(audio_path), media_type="audio/wav",
                        filename=f"{arrangement}_{part}.wav")


# ── MIDI ─────────────────────────────────────────────────────────────────────

@router.get("/export/{session_id}/midi")
async def export_midi(session_id: str):
    session_dir = get_session_dir(session_id)
    midi_path = session_dir / "transformed.mid"
    if not midi_path.exists():
        raise HTTPException(404, "MIDI not found. Run /transform first.")
    return FileResponse(str(midi_path), media_type="audio/midi",
                        filename="arrangement.mid")


# ── MusicXML (sheet music, rendered client-side by OSMD) ────────────────────

def _build_score(notes: List[Dict], tempo: float):
    """Build a music21 Score from note dicts. No external tool needed."""
    from music21 import stream, note as m21note, tempo as m21tempo, metadata

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Arrangement Transformer"

    rh = stream.Part(); rh.partName = "Right Hand"
    lh = stream.Part(); lh.partName = "Left Hand"
    rh.append(m21tempo.MetronomeMark(number=int(tempo)))

    for n in sorted(notes, key=lambda x: x["onset"]):
        pitch_midi = int(n["pitch"])
        dur_q = max(0.125, (float(n["offset"]) - float(n["onset"])) * (tempo / 60.0))
        m21n = m21note.Note()
        m21n.pitch.midi = pitch_midi
        m21n.quarterLength = round(dur_q, 3)
        m21n.offset = float(n["onset"]) * (tempo / 60.0)
        m21n.volume.velocity = int(n.get("velocity", 80))
        (rh if pitch_midi >= 48 else lh).append(m21n)

    score.append(rh)
    score.append(lh)
    return score


@router.get("/export/{session_id}/musicxml")
async def export_musicxml(session_id: str):
    """Export sheet music as MusicXML for in-browser rendering via OSMD."""
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file    = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found. Run /transform first.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    xml_path = session_dir / "sheet_music.xml"
    if not xml_path.exists():
        try:
            score = _build_score(transformed["transformed_notes"], seg_data["tempo_bpm"])
            score.write("musicxml", fp=str(xml_path))
        except Exception as e:
            raise HTTPException(500, f"MusicXML generation failed: {e}")

    if not xml_path.exists():
        raise HTTPException(500, "MusicXML file was not created.")

    return FileResponse(
        str(xml_path),
        media_type="application/vnd.recordare.musicxml+xml",
        filename="sheet_music.xml",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ── CDEF ─────────────────────────────────────────────────────────────────────

@router.get("/export/{session_id}/cdef")
async def export_cdef(session_id: str):
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file    = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found. Run /transform first.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    notes         = transformed["transformed_notes"]
    tempo         = seg_data["tempo_bpm"]
    time_sig_num  = int(seg_data["time_signature"].split("/")[0])

    cdef_text = notes_to_cdef(notes, tempo, time_sig_num)
    cdef_path = session_dir / "arrangement.cdef"
    cdef_path.write_text(cdef_text, encoding="utf-8")

    return FileResponse(str(cdef_path), media_type="text/plain",
                        filename="arrangement.cdef")


@router.get("/export/{session_id}/cdef-text")
async def get_cdef_text(session_id: str):
    """Return CDEF as JSON string for inline display in frontend."""
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file    = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    notes        = transformed["transformed_notes"]
    tempo        = seg_data["tempo_bpm"]
    time_sig_num = int(seg_data["time_signature"].split("/")[0])

    cdef_text = notes_to_cdef(notes, tempo, time_sig_num)
    return {"cdef": cdef_text}


# ── cleanup ──────────────────────────────────────────────────────────────────

@router.delete("/cleanup/{session_id}")
async def cleanup(session_id: str):
    cleanup_session(session_id)
    return {"status": "cleaned", "session_id": session_id}
