from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import json
import soundfile as sf
from pathlib import Path

from app.services.session import get_session_dir, cleanup_session
from app.services.cdef_format import notes_to_cdef
from app.services.audio_utils import render_notes_to_audio

router = APIRouter()


@router.get("/export/{session_id}/audio/original")
async def export_original_recording(session_id: str):
    """Serve the original uploaded WAV."""
    session_dir = get_session_dir(session_id)
    wav_path = session_dir / "audio.wav"
    if not wav_path.exists():
        raise HTTPException(404, "Original audio not found.")
    return FileResponse(str(wav_path), media_type="audio/wav", filename="original.wav")


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

    return FileResponse(str(audio_path), media_type="audio/wav", filename="transcription_preview.wav")


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

    return FileResponse(str(audio_path), media_type="audio/wav", filename="arrangement_preview.wav")


def _generate_pdf_sheet_music(notes, tempo, output_path: Path):
    """Render notes to PDF sheet music via music21."""
    import music21
    from music21 import stream, note as m21note, tempo as m21tempo, metadata

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Arrangement Transformer Output"

    rh_part = stream.Part()
    lh_part = stream.Part()
    rh_part.partName = "Right Hand"
    lh_part.partName = "Left Hand"

    mm = m21tempo.MetronomeMark(number=tempo)
    rh_part.append(mm)

    MELODY_ZONE_LOW = 48

    for n in notes:
        pitch_midi = int(n["pitch"])
        dur_quarters = max(0.25, (n["offset"] - n["onset"]) * (tempo / 60.0))
        m21_note = m21note.Note()
        m21_note.pitch.midi = pitch_midi
        m21_note.quarterLength = round(dur_quarters, 2)
        m21_note.offset = n["onset"] * (tempo / 60.0)
        m21_note.volume.velocity = int(n.get("velocity", 80))

        if pitch_midi >= MELODY_ZONE_LOW:
            rh_part.append(m21_note)
        else:
            lh_part.append(m21_note)

    score.append(rh_part)
    score.append(lh_part)

    score.write("musicxml.pdf", fp=str(output_path))


@router.get("/export/{session_id}/midi")
async def export_midi(session_id: str):
    session_dir = get_session_dir(session_id)
    midi_path = session_dir / "transformed.mid"
    if not midi_path.exists():
        raise HTTPException(404, "MIDI not found. Run /transform first.")
    return FileResponse(
        str(midi_path),
        media_type="audio/midi",
        filename="arrangement.mid",
    )


@router.get("/export/{session_id}/cdef")
async def export_cdef(session_id: str):
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found. Run /transform first.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    notes = transformed["transformed_notes"]
    tempo = seg_data["tempo_bpm"]
    time_sig = seg_data["time_signature"]
    time_sig_num = int(time_sig.split("/")[0])

    cdef_text = notes_to_cdef(notes, tempo, time_sig_num)
    cdef_path = session_dir / "arrangement.cdef"
    cdef_path.write_text(cdef_text, encoding="utf-8")

    return FileResponse(
        str(cdef_path),
        media_type="text/plain",
        filename="arrangement.cdef",
    )


@router.get("/export/{session_id}/pdf")
async def export_pdf(session_id: str):
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found. Run /transform first.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    pdf_path = session_dir / "sheet_music.pdf"
    _generate_pdf_sheet_music(
        transformed["transformed_notes"],
        seg_data["tempo_bpm"],
        pdf_path,
    )

    if not pdf_path.exists():
        raise HTTPException(500, "PDF generation failed.")

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="sheet_music.pdf",
    )


@router.get("/export/{session_id}/cdef-text")
async def get_cdef_text(session_id: str):
    """Return CDEF as JSON string for inline display in frontend."""
    session_dir = get_session_dir(session_id)

    transformed_file = session_dir / "transformed.json"
    segments_file = session_dir / "segments.json"

    if not transformed_file.exists():
        raise HTTPException(404, "Transform not found.")

    with open(transformed_file) as f:
        transformed = json.load(f)
    with open(segments_file) as f:
        seg_data = json.load(f)

    notes = transformed["transformed_notes"]
    tempo = seg_data["tempo_bpm"]
    time_sig_num = int(seg_data["time_signature"].split("/")[0])

    cdef_text = notes_to_cdef(notes, tempo, time_sig_num)
    return {"cdef": cdef_text}


@router.delete("/cleanup/{session_id}")
async def cleanup(session_id: str):
    cleanup_session(session_id)
    return {"status": "cleaned", "session_id": session_id}
