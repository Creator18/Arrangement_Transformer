from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any

from app.services.session import get_session_dir, cleanup_session
from app.services.cdef_format import notes_to_cdef
from app.services.sampler import render_notes as render_notes_to_audio

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


# ── per-layer audio (bass / pattern / melody, by `layer` field) ──────────────
# Different from /audio/{arrangement}/{part} which filters by pitch range.
# This endpoint filters by the per-note `layer` label produced by pYIN
# (bass) and the pattern_classifier (pattern / melody).  When a real
# audio-level stem exists on disk (bass.wav, pattern.wav, melody.wav) we
# serve that directly; otherwise the per-layer notes are synthesised via
# the SF2 sampler.  Used by ResultsPage's "Layers" review tab.

_LAYER_NAMES = ("bass", "bass_pattern", "bass_melody",
                "pattern", "melody")

# Which layers correspond to which REPET-SIM cluster-bearing stems.  Used
# by the cluster endpoints to find the right entry in pattern_clusters.json.
_CLUSTER_LAYERS = {
    "pattern":      "pattern",
    "bass_pattern": "bass_pattern",
}


@router.get("/export/{session_id}/audio/layer/{layer}")
async def export_layer_audio(session_id: str, layer: str):
    if layer not in _LAYER_NAMES:
        raise HTTPException(400, f"layer must be one of {list(_LAYER_NAMES)}")

    session_dir = get_session_dir(session_id)

    # 1. Prefer real audio-level stem when present (REPET path)
    components_dir = session_dir / "components"
    stem_path = components_dir / f"{layer}.wav"
    if stem_path.exists():
        return FileResponse(str(stem_path), media_type="audio/wav",
                            filename=f"{layer}.wav")

    # 2. Fall back: synthesize from layer-labeled notes
    transcription_file = session_dir / "transcription.json"
    if not transcription_file.exists():
        raise HTTPException(404, "Transcription not found.")

    with open(transcription_file) as f:
        transcription = json.load(f)

    layer_notes = [n for n in transcription.get("notes", [])
                   if n.get("layer") == layer]

    cache_name = f"layer_{layer}.wav"
    audio_path = session_dir / cache_name

    if not audio_path.exists():
        sr = 22050
        audio = render_notes_to_audio(layer_notes, sr=sr)
        sf.write(str(audio_path), audio, sr, subtype="PCM_16")

    return FileResponse(str(audio_path), media_type="audio/wav",
                        filename=cache_name)


# ── pattern clusters (Pattern 1 … N per pattern stem) ─────────────────────────

@router.get("/export/{session_id}/pattern-clusters")
async def get_pattern_clusters(session_id: str):
    """
    Return per-stem cluster metadata so the UI knows how many Pattern-N
    rows to render under each pattern layer.

    Response shape:
      {
        "pattern":      {"n_clusters": 3, "granularity": "bar",
                         "cluster_bar_counts": {"0": 12, "1": 4, "2": 2}},
        "bass_pattern": {"n_clusters": 2, ...},
      }
    """
    session_dir = get_session_dir(session_id)
    clusters_file = session_dir / "pattern_clusters.json"
    if not clusters_file.exists():
        return {}
    with open(clusters_file) as f:
        data = json.load(f)
    # Strip audio_paths (server-side absolute paths shouldn't leak to clients)
    out = {}
    for stem, meta in data.items():
        out[stem] = {
            "n_clusters":         int(meta.get("n_clusters", 0)),
            "granularity":        meta.get("granularity", "bar"),
            "cluster_bar_counts": meta.get("cluster_bar_counts", {}),
        }
    return out


@router.get("/export/{session_id}/audio/pattern-cluster/{layer}/{cluster_idx}")
async def export_pattern_cluster_audio(session_id: str, layer: str, cluster_idx: int):
    """
    Serve the audio for one cluster of one pattern stem.

    Order of preference:
      1. The concatenated real-audio slice written by separation.py
         (`components/<layer>_cluster_<i>.wav`).  Already on disk after
         separation — zero synthesis cost, instant response.
      2. SF2/SFZ resynth from notes with matching pattern_idx.  Used only
         when separation didn't produce the slice (older sessions, or a
         cluster with too few bars).

    The real-audio slice IS what REPET-SIM actually extracted for that
    cluster, which is the most honest signal for review-tab purposes
    anyway.  Synthesis adds latency without adding fidelity.
    """
    if layer not in _CLUSTER_LAYERS:
        raise HTTPException(400, f"layer must be one of {list(_CLUSTER_LAYERS)}")

    session_dir = get_session_dir(session_id)

    # 1. Prefer the pre-written real-audio slice (instant — no synthesis)
    components_dir = session_dir / "components"
    wav_path = components_dir / f"{layer}_cluster_{cluster_idx}.wav"
    if wav_path.exists():
        return FileResponse(str(wav_path), media_type="audio/wav",
                            filename=wav_path.name)

    # 2. Fall back to SF2/SFZ resynth from layer + pattern_idx filtered notes
    transcription_file = session_dir / "transcription.json"
    if transcription_file.exists():
        with open(transcription_file) as f:
            transcription = json.load(f)
        notes = [
            n for n in transcription.get("notes", [])
            if n.get("layer") == layer and int(n.get("pattern_idx", -1)) == cluster_idx
        ]
        if len(notes) >= 2:
            cache_name = f"cluster_{layer}_{cluster_idx}.wav"
            audio_path = session_dir / cache_name
            if not audio_path.exists():
                sr = 22050
                audio = render_notes_to_audio(notes, sr=sr)
                sf.write(str(audio_path), audio, sr, subtype="PCM_16")
            return FileResponse(str(audio_path), media_type="audio/wav",
                                filename=cache_name)

    raise HTTPException(404, f"No audio for cluster {cluster_idx} of {layer}.")


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

    # Fast path: part='all' is identical to the main preview the LISTEN player
    # already pre-renders during /previews — same note list, same SR.  Serving
    # the cached preview WAV directly is instant (file response) instead of
    # synthesising a duplicate render through Salamander on first click.
    if part == "all":
        session_dir = get_session_dir(session_id)
        preview_name = ("transcription_preview.wav" if arrangement == "transcription"
                        else "arrangement_preview.wav")
        preview_path = session_dir / preview_name
        if preview_path.exists():
            return FileResponse(str(preview_path), media_type="audio/wav",
                                filename=preview_name)
        # If /previews hasn't run yet, fall through to the normal lazy path below.

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
