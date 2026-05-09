# Arrangement Transformer

> Any piano piece. Any keyboard.

A tool that takes a piano audio file, transcribes it using ML, and transforms it into an arrangement that fits a different octave range (typically 7-octave to 5-octave keyboard), while maintaining maximum musical fidelity.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React, Tailwind CSS, Framer Motion |
| Backend | FastAPI (Python 3.10+) |
| Transcription | Basic Pitch (Spotify) |
| Stem Separation | Spleeter (Deezer) |
| Audio Analysis | Librosa, FFmpeg |
| Sheet Music | music21, pretty-midi |
| Deployment | Frontend → Vercel · Backend → HuggingFace Spaces |

## Quick Start (Windows PowerShell)

### Backend
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend (open a second terminal)
```powershell
cd frontend
npm install
npm run dev
```

> **Note:** PowerShell 5.1 (Windows default) does not support `&&`. Run each command on its own line as shown above.

## Pipeline

```
Upload audio
→ Convert to WAV (FFmpeg)
→ Transcribe (Basic Pitch) → note list + octave range
→ Separate stems (Spleeter) → bass + melody/voice
→ Detect tempo + time signature (Librosa)
→ Segment by measures
→ Per segment: mel spectrograms → JS divergence → star detection
→ Transform arrangement (priority hierarchy + ghost notes + tempo-aware fills)
→ Score fidelity (JS divergence on full arrangement spectrograms)
→ Export: PDF + MIDI + CDEF
```

## CDEF Format

A custom human-readable notation format:

```
RH: C4-E4 . G4 . | F4 A4 . C5 |
LH: C2 . . G2 | F2 . . C2 |
```

- Each beat = one unit of horizontal space  
- Notes: letter + octave (C4, D#4, G3)  
- Multiple notes in one beat: dash-separated (C4-E4-G4)  
- Rest: `.`  
- Measure separator: `|`  
- `RH` = right hand (melody + voice), `LH` = left hand (bass)
