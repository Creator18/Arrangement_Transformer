# Arrangement Transformer

Turns a pop song (audio) into a playable piano arrangement. Research code plus a web app (FastAPI backend,
React frontend).

## The idea: keep, don't write

A piano arrangement keeps the melody, compresses the harmony into two hands and leaves out most of what the band
plays. Most audio-to-piano systems *generate* the arrangement token by token. This project does the opposite:

1. **Hear everything an arrangement could keep.** A pitch-aligned spectral transformer (the "arranger") reads the
   song and outputs per-key note onsets, held notes and velocities.
2. **Select, never write.** A note-level clean-up model decides for each detected note whether to keep it, drop it,
   or merge it into the previous strike of the same key (one note heard as two). It never adds a note and never
   extends a note beyond what the arranger heard.
3. **Learn what humans leave out.** Many songs have several human arrangements, and they disagree. Instead of
   imitating any one of them, the clean-up learns only which of our notes **no** human arrangement keeps, and drops
   those.

Early generative attempts (a decoder that wrote arrangements token by token) scored much worse and sounded worse,
which is why the project moved to selection.

## How it works

| Part | What it is | Trained on |
|---|---|---|
| Arranger | 19.9M-parameter transformer; harmonic CQT input; half-octave x 32 ms pitch-aligned tokens; factorised time / pitch attention with learned soft time windows | Renders only: 227,860 clips of POP909 and Music2MIDI arrangements played by 14 instrument families, plus extra parts that are *not* in the target (doublings, pads, counter-lines), drums and loud mastering. Targets are the written notes at their written times |
| Clean-up (keep / drop / merge) | 6-layer transformer over the notes of a 20 s window, reading the arranger's probabilities around each note | Renders, with labels from an exact search for the keep / drop / merge plan closest to the written arrangement |
| "Never kept" drop | Extra output of the clean-up | 840 real Music2MIDI songs; a note is "never kept" if no human arrangement of the song has its pitch class within 150 ms |
| Sustain pedal | Rule: pedal down while the song's sound rings past released keys | n/a |

## Results so far

All numbers come from this repo's research log (`CLAUDE.md`, `NEW_BASE_PLAN.md`). Real-song results use 106 held-out
Music2MIDI songs.

**Arranger, at the app's 0.9 onset bar**

| Model | Dense multi-instrument renders: F1 | Double presses per 1,000 notes | Real songs: strict note F1 (P / R) | Real songs: notes/s (human 8.0) |
|---|---|---|---|---|
| Octave tokens, global attention | 0.785 | 121 | 0.248 (0.234 / 0.279) | 9.1 |
| Learned soft time windows | 0.793 | 102 | 0.245 (0.241 / 0.265) | 8.3 |
| Half-octave tokens | 0.804 | 105 | 0.249 (0.239 / 0.273) | 8.8 |
| **Final arranger** (both) | **0.867** | **32** | **0.251** (0.228 / 0.298) | 10.9 |

**Clean-up, 3,212 held-out render clips** (note F1 with note ends, onset within 50 ms)

| | F1 with ends |
|---|---|
| Arranger output at the 0.8 bar | 0.467 |
| Clean-up, keep / drop only | 0.519 |
| Clean-up with merging | 0.526 |
| Best possible keep / drop / merge (exact search) | 0.619 |

**What humans never keep** (925 real songs, 1,520 human arrangements)

- 32.3% of our notes appear in no human arrangement of the song, against 25.2% for one human arrangement scored
  against the others (the arrangers' own disagreement).
- Short, out-of-key notes next to a sung note (singer slides) are never kept 84.5% of the time.
- The "never kept" output, on 89 held-out songs: AUC 0.72. At the deployed setting it drops 7.8% of notes, about 75%
  of them never kept by any human, while losing 3.0% of the notes humans do keep (11.4 to 10.5 notes per second).

**Solo piano** (the arranger's starting point): note F1 0.960 on 12 real MAESTRO recordings, against 0.966 for the
high-resolution piano transcription model of Kong et al.

### Caveats

- **Strict F1 against one human arrangement stays near 0.25.** Arrangers disagree with each other, so this metric
  understates quality; the multi-arrangement measures above were built partly for this reason.
- **The real-song test set is a carved split.** Only part of Music2MIDI could be downloaded, so these numbers are
  not comparable with published Music2MIDI results.
- **Listening results come from one listener** (the author), without blinding. They are reported as design
  decisions (for example the 0.8 onset bar and the clean-up settings), not as a perceptual study.
- **No head-to-head with published song-to-piano systems yet.** It is planned on the same 106 songs.

## Repository layout

```
backend/
  app/            FastAPI app: routers, services (transcription, clean-up, previews, export)
  app/ml/spectral model, training and clean-up code (cleanup*_train.py)
  eval/           render pipeline, evaluation scripts, SLURM job scripts for the training cluster
  serve.ps1       starts the backend with a chosen arranger / clean-up setup (Windows)
frontend/         React + Vite app (upload, piano roll, sheet music, playback)
CLAUDE.md         research log: what was tried, results, decisions
NEW_BASE_PLAN.md  the arranger's training plan and results
```

## Running it

**Requirements:** Python 3.11, Node 18+, ffmpeg on the PATH.

```bash
cd backend
python -m venv venv
venv/Scripts/activate          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

madmom (used by some clean-up versions) needs `pip install Cython` first, then
`pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom"`. pretty_midi needs
`setuptools<81`.

**Model weights are not included in this repository.** The backend expects the arranger and clean-up checkpoints
under `backend/app/ml/spectral/checkpoints/`. Without them the training code still runs, but the app cannot transcribe
songs. Playback previews use the Salamander Grand Piano (sfizz), downloaded separately.

Start the backend (Windows):

```powershell
.\serve.ps1 -Arranger app/ml/spectral/checkpoints/base20m/best.pt -Threshold 0.8 -Cleanup app/ml/spectral/checkpoints/cleanup4/best.pt -Keep 0.3 -Merge 0.5 -NK 0.7 -Sustain
```

On macOS / Linux, set the same options as environment variables (see `serve.ps1`) and run
`python -m uvicorn app.main:app --port 8000`.

Start the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

## Reproducing the models

Training ran on a SLURM GPU cluster; the job scripts in `backend/eval/*.sh` show the exact commands.

1. Render training clips: `backend/eval/render_base.py` (needs POP909, Music2MIDI and General MIDI soundfonts,
   none of which are included).
2. Train the arranger: `backend/app/ml/spectral/train.py` (settings in `NEW_BASE_PLAN.md`).
3. Clean-up data and merge labels: `backend/eval/cleanup_data.py`, `backend/eval/merge_oracle.py --labels`.
4. Clean-up: `cleanup_train.py`, then `cleanup2_train.py`, `cleanup3_train.py`, `cleanup4_train.py` (each fine-tunes
   the previous one).
5. Human-arrangement labels: `backend/eval/never_kept.py`, then `backend/eval/human_data.py`.

## Data

Datasets are not redistributed here. POP909 and Music2MIDI must be obtained from their authors. Music2MIDI provides
arrangements and video links; song audio is not part of this repository.
