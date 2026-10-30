# Slim CPU image of the song-to-piano backend (build from the repo root):
#   docker build -f deploy/Dockerfile -t arrangement-cpu .
#   docker run --cpus=2 -p 8000:8000 arrangement-cpu
# Needs deploy/models (weights-only checkpoints) and deploy/piano (slim Salamander); see deploy/README.md.

FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements.txt /tmp/requirements.txt
# CPU-only torch first (the default wheels pull CUDA on x86)
RUN pip install --no-cache-dir --prefix=/install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt
# pysfizz: no Linux Arm wheel, and the PyPI sdist lacks its submodules, so build the tagged source
RUN git clone --depth 1 --branch v0.1.3 --recurse-submodules --shallow-submodules \
        https://github.com/tiianhk/pysfizz /tmp/pysfizz \
    && pip install --no-cache-dir --no-deps --prefix=/install /tmp/pysfizz

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
WORKDIR /srv
COPY backend/app app
COPY deploy/models models
COPY deploy/piano piano
# The hosted setup: base 0.8 + clean-up v4 without beats (keep 0.3, merge 0.5, never-kept 0.7) + sustain
ENV ARRANGEMENT_PIANO_MODE=spectral \
    ARRANGEMENT_PIANO_ONLY=1 \
    ARRANGEMENT_FORCE_SONG=1 \
    ARRANGEMENT_SONG_ARRANGER=models/base20m/best.pt \
    ARRANGEMENT_SONG_ARRANGER_THR=0.8 \
    ARRANGEMENT_SONG_ARRANGER_FRAME_THR=0.5 \
    ARRANGEMENT_SPECTRAL_ONSET_THR=0.85 \
    ARRANGEMENT_CLEANUP=models/cleanup4_nobeat/best.pt \
    ARRANGEMENT_CLEANUP_KEEP=0.3 \
    ARRANGEMENT_CLEANUP_MERGE=0.5 \
    ARRANGEMENT_CLEANUP_NK=0.7 \
    ARRANGEMENT_SUSTAIN=1 \
    ARRANGEMENT_SFZ="/srv/piano/Salamander Grand Piano V3.sfz" \
    ARRANGEMENT_RENDER_WORKERS=2 \
    OMP_NUM_THREADS=2 \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
