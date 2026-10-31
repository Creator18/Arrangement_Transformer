# CPU hosting: slim Docker image of the backend

The hosted setup on CPU, with no GPU:
- base 0.8;
- clean-up v4 without beats (keep 0.3, merge 0.5, never-kept 0.7);
- sustain on.

The frontend is a static site and is hosted separately.

## Build

From the repo root:

```bash
# 1. weights-only checkpoints (drops optimizer state: base 227 MB -> 76 MB)
backend/venv/bin/python - <<'EOF'
import torch, pathlib
for src, dst in [("backend/app/ml/spectral/checkpoints/base20m/best.pt", "deploy/models/base20m/best.pt"),
                 ("backend/app/ml/spectral/checkpoints/cleanup4_nobeat/best.pt", "deploy/models/cleanup4_nobeat/best.pt")]:
    s = torch.load(src, map_location="cpu", weights_only=False)
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in s.items() if k not in ("opt", "sched", "scaler")}, dst)
EOF
# 2. slim piano (8 of 16 velocity layers, 44.1 kHz 16-bit FLAC: 715 MB -> 84 MB)
backend/venv/bin/python deploy/slim_salamander.py SalamanderGrandPiano-master/SalamanderGrandPiano-master deploy/piano
# 3. image (776 MB compressed, 3.0 GB unpacked; linux/arm64 here, builds for amd64 too)
docker build -f deploy/Dockerfile -t arrangement-cpu .
docker run --cpus=2 -p 8000:8000 arrangement-cpu
```

## What is left out

| Left out | Why it's safe |
|---|---|
| Demucs | Stems are never separated on the hosted path. |
| madmom | v4 without beats needs no beats. |
| basic-pitch | Not used on the hosted path. |
| `piano_transcription_inference` | Not used on the hosted path. |

At startup the warmup reports these components as missing, which is harmless.

## Image fixes

- **pysfizz:**
  - it has no Linux Arm wheel, and the PyPI source package is missing its submodules, so the image builds tag v0.1.3
    from git;
  - it is installed with `--no-deps`, because otherwise it upgrades numpy to 2.x under SciPy 1.13 and numba.
- **Parallel preview render:** forked render workers segfaulted inside sfizz on Linux. `sfz_sampler.py` now starts
  them with `spawn`, which is what macOS already does.

## Timing at 2 CPUs (2026-10-30)

Setup:
- colima on an M-series MacBook Air, arm64 Linux, `docker run --cpus=2`;
- run with `deploy/bench.py`, which makes the frontend's calls;
- MP3 uploads.

| Song | Length | Notes | Transcribe | Previews | Total | Peak memory |
|---|---|---|---|---|---|---|
| c36cc757 | 2:45 | 879 | 88 s | 6 s | 100 s | 1.9 GB |
| 15ace7a2 | 3:40 | 2,124 | 121 s | 15 s | 137 s | 2.5 GB |
| 242be364 | 4:33 | 2,605 | 153 s | 17 s | 173 s | 2.5 GB |

- **Idle:** 0.43 GB. Startup warmup takes 2.4 s.
- **Speed:** about 0.6 s of work per second of song on two Apple M-series cores. Cloud vCPUs are usually slower per
  core, so plan on about 1–1.5× the song's length per song.
- **Your Windows laptop (2 threads, from CLAUDE.md):** 137 s for a 3:08 song with v4 without beats. The same ballpark.

## Which free host fits

| Host | Fits? |
|---|---|
| **Oracle Cloud Always Free, Arm** | Yes. Now reportedly 2 cores / 12 GB, so one song at a time with plenty of memory to spare. It is the same arm64 build as tested here. |
| **Google Cloud Run** | Yes, with a 4 GiB instance. Rough capacity from the free monthly tier (180k vCPU-s, 360k GB-s): about 430–640 songs a month at 2 vCPU. Cold starts pull the 776 MB image. |
| Render / Koyeb | No. They give 512 MB of RAM; this needs 2.5 GB at peak. |
| Hugging Face Spaces, CPU Basic | No, not for free: Docker Spaces need PRO. |

## Deployment plan (2026-10-31, not started)

### Architecture

- **Backend:** this image.
  - **Oracle Always Free (Arm)** first. It is the same arm64 build as timed above.
  - **Google Cloud Run** as fallback: x86, same Dockerfile, 4 GiB instance.
- **Frontend:** a static build on Vercel, Cloudflare Pages or GitHub Pages. `frontend/src/utils/api.js` reads
  `VITE_API_URL` at build time.
- **Models and piano:** git-ignored, so the image is built locally and pushed to a **private** registry (for example
  GHCR). The server pulls it with a token. Nothing is downloaded at startup.
- **HTTPS:**
  - Cloud Run gives an HTTPS URL automatically.
  - On Oracle, put Caddy in front, with automatic Let's Encrypt and a free hostname such as DuckDNS.

### App changes before going public

These don't depend on the host.

1. **Background jobs.**
   - Arranging a song takes 2–3 minutes inside one HTTP request today.
   - Proxies (Vercel, Cloudflare) cut requests off at about 100 s.
   - Change it to: start a job, then poll its status. That needs changes in both the backend and the frontend.
2. **One song at a time.**
   - A queue, plus a "busy, try again shortly" reply.
   - Never run songs side by side on 2 cores.
3. **Caps.**
   - A maximum song length (for example 6 min), or the start/end range picker from CLAUDE.md.
   - An upload size limit.
   - Basic per-IP rate limiting.
4. **Session cleanup.** Each session keeps about 50 MB of audio, so delete sessions after a few hours.
5. **CORS.** `allow_origins` in `app/main.py` lists `"https://*.vercel.app"`, but that setting doesn't accept wildcards,
   so it never matches. Use the real domain or `allow_origin_regex`.
6. **Link fetch.** YouTube usually blocks cloud IPs ("Sign in to confirm you're not a bot"), so turn it off by default
   and make file upload the main way in.

### Only the user can do

- Create the Oracle or Google Cloud account. Both ask for a card even on the free tier.
- Choose an Oracle region with free Arm capacity.

### Risks

- **Oracle Arm limits:**
  - reportedly cut to 2 cores / 12 GB in June 2026;
  - going over the free quota can get the server deleted;
  - Oracle can reclaim Always Free servers that sit mostly idle.
  - Check Oracle's current docs.
- **Cloud Run cold starts:** the first visitor after a quiet spell waits while the 776 MB image is pulled and the models
  load.
- **Licences:**
  - Salamander is CC-BY, so credit it on the site.
  - Check that POP909 and Music2MIDI allow a public demo of a model trained on them.

### Order

1. App changes, tested in this image.
2. The user picks the host and creates the account.
3. Push the image to a private registry.
4. Start the server, with HTTPS.
5. Deploy the frontend with `VITE_API_URL`.
6. Smoke test with a few songs.
