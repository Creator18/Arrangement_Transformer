"""
Which signal should feed the beat tracker?

Can HPSS replace Demucs as the harmonic source for beat tracking?

The requirement is NOT "HPSS is accurate against some external truth" — the
whole pipeline is tuned against the grid Demucs currently produces, so the
question is whether HPSS AGREES with Demucs closely enough that nothing
downstream notices.  So Demucs is the reference here, deliberately.

Three sources compared, plus wall-clock for each:
  demucs  — current: 4-stem separation, drums removed, harmonic mix
  hpss    — librosa harmonic/percussive split (the cheap candidate)
  raw     — no separation at all (the free option, for reference)
"""
import sys, time, glob, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, librosa, mir_eval
from app.services import separation as S

EXCERPT = 30.0
N_PIECES = 6

def beats_from(audio, sr):
    rnn, dbn = S._get_madmom()
    act = rnn(librosa.resample(audio, orig_sr=sr, target_sr=44100)
              if sr != 44100 else audio)
    out = dbn(act)
    return np.asarray([b[0] for b in out], dtype=float)

files = sorted(glob.glob('eval/data/audio/*.wav'))[:N_PIECES]
rows = {'demucs': [], 'hpss': [], 'raw': []}
times = {'demucs': 0.0, 'hpss': 0.0, 'raw': 0.0}

for f in files:
    y, sr = librosa.load(f, sr=44100, mono=True, duration=EXCERPT)
    if len(y) < sr * 5:
        continue

    t0 = time.perf_counter()
    try:
        stems = S._get_separator()
        harm = S._demucs_harmonic(y, sr) if hasattr(S, '_demucs_harmonic') else None
    except Exception:
        harm = None
    if harm is None:
        # Fall back to the documented behaviour: full separation, drop drums.
        from demucs.apply import apply_model
        import torch
        model = S._get_separator()
        wav = torch.tensor(np.stack([y, y]))[None]
        with torch.no_grad():
            src = apply_model(model, wav, device='cuda' if torch.cuda.is_available() else 'cpu')[0]
        names = model.sources
        keep = [i for i, n in enumerate(names) if n != 'drums']
        harm = src[keep].sum(0).mean(0).cpu().numpy()
    times['demucs'] += time.perf_counter() - t0
    ref = beats_from(harm, sr)

    t0 = time.perf_counter()
    h, _ = librosa.effects.hpss(y)
    times['hpss'] += time.perf_counter() - t0
    est_h = beats_from(h, sr)

    times['raw'] += 0.0
    est_r = beats_from(y, sr)

    if len(ref) < 3:
        continue
    for name, est in (('demucs', ref), ('hpss', est_h), ('raw', est_r)):
        if len(est) < 3:
            rows[name].append(0.0); continue
        f1 = mir_eval.beat.f_measure(ref, est)
        rows[name].append(f1)

print(f"\npieces: {len(rows['demucs'])}   excerpt: {EXCERPT:.0f}s   reference = demucs grid\n")
print(f"{'source':<10}{'beat F vs demucs':>19}{'separation time':>18}")
print('-' * 47)
for k in ('demucs', 'hpss', 'raw'):
    v = rows[k]
    m = float(np.mean(v)) if v else float('nan')
    print(f"{k:<10}{m:>18.3f}{times[k]:>17.2f}s")


# ── Result, 2026-08-08 (6 pieces, 30s excerpts, synthesised solo piano) ──────
#
#   source   beat F vs demucs   separation time
#   demucs             1.000            10.70s     (self — sanity check)
#   hpss               0.798            13.03s
#   raw                1.000             0.00s
#
# HPSS is a dead end: SLOWER than Demucs (librosa HPSS at 44.1kHz is
# expensive) and it moves the grid.  It was my recommendation and the
# measurement refuted it.
#
# Raw audio reproduces the Demucs grid EXACTLY at zero cost.  On solo piano
# the "drum-less harmonic mix" and the raw mix are the same signal, so
# Demucs spends the pipeline's largest slice producing what it was handed.
#
# Before acting on this: rerun on SONG material, where raw should diverge and
# Demucs should earn its cost, and confirm with the 50-piece transcription
# eval that nothing downstream shifted.
