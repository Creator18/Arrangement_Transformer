"""
Song -> piano by projection onto an 88-key dictionary.

Onsets & Frames is trained on MAESTRO, solo piano.  Pointed at a song it does
not fail gracefully — it is answering "what piano notes are in this audio" when
the honest question is "what piano notes best APPROXIMATE this audio", and for
a guitar chord or a vocal line those are different questions.

This answers the second one.  Render all 88 keys once as spectral templates,
then decompose the song's magnitude spectrogram against that fixed dictionary
with KL multiplicative updates.  Whatever activates is, by construction, the
best piano approximation available — arrangement rather than transcription.

The dictionary is HARMONIC, not piano
-------------------------------------
The obvious dictionary is 88 rendered piano notes, and that was tried first.
It loses, because it matches on partial AMPLITUDES — which is the definition of
timbre, and so the one part of a note that does not survive changing
instrument.  Partial POSITIONS do survive: a C4 puts energy at f0, 2f0, 3f0
whether it is sung, plucked or struck.  build_harmonic_dictionary keeps the
positions and discards the amplitudes.

Measured on held-out pieces (backend/eval/harmonic_dict_test.py), as "% of
activation energy landing on a pitch that is sounding or ringing":

    voice              piano dict   harmonic dict     octave ghosting
    grand piano            64.8%          69.6%        5.4% -> 5.1%
    nylon guitar           61.7%          64.7%        7.0% -> 5.8%
    string ensemble        47.9%          64.9%       10.6% -> 3.3%
    choir aahs             34.4%          45.1%       12.9% -> 7.3%
    church organ           41.5%          57.2%       15.9% -> 10.8%

It wins on every instrument and every column, sustained sources most, and
octave ghosting FALLS everywhere.  The sub-octave ambiguity a comb is supposed
to suffer from does not materialise under KL divergence: a comb at f0/2 has
teeth at 1.5*f0 and 2.5*f0 where the signal has nothing, and predicting absent
energy is penalised.

On a melody line it also beats pYIN (frame pitch accuracy 60.2% vs 48.7%, and
54.5% vs 46.7% with vibrato), so the melody does not need a separate
monophonic pitch tracker.  The piano dictionary's melody failure was almost
purely octave error — pitch-class accuracy 63.0% against exact-pitch 33.1%,
i.e. the right note in the wrong register.

DRUMS MUST STILL BE REMOVED
---------------------------
Percussion costs 20.5 points on a mixture and Demucs recovers essentially all
of it (63.0% vs 63.3% for audio that never had drums).  It does not corrupt
pitch — ghosting is flat across the drum conditions — but it raises a broadband
floor that swamps quiet notes.  So this reads the drumless stems separation
already wrote, never the raw mix.

The extractor is the bottleneck, not the projection
---------------------------------------------------
Note-level scores, tuned and tested on disjoint pieces: COnP 0.575 on matched
piano, 0.375 on a full mixture, with precision 0.92 and recall 0.42 in the
matched case.  It is precise and it under-reads.  That profile is the right way
round for an ARRANGEMENT — a piano reduction has to drop notes anyway, and
what comes out is mostly correct — but it is well short of what the
decomposition supports, and it is the obvious place for the next improvement.
Everything it ignores is already in the repo: the madmom beat grid, spectral
flux, and the harmonic relationship between simultaneously active rows.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

# These must match what backend/eval measured; the eval scripts import them
# from here rather than keeping their own copies, because two drifting copies
# of a feature definition is exactly how a measurement stops describing the
# thing that ships.
SR = 22050
N_FFT = 8192          # 2.7 Hz bins — A0 is 27.5 Hz, so the window has to be
HOP = 512             # long enough to resolve the bottom octave at all
FPS = SR / HOP
LOW, HIGH = 21, 108
N_PITCH = HIGH - LOW + 1

# Re-tuned on REAL uploads, not synthetic renders (eval/threshold_retune.py).
#
# The reasoning that produced 0.90 was wrong.  The claim was that real audio
# carries noise and room, so more activations clear the bar; measured, real
# uploads are LESS dense than synthetic mixtures at the same q — 8.06 notes/s
# against 10.98 at q=0.90.  Density is monotonic in q, though, so it transfers
# an operating point across domains even when the absolute levels differ:
#
#   synthetic, with ground truth   F1 peaks at q=0.96, emitting 4.98 notes/s
#   real uploads                   4.98 notes/s arrives at q=0.94
#
# Note that peak is BELOW what a pianist plays (MAESTRO averages 7.64 notes/s).
# That is not a bug to correct — past this point the notes the extractor adds
# are mostly wrong, so matching a human's note rate would buy density with
# precision, which is the trade that made the output sound noisy.
THRESH_Q = 0.94       # activation quantile

# One decomposition of the drumless mix, two voices decoded from it with a
# non-crossing constraint.  Supersedes PER_STEM, which is kept as a fallback so
# the two can be compared rather than only replaced.
TWO_VOICE = True
PER_STEM = True           # route each Demucs stem to its own method

# Reduce the harmony HERE rather than at /transform.
#
# The two monophonic decodes (vocals->melody, bass->bass) impose a model — one
# note at a time, a dwell cost, onsets pulled to the beat — and by ear they are
# the parts that work.  The harmony was peak-picked instead, which imposes
# nothing and reports every activation over a threshold: ~2,458 notes, about
# two thirds of everything, 88% of them at or above G#3, piling into the same
# register as the tune and masking it.
#
# It was only ever thinned by the arpeggiator at transform time, so the
# full-range output the UI calls "Original" was a dump rather than an
# arrangement — 2,525 notes above C4 with 14 sounding at once.  Generating the
# figure at the source makes both views arrangements at different keyboard
# sizes, and then compression really is only compression.
REDUCE_HARMONY = True

# NO GENERATED ACCOMPANIMENT.
#
# The figure was interfering rather than accompanying, and by ear it was
# actively damaging the piece.  Two measured reasons it could not have worked
# yet:
#
#   Its harmony is wrong often enough to matter.  With the key finally detected
#   correctly (E major), the generated notes taken alone still measure G major —
#   chord inference is picking out-of-key roots, because it reads peak-picked
#   notes at recall 0.42 rather than the clean voicings it scores 76% on.
#
#   It plays the same figure in every bar regardless of the vocal.  A human
#   arranger thins under the singing and fills the rests, and POP909 says the
#   vocal is silent 42% of the time.  Constant density reads as added material,
#   which is exactly what it sounded like.
#
# So the harmony stems are not decomposed at all: melody from the vocals stem
# and bass from the bass stem, which are the two parts that work.  A lead line
# over a bass line is a legitimate reduction, not a placeholder.  It is also
# most of the speed problem — the harmony run used the full 88-column
# dictionary, the widest of the three.
INCLUDE_HARMONY = False

WHITEN_NU = 0.0       # spectral whitening; measured as a strictly worse lever
                      # than N_HARMONICS below, so it is left off

# How many partials each template carries — the lever on the register collapse.
#
# At 40, a bottom-octave template packs forty teeth into the lowest ~1 kHz,
# precisely where a produced record keeps its energy, so it can explain almost
# any low-frequency content; a treble template gets a handful of sparse teeth
# and cannot compete.  Measured on real uploads, that put 95% of all emitted
# notes below G#3.
#
#   n_harm   synth F1   synth bass%   REAL bass%   real notes/s
#       40      0.451          34%          95%          4.78
#       24      0.437          31%          73%          5.86
#       16      0.350          35%          60%          7.10
#       12      0.324          36%          49%          8.16
#        8      0.351          36%          46%          9.33
#        6      0.343          32%          48%          8.68
#
# Note what the synthetic column is and is not saying.  Synthetic bass share
# sits at 31-36% for EVERY setting — those mixtures never had the collapse, so
# synthetic F1 cannot reward fixing it and only charges for the reduced
# expressiveness.  "Keep the setting where F1 is intact" would therefore
# optimise for the one domain that does not have the bug, and would leave real
# audio at 95%.
#
# 8 is chosen: the best register balance on real audio, and the best F1 among
# the settings that achieve one.  The cost is real and worth stating — roughly
# 0.10 of synthetic F1 against a 95% -> 46% correction on the material that
# actually gets uploaded.
N_HARMONICS = 8
MIN_DUR = 0.05        # shorter than this is a peak-picking artefact
SMOOTH = 3            # frames of moving average per activation row
NMF_ITERS = 150

# Which dictionary the projection runs on.  'harmonic' is the default because
# it measured better on every instrument and every metric — see
# backend/eval/harmonic_dict_test.py.  Override with ARRANGEMENT_DICTIONARY=piano
# to A/B by ear without a code change.
DICTIONARY = 'harmonic'          # 'harmonic' | 'piano'
HARMONIC_ROLLOFF = 1.5           # chosen on held-out pieces, not by eye

_DICT_PATH = pathlib.Path(__file__).with_name("_piano_dict88.npy")
_W: np.ndarray | None = None
_W_KIND: str | None = None
_W_BY_FFT: dict = {}


def _to_mel(W: np.ndarray, n_fft: int) -> np.ndarray:
    """Project a linear-frequency dictionary through the mel filterbank.

    Applied to the DICTIONARY as well as the signal, deliberately.  NMF solves
    V ~ W H, and the filterbank M is linear, so M V ~ (M W) H has the same
    solution H — projecting both sides keeps the two consistent.  Rebuilding
    the combs directly on mel centre frequencies would work too but risks the
    two sides drifting apart, which is the failure Test 1 was built to catch.
    """
    if not USE_MEL:
        return W
    Wm = mel_filterbank(n_fft) @ W
    n = np.linalg.norm(Wm, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return (Wm / n).astype(np.float32)


def get_dictionary_for(n_fft: int) -> np.ndarray:
    """Harmonic dictionary at a given FFT size, memoised."""
    if n_fft == N_FFT:
        return get_dictionary()
    if n_fft not in _W_BY_FFT:
        _W_BY_FFT[n_fft] = _to_mel(build_harmonic_dictionary(
            HARMONIC_ROLLOFF, N_HARMONICS, n_fft), n_fft)
    return _W_BY_FFT[n_fft]


# FFT size per stem.  N_FFT = 8192 gives 2.7 Hz bins, which the BOTTOM octave
# genuinely needs — A0 is 27.5 Hz and a coarser window cannot resolve it.
# Nothing above E2 (82 Hz) needs that, and the FFT size is what makes the NMF
# matrices 4097 rows deep, which is where the cost actually lives: restricting
# the dictionary to a stem's own pitch range saved only 7%, because W @ H and
# V / (WH) are (bins x T) regardless of how many columns W has.
#
# So the lever is bins, not columns.  Halving the FFT for the vocals stem halves
# its matrices outright.
# ── mel front end ────────────────────────────────────────────────────────────
#
# A linear FFT spends its resolution uniformly: 2.7 Hz everywhere.  To separate
# adjacent semitones you need better than half a semitone, which is ~0.8 Hz at
# A0 and ~145 Hz at 5 kHz.  So linear is THREE TIMES TOO COARSE at the bottom
# and FIFTY TIMES FINER THAN NEEDED at the top — thousands of bins resolving
# detail no note has.
#
# Mel spacing puts the resolution where pitch discrimination needs it, which
# also removes the reason the two registers used different FFT sizes.
#
# 1024 bins, not 512: mel is only logarithmic ABOVE ~1 kHz and roughly linear
# below, so 512 would give ~5.5 Hz in the bass — the same marginal resolution
# as the 4096 FFT that cannot separate E2 from F2 (4.9 Hz apart).  1024 keeps
# ~2.7 Hz down low while still being 4x smaller than 4097.
#
# The STFT stays at n_fft=8192: a filterbank cannot manufacture resolution the
# transform never had.  Only the REPRESENTATION shrinks, and the NMF cost is
# dominated by (bins x T), so that is where the saving comes from.
# MEASURED AND REJECTED — kept so the comparison can be re-run.
#
#                     correct   decay   octave-ghost   fifth   other   accounted
#   LINEAR 4097 bins    41.3%   47.4%           2.0%    1.1%    8.2%       88.7%
#   MEL    1024 bins    25.8%   37.7%           5.1%    3.5%   27.9%       63.5%
#
# The argument for mel was that resolution above 1 kHz is fifty times finer
# than needed, since semitones up there are ~145 Hz apart.  True of
# FUNDAMENTALS, and irrelevant: the upper HARMONICS of low notes live up there
# too, spaced by their fundamental rather than by the local semitone.  A 40 Hz
# bass note has harmonics 40 Hz apart at 4 kHz.  Mel's wide bins smear exactly
# the comb structure that distinguishes a note from its octave.
#
# Note the trap in the summary statistic: the mel dictionary is LESS
# self-similar by the obvious measure — max cos(p, p+12) falls from 0.932 to
# 0.619 — and discriminates worse anyway.  What is lost is precise harmonic
# POSITION, which that metric does not see.
#
# The general lesson: a front end suited to a LEARNED model is not suited to a
# TEMPLATE DECOMPOSITION.  Mirelo can use mel because their network learns
# whatever structure survives it; matching cannot, because it depends on the
# positions mel destroys.
import os as _os
USE_MEL = _os.environ.get('FORCE_MEL', '0') == '1'
MEL_BINS = 1024

N_FFT_HIGH = 4096     # 5.4 Hz bins — ample from E2 upward
HIGH_FFT_MIN_PITCH = 40   # E2; below this, 8192 is required


_MEL_FB: dict = {}


def mel_filterbank(n_fft: int) -> np.ndarray:
    """(MEL_BINS, 1+n_fft/2) filterbank, memoised."""
    import librosa
    if n_fft not in _MEL_FB:
        _MEL_FB[n_fft] = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=MEL_BINS,
                                             norm=None).astype(np.float32)
    return _MEL_FB[n_fft]


def stft_mag(y: np.ndarray, n_fft: int = None) -> np.ndarray:
    import librosa
    n_fft = n_fft or N_FFT
    V = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP))
    return (mel_filterbank(n_fft) @ V) if USE_MEL else V


def build_dictionary(velocity: int = 80) -> np.ndarray:
    """One spectral template per key, taken from the sustain, not the attack."""
    # render_notes, NOT render_notes_sfz: the shipped chain applies tone
    # shaping and the room, and every measurement in backend/eval was made
    # against templates that carry them.  Building from the dry sampler here
    # would ship a different dictionary than the one the numbers describe —
    # the same mistake compare_pianos.py made by bypassing the reverb stage.
    from app.services.sampler import render_notes
    cols = []
    for p in range(LOW, HIGH + 1):
        y = np.asarray(render_notes(
            [{"pitch": p, "onset": 0.0, "offset": 1.5, "velocity": velocity}],
            sr=SR), dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        S = stft_mag(y)
        # Skip the attack: the transient is broadband and shared across every
        # key, so including it makes all 88 templates look alike exactly where
        # they need to differ.
        a, b = int(0.08 * FPS), int(1.1 * FPS)
        col = S[:, a:min(b, S.shape[1])].mean(axis=1) if S.shape[1] > a else S.mean(axis=1)
        n = np.linalg.norm(col)
        cols.append(col / n if n > 1e-9 else col)
    return np.stack(cols, axis=1)          # (freq, 88)


def build_harmonic_dictionary(rolloff: float = 1.0,
                              n_harmonics: int = None,
                              n_fft: int = None) -> np.ndarray:
    """Timbre-neutral templates: a harmonic comb per pitch, no instrument.

    The piano dictionary matches on partial AMPLITUDES, which is the part of a
    note that changes with the instrument — so it is matching on precisely the
    thing that does not generalise.  Partial POSITIONS do generalise: a C4 puts
    energy at f0, 2f0, 3f0 whether it is sung, plucked or struck.  These
    templates keep the positions and discard the amplitudes, replacing them
    with a generic 1/h**rolloff decay.

    The known cost is octave confusion.  A comb at f0/2 contains every harmonic
    of f0 plus extra teeth between them, so a sub-octave can always explain a
    note at least as well.  The piano dictionary resisted that because its
    partial amplitudes are distinctive enough to break the tie — which is
    exactly the information being thrown away here.  Whether the trade is worth
    it is what backend/eval/harmonic_dict_test.py measures.
    """
    import librosa
    n_harmonics = N_HARMONICS if n_harmonics is None else n_harmonics
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft or N_FFT)
    cols = []
    for p in range(LOW, HIGH + 1):
        f0 = 440.0 * 2.0 ** ((p - 69) / 12.0)
        col = np.zeros_like(freqs)
        for h in range(1, n_harmonics + 1):
            f = f0 * h
            if f >= freqs[-1]:
                break
            # Width grows with frequency: a fixed number of CENTS, not of Hz,
            # so a slightly mistuned source stays matched right up the series
            # instead of drifting off the template in the top octaves.
            sigma = max(2.0 * (freqs[1] - freqs[0]), f * 0.012)
            col += (1.0 / h ** rolloff) * np.exp(-0.5 * ((freqs - f) / sigma) ** 2)
        n = np.linalg.norm(col)
        cols.append(col / n if n > 1e-9 else col)
    return np.stack(cols, axis=1)


def get_dictionary() -> np.ndarray:
    """The active dictionary, memoised.

    The harmonic one is computed from a formula in well under a second, so it
    is never cached to disk — only the piano one, which costs 88 sampler
    renders, is worth a cache file.
    """
    global _W, _W_KIND
    import os
    kind = os.environ.get('ARRANGEMENT_DICTIONARY', DICTIONARY).lower()
    if _W is not None and _W_KIND == kind:
        return _W

    if kind == 'harmonic':
        W = build_harmonic_dictionary(HARMONIC_ROLLOFF)
        _W, _W_KIND = _to_mel(W, N_FFT), kind
        return _W

    if _DICT_PATH.exists():
        try:
            _W, _W_KIND = np.load(_DICT_PATH), kind
            return _W
        except Exception:                                   # noqa: BLE001
            pass
    _W, _W_KIND = build_dictionary(), kind
    try:
        np.save(_DICT_PATH, _W)
    except Exception as e:                                  # noqa: BLE001
        print(f"[projection] could not cache dictionary: {e}")
    return _W


def whiten(V: np.ndarray, nu: float = WHITEN_NU) -> np.ndarray:
    """Flatten the mix's spectral tilt before decomposing.

    Real records are strongly bottom-heavy, and the dictionary's low templates
    pack forty harmonics into the bottom couple of kHz — exactly where that
    energy is.  So low templates are the cheapest explanation for almost
    anything, and NMF takes it: measured on real uploads, 94% of all emitted
    notes landed below G#3, at every threshold, where synthetic mixtures ran
    roughly 50/50.  No threshold fixes that, because it is a ratio and a
    threshold only moves a level.

    Dividing by a log-frequency-smoothed envelope removes the tilt while
    leaving the peaks that carry pitch, so templates compete on how well they
    explain STRUCTURE rather than on where the energy happens to sit.  nu=0 is
    no whitening, nu=1 is full; partial is usual, since flattening completely
    also amplifies noise in near-empty bands.
    """
    if nu <= 0:
        return V
    env = V.mean(axis=1)
    n = len(env)
    # Smooth in log-frequency: constant width in octaves, not in Hz, so the
    # bass is not smoothed into a single number while the treble stays spiky.
    idx = np.arange(1, n)
    edges = np.unique(np.geomspace(1, n - 1, 48).astype(int))
    sm = env.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            sm[a:b] = env[a:b].mean()
    sm[0] = env[0]
    sm = np.maximum(sm, env.max() * 1e-6)
    return (V / sm[:, None] ** nu).astype(np.float32)


def nmf_h(V: np.ndarray, W: np.ndarray, iters: int = NMF_ITERS) -> np.ndarray:
    """KL multiplicative updates for H with W held fixed.

    KL rather than Frobenius because the target is a magnitude spectrogram,
    where the loud partials would otherwise dominate the fit and quiet inner
    voices would be discarded outright.  W is given, so only H is solved for,
    which is why this is a few lines rather than sklearn's NMF.
    """
    eps = 1e-9
    H = np.full((W.shape[1], V.shape[1]), V.mean() / W.shape[1] + eps)
    Wt = W.T
    ones = Wt @ np.ones_like(V)
    for _ in range(iters):
        H *= (Wt @ (V / (W @ H + eps))) / (ones + eps)
    return H


def extract_notes(H: np.ndarray, thresh_q: float = THRESH_Q,
                  off_frac: float = 0.35) -> List[tuple]:
    """Activation matrix -> [(onset_s, offset_s, midi_pitch), ...]."""
    if SMOOTH > 1:
        k = np.ones(SMOOTH) / SMOOTH
        H = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, H)

    pos = H[H > H.max() * 1e-4]
    if pos.size == 0:
        return []
    thresh = float(np.quantile(pos, thresh_q))

    notes = []
    for i in range(H.shape[0]):
        row = H[i]
        active = row > thresh
        if not active.any():
            continue
        edges = np.diff(active.astype(np.int8))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if active[0]:
            starts.insert(0, 0)
        if active[-1]:
            ends.append(len(row))

        for s, e in zip(starts, ends):
            seg = row[s:e]
            if seg.size == 0:
                continue
            # Split each run where the row climbs again, so a repeated key does
            # not read as one held note — the failure the piano roll makes
            # most visible.
            d = np.diff(seg, prepend=seg[0])
            cuts = [0]
            for r in np.where(d > off_frac * seg.max())[0]:
                if r - cuts[-1] >= max(2, int(MIN_DUR * FPS)):
                    cuts.append(int(r))
            cuts.append(len(seg))
            for a, b in zip(cuts[:-1], cuts[1:]):
                on_t, off_t = (s + a) / FPS, (s + b) / FPS
                if off_t - on_t >= MIN_DUR:
                    notes.append((on_t, off_t, i + LOW))
    return notes


def velocities(H: np.ndarray, notes: List[tuple]) -> List[int]:
    """Peak activation -> MIDI velocity.

    Activation height is the only dynamic information the decomposition
    carries, and a flat velocity makes an arrangement sound like a punched
    piano roll, so it is what the dynamics come from.
    """
    peaks = []
    for on, off, p in notes:
        a, b = int(on * FPS), max(int(off * FPS), int(on * FPS) + 1)
        row = H[p - LOW, a:b]
        peaks.append(float(row.max()) if row.size else 0.0)
    if not peaks:
        return []
    lo, hi = np.percentile(peaks, 5), np.percentile(peaks, 95)
    out = []
    for v in peaks:
        t = 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0, 1))
        out.append(int(round(40 + 70 * t)))          # 40..110, a musical range
    return out


def project_notes(y: np.ndarray, sr: int,
                  beat_times=None, downbeats=None,
                  beats_per_bar: int = 4) -> List[Dict[str, Any]]:
    """Audio (already drumless) -> note dicts in the pipeline's shape.

    The melody register is DECODED rather than peak-picked when a beat grid is
    available.  That has to happen here, not downstream, because it needs the
    activation matrix — and this is the only place that exists.  Peak-picking
    still handles everything below the melody split, where polyphony is the
    point and a single-line decode would be wrong.
    """
    import librosa
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(np.asarray(y, dtype=np.float32),
                             orig_sr=sr, target_sr=SR)
    H = nmf_h(stft_mag(np.asarray(y, dtype=np.float32)), get_dictionary())
    est = sorted(extract_notes(H), key=lambda n: (n[0], n[2]))
    vels = velocities(H, est)
    notes = [{"pitch": int(p), "onset": float(on), "offset": float(off),
              "velocity": int(v)}
             for (on, off, p), v in zip(est, vels)]

    try:
        from app.services import melody_decode
        if melody_decode.DECODE_MELODY and beat_times is not None:
            line = melody_decode.decode(H, LOW, beat_times, downbeats,
                                        beats_per_bar)
            if line:
                kept = [n for n in notes if n["pitch"] < melody_decode.MEL_LO]
                print(f"[projection] melody decoded: {len(line)} notes replace "
                      f"{len(notes) - len(kept)} peak-picked")
                notes = sorted(kept + line,
                               key=lambda n: (n["onset"], n["pitch"]))
    except Exception as e:                                  # noqa: BLE001
        # A failed decode must not cost the whole arrangement — fall back to
        # the peak-picked notes, which are what shipped before this existed.
        print(f"[projection] melody decode skipped ({type(e).__name__}: {e})")
    return notes


HARMONY_STEMS = ("guitar", "other", "piano")

# Imported here rather than inside the function so the ranges the decoder
# will use are the same ones the decomposition is restricted to.
from app.services import melody_decode as MD      # noqa: E402
from app.services.melody_decode import (          # noqa: E402
    VOCAL_LO as MD_VOCAL_LO, VOCAL_HI as MD_VOCAL_HI,
    BASS_LO as MD_BASS_LO, BASS_HI as MD_BASS_HI)


def _activations(path: pathlib.Path, lo: int = None, hi: int = None):
    """Load a stem and decompose it. None if the file is missing or silent.

    `lo`/`hi` restrict the decomposition to the pitches that stem can actually
    contain, which is free speed: NMF cost scales with the number of dictionary
    columns, and a bass stem needs 32 of the 88 rather than all of them.  The
    result is padded back to full width so callers keep the (88, T) contract —
    melody_decode slices by absolute MIDI number and must not have to care.
    """
    import librosa
    if not path.exists():
        return None
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    pk = float(np.abs(y).max())
    if pk < 1e-4:
        return None                      # an empty stem, e.g. piano on a rock track
    n_fft = (N_FFT_HIGH if lo is not None and int(lo) >= HIGH_FFT_MIN_PITCH
             else N_FFT)
    V = stft_mag((y / pk).astype(np.float32), n_fft)
    W = get_dictionary_for(n_fft)
    if lo is None or hi is None:
        return nmf_h(V, W)
    a = max(0, int(lo) - LOW)
    b = min(W.shape[1], int(hi) - LOW + 1)
    H = nmf_h(V, W[:, a:b])
    out = np.zeros((W.shape[1], H.shape[1]), dtype=H.dtype)
    out[a:b] = H
    return out


def _activations_mixed(paths):
    """Mix several stems, then decompose ONCE."""
    import librosa
    sigs = []
    for path in paths:
        if not path.exists():
            continue
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        if float(np.abs(y).max()) >= 1e-4:
            sigs.append(y)
    if not sigs:
        return None
    n = max(len(y) for y in sigs)
    mix = np.zeros(n, dtype=np.float32)
    for y in sigs:
        mix[: len(y)] += y
    pk = float(np.abs(mix).max())
    if pk < 1e-4:
        return None
    return nmf_h(stft_mag((mix / pk).astype(np.float32)), get_dictionary())


def key_from_activations(*Hs) -> Optional[tuple]:
    """Estimate (root, mode) by folding activation energy into pitch classes.

    Reads the key off the DECOMPOSITION rather than off decoded notes.  The
    decoder is lossy on purpose — it commits to one pitch per frame, drops notes
    under MIN_NOTE_S, and is bounded to a register — so by the time we have
    notes most of the harmonic evidence has been thrown away.  The activation
    matrix still has all of it.  Measured against POP909's human key labels on
    30 s excerpts: decoded notes 27.5% exact / 72.5% allowing relative-key
    confusion, this 75.0% / 85.0%.

    Each stem is normalised before pooling.  `_activations` peak-normalises
    every stem independently, so the raw magnitudes carry no common unit and
    summing them would just let the loudest stem cast the whole vote.

    This is deliberately NOT used to constrain the dictionary.  Penalising or
    removing out-of-scale templates was measured (eval/scale_penalty_test.py)
    and made the decomposition monotonically worse even given a perfect key.
    The key is for choosing among notes we already have, downstream.
    """
    from app.services.nmf_voice_leading import key_from_pitch_classes
    pcv = np.zeros(12, dtype=np.float64)
    for H in Hs:
        if H is None:
            continue
        tot = float(H.sum())
        if tot <= 0:
            continue
        rows = H.sum(axis=1) / tot
        for i in range(rows.shape[0]):
            pcv[(LOW + i) % 12] += float(rows[i])
    return key_from_pitch_classes(pcv)


def project_by_stem(comp_dir: pathlib.Path, beat_times=None, downbeats=None,
                    beats_per_bar: int = 4):
    """Route each Demucs stem to the method built for it.

    Everything used to be summed into harmonic.wav and then split by frequency
    (240 Hz) or pitch (MIDI 56).  Neither is a source separation, so the melody
    competed with a guitar for the same activations and any vocal note below G#3
    was handed to the accompaniment rebuild and overwritten — measured at ~30% of
    the vocal on a real upload.

    Notes come back tagged with a ROLE, not a hand.  Role is musical and comes
    from the stem; hand is physical and is assigned last by
    enforce_hand_constraints, which can see the whole texture's geometry.  That
    also gives the reduction stage a priority order it never had:
    melody > bass > harmony, instead of pruning by whichever note was loudest.
    """
    from app.services import melody_decode as MD

    out: List[Dict[str, Any]] = []
    used = []

    # melody — the vocals stem, decoded over the full vocal register
    Hv = _activations(comp_dir / "stem_vocals.wav",
                      MD_VOCAL_LO, MD_VOCAL_HI)
    if Hv is not None and beat_times is not None:
        line = MD.decode(Hv, LOW, beat_times, downbeats, beats_per_bar,
                         lo=MD.VOCAL_LO, hi=MD.VOCAL_HI)
        for n in line:
            n["role"] = "melody"
        out += line
        used.append(f"vocals->melody {len(line)}")

    # bass — monophonic by nature, and the cleanest stem Demucs produces
    Hb = _activations(comp_dir / "stem_bass.wav",
                      MD_BASS_LO, MD_BASS_HI)
    if Hb is not None and beat_times is not None:
        line = MD.decode(Hb, LOW, beat_times, downbeats, beats_per_bar,
                         lo=MD.BASS_LO, hi=MD.BASS_HI)
        for n in line:
            n["role"] = "bass"
        out += line
        used.append(f"bass->bass {len(line)}")

    # harmony — guitar + other + piano, peak-picked.  These feed chord
    # inference and are then REPLACED by the arpeggiated figure, so their
    # individual pitches matter less than the chroma they imply.
    #
    # Summed in the AUDIO domain, not the activation domain.  Decomposing each
    # stem separately and adding the results cost three NMF runs to produce
    # something one run gives: per-stem routing had quietly turned one
    # decomposition into four, and at ~0.33 s of compute per second of audio
    # that took a 4:44 track from ~94 s to ~380 s. Mixing first is also more
    # correct — a chord voiced across guitar and keys is one harmonic event, and
    # decomposing it as two competing explanations is not what we want.
    Hh = (_activations_mixed([comp_dir / f"stem_{n}.wav" for n in HARMONY_STEMS])
          if INCLUDE_HARMONY else None)
    if Hh is not None:
        est = sorted(extract_notes(Hh), key=lambda n: (n[0], n[2]))
        vels = velocities(Hh, est)
        harm = [{"pitch": int(p), "onset": float(a), "offset": float(b),
                 "velocity": int(v), "role": "harmony"}
                for (a, b, p), v in zip(est, vels)]
        out += harm
        used.append(f"harmony->{len(harm)}")

    if not out:
        return None
    out.sort(key=lambda n: (n["onset"], n["pitch"]))

    if REDUCE_HARMONY and INCLUDE_HARMONY and beat_times is not None:
        try:
            from app.services.arpeggiator import arpeggiate
            from app.services.transformation import enforce_melody_on_top
            from app.services.nmf_voice_leading import detect_scale
            from app.services.music_theory import _scale_pitch_classes
            # Detect the key from DETECTED notes only — never from generated
            # ones.  The harmony written below is invented material, and once it
            # lands in transcription.json it is 52% of all notes, so letting it
            # vote turns key detection into a poll of our own output.  Measured
            # on a real upload: melody+bass alone gave E major (correct), the
            # generated harmony gave G major, and all notes together gave A
            # major — the average of a right and a wrong answer, which then had
            # drop_non_scale enforcing a key the song was never in and deleting
            # the tune's own leading tone.
            detected = [n for n in out if n.get("role") in ("melody", "bass")]
            key = key_from_activations(Hv, Hb)
            if key is None:
                root, scale_name, _ = detect_scale(detected or out)
            else:
                root, scale_name = key
            try:
                scale_pcs = _scale_pitch_classes(root, scale_name)
            except Exception:                                   # noqa: BLE001
                scale_pcs = None
            before = len(out)
            out = arpeggiate(out, np.asarray(beat_times, float),
                             np.asarray(downbeats, float) if downbeats is not None else None,
                             beats_per_bar, scale_pcs, root)
            out = enforce_melody_on_top(out)
            out.sort(key=lambda n: (n["onset"], n["pitch"]))
            for n in out:
                n["key_root"], n["key_scale"] = int(root), str(scale_name)
            used.append(f"harmony reduced {before}->{len(out)}, "
                        f"key {root} {scale_name} from {len(detected)} detected notes")
        except Exception as e:                                  # noqa: BLE001
            print(f"[projection] upstream harmony reduction skipped "
                  f"({type(e).__name__}: {e})")

    if not INCLUDE_HARMONY:
        # Still establish and stamp the key, or /transform re-derives it.
        try:
            from app.services.nmf_voice_leading import detect_scale
            key, src = key_from_activations(Hv, Hb), "activations"
            if key is None:                       # both stems missing or silent
                root, scale_name, _ = detect_scale(out)
                src = "decoded notes (fallback)"
            else:
                root, scale_name = key
            for n in out:
                n["key_root"], n["key_scale"] = int(root), str(scale_name)
            used.append(f"key {root} {scale_name} from {src}, "
                        f"no generated accompaniment")
        except Exception as e:                                  # noqa: BLE001
            print(f"[projection] key detection skipped ({type(e).__name__}: {e})")

    print(f"[projection] per-stem: {', '.join(used)}")
    return out


def project_two_voices(comp_dir: pathlib.Path, beat_times=None, downbeats=None,
                       beats_per_bar: int = 4):
    """One decomposition of the drumless mix, two voices decoded from it.

    The unified path.  Per-stem routing solved melody and bass against DIFFERENT
    evidence — separate Demucs stems, separately peak-normalised, separately
    decomposed, at two FFT sizes — and then merged two answers that had never
    been reconciled.  Everything downstream assumes one grid, and the parts kept
    disagreeing about the register they share.

    Here the whole thing rests on a single (88, T) matrix, so the two voices are
    competing for the same energy and the non-crossing constraint has something
    to bite on.  It is also cheaper: one NMF instead of two.

    Note the range starts at BASS_LO (24), below HIGH_FFT_MIN_PITCH, so this
    always takes the 8192-bin front end.  The 4096 path only ever existed to
    keep the isolated vocals stem affordable, and no longer runs for songs.
    """
    H = _activations(comp_dir / "harmonic.wav", MD_BASS_LO, MD_VOCAL_HI)
    if H is None or beat_times is None:
        return None
    out = MD.decode_two_voices(H, LOW, beat_times, downbeats, beats_per_bar)
    if not out:
        return None

    key = key_from_activations(H)
    if key is not None:
        root, scale_name = key
        for n in out:
            n["key_root"], n["key_scale"] = int(root), str(scale_name)
    n_bass = sum(1 for n in out if n.get("role") == "bass")
    print(f"[projection] two-voice: bass {n_bass}, melody {len(out) - n_bass}"
          + (f", key {key[0]} {key[1]}" if key else ""))
    return out


def project_from_stems(comp_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Project the DRUMLESS mix separation already wrote.

    harmonic.wav is Demucs' bass+other+vocals with percussion removed, at its
    original band balance.  Drum removal is worth 20 points here.

    It must NOT be reconstructed by summing bass.wav and upper.wav: those are
    peak-normalised independently, so their sum carries far more low end than
    the mix did — measured at 1.54x on a real upload.  Doing that made the
    decomposition emit 7.29 notes/s below G#3 against 0.77 above it, which is
    not what the eval measured, because the eval used the balanced signal.
    The fallback below is kept for sessions separated before harmonic.wav
    existed, and it says so rather than silently producing worse notes.
    """
    import librosa, json
    # The beat grid separation already found, for the melody decode's metrical
    # prior.  Read from components.json rather than recomputed, so the decode
    # is on the same grid every other stage uses.
    beats = downs = None
    bpb = 4
    meta = comp_dir.parent / "components.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            hop = float(m.get("hop_length", 512))
            msr = float(m.get("sr", 22050))
            bpb = int(m.get("beats_per_bar", 4))
            if m.get("beat_frames"):
                beats = np.asarray(m["beat_frames"], dtype=float) * hop / msr
            if m.get("downbeat_frames"):
                downs = np.asarray(m["downbeat_frames"], dtype=float) * hop / msr
        except Exception:                                   # noqa: BLE001
            pass

    # Per-stem routing when separation wrote the stems; otherwise the old
    # single-signal path, which is also the fallback for sessions separated
    # before stems existed.
    if TWO_VOICE and (comp_dir / "harmonic.wav").exists():
        two = project_two_voices(comp_dir, beats, downs, bpb)
        if two:
            return two
        print("[projection] two-voice path produced nothing; falling back")

    if PER_STEM and (comp_dir / "stem_vocals.wav").exists():
        by_stem = project_by_stem(comp_dir, beats, downs, bpb)
        if by_stem:
            return by_stem

    h = comp_dir / "harmonic.wav"
    if h.exists():
        y, sr_used = librosa.load(str(h), sr=SR, mono=True)
        return project_notes(y, sr_used, beats, downs, bpb)

    print("[projection] harmonic.wav missing (session predates it); falling "
          "back to summed stems, which over-weights the bass")
    parts, sr_used = [], SR
    for name in ("bass.wav", "upper.wav"):
        p = comp_dir / name
        if p.exists():
            y, sr_used = librosa.load(str(p), sr=SR, mono=True)
            parts.append(y)
    if not parts:
        raise FileNotFoundError("no drumless audio for projection")
    n = max(len(p) for p in parts)
    mix = np.zeros(n, dtype=np.float32)
    for p in parts:
        mix[: len(p)] += p
    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * 0.95
    return project_notes(mix, sr_used, beats, downs, bpb)
