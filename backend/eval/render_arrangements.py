"""Piano arrangements -> many audio renders each, every one labelled with that ONE arrangement (CPU).

The user's design.  For every arrangement (POP909: MELODY/BRIDGE/PIANO tracks;
Music2MIDI human arrangements), rendered whole and cut into 20 s windows:

  >= 2 pure PIANO renders     each through a different sampled piano (SFZ, as the MAESTRO set)
  N  MULTI-INSTRUMENT renders  General MIDI soundfont through FluidSynth, each with random settings:

    COVERAGE       every combination of the settings below (252) is used in turn across the
                   dataset (a fixed shuffled order) — none is left to chance
    instruments    1 .. 6, chosen freely (any instrument for any role; bass from bass-range voices)
    swap rate      never / per 8-bar phrase / every 2-4 bars, aligned to downbeats
    articulation   only when the chosen instrument suits the passage: a continuous instrument
                   (strings, voice, pad, winds, organ) on connected notes lets them flow; a plucked
                   or staccato one (guitar, keys, mallets, pizzicato) on gapped notes keeps them
                   short; any other pairing plays the notes as written
    pitch          never altered: no pitch bend and no vibrato (modulation held at 0)
    layering       level 0 none .. 3 heavy, consistent or random (toggling bar to bar):
                     chord split   the notes of one chord on different instruments (any role)
                     unison        the same note on 2+ instruments
                     extra parts   NOT in the target: melody octave doubling, an in-chord line
                                   under the melody, a pad holding the chord, arpeggios / strums
    stems          every instrument renders as its own stem with its own volume and bass /
                   treble EQ (shelving +-9 dB at 200 Hz / 3 kHz), then they are mixed
    drums          on or off by the combination, from the beat grid (kick, backbeat, hats, fills)

  LEVELS (--levels)  each PART of the song (POP909's own labels: melody = the vocal line, bridge =
                   the secondary melody, the piano accompaniment, and its bass = the lowest accompaniment
                   note of each chord) gets its own loudness over time, so one part can sit far
                   above another: a loud lead over a faint accompaniment, a loud accompaniment
                   over a quiet lead, a faint inner part with a solid bass, ... (see level_plan).
                   Levels change from section to section with ramps, plus small per-note
                   variation.  Multi renders: every part of every instrument is its own stem
                   scaled by that part's curve.  Piano renders: the pianist's own balance, by
                   velocity.  The target is still every note, however faint.

The label of every render is only the arrangement's notes (onset / frame rolls,
velocity), so audio of any orchestration points at one answer.

    <out>/<split>/<src><id>_r<k>_<w>.npz   y (int16, 24 kHz), on, fr, piano ("salamander" ... or "multi"),
                                           render json; + .vel.npy
Splits: POP909 song % 10 (0 val, 1 test); Music2MIDI by song (eval/m2m_bysong.py units).

    python eval/render_arrangements.py --pop909 <dir> --m2m <dir> --units m2m_units.json \\
        --sf2 OmegaGMGS2.sf2 --out <dir> --shard i/n [--multi 4]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
SECONDS, SR = 20.0, 24000

# General MIDI programs by family
CONTINUOUS = {"strings": [48, 49, 50, 51, 40, 41, 42, 44], "voice": [52, 53, 54, 85], "pad": [88, 89, 90, 91, 92, 94, 95],
              "organ": [16, 17, 18, 19, 20], "winds": [73, 71, 68, 74, 75, 64, 65, 66, 69, 70],
              "brass": [56, 57, 60, 61, 62, 63], "lead": [80, 81, 82, 83, 84, 86, 87], "epiano": [4, 5, 21]}
PLUCKED = {"guitar": [24, 25, 26, 27, 28, 29, 30], "keys": [0, 1, 2, 6, 7, 3], "mallet": [8, 9, 11, 12, 13, 108],
           "pluck": [45, 46, 105, 107, 114, 10]}
BASS_CONT, BASS_PLUCK = [32, 35, 39, 43, 58], [33, 34, 36, 37, 38]
DRUM_KITS = [0, 8, 16, 24, 25, 32, 40, 48]
LEAD_P, LEAD_PROGS = 0.0, None                    # melody on a lead instrument (set by eval/render_base.py)


def split_of_pop(n):
    return "validation" if n % 10 == 0 else "test" if n % 10 == 1 else "train"


# ---------------------------------------------------------------- arrangements
def load_pop909(d: pathlib.Path, with_parts=False):
    """notes (N,4) [on, off, pitch, vel], role per note (0 melody, 1 bass, 2 accomp), beats (s).
    with_parts: also the PART of each note for levels (0 melody, 1 bass, 2 accompaniment, 3 bridge)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
    rows, roles, bridge = [], [], []
    for inst in pm.instruments:
        r = 0 if inst.name.upper() == "MELODY" else 2
        for n in inst.notes:
            rows.append([n.start, n.end, n.pitch, n.velocity]); roles.append(r); bridge.append(inst.name.upper() == "BRIDGE")
    notes, roles, bridge = np.array(rows, float), np.array(roles), np.array(bridge)
    o = np.argsort(notes[:, 0], kind="stable"); notes, roles, bridge = notes[o], roles[o], bridge[o]
    roles = mark_bass(notes, roles)
    parts = np.where(bridge & (roles == 2), 3, roles)
    rows = [l.split() for l in (d / "beat_midi.txt").read_text().splitlines() if l.strip()]
    beats = np.array([float(r[0]) for r in rows]); flags = np.array([float(r[2]) if len(r) > 2 else 0 for r in rows])
    if with_parts:
        return notes, roles, beats, flags, parts
    return notes, roles, beats, flags


def mark_bass(notes, roles):
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    for e in np.unique(ev):
        ix = np.flatnonzero((ev == e) & (roles == 2))
        if len(ix):
            j = ix[np.argmin(notes[ix, 2])]
            if notes[j, 2] < 60:
                roles[j] = 1
    return roles


def load_m2m(path: pathlib.Path):
    """Music2MIDI arrangement (flattened): melody = the top note of each onset group above C4."""
    notes = np.load(path).astype(float)
    notes = notes[np.argsort(notes[:, 0], kind="stable")]
    roles = np.full(len(notes), 2)
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    for e in np.unique(ev):
        ix = np.flatnonzero(ev == e); j = ix[np.argmax(notes[ix, 2])]
        if notes[j, 2] >= 60:
            roles[j] = 0
    roles = mark_bass(notes, roles)
    beats = estimate_beats(notes)
    return notes, roles, beats, None


def estimate_beats(notes):
    import librosa
    import scipy.signal
    if not hasattr(scipy.signal, "hann"):              # librosa's beat tracker still calls the removed name
        scipy.signal.hann = scipy.signal.windows.hann
    fps = 100; T = int(notes[:, 1].max() * fps) + fps
    env = np.zeros(T)
    np.add.at(env, np.clip((notes[:, 0] * fps).astype(int), 0, T - 1), notes[:, 3] / 127)
    env = np.convolve(env, np.hanning(7), "same")
    _, b = librosa.beat.beat_track(onset_envelope=env, sr=fps * 512, hop_length=512, units="time")
    return np.asarray(b)


# ---------------------------------------------------------------- labels
def rolls(notes, t0, T, fps):
    on = np.zeros((88, T), np.uint8); fr = np.zeros((88, T), np.uint8); vel = np.zeros((88, T), np.uint8)
    for s, e, p, v in notes:
        k = int(p) - 21
        if not 0 <= k < 88 or e <= t0 or s >= t0 + SECONDS:
            continue
        a, b = int(round((s - t0) * fps)), int(round((e - t0) * fps))
        fr[k, max(0, a):min(T, max(b, a + 1))] = 1
        if 0 <= a < T:
            on[k, a] = 1; vel[k, a] = max(vel[k, a], int(np.clip(v, 1, 127)))
    return on, fr, vel


# ---------------------------------------------------------------- multi-instrument renders
CONT_PROGS = {p for v in CONTINUOUS.values() for p in v} | set(BASS_CONT)
PLUCK_PROGS = {p for v in PLUCKED.values() for p in v} | set(BASS_PLUCK)


def combos():
    """Every combination of the render settings, in a fixed shuffled order.  Renders take them in
    turn across the whole dataset, so every combination is used evenly — nothing is left to chance."""
    out = []
    for n in range(1, 7):
        for swap in ("never", "section", "bars"):
            for layer in range(4):
                for style in (("consistent",) if layer == 0 else ("consistent", "random")):
                    for drums in (False, True):
                        out.append(dict(instruments=n, swap=swap, layering=layer, layering_style=style, drums=drums))
    random.Random(0).shuffle(out)
    return out


COMBOS = combos()


def articulation(notes, roles):
    """Per note: True = connected / legato (the next note of the same voice starts as this one ends)."""
    leg = np.zeros(len(notes), bool)
    for r in (0, 1, 2):
        ix = np.flatnonzero(roles == r)
        for x, y in zip(ix[:-1], ix[1:]):
            ioi = notes[y, 0] - notes[x, 0]
            if ioi > 0.03:
                leg[x] = (notes[y, 0] - notes[x, 1]) / ioi < 0.2
    return leg


def downbeats(beats, flags=None):
    if flags is not None and len(flags) == len(beats) and flags.sum() > 2:
        return beats[flags > 0]
    return beats[::4]


def shelf(y, f0, gain_db, high):
    """RBJ shelving EQ (S = 1): a bass (low shelf) or treble (high shelf) boost or cut."""
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / SR; c, sn = np.cos(w), np.sin(w); al = sn / 2 * np.sqrt(2)
    sq = 2 * np.sqrt(A) * al
    if high:
        bb = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        aa = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    else:
        bb = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        aa = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    return lfilter(np.array(bb) / aa[0], np.array(aa) / aa[0], y)


def _gen_api():
    import ctypes
    import fluidsynth
    L = fluidsynth._fl
    L.fluid_synth_set_gen.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    return L


def synth_stem(events, sf2, end, drum=False, tone=None):
    """One instrument on its own FluidSynth: events (t, kind, a, b) -> mono stem.
    tone (optional): release / decay offsets (timecents), sustain offset (cB), reverb room and
    chorus.  The vibrato control is held at 0.  "bend" events (value in semitones, range +-12)
    exist ONLY for slides between two written notes of a legato line: they land exactly on the
    next written pitch at its written time; no pitch outside the arrangement is ever held."""
    import fluidsynth
    fs = fluidsynth.Synth(samplerate=float(SR), gain=0.3); sfid = fs.sfload(str(sf2)); ch = 9 if drum else 0
    tone = tone or {}
    if "reverb" in tone:
        fs.set_reverb(*tone["reverb"])
    if "chorus" in tone:
        fs.set_chorus(*tone["chorus"])
    L = _gen_api()

    def apply_gens():
        if not tone.get("vibrato"):
            fs.cc(ch, 1, 0)
        for gen, key in ((38, "release_tc"), (36, "decay_tc"), (37, "sustain_cb")):
            if key in tone:
                L.fluid_synth_set_gen(fs.synth, ch, gen, float(tone[key]))
        if not drum:                                                     # pitch-bend range +-12 (RPN 0)
            for c, v in ((101, 0), (100, 0), (6, 12), (38, 0)):
                fs.cc(ch, c, v)
        if "chorus_send" in tone:
            fs.cc(ch, 93, int(tone["chorus_send"]))
    apply_gens()
    out, t_now = [], 0.0
    held, last_on = {}, {}                  # shared pitches on ONE instrument: a second voice starting within 50 ms
                                            # is the same strike; the key releases only when its LAST voice ends
    order = {"prog": 0, "kit": 0, "cc": 1, "off": 2, "bend": 3, "on": 4}
    for t, kind, x, y in sorted(events, key=lambda e: (e[0], order.get(e[1], 5))):
        n = int(round((t - t_now) * SR))
        if n > 0:
            out.append(np.asarray(fs.get_samples(n), np.float32)[::2]); t_now += n / SR
        if kind == "prog":
            fs.program_select(ch, sfid, 0, x); apply_gens()
        elif kind == "kit":
            fs.program_select(ch, sfid, 128, x)
        elif kind == "cc" and (x != 1 or tone.get("vibrato")):
            fs.cc(ch, x, y)
        elif kind == "bend":
            fs.pitch_bend(ch, int(np.clip(x / 12.0 * 8191, -8192, 8191)))
        elif kind == "on":
            if drum or not tone.get("shared_rule") or not (held.get(x, 0) > 0 and t - last_on.get(x, -9) <= 0.05):
                fs.noteon(ch, x, y); last_on[x] = t
            held[x] = held.get(x, 0) + 1
        elif kind == "off":
            held[x] = held.get(x, 0) - 1
            if drum or not tone.get("shared_rule") or held[x] <= 0:
                fs.noteoff(ch, x); held[x] = 0
    n = int(round((end - t_now) * SR))
    if n > 0:
        out.append(np.asarray(fs.get_samples(n), np.float32)[::2])
    fs.delete()
    return np.concatenate(out).astype(np.float64) / 32768.0 if out else np.zeros(1)


def attack_shape(y, onsets, attack_s, floor):
    """A slow attack AFTER rendering (sampled sustains ignore the attack generator): at each note
    start the stem fades in from `floor` over attack_s seconds.  Pitch untouched."""
    if attack_s <= 0.005 or not len(onsets):
        return y
    g = np.ones(len(y)); n = int(attack_s * SR); ramp = floor + (1 - floor) * np.linspace(0, 1, n) ** 1.5
    for t in onsets:
        a = int(t * SR)
        if 0 <= a < len(y):
            b = min(len(y), a + n); g[a:b] = np.minimum(g[a:b], ramp[:b - a])
    return y * g


def tone_for_stem(rng, level):
    """Random tone for one instrument; level 0 = defaults .. 3 = heavy.  Returns (tone, label)."""
    if level == 0:
        return {}, dict(attack_s=0.0, release_tc=0.0, decay_tc=0.0, sustain_cb=0.0, lowpass=0, highpass=0, chorus=0)
    k = level / 3.0
    t = dict(release_tc=float(rng.uniform(-1200, 4800 * k)), decay_tc=float(rng.uniform(-1200 * k, 2400 * k)),
             sustain_cb=float(rng.uniform(0, 300 * k)))
    if rng.random() < .4 * k + .1:
        t["chorus_send"] = int(rng.integers(20, 110))
    lab = dict(attack_s=float(rng.choice([0, 0, .02, .05, .1, .2, .35]) * (1 if rng.random() < .3 + .5 * k else 0)),
               release_tc=t["release_tc"], decay_tc=t["decay_tc"], sustain_cb=t["sustain_cb"],
               lowpass=int(rng.choice([0, 0, 2500, 4000, 7000]) if rng.random() < .5 * k else 0),
               highpass=int(rng.choice([0, 80, 150, 300]) if rng.random() < .4 * k else 0),
               chorus=int(t.get("chorus_send", 0)))
    return t, lab


def filt(y, lowpass, highpass):
    from scipy.signal import butter, lfilter
    if lowpass:
        b, a = butter(2, lowpass / (SR / 2)); y = lfilter(b, a, y)
    if highpass:
        b, a = butter(2, highpass / (SR / 2), btype="high"); y = lfilter(b, a, y)
    return y


def master(y, rng, level):
    """Production on the whole mix: compression, saturation, tilt EQ, noise, sometimes lo-fi.
    Returns (audio, label)."""
    from scipy.signal import butter, lfilter
    lab = dict(comp_ratio=1.0, comp_thresh_db=0.0, drive=1.0, tilt_db=0.0, noise_db=-120.0, lofi=0)
    if level == 0:
        return y, lab
    k = level / 3.0
    pk = np.abs(y).max() or 1.0; y = y / pk
    if rng.random() < .3 + .6 * k:                                       # RMS compressor
        ratio, thr = float(rng.uniform(2, 2 + 8 * k)), float(rng.uniform(-30, -10))
        env = np.sqrt(np.convolve(y ** 2, np.ones(int(.01 * SR)) / int(.01 * SR), "same")) + 1e-9
        lvl = 20 * np.log10(env); over = np.maximum(0, lvl - thr)
        gain_db = -over * (1 - 1 / ratio)
        gain_db = np.convolve(gain_db, np.ones(int(.03 * SR)) / int(.03 * SR), "same")
        y = y * 10 ** (gain_db / 20); lab.update(comp_ratio=ratio, comp_thresh_db=thr)
    if rng.random() < .2 + .5 * k:                                       # saturation
        d = float(rng.uniform(1, 1 + 4 * k)); y = np.tanh(d * y) / np.tanh(d); lab["drive"] = d
    if rng.random() < .5:                                                # tilt
        tl = float(rng.uniform(-4, 4)); y = shelf(y, 1000.0, tl, high=True); lab["tilt_db"] = tl
    if rng.random() < .25 * k + .05:                                     # noise floor (hiss / room)
        ndb = float(rng.uniform(-55, -32)); nz = rng.standard_normal(len(y))
        b, a = butter(1, float(rng.uniform(800, 8000)) / (SR / 2)); nz = lfilter(b, a, nz); nz /= np.abs(nz).max() + 1e-9
        y = y + nz * 10 ** (ndb / 20); lab["noise_db"] = ndb
    if rng.random() < .08 * k:                                           # phone / lo-fi recording
        b, a = butter(2, [300 / (SR / 2), 3400 / (SR / 2)], btype="band"); y = lfilter(b, a, y); lab["lofi"] = 1
    return y, lab


def slides_for(notes_idx, notes, prog_of, rng, slide_p):
    """Which consecutive notes of a legato line become a slide on a CONTINUOUS instrument:
    the next note is close (<= 2 semitones, sometimes <= 4) and starts as the previous ends.
    Returns {j: True} for every note j REACHED by a slide (labels unchanged: still its own note)."""
    out = {}
    for a, b in zip(notes_idx[:-1], notes_idx[1:]):
        if prog_of(a) not in CONT_PROGS or prog_of(b) != prog_of(a):
            continue
        gap, iv = notes[b, 0] - notes[a, 1], abs(notes[b, 2] - notes[a, 2])
        if gap < 0.04 and 0 < iv <= (4 if rng.random() < .2 else 2) and rng.random() < slide_p:
            out[b] = True
    return out



# ---------------------------------------------------------------- sung melody
VOWEL_PROGS = [52, 53, 54, 85]          # GM choir aahs, voice oohs, synth voice, voice lead
CONSONANTS = ["", "", "d", "t", "n", "m", "l", "b", "p"]   # "" = a bare vowel onset


def consonant(kind, f0, sr, rng):
    """A short consonant attack made here (the vowel itself comes from the soundfont).
    Plosives (d t b p): a filtered noise burst.  Nasals / liquids (n m l): a soft hum AT THE
    NOTE'S OWN PITCH, so no pitch is ever added.  Returns (audio, its length in seconds)."""
    from scipy.signal import butter, lfilter
    if kind in ("d", "t", "b", "p"):
        dur = {"t": .03, "p": .025, "d": .02, "b": .018}[kind] * float(rng.uniform(.8, 1.3))
        n = int(dur * sr); x = rng.standard_normal(n)
        lo, hi = {"t": (3000, 8000), "p": (500, 3000), "d": (1500, 5000), "b": (200, 1500)}[kind]
        b, a = butter(2, [lo / (sr / 2), min(hi, sr / 2 - 100) / (sr / 2)], btype="band")
        env = np.exp(-np.linspace(0, 6, n))
        return lfilter(b, a, x) * env * float(rng.uniform(.15, .35)), dur
    dur = {"n": .06, "m": .07, "l": .045}[kind] * float(rng.uniform(.8, 1.4))
    n = int(dur * sr); t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h ** (2.2 if kind != "l" else 1.6) for h in range(1, 6))
    b, a = butter(2, (900 if kind != "l" else 1800) / (sr / 2))
    env = np.minimum(1, t / .01) * np.minimum(1, (dur - t) / .015)
    return lfilter(b, a, x) * env * float(rng.uniform(.15, .3)), dur


def vocal_stem(mel, sf2, end, rng, slide_p=0.0):
    """The melody SUNG: random syllable chains per phrase.  Each syllable = a made consonant
    attack + a soundfont vowel at the exact written pitch (no vibrato).  Phrases split at rests;
    within a phrase a syllable sometimes carries over several notes (melisma) with no new
    consonant — and with slide_p, a close melisma step (<= 2 semitones, sometimes 4) is SUNG as a
    slide: the voice glides onto the next written pitch, landing exactly at its written time.
    Returns (audio, lyric, set of note indices reached by a slide)."""
    import fluidsynth
    fs = fluidsynth.Synth(samplerate=float(SR), gain=0.3); sfid = fs.sfload(str(sf2))
    progs = list(rng.choice(VOWEL_PROGS, size=int(rng.integers(2, 5)), replace=False))
    for c, pr in enumerate(progs):
        fs.program_select(c, sfid, 0, int(pr)); fs.cc(c, 1, 0); fs.cc(c, 91, int(rng.integers(10, 60)))
        for cc, v in ((101, 0), (100, 0), (6, 12), (38, 0)):            # pitch-bend range +-12
            fs.cc(c, cc, v)
    ev, cons, syl, slid = [], [], [], set()
    prev_end, ch, k = -9.0, 0, ""
    base = None                                                           # (index, pitch) of the held note in a slide chain
    for j, (s, e, p, v) in enumerate(mel):
        new_phrase = s - prev_end > 0.3
        melisma = (not new_phrase) and s - prev_end < 0.06 and rng.random() < .3
        lim = 4 if rng.random() < .2 else 2
        slide = (melisma and base is not None and 0 < abs(p - mel[j - 1, 2]) <= lim
                 and abs(p - base[1]) <= 12 and rng.random() < slide_p)
        if not melisma:                                                   # a new syllable
            ch = int(rng.integers(0, len(progs))); k = str(rng.choice(CONSONANTS))
            syl.append(k + ["a", "oo", "u", "o"][VOWEL_PROGS.index(progs[ch]) % 4])
            if k:
                cons.append((s, k, 440.0 * 2 ** ((p - 69) / 12)))
        nxt = mel[j + 1, 0] if j + 1 < len(mel) else None
        ee = max(e, float(nxt) - 0.01) if nxt is not None and nxt - e < 0.25 else e
        if slide:                                                         # hold the base note, glide onto p
            slid.add(j)
            ev = [x for x in ev if not (x[1] == "off" and x[2] == ch and x[3] == base[1] and x[0] >= mel[j - 1, 0])]
            g = float(rng.uniform(.04, .15)); t0 = max(mel[j - 1, 0], s - g)
            st, en = mel[j - 1, 2] - base[1], p - base[1]
            ev += [(t0 + q * (s - t0), "bend", ch, float(st + q * (en - st)), 0) for q in np.linspace(0, 1, 8)]
            ev += [(ee, "off", ch, int(base[1]), 0), (ee + .001, "bend", ch, 0.0, 0)]
        else:
            ev += [(s, "bend", ch, 0.0, 0), (s, "on", ch, int(p), int(v)), (ee, "off", ch, int(p), 0)]
            base = (j, int(p))
        prev_end = e
    out, t_now = [], 0.0
    order = {"off": 0, "bend": 1, "on": 2}
    for t, kind, c, x, y in sorted(ev, key=lambda z: (z[0], order[z[1]])):
        n = int(round((t - t_now) * SR))
        if n > 0:
            out.append(np.asarray(fs.get_samples(n), np.float32)[::2]); t_now += n / SR
        if kind == "on":
            fs.noteon(c, x, y)
        elif kind == "off":
            fs.noteoff(c, x)
        else:
            fs.pitch_bend(c, int(np.clip(x / 12.0 * 8191, -8192, 8191)))
    n = int(round((end - t_now) * SR))
    if n > 0:
        out.append(np.asarray(fs.get_samples(n), np.float32)[::2])
    fs.delete()
    y = np.concatenate(out).astype(np.float64) / 32768.0 if out else np.zeros(int(end * SR))
    peak = np.abs(y).max() or 1.0
    for t, kind, f0 in cons:                                              # consonant just before / at the vowel
        x, dur = consonant(kind, f0, SR, rng)
        a = max(0, int((t - (dur if kind in ("n", "m", "l") else 0.0)) * SR))
        b = min(len(y), a + len(x)); y[a:b] += x[:b - a] * peak
    return y, " ".join(syl[:24]), slid



def render_multi(notes, roles, beats, flags, sf2, combo, rng, vocal_p=0.0, tone_level=None, slide_p=0.0,
                 parts=None, art_both=False, voices=None, voice_k=0, exposed_p=0.0, sustain=False, attack_delay=False,
                 heavy=False, vibrato=False, rolls=False, orn=None, shared_rule=True, lead_p=0.0, lead_progs=None):
    """One orchestration of the whole arrangement.  Each instrument renders as its own stem with its
    own volume and bass/treble EQ, then the stems are mixed.  No pitch bend, no vibrato."""
    n_inst, layer, random_layer = combo["instruments"], combo["layering"], combo["layering_style"] == "random"
    end = float(notes[:, 1].max()) + 2.0
    db = downbeats(beats, flags) if len(beats) > 8 else np.arange(0, end, 2.0)
    bar = float(np.median(np.diff(db))) if len(db) > 2 else 2.0
    if combo["swap"] == "never":
        bounds = [0.0]
    else:
        step = 8 if combo["swap"] == "section" else int(rng.integers(2, 5))
        bounds = [0.0] + [float(x) for x in db[step::step]]
    seg_of = lambda t: int(np.searchsorted(bounds, t, side="right") - 1)
    roles_of_slot = [(0, 1, 2)[i] if i < 3 else int(rng.integers(0, 3)) for i in range(n_inst)]
    slots_of = {r: [i for i in range(n_inst) if roles_of_slot[i] == r] for r in (0, 1, 2)}
    for r in (0, 1, 2):                                                    # a role with no instrument borrows one
        if not slots_of[r]:
            slots_of[r] = [int(rng.integers(0, n_inst))]

    def pool(r):
        if r == 1:
            return BASS_CONT + BASS_PLUCK + [42, 43, 32]
        return sorted((CONT_PROGS | PLUCK_PROGS) - set(BASS_CONT) - set(BASS_PLUCK))
    prog = {(sg, i): int(rng.choice(pool(roles_of_slot[i]))) for sg in range(len(bounds)) for i in range(n_inst)}
    if lead_progs and rng.random() < lead_p:                              # a LEAD instrument plays the melody
        for (sg, i) in prog:                                              # (synth lead, guitar, violin, flute, sax ...)
            if i == slots_of[0][0]:
                prog[(sg, i)] = int(rng.choice(lead_progs))
    if heavy and rng.random() < .7:                                       # rock / EDM sounds
        for (sg, i) in prog:
            if roles_of_slot[i] == 1:
                prog[(sg, i)] = int(rng.choice([38, 39, 33, 34]))           # synth / electric bass
            elif rng.random() < .5:
                prog[(sg, i)] = int(rng.choice([29, 30, 80, 81, 87, 62]))   # overdrive / distortion, saw / square leads, synth brass
    head = {i: [] for i in range(n_inst)}                                # program changes / reverb send
    for sg, t0 in enumerate(bounds):
        for i in range(n_inst):
            head[i].append((t0, "prog", prog[(sg, i)], 0))
    for i in range(n_inst):
        head[i].append((0.0, "cc", 91, int(rng.integers(0, 70))))
    lev = parts is not None                                               # part-level mode
    if lev:
        end_ = float(notes[:, 1].max()) + 2.0
        db_ = downbeats(beats, flags) if len(beats) > 8 else np.arange(0, end_, 2.0)
        scen, gain, lev_desc = level_plan(end_, db_, rng, exposed_p)
        jit = rng.normal(0, 1.5, len(notes))                              # small per-note variation (dB)
    from collections import defaultdict
    stems = defaultdict(list)                                             # (instrument, part) -> note events
    PT = lambda j: int(parts[j]) if lev else 0
    on_layer = {}

    def active(t, kind, need):
        if layer < need:
            return False
        if not random_layer:
            return True
        k = (int(t // bar), kind)
        if k not in on_layer:
            on_layer[k] = rng.random() < .55
        return on_layer[k]
    leg = articulation(notes, roles)
    # BOTH WAYS (art_both): also the reverse of the natural articulation, chosen per instrument
    # per 2-bar phrase (never changing inside one): sustained instruments playing a connected
    # passage SHORT (staccato over long written notes), and sustained instruments or keys
    # (pedal) HOLDING notes over written gaps.  The target stays the written notes.
    art_mode, lenc = {}, np.zeros(len(notes), np.float32)            # 0 as written, 1 longer, 2 shorter
    HOLD_PROGS = set(CONT_PROGS) | set(PLUCKED["keys"]) | set(CONTINUOUS["epiano"])

    def art_of(i, t):
        k = (i, int(t // (2 * bar)))
        if k not in art_mode:
            art_mode[k] = str(rng.choice(["natural", "short", "hold"], p=[.5, .25, .25]))
        return art_mode[k]
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    sung = bool(rng.random() < vocal_p)                                   # the melody is SUNG in this render
    tl = int(rng.integers(0, 4)) if tone_level is None else int(tone_level)   # tone variation level 0..3
    slide_on = bool(slide_p > 0 and rng.random() < .6)
    note_slot = np.full(len(notes), -1)                                   # which instrument plays each note
    mel_idx = np.flatnonzero(roles == 0)
    slid = slides_for(list(mel_idx), notes, lambda j: prog[(seg_of(notes[j, 0]), slots_of[0][0])], rng,
                      slide_p) if (slide_on and not sung) else {}
    for j, (s, e, p, v) in enumerate(notes):
        r = int(roles[j]); slots = slots_of[r]; sg = seg_of(s)
        group = [x for x in np.flatnonzero(ev == ev[j]) if roles[x] == r]
        if len(slots) > 1 and len(group) > 1 and active(s, "split", 1):     # chord split, any role
            targets = [slots[group.index(j) % len(slots)]]
        else:
            targets = [slots[0]]
        if sung and r == 0:                                               # the singer has it; others only double it
            targets = targets[1:] if len(targets) > 1 else []
        bare = sung and r == 0 and lev and scen == "exposed"              # an exposed voice: nothing doubles it
        if targets and n_inst > 1 and not bare and active(s, "unison", 1) and rng.random() < .6:   # the same note on another instrument
            targets.append(int(rng.choice([i for i in range(n_inst) if i != targets[0]])))
        if sung and r == 0 and not targets and n_inst > 1 and not bare and active(s, "unison", 1) and rng.random() < .4:
            targets = [int(rng.choice([i for i in range(n_inst) if i not in slots_of[0]] or [0]))]
        for k, i in enumerate(dict.fromkeys(targets)):
            pr = prog[(sg, i)]; ee = e
            if leg[j] and pr in CONT_PROGS:                                  # continuous instrument + connected passage: flow
                nxt = notes[j + 1:, 0][roles[j + 1:] == r]
                if len(nxt) and nxt[0] - e < 0.25:
                    ee = max(e, float(nxt[0]) - 0.005)
            elif not leg[j] and pr in PLUCK_PROGS:                           # plucked instrument + gapped passage: short
                ee = s + max(0.05, (e - s) * float(rng.uniform(.4, .9)))
            if art_both:
                mode = art_of(i, s)
                if mode == "short" and pr in CONT_PROGS:                     # staccato even where the line is long
                    ee = s + max(0.05, (e - s) * float(rng.uniform(.2, .6)))
                elif mode == "hold" and pr in HOLD_PROGS:                    # ringing over the written gap
                    nxt = notes[j + 1:, 0][roles[j + 1:] == r]
                    gap_end = float(nxt[0]) - 0.005 if len(nxt) else e + 1.0
                    ee = max(e, min(gap_end, e + float(rng.uniform(.2, 1.2))))
            if k == 0:
                lenc[j] = 1.0 if ee > e + 0.03 else 2.0 if ee < e - 0.03 else 0.0
            vv = int(np.clip(v * (1.0 if k == 0 else rng.uniform(.3, .9)), 1, 127))
            if k == 0:
                note_slot[j] = i
            if lev:
                vv = int(np.clip(round(vv * 10 ** (jit[j] / 40)), 1, 127))
            if k == 0 and r == 0 and j in slid and i == slots_of[0][0]:
                pass                                                      # reached by a slide: see below
            else:
                stems[(i, PT(j))] += [(s, "on", int(p), vv), (ee, "off", int(p), 0)]
        if r == 0 and active(s, "octave", 2) and rng.random() < .7:          # extra parts, NOT in the target
            q = int(p) + (12 if p < 72 else -12); i = int(rng.choice(slots))
            stems[(i, PT(j))] += [(s, "on", q, int(v * .6)), (e, "off", q, 0)]
        if r == 0 and active(s, "under", 3) and rng.random() < .5:           # an in-chord tone under the melody
            snd = {int(x) % 12 for x in notes[(notes[:, 0] <= s) & (notes[:, 1] > s) & (roles == 2), 2]}
            cand = [int(p) - d for d in range(3, 10) if (int(p) - d) % 12 in snd]
            if cand:
                i = int(rng.choice(slots_of[2])); stems[(i, 2 if lev else 0)] += [(s, "on", cand[0], int(v * .5)), (e, "off", cand[0], 0)]
    if slid:                                                              # slides: hold the earlier note, bend onto the next
        i0 = slots_of[0][0]; ev0 = stems[(i0, 0)]
        chain_base, prev = None, None
        for j in mel_idx:
            if j in slid and prev is not None:
                base = chain_base if chain_base is not None else prev
                semis = notes[j, 2] - notes[base, 2]
                if abs(semis) <= 12:
                    chain_base = base
                    # extend the sounding base note to this note's end; glide lands on time
                    ev0[:] = [e for e in ev0 if not (e[1] == "off" and e[2] == int(notes[base, 2]) and abs(e[0] - notes[prev, 1]) < 0.3 and e[0] <= notes[j, 1])]
                    g = float(rng.uniform(.04, .15)); t0 = max(notes[prev, 0], notes[j, 0] - g)
                    start = notes[prev, 2] - notes[base, 2]
                    for q in np.linspace(0, 1, 8):
                        ev0.append((t0 + q * (notes[j, 0] - t0), "bend", float(start + q * (semis - start)), 0))
                    ev0.append((notes[j, 1], "off", int(notes[base, 2]), 0))
                    ev0.append((notes[j, 1] + .001, "bend", 0.0, 0))
                    prev = j
                    continue
            chain_base, prev = None, j
    extra = {}
    if layer >= 2:                                                           # a pad holding each bar's chord
        pad = [(0.0, "prog", int(rng.choice(CONTINUOUS["pad"] + CONTINUOUS["strings"])), 0)]
        for b0 in np.concatenate([[0.0], db]):
            if active(b0, "pad", 2):
                pcs = sorted({int(x) % 12 for x in notes[(notes[:, 0] < b0 + bar) & (notes[:, 1] > b0) & (roles == 2), 2]})[:4]
                pad += [x for q in pcs for x in ((b0, "on", 48 + q, 50), (b0 + bar * .98, "off", 48 + q, 0))]
        extra["pad"] = pad
    if layer >= 3:                                                           # arpeggiate / strum the chords
        arp = [(0.0, "prog", int(rng.choice(PLUCKED["guitar"] + PLUCKED["mallet"])), 0)]; div = int(rng.choice([8, 16]))
        for b0 in np.concatenate([[0.0], db]):
            if active(b0, "arp", 3):
                pcs = sorted({int(x) for x in notes[(notes[:, 0] < b0 + bar) & (notes[:, 1] > b0) & (roles == 2), 2]})
                for k, t in enumerate(np.arange(b0, b0 + bar, bar / div)):
                    if pcs:
                        q = pcs[k % len(pcs)]; arp += [(t, "on", q, int(rng.integers(45, 80))), (t + bar / div * .9, "off", q, 0)]
        extra["arp"] = arp
    drums = []; kicks = []
    if heavy and len(beats) > 4:
        # LOUD, LOCKED TO THE MUSIC: kick on the bass notes (or four on the floor), snare / clap on 2 and 4,
        # 8th / 16th hats, crashes on downbeats, fills; sometimes extra hits on non-bass notes.  Never in the target.
        drums = [(0.0, "kit", int(rng.choice([16, 24, 25, 0, 8])), 0)]; bd = float(np.median(np.diff(beats)))
        style = str(rng.choice(["bass", "four", "rock"])); sub = int(rng.choice([2, 4])); clap = rng.random() < .4
        kt = list(notes[roles == 1, 0]) if style == "bass" else list(beats) if style == "four" else \
            list(beats[::2]) + list(notes[roles == 1, 0])
        for t in sorted(set(round(x, 3) for x in kt)):
            drums += [(t, "on", 36, int(rng.integers(100, 128))), (t + .1, "off", 36, 0)]; kicks.append(t)
        for i, t in enumerate(beats):
            if i % 2 == 1:
                q = 39 if clap else 38; drums += [(t, "on", q, int(rng.integers(100, 128))), (t + .1, "off", q, 0)]
            for h in range(sub):
                q = 42 if rng.random() < .85 else 46
                drums += [(t + h * bd / sub, "on", q, int(rng.integers(60, 110))), (t + h * bd / sub + .05, "off", q, 0)]
            if i % 16 == 0:
                drums += [(t, "on", 49, 120), (t + 1.0, "off", 49, 0)]
            if i % 16 == 15 and rng.random() < .6:
                for h in range(4):
                    q = int(rng.choice([45, 47, 48, 50])); drums += [(t + h * bd / 4, "on", q, 115), (t + h * bd / 4 + .1, "off", q, 0)]
        for t in notes[roles != 1, 0]:                                     # extra hits on non-bass notes
            if rng.random() < .15:
                q = int(rng.choice([38, 45, 47, 50, 39])); drums += [(t, "on", q, int(rng.integers(90, 128))), (t + .1, "off", q, 0)]
    elif combo["drums"] and len(beats) > 4:
        drums = [(0.0, "kit", int(rng.choice(DRUM_KITS)), 0)]; sub = int(rng.choice([2, 4])); bd = float(np.median(np.diff(beats)))
        for i, t in enumerate(beats):
            if i % 2 == 0 or rng.random() < .1:
                drums += [(t, "on", 36, int(rng.integers(80, 120))), (t + .1, "off", 36, 0)]
            if i % 2 == 1:
                drums += [(t, "on", 38, int(rng.integers(80, 120))), (t + .1, "off", 38, 0)]
            for h in range(sub):
                q = 42 if rng.random() < .9 else 46
                drums += [(t + h * bd / sub, "on", q, int(rng.integers(40, 90))), (t + h * bd / sub + .05, "off", q, 0)]
            if i % 16 == 15 and rng.random() < .5:
                for h in range(4):
                    q = int(rng.choice([45, 47, 48, 50])); drums += [(t + h * bd / 4, "on", q, 100), (t + h * bd / 4 + .1, "off", q, 0)]
    # ATTACK DELAY (--attack-delay): a player's sound comes in after the written start -- slow-attack
    # instruments (strings, winds, voice, pads, brass) by 0..+120 ms, plucked / struck ones -20..+40 ms --
    # chosen per instrument per 2-bar phrase (a player is consistent), plus a small per-note jitter.  Only
    # the AUDIO moves; the target keeps the written start, so the model must place it on time.
    SLOW = set(CONTINUOUS["strings"]) | set(CONTINUOUS["winds"]) | set(CONTINUOUS["voice"]) | set(CONTINUOUS["pad"]) \
        | set(CONTINUOUS["brass"]) | set(CONTINUOUS["lead"])
    dly = {}

    def delay(i, t):
        if not attack_delay:
            return 0.0
        k_ = (i, int(t // (2 * bar)))
        if k_ not in dly:
            pr_ = prog[(seg_of(t), i)] if i >= 0 else 52
            if pr_ in SLOW:
                dly[k_] = float(rng.uniform(-.03, 0)) if rng.random() < .1 else float(rng.uniform(0, .12))
            else:
                dly[k_] = float(rng.uniform(-.02, .04))
        return dly[k_]
    rolled = set()
    if rolls:                                                              # strummed / rolled chords: SOUND only
        STRUM = set(PLUCKED["guitar"]) | set(PLUCKED["keys"]) | set(PLUCKED["pluck"]) | {46, 29, 30}
        for (i, pt), ev in stems.items():
            ons = sorted([e for e in ev if e[1] == "on"], key=lambda e: e[0]); grp = []
            for e in ons + [(1e9, "on", 0, 0)]:
                if grp and e[0] - grp[0][0] > .015:
                    if len(grp) >= 3 and prog[(seg_of(grp[0][0]), i)] in STRUM and rng.random() < .35:
                        step = float(rng.uniform(.01, .04)); order = sorted(grp, key=lambda g: g[2], reverse=rng.random() < .25)
                        sh = {(g[2], round(g[0], 4)): k_ * step for k_, g in enumerate(order)}
                        ev[:] = [(t + sh.get((x, round(t, 4)), 0.0), kd, x, y_) if kd == "on" else (t, kd, x, y_) for t, kd, x, y_ in ev]
                        rolled.update((i, round(g[0], 2)) for g in grp)
                    grp = []
                grp.append(e)
    if attack_delay:
        for (i, pt), ev in stems.items():
            ev[:] = [(t + delay(i, t) + (float(np.clip(rng.normal(0, .008), -.02, .02)) if kd == "on" else 0.0), kd, x, y_)
                     for t, kd, x, y_ in ev]
    VIB = set(CONTINUOUS["strings"]) | set(CONTINUOUS["voice"]) | set(CONTINUOUS["winds"]) | set(CONTINUOUS["lead"]) \
        | set(CONTINUOUS["brass"]) | set(PLUCKED["guitar"]) | {29, 30, 31}
    vib = {}                                                               # (instrument, block) -> 0 none / 1 slight / 2 wide
    if vibrato:
        for i in range(n_inst):
            for b0 in np.arange(0.0, end, 2 * bar):
                c_ = 0
                if prog[(seg_of(b0), i)] in VIB:
                    u_ = rng.random(); c_ = 0 if u_ < .4 else 1 if u_ < .75 else 2
                vib[(i, int(b0 // (2 * bar)))] = c_
                head[i].append((b0, "cc", 1, 0 if c_ == 0 else int(rng.integers(20, 61)) if c_ == 1 else int(rng.integers(70, 128))))
    mix = np.zeros(int(end * SR) + SR); eq = []
    lyric = ""
    if sung:
        mel_ix = np.flatnonzero(roles == 0)
        if voices:                                                        # a real singer (VocalSet clone, eval/sing_batch.py)
            vy, vm = voices[voice_k % len(voices)]
            y = np.zeros(int(end * SR) + SR); y[:min(len(vy), len(y))] = vy[:len(y)]
            lyric = f"{vm['tts_voice']} -> VocalSet speaker {vm['speaker']}"; vslid = vm["slid"]
            if len(vm["length_class"]) == len(mel_ix):                    # the singer's own articulation (both ways)
                lenc[mel_ix] = np.asarray(vm["length_class"], np.float32)
        else:
            mel_n = notes[roles == 0].copy()
            if attack_delay:                                              # the singer comes in late too
                dv = np.array([delay(-1, t) for t in mel_n[:, 0]]); mel_n[:, 0] += dv; mel_n[:, 1] += dv
            y, lyric, vslid = vocal_stem(mel_n, sf2, end, rng, slide_p if slide_on else 0.0)
        if lev:
            y = y * 10 ** (gain(0, np.arange(len(y)) / SR) / 20)
        slid = {int(mel_ix[q]): True for q in vslid}
        y = shelf(shelf(y, 200.0, float(rng.uniform(-6, 3)), high=False), 3000.0, float(rng.uniform(-3, 6)), high=True)
        g = float(rng.uniform(.7, 1.0)); n = min(len(y), len(mix)); mix[:n] += g * y[:n]
        eq.append(["voice", 0, 0, round(g, 2)])
    # SUSTAIN CASES (--sustain): how a note's sound behaves around its written length varies; the
    # label is always the written key-down / key-up.  Per instrument, per 2-bar block:
    #   pedal    keys / e-piano with the sustain pedal down: notes ring past their written release
    #   swells   sustained instruments get louder / softer through the block (expression, volume only)
    #   dips     a long sustained note dips in volume mid-note and recovers: still ONE note
    pedal_on = {i: [] for i in range(n_inst)}; dip_note = np.zeros(len(notes), np.float32)
    if sustain:
        HOLD_KEYS = set(PLUCKED["keys"]) | set(CONTINUOUS["epiano"])
        blocks = np.arange(0.0, end, 2 * bar)
        for i in range(n_inst):
            cc = []
            for b0 in blocks:
                pr_ = prog[(seg_of(b0), i)]
                if pr_ in HOLD_KEYS and rng.random() < .4:
                    b1 = min(end, b0 + 2 * bar * float(rng.uniform(.5, 1.0)))
                    pedal_on[i].append((float(b0), float(b1))); cc += [(b0, "cc", 64, 127), (b1, "cc", 64, 0)]
                elif pr_ in CONT_PROGS and rng.random() < .3:
                    v0, v1 = int(rng.integers(40, 128)), int(rng.integers(40, 128))
                    cc += [(b0 + q * 2 * bar, "cc", 11, int(v0 + (v1 - v0) * q)) for q in np.linspace(0, 1, 9)]
                    cc.append((b0 + 2 * bar, "cc", 11, 127))
            head[i] += cc
        for j, (s_, e_, p_, v_) in enumerate(notes):
            i = int(note_slot[j])
            if i >= 0 and e_ - s_ >= .4 and prog[(seg_of(s_), i)] in CONT_PROGS and rng.random() < .25:
                m_ = s_ + (e_ - s_) * float(rng.uniform(.3, .7)); w_ = float(rng.uniform(.08, .2))
                head[i] += [(m_ - w_ / 2, "cc", 11, 127), (m_, "cc", 11, int(rng.integers(30, 70))), (m_ + w_ / 2, "cc", 11, 127)]
                dip_note[j] = 1.0
    layers = [(f"inst{i}" + (f"p{pt}" if lev else ""), head[i] + ev) for (i, pt), ev in sorted(stems.items())] \
        + list(extra.items()) + ([("drums", drums)] if drums else [])
    inst_tone, inst_gain, inst_eq = {}, {}, {}
    perc_db = float(rng.uniform(0, 12)) if heavy else 0.0; perc_gain = 10 ** (perc_db / 20)
    pump = None
    if heavy and kicks and rng.random() < .5:
        tt = np.arange(int(end * SR) + SR) / SR; pump = np.ones_like(tt)
        depth, rel = float(rng.uniform(.3, .75)), float(rng.uniform(.08, .25))
        for tk in kicks:
            a0 = int(tk * SR); a1 = min(len(tt), a0 + int(4 * rel * SR))
            if a0 < len(tt):
                pump[a0:a1] = np.minimum(pump[a0:a1], 1 - depth * np.exp(-(tt[a0:a1] - tk) / rel))
    room = None
    if tl:
        room = (float(rng.uniform(.1, .2 + .8 * tl / 3)), float(rng.uniform(0, 1)), float(rng.uniform(.3, 1)),
                float(rng.uniform(.2, .5 + .5 * tl / 3)))
    if sustain and rng.random() < .3:                                     # very wet: long tails everywhere
        room = (float(rng.uniform(.8, 1.0)), float(rng.uniform(0, .4)), float(rng.uniform(.6, 1)), float(rng.uniform(.6, .95)))
    stem_tone = {}
    for key, events in layers:
        ik = key.split("p")[0] if key.startswith("inst") else key        # one instrument = one tone, all its parts
        if ik not in inst_tone:
            inst_tone[ik] = tone_for_stem(rng, tl if key != "drums" else min(tl, 1))
            if attack_delay and key.startswith("inst") and rng.random() < .6:   # slow swells, up to 0.5 s
                inst_tone[ik][1]["attack_s"] = float(rng.uniform(.03, .5))
            if sustain and key.startswith("inst"):                         # decay / ring / cut, per instrument
                t_, l_ = inst_tone[ik]; c_ = rng.random()
                if c_ < .3:     # fades while held
                    t_["sustain_cb"] = l_["sustain_cb"] = float(rng.uniform(600, 960)); t_["decay_tc"] = l_["decay_tc"] = float(rng.uniform(-1200, 1200))
                elif c_ < .6:   # rings long after release
                    t_["release_tc"] = l_["release_tc"] = float(rng.uniform(1800, 4200))
                elif c_ < .8:   # stops dead at release
                    t_["release_tc"] = l_["release_tc"] = float(rng.uniform(-4800, -2400))
            inst_eq[ik] = (float(rng.uniform(-9, 9)), float(rng.uniform(-9, 9)))
            inst_gain[ik] = float(rng.uniform(.4, 1.0)) * (.6 if key in ("pad", "arp") else 1.0)
        tone, tlab = dict(inst_tone[ik][0]), inst_tone[ik][1]
        if room:
            tone["reverb"] = room
        if vibrato:
            tone["vibrato"] = True
        if shared_rule:
            tone["shared_rule"] = True
        if tone.get("chorus_send"):
            tone["chorus"] = (int(rng.integers(2, 6)), float(rng.uniform(1, 4)), float(rng.uniform(.2, 2)), float(rng.uniform(2, 12)), 0)
        y = synth_stem(events, sf2, end, drum=key == "drums", tone=tone)
        ons = sorted({round(e[0], 3) for e in events if e[1] == "on"})
        y = attack_shape(y, ons, tlab["attack_s"], float(rng.uniform(.1, .5)))
        y = filt(y, tlab["lowpass"], tlab["highpass"]); stem_tone[ik] = tlab
        lo, hi = inst_eq[ik]
        y = shelf(shelf(y, 200.0, lo, high=False), 3000.0, hi, high=True)    # this layer's own bass / treble
        g = inst_gain[ik]
        if heavy and key == "drums":                                       # drums up to 12 dB over the band
            g *= perc_gain
        if heavy and key != "drums" and pump is not None:                  # the band ducks on every kick
            y = y * pump[:len(y)] if len(y) <= len(pump) else y * np.pad(pump, (0, len(y) - len(pump)), constant_values=1)
        if lev and key != "drums":                                        # this part's loudness over time
            pt = int(key.split("p")[1]) if key.startswith("inst") else 2  # pad / arp follow the accompaniment
            y = y * 10 ** (gain(pt, np.arange(len(y)) / SR) / 20)
        n = min(len(y), len(mix)); mix[:n] += g * y[:n]; eq.append([key, round(lo, 1), round(hi, 1), round(g, 2)])
    mix, mlab = master(mix, rng, tl)
    pk = np.abs(mix).max()
    # per-note labels: how each written note was SOUNDED (aux targets / diagnostics)
    meta = np.zeros((len(notes), 18), np.float32)                         # attack_s, release_tc, decay_tc, sustain_cb, slid, legato, sung, level_db, tone_valid, length class, ring class, faded, dip, delay_ms, vibrato class, trill/tremolo, rolled, drums dB
    for j in range(len(notes)):
        lab = stem_tone.get(f"inst{note_slot[j]}") if note_slot[j] >= 0 else None
        if lab:
            meta[j, :4] = [lab["attack_s"], lab["release_tc"], lab["decay_tc"], lab["sustain_cb"]]
        meta[j, 4] = 1.0 if j in slid else 0.0; meta[j, 5] = float(leg[j]); meta[j, 6] = 1.0 if (sung and roles[j] == 0) else 0.0
        meta[j, 8] = 1.0 if lab else 0.0                                 # tone columns known for this note
        meta[j, 9] = lenc[j]
        if sustain:                                                       # ring after release: 0 short, 1 some, 2 long
            rc = 0 if not lab else (0 if lab["release_tc"] <= 0 else 1 if lab["release_tc"] <= 1800 else 2)
            if note_slot[j] >= 0 and any(a_ <= notes[j, 1] < b_ for a_, b_ in pedal_on.get(int(note_slot[j]), [])):
                rc = 2                                                    # pedal down at the key release
            if room and room[0] > .7 and room[3] > .5:
                rc = max(rc, 1)
            meta[j, 10] = rc
            meta[j, 11] = 1.0 if (lab and lab["sustain_cb"] >= 600 and notes[j, 1] - notes[j, 0] > .7 * 2 ** (lab["decay_tc"] / 1200)) else 0.0
            meta[j, 12] = dip_note[j]
        if attack_delay:
            meta[j, 13] = 1000 * delay(-1 if (sung and roles[j] == 0) else int(note_slot[j]), notes[j, 0])
        if note_slot[j] >= 0:
            meta[j, 14] = vib.get((int(note_slot[j]), int(notes[j, 0] // (2 * bar))), 0)
            meta[j, 16] = 1.0 if (int(note_slot[j]), round(notes[j, 0], 2)) in rolled else 0.0
        meta[j, 15] = orn[j] if orn is not None else 0.0
        meta[j, 17] = perc_db
    if lev:                                                               # how loud each written note came out, vs the loudest sounding
        ldb = np.array([gain(PT(j), notes[j, 0]) + jit[j] + 20 * np.log10(
            inst_gain.get(f"inst{note_slot[j]}", .7 if (sung and roles[j] == 0) else 1.0) + 1e-6) for j in range(len(notes))])
        meta[:, 7] = relative_levels(notes, ldb)
    desc = dict(combo, tone_level=tl, slides=len(slid), levels=lev_desc if lev else None, reverb_room=room, master=mlab, sung=sung, lyric=lyric, programs={f"inst{i}": sorted({prog[(sg, i)] for sg in range(len(bounds))}) for i in range(n_inst)},
                layers_bass_treble_gain=eq)
    # everything that SOUNDS (pitched; drums out): every instrument's notes as played (incl. extra parts,
    # doublings, delays) + the sung melody -- the label for a full transcriber of the song
    allv = []
    for key, events in layers:
        if key == "drums":
            continue
        held = {}
        for t, kd, x, y_ in sorted(events, key=lambda e: e[0]):
            if kd == "on":
                held.setdefault(x, []).append(t)
            elif kd == "off" and held.get(x):
                allv.append((held[x].pop(0), t, x, 80))
    if sung:
        mn = notes[roles == 0]
        dv = np.array([delay(-1, t) for t in mn[:, 0]]) if attack_delay else np.zeros(len(mn))
        allv += [(s_ + d_, e_ + d_, p_, 80) for (s_, e_, p_, _v), d_ in zip(mn, dv)]
    desc["all_notes"] = np.array(sorted(allv), float) if allv else np.zeros((0, 4))
    if lev:                                                               # loudness of the whole mix varies too
        mix = mix * 10 ** (float(rng.uniform(-12, 0)) / 20) / max(pk, 1e-9) * 0.9
        return mix.astype(np.float32), desc, meta
    return (mix / pk * 0.9 if pk > 0 else mix).astype(np.float32), desc, meta


# ---------------------------------------------------------------- part levels
N_PARTS = 4                                     # melody, bass, accompaniment, bridge
SCENARIOS = ("lead_over", "faint_inner", "accomp_over", "random", "bass_heavy", "even", "close")
# "exposed" (chosen with --exposed-p): sections where the band drops out or nearly (-35..-60 dB) and
# the lead stands alone, alternating with ordinary lead-over sections.


def level_plan(end, db, rng, exposed_p=0.0):
    """Loudness of each part over time (dB, 0 = the loudest possible).  One scenario per render,
    re-drawn per section (4 or 8 bars) with ramps between sections.  Returns (scenario,
    gain(part, t) -> dB, section table)."""
    sc = "exposed" if rng.random() < exposed_p else SCENARIOS[int(rng.integers(len(SCENARIOS)))]

    def draw():
        u = lambda a, b: float(rng.uniform(a, b))
        if sc == "lead_over":       # the voice far above everything
            g = [0, u(-24, -6), u(-30, -8), u(-30, -6)]
        elif sc == "faint_inner":   # loud lead, solid bass, a faint accompaniment (the case heard missing)
            g = [0, u(-10, -2), u(-40, -18), u(-36, -14)]
        elif sc == "accomp_over":   # the band above a quiet voice
            g = [u(-24, -6), u(-12, 0), 0, u(-18, 0)]
        elif sc == "bass_heavy":
            g = [u(-12, 0), 0, u(-30, -8), u(-24, -6)]
        elif sc == "random":
            g = [u(-36, 0) for _ in range(N_PARTS)]
        elif sc == "exposed":       # the lead alone (band out or nearly), or an ordinary lead-over section
            g = [0, u(-60, -35), u(-60, -40), u(-60, -40)] if rng.random() < .6 else [0, u(-18, -4), u(-24, -8), u(-24, -8)]
        elif sc == "close":         # near each other but not equal: any order, spread 3-10 dB
            g = list(rng.permutation(np.linspace(0, -u(3, 10), N_PARTS)) + rng.normal(0, .7, N_PARTS))
        else:
            g = [u(-4, 0) for _ in range(N_PARTS)]
        g = np.array(g, float); return g - g.max()
    bars = np.concatenate([[0.0], db, [end]]) if len(db) else np.arange(0, end + 2, 2.0)
    step = int(rng.choice([4, 8])); t_sec = list(bars[::step]) + [end + 1]
    G = [draw()]
    for _ in t_sec[1:]:
        G.append(draw() if rng.random() < .5 else G[-1])
    G = np.array(G)
    ramp = [float(rng.uniform(0, 2.0)) for _ in t_sec]                    # seconds of crossfade into each section
    grid = np.arange(0, end + 2, 0.02)
    curves = np.zeros((N_PARTS, len(grid)))
    for k in range(len(t_sec) - 1):
        a, b = t_sec[k], t_sec[k + 1]; m = (grid >= a) & (grid < b)
        prev = G[k - 1] if k else G[0]
        w = np.clip((grid[m] - a) / max(ramp[k], 1e-3), 0, 1)
        curves[:, m] = prev[:, None] * (1 - w) + G[k][:, None] * w
    curves[:, grid >= t_sec[-1]] = G[-1][:, None]

    def gain(part, t):
        return np.interp(t, grid, curves[int(part)])
    return sc, gain, dict(scenario=sc, section_bars=step, sections=G.round(1).tolist())


def union_shared_old(notes, parts):
    """The FIRST shared-note convention (arr_union / arr_attack): same-pitch notes that overlap, or touch
    (<= 30 ms) from different parts, become ONE note from the first start to the latest end."""
    groups = []
    for p in np.unique(notes[:, 2]):
        ix = np.flatnonzero(notes[:, 2] == p); ix = ix[np.argsort(notes[ix, 0], kind="stable")]
        cur = [int(ix[0]), float(notes[ix[0], 0]), float(notes[ix[0], 1]), float(notes[ix[0], 3])]
        for j in ix[1:]:
            s_, e_, v_ = float(notes[j, 0]), float(notes[j, 1]), float(notes[j, 3])
            if s_ < cur[2] - 0.02 or (s_ - cur[2] <= 0.03 and parts[j] != parts[cur[0]]):
                cur[2] = max(cur[2], e_); cur[3] = max(cur[3], v_)
            else:
                groups.append(cur); cur = [int(j), s_, e_, v_]
        groups.append(cur)
    groups.sort(key=lambda g: (g[1], notes[g[0], 2]))
    merged = np.array([[g[1], g[2], notes[g[0], 2], g[3]] for g in groups], float)
    return merged, np.array([g[0] for g in groups], int)


def union_shared(notes, parts, attack_tol=0.05):
    """A pitch shared by two voices, played the way a pianist would (the user's rule):
      ATTACK   if the second voice starts within `attack_tol` of the first -> ONE strike; if it starts
               noticeably later -> its own re-strike at that moment
      RELEASE  the key never releases and re-strikes because the SHORTER note ended: once struck it stays
               held until the longest overlapping note ends
    Notes that do not overlap (repeats, back-to-back notes) are left exactly as written.
    Returns (notes sorted by onset, index of the original note each output note starts from)."""
    groups = []
    for p in np.unique(notes[:, 2]):
        ix = np.flatnonzero(notes[:, 2] == p); ix = ix[np.argsort(notes[ix, 0], kind="stable")]
        cur = [int(ix[0]), float(notes[ix[0], 0]), float(notes[ix[0], 1]), float(notes[ix[0], 3])]
        for j in ix[1:]:
            s_, e_, v_ = float(notes[j, 0]), float(notes[j, 1]), float(notes[j, 3])
            if s_ < cur[2] - 0.02:                                       # overlaps the note being held
                if s_ - cur[1] <= attack_tol:                             # same attack: one strike
                    cur[2] = max(cur[2], e_); cur[3] = max(cur[3], v_)
                else:                                                     # a distinct later entry: re-strike,
                    end_all = max(cur[2], e_)                             # then hold to the longest end
                    cur[2] = s_; groups.append(cur); cur = [int(j), s_, end_all, v_]
            else:
                groups.append(cur); cur = [int(j), s_, e_, v_]
        groups.append(cur)
    groups.sort(key=lambda g: (g[1], notes[g[0], 2]))
    merged = np.array([[g[1], g[2], notes[g[0], 2], g[3]] for g in groups], float)
    return merged, np.array([g[0] for g in groups], int)


def relative_levels(notes, level_db):
    """Each note's level relative to the loudest note sounding at its onset (dB, <= 0)."""
    out = np.zeros(len(notes), np.float32); on, off = notes[:, 0], notes[:, 1]
    for j in range(len(notes)):
        live = (on <= on[j] + 1e-3) & (off > on[j])
        out[j] = level_db[j] - level_db[live].max() if live.any() else 0.0
    return out


def piano_levels(pm, notes, parts, gain, rng):
    """A copy of the arrangement with each note's velocity set by its part's level (+ a little
    per-note variation): the pianist's own balance.  Returns (pm, level_db per note)."""
    import pretty_midi
    jit = rng.normal(0, 1.5, len(notes))
    lv = np.array([gain(parts[j], notes[j, 0]) for j in range(len(notes))]) + jit
    out = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    for (s_, e_, p_, v_), d in zip(notes, lv):
        v = int(np.clip(round(v_ * 10 ** (d / 40)), 1, 127))          # velocity ~ sqrt(amplitude)
        inst.notes.append(pretty_midi.Note(v, int(p_), float(s_), float(e_)))
    out.instruments.append(inst)
    return out, lv


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909", required=True); ap.add_argument("--m2m", required=True)
    ap.add_argument("--units", required=True); ap.add_argument("--sf2", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--pianos", type=int, default=2, help="pure piano renders per arrangement (>= 2)")
    ap.add_argument("--extra-piano-p", type=float, default=0.3, help="chance of a 3rd piano render")
    ap.add_argument("--multi", type=int, default=4, help="multi-instrument renders per arrangement")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tempo", default="", help="speed the written notes up before rendering: one factor per song from this list, e.g. 1.3,1.6,2.0")
    ap.add_argument("--splits", default="", help="only these splits, e.g. validation")
    ap.add_argument("--union-mode", choices=["new", "old"], default="new",
                    help="new = the user's revised rule (50 ms attack, never release + re-press); old = merge every overlap")
    ap.add_argument("--union-shared", action="store_true",
                    help="shared pitches: one strike if two voices start within 50 ms, else a re-strike; never a release + re-press when the shorter note ends (target and piano renders)")
    ap.add_argument("--heavy-p", type=float, default=0.0, help="share of multi renders with rock / EDM drums (loud, on the bass), genre sounds, sidechain")
    ap.add_argument("--vibrato", action="store_true", help="mod-wheel vibrato per instrument per phrase (target: one note)")
    ap.add_argument("--trills", action="store_true", help="some long melody notes become trills (alternating notes, in the target)")
    ap.add_argument("--rolls", action="store_true", help="rolled / strummed chords (sound only) and tremolo rolls (alternating notes, in the target)")
    ap.add_argument("--attack-delay", action="store_true",
                    help="sound comes in late / swells slowly (per instrument per phrase); target keeps the written start")
    ap.add_argument("--sustain", action="store_true",
                    help="sustain / decay / release cases (pedal, fades while held, long rings, dead stops, swells, dips, held vs re-struck) + labels; needs --levels --art-both")
    ap.add_argument("--vocal-p", type=float, default=0.0, help="share of multi renders whose melody is sung")
    ap.add_argument("--multi-only", action="store_true", help="no piano renders (a fine-tuning batch)")
    ap.add_argument("--tone", default="random", help="tone variation level 0-3, or random per render")
    ap.add_argument("--slide-p", type=float, default=0.0, help="chance an eligible legato step becomes a slide")
    ap.add_argument("--reverse", action="store_true", help="walk the shard from its end (a helper job meets the main one midway)")
    ap.add_argument("--tag", default="r", help="render-name prefix, so a new batch never overwrites the old")
    ap.add_argument("--levels", action="store_true", help="part-level variation (POP909 only: its parts are labelled)")
    ap.add_argument("--voices", default="", help="dir of cloned voices (eval/sing_batch.py + eval/rvc_sing.py)")
    ap.add_argument("--exposed-p", type=float, default=0.0, help="share of level renders with exposed-lead sections")
    ap.add_argument("--art-both", action="store_true",
                    help="articulation both ways: also staccato over long notes and holds over gaps (+ length label)")
    a = ap.parse_args()
    import pretty_midi
    import soundfile  # noqa: F401
    from app.ml.spectral import frontend as S, render as R
    piano_names = sorted(R.pianos())
    U = json.loads(pathlib.Path(a.units).read_text())
    m2m_split = {i: k for k, ids in U["splits"].items() for i in ids}
    jobs = [("pop", d.name, split_of_pop(int(d.name)), d) for d in sorted(pathlib.Path(a.pop909).iterdir())
            if d.is_dir() and d.name.isdigit()]
    if not a.levels:                                                      # Music2MIDI has no part labels
        jobs += [("m2m", i, m2m_split[i], pathlib.Path(a.m2m) / "midi_numpy" / f"{i}.npy")
                 for i in sorted(m2m_split) if (pathlib.Path(a.m2m) / "midi_numpy" / f"{i}.npy").exists()]
    if a.splits:
        jobs = [j for j in jobs if j[2] in a.splits.split(",")]
    si, sn = (int(v) for v in a.shard.split("/"))
    T = S.n_frames(int(SECONDS * S.SR))
    done = 0
    mine = list(enumerate(jobs))[si::sn][: a.limit or None]
    for jx, (src, sid, split, path) in (mine[::-1] if a.reverse else mine):
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash((src, sid))) % 2 ** 32)
        song_voices = []
        if a.voices and src == "pop":                                     # this song's cloned singers, at the render rate
            import librosa
            for vj in sorted(pathlib.Path(a.voices).glob(f"pop{sid}_v*.json")):
                vw = vj.with_name(vj.name[:-5] + ".voice.wav")
                if vw.exists():
                    vy, vsr = soundfile.read(str(vw), dtype="float32")
                    vy = vy.mean(1) if vy.ndim > 1 else vy
                    song_voices.append((librosa.resample(vy, orig_sr=vsr, target_sr=SR) if vsr != SR else vy,
                                        json.loads(vj.read_text())))
        parts = None
        if a.levels:
            notes, roles, beats, flags, parts = load_pop909(path, with_parts=True)
        else:
            notes, roles, beats, flags = load_pop909(path) if src == "pop" else load_m2m(path)
        if len(notes) < 10:
            continue
        if a.sustain:                                                     # held vs re-struck: some long notes become
            dur = notes[:, 1] - notes[:, 0]                                   # TWO real presses with a tiny gap
            cut = np.flatnonzero((dur >= .5) & (rng.random(len(notes)) < .06))
            if len(cut):
                m_ = notes[cut, 0] + dur[cut] * rng.uniform(.35, .65, len(cut)); g_ = rng.uniform(.02, .06, len(cut))
                second = notes[cut].copy(); second[:, 0] = m_ + g_
                notes = notes.copy(); notes[cut, 1] = m_
                notes = np.concatenate([notes, second]); roles = np.concatenate([roles, roles[cut]])
                if parts is not None:
                    parts = np.concatenate([parts, parts[cut]])
                o_ = np.argsort(notes[:, 0], kind="stable"); notes, roles = notes[o_], roles[o_]
                parts = parts[o_] if parts is not None else None
        if a.union_shared and a.union_mode == "new" and parts is not None:   # MORE SHARED NOTES: an accompaniment instrument
            add = []                                                       # also plays a melody pitch, entering together
            for j in np.flatnonzero((roles == 0) & (notes[:, 1] - notes[:, 0] >= .4)):    # (one strike) or clearly later
                if rng.random() < .08:                                     # (a re-strike), ending before or after it
                    s_, e_, p_, v_ = notes[j]; d_ = e_ - s_
                    st = s_ + (float(rng.uniform(0, .04)) if rng.random() < .5 else float(rng.uniform(.08, min(.4, d_ - .15))))
                    en = (st + float(rng.uniform(.12, max(.13, e_ - st - .05)))) if rng.random() < .5 else e_ + float(rng.uniform(.1, .6))
                    add.append([st, en, p_, v_ * float(rng.uniform(.5, 1.0))])
            if add:
                notes = np.concatenate([notes, np.array(add, float)])
                roles = np.concatenate([roles, np.full(len(add), 2)]); parts = np.concatenate([parts, np.full(len(add), 2)])
                o_ = np.argsort(notes[:, 0], kind="stable"); notes, roles, parts = notes[o_], roles[o_], parts[o_]
        orn = np.zeros(len(notes), np.float32)                             # 1 = part of a trill / tremolo roll
        if a.trills or a.rolls:
            new, drop = [], set()
            if a.trills:                                                      # a long melody note -> a TRILL: the note
                for j in np.flatnonzero((roles == 0) & (notes[:, 1] - notes[:, 0] >= .4)):   # and the one above it, alternating
                    if rng.random() < .05:
                        s_, e_, p_, v_ = notes[j]; up = p_ + int(rng.choice([1, 2])); rate = float(rng.uniform(7, 12))
                        ts = np.arange(s_, e_ - .5 / rate, 1 / rate)
                        new += [([t, t + .9 / rate, p_ if k % 2 == 0 else up, v_], j) for k, t in enumerate(ts)]; drop.add(int(j))
            if a.rolls:                                                       # two chord tones -> a TREMOLO roll
                acc = np.flatnonzero((roles == 2) & (notes[:, 1] - notes[:, 0] >= .5))
                for j in acc:
                    if int(j) in drop or rng.random() >= .03:
                        continue
                    mate = [m for m in acc if m != j and int(m) not in drop and abs(notes[m, 0] - notes[j, 0]) < .03
                            and 3 <= abs(notes[m, 2] - notes[j, 2]) <= 12]
                    if mate:
                        m = int(mate[0]); s_, e_ = notes[j, 0], min(notes[j, 1], notes[m, 1]); rate = float(rng.uniform(8, 14))
                        ts = np.arange(s_, e_ - .5 / rate, 1 / rate)
                        new += [([t, t + .9 / rate, notes[j, 2] if k % 2 == 0 else notes[m, 2], notes[j, 3]], j if k % 2 == 0 else m)
                                for k, t in enumerate(ts)]; drop.update((int(j), m))
            if new:
                keepm = np.array([i for i in range(len(notes)) if i not in drop], int)
                src_ix = np.array([j for _, j in new], int)
                notes = np.concatenate([notes[keepm], np.array([r for r, _ in new], float)])
                roles = np.concatenate([roles[keepm], roles[src_ix]])
                if parts is not None:
                    parts = np.concatenate([parts[keepm], parts[src_ix]])
                orn = np.concatenate([np.zeros(len(keepm), np.float32), np.ones(len(new), np.float32)])
                o_ = np.argsort(notes[:, 0], kind="stable"); notes, roles, orn = notes[o_], roles[o_], orn[o_]
                parts = parts[o_] if parts is not None else None
        if a.tempo:                                                       # faster music, labels still exact
            sp = float(rng.choice([float(x) for x in a.tempo.split(",")]))
            notes = notes.copy(); notes[:, :2] /= sp; beats = beats / sp
        notes_t, keep = notes, np.arange(len(notes))                      # the TARGET notes (and piano audio)
        if a.union_shared and parts is not None:
            notes_t, keep = union_shared(notes, parts) if a.union_mode == "new" else union_shared_old(notes, parts)
        parts_t = parts[keep] if parts is not None else None
        n_win = max(1, int(math.ceil(notes[:, 1].max() / SECONDS)))
        if (out / f"{src}{sid}_{a.tag}{(0 if a.multi_only else a.pianos) + a.multi - 1}_{n_win - 1:03d}.npz").exists():
            continue
        pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
        inst.notes = [pretty_midi.Note(int(np.clip(v, 1, 127)), int(p), float(s), float(e)) for s, e, p, v in notes_t]
        pm.instruments.append(inst)
        n_piano = 0 if a.multi_only else a.pianos + (1 if rng.random() < a.extra_piano_p else 0)
        pianos = list(rng.choice(piano_names, size=min(n_piano, len(piano_names)), replace=False))
        renders = [("piano", p) for p in pianos] + [("multi", k) for k in range(a.multi)]
        for k, (kind, arg) in enumerate(renders):
            pedal_k, roll_k = [], {}                                      # piano pedalling / rolls: set by the levels branch only
            if kind == "multi":
                y, desc, meta = render_multi(notes, roles, beats, flags, a.sf2, COMBOS[(jx * a.multi + arg) % len(COMBOS)], rng,
                                             vocal_p=a.vocal_p, tone_level=None if a.tone == "random" else int(a.tone),
                                             slide_p=a.slide_p, parts=parts, art_both=a.art_both,
                                             voices=song_voices, voice_k=arg, exposed_p=a.exposed_p, sustain=a.sustain,
                                             attack_delay=a.attack_delay, heavy=rng.random() < a.heavy_p,
                                             vibrato=a.vibrato, rolls=a.rolls, orn=orn,
                                             shared_rule=a.union_shared and a.union_mode == "new",
                                             lead_p=LEAD_P, lead_progs=LEAD_PROGS)
                all_notes = desc.pop("all_notes")
            elif a.levels:                                                # this pianist's balance between the parts
                end_ = float(notes[:, 1].max()) + 2.0
                _sc, gain, _d = level_plan(end_, downbeats(beats, flags), rng)
                pm_k, lv = piano_levels(pm, notes_t, parts_t, gain, rng)
                pedal_k = []
                roll_k = {}                                               # this pianist rolls some chords (sound only)
                if a.rolls:
                    on_ = notes_t[:, 0]
                    for s0 in np.unique(np.round(on_, 2)):
                        ch_ = np.flatnonzero(np.abs(on_ - s0) < .03)
                        if len(ch_) >= 3 and rng.random() < .3:
                            step = float(rng.uniform(.01, .04)); order = sorted(ch_, key=lambda q: notes_t[q, 2], reverse=rng.random() < .25)
                            roll_k.update({(int(notes_t[q, 2]), round(float(notes_t[q, 0]), 3)): k_ * step for k_, q in enumerate(order)})
                if a.sustain:                                             # this pianist's pedalling (sound only)
                    db_ = downbeats(beats, flags); bar_ = float(np.median(np.diff(db_))) if len(db_) > 2 else 2.0
                    pedal_k = [(float(b0), float(b0 + 2 * bar_ * rng.uniform(.5, 1.0))) for b0 in np.arange(0, end_, 2 * bar_)
                               if rng.random() < .4]
                rel = relative_levels(notes_t, lv)
            for w in range(n_win):
                dst = out / f"{src}{sid}_{a.tag}{k}_{w:03d}.npz"
                if dst.exists():
                    continue
                t0 = w * SECONDS
                if kind == "piano":
                    pmk = pm_k if a.levels else pm
                    yy, on, fr = R.render_window(pmk, t0, SECONDS, arg, pedal=pedal_k if a.sustain else None,
                                                 onset_shift=roll_k if a.rolls else None)
                    vel = R.velocity_roll(pmk, t0, SECONDS, arg); desc_k = {"piano": arg}
                    on_all, fr_all = on, fr                               # solo piano: everything that sounds = the notes
                    if a.levels:
                        desc_k["levels"] = _d
                else:
                    seg = y[int(t0 * SR):int((t0 + SECONDS) * SR)]
                    yy = np.pad(seg, (0, int(SECONDS * SR) - len(seg)))
                    on, fr, vel = rolls(notes_t, t0, T, S.FPS); desc_k = desc
                    on_all, fr_all, _ = rolls(all_notes, t0, T, S.FPS) if len(all_notes) else (on * 0, fr * 0, None)
                tmp = dst.with_name(dst.stem + ".tmp.npz")
                sel = (notes_t[:, 1] > t0) & (notes_t[:, 0] < t0 + SECONDS)
                W = 20 if (a.heavy_p > 0 or a.vibrato or a.trills or a.rolls) else 16 if a.attack_delay else 15 if a.sustain else 12 if a.art_both else 11
                nm = np.zeros((0, W if a.levels else 9), np.float32)
                if kind == "multi":                                       # onset (window s), pitch, then the sound labels
                    mt = meta[keep]                                       # each target note's sound = its first voice
                    m_ = mt[sel][:, :W - 2] if a.levels else mt[sel][:, :7]
                    nm = np.concatenate([np.c_[notes_t[sel, 0] - t0, notes_t[sel, 2]].astype(np.float32), m_], 1)
                elif a.levels:                                            # piano: only the level is known (tone_valid 0)
                    lat = R.latencies()[arg]
                    m_ = np.zeros((int(sel.sum()), W - 2), np.float32); m_[:, 7] = rel[sel]
                    if a.sustain:                                         # piano: pedal ring, long notes fade
                        m_[:, 10] = [2 if any(p0 <= e_ < p1 for p0, p1 in pedal_k) else 0 for e_ in notes_t[sel, 1]]
                        m_[:, 11] = (notes_t[sel, 1] - notes_t[sel, 0] > 2.5).astype(np.float32)
                    if W >= 20:
                        m_[:, 15] = orn[keep][sel]
                        m_[:, 16] = [1.0 if (int(p_), round(float(s_), 3)) in roll_k and roll_k[(int(p_), round(float(s_), 3))] > 0
                                     else 0.0 for s_, p_ in notes_t[sel][:, [0, 2]]]
                    nm = np.concatenate([np.c_[notes_t[sel, 0] - t0 + lat, notes_t[sel, 2]].astype(np.float32), m_], 1)
                np.savez_compressed(tmp, y=(np.clip(yy, -1, 1) * 32767).astype(np.int16), on=on, fr=fr,
                                    piano=str(arg) if kind == "piano" else "multi", render=json.dumps(desc_k, default=float),
                                    note_meta=nm, on_all=on_all, fr_all=fr_all)
                tmp.replace(dst)
                vt = dst.with_name(dst.stem + ".vel.tmp.npy"); np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
                done += 1
        print(f"  {split} {src}{sid}: {len(renders)} renders x {n_win} windows", flush=True)
    print(f"{done} clips written", flush=True)


if __name__ == "__main__":
    main()
