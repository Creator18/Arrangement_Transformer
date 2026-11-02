"""Random-note clips for the arranger's "identify everything" curriculum (the user's design, 2026-11-02).

No melody, no musical sense: notes are drawn at random and every quality is drawn at random too, and the model must
say exactly what was played.  Difficulty rises by LEVEL; the trainer moves up a level once it is near-perfect.
At most 10 notes sound at once (ten fingers).  A note combination may come back, but only with freshly drawn
qualities (instrument, soundfont, tone, effects, loudness, mastering): the same notes, a different sound.

Sound comes from the base renders' own chain (eval/render_base.py): 14 instrument families from the soundfonts,
their bank variations, envelopes / filters / EQ / drive / compression / delay / tremolo, the loud master.  Nothing
moves pitch (no bends, vibrato or chorus).  Target = every note played, at its written time.

    make_clip(level, rng, seconds) -> y (float32, 24 kHz), notes (N, 5: onset, offset, pitch, velocity, instrument
                                      slot), tags (N, len(TAGS) booleans), meta (dict: every quality drawn)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import render_arrangements as RA          # noqa: E402
import render_base as RB                  # noqa: E402

SR = RA.SR
MAX_POLY = 10
TAGS = ("octave_pair", "semitone_neighbour", "quick_repeat", "short", "quiet_under_loud", "soft_attack", "low", "high", "tremolo",
        "inaudible")

# per level: notes at once, note lengths, gaps, what is allowed, how the sound is varied
LEVELS = {
    1: dict(poly=1, len=(0.20, 2.0), gap=(0.10, 0.6), octave=False, semitone=False, repeat=False, quiet=False,
            insts=(1, 1), tone=0, fx=False, master_p=0.0, drums_p=0.0, soft_p=0.0, families="plain"),
    2: dict(poly=2, len=(0.10, 2.0), gap=(0.0, 0.5), octave=True, semitone=False, repeat=False, quiet=False,
            insts=(1, 1), tone=1, fx=False, master_p=0.0, drums_p=0.0, soft_p=0.0, families="all", trem_p=0.25),
    3: dict(poly=3, len=(0.08, 2.0), gap=(0.0, 0.4), octave=True, semitone=True, repeat=True, quiet=False,
            insts=(1, 1), tone=2, fx=True, master_p=0.2, drums_p=0.0, soft_p=0.1, families="all", trem_p=0.25),
    4: dict(poly=5, len=(0.05, 2.5), gap=(0.0, 0.3), octave=True, semitone=True, repeat=True, quiet=False,
            insts=(1, 2), tone=2, fx=True, master_p=0.4, drums_p=0.0, soft_p=0.2, families="all", trem_p=0.25),
    5: dict(poly=7, len=(0.04, 3.0), gap=(0.0, 0.3), octave=True, semitone=True, repeat=True, quiet=True,
            insts=(1, 3), tone=3, fx=True, master_p=0.6, drums_p=0.2, soft_p=0.3, families="all", trem_p=0.25),
    6: dict(poly=10, len=(0.03, 3.0), gap=(0.0, 0.3), octave=True, semitone=True, repeat=True, quiet=True,
            insts=(1, 3), tone=3, fx=True, master_p=0.75, drums_p=0.4, soft_p=0.3, families="all", trem_p=0.25),
}
PLAIN = [0, 1, 4, 5, 24, 25, 48, 73, 19]          # level 1: piano, e-piano, guitar, strings, flute, organ
# playable ranges per instrument family (MIDI), so every labelled note is actually heard
RANGE = {"bass": (28, 60), "keys": (21, 108), "guitar": (40, 88), "strings": (36, 96), "pad": (36, 96),
         "voice": (45, 84), "organ": (36, 96), "winds": (55, 96), "brass": (40, 84), "lead": (48, 96),
         "mallet": (48, 100), "pluck": (40, 88)}


ALLOWED = None                  # programs whose every soundfont has a preset that sounds the key (eval/random_calib.py)


def init(sf2s, table_path, seed=0, presets=None):
    """Point the shared synthesis chain at the soundfonts (as render_base.main does).  presets: random_calib.py's
    json; presets that sound off the key are never used, and a program is used only if every soundfont has one."""
    global ALLOWED
    table = json.loads(pathlib.Path(table_path).read_text())
    RB._S["sf2s"] = list(sf2s)
    RB._S["table"] = {s: table[s] for s in RB._S["sf2s"]}
    RB._S["rng"] = np.random.default_rng(seed)
    RB.widen_families()
    if presets and pathlib.Path(presets).exists():
        bad = {tuple(b) for b in json.loads(pathlib.Path(presets).read_text())["bad"]}
        RB.BAD_VARIANT |= bad
        pool = sorted(RA.CONT_PROGS | RA.PLUCK_PROGS | set(RB.LEAD) | set(PLAIN))
        ALLOWED = {p for p in pool if all(
            [b for b, _n in RB._S["table"][sf]["progs"].get(str(p), []) if (pathlib.Path(sf).name, b, p) not in bad]
            for sf in RB._S["sf2s"])}


def family_of(prog):
    for fam, progs in list(RA.CONTINUOUS.items()) + list(RA.PLUCKED.items()):
        if prog in progs:
            return fam
    if prog in RA.BASS_CONT or prog in RA.BASS_PLUCK:
        return "bass"
    return "keys"


def pick_program(rng, families):
    pool = PLAIN if families == "plain" else sorted(RA.CONT_PROGS | RA.PLUCK_PROGS | set(RB.LEAD))
    if ALLOWED is not None:
        pool = [p for p in pool if p in ALLOWED] or [0]
    return int(rng.choice(pool))


def draw_notes(L, rng, seconds, insts):
    """Random notes: (onset, offset, pitch, velocity, slot).  Never more than L['poly'] sounding at once."""
    out = []
    t = float(rng.uniform(0.05, 0.4))
    while t < seconds - 0.1:
        k = int(rng.integers(1, L["poly"] + 1))
        slot = int(rng.integers(len(insts)))
        lo, hi = RANGE.get(family_of(insts[slot]), (36, 96))
        pitches = set()
        base = int(rng.integers(lo, hi + 1))
        pitches.add(base)
        while len(pitches) < k:
            r = rng.random()
            if L["octave"] and r < 0.2:
                p = base + int(rng.choice([-12, 12, 24, -24]))
            elif L["semitone"] and r < 0.4:
                p = base + int(rng.choice([-2, -1, 1, 2]))
            else:
                p = int(rng.integers(lo, hi + 1))
            if lo <= p <= hi:
                pitches.add(p)
        # sounding now: drop pitches that would push polyphony above the level's cap
        sounding = sum(1 for n in out if n[0] <= t < n[1])
        room = max(0, min(L["poly"], MAX_POLY) - sounding)
        pitches = sorted(pitches)[:room] if room else []
        for p in pitches:
            dur = float(np.exp(rng.uniform(np.log(L["len"][0]), np.log(L["len"][1]))))
            if any(n[2] == p and n[0] <= t < n[1] for n in out):
                continue                                  # the same key cannot strike while it is held
            v = int(rng.integers(35, 127))
            out.append([t + float(rng.uniform(0, 0.02)) * (len(pitches) > 1), min(t + dur, seconds - 0.02), p, v, slot])
        if L["quiet"] and len(pitches) >= 2 and rng.random() < 0.3:      # one quiet note under loud ones
            q = out[-1]; q[3] = int(rng.integers(20, 45))
            for n in out[-len(pitches):-1]:
                n[3] = max(n[3], 100)
        if L["repeat"] and pitches and rng.random() < 0.15:             # quick repeats of one key
            p, gap = pitches[0], float(rng.uniform(0.08, 0.2))
            t0 = t + gap
            for _ in range(int(rng.integers(1, 4))):
                if t0 > seconds - 0.1:
                    break
                out.append([t0, min(t0 + gap * 0.8, seconds - 0.02), p, int(rng.integers(50, 127)), slot])
                t0 += gap
        t += float(rng.uniform(*L["gap"])) + (min(n[1] for n in out[-max(1, len(pitches)):]) - t if L["poly"] == 1 and out else 0) \
            + float(rng.uniform(0.03, 0.15))
    return enforce_poly(np.array(out, float).reshape(-1, 5), min(L["poly"], MAX_POLY))


def enforce_poly(n, cap):
    """drop any note that would make more than cap notes sound at once (strict, after all the additions)."""
    n = n[np.argsort(n[:, 0], kind="stable")]
    keep = []
    for i, (s, e, p, v, k) in enumerate(n):
        if sum(1 for j in keep if n[j, 0] <= s < n[j, 1]) < cap and not any(n[j, 2] == p and n[j, 0] <= s < n[j, 1] for j in keep):
            keep.append(i)
    return n[keep]


def tag_notes(n):
    s, e, p, v = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    T = np.zeros((len(n), len(TAGS)), bool)
    for i in range(len(n)):
        ov = (s < e[i]) & (e > s[i]); ov[i] = False
        d = np.abs(p - p[i])
        T[i, 0] = np.any(ov & (d % 12 == 0) & (d > 0))
        T[i, 1] = np.any(ov & (d >= 1) & (d <= 2))
        T[i, 2] = np.any((p == p[i]) & (np.abs(s - s[i]) < 0.25) & (np.arange(len(n)) != i))
        T[i, 3] = e[i] - s[i] < 0.1
        T[i, 4] = v[i] < 50 and np.any((np.abs(s - s[i]) < 0.05) & (v >= 90))
        T[i, 6] = p[i] < 48
        T[i, 7] = p[i] >= 84
    return T


def make_clip(level, rng, seconds=8.0, notes=None):
    """One clip.  notes given = the same notes re-voiced with fresh qualities."""
    L = LEVELS[level]
    n_inst = int(rng.integers(L["insts"][0], L["insts"][1] + 1))
    insts = [pick_program(rng, L["families"]) for _ in range(n_inst)]
    if notes is None:
        notes = draw_notes(L, rng, seconds, insts)
    else:
        notes = notes.copy(); notes[:, 4] = rng.integers(n_inst, size=len(notes))
    RB._S["rng"] = np.random.default_rng(int(rng.integers(2 ** 31)))
    meta = dict(level=level, seconds=seconds, programs=insts, stems=[])
    y = np.zeros(int(seconds * SR))
    soft = np.zeros(len(notes), bool)
    trem_note = np.zeros(len(notes), bool)
    for k, prog in enumerate(insts):
        sel = notes[:, 4] == k
        if not sel.any():
            continue
        tone, tlab = RA.tone_for_stem(rng, L["tone"]) if L["tone"] else ({}, {})
        trem = None
        if L.get("trem_p") and rng.random() < L["trem_p"]:              # tremolo: the answer stays ONE held note
            trem = dict(kind=str(rng.choice(["restrike", "volume"], p=[0.6, 0.4])), hz=float(rng.uniform(8, 16)),
                        depth=float(rng.uniform(0.4, 0.9)))
        ev = [(0.0, "prog", prog, 0)]
        for i in np.nonzero(sel)[0]:
            s, e, p, v, _ = notes[i]
            per = 1.0 / trem["hz"] if trem else 0.0
            if trem and e - s >= 4 * per:
                trem_note[i] = True
            if trem and trem["kind"] == "restrike" and trem_note[i]:
                # the key re-struck fast (mandolin, tremolo strings, a picked tremolo), short lifts, velocity jitter
                lift = float(rng.uniform(0.1, 0.3)) * per
                for t in np.arange(s, e - per / 2, per):
                    vv = int(np.clip(v + rng.integers(-12, 13), 1, 127))
                    ev += [(float(t), "on", int(p), vv), (float(min(e, t + per - lift)), "off", int(p), 0)]
            else:
                ev += [(float(s), "on", int(p), int(v)), (float(e), "off", int(p), 0)]
        stem = RB.synth_stem(ev, None, seconds, tone=tone) if L["fx"] else _plain_stem(ev, seconds, tone)
        stem = stem[:len(y)]
        if trem and trem["kind"] == "volume":                          # the level wobbles (tremolo pedal, Leslie)
            t_ = np.arange(len(stem)) / SR
            stem = stem * (1 - trem["depth"] * 0.5 * (1 + np.sin(2 * np.pi * trem["hz"] * t_ + rng.uniform(0, 6.3))))
        if trem:
            tlab = dict(tlab, tremolo=trem)
        if L["soft_p"] and rng.random() < L["soft_p"]:                 # soft attacks (sound fades in, label stays)
            att = float(rng.choice([0.05, 0.1, 0.2, 0.35]))
            stem = RA.attack_shape(stem, notes[sel, 0], att, float(rng.uniform(0.0, 0.3)))
            soft[sel] = True; tlab = dict(tlab, attack_s=att)
        if tlab.get("lowpass") or tlab.get("highpass"):
            stem = RA.filt(stem, tlab.get("lowpass", 0), tlab.get("highpass", 0))
        g = 10 ** (float(rng.uniform(-9, 0)) / 20) if n_inst > 1 else 1.0
        y[:len(stem)] += g * stem
        meta["stems"].append(dict(program=prog, gain=g, tone=tlab))
    if L["drums_p"] and rng.random() < L["drums_p"]:                   # drums underneath, never in the target
        bpm = float(rng.uniform(70, 160)); beat = 60 / bpm
        ev = [(0.0, "kit", int(rng.choice(RA.DRUM_KITS)), 0)]
        for b in np.arange(0, seconds - 0.1, beat / 2):
            for drum, p_ in ((36, .5), (38, .3), (42, .8)):
                if rng.random() < p_:
                    ev += [(float(b), "on", drum, int(rng.integers(60, 120))), (float(b) + 0.05, "off", drum, 0)]
        d = RB.synth_stem(ev, None, seconds, drum=True)[:len(y)]
        y[:len(d)] += 10 ** (float(rng.uniform(-12, -3)) / 20) * d
        meta["drums_bpm"] = bpm
    if rng.random() < L["master_p"]:
        y, mlab = RB.master(y, rng, int(rng.integers(1, 4))); meta["master"] = mlab
    pk = np.abs(y).max()
    y = (y / pk * float(rng.uniform(0.3, 0.95))) if pk > 0 else y
    T = tag_notes(notes); T[:, TAGS.index("soft_attack")] = soft; T[:, TAGS.index("tremolo")] = trem_note
    T[:, TAGS.index("inaudible")] = ~audible(y, notes)
    return y.astype(np.float32), notes, T, meta


def audible(y, notes):
    """per note: does its pitch clearly sound in the final mix?  (its CQT bin rises > 3 dB over the 100 ms before
    its start, or is loud on its own, and is above -45 dB of the clip's peak.)  Inaudible notes are neither taught
    nor scored: hearing them is impossible, so they would only teach the model to imagine notes."""
    import librosa
    hop = 256
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y.astype(np.float64), sr=SR, hop_length=hop, fmin=librosa.midi_to_hz(21),
                                                   n_bins=88 * 3, bins_per_octave=36, tuning=0.0)), ref=np.max)
    fps = SR / hop
    out = np.ones(len(notes), bool)
    for i, (s, e, p, v, k) in enumerate(notes):
        b = int((p - 21) * 3) + 1
        a0 = int((s + 0.02) * fps); a1 = max(a0 + 1, int(min(e, s + 0.15) * fps))
        dur = C[max(0, b - 1):b + 2, a0:a1]
        pre = C[max(0, b - 1):b + 2, max(0, int((s - 0.12) * fps)):max(1, int((s - 0.02) * fps))]
        if dur.size == 0:
            continue
        top = dur.max()
        out[i] = top > -45 and (pre.size == 0 or top - pre.max() > 3 or top > -25)
    return out


def _plain_stem(ev, seconds, tone):
    """level 1-2: the soundfont sound without the per-stem effects chain."""
    rng = RB._S["rng"]
    fx, RB.stem_fx = RB.stem_fx, (lambda y, r: y)
    try:
        return RB.synth_stem(ev, None, seconds, tone=tone)
    finally:
        RB.stem_fx = fx


def rolls(notes, seconds):
    """onset / frame / velocity rolls (88, T) at the model's frame rate, written times."""
    from app.ml.spectral import frontend as S
    T = S.n_frames(int(seconds * S.SR))
    return RA.rolls([(s, e, p, v) for s, e, p, v, _ in notes], 0.0, T, S.FPS)


def max_poly(n):
    ts = np.unique(n[:, 0])
    return int(max(((n[:, 0] <= t) & (n[:, 1] > t)).sum() for t in ts)) if len(n) else 0


if __name__ == "__main__":
    import argparse
    import time
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf2s", required=True); ap.add_argument("--sf-table", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--per-level", type=int, default=3)
    a = ap.parse_args()
    init(a.sf2s.split(","), a.sf_table)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for lv in LEVELS:
        for i in range(a.per_level):
            t0 = time.time()
            y, n, T, meta = make_clip(lv, rng)
            sf.write(out / f"level{lv}_{i}.wav", y, SR)
            (out / f"level{lv}_{i}.json").write_text(json.dumps(dict(meta, notes=n.tolist(),
                                                                     tags={t: int(T[:, k].sum()) for k, t in enumerate(TAGS)}), default=float))
            print(f"level {lv} clip {i}: {len(n)} notes, max at once {max_poly(n)}, programs {meta['programs']}, "
                  f"{time.time() - t0:.1f} s to make", flush=True)

