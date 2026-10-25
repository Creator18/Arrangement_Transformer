"""Modular renders (the user's design, 2026-10-22): POP909 arrangements played by instruments, where the
ONLY thing that varies is one named effect -- so each adapter trained on a module explains one property.

HARD RULE: the written notes are exactly what is played and exactly the target.  Nothing is added (no octave
doublings, inner lines, pads, arpeggios, trills made from long notes) and nothing is removed or merged.  Only
HOW each written note sounds varies.

  plain        stock soundfont instruments in combinations (1-6 instruments, swaps, chord splits); default
               tone, no reverb / chorus send, flat EQ, equal levels, notes exactly as written, no drums, no voice
  timbre       envelope (attack/decay/sustain/release), low/high-pass, chorus
  reverb       room size / damping / width / level + per-instrument reverb send
  eq           per-instrument bass / treble shelves (+-9 dB at 200 Hz / 3 kHz) + whole-mix tilt
  dynamics     compression, saturation, noise floor, lo-fi
  levels       each part's loudness over time (lead over, faint accompaniment, bass-heavy, close, ...)
  exposed      the band drops out (-35..-60 dB) under the melody in some sections
  articulation played shorter or longer than written (staccato over long notes, holds over rests)
  connected    written fast figures (repeated / alternating notes) played CONNECTED (each note rings into
               the next: a tremolo that feels joint) or DETACHED (completely separate staccato strikes)
  noteend      sustain pedal on keys, very long releases, dead stops
  decay        notes fading out while still held
  expression   volume swells through phrases and dips mid-note (volume only)
  late         an instrument's sound arrives 0-120 ms late (slow-attack families) and swells in slowly
  fast         the whole arrangement sped up 1.2-2x (labels exact)
  glides       pitch glides between consecutive written notes, scoops into notes, falls off phrase ends
               (only written notes are attacked; target = the written notes)
  vibrato      pitch wobble (mod wheel) on strings / voice / winds / leads / brass / guitars
  rolls        chord notes swept 10-40 ms apart (strum / arpeggiando); target = the written chord
  unison       a note also played by a second instrument at the same time (same pitch, same time)
  drums        a normal drum kit on the beat grid (unpitched, never in the target)
  heavydrums   loud rock / EDM drums locked to the bass notes, 0-12 dB over the band
  sidechain    the band ducks on every beat (pumping), no drums
  genre        distorted guitars, saw leads, synth bass, synth brass
  singer       the melody sung (soundfont vowels + consonants) instead of played

    python eval/render_modular.py --module plain --pop909 <dir> --sf2 OmegaGMGS2.sf2 --out <dir> --shard i/n
writes <out>/<split>/pop<id>_<tag><k>_<w>.npz (y, on, fr, piano="multi", render json, note_meta) + .vel.npy
"""
from __future__ import annotations

import argparse, json, math, os, pathlib, sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import render_arrangements as RA                                     # noqa: E402  (shared, proven pieces)
from render_arrangements import (SR, SECONDS, CONTINUOUS, PLUCKED, CONT_PROGS, PLUCK_PROGS, BASS_CONT,  # noqa: E402
                                 BASS_PLUCK, DRUM_KITS, load_pop909, split_of_pop, rolls, downbeats, shelf,
                                 synth_stem, tone_for_stem, attack_shape, filt, level_plan, vocal_stem)

MODULES = ["plain", "timbre", "reverb", "eq", "dynamics", "levels", "exposed", "articulation", "connected",
           "noteend", "decay", "expression", "late", "fast", "glides", "vibrato", "rolls", "unison", "drums",
           "heavydrums", "sidechain", "genre", "singer"]
SLOW = set(CONTINUOUS["strings"]) | set(CONTINUOUS["winds"]) | set(CONTINUOUS["voice"]) | set(CONTINUOUS["pad"]) \
    | set(CONTINUOUS["brass"]) | set(CONTINUOUS["lead"])
VIB = SLOW - set(CONTINUOUS["pad"]) | set(PLUCKED["guitar"]) | {29, 30, 31}
GLIDE = SLOW - set(CONTINUOUS["pad"]) | {29, 30, 31} | set(PLUCKED["guitar"])
HOLD_KEYS = set(PLUCKED["keys"]) | set(CONTINUOUS["epiano"])
STRUM = set(PLUCKED["guitar"]) | set(PLUCKED["keys"]) | set(PLUCKED["pluck"]) | {46, 29, 30}


def orchestrate(notes, roles, beats, flags, rng, n_inst, swap, split, genre=False, prefer=None):
    """Which instrument plays each written note (exactly one), and each instrument's programs."""
    end = float(notes[:, 1].max()) + 2.0
    db = downbeats(beats, flags) if len(beats) > 8 else np.arange(0, end, 2.0)
    bar = float(np.median(np.diff(db))) if len(db) > 2 else 2.0
    step = 8 if swap == "section" else int(rng.integers(2, 5))
    bounds = [0.0] if swap == "never" else [0.0] + [float(x) for x in db[step::step]]
    seg = lambda t: int(np.searchsorted(bounds, t, side="right") - 1)
    role_of = [(0, 1, 2)[i] if i < 3 else int(rng.integers(0, 3)) for i in range(n_inst)]
    slots = {r: [i for i in range(n_inst) if role_of[i] == r] or [int(rng.integers(0, n_inst))] for r in (0, 1, 2)}
    pool = lambda r: (BASS_CONT + BASS_PLUCK + [42, 43, 32]) if r == 1 else sorted((CONT_PROGS | PLUCK_PROGS) - set(BASS_CONT) - set(BASS_PLUCK))
    prog = {(g, i): int(rng.choice(pool(role_of[i]))) for g in range(len(bounds)) for i in range(n_inst)}
    if prefer:                                                         # a module that needs capable instruments on a part
        for (g, i) in prog:
            if role_of[i] in prefer and rng.random() < .8:
                prog[(g, i)] = int(rng.choice(sorted(prefer[role_of[i]])))
    if genre:
        for (g, i) in prog:
            if role_of[i] == 1:
                prog[(g, i)] = int(rng.choice([38, 39, 33, 34]))
            elif rng.random() < .6:
                prog[(g, i)] = int(rng.choice([29, 30, 80, 81, 87, 62]))
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    slot = np.zeros(len(notes), int)
    for j in range(len(notes)):
        r = int(roles[j]); sl = slots[r]
        group = [x for x in np.flatnonzero(ev == ev[j]) if roles[x] == r]
        slot[j] = sl[group.index(j) % len(sl)] if (split and len(sl) > 1 and len(group) > 1) else sl[0]
    return dict(end=end, bar=bar, db=db, bounds=bounds, seg=seg, role_of=role_of, slots=slots, prog=prog, slot=slot)


def render(notes, roles, parts, beats, flags, sf2, rng, module, combo):
    """One orchestration with one module's effect, or several at once ("a+b+c", for the joint refinement:
    each effect is applied exactly as on its own).  Returns (audio, desc, per-note effect value)."""
    n_inst, swap, split = combo
    mods = set(module.split("+"))
    if "rolls" in mods:
        split = False                                                  # keep each chord on one instrument to sweep it
    prefer = {}
    for m_, pf in (("glides", {0: GLIDE}), ("vibrato", {0: VIB, 2: VIB}), ("rolls", {2: STRUM})):
        if m_ in mods:
            for r_, ps in pf.items():
                prefer[r_] = prefer.get(r_, set()) | ps
    O = orchestrate(notes, roles, beats, flags, rng, n_inst, swap, split, genre="genre" in mods, prefer=prefer or None)
    end, bar, seg, prog, slot = O["end"], O["bar"], O["seg"], O["prog"], O["slot"]
    blk = lambda t: int(t // (2 * bar))
    fx = np.zeros(len(notes), np.float32)                              # per-note effect value (diagnostics)
    sung = "singer" in mods
    # ---- note events: exactly the written notes (start, end, pitch, velocity), one instrument each
    on_t, off_t = notes[:, 0].copy(), notes[:, 1].copy()
    if "articulation" in mods:                                       # played shorter / longer than written
        mode = {}
        for j in range(len(notes)):
            k = (slot[j], blk(notes[j, 0]))
            mode.setdefault(k, str(rng.choice(["short", "hold", "as written"], p=[.4, .4, .2])))
            s, e = notes[j, 0], notes[j, 1]
            if mode[k] == "short":
                off_t[j] = s + max(.05, (e - s) * float(rng.uniform(.2, .6))); fx[j] = 2
            elif mode[k] == "hold":
                nxt = [notes[q, 0] for q in range(j + 1, min(len(notes), j + 40)) if slot[q] == slot[j] and notes[q, 2] == notes[j, 2]]
                cap = (nxt[0] - .005) if nxt else e + 1.2
                off_t[j] = max(e, min(cap, e + float(rng.uniform(.2, 1.2)))); fx[j] = 1
    if "connected" in mods:                                          # written fast figures: joint vs separate
        mode = {}
        for j in range(len(notes) - 1):
            same = [q for q in range(j + 1, min(len(notes), j + 12)) if slot[q] == slot[j]]
            if not same:
                continue
            q = same[0]; ioi = notes[q, 0] - notes[j, 0]
            if ioi <= .18:                                             # part of a fast figure
                k = (slot[j], blk(notes[j, 0])); mode.setdefault(k, "connected" if rng.random() < .5 else "detached")
                if mode[k] == "connected":                             # rings into the next note
                    off_t[j] = notes[q, 0] + float(rng.uniform(.02, .06)); fx[j] = 1
                else:                                                  # fully separate strike
                    off_t[j] = notes[j, 0] + max(.025, ioi * float(rng.uniform(.3, .6))); fx[j] = 2
    delay = {}
    if "late" in mods:                                               # the sound arrives late
        for j in range(len(notes)):
            i = int(slot[j]); k = (i, blk(notes[j, 0])); pr = prog[(seg(notes[j, 0]), i)]
            if k not in delay:
                delay[k] = (float(rng.uniform(-.03, 0)) if rng.random() < .1 else float(rng.uniform(0, .12))) \
                    if pr in SLOW else float(rng.uniform(-.02, .04))
            d = delay[k] + float(np.clip(rng.normal(0, .008), -.02, .02))
            on_t[j] += d; off_t[j] += d; fx[j] = 1000 * d
    stems = defaultdict(list)
    for j in range(len(notes)):
        if sung and roles[j] == 0:
            continue                                                   # the singer has the melody
        i = int(slot[j]); p_, v_ = int(notes[j, 2]), int(np.clip(notes[j, 3], 1, 127))
        stems[i] += [(on_t[j], "on", p_, v_), (off_t[j], "off", p_, 0)]
        if "unison" in mods and rng.random() < .35 and n_inst > 1:   # also on a second instrument, same time
            i2 = int(rng.choice([x for x in range(n_inst) if x != i]))
            stems[i2] += [(on_t[j], "on", p_, int(v_ * float(rng.uniform(.5, 1.0)))), (off_t[j], "off", p_, 0)]; fx[j] = 1
    head = {i: [] for i in range(n_inst)}
    for g, t0 in enumerate(O["bounds"]):
        for i in range(n_inst):
            head[i].append((t0, "prog", prog[(g, i)], 0))
    for i in range(n_inst):                                            # plain: no reverb / chorus send
        head[i] += [(0.0, "cc", 91, int(rng.integers(20, 110)) if "reverb" in mods else 0), (0.0, "cc", 93, 0)]
    # ---- per-instrument controller effects
    tones = {i: {} for i in range(n_inst)}
    for i in range(n_inst):                                            # combined: each effect adds its own settings
        if "timbre" in mods:
            t_, lab = tone_for_stem(rng, int(rng.integers(1, 4))); tones[i].update(t_); tones[i]["_lab"] = lab
        if "decay" in mods and rng.random() < .7:
            tones[i].update({"sustain_cb": float(rng.uniform(600, 960)), "decay_tc": float(rng.uniform(-1200, 1200))})
        if "noteend" in mods:
            c = rng.random()
            tones[i].update({"release_tc": float(rng.uniform(1800, 4200))} if c < .4 else
                            {"release_tc": float(rng.uniform(-4800, -2400))} if c < .8 else {})
        if "late" in mods and rng.random() < .6:
            tones[i]["_attack"] = float(rng.uniform(.03, .5))
    if "noteend" in mods:                                            # sustain pedal on keys / e-piano
        for i in range(n_inst):
            for b0 in np.arange(0.0, end, 2 * bar):
                if prog[(seg(b0), i)] in HOLD_KEYS and rng.random() < .5:
                    b1 = min(end, b0 + 2 * bar * float(rng.uniform(.5, 1.0)))
                    head[i] += [(b0, "cc", 64, 127), (b1, "cc", 64, 0)]
    if "expression" in mods:
        for i in range(n_inst):
            for b0 in np.arange(0.0, end, 2 * bar):
                if prog[(seg(b0), i)] in CONT_PROGS and rng.random() < .5:
                    v0, v1 = int(rng.integers(40, 128)), int(rng.integers(40, 128))
                    head[i] += [(b0 + q * 2 * bar, "cc", 11, int(v0 + (v1 - v0) * q)) for q in np.linspace(0, 1, 9)]
                    head[i].append((b0 + 2 * bar, "cc", 11, 127))
        for j in range(len(notes)):
            i = int(slot[j])
            if notes[j, 1] - notes[j, 0] >= .4 and prog[(seg(notes[j, 0]), i)] in CONT_PROGS and rng.random() < .3:
                m = notes[j, 0] + (notes[j, 1] - notes[j, 0]) * float(rng.uniform(.3, .7)); w = float(rng.uniform(.08, .2))
                head[i] += [(m - w / 2, "cc", 11, 127), (m, "cc", 11, int(rng.integers(30, 70))), (m + w / 2, "cc", 11, 127)]; fx[j] = 1
    if "vibrato" in mods:
        for i in range(n_inst):
            tones[i]["vibrato"] = True
            for b0 in np.arange(0.0, end, 2 * bar):
                c = 0
                if prog[(seg(b0), i)] in VIB:
                    u = rng.random(); c = 0 if u < .3 else 1 if u < .65 else 2
                head[i].append((b0, "cc", 1, 0 if c == 0 else int(rng.integers(20, 61)) if c == 1 else int(rng.integers(70, 128))))
                for j in np.flatnonzero((slot == i) & (notes[:, 0] >= b0) & (notes[:, 0] < b0 + 2 * bar)):
                    fx[j] = c
    if "glides" in mods:                                             # only on single-line instruments
        for i in range(n_inst):
            js = np.flatnonzero(slot == i)
            if len(js) < 2 or np.mean(np.diff(notes[js, 0]) < .03) > .2:   # chords on this instrument: no bends
                continue
            for a_, b_ in zip(js[:-1], js[1:]):
                pr = prog[(seg(notes[b_, 0]), i)]
                if pr not in GLIDE:
                    continue
                gap, iv = on_t[b_] - off_t[a_], notes[b_, 2] - notes[a_, 2]
                if fx[a_] != 1 and gap < .05 and 0 < abs(iv) <= 12 and rng.random() < .7:   # glide INTO the next note (no chains)
                    g = float(rng.uniform(.04, .25)); t0 = max(on_t[a_] + .02, on_t[b_] - g)
                    stems[i] = [e for e in stems[i] if not (e[1] == "off" and e[2] == int(notes[a_, 2]) and abs(e[0] - off_t[a_]) < 1e-6)]
                    stems[i] = [e for e in stems[i] if not (e[1] == "on" and e[2] == int(notes[b_, 2]) and abs(e[0] - on_t[b_]) < 1e-6)]
                    stems[i] += [(t0 + q * (on_t[b_] - t0), "bend", float(q * iv), 0) for q in np.linspace(0, 1, 10)]
                    stems[i] += [(off_t[b_], "off", int(notes[a_, 2]), 0), (off_t[b_] + .002, "bend", 0.0, 0)]
                    fx[b_] = 1                                           # reached by a glide (no new attack)
                elif rng.random() < .25:                                # a scoop into the note
                    sc = -float(rng.uniform(.5, 3)); d = float(rng.uniform(.05, .15))
                    stems[i] += [(on_t[b_] + q * d, "bend", sc * (1 - q), 0) for q in np.linspace(0, 1, 6)]; fx[b_] = 2
    if "rolls" in mods:                                              # chord notes swept apart (sound only)
        for i in range(n_inst):
            ons = sorted([e for e in stems[i] if e[1] == "on"], key=lambda e: e[0]); grp = []
            for e in ons + [(1e9, "on", 0, 0)]:
                if grp and e[0] - grp[0][0] > .015:
                    if len(grp) >= 2 and prog[(seg(grp[0][0]), i)] in STRUM and rng.random() < .6:
                        step = float(rng.uniform(.01, .04)); order = sorted(grp, key=lambda g_: g_[2], reverse=rng.random() < .25)
                        sh = {(g_[2], round(g_[0], 4)): k_ * step for k_, g_ in enumerate(order)}
                        stems[i] = [(t + sh.get((x, round(t, 4)), 0.0), kd, x, y_) if kd == "on" else (t, kd, x, y_) for t, kd, x, y_ in stems[i]]
                        for g_ in grp:
                            for j in np.flatnonzero((np.abs(notes[:, 0] - g_[0]) < 1e-6) & (notes[:, 2] == g_[2])):
                                fx[j] = 1
                    grp = []
                grp.append(e)
    # ---- render each instrument
    room = (float(rng.uniform(.3, 1.0)), float(rng.uniform(0, 1)), float(rng.uniform(.3, 1)), float(rng.uniform(.3, .95))) \
        if "reverb" in mods else (0.0, 0.0, 0.0, 0.0)
    gain_fn = None
    if mods & {"levels", "exposed"}:
        _sc, gain_fn, _d = level_plan(end, O["db"], rng, 1.0 if "exposed" in mods else 0.0)
    mix = np.zeros(int(end * SR) + SR)
    for i in range(n_inst):
        if not stems[i]:
            continue
        tone = {k: v for k, v in tones[i].items() if not k.startswith("_")}
        tone["reverb"] = room
        y = synth_stem(head[i] + stems[i], sf2, end, tone=tone)
        if "timbre" in mods:
            lab = tones[i]["_lab"]
            y = attack_shape(y, sorted({e[0] for e in stems[i] if e[1] == "on"}), lab["attack_s"], float(rng.uniform(.1, .5)))
            y = filt(y, lab["lowpass"], lab["highpass"])
        if "late" in mods and "_attack" in tones[i]:
            y = attack_shape(y, sorted({e[0] for e in stems[i] if e[1] == "on"}), tones[i]["_attack"], float(rng.uniform(.1, .4)))
        if "eq" in mods:
            y = shelf(shelf(y, 200.0, float(rng.uniform(-9, 9)), high=False), 3000.0, float(rng.uniform(-9, 9)), high=True)
        if gain_fn is not None:                                        # this instrument's part loudness over time
            pt = O["role_of"][i] if O["role_of"][i] != 2 else 2
            y = y * 10 ** (gain_fn(pt, np.arange(len(y)) / SR) / 20)
        n = min(len(y), len(mix)); mix[:n] += y[:n]
    if sung:                                                           # the melody sung instead of played
        y, lyric, _sl = vocal_stem(notes[roles == 0], sf2, end, rng, 0.0)
        n = min(len(y), len(mix)); mix[:n] += float(rng.uniform(.7, 1.0)) * y[:n] / (np.abs(y).max() + 1e-9) * (np.abs(mix).max() or 1.0)
    if mods & {"drums", "heavydrums"}:
        dk = "heavydrums" if "heavydrums" in mods else "drums"
        bd = float(np.median(np.diff(beats))) if len(beats) > 4 else .5
        dr = [(0.0, "kit", int(rng.choice(DRUM_KITS if dk == "drums" else [16, 24, 25, 0, 8])), 0)]
        kt = list(notes[roles == 1, 0]) if dk == "heavydrums" and rng.random() < .5 else list(beats if dk == "heavydrums" else beats[::2])
        for t in sorted(set(round(x, 3) for x in kt)):
            dr += [(t, "on", 36, int(rng.integers(90, 128))), (t + .1, "off", 36, 0)]
        for k_, t in enumerate(beats):
            if k_ % 2 == 1:
                dr += [(t, "on", 38, int(rng.integers(90, 128))), (t + .1, "off", 38, 0)]
            for h in range(2 if dk == "drums" else int(rng.choice([2, 4]))):
                dr += [(t + h * bd / 2, "on", 42, int(rng.integers(50, 100))), (t + h * bd / 2 + .05, "off", 42, 0)]
        y = synth_stem(dr, sf2, end, drum=True, tone={"reverb": room})
        g_db = float(rng.uniform(-12, -3)) if dk == "drums" else float(rng.uniform(0, 12))
        y = y / (np.abs(y).max() + 1e-9) * (np.abs(mix).max() or 1.0) * 10 ** (g_db / 20)
        n = min(len(y), len(mix)); mix[:n] += y[:n]
    if "sidechain" in mods:                                          # pumping on every beat, no drums
        tt = np.arange(len(mix)) / SR; pump = np.ones_like(tt); depth, rel = float(rng.uniform(.3, .75)), float(rng.uniform(.08, .25))
        for tk in beats:
            a0 = int(tk * SR); a1 = min(len(tt), a0 + int(4 * rel * SR))
            if a0 < len(tt):
                pump[a0:a1] = np.minimum(pump[a0:a1], 1 - depth * np.exp(-(tt[a0:a1] - tk) / rel))
        mix = mix * pump
    if "dynamics" in mods:
        mix, _ = RA.master(mix, rng, int(rng.integers(1, 4)))
    if "eq" in mods:
        mix = shelf(mix, 1000.0, float(rng.uniform(-4, 4)), high=True)
    pk = np.abs(mix).max()
    mix = mix / pk * 0.9 if pk > 0 else mix
    desc = dict(module=module, instruments=n_inst, swap=swap, chord_split=split,
                programs={f"inst{i}": sorted({prog[(g, i)] for g in range(len(O["bounds"]))}) for i in range(n_inst)})
    return mix.astype(np.float32), desc, fx


EXCLUSIVE = [{"drums", "heavydrums"}, {"levels", "exposed"}]


def pick_combo(rng, kmin, kmax):
    """2-5 effects (never plain), no two from the same exclusive pair, as "a+b+c"."""
    pool = [m for m in MODULES if m != "plain"]
    k = int(rng.integers(kmin, kmax + 1)); out = []
    for m in rng.permutation(pool):
        if len(out) == k:
            break
        if any(m in ex and set(out) & ex for ex in EXCLUSIVE):
            continue
        out.append(str(m))
    return "+".join(sorted(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help=f"one of {MODULES}, or \"combo\": 2-5 effects at once per render")
    ap.add_argument("--combo-k", default="2,5", help="combo: min,max effects per render")
    ap.add_argument("--pop909", required=True); ap.add_argument("--sf2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--renders", type=int, default=3, help="orchestrations per song")
    ap.add_argument("--splits", default="train,validation"); ap.add_argument("--tag", default="m")
    a = ap.parse_args()
    assert a.module == "combo" or a.module in MODULES, a.module
    from app.ml.spectral import frontend as S
    jobs = [d for d in sorted(pathlib.Path(a.pop909).iterdir()) if d.is_dir() and d.name.isdigit()
            and split_of_pop(int(d.name)) in a.splits.split(",")]
    si, sn = (int(v) for v in a.shard.split("/"))
    mine = jobs[si::sn][: a.limit or None]
    T = S.n_frames(int(SECONDS * S.SR)); done = 0
    for d in mine:
        split = split_of_pop(int(d.name)); out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash((a.module, d.name))) % 2 ** 32)
        notes, roles, beats, flags, parts = load_pop909(d, with_parts=True)
        if len(notes) < 10:
            continue
        if a.module == "fast":                                         # the whole arrangement faster (combos: below)
            sp = float(rng.choice([1.2, 1.4, 1.6, 2.0])); notes = notes.copy(); notes[:, :2] /= sp; beats = beats / sp
        n_win = max(1, int(math.ceil(notes[:, 1].max() / SECONDS)))
        if (out / f"pop{d.name}_{a.tag}{a.renders - 1}_{n_win - 1:03d}.npz").exists():
            continue
        base_notes, base_beats = notes, beats
        for k in range(a.renders):
            notes, beats, module = base_notes, base_beats, a.module
            if a.module == "combo":
                module = pick_combo(rng, *(int(v) for v in a.combo_k.split(",")))
                if "fast" in module.split("+"):
                    sp = float(rng.choice([1.2, 1.4, 1.6, 2.0])); notes = notes.copy(); notes[:, :2] /= sp; beats = beats / sp
            n_win = max(1, int(math.ceil(notes[:, 1].max() / SECONDS)))
            combo = (int(rng.integers(1, 7)), str(rng.choice(["never", "section", "bars"])), bool(rng.random() < .5))
            y, desc, fx = render(notes, roles, parts, beats, flags, a.sf2, rng, module, combo)
            for w in range(n_win):
                dst = out / f"pop{d.name}_{a.tag}{k}_{w:03d}.npz"
                t0 = w * SECONDS
                seg = y[int(t0 * SR):int((t0 + SECONDS) * SR)]; yy = np.pad(seg, (0, int(SECONDS * SR) - len(seg)))
                on, fr, vel = rolls(notes, t0, T, S.FPS)
                sel = (notes[:, 1] > t0) & (notes[:, 0] < t0 + SECONDS)
                nm = np.c_[notes[sel, 0] - t0, notes[sel, 2], fx[sel]].astype(np.float32)   # onset, pitch, effect value
                tmp = dst.with_name(f"{dst.stem}.tmp{os.getpid()}.npz")      # per process: a duplicate job can never collide
                np.savez_compressed(tmp, y=(np.clip(yy, -1, 1) * 32767).astype(np.int16), on=on, fr=fr, piano="multi",
                                    render=json.dumps(desc, default=float), note_meta=nm)
                tmp.replace(dst)
                vt = dst.with_name(f"{dst.stem}.vel.tmp{os.getpid()}.npy"); np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
                done += 1
        print(f"  {split} pop{d.name}: {a.renders} renders x {n_win} windows", flush=True)
    print(f"{done} clips written", flush=True)


if __name__ == "__main__":
    main()
