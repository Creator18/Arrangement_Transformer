"""Renders for the NEW BASE (NEW_BASE_PLAN.md): eval/render_arrangements.py's renderer -- POP909 + Music2MIDI
training arrangements, the arrangement + extra parts + drums, target = the arrangement at its written times --
widened in sound:

  soundfonts   every instrument stem picks one of several soundfonts (OmegaGMGS2, FluidR3 GM2, MuseScore_General)
               and one of that soundfont's bank variations of its program (detuned / chorus presets kept as
               sounds; sound effects, unpitched percussion and the few off-family variations excluded)
  families     14: acoustic piano & keys, electric pianos, organs, acoustic guitars, electric guitars, bass,
               bowed strings, brass, woodwinds & reeds, synth leads, synth pads, voice, tuned percussion,
               world plucked & bowed (GM program ranges; the renderer's articulation logic keeps working)
  lead lines   with --lead-p the melody is given to a lead instrument (synth lead, guitar, violin, flute, sax,
               trumpet ...), so fast instrumental melodies are common
  tone         the renderer's envelopes / filters / bass-treble EQ / reverb, plus per stem: drive, mid-band EQ,
               compression, delay, tremolo (loudness LFO).  NOTHING that moves pitch: chorus is off, no
               vibrato, no bends.
  mastering    on top of the renderer's bus processing: bass lift, bus compression and a peak limiter driven
               3-18 dB, so render loudness covers where real masters sit (measured: real songs ~0.8 spreads
               above the old renders and 1.7x wider, eval/fe_shift.py)

Everything else (splits, windows, piano renders, extra parts, drums, sustain cases, attack delays, singer)
is render_arrangements.main, unchanged; its own options pass through.

    python eval/render_base.py --sf2s a.sf2,b.sf2,c.sf2 --sf-table sf_table.json --lead-p 0.4 \\
        [render_arrangements options: --pop909 --m2m --units --sf2 --out --shard --multi ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
import render_arrangements as RA                                     # noqa: E402

# GM programs never used: orchestra hit, unpitched percussion, sound effects
EXCLUDE = {55, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127}
# bank variations whose sound is not their program's family (checked by name)
BAD_VARIANT = {("OmegaGMGS2.sf2", 1, 9), ("OmegaGMGS2.sf2", 1, 13), ("OmegaGMGS2.sf2", 1, 14), ("OmegaGMGS2.sf2", 1, 15)}
# melody instruments for --lead-p: synth leads, electric guitars, violin / fiddle, flutes, reeds, saxes, brass
LEAD = [80, 81, 82, 83, 84, 87, 26, 27, 29, 30, 40, 110, 73, 72, 75, 71, 68, 64, 65, 66, 56, 59, 61]


def widen_families():
    """The renderer's instrument pools, widened to the 14 families (articulation sets follow)."""
    C, P = RA.CONTINUOUS, RA.PLUCKED
    C["strings"] = sorted(set(C["strings"]) | {110})                       # + fiddle
    C["pad"] = sorted(set(C["pad"]) | {93, 96, 97, 99, 101, 102})          # + metal pad, ice rain, soundtrack, atmosphere ...
    C["organ"] = sorted(set(C["organ"]) | {21, 22, 23})                    # + accordion, harmonica, bandoneon
    C["winds"] = sorted(set(C["winds"]) | {72, 76, 77, 78, 79, 109, 111})  # + piccolo, bottle, shakuhachi, whistle, ocarina, bagpipe, shanai
    P["guitar"] = sorted(set(P["guitar"]) | {31})                          # + guitar harmonics
    P["mallet"] = sorted(set(P["mallet"]) | {10, 14, 15, 98, 100, 112, 114})   # + music box, bells, dulcimer, crystal, tinkle, steel drums
    P["pluck"] = sorted(set(P["pluck"]) | {104, 105, 106, 107})            # + sitar, banjo, shamisen, koto
    RA.CONT_PROGS = {p for v in C.values() for p in v} | set(RA.BASS_CONT)
    RA.PLUCK_PROGS = {p for v in P.values() for p in v} | set(RA.BASS_PLUCK)
    assert not (RA.CONT_PROGS | RA.PLUCK_PROGS) & EXCLUDE


_S = {"rng": np.random.default_rng(0), "table": {}, "sf2s": []}


def variants(sf2, prog):
    name = pathlib.Path(sf2).name
    return [b for b, _n in _S["table"][sf2]["progs"].get(str(prog), []) if (name, b, prog) not in BAD_VARIANT] or [0]


def synth_stem(events, sf2, end, drum=False, tone=None):
    """RA.synth_stem with a random soundfont per stem and a bank variation per program."""
    import fluidsynth
    rng = _S["rng"]
    sf2 = str(rng.choice(_S["sf2s"]))
    fs = fluidsynth.Synth(samplerate=float(RA.SR), gain=0.3); sfid = fs.sfload(sf2); ch = 9 if drum else 0
    tone = dict(tone or {}); tone.pop("chorus_send", None); tone.pop("chorus", None)   # chorus moves pitch: off
    if "reverb" in tone:
        fs.set_reverb(*tone["reverb"])
    L = RA._gen_api()
    bank_of = {}

    def apply_gens():
        fs.cc(ch, 1, 0); fs.cc(ch, 93, 0)                                   # no vibrato, no chorus send
        for gen, key in ((38, "release_tc"), (36, "decay_tc"), (37, "sustain_cb")):
            if key in tone:
                L.fluid_synth_set_gen(fs.synth, ch, gen, float(tone[key]))
    apply_gens()
    kits = [p for _b, p in _S["table"][sf2]["kits"]] or [0]
    out, t_now = [], 0.0
    order = {"prog": 0, "kit": 0, "cc": 1, "off": 2, "bend": 3, "on": 4}
    for t, kind, x, y in sorted(events, key=lambda e: (e[0], order.get(e[1], 5))):
        n = int(round((t - t_now) * RA.SR))
        if n > 0:
            out.append(np.asarray(fs.get_samples(n), np.float32)[::2]); t_now += n / RA.SR
        if kind == "prog":
            if x not in bank_of:
                bank_of[x] = int(rng.choice(variants(sf2, x)))
            fs.program_select(ch, sfid, bank_of[x], x); apply_gens()
        elif kind == "kit":
            fs.program_select(ch, sfid, 128, x if x in kits else int(rng.choice(kits)))
        elif kind == "cc" and x not in (1, 93):
            fs.cc(ch, x, y)
        elif kind == "on":
            fs.noteon(ch, x, y)
        elif kind == "off":
            fs.noteoff(ch, x)
        # "bend": never rendered (no pitch movement)
    n = int(round((end - t_now) * RA.SR))
    if n > 0:
        out.append(np.asarray(fs.get_samples(n), np.float32)[::2])
    fs.delete()
    y = np.concatenate(out).astype(np.float64) / 32768.0 if out else np.zeros(1)
    return y if drum else stem_fx(y, rng)


def peak_eq(y, f0, gain_db, q=1.0):
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / RA.SR; al = np.sin(w) / (2 * q); c = np.cos(w)
    b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], y)


def compress(y, thr_db, ratio, att=0.005, rel=0.08):
    env = np.sqrt(np.convolve(y ** 2, np.ones(int(att * RA.SR) or 1) / (int(att * RA.SR) or 1), "same")) + 1e-9
    over = np.maximum(0, 20 * np.log10(env) - thr_db)
    g = -over * (1 - 1 / ratio)
    k = int(rel * RA.SR) or 1
    return y * 10 ** (np.convolve(g, np.ones(k) / k, "same") / 20)


def stem_fx(y, rng):
    """Per-stem tone beyond the renderer's own (no pitch effects)."""
    if not np.any(y):
        return y
    pk = np.abs(y).max()
    if rng.random() < .25:                                                 # drive / distortion
        d = float(rng.uniform(1.5, 8)); y = np.tanh(d * y / pk) / np.tanh(d) * pk
    if rng.random() < .35:                                                 # a mid band boosted or cut
        y = peak_eq(y, float(rng.uniform(300, 3500)), float(rng.uniform(-9, 9)), float(rng.uniform(.5, 2)))
    if rng.random() < .3:                                                  # compression
        y = compress(y, float(20 * np.log10(pk + 1e-9) - rng.uniform(6, 20)), float(rng.uniform(2, 8)))
    if rng.random() < .2:                                                  # delay (echoes of the same pitches)
        dt = int(rng.uniform(.1, .5) * RA.SR); fb = float(rng.uniform(.2, .5)); mix = float(rng.uniform(.1, .35))
        e = np.zeros_like(y); src = y.copy()
        for k in range(1, 6):
            if k * dt >= len(y):
                break
            e[k * dt:] += src[:len(y) - k * dt] * fb ** (k - 1)
        y = y + mix * e
    if rng.random() < .1:                                                  # tremolo (loudness LFO, pitch untouched)
        t = np.arange(len(y)) / RA.SR; depth = float(rng.uniform(.2, .6))
        y = y * (1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * float(rng.uniform(3, 8)) * t))
    return y


_master0 = RA.master


def master(y, rng, level):
    """The renderer's bus processing, then a loud master on most renders."""
    from scipy.ndimage import maximum_filter1d
    y, lab = _master0(y, rng, level)
    lab.update(bass_db=0.0, bus_ratio=1.0, limit_drive_db=0.0)
    if rng.random() < _S.get("master_p", .75):
        pk = np.abs(y).max() or 1.0; y = y / pk
        if rng.random() < .7:                                              # real masters carry more bass (eval/fe_shift.py)
            b = float(rng.uniform(2, 10)); y = RA.shelf(y, 120.0, b, high=False); lab["bass_db"] = b
        r = float(rng.uniform(1.5, 4)); y = compress(y, -18.0, r, att=.01, rel=.15); lab["bus_ratio"] = r
        pk = np.abs(y).max() or 1.0; y = y / pk
        drive = float(rng.uniform(3, 18)); y = y * 10 ** (drive / 20); lab["limit_drive_db"] = drive
        env = maximum_filter1d(np.abs(y), int(.005 * RA.SR))                   # 5 ms look-ahead peak limiter
        k = int(.03 * RA.SR); env = np.convolve(env, np.ones(k) / k, "same")
        y = y * np.minimum(1.0, 0.98 / np.maximum(env, 1e-9))
        y = np.clip(y, -1, 1)
    return y, lab


_render_multi0 = RA.render_multi


def render_multi(*args, **kw):
    rng = args[6] if len(args) > 6 else kw["rng"]
    _S["rng"] = np.random.default_rng(int(rng.integers(2 ** 31)))      # stems' soundfont / bank / fx: reproducible
    return _render_multi0(*args, **kw)


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)     # else "--sf2" is taken as "--sf2s"
    ap.add_argument("--sf2s", required=True, help="comma-separated soundfonts")
    ap.add_argument("--sf-table", required=True, help="eval/sf2_table.py output: programs / banks / kits per soundfont")
    ap.add_argument("--lead-p", type=float, default=0.4)
    ap.add_argument("--master-p", type=float, default=0.75)
    a, rest = ap.parse_known_args()
    table = json.loads(pathlib.Path(a.sf_table).read_text())
    _S["sf2s"] = [s for s in a.sf2s.split(",")]
    _S["table"] = {s: table[s] for s in _S["sf2s"]}
    _S["master_p"] = a.master_p
    widen_families()
    RA.LEAD_P, RA.LEAD_PROGS = a.lead_p, LEAD
    RA.synth_stem, RA.master, RA.render_multi = synth_stem, master, render_multi
    # Targets are ALWAYS the written times (user rule): the piano path would move each label later by the
    # sampler's measured latency (5-16 ms) -- to where the sound starts.  No shift here, like every other render.
    import collections
    from app.ml.spectral import render as R
    R.latencies = lambda: collections.defaultdict(float)
    sys.argv = [sys.argv[0]] + rest
    RA.main()


if __name__ == "__main__":
    main()
