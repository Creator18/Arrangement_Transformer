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


def split_of_pop(n):
    return "validation" if n % 10 == 0 else "test" if n % 10 == 1 else "train"


# ---------------------------------------------------------------- arrangements
def load_pop909(d: pathlib.Path):
    """notes (N,4) [on, off, pitch, vel], role per note (0 melody, 1 bass, 2 accomp), beats (s)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
    rows, roles = [], []
    for inst in pm.instruments:
        r = 0 if inst.name.upper() == "MELODY" else 2
        for n in inst.notes:
            rows.append([n.start, n.end, n.pitch, n.velocity]); roles.append(r)
    notes, roles = np.array(rows, float), np.array(roles)
    o = np.argsort(notes[:, 0], kind="stable"); notes, roles = notes[o], roles[o]
    roles = mark_bass(notes, roles)
    rows = [l.split() for l in (d / "beat_midi.txt").read_text().splitlines() if l.strip()]
    beats = np.array([float(r[0]) for r in rows]); flags = np.array([float(r[2]) if len(r) > 2 else 0 for r in rows])
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


def synth_stem(events, sf2, end, drum=False):
    """One instrument on its own FluidSynth: events (t, kind, a, b) -> mono stem.  The vibrato
    (modulation) control is held at 0 and no pitch bend is ever sent: no note's pitch moves."""
    import fluidsynth
    fs = fluidsynth.Synth(samplerate=float(SR), gain=0.3); sfid = fs.sfload(str(sf2)); ch = 9 if drum else 0
    fs.cc(ch, 1, 0)
    out, t_now = [], 0.0
    for t, kind, x, y in sorted(events, key=lambda e: (e[0], e[1] not in ("prog", "kit", "cc"))):
        n = int(round((t - t_now) * SR))
        if n > 0:
            out.append(np.asarray(fs.get_samples(n), np.float32)[::2]); t_now += n / SR
        if kind == "prog":
            fs.program_select(ch, sfid, 0, x); fs.cc(ch, 1, 0)
        elif kind == "kit":
            fs.program_select(ch, sfid, 128, x)
        elif kind == "cc" and x != 1:
            fs.cc(ch, x, y)
        elif kind == "on":
            fs.noteon(ch, x, y)
        elif kind == "off":
            fs.noteoff(ch, x)
    n = int(round((end - t_now) * SR))
    if n > 0:
        out.append(np.asarray(fs.get_samples(n), np.float32)[::2])
    fs.delete()
    return np.concatenate(out).astype(np.float64) / 32768.0 if out else np.zeros(1)


def render_multi(notes, roles, beats, flags, sf2, combo, rng):
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
    stems = {i: [] for i in range(n_inst)}
    for sg, t0 in enumerate(bounds):
        for i in range(n_inst):
            stems[i].append((t0, "prog", prog[(sg, i)], 0))
    for i in range(n_inst):
        stems[i].append((0.0, "cc", 91, int(rng.integers(0, 70))))
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
    ev = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    for j, (s, e, p, v) in enumerate(notes):
        r = int(roles[j]); slots = slots_of[r]; sg = seg_of(s)
        group = [x for x in np.flatnonzero(ev == ev[j]) if roles[x] == r]
        if len(slots) > 1 and len(group) > 1 and active(s, "split", 1):     # chord split, any role
            targets = [slots[group.index(j) % len(slots)]]
        else:
            targets = [slots[0]]
        if n_inst > 1 and active(s, "unison", 1) and rng.random() < .6:      # the same note on another instrument
            targets.append(int(rng.choice([i for i in range(n_inst) if i != targets[0]])))
        for k, i in enumerate(dict.fromkeys(targets)):
            pr = prog[(sg, i)]; ee = e
            if leg[j] and pr in CONT_PROGS:                                  # continuous instrument + connected passage: flow
                nxt = notes[j + 1:, 0][roles[j + 1:] == r]
                if len(nxt) and nxt[0] - e < 0.25:
                    ee = max(e, float(nxt[0]) - 0.005)
            elif not leg[j] and pr in PLUCK_PROGS:                           # plucked instrument + gapped passage: short
                ee = s + max(0.05, (e - s) * float(rng.uniform(.4, .9)))
            vv = int(np.clip(v * (1.0 if k == 0 else rng.uniform(.3, .9)), 1, 127))
            stems[i] += [(s, "on", int(p), vv), (ee, "off", int(p), 0)]
        if r == 0 and active(s, "octave", 2) and rng.random() < .7:          # extra parts, NOT in the target
            q = int(p) + (12 if p < 72 else -12); i = int(rng.choice(slots))
            stems[i] += [(s, "on", q, int(v * .6)), (e, "off", q, 0)]
        if r == 0 and active(s, "under", 3) and rng.random() < .5:           # an in-chord tone under the melody
            snd = {int(x) % 12 for x in notes[(notes[:, 0] <= s) & (notes[:, 1] > s) & (roles == 2), 2]}
            cand = [int(p) - d for d in range(3, 10) if (int(p) - d) % 12 in snd]
            if cand:
                i = int(rng.choice(slots_of[2])); stems[i] += [(s, "on", cand[0], int(v * .5)), (e, "off", cand[0], 0)]
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
    drums = []
    if combo["drums"] and len(beats) > 4:
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
    mix = np.zeros(int(end * SR) + SR); eq = []
    layers = [(f"inst{i}", stems[i]) for i in range(n_inst)] + list(extra.items()) + ([("drums", drums)] if drums else [])
    for key, events in layers:
        y = synth_stem(events, sf2, end, drum=key == "drums")
        lo, hi = float(rng.uniform(-9, 9)), float(rng.uniform(-9, 9))
        y = shelf(shelf(y, 200.0, lo, high=False), 3000.0, hi, high=True)    # this layer's own bass / treble
        g = float(rng.uniform(.4, 1.0)) * (.6 if key in ("pad", "arp") else 1.0)
        n = min(len(y), len(mix)); mix[:n] += g * y[:n]; eq.append([key, round(lo, 1), round(hi, 1), round(g, 2)])
    pk = np.abs(mix).max()
    desc = dict(combo, programs={f"inst{i}": sorted({prog[(sg, i)] for sg in range(len(bounds))}) for i in range(n_inst)},
                layers_bass_treble_gain=eq)
    return (mix / pk * 0.9 if pk > 0 else mix).astype(np.float32), desc


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
    a = ap.parse_args()
    import pretty_midi
    import soundfile  # noqa: F401
    from app.ml.spectral import frontend as S, render as R
    piano_names = sorted(R.pianos())
    U = json.loads(pathlib.Path(a.units).read_text())
    m2m_split = {i: k for k, ids in U["splits"].items() for i in ids}
    jobs = [("pop", d.name, split_of_pop(int(d.name)), d) for d in sorted(pathlib.Path(a.pop909).iterdir())
            if d.is_dir() and d.name.isdigit()]
    jobs += [("m2m", i, m2m_split[i], pathlib.Path(a.m2m) / "midi_numpy" / f"{i}.npy")
             for i in sorted(m2m_split) if (pathlib.Path(a.m2m) / "midi_numpy" / f"{i}.npy").exists()]
    si, sn = (int(v) for v in a.shard.split("/"))
    T = S.n_frames(int(SECONDS * S.SR))
    done = 0
    for jx, (src, sid, split, path) in list(enumerate(jobs))[si::sn][: a.limit or None]:
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash((src, sid))) % 2 ** 32)
        notes, roles, beats, flags = load_pop909(path) if src == "pop" else load_m2m(path)
        if len(notes) < 10:
            continue
        n_win = max(1, int(math.ceil(notes[:, 1].max() / SECONDS)))
        if (out / f"{src}{sid}_r{a.pianos + a.multi - 1}_{n_win - 1:03d}.npz").exists():
            continue
        pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
        inst.notes = [pretty_midi.Note(int(np.clip(v, 1, 127)), int(p), float(s), float(e)) for s, e, p, v in notes]
        pm.instruments.append(inst)
        n_piano = a.pianos + (1 if rng.random() < a.extra_piano_p else 0)
        pianos = list(rng.choice(piano_names, size=min(n_piano, len(piano_names)), replace=False))
        renders = [("piano", p) for p in pianos] + [("multi", k) for k in range(a.multi)]
        for k, (kind, arg) in enumerate(renders):
            if kind == "multi":
                y, desc = render_multi(notes, roles, beats, flags, a.sf2, COMBOS[(jx * a.multi + arg) % len(COMBOS)], rng)
            for w in range(n_win):
                dst = out / f"{src}{sid}_r{k}_{w:03d}.npz"
                if dst.exists():
                    continue
                t0 = w * SECONDS
                if kind == "piano":
                    yy, on, fr = R.render_window(pm, t0, SECONDS, arg)
                    vel = R.velocity_roll(pm, t0, SECONDS, arg); desc_k = {"piano": arg}
                else:
                    seg = y[int(t0 * SR):int((t0 + SECONDS) * SR)]
                    yy = np.pad(seg, (0, int(SECONDS * SR) - len(seg)))
                    on, fr, vel = rolls(notes, t0, T, S.FPS); desc_k = desc
                tmp = dst.with_name(dst.stem + ".tmp.npz")
                np.savez_compressed(tmp, y=(np.clip(yy, -1, 1) * 32767).astype(np.int16), on=on, fr=fr,
                                    piano=str(arg) if kind == "piano" else "multi", render=json.dumps(desc_k))
                tmp.replace(dst)
                vt = dst.with_name(dst.stem + ".vel.tmp.npy"); np.save(vt, vel); vt.replace(dst.with_name(dst.stem + ".vel.npy"))
                done += 1
        print(f"  {split} {src}{sid}: {len(renders)} renders x {n_win} windows", flush=True)
    print(f"{done} clips written", flush=True)


if __name__ == "__main__":
    main()
