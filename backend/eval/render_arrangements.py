"""Piano arrangements -> many audio renders each, every one labelled with that ONE arrangement (CPU).

The user's design.  For every arrangement (POP909: MELODY/BRIDGE/PIANO tracks;
Music2MIDI human arrangements), rendered whole and cut into 20 s windows:

  >= 2 pure PIANO renders     each through a different sampled piano (SFZ, as the MAESTRO set)
  N  MULTI-INSTRUMENT renders  General MIDI soundfont through FluidSynth, each with random settings:

    instruments    1-2 up to a full band (--max-inst)
    swap rate      never / per section (8 bars) / every 2-4 bars
    articulation   per voice and passage, from the piano's own playing: notes with clear
                   gaps -> plucked / staccato instruments (guitar chugs, pizzicato, keys),
                   kept short; connected, legato notes -> continuous instruments (strings,
                   voice, pads, winds, organ) flowing note to note, sometimes with slides
                   and vibrato
    layering       level 0 none .. 3 heavy, each consistent (fixed for the section) or random
                   (toggling bar to bar):
                     chord split   notes of one chord on different instruments
                     unison        the same note on 2+ instruments, each its own volume and
                                   brightness (bass/treble shaping)
                     extra parts   NOT in the target: melody octave doubling, a pad holding
                                   the chord, chord arpeggios / strums, a line under the melody
    drums          sometimes, from the beat grid (kick, backbeat snare, 8th / 16th hats, fills)
    mix            per-instrument volume, pan, reverb

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
    beats = np.array([float(l.split()[0]) for l in (d / "beat_midi.txt").read_text().splitlines() if l.strip()])
    return notes, roles, beats


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
    return notes, roles, beats


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


# ---------------------------------------------------------------- multi-instrument plan
def articulation(notes, roles):
    """Per note: True = connected / legato (next note of the same voice starts as this one ends)."""
    leg = np.zeros(len(notes), bool)
    for r in (0, 1, 2):
        ix = np.flatnonzero(roles == r)
        for a, b in zip(ix[:-1], ix[1:]):
            ioi = notes[b, 0] - notes[a, 0]
            if ioi <= 0.03:
                continue
            gap = notes[b, 0] - notes[a, 1]
            leg[a] = gap / ioi < 0.2
    return leg


def plan_render(notes, roles, beats, rng):
    """Random settings for one multi-instrument render -> list of (channel events) and its description."""
    n_inst = int(rng.choice([1, 2, 3, 4, 5, 6], p=[.12, .2, .22, .2, .14, .12]))
    swap = rng.choice(["never", "section", "bars"], p=[.35, .4, .25])
    layer = int(rng.choice([0, 1, 2, 3], p=[.25, .3, .25, .2])); layer_random = bool(rng.random() < .5)
    drums = bool(rng.random() < .6)
    bar = 4 * float(np.median(np.diff(beats))) if len(beats) > 4 else 2.0
    seg_len = {"never": 1e9, "section": 8 * bar, "bars": float(rng.integers(2, 5)) * bar}[swap]
    leg = articulation(notes, roles)
    desc = dict(instruments=n_inst, swap=str(swap), layering=layer, layering_random=layer_random, drums=drums)
    return dict(n_inst=n_inst, seg_len=seg_len, layer=layer, layer_random=layer_random, drums=drums, bar=bar, leg=leg), desc


def pick_program(role, legato, rng):
    if role == 1:
        return int(rng.choice(BASS_CONT if legato else BASS_PLUCK))
    fam = CONTINUOUS if legato else PLUCKED
    if role == 0 and legato:
        fam = {k: fam[k] for k in ("voice", "strings", "winds", "brass", "lead", "organ")}
    return int(rng.choice(fam[rng.choice(list(fam))]))


def render_multi(notes, roles, beats, sf2, rng):
    """FluidSynth render of one random orchestration of the whole arrangement -> mono float32 at SR."""
    import fluidsynth
    P = plan_render(notes, roles, beats, rng); plan, desc = P
    end = float(notes[:, 1].max()) + 2.0
    fs = fluidsynth.Synth(samplerate=float(SR), gain=0.25)
    sfid = fs.sfload(str(sf2))
    ev = []                                                                 # (time, kind, ch, a, b)
    n_seg = int(math.ceil(end / plan["seg_len"])) if plan["seg_len"] < 1e8 else 1
    chans = [c for c in range(16) if c != 9]
    # per segment: an instrument for each role; the extra instruments are layers
    seg_prog = []
    for s in range(n_seg):
        progs = {r: [] for r in (0, 1, 2)}
        for i in range(plan["n_inst"]):
            r = (0, 1, 2)[i] if i < 3 else int(rng.integers(0, 3))
            progs[r].append(i)
        seg_prog.append(progs)
    # an instrument slot = a channel; its program may change per segment
    slot_prog = {}
    for s in range(n_seg):
        for r in (0, 1, 2):
            for i in seg_prog[s][r]:
                legish = rng.random() < .5
                slot_prog[(s, i)] = pick_program(r, legish, rng)
    for i in range(plan["n_inst"]):
        ch = chans[i]
        ev.append((0.0, "cc", ch, 7, int(rng.integers(70, 120)))); ev.append((0.0, "cc", ch, 10, int(rng.integers(20, 108))))
        ev.append((0.0, "cc", ch, 91, int(rng.integers(0, 80)))); ev.append((0.0, "cc", ch, 74, int(rng.integers(30, 127))))
    for s in range(n_seg):
        for i in range(plan["n_inst"]):
            ev.append((s * min(plan["seg_len"], 1e8), "prog", chans[i], slot_prog[(s, i)], 0))
    seg_of = lambda t: min(n_seg - 1, int(t // plan["seg_len"])) if n_seg > 1 else 0
    bar_of = lambda t: int(t // plan["bar"])
    layer_on = {}                                                           # (bar, kind) -> bool
    def active(t, kind, level_needed):
        if plan["layer"] < level_needed:
            return False
        if not plan["layer_random"]:
            return True
        k = (bar_of(t), kind)
        if k not in layer_on:
            layer_on[k] = rng.random() < .55
        return layer_on[k]
    ev_ix = np.concatenate([[0], np.cumsum(np.diff(notes[:, 0]) > 0.03)])
    for j, (s, e, p, v) in enumerate(notes):
        r = int(roles[j]); sg = seg_of(s); slots = seg_prog[sg][r] or [0]
        legato = plan["leg"][j]
        if legato:                                                          # continuous: flow into the next note
            nxt = notes[j + 1:, 0][(roles[j + 1:] == r)]
            e = max(e, float(nxt[0]) - 0.005) if len(nxt) and nxt[0] - e < 0.25 else e
        else:                                                               # detached: keep (or make) it short
            e = s + min(e - s, max(0.06, (e - s) * float(rng.uniform(.5, 1.0))))
        # chord split: one instrument per chord note, else the role's first instrument
        group = np.flatnonzero(ev_ix == ev_ix[j])
        if r == 2 and len(slots) > 1 and active(s, "split", 1) and len(group) > 1:
            targets = [slots[int(np.searchsorted(group, j)) % len(slots)]]
        else:
            targets = [slots[0]]
        if len(slots) > 1 and active(s, "unison", 1) and rng.random() < .6:     # unison doubling
            targets.append(slots[int(rng.integers(1, len(slots)))])
        for k, slot in enumerate(dict.fromkeys(targets)):
            vv = int(np.clip(v * (1.0 if k == 0 else rng.uniform(.3, .9)), 1, 127))
            ev.append((s, "on", chans[slot], int(p), vv)); ev.append((e, "off", chans[slot], int(p), 0))
        # extra parts, NOT in the target
        if r == 0 and active(s, "octave", 2) and rng.random() < .7:
            q = int(p) + (12 if p < 72 else -12)
            ev.append((s, "on", chans[slots[0]], q, int(v * .6))); ev.append((e, "off", chans[slots[0]], q, 0))
        if r == 0 and active(s, "under", 3) and rng.random() < .5:
            ev.append((s, "on", chans[slots[-1]], int(p) - 4, int(v * .5))); ev.append((e, "off", chans[slots[-1]], int(p) - 4, 0))
    if plan["layer"] >= 2:                                                  # a pad holding each bar's chord
        pad_ch = chans[min(plan["n_inst"], 14)]
        ev.append((0.0, "prog", pad_ch, int(rng.choice(CONTINUOUS["pad"] + CONTINUOUS["strings"])), 0))
        ev.append((0.0, "cc", pad_ch, 7, int(rng.integers(40, 80))))
        for b0 in np.arange(0, end, plan["bar"]):
            if not active(b0, "pad", 2):
                continue
            pcs = notes[(notes[:, 0] < b0 + plan["bar"]) & (notes[:, 1] > b0) & (roles == 2), 2]
            for q in sorted({int(x) % 12 for x in pcs})[:4]:
                ev.append((b0, "on", pad_ch, 48 + q, 50)); ev.append((b0 + plan["bar"] * .98, "off", pad_ch, 48 + q, 0))
    if plan["layer"] >= 3:                                                  # arpeggiate / strum the chords
        arp_ch = chans[min(plan["n_inst"] + 1, 14)]
        ev.append((0.0, "prog", arp_ch, int(rng.choice(PLUCKED["guitar"] + PLUCKED["mallet"])), 0))
        step = plan["bar"] / rng.choice([8, 16])
        for b0 in np.arange(0, end, plan["bar"]):
            if not active(b0, "arp", 3):
                continue
            pcs = sorted({int(x) for x in notes[(notes[:, 0] < b0 + plan["bar"]) & (notes[:, 1] > b0) & (roles == 2), 2]})
            if not pcs:
                continue
            for k, t in enumerate(np.arange(b0, b0 + plan["bar"], step)):
                q = pcs[k % len(pcs)]
                ev.append((t, "on", arp_ch, q, int(rng.integers(45, 80)))); ev.append((t + step * .9, "off", arp_ch, q, 0))
    if plan["drums"] and len(beats) > 4:                                    # drums on the beat grid
        ev.append((0.0, "drumkit", 9, int(rng.choice(DRUM_KITS)), 0)); ev.append((0.0, "cc", 9, 7, int(rng.integers(60, 110))))
        sub = int(rng.choice([2, 4])); bd = np.diff(beats).mean()
        for i, b in enumerate(beats):
            if i % 2 == 0 or rng.random() < .1:
                ev.append((b, "on", 9, 36, int(rng.integers(80, 120))))
            if i % 2 == 1:
                ev.append((b, "on", 9, 38, int(rng.integers(80, 120))))
            for h in range(sub):
                ev.append((b + h * bd / sub, "on", 9, 42 if rng.random() < .9 else 46, int(rng.integers(40, 90))))
            if i % 16 == 15 and rng.random() < .5:                         # fill into the next section
                for h in range(4):
                    ev.append((b + h * bd / 4, "on", 9, int(rng.choice([45, 47, 48, 50])), 100))
    # slides / vibrato on continuous melody instruments
    if rng.random() < .4:
        for i in seg_prog[0][0][:1]:
            ev.append((0.0, "cc", chans[i], 1, int(rng.integers(0, 60))))
    ev.sort(key=lambda x: (x[0], x[1] != "prog"))
    out, t_now = [], 0.0
    for t, kind, ch, a, b in ev:
        if t > t_now:
            n = int(round((t - t_now) * SR))
            if n > 0:
                out.append(np.asarray(fs.get_samples(n), np.float32)[::2]); t_now += n / SR
        if kind == "prog":
            fs.program_select(ch, sfid, 0, a)
        elif kind == "drumkit":
            fs.program_select(ch, sfid, 128, a)
        elif kind == "cc":
            fs.cc(ch, a, b)
        elif kind == "on":
            fs.noteon(ch, a, b)
        elif kind == "off":
            fs.noteoff(ch, a)
    out.append(np.asarray(fs.get_samples(int(2 * SR)), np.float32)[::2])
    fs.delete()
    y = np.concatenate(out) / 32768.0
    pk = np.abs(y).max()
    return (y / pk * 0.9 if pk > 0 else y), desc


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
    for src, sid, split, path in jobs[si::sn][: a.limit or None]:
        out = pathlib.Path(a.out) / split; out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash((src, sid))) % 2 ** 32)
        notes, roles, beats = load_pop909(path) if src == "pop" else load_m2m(path)
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
                y, desc = render_multi(notes, roles, beats, a.sf2, rng)
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
