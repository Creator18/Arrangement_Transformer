"""How much of the solo voice does fill-and-remove recover on REAL songs, with the solo bars found by the detector?

eval/aah_fill_test.py measured the fix on POP909 melodies sung as a synthetic "aah".  Here the voice is a real
singer in a real mix: the 106 held-out Music2MIDI test songs (by-song split, the songs v1 trained on excluded, as in
eval/s2p_evaluate.py).

  solo regions   detected: the solo-voice detector (app/ml/spectral/solo_train.py) per 16 ms frame, >= --det-thr,
                 read per beat (a beat is solo when its mean probability is over the bar), runs of >= 2 beats.
                 true: the detector's own label (Demucs vocal vs rest curves, eval/m2m_vocal_curves.py, >= 12 dB
                 vocal dominance) read the same way.  Scoring always uses the TRUE solo frames.
  fill           eval/aah_fill_test.py's best variant, with no POP909 labels: beats from librosa (the app's fast
                 path, 4 beats per bar); accompaniment = the non-melody notes (trained split) of the base's own
                 transcription OUTSIDE the solo regions; each solo bar gets the source bar whose accompaniment chroma
                 best fits the voice's pitch classes in that bar (nearest bar if the voice is silent); notes kept
                 below the voice's range (moved down by octaves), added at the voice's level, played as
                   pluck        a single soft plucked tone (eval/aah_test.py; the POP909 test's sound)
                   band         an instrumental mix like the base's training renders: bass + keys / guitar /
                                e-piano + (mostly) a quieter pad / strings / organ, FluidSynth General MIDI stems
                   band+drums   the same plus a drum kit on the beats (drums add no notes to remove)
                 Fill and removal use the DETECTED solo regions (the true ones cost only 0.4 points, first run).
                 Third run: band+drums only (best of the three), union only (never worse than plain), regions from
                 detector v1 AND v2 (checkpoints/solo_det2, P(quiet) + P(silent)), and six removal rules (RULES);
                 every added note in the true solo parts is classed as voice / fill leftover / other.
  remove         drop transcribed notes inside the solo regions that match an added note (same key, start within
                 50 ms), unless the plain pass found them too.  "replace": inside the solo regions the fill pass's
                 kept notes REPLACE the plain pass's (first run: some songs lost voice notes the plain pass had).
                 "union": every plain-pass note stays and only NEW fill-pass notes are added (no plain note shares
                 their key within 50 ms), so the fix can only add.

Reference voice: the skyline of the human arrangement (top note of each onset event, >= C4; 78% of a human's own
melody in POP909, eval/pop909_melody_role.py).  Scores are pooled over songs (matched notes / all notes):
recall at 50 ms and 150 ms onsets (exact pitch), and at 150 ms by pitch class (octave choices not penalised).
Pitch class is the measure that counts here: in solo passages the human arranger mostly writes the sung line an
octave ABOVE where it is sung (smoke test, 3 songs: 22 of 25 matches at +12), so exact pitch reads ~0.

    python eval/solo_fill_real.py --ckpt <base best.pt> --det <solo_det best.pt> --out <dir> [--shard k --nshards n]
    python eval/solo_fill_real.py --out <dir> --merge
"""
import argparse
import json
import pathlib
import sys

import mir_eval
import numpy as np
import pretty_midi
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.ml.spectral import frontend as S  # noqa: E402
from app.ml.spectral.solo_train import CLIP, SoloNet  # noqa: E402
from app.services import spectral_transcription as ST  # noqa: E402
from eval.aah_fill_test import mono  # noqa: E402
from eval.aah_test import SR, pluck  # noqa: E402
from eval.rhythm_groups import melody_split, top_voice  # noqa: E402

assert SR == S.SR
FPS = S.SR / S.HOP


def target(vdb, rdb, vmax):
    """The v1 detector's label (checkpoints/solo_det): vocal active, soft 0 at 6 dB -> 1 at 18 dB of vocal dominance."""
    active = (vdb > vmax - 35) & (vdb > -60)
    return (active * np.clip((vdb - rdb - 6) / 12, 0, 1)).astype(np.float32)


class SoloNetV1(SoloNet):
    """v1's head: one logit per 16 ms frame (the current SoloNet is v2, three classes)."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(self.proj.out_features, 2)

    def forward(self, x):
        B, _, _, Tn = x.shape
        h = self.proj(self.conv(x).permute(0, 3, 1, 2).flatten(2))
        h = self.enc(h + self.pos[:, : h.shape[1]])
        return self.head(h).reshape(B, -1)[:, :Tn]


def detector(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = S.SpectralFrontend().to(dev).eval()
    fe.mu.copy_(ck["fe_mu"])
    fe.sd.copy_(ck["fe_sd"])
    v1 = ck["model"]["head.weight"].shape[0] == 2           # v1: one logit / frame; v2: three classes / frame
    net = (SoloNetV1() if v1 else SoloNet()).to(dev).eval()
    net.load_state_dict(ck["model"])

    @torch.no_grad()
    def run(y):
        """Per-frame P(solo voice) for a whole song, in the detector's 10 s clips (50% overlap, centre kept)."""
        n = S.n_frames(len(y))
        acc, cnt = np.zeros(n), np.zeros(n)
        step = CLIP // 2
        for a in range(0, max(1, len(y) - step), step):
            x = y[a:a + CLIP]
            x = np.pad(x, (0, CLIP - len(x))).astype(np.float32)
            lg = net(fe(torch.from_numpy(x)[None].to(dev)))
            if v1:
                p = torch.sigmoid(lg.float())[0].cpu().numpy()
            else:                                            # P(quiet band) + P(silent band)
                p = torch.softmax(lg.float(), -1)[0, :, 1:].sum(-1).cpu().numpy()
            f0 = a // S.HOP
            m = min(len(p), n - f0)
            lo, hi = (0 if a == 0 else len(p) // 4), (m if a + CLIP >= len(y) else min(m, 3 * len(p) // 4))
            acc[f0 + lo:f0 + hi] += p[lo:hi]
            cnt[f0 + lo:f0 + hi] += 1
        return acc / np.maximum(cnt, 1)
    return run


def truth(curves, n):
    z = np.load(curves)
    tt = np.arange(n) / FPS
    src = np.arange(len(z["vocal_db"])) * float(z["hop"])
    vdb, rdb = np.interp(tt, src, z["vocal_db"]), np.interp(tt, src, z["rest_db"])
    return target(vdb, rdb, float(np.percentile(z["vocal_db"], 99.5)))


def beats_of(y):
    import librosa
    import scipy.signal
    if not hasattr(scipy.signal, "hann"):
        scipy.signal.hann = scipy.signal.windows.hann
    _, b = librosa.beat.beat_track(y=y, sr=SR, hop_length=512)
    return librosa.frames_to_time(b, sr=SR, hop_length=512)


def solo_beats(p, beats, thr):
    """Beat spans whose mean probability >= thr, in runs of at least 2 beats: list of (start, end) seconds."""
    spans = [(a, b) for a, b in zip(beats[:-1], beats[1:])]
    on = [p[int(a * FPS):max(int(a * FPS) + 1, int(b * FPS))].mean() >= thr for a, b in spans]
    runs, cur = [], []
    for s, o in zip(spans, on):
        if o:
            cur.append(s)
        else:
            if len(cur) >= 2:
                runs.append((cur[0][0], cur[-1][1]))
            cur = []
    if len(cur) >= 2:
        runs.append((cur[0][0], cur[-1][1]))
    return runs


def inside(t, runs):
    return any(a <= t < b for a, b in runs)


def choose_fill(plain, runs, beats):
    """The notes to add under the solo runs, taken from the song's own accompaniment elsewhere: (notes, voice floor)."""
    arr = np.array([[g["onset"], g["offset"], g["pitch"]] for g in plain], float)
    out_ix = [i for i, g in enumerate(plain) if not inside(g["onset"], runs)]
    acc = []
    if len(out_ix) > 10:
        sub = arr[out_ix]
        mel = melody_split(sub)
        acc = sub[~mel]
    solo_v = mono([g for g in plain if inside(g["onset"], runs)])
    lo = (np.percentile([g["pitch"] for g in solo_v], 5) - 2) if solo_v else 60
    bars = [(beats[i], beats[i + 4]) for i in range(0, len(beats) - 4, 4)]
    src = [(a, b) for a, b in bars if not any(a < r1 and b > r0 for r0, r1 in runs)]

    def chroma(notes, a, b):
        c = np.zeros(12)
        for on, off, p in notes:
            c[int(p) % 12] += max(0.0, min(off, b) - max(on, a))
        return c / (np.linalg.norm(c) + 1e-9)
    acc_l = [tuple(x) for x in acc]
    src_c = [chroma([x for x in acc_l if a <= x[0] < b], a, b) for a, b in src]
    src_n = [sum(a <= x[0] < b for x in acc_l) for a, b in src]
    added = []
    for b0, b1 in bars:
        if not any(b0 < r1 and b1 > r0 for r0, r1 in runs) or not src:
            continue
        vc = chroma([(g["onset"], g["offset"], g["pitch"]) for g in solo_v], b0, b1)
        cand = [j for j in range(len(src)) if src_n[j] >= 4] or list(range(len(src)))
        if vc.sum() > 0:
            j = max(cand, key=lambda j: (float(vc @ src_c[j]), -abs(src[j][0] - b0)))
        else:
            j = min(cand, key=lambda j: abs(src[j][0] - b0))
        sa, sb = src[j]
        k = (b1 - b0) / (sb - sa)
        for on, off, p in acc_l:
            if sa <= on < sb:
                ns, ne = b0 + (on - sa) * k, min(b0 + (off - sa) * k, b1 + 0.3)
                if not inside(ns, runs):
                    continue
                while p > lo - 1:
                    p -= 12
                if p >= 28:
                    added.append(pretty_midi.Note(80, int(p), ns, ne))
    return added, lo


def band(added, n, runs, beats, rng, sf2s, drums):
    """The chosen notes played as an instrumental mix, the way the base's training renders sound
    (eval/render_arrangements.py's FluidSynth stems and General MIDI families): the lowest note of each onset event
    on a bass (moved into 28-52 by octaves), the rest on a keys / guitar / e-piano and, most of the time, doubled
    by a quieter pad / strings / organ; with `drums`, a kit (kick, snare, hats) on the beats inside the solo runs.
    Returns (audio, the pitched notes actually played) so removal works on the real keys."""
    import render_arrangements as RA
    end = n / SR + 0.5
    ns = sorted(added, key=lambda f: (f.start, f.pitch))
    ev = np.concatenate([[0], np.cumsum(np.diff([f.start for f in ns]) > 0.03)]) if ns else []
    bass, chord = [], []
    for e in np.unique(ev):
        grp = [ns[i] for i in np.flatnonzero(ev == e)]
        b, rest = grp[0], grp[1:]
        p = b.pitch
        while p > 52:
            p -= 12
        bass.append(pretty_midi.Note(b.velocity, p, b.start, b.end))
        chord += rest if rest else ([b] if b.pitch > 52 else [])
    sf2 = sf2s[int(rng.integers(len(sf2s)))]
    progs = dict(bass=int(rng.choice(RA.BASS_PLUCK + RA.BASS_CONT)),
                 keys=int(rng.choice(RA.PLUCKED["guitar"] + RA.PLUCKED["keys"] + RA.CONTINUOUS["epiano"])),
                 pad=int(rng.choice(RA.CONTINUOUS["strings"] + RA.CONTINUOUS["pad"] + RA.CONTINUOUS["organ"])))

    def stem(notes, prog, vel=(70, 105)):
        evs = [(0.0, "prog", prog, 0)]
        for f in notes:
            evs += [(f.start, "on", f.pitch, int(rng.integers(*vel))), (max(f.end, f.start + 0.05), "off", f.pitch, 0)]
        y = RA.synth_stem(evs, sf2, end)
        return y / (np.abs(y).max() + 1e-9)
    out = stem(bass, progs["bass"]) * rng.uniform(0.8, 1.0)
    out = out + stem(chord, progs["keys"]) * rng.uniform(0.6, 1.0)
    if rng.random() < 0.7:
        out = out + stem(chord, progs["pad"], (50, 80)) * rng.uniform(0.25, 0.5)
    if drums:
        evs = [(0.0, "kit", int(rng.choice(RA.DRUM_KITS)), 0)]
        bt = [t for t in beats if inside(t, runs)]
        bd = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.5
        for i, t in enumerate(bt):
            k = 36 if i % 2 == 0 else 38
            evs += [(t, "on", k, int(rng.integers(80, 120))), (t + 0.1, "off", k, 0)]
            for h in range(2):
                evs += [(t + h * bd / 2, "on", 42, int(rng.integers(40, 90))), (t + h * bd / 2 + 0.05, "off", 42, 0)]
        y = RA.synth_stem(evs, sf2, end, drum=True)
        out = out + y / (np.abs(y).max() + 1e-9) * rng.uniform(0.4, 0.8)
    out = np.pad(out, (0, max(0, n - len(out))))[:n]
    return out, bass + chord


def fill_pass(y, runs, added, style, ck, thr, level, tmp, rng, sf2s):
    """Play `added` under the solo runs (style pluck / band / band+drums) and transcribe: (notes, played)."""
    if style == "pluck":
        add, played = pluck(added, len(y), 1.0), added
    else:
        add, played = band(added, len(y), runs, BEATS, rng, sf2s, drums=style == "band+drums")
    mix = y.astype(np.float64).copy()
    for r0, r1 in runs:                              # the added part at the voice's own level, run by run
        a, b = int(r0 * SR), int(r1 * SR)
        ra = np.sqrt((add[a:b] ** 2).mean() + 1e-12)
        if ra > 1e-6:
            mix[a:b] += level * add[a:b] * np.sqrt((y[a:b].astype(np.float64) ** 2).mean()) / ra
    mix = (mix * min(1.0, 0.95 / (np.abs(mix).max() + 1e-9))).astype(np.float32)
    sf.write(tmp, mix, SR)
    return ST.transcribe(tmp, onset_thr=thr, ckpt=ck, frame_thr=0.5), played


# removal rules for the NEW notes (fill-pass notes in the solo runs that the plain pass did not have):
#   strict         same key as a played note, start within 50 ms (the rule so far)
#   floor          strict, and also every new note below the voice floor
#   oct+wide       same pitch class as a played note, within 100 ms
#   +voice         and keep only new notes the SUNG LINE follows: a pYIN pitch track of the mix inside the solo
#                  runs (there the mix is mostly the voice) sits within 1 semitone of the note for >= 40% of the
#                  50-200 ms after its start (past a scoop at the start); "-pc" accepts the track in any octave
#   +voice all     the voice filter on EVERY note in the solo runs, the plain pass's too (can lose voice notes)
RULES = {"strict": (False, 0.05, False, None), "floor": (False, 0.05, True, None),
         "oct+wide": (True, 0.10, False, None), "strict+voice": (False, 0.05, False, "exact"),
         "floor+voice": (False, 0.05, True, "exact"), "floor+voice-pc": (False, 0.05, True, "pc"),
         "floor+voice all": (False, 0.05, True, "all")}
TRACK_FPS = 16000 / 256


def voice_track(y, runs):
    """MIDI pitch per 16 ms of the dominant sung line inside the runs (pYIN at 16 kHz), nan elsewhere / unvoiced."""
    import librosa
    tr = np.full(int(len(y) / SR * TRACK_FPS) + 2, np.nan)
    merged = []
    for r0, r1 in sorted(runs):
        if merged and r0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r1)
        else:
            merged.append([r0, r1])
    for r0, r1 in merged:
        z = librosa.resample(y[int(r0 * SR):int(r1 * SR)].astype(np.float32), orig_sr=SR, target_sr=16000)
        if len(z) < 2048:
            continue
        f0, v, _ = librosa.pyin(z, fmin=80, fmax=1100, sr=16000, frame_length=1024, hop_length=256)
        m = np.where(v, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
        i0 = int(r0 * TRACK_FPS)
        k = min(len(m), len(tr) - i0)
        tr[i0:i0 + k] = m[:k]
    return tr


def follows(g, tr, pc=False):
    """The sung line sits on this note for >= 40% of 50-200 ms after its start (nan counts as no)."""
    a = int((g["onset"] + 0.05) * TRACK_FPS)
    b = max(a + 1, int(min(g["offset"] + 0.05, g["onset"] + 0.2) * TRACK_FPS))
    seg = tr[a:b]
    if not len(seg):
        return False
    d = np.abs(seg - g["pitch"])
    if pc:
        d = np.minimum(d % 12, 12 - d % 12)
    return bool(np.mean(d <= 1.0) >= 0.4)


def union_with(plain, got, played, runs, lo, rule, tr=None):
    """Every plain-pass note, plus the fill pass's new notes in the solo runs that survive `rule`: (notes, new)."""
    oct_, tol, floor, voice = RULES[rule]
    seen = lambda g: any(g["pitch"] == x["pitch"] and abs(g["onset"] - x["onset"]) < 0.05 for x in plain)
    same = (lambda p, q: p % 12 == q % 12) if oct_ else (lambda p, q: p == q)
    new = [g for g in got if inside(g["onset"], runs) and not seen(g)
           and not any(same(g["pitch"], f.pitch) and abs(g["onset"] - f.start) < tol for f in played)
           and not (floor and g["pitch"] < lo)]
    base = plain
    if voice:
        new = [g for g in new if follows(g, tr, pc=voice == "pc")]
        if voice == "all":
            base = [g for g in plain if not inside(g["onset"], runs) or follows(g, tr)]
    return sorted(base + new, key=lambda g: g["onset"]), new


def leftover(g, played):
    """A new note that is (probably) the fill itself: a played note's pitch class within 150 ms."""
    return any(g["pitch"] % 12 == f.pitch % 12 and abs(g["onset"] - f.start) < 0.15 for f in played)


BEATS = []


def match(ref, est, tol, pc=False, pairs=False):
    """Number of ref notes matched by est (onset within tol, same pitch or pitch class); with pairs, the
    octave offsets (ref - est, semitones) of the matched pairs too."""
    if not len(ref) or not len(est):
        return (0, []) if pairs else 0
    f = (lambda p: 60 + int(p) % 12) if pc else (lambda p: p)
    ri = np.array([[r[0], max(r[1], r[0] + 1e-3)] for r in ref])
    rp = np.array([pretty_midi.note_number_to_hz(f(r[2])) for r in ref])
    ei = np.array([[g["onset"], max(g["offset"], g["onset"] + 1e-3)] for g in est])
    ep = np.array([pretty_midi.note_number_to_hz(f(g["pitch"])) for g in est])
    m = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=tol, offset_ratio=None)
    if pairs:
        return len(m), [int(ref[i][2] - est[j]["pitch"]) for i, j in m]
    return len(m)


DETS = ["v1", "v2"]
SYSTEMS = ["plain"] + [f"{d} {r}" for d in DETS for r in RULES]


def summarize(rows, out):
    """Pooled scores over songs (matched notes / all voice notes) from the per-song rows; writes the json."""
    ref_n = sum(r["ref"] for r in rows)
    cover = np.sum([r["cover"] for r in rows], 0)
    print(f"\n{len(rows)} songs; {ref_n} voice notes (human skyline) in true solo parts; "
          f"true solo {cover[2] / FPS / 60:.1f} min")
    for d in DETS:
        c = np.sum([r["det"][d]["cover"] for r in rows], 0)
        print(f"detector {d} frames: recall {c[0] / max(c[2], 1):.3f}  precision {c[0] / max(c[1], 1):.3f}  "
              f"flagged {c[1] / FPS / 60:.1f} min")
    print(f"\n   {'system':<18}{'R 50ms':>8}{'R 150ms':>9}{'R pc':>7}{'notes/voice':>13}"
          f"{'added':>8}{'voice':>8}{'leftover':>10}{'other':>8}")
    summ = {}
    for k in SYSTEMS:
        m50, m150, mpc, ne = np.sum([r[k] for r in rows], 0)
        o = np.array([d for r in rows for d in r[k + " oct"]])
        summ[k] = dict(r50=m50 / max(ref_n, 1), r150=m150 / max(ref_n, 1), rpc=mpc / max(ref_n, 1),
                       per_voice=ne / max(ref_n, 1),
                       octave_offsets={int(d): int((o == d).sum()) for d in np.unique(o)} if len(o) else {})
        nw = np.sum([r[k + " new"] for r in rows if k + " new" in r], 0) if k != "plain" else np.zeros(4)
        summ[k]["added"] = dict(n=int(nw[0]), voice=int(nw[1]), leftover=int(nw[2]), other=int(nw[3]))
        s = summ[k]
        tail = (f"{int(nw[0]):>8d}{nw[1] / max(nw[0], 1):>8.0%}{nw[2] / max(nw[0], 1):>10.0%}{nw[3] / max(nw[0], 1):>8.0%}"
                if k != "plain" else "")
        print(f"   {k:<18}{s['r50']:>8.3f}{s['r150']:>9.3f}{s['rpc']:>7.3f}{s['per_voice']:>13.2f}" + tail)
    print("added = new notes in the true solo parts; voice = same pitch class as a sung note within 150 ms;\n"
          "leftover = a played fill note's pitch class within 150 ms; other = neither (drums, bleed, ...)")
    print("\nhuman minus transcribed pitch, pitch-class matches (semitones: count):")
    for k in SYSTEMS:
        print(f"   {k:<18}{summ[k]['octave_offsets']}")
    f = out / "solo_fill_real.json"
    f.write_text(json.dumps(dict(songs=rows, summary=summ, ref_notes=ref_n,
                                 det_recall=float(cover[0] / max(cover[2], 1)),
                                 det_precision=float(cover[0] / max(cover[1], 1)),
                                 true_solo_min=float(cover[2] / FPS / 60)), indent=1))
    print(f"\nwrote {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", default="/scratch/basu.ang/m2m")
    ap.add_argument("--audio", default="/scratch/basu.ang/solo_det/m2m")
    ap.add_argument("--curves", default="/scratch/basu.ang/solo_det/m2m_curves")
    ap.add_argument("--ckpt")
    ap.add_argument("--det", help="v1 detector (checkpoints/solo_det)")
    ap.add_argument("--det2", help="v2 detector (checkpoints/solo_det2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--sf2s", default="/scratch/basu.ang/datasets/OmegaGMGS2.sf2", help="comma-separated soundfonts")
    ap.add_argument("--keep-audio", type=int, default=0, help="keep the mix and filled audio of the first N songs")
    ap.add_argument("--merge", action="store_true", help="pool the shards' rows in --out and print the summary")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        rows = [r for f in sorted(out.glob("rows_*.json")) for r in json.loads(f.read_text())]
        summarize(rows, out)
        return
    m2m = pathlib.Path(a.m2m)
    U = json.loads((m2m / "m2m_units.json").read_text())
    old = set(json.loads((m2m / "m2m_splits.json").read_text())["train"])
    test = [i for i in U["splits"]["test"] if i not in old][: a.limit or None][a.shard::a.nshards]
    dets = {"v1": detector(a.det), "v2": detector(a.det2)}
    ck = pathlib.Path(a.ckpt)
    sf2s = [pathlib.Path(x) for x in a.sf2s.split(",")]
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))           # render_arrangements
    done = out / f"rows_{a.shard}.json"
    rows = json.loads(done.read_text()) if done.exists() else []       # resume a killed shard
    test = [i for i in test if i not in {r["song"] for r in rows}]
    for sid in test:
        y = np.load(pathlib.Path(a.audio) / f"{sid}.npy").astype(np.float32)
        n = S.n_frames(len(y))
        t = truth(pathlib.Path(a.curves) / f"{sid}.npz", n)
        tf = t >= 0.5
        beats = beats_of(y)
        BEATS[:] = list(beats)
        wav = out / f"{sid}_mix.wav"
        sf.write(wav, y, SR)
        plain = ST.transcribe(wav, onset_thr=a.thr, ckpt=ck, frame_thr=0.5)
        res, new, meta = {"plain": plain}, {}, {}
        probs = {d: dets[d](y) for d in DETS}
        allruns = {d: solo_beats(probs[d], beats, a.det_thr) for d in DETS}
        tr = voice_track(y, [r for d in DETS for r in allruns[d]])
        for d in DETS:
            p = probs[d]
            df = p >= a.det_thr
            runs = allruns[d]
            added, lo = choose_fill(plain, runs, beats) if runs else ([], 0)
            meta[d] = dict(det_s=round(df.sum() / FPS, 1), added=len(added),
                           cover=[int((tf & df).sum()), int(df.sum()), int(tf.sum())])
            if added:
                rng = np.random.default_rng(int(sid) % (2 ** 31))     # the same band for a song in every run
                got, played = fill_pass(y, runs, added, "band+drums", ck, a.thr, a.level,
                                        out / f"{sid}_fill_{d}.wav", rng, sf2s)
            for r in RULES:
                res[f"{d} {r}"], new[f"{d} {r}"] = union_with(plain, got, played, runs, lo, r, tr) if added else (plain, [])
                new[f"{d} {r}"] = (new[f"{d} {r}"], played if added else [])
        hum = np.load(m2m / "midi_numpy" / f"{sid}.npy")
        hum = hum[np.lexsort((-hum[:, 2], hum[:, 0]))]
        mel = hum[top_voice(hum)]
        in_t = lambda x: tf[min(n - 1, int(x * FPS))]
        ref = [r for r in mel if in_t(r[0])]
        row = dict(song=sid, solo_s=round(tf.sum() / FPS, 1), ref=len(ref), det=meta, cover=meta["v1"]["cover"])
        for k in SYSTEMS:
            ns = res[k]
            est = [g for g in ns if in_t(g["onset"])]
            npc, offs = match(ref, est, 0.15, pc=True, pairs=True)
            row[k] = [match(ref, est, 0.05), match(ref, est, 0.15), npc, len(est)]
            row[k + " oct"] = offs
            if k in new:                                     # what the added notes are, inside the true solo parts
                nn, played = new[k]
                nn = [g for g in nn if in_t(g["onset"])]
                voice = [g for g in nn if any(g["pitch"] % 12 == int(r[2]) % 12 and abs(g["onset"] - r[0]) < 0.15
                                              for r in ref)]
                rest = [g for g in nn if g not in voice]
                left = [g for g in rest if leftover(g, played)]
                row[k + " new"] = [len(nn), len(voice), len(left), len(rest) - len(left)]
        row["ref_notes"] = [[round(float(r[0]), 3), int(r[2])] for r in ref]
        row["plain_notes"] = [[round(g["onset"], 3), g["pitch"]] for g in plain if in_t(g["onset"])]
        rows.append(row)
        if len(rows) > a.keep_audio:
            for f in [wav] + [out / f"{sid}_fill_{d}.wav" for d in DETS]:
                f.unlink(missing_ok=True)
        (out / f"rows_{a.shard}.json").write_text(json.dumps(rows))
        print(f"{sid}: true solo {row['solo_s']:5.1f}s  v1 {meta['v1']['det_s']:5.1f}s  v2 {meta['v2']['det_s']:5.1f}s  "
              f"voice notes {len(ref):3d}  "
              + "  ".join(f"{k} {row[k][2]}" for k in SYSTEMS), flush=True)
    if a.nshards == 1:
        summarize(rows, out)


if __name__ == "__main__":
    main()
