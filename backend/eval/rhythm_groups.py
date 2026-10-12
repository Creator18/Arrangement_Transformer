"""Rhythm groups: label every passage of every arrangement by its ACCOMPANIMENT rhythm.

The user's design (memory: rhythm-group-design): passages are grouped across
ALL songs by the rhythm of the accompaniment only — not pitch, harmony,
melody or song — because how a passage is arranged depends on its rhythmic
feel.  Passage length follows the song's meter.  Melody is separated so it
can ride on any group.  Three stages:

  meter        beats, bars and meter COUNTED FROM THE NOTES.  Music2MIDI notes
               come from MuseScore scores warped onto the recording, so the
               score's grid is still in them: an accent envelope (onsets,
               weighted up for bass notes and chords) -> dynamic-programming
               beat tracking (local, so warping is followed) -> the meter
               whose downbeats carry the strongest accents (2, 3 or 4 beats
               per bar) and the beat subdivision (duple vs triple: 6/8 is two
               dotted beats split in three).  POP909 ships true beats and
               downbeats, which score the method before it is trusted.
  fingerprint  melody / accompaniment split (POP909: its own MELODY track;
               Music2MIDI: top voice, tuned against POP909's real split), then
               per passage a beat-relative onset grid of the accompaniment —
               12 slots per beat (holds 16ths and triplets), bass and middle
               registers separately, binary: rhythm only.
  cluster      within each meter, k-means on the fingerprints with k chosen
               from the data: the smallest k whose groups are tight enough,
               capped so they never fragment into a plethora.  Each group gets
               a membership threshold (its members' 10th-percentile
               similarity); passages below it stay unassigned.  Writes group
               ids per passage, group sizes, and text pictures of each group's
               rhythm plus example passages for spot checks.

    python eval/rhythm_groups.py --stage meter --source pop909 --out <dir> [--shard i/n]
    python eval/rhythm_groups.py --stage fingerprint --source m2m --out <dir> [--shard i/n]
    python eval/rhythm_groups.py --stage cluster --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
import scipy.signal, scipy.signal.windows                       # noqa: E402
if not hasattr(scipy.signal, "hann"):                          # librosa 0.10 on new scipy
    scipy.signal.hann = scipy.signal.windows.hann
sys.path.insert(0, str(ROOT))
FPS = 100                       # accent-envelope frame rate
SLOTS = 12                      # grid slots per beat: 16ths (3 per 8th... 12/4) and triplets (12/3)
BASS_SPLIT = 55                 # G3: below = bass register of the accompaniment


# ── sources ──────────────────────────────────────────────────────────────────
def pieces(source, data, shard):
    """yield (id, notes [on,off,pitch,vel], melody_mask or None, true_beats or None, true_downbeats or None)."""
    si, sn = (int(v) for v in shard.split("/"))
    if source == "pop909":
        import pretty_midi
        dirs = sorted((data / "POP909").glob("[0-9][0-9][0-9]"))[si::sn]
        for d in dirs:
            pm = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
            rows, mel = [], []
            for inst in pm.instruments:
                for n in inst.notes:
                    rows.append((n.start, n.end, n.pitch, n.velocity)); mel.append(inst.name == "MELODY")
            b = np.loadtxt(d / "beat_midi.txt")
            yield d.name, np.array(rows), np.array(mel), b[:, 0], b[b[:, 2] > 0, 0]
    else:
        m2m = data / "music2midi"
        sp = json.loads((m2m / "m2m_splits.json").read_text())
        npy = m2m / "midi_numpy" if (m2m / "midi_numpy").is_dir() else m2m / "music2midi-dataset" / "midi_numpy"
        ids = sorted(sp["train"] + sp["validation"] + sp["test"])[si::sn]
        for s in ids:
            yield s, np.load(npy / f"{s}.npy"), None, None, None


# ── meter, counted from the notes ────────────────────────────────────────────
def accent_envelope(notes):
    T = int((notes[:, 0].max() + 2) * FPS)
    env = np.zeros(T)
    for on, off, p, v in notes:
        i = int(on * FPS)
        env[i] += 1.0 + (1.0 if p < BASS_SPLIT else 0.0)
    return np.convolve(env, np.hanning(5), "same")


def track_beats(notes):
    import librosa
    env = accent_envelope(notes)
    # librosa's DP tracker on our own envelope: sr / hop = FPS
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=FPS * 100, hop_length=100,
                                           start_bpm=110, tightness=120, trim=False)
    return np.asarray(beats) / FPS, float(np.atleast_1d(tempo)[0])


def harmonic_novelty(notes, beats, span=2):
    """Per beat: how much the pitch-class content changes across it (chords change at bars)."""
    on, p, off = notes[:, 0], notes[:, 2].astype(int), notes[:, 1]
    B = len(beats)
    ch = np.zeros((B, 12))
    for i in range(B - 1):
        sel = (on < beats[i + 1]) & (off > beats[i])                  # sounding in beat i
        np.add.at(ch[i], p[sel] % 12, 1.0)
    ch /= np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9
    nov = np.zeros(B)
    for i in range(span, B - span):
        a, b = ch[i - span:i].mean(0), ch[i:i + span].mean(0)
        nov[i] = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return nov


def beat_accents(notes, beats):
    on, p = notes[:, 0], notes[:, 2]
    idx = np.clip(np.searchsorted(beats, on, "right") - 1, 0, len(beats) - 2)
    frac = (on - beats[idx]) / np.maximum(1e-3, beats[idx + 1] - beats[idx])
    near = frac < 0.12
    acc = np.zeros(len(beats))
    np.add.at(acc, idx[near], 1.0 + (p[near] < BASS_SPLIT) * 2.0)
    return acc / (acc.std() + 1e-9) + 2.0 * harmonic_novelty(notes, beats) / 0.2


def settle_tempo(notes, beats, tempo):
    """Fast beat -> keep every other beat when one phase carries clearly more accent.

    A tracker on a ballad locks onto eighth notes (7 of 21 POP909 songs at 2x);
    bass notes and chord changes land on the quarter-note beat, so when one of
    the two alternating phases is markedly stronger, the beat is half as fast.
    """
    # One metrical level for every song — the 60-110 BPM "foot-tapping" beat —
    # so a pattern looks the same however the tracker locked on (POP909: 16 of
    # 101 songs tracked at 2x, 7 at 0.5x).  Halve a fast beat on its stronger
    # phase; split a slow beat at its midpoints.
    if len(beats) < 16:
        return beats, tempo
    if tempo > 110:
        a = beat_accents(notes, beats)
        e, o = a[0::2].mean(), a[1::2].mean()
        return beats[0 if e >= o else 1::2], tempo / 2
    if tempo < 58:
        mid = (beats[:-1] + beats[1:]) / 2
        return np.sort(np.concatenate([beats, mid])), tempo * 2
    return beats, tempo


def meter_from_beats(notes, beats):
    """(beats_per_bar, downbeat phase, subdivision 2|3, downbeat times)."""
    if len(beats) < 16:
        return 4, 0, 2, beats[::4]
    on = notes[:, 0]
    idx = np.clip(np.searchsorted(beats, on, "right") - 1, 0, len(beats) - 2)
    frac = (on - beats[idx]) / np.maximum(1e-3, beats[idx + 1] - beats[idx])
    a = beat_accents(notes, beats)            # bass onsets + harmonic change at each beat
    best, bm, bp = -1e9, 4, 0
    for m, prior in ((4, 0.15), (3, 0.0), (2, -0.05)):
        for ph in range(m):
            down = a[ph::m].mean(); rest = np.delete(a, np.arange(ph, len(a), m)).mean()
            sc = (down - rest) / (a.std() + 1e-9) + prior
            if sc > best:
                best, bm, bp = sc, m, ph
    # subdivision: are off-beat onsets at thirds or halves of the beat?
    off = frac[(frac > 0.12) & (frac < 0.88)]
    thirds = np.mean(np.minimum(np.abs(off - 1 / 3), np.abs(off - 2 / 3)) < 0.06) if len(off) else 0
    halves = np.mean(np.abs(off - 0.5) < 0.06) if len(off) else 0
    sub = 3 if thirds > 1.3 * halves else 2
    return bm, bp, sub, beats[bp::bm]


def meter_name(m, sub):
    return {(4, 2): "4/4", (3, 2): "3/4", (2, 2): "2/4", (2, 3): "6/8", (4, 3): "12/8", (3, 3): "9/8"}[(m, sub)]


# ── melody / accompaniment and fingerprints ──────────────────────────────────
def top_voice(notes):
    """Skyline melody: highest note of each onset event, if at or above C4."""
    on = notes[:, 0]
    ev = np.concatenate([[0], np.cumsum(np.diff(on) > 0.03)])
    mel = np.zeros(len(notes), bool)
    for e in np.unique(ev):
        ix = np.flatnonzero(ev == e)
        j = ix[np.argmax(notes[ix, 2])]
        mel[j] = notes[j, 2] >= 60
    return mel


def passage_bars(m, sub):
    """Bars per passage so a passage spans ~4 quarter-beats of music: meter-dependent."""
    return {"4/4": 1, "3/4": 2, "2/4": 2, "6/8": 2, "12/8": 1, "9/8": 1}[meter_name(m, sub)]


def fingerprints(notes, mel, beats, m, ph, sub):
    """One rhythm vector per passage: accompaniment onsets on a beat-relative grid."""
    acc = notes[~mel]
    bpb = passage_bars(m, sub) * m
    down = np.arange(ph, len(beats) - bpb, bpb)
    out = []
    for k, b0 in enumerate(down):
        g = np.zeros((2, bpb * SLOTS), np.uint8)
        for j in range(bpb):
            t0, t1 = beats[b0 + j], beats[b0 + j + 1]
            sel = acc[(acc[:, 0] >= t0 - 0.02) & (acc[:, 0] < t1 - 0.02)]
            for on, off, p, v in sel:
                s = int(np.clip(round((on - t0) / (t1 - t0) * SLOTS), 0, SLOTS - 1))
                g[0 if p < BASS_SPLIT else 1, j * SLOTS + s] = 1
        out.append((float(beats[b0]), g.ravel()))
    return out


def canonical(v, bpb):
    """Rotate a passage by whole beats so its strongest bass attack sits on beat 1.

    A misplaced downbeat (18 of 68 well-tracked POP909 songs) only rotates the
    passage; rotating every passage to the same anchor puts the same rhythm in
    the same group regardless.  Falls back to the mid register without bass.
    """
    g = v.reshape(2, bpb, SLOTS)
    reg = 0 if g[0].sum() else 1
    k = int(np.argmax(g[reg, :, 0] * 2 + g[reg].sum(1) * 0.1))
    return np.roll(g, -k, axis=1).ravel()


def picture(v, bpb):
    """Text picture of a rhythm vector: one row per register, '|' per beat."""
    g = v.reshape(2, bpb, SLOTS)
    rows = []
    for r, name in ((1, "mid "), (0, "bass")):
        cells = ["".join("x" if g[r, b, s] > 0.5 else ("." if s % 3 == 0 else " ") for s in range(SLOTS))
                 for b in range(bpb)]
        rows.append(f"      {name} |" + "|".join(cells) + "|")
    return "\n".join(rows)


# ── stages ───────────────────────────────────────────────────────────────────
def stage_meter_fp(args, with_fp):
    from mir_eval.beat import f_measure
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}_{args.shard.replace('/', 'of')}"
    rows, evals = [], []
    for pid, notes, mel, tb, td in pieces(args.source, pathlib.Path(args.data), args.shard):
        if len(notes) < 50:
            continue
        o = np.argsort(notes[:, 0], kind="stable")
        notes = notes[o]
        mel = mel[o] if mel is not None else None
        beats, tempo = track_beats(notes)
        beats, tempo = settle_tempo(notes, beats, tempo)
        m, ph, sub, downs = meter_from_beats(notes, beats)
        rec = dict(id=pid, source=args.source, tempo=tempo, beats_per_bar=m, subdivision=sub,
                   meter=meter_name(m, sub), n_beats=len(beats))
        if tb is not None:                                   # POP909: score the counting
            fb = f_measure(np.asarray(tb), beats)
            fd = f_measure(np.asarray(td), np.asarray(downs))
            sky = top_voice(notes)
            tp = (sky & mel).sum(); prec = tp / max(1, sky.sum()); rec_ = tp / max(1, mel.sum())
            rec.update(beat_f=fb, downbeat_f=fd, skyline_p=prec, skyline_r=rec_)
            evals.append((fb, fd, prec, rec_, m == 4))
        if with_fp:
            mmask = mel if mel is not None else top_voice(notes)
            fps = fingerprints(notes, mmask, beats, m, ph, sub)
            rec["passages"] = [(t, v.tolist()) for t, v in fps]
        rows.append(rec)
        print(f"  {pid} {rec['meter']} {tempo:.0f} bpm"
              + (f"  beatF {rec['beat_f']:.2f} downF {rec['downbeat_f']:.2f}" if tb is not None else ""), flush=True)
    (out / f"{'fp' if with_fp else 'meter'}_{tag}.json").write_text(json.dumps(rows))
    if evals:
        e = np.array(evals, float)
        print(f"\nPOP909 check ({len(e)} songs): beat F {e[:, 0].mean():.3f}, downbeat F {e[:, 1].mean():.3f}, "
              f"meter=4 on {e[:, 4].mean():.0%} (POP909 is almost all 4/4); top-voice melody vs the real "
              f"MELODY track: precision {e[:, 2].mean():.3f}, recall {e[:, 3].mean():.3f}")


def stage_cluster(args):
    from sklearn.cluster import KMeans
    out = pathlib.Path(args.out)
    recs = [r for f in sorted(out.glob("fp_*.json")) for r in json.loads(f.read_text())]
    by_meter = {}
    for r in recs:
        for t, v in r.get("passages", []):
            v = np.array(v, np.float32)
            if v.sum() == 0:
                continue                                     # silent accompaniment: no rhythm to group
            by_meter.setdefault(r["meter"], []).append((r["source"], r["id"], t, v))
    report, labels = [], {}
    for meter, items in sorted(by_meter.items(), key=lambda x: -len(x[1])):
        bpb = items[0][3].shape[0] // (2 * SLOTS)
        X = np.stack([canonical(v, bpb) for *_, v in items]); X /= np.linalg.norm(X, axis=1, keepdims=True)
        if len(X) < 200:
            report.append(f"{meter}: {len(X)} passages — too few to group, left unassigned")
            continue
        sub = X[np.random.default_rng(0).choice(len(X), min(len(X), 40000), replace=False)]
        tight, chosen = {}, None
        for k in (4, 6, 8, 10, 12, 16, 20, 24, 30, 40):       # cap: never more than 40 groups per meter
            km = KMeans(k, n_init=3, random_state=0).fit(sub)
            sims = (sub * km.cluster_centers_[km.labels_] / np.linalg.norm(km.cluster_centers_[km.labels_], axis=1, keepdims=True)).sum(1)
            tight[k] = float(np.median(sims))
            # smallest k past which adding groups stops buying tightness (< 0.01 per step)
            ks = sorted(tight)
            if len(ks) >= 2 and tight[ks[-1]] - tight[ks[-2]] < 0.01:
                chosen = ks[-2]; break
        chosen = chosen or max(tight)
        km = KMeans(chosen, n_init=5, random_state=0).fit(X)
        C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        sim = (X * C[km.labels_]).sum(1)
        thr = {g: float(np.percentile(sim[km.labels_ == g], 10)) for g in range(chosen)}
        order = np.argsort(-np.bincount(km.labels_, minlength=chosen))
        name = {g: f"{meter}-{chr(65 + i) if i < 26 else 'Z' + str(i)}" for i, g in enumerate(order)}
        report.append(f"\n{meter}: {len(X)} passages -> {chosen} groups "
                      f"(median similarity by k: " + ", ".join(f"{k}:{v:.2f}" for k, v in tight.items()) + ")")
        for g in order:
            mem = np.flatnonzero(km.labels_ == g)
            inside = mem[sim[mem] >= thr[g]]
            report.append(f"  {name[g]}: {len(mem)} passages ({len(mem) / len(X):.1%}), "
                          f"threshold sim {thr[g]:.2f}, {len(set(items[i][1] for i in mem))} songs")
            report.append(picture(km.cluster_centers_[g] / km.cluster_centers_[g].max(), bpb))
            ex = np.random.default_rng(int(g)).choice(inside, min(3, len(inside)), replace=False)
            report.append("      e.g. " + "; ".join(f"{items[i][0]} {items[i][1]} @ {items[i][2]:.1f}s" for i in ex))
        for i, (src, pid, t, _) in enumerate(items):
            labels.setdefault(f"{src}:{pid}", []).append(
                (t, name[km.labels_[i]] if sim[i] >= thr[km.labels_[i]] else None))
    (out / "groups_report.txt").write_text("\n".join(report))
    (out / "group_labels.json").write_text(json.dumps(labels))
    print("\n".join(report))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["meter", "fingerprint", "cluster"])
    ap.add_argument("--source", default="pop909", choices=["pop909", "m2m"])
    ap.add_argument("--data", default=str(ROOT.parent / "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    if args.stage == "cluster":
        stage_cluster(args)
    else:
        stage_meter_fp(args, args.stage == "fingerprint")


if __name__ == "__main__":
    main()
