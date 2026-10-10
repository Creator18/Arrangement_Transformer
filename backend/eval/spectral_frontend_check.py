"""Verify the spectral front end does what its design claims, before anything trains on it.

Each test checks one property the plan depends on:

  1 ANCHOR      a pure tone lands on the bin its pitch says it should
                (absolute frequency survives the log axis)
  2 HARMONICS   a harmonic tone's overtones align at the fundamental's bin
                in every HCQT channel — and whether the higher channels
                resolve bass pitch the capped fundamental channel cannot
  3 TUNING      a tone 30 cents sharp reads ~30 cents sharp
  4 BASS BLUR   onset timing error by register, CQT flux vs the onset stream
  5 SPEED       cost per training batch, against the transcriber's ~0.6 s/step
  6 LABELS      on a rendered MAESTRO excerpt: does energy at the labelled
                pitch bins separate active from inactive frames, and does that
                separation peak at ZERO lag?  This is the check that features
                and notation line up in time — the thing a misaligned training
                set gets silently wrong.

    venv/Scripts/python.exe eval/spectral_frontend_check.py
"""
from __future__ import annotations

import pathlib
import sys
import time
import warnings

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.ml.spectral import frontend as S          # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAESTRO = ROOT.parent / "data" / "maestro-v3.0.0"


def hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)


def tone(f, dur=2.0, harmonics=1, decay=0.0, onset=0.0):
    t = np.arange(int(dur * S.SR)) / S.SR
    env = np.where(t >= onset, np.exp(-decay * np.clip(t - onset, 0, None)), 0.0)
    y = sum((0.7 ** (k - 1)) * np.sin(2 * np.pi * f * k * t) for k in range(1, harmonics + 1))
    return (y * env).astype(np.float32)


def run(model, y):
    with torch.no_grad():
        return model(torch.from_numpy(y)[None].to(DEV))


def test_anchor(hc):
    print("1 ANCHOR — pure tones, argmax bin of the h=1 channel")
    errs = []
    for p in (33, 45, 57, 69, 81, 93):
        m = run(hc, tone(hz(p)))[0, S.HARMONICS.index(1)].mean(-1).cpu().numpy()
        got, want = int(m.argmax()), S.midi_to_bin(p)
        errs.append(got - want)
        print(f"   MIDI {p:>3} ({hz(p):7.1f} Hz)  expected bin {want:>3}  got {got:>3}  "
              f"error {got - want:+d} bins")
    ok = all(abs(e) <= 1 for e in errs)
    print(f"   -> {'PASS' if ok else 'FAIL'}: within one bin (33 cents) everywhere\n")
    return ok


def peak_width(v, i):
    """Width at half maximum around index i, in bins."""
    half = v[i] / 2
    lo = i
    while lo > 0 and v[lo] > half:
        lo -= 1
    hi = i
    while hi < len(v) - 1 and v[hi] > half:
        hi += 1
    return hi - lo


def test_harmonics(hc):
    print("2 HARMONICS — 5-partial tone; per channel: peak position and sharpness")
    for p, name in ((33, "A1 bass"), (60, "C4 middle")):
        want = S.midi_to_bin(p)
        H = run(hc, tone(hz(p), harmonics=5))[0].mean(-1).cpu().numpy()
        print(f"   {name} (expected bin {want})")
        for ci, h in enumerate(S.HARMONICS):
            if h < 1:
                continue
            v = H[ci]
            # look near the fundamental's bin: does channel h put energy THERE
            w = slice(max(0, want - 6), want + 7)
            i = int(np.argmax(v[w])) + w.start
            print(f"     h={h}: peak at bin {i:>3} ({i - want:+d})  "
                  f"half-max width {peak_width(v, i):>2} bins")
    print("   -> aligned channels peak at the same bin; NARROWER width = sharper pitch.\n"
          "      If bass h>=3 is much narrower than bass h=1, the harmonic channels\n"
          "      are recovering the pitch resolution the capped window gives up.\n")


def test_tuning(hc):
    print("3 TUNING — A4 at +30 cents, parabolic peak interpolation")
    v = run(hc, tone(hz(69) * 2 ** (30 / 1200)))[0, S.HARMONICS.index(1)].mean(-1).cpu().numpy()
    i = int(v.argmax())
    a, b, c = np.log(v[i - 1] + 1e-9), np.log(v[i] + 1e-9), np.log(v[i + 1] + 1e-9)
    frac = 0.5 * (a - c) / (a - 2 * b + c)
    cents = (i + frac - S.midi_to_bin(69)) * (1200 / S.BINS_PER_OCT)
    print(f"   measured {cents:+.1f} cents (true +30.0)")
    ok = abs(cents - 30) < 8
    print(f"   -> {'PASS' if ok else 'FAIL'}: within 8 cents\n")
    return ok


def test_bass_blur(fe):
    print("4 BASS BLUR — plucked tone starting at exactly 1.000 s")
    onset_true = 1.0
    print(f"   {'note':<10}{'CQT flux err':>14}{'onset stream err':>18}")
    for p, name in ((33, "A1"), (45, "A2"), (57, "A3"), (81, "A5")):
        y = tone(hz(p), dur=2.5, harmonics=4, decay=3.0, onset=onset_true)
        x = run(fe, y)[0].cpu().numpy()
        b = S.midi_to_bin(p)
        h1 = x[S.HARMONICS.index(1), b - 1:b + 2].mean(0)
        cqt_flux = np.diff(h1, prepend=h1[0])
        on = x[-1, b - 1:b + 2].mean(0)
        f0 = int(0.5 * S.FPS)          # search after 0.5 s
        t_cqt = (np.argmax(cqt_flux[f0:]) + f0) / S.FPS
        t_on = (np.argmax(on[f0:]) + f0) / S.FPS
        print(f"   {name:<10}{(t_cqt - onset_true) * 1000:>11.0f} ms"
              f"{(t_on - onset_true) * 1000:>15.0f} ms")
    print("   -> the onset stream should stay near 0 ms in the bass where CQT drifts.\n")


def test_equivalence(hc):
    print("5a EQUIVALENCE — FFT path vs direct-convolution reference, random audio")
    wav = torch.randn(2, 3 * S.SR, device=DEV) * 0.1
    with torch.no_grad():
        a = hc(wav, method="conv")
        b = hc(wav, method="fft")
    rel = float((a - b).abs().max() / a.abs().max())
    print(f"   max abs difference / max value = {rel:.2e}")
    ok = rel < 1e-3
    print(f"   -> {'PASS' if ok else 'FAIL'}: same transform, different arithmetic\n")
    return ok


def test_speed(fe):
    print("5 SPEED — batch 8 x 8 s, forward only")
    wav = torch.randn(8, 8 * S.SR, device=DEV) * 0.1
    with torch.no_grad():
        fe(wav)                        # warm-up
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            x = fe(wav)
            tok, coords = S.patchify(x)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    print(f"   {dt * 1000:.0f} ms per batch   features {tuple(x.shape)}   "
          f"tokens {tuple(tok.shape)}  ({tok.shape[1] / 8:.0f} tokens/s)")
    print(f"   -> {'PASS' if dt < 0.15 else 'SLOW'}: under 25% of the ~0.6 s training step\n")


def test_labels(fe):
    print("6 LABELS — rendered MAESTRO excerpt, feature vs notation alignment")
    import pretty_midi
    from app.services.sampler import render_notes
    import csv
    rows = list(csv.DictReader(open(MAESTRO / "maestro-v3.0.0.csv", encoding="utf-8")))
    row = next(r for r in rows if r["split"] == "test")
    pm = pretty_midi.PrettyMIDI(str(MAESTRO / row["midi_filename"]))
    dur = 30.0
    notes = [{"pitch": n.pitch, "onset": n.start, "offset": min(n.end, dur),
              "velocity": n.velocity}
             for i in pm.instruments for n in i.notes if n.start < dur]
    y = render_notes(notes, sr=S.SR).astype(np.float32)[: int(dur * S.SR)]
    print(f"   {row['canonical_composer']} — {row['canonical_title']}, first {dur:.0f} s, "
          f"{len(notes)} notes")
    x = run(fe, y)[0].cpu().numpy()
    T = x.shape[-1]
    # Frames are CENTRED on t*HOP, so a time maps to round(t*fps), not floor —
    # data.track_rolls already does this.  The sampler also starts notes late:
    # measured median 11.7 ms (5-16 ms by pitch), so rendered-audio labels are
    # shifted by that much.  Real recordings will not have this offset.
    LAT = 0.0117
    def fr(t):
        return int(round((t + LAT) * S.FPS))
    roll = np.zeros((S.N_KEYS, T), bool)
    for n in notes:
        a, b = fr(n["onset"]), fr(n["offset"])
        roll[n["pitch"] - 21, a:max(b, a + 1)] = True
    centre = np.array([S.midi_to_bin(p) for p in range(21, 109)])

    def auc(feat, lag):
        f = np.roll(feat, lag, axis=1)
        pos, neg = f[roll], f[~roll]
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))
        return (r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg)

    for label, feat in (("h=1 only", x[S.HARMONICS.index(1)][centre]),
                        ("all harmonics", x[:len(S.HARMONICS)][:, centre].mean(0))):
        lags = range(-6, 7)
        a = [auc(feat, L) for L in lags]
        best = list(lags)[int(np.argmax(a))]
        print(f"   {label:<14} AUC at lag 0 = {a[6]:.3f}   best lag {best:+d} frames "
              f"({best * 16:+d} ms)")
    # Sustained frames are biased by the sampler's room reverb, which keeps a
    # note ringing past its labelled offset.  Onsets have no tail, so they
    # measure TIME alignment without that confound.
    onroll = np.zeros((S.N_KEYS, T), bool)
    for n in notes:
        onroll[n["pitch"] - 21, min(fr(n["onset"]), T - 1)] = True
    ofeat = x[-1][centre]
    def onset_auc(lag):
        f = np.roll(ofeat, lag, axis=1)
        pos, neg = f[onroll], f[~onroll]
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))
        return (r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg)
    lags = list(range(-6, 7))
    oa = [onset_auc(L) for L in lags]
    ob = lags[int(np.argmax(oa))]
    print(f"   {'onset stream':<14} AUC at lag 0 = {oa[6]:.3f}   best lag {ob:+d} frames "
          f"({ob * 16:+d} ms)   <- onsets only, no reverb tail")
    print("   -> AUC well above 0.5 means the pitch axis matches the notation;\n"
          "      best ONSET lag 0 means the time axis does too.  A sustained-frame\n"
          "      lag that onsets don't share is reverb, not misalignment.\n")


def main():
    print(f"device {DEV}   SR {S.SR}  hop {S.HOP} ({1000 / S.FPS:.0f} ms)  "
          f"{S.N_BINS} bins  harmonics {S.HARMONICS}\n")
    fe = S.SpectralFrontend().to(DEV).eval()
    test_anchor(fe.hcqt)
    test_harmonics(fe.hcqt)
    test_tuning(fe.hcqt)
    test_bass_blur(fe)
    test_equivalence(fe.hcqt)
    test_speed(fe)
    test_labels(fe)


if __name__ == "__main__":
    main()
