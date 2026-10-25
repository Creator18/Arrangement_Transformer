"""Speech-to-singing: a melody sung with real human phonemes, at EXACTLY the written notes.

The user's design.  A text-to-speech voice (Kokoro, Apache-2.0) speaks each syllable once; the
WORLD vocoder (pyworld) splits it into pitch, spectral envelope (the vocal tract) and aperiodicity
(breath).  Each written note is then rebuilt from those parts:

    consonant   the syllable's onset, placed just BEFORE the note so the vowel lands on time
    vowel       its steady middle, stretched to the note's length
    pitch       set to the written note, with only a slow random drift of a few cents (user-approved
                2026-10-16: +-3-5 cents, never vibrato); a melisma keeps the vowel and steps
                to each new note; an approved slide glides onto the next written note, landing
                on it at its written time
    ending      a short natural release; a rest between phrases is silence (a breath)

A voice never changes inside a phrase: one TTS voice per render (the voice clone and the timbre
variation that follow are also fixed per phrase).  The output is a mono stem plus the set of
notes reached by a slide, so the labels stay 1:1 with the notation.

    from eval.sing_tts import Singer
    y, slid, f0 = Singer(voice="af_heart").sing(notes, rng)    # notes (N, 4) [on, off, pitch, vel]
"""
from __future__ import annotations

import numpy as np

SR = 24000            # Kokoro's rate (and the renders')
FP = 5.0              # WORLD frame period, ms
CONS = ["l", "d", "t", "b", "p", "n", "m", "s", "h", "k", "g", "w", "y", "r", ""]
VOWELS = {"ah": "ah", "ee": "ee", "oo": "oo", "oh": "oh", "ay": "ay"}
SPELL = {("", "ah"): "ah", ("", "ee"): "ee", ("", "oo"): "ooh", ("", "oh"): "oh", ("", "ay"): "ay"}


def spelling(c, v):
    if (c, v) in SPELL:
        return SPELL[(c, v)]
    return c + {"ah": "ah", "ee": "ee", "oo": "oo", "oh": "oh", "ay": "ay"}[v]


class Singer:
    def __init__(self, voice="af_heart", lang="a", device=None):
        from kokoro import KPipeline
        self.voice = voice
        self.tts = KPipeline(lang_code=lang, device=device)
        self.cache = {}

    # ------------------------------------------------------------ one spoken syllable -> parts
    def syllable(self, c, v):
        key = (c, v)
        if key in self.cache:
            return self.cache[key]
        import pyworld as pw
        text = c + {"ah": "aaah", "ee": "eeee", "oo": "oooo", "oh": "ohhh", "ay": "aaay"}[v] + "."   # elongated: a longer held vowel
        y = np.concatenate([np.asarray(a, np.float64) for _, _, a in self.tts(text, voice=self.voice)])
        # trim leading / trailing silence
        e = np.convolve(y ** 2, np.ones(240) / 240, "same"); on = np.flatnonzero(e > e.max() * 1e-3)
        y = y[max(0, on[0] - 120):on[-1] + 120] if len(on) else y
        f0, t = pw.harvest(y, SR, f0_floor=60, f0_ceil=900, frame_period=FP)
        sp = pw.cheaptrick(y, f0, t, SR); ap = pw.d4c(y, f0, t, SR)
        voiced = np.flatnonzero(f0 > 0)
        if len(voiced) < 6:                                    # TTS gave nothing usable
            self.cache[key] = None
            return None
        v0, v1 = voiced[0], voiced[-1]
        # consonant = everything before voicing, plus the transition into the vowel
        # (sonorants l/m/n/w/y/r are voiced: their first ~50 ms are the consonant)
        trans = int(round((50 if c in ("l", "m", "n", "w", "y", "r") else 25) / FP)) if c else 0
        c_end = min(v0 + trans, v0 + (v1 - v0) // 3)
        # steady vowel: the middle half of the voiced part after the consonant
        a, b = c_end + (v1 - c_end) // 4, c_end + 3 * (v1 - c_end) // 4
        part = dict(c_sp=sp[:c_end], c_ap=ap[:c_end], c_voiced=f0[:c_end] > 0,
                    v_sp=sp[a:max(b, a + 2)], v_ap=ap[a:max(b, a + 2)],
                    tail_sp=sp[b:v1 + 1], tail_ap=ap[b:v1 + 1], nfft=sp.shape[1])
        self.cache[key] = part
        return part

    # ------------------------------------------------------------ the melody
    def sing(self, notes, rng, slide_p=0.0, phrase_gap=0.35, release_s=0.08, drift_cents=4.0,
             content_hz=None, art_both=False):
        """notes (N, 4) [onset, offset, pitch, velocity] of ONE melody line.  Returns (audio, slid, f0)
        where slid = indices of notes reached by a slide and f0 = the exact pitch track (Hz, 0 =
        unvoiced, one value per 5 ms) including the drift.  content_hz: synthesize the audio at
        this constant pitch instead (the CONTENT input for the voice clone, which is then given
        f0 directly), so no sung pitch passes through the vocoder.
        art_both: articulation both ways, chosen once per PHRASE: as written, clipped short
        (30-70% of each note), or held over the written gaps.  self.length_class then holds, per
        input note, 0 as written / 1 longer / 2 shorter (the side-output label)."""
        import pyworld as pw
        notes = np.asarray(notes, float); order = np.argsort(notes[:, 0], kind="stable"); notes = notes[order]
        lenc = np.zeros(len(notes), np.float32); mode = "natural"
        end = float(notes[:, 1].max()) + 1.0
        T = int(round(end * 1000 / FP)) + 1
        nfft = None
        f0 = np.zeros(T); SP = None; AP = None
        hz = lambda p: 440.0 * 2 ** ((p - 69) / 12)
        fr = lambda s: int(round(s * 1000 / FP))
        # syllables: a new one at the start of each phrase and on most notes; sometimes a melisma
        slid, syl, prev = set(), None, None
        for j, (s, e, p, v) in enumerate(notes):
            new_phrase = prev is None or s - notes[prev, 1] > phrase_gap
            if new_phrase and art_both:                       # one articulation per phrase
                mode = str(rng.choice(["natural", "short", "hold"], p=[.5, .25, .25]))
            nxt_on = notes[j + 1, 0] if j + 1 < len(notes) else s + 30
            e_w = e                                            # the written end (the label)
            if mode == "short":
                e = s + max(0.06, (e_w - s) * float(rng.uniform(.3, .7)))
            elif mode == "hold" and nxt_on - e_w < 1.2:
                e = max(e_w, nxt_on - 0.02)
            lenc[j] = 1.0 if e > e_w + 0.03 else 2.0 if e < e_w - 0.03 else 0.0
            melisma = (not new_phrase) and syl is not None and rng.random() < 0.18
            if not melisma:
                for _ in range(8):
                    syl = (str(rng.choice(CONS)), str(rng.choice(list(VOWELS))))
                    if self.syllable(*syl) is not None:
                        break
            part = self.syllable(*syl)
            if part is None:
                prev = j; continue
            if SP is None:
                nfft = part["nfft"]; SP = np.full((T, nfft), 1e-12); AP = np.ones((T, nfft))
            a, b = fr(s), max(fr(e), fr(s) + 2)
            g = 10 ** ((v - 100) / 40)                         # quieter notes, softer voice (amplitude)
            # consonant before the onset (not for a melisma: the vowel just continues)
            if not melisma:
                n = len(part["c_sp"]); c0 = max(0, a - n)
                SP[c0:a] = part["c_sp"][n - (a - c0):] * g ** 2; AP[c0:a] = part["c_ap"][n - (a - c0):]
                f0[c0:a] = np.where(part["c_voiced"][n - (a - c0):], hz(p), 0.0)
            # vowel held for the note: loop the steady frames smoothly
            m = len(part["v_sp"]); idx = (np.arange(b - a) * 0.5) % (2 * m - 2 if m > 1 else 1)
            idx = np.where(idx > m - 1, 2 * m - 2 - idx, idx)          # ping-pong, no jump
            lo = np.floor(idx).astype(int); w = (idx - lo)[:, None]; hi = np.minimum(lo + 1, m - 1)
            SP[a:b] = (part["v_sp"][lo] * (1 - w) + part["v_sp"][hi] * w) * g ** 2
            AP[a:b] = part["v_ap"][lo] * (1 - w) + part["v_ap"][hi] * w
            # a slow random drift around the written pitch (a few cents; never vibrato)
            n_ = b - a
            if drift_cents > 0 and n_ > 4:
                k_ = max(2, n_ // 40 + 2)                      # ~one control point per 200 ms
                d_ = np.interp(np.arange(n_), np.linspace(0, n_ - 1, k_), rng.normal(0, drift_cents / 2, k_))
                f0[a:b] = hz(p) * 2 ** (np.clip(d_, -drift_cents - 1, drift_cents + 1) / 1200)
            else:
                f0[a:b] = hz(p)
            # slide onto THIS note from the previous one (legato, close), landing on time
            if prev is not None:
                k = max(2, fr(float(rng.uniform(.04, .15))))
                k0 = max(fr(notes[prev, 0]) + 1, a - k)
            if (prev is not None and not new_phrase and slide_p > 0 and abs(p - notes[prev, 2]) <= 4
                    and s - notes[prev, 1] < 0.08 and k0 < a and rng.random() < slide_p):  # (no room: no slide)
                f0[k0:a] = hz(notes[prev, 2] + (p - notes[prev, 2]) * np.linspace(0, 1, a - k0, endpoint=False))
                slid.add(j)
            # natural release if a gap follows
            nxt = notes[j + 1, 0] if j + 1 < len(notes) else end
            if nxt - e > 0.03:
                r = min(fr(release_s), fr(nxt) - b)
                if r > 1:
                    fade = np.linspace(1, 0, r)[:, None] ** 2
                    SP[b:b + r] = SP[b - 1] * fade + 1e-12; AP[b:b + r] = AP[b - 1]; f0[b:b + r] = hz(p)
            prev = j
        self.length_class = np.empty_like(lenc); self.length_class[order] = lenc
        slid = {int(order[q]) for q in slid}                  # back to the caller's note order
        if SP is None:
            return np.zeros(int(end * SR)), slid, f0
        src = np.where(f0 > 0, content_hz, 0.0) if content_hz else f0
        y = pw.synthesize(src, np.ascontiguousarray(SP), np.ascontiguousarray(AP), SR, FP)
        return y / (np.abs(y).max() + 1e-9) * 0.9, slid, f0
