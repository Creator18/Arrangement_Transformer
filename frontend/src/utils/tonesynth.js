/**
 * tonesynth.js — a small Web Audio voice for the interactive keyboard.
 *
 * Purpose is proof, not beauty: clicking a key should sound the pitch that
 * key claims to be, so anyone can check by ear that the roll and the
 * keyboard agree about what note is what.  Pitch accuracy is the whole
 * requirement, so the frequency comes straight from the MIDI number with no
 * sample mapping or tuning table in between to be wrong.
 *
 * Deliberately synthesised rather than loading Salamander samples: the
 * sampler is ~1GB server-side, and a click-to-hear check must be instant and
 * offline.  A few detuned partials with a percussive envelope is enough to
 * read as "piano-ish" and, more importantly, is unambiguous in pitch.
 */

let ctx = null

function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext
    if (!AC) return null
    ctx = new AC()
  }
  // Browsers start the context suspended until a user gesture; every call
  // here is inside a click handler, so this is the right place to resume.
  if (ctx.state === 'suspended') ctx.resume()
  return ctx
}

export function midiToFreq(midi) {
  return 440 * Math.pow(2, (midi - 69) / 12)
}

/**
 * Strike a note. Returns a stop() for held/pointer-up behaviour, though the
 * envelope decays on its own like a real struck string.
 */
export function playNote(midi, { velocity = 0.8, duration = 1.6 } = {}) {
  const ac = audio()
  if (!ac) return () => {}

  const f0 = midiToFreq(midi)
  const now = ac.currentTime
  const out = ac.createGain()
  out.connect(ac.destination)

  // Lower notes carry more energy in the upper partials; higher notes are
  // closer to pure. Without this the top octave sounds harsh and the bottom
  // sounds hollow.
  const bright = Math.max(0.15, 1 - (midi - 21) / 100)
  const partials = [
    [1, 1.0],
    [2, 0.42 * bright],
    [3, 0.16 * bright],
    [4, 0.09 * bright],
  ]

  const peak = Math.min(0.9, velocity) * 0.22
  out.gain.setValueAtTime(0.0001, now)
  out.gain.exponentialRampToValueAtTime(peak, now + 0.006)   // hammer strike
  out.gain.exponentialRampToValueAtTime(0.0001, now + duration)

  const oscs = partials.map(([mult, amp]) => {
    const o = ac.createOscillator()
    const g = ac.createGain()
    o.type = 'triangle'
    o.frequency.value = f0 * mult
    // Real strings are slightly inharmonic; a touch of detune stops the
    // partials phase-locking into an organ-like buzz.
    o.detune.value = (mult - 1) * 1.5
    g.gain.value = amp
    o.connect(g); g.connect(out)
    o.start(now)
    o.stop(now + duration + 0.05)
    return o
  })

  return () => {
    const t = ac.currentTime
    try {
      out.gain.cancelScheduledValues(t)
      out.gain.setValueAtTime(Math.max(0.0001, out.gain.value), t)
      out.gain.exponentialRampToValueAtTime(0.0001, t + 0.18)
      oscs.forEach((o) => o.stop(t + 0.2))
    } catch { /* already stopped */ }
  }
}
