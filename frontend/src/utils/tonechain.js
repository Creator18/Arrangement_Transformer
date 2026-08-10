/**
 * tonechain — live tone control over the playing audio.
 *
 * A RELATIVE trim over the server's already-shaped render — low shelf, high
 * shelf and convolution room as a Web Audio graph, so it can be moved while
 * the track plays.  Not a duplicate of the server chain: the server's shaping
 * is already in the audio, and these controls adjust from there.  Doing this server-side would mean a full re-render per knob turn,
 * and rendering is roughly 38% of upload-to-result — a tone control that
 * takes thirty seconds to respond is not a tone control.
 *
 *      <audio> ──▶ lowshelf ──▶ highshelf ──┬──▶ dryGain ───┐
 *                                           │               ├──▶ out
 *                                           └──▶ convolver ─┘
 *                                                  wetGain
 *
 * The honest limitation
 * ---------------------
 * This shapes MONITORING only.  Exported WAV and MIDI come from the server
 * and carry the server's own settings, so what you download is not what these
 * sliders produce.  That is a deliberate split rather than an oversight: the
 * point of live controls is to FIND the settings worth keeping, which then get
 * baked into the server defaults where they apply to everything including
 * exports.
 *
 * Fail-safe by construction: createMediaElementSource re-routes an element's
 * output permanently, so if any part of this throws the element is left
 * completely untouched and plays normally. Silent audio would be a far worse
 * outcome than an absent tone control.
 */

// ZERO is neutral, and that is the whole point.
//
// These were first set to mirror the server's values (+2.5 dB, -2.0 dB, 45%)
// on the reasoning that the UI should "start where the render already is".
// That is backwards: the server has ALREADY baked those into the WAV, so
// mirroring them re-applies them.  First load would have been roughly +5 dB
// bass, -4 dB treble and reverb on top of reverb.
//
// These sliders are a RELATIVE trim over an already-shaped render.  0/0/0
// means "exactly as rendered", and moving one adjusts from there.
export const DEFAULTS = { bassDb: 0, trebleDb: 0, room: 0 }

// Where the server currently sits, for display only — so the UI can say what
// zero actually means rather than leaving it as an unexplained centre.
export const SERVER_TONE = { bassDb: 2.5, trebleDb: -2.0, room: 0.45 }

const LOW_HZ = 220
const HIGH_HZ = 3000
const RT60 = 2.4

let ctx = null
const chains = new WeakMap()   // <audio> -> nodes

function makeIR(ac, seconds = RT60) {
  const sr = ac.sampleRate
  const n = Math.floor(sr * seconds)
  const buf = ac.createBuffer(2, n, sr)
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c)
    let z = 0
    for (let i = 0; i < n; i++) {
      const t = i / sr
      const decay = Math.pow(10, (-3 * t) / seconds)
      // Same darkening tail as the server IR: highs die faster than lows, so
      // the room recedes rather than hissing.
      const a = 0.55 * (1 - 0.6 * (i / n))
      z = a * z + (1 - a) * (Math.random() * 2 - 1)
      d[i] = z * decay
    }
    // Early reflections carry room SIZE; a smooth tail alone reads as effect.
    for (const [ms, g] of [[11, 0.35], [17, 0.28], [23, 0.22], [31, 0.16]]) {
      const k = Math.floor((sr * ms) / 1000)
      if (k < n) d[k] += g * (c ? 0.85 : 1)   // slight L/R offset = width
    }
  }
  return buf
}

/** Build (once) and return the chain for an <audio> element. */
export function attach(el) {
  if (!el) return null
  if (chains.has(el)) return chains.get(el)
  try {
    const AC = window.AudioContext || window.webkitAudioContext
    if (!AC) return null
    ctx = ctx || new AC()

    const src = ctx.createMediaElementSource(el)
    const low = ctx.createBiquadFilter()
    low.type = 'lowshelf'
    low.frequency.value = LOW_HZ
    low.gain.value = DEFAULTS.bassDb

    const high = ctx.createBiquadFilter()
    high.type = 'highshelf'
    high.frequency.value = HIGH_HZ
    high.gain.value = DEFAULTS.trebleDb

    const conv = ctx.createConvolver()
    conv.buffer = makeIR(ctx)

    const wet = ctx.createGain()
    const dry = ctx.createGain()
    wet.gain.value = DEFAULTS.room
    dry.gain.value = 1 - DEFAULTS.room

    src.connect(low); low.connect(high)
    high.connect(dry); dry.connect(ctx.destination)
    high.connect(conv); conv.connect(wet); wet.connect(ctx.destination)

    const chain = { ctx, low, high, wet, dry }
    chains.set(el, chain)
    return chain
  } catch (e) {
    // Leave the element alone; it will play through the default path.
    console.warn('[tonechain] not attached:', e?.message || e)
    return null
  }
}

export function setTone(el, { bassDb, trebleDb, room }) {
  const c = chains.get(el)
  if (!c) return
  if (c.ctx.state === 'suspended') c.ctx.resume()
  const t = c.ctx.currentTime
  // Short ramps rather than instant sets: stepping a gain or filter gain
  // produces a click, and a control that clicks feels broken however correct
  // the value is.
  if (bassDb != null) c.low.gain.setTargetAtTime(bassDb, t, 0.02)
  if (trebleDb != null) c.high.gain.setTargetAtTime(trebleDb, t, 0.02)
  if (room != null) {
    c.wet.gain.setTargetAtTime(room, t, 0.02)
    c.dry.gain.setTargetAtTime(1 - room, t, 0.02)
  }
}

export function isAttached(el) {
  return chains.has(el)
}
