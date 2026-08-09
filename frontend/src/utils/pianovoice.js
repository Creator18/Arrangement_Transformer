/**
 * pianovoice.js — play a single key using the REAL instrument.
 *
 * Replaces the Web Audio synth that shipped first.  That version proved the
 * pitch was right but not the timbre, and the point of a playable keyboard is
 * to check our output by ear — so it has to be the same piano the
 * arrangements are rendered with (Salamander through sfizz), not a
 * lookalike.  A note that sounds different from the track is a worse test
 * than no test, because a real timbre problem would hide inside the
 * difference.
 *
 * The samples come from /export/note/{midi}, rendered server-side and cached
 * to disk there; here they are decoded once into an AudioBuffer and reused,
 * so a key is only ever fetched the first time it is pressed.
 */

const BASE = import.meta.env.VITE_API_URL || ''

let ctx = null
const buffers = new Map()      // midi -> AudioBuffer
const pending = new Map()      // midi -> Promise<AudioBuffer>

function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext
    if (!AC) return null
    ctx = new AC()
  }
  // Contexts start suspended until a user gesture; every caller here is
  // inside a click handler, which is exactly where resuming is allowed.
  if (ctx.state === 'suspended') ctx.resume()
  return ctx
}

async function load(midi, velocity) {
  if (buffers.has(midi)) return buffers.get(midi)
  if (pending.has(midi)) return pending.get(midi)

  const ac = audio()
  if (!ac) return null

  const p = fetch(`${BASE}/export/note/${midi}?velocity=${velocity}`)
    .then((r) => {
      if (!r.ok) throw new Error(`note ${midi}: HTTP ${r.status}`)
      return r.arrayBuffer()
    })
    .then((buf) => ac.decodeAudioData(buf))
    .then((decoded) => {
      buffers.set(midi, decoded)
      pending.delete(midi)
      return decoded
    })
    .catch((e) => {
      pending.delete(midi)
      throw e
    })

  pending.set(midi, p)
  return p
}

/**
 * Strike a key. Fire-and-forget; the first press of a given pitch waits on
 * the fetch, every later one is instant.
 */
export function playNote(midi, { velocity = 80 } = {}) {
  const ac = audio()
  if (!ac) return

  const start = (buf) => {
    if (!buf) return
    const src = ac.createBufferSource()
    const gain = ac.createGain()
    src.buffer = buf
    gain.gain.value = 0.9
    src.connect(gain)
    gain.connect(ac.destination)
    src.start()
  }

  const cached = buffers.get(midi)
  if (cached) start(cached)
  else load(midi, velocity).then(start).catch(() => { /* keyboard stays silent */ })
}

/**
 * Warm the cache for a range of keys so the first click is not the one that
 * pays the latency. Called when the roll mounts, at low priority.
 */
export function prefetchRange(lo, hi, velocity = 80) {
  for (let m = Math.max(21, lo); m <= Math.min(108, hi); m++) {
    load(m, velocity).catch(() => {})
  }
}
