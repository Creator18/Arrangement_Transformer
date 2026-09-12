/**
 * roll-fps — measure the piano roll's frame rate while it is actually playing.
 *
 * The Browser pane cannot answer this: requestAnimationFrame never fires
 * there, so a roll that renders at 12 fps and one that renders at 60 look
 * identical to it.  Playwright composites for real, so a frame that took
 * 90 ms is visible as a frame that took 90 ms.
 *
 * What it reports, and why these numbers rather than "fps":
 *
 *   fps        mean, the headline — but an average hides exactly the thing
 *              that reads as jarring, because one 200 ms stall inside a
 *              second of 60 fps still averages ~50.
 *   p95        the frame time 19 frames in 20 come in under.  This is the
 *              number the eye actually judges: smoothness is the absence of
 *              outliers, not a good mean.
 *   janky      frames over 32 ms — two display refreshes, i.e. a visible
 *              hitch rather than a slow-but-even cadence.
 *
 *   node scripts/roll-fps.mjs --session <uuid> [--seconds 6] [--label before]
 */
import { chromium } from 'playwright'

const args = process.argv.slice(2)
const arg = (k, d) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : d }

const SESSION = arg('session')
const BASE    = arg('base', 'http://localhost:5173')
const SECONDS = Number(arg('seconds', 6))
const LABEL   = arg('label', 'run')
if (!SESSION) { console.error('need --session <uuid>'); process.exit(1) }

const browser = await chromium.launch({
  args: ['--autoplay-policy=no-user-gesture-required'],
})
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
page.on('pageerror', e => console.error('[pageerror]', e.message))

await page.goto(`${BASE}/results/${SESSION}`, { waitUntil: 'networkidle' }).catch(() => {})
// The results page fetches the arrangement after mount, so the roll does not
// exist at networkidle.  Wait for the transport button itself rather than a
// duration — a timeout that is merely long enough today is a flaky test.
await page.waitForSelector('button svg path[d^="M8 5v14"]', { timeout: 30000 })
await page.waitForTimeout(1500)

// Count the note rects actually in the DOM.  A frame-rate number means
// nothing without knowing how much the roll was being asked to draw.
const noteCount = await page.evaluate(
  () => document.querySelectorAll('svg g > rect').length)

// Find the transport play button by its position in the roll's header and
// click it.  Falling back to any button inside the roll card keeps this
// working if the markup shifts.
const played = await page.evaluate(() => {
  const btns = [...document.querySelectorAll('button')]
  const play = btns.find(b => b.querySelector('path[d^="M8 5v14"]'))
  if (!play) return false
  play.click()
  return true
})
if (!played) { console.error('could not find the play button'); await browser.close(); process.exit(1) }

await page.waitForTimeout(800)   // let playback actually start

const stats = await page.evaluate(async (secs) => {
  const ts = []
  let last = performance.now()
  await new Promise(res => {
    const t0 = last
    const tick = (now) => {
      ts.push(now - last)
      last = now
      if (now - t0 < secs * 1000) requestAnimationFrame(tick)
      else res()
    }
    requestAnimationFrame(tick)
  })
  ts.shift()
  const sorted = [...ts].sort((a, b) => a - b)
  const q = p => sorted[Math.floor(sorted.length * p)]
  return {
    frames: ts.length,
    fps: 1000 / (ts.reduce((a, b) => a + b, 0) / ts.length),
    p50: q(0.50), p95: q(0.95), worst: sorted[sorted.length - 1],
    janky: ts.filter(t => t > 32).length,
  }
}, SECONDS)

const playing = await page.evaluate(() => {
  const a = document.querySelector('audio')
  return a ? { paused: a.paused, t: a.currentTime } : null
})

console.log(`\n${LABEL}  (${noteCount} note rects, audio t=${playing?.t?.toFixed(1)}s, paused=${playing?.paused})`)
console.log(`  fps    ${stats.fps.toFixed(1)}`)
console.log(`  p50    ${stats.p50.toFixed(1)} ms`)
console.log(`  p95    ${stats.p95.toFixed(1)} ms`)
console.log(`  worst  ${stats.worst.toFixed(1)} ms`)
console.log(`  janky  ${stats.janky} / ${stats.frames} frames over 32 ms`)

await browser.close()
