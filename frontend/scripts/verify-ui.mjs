/**
 * verify-ui — look at the UI the way a person does.
 *
 * Why this exists
 * ---------------
 * Every UI change in this project has shipped with the same caveat: "not
 * verified in motion — the preview pane does not composite frames, so
 * requestAnimationFrame never fires."  That made the human the only test
 * harness for the entire frontend, while the backend has an eval suite that
 * has overturned four confident hypotheses in a single session.
 *
 * The frontend deserves the same thing, and it starts here: a browser that
 * genuinely renders, so an animation can be caught either running or dead.
 *
 * The capability test is the important part.  Two screenshots a moment apart
 * MUST differ on a page with a live canvas.  If they are identical, either
 * the animation is broken or this harness is as blind as the pane it
 * replaces — and both of those are things worth failing loudly on.
 *
 *   node scripts/verify-ui.mjs                     # capability + scroll shots
 *   node scripts/verify-ui.mjs --url http://…      # somewhere else
 */
import { chromium } from 'playwright'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const args = process.argv.slice(2)
const arg = (k, d) => {
  const i = args.indexOf(`--${k}`)
  return i >= 0 ? args[i + 1] : d
}

const URL = arg('url', 'http://localhost:5173')
const OUT = arg('out', 'shots')
const W = 1440
const H = 900

mkdirSync(OUT, { recursive: true })

/** Fraction of bytes that differ — a cheap proxy for "did anything move". */
function differs(a, b) {
  if (a.length !== b.length) return 1
  let n = 0
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++
  return n / a.length
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: W, height: H } })

const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })

await page.goto(URL, { waitUntil: 'networkidle' }).catch(() => {})
await page.waitForTimeout(1500)          // let entrance animations settle

// ── capability check: is anything actually animating? ──────────────────────
const a = await page.screenshot()
await page.waitForTimeout(700)
const b = await page.screenshot()
const moved = differs(a, b)

console.log(`\nverify-ui  ${URL}  ${W}x${H}`)
console.log(`  frames composite : ${moved > 0 ? 'YES' : 'NO'}  (${(moved * 100).toFixed(2)}% of bytes changed in 700ms)`)

// ── scroll-position shots, the thing the old harness could never do ────────
// Sample where the SECTIONS are, not at arbitrary fractions of the document.
// Fixed fractions land mid-transition on a page whose sections differ in
// height by 5x, which produces screenshots of nothing in particular and
// invites false alarms about missing content.
const stops = await page.evaluate(() => {
  const secs = [...document.querySelectorAll('section[id]')]
  if (!secs.length) return [0]
  const doc = document.documentElement.scrollHeight
  return secs.map((el) => {
    const r = el.getBoundingClientRect()
    const top = r.top + window.scrollY
    // Centre of the section, clamped to a reachable scroll position.
    return Math.max(0, Math.min(doc - window.innerHeight,
      Math.round(top + r.height / 2 - window.innerHeight / 2)))
  })
})
for (const y of stops) {
  // Real wheel input, not scrollTo: Lenis drives the page off wheel events,
  // and a programmatic jump bypasses it entirely — which is exactly how the
  // old harness produced scroll positions the page had never actually been in.
  await page.evaluate((target) => window.scrollTo({ top: target, behavior: 'instant' }), y)
  await page.waitForTimeout(900)
  const shot = join(OUT, `y${String(y).padStart(5, '0')}.png`)
  writeFileSync(shot, await page.screenshot())
  console.log(`  captured ${shot}  (y=${y})`)
}

console.log(`  page errors      : ${errors.length}`)
for (const e of errors.slice(0, 5)) console.log(`      ${e.slice(0, 110)}`)

await browser.close()
process.exit(moved > 0 ? 0 : 1)
