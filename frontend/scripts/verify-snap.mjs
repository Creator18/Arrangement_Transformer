/**
 * Does one wheel gesture advance exactly one act?
 *
 * Scroll snapping was shipped on the strength of "it builds".  This checks
 * the actual claim: a single wheel gesture should land on the next act, and
 * the pipeline — five viewports of scrubbed narrative — should NOT snap past,
 * it should scrub through.
 */
import { chromium } from 'playwright'

const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1440, height: 900 } })
await p.goto('http://localhost:5173/', { waitUntil: 'networkidle' }).catch(() => {})
await p.waitForTimeout(1800)

const tops = await p.evaluate(() =>
  ['overture', 'console', 'pipeline', 'coda'].map((id) => {
    const e = document.getElementById(id)
    return e ? Math.round(e.getBoundingClientRect().top + window.scrollY) : null
  }))
console.log('act tops:', tops.join(', '))

const y = () => p.evaluate(() => Math.round(window.scrollY))
const nearest = (v) => {
  let bi = 0, bd = Infinity
  tops.forEach((t, i) => { const d = Math.abs(t - v); if (d < bd) { bd = d; bi = i } })
  return { act: bi, off: bd }
}

console.log('\n gesture | scrollY | nearest act | offset')
console.log('---------|---------|-------------|-------')
for (let g = 1; g <= 5; g++) {
  await p.mouse.move(720, 450)
  await p.mouse.wheel(0, 320)          // one deliberate gesture
  await p.waitForTimeout(1400)         // snap animation + lock
  const v = await y()
  const n = nearest(v)
  console.log(`    ${g}    | ${String(v).padStart(7)} |      ${n.act}      | ${n.off}px`)
}

// Inside the pipeline, a gesture must MOVE without jumping a whole act.
await p.evaluate((t) => window.scrollTo({ top: t + 1200, behavior: 'instant' }), tops[2])
await p.waitForTimeout(700)
const before = await y()
await p.mouse.wheel(0, 320)
await p.waitForTimeout(1000)
const after = await y()
console.log(`\npipeline scrub: ${before} -> ${after}  (moved ${after - before}px)`)
console.log(after - before > 0 && after - before < 900
  ? '  PASS  scrubs inside the section instead of snapping past it'
  : '  FAIL  either stuck, or jumped a whole act')

await b.close()

// ── Result, 2026-08-09 ───────────────────────────────────────────────────────
//
//   act tops: 0, 900, 1800, 6300
//
//   gesture | scrollY | nearest act | offset
//       1   |     900 |      1      | 0px
//       2   |    1800 |      2      | 0px
//       3   |    2120 |      2      | 320px   <- inside the pipeline
//       4   |    2440 |      2      | 640px
//       5   |    2760 |      2      | 960px
//
//   pipeline scrub: 3000 -> 3320 (moved 320px)  PASS
//
// Both halves hold: gestures land EXACTLY on act boundaries (0px offset, not
// approximately), and once inside the pipeline the same gesture scrubs 320px
// instead of jumping a whole act.  The data-free-scroll exemption is doing
// what it was written for.
