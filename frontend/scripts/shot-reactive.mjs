// Music-reactive backdrop check: screenshots paused vs playing, and how much the canvas moves.
//   node scripts/shot-reactive.mjs <session-id>
import { chromium } from 'playwright'

const sid = process.argv[2]
const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] })
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto(`http://localhost:5173/results/${sid}`, { waitUntil: 'networkidle' })
await page.waitForTimeout(3000)
const energy = () => page.evaluate(() => {
  const c = document.querySelector('canvas'); const g = c.getContext('2d')
  const d = g.getImageData(0, 0, c.width, c.height).data
  let s = 0; for (let i = 0; i < d.length; i += 16) s += d[i] + d[i + 1] + d[i + 2]
  return s / (d.length / 16)
})
await page.screenshot({ path: 'shots/reactive_paused.png' })
const paused = await energy()
await page.mouse.click(367, 110)
await page.waitForTimeout(12000)
await page.screenshot({ path: 'shots/reactive_playing.png' })
const playing = await energy()
console.log(`canvas brightness: paused ${paused.toFixed(2)}  playing ${playing.toFixed(2)}`)
await browser.close()
