// Screenshot the piano-roll keyboard mid-playback, to check key shape, glow and shadows.
//   node scripts/shot-keys.mjs <session-id> [out.png]
import { chromium } from 'playwright'

const sid = process.argv[2]
const out = process.argv[3] || 'shots/keys.png'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto(`http://localhost:5173/results/${sid}`, { waitUntil: 'networkidle' })
await page.waitForTimeout(2500)
// the round orange play button beside the track name
await page.mouse.click(367, 110)
await page.waitForTimeout(9000)
const svg = page.locator('svg').filter({ hasText: 'KEY' }).first()
await svg.screenshot({ path: out })
await page.screenshot({ path: out.replace('.png', '_full.png') })
await browser.close()
console.log('wrote', out)
