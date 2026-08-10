import { useEffect, useRef } from 'react'

/**
 * AuroraRibbons — the backdrop, rebuilt on simplex noise.
 *
 * Replaces StringSpace, and the swap is a simplification rather than a
 * restyle: that component pulled three.js for a 734KB chunk to draw lines
 * receding into depth.  Ribbons driven by 2D noise read MORE like sound, not
 * less — a waveform is a line displaced over time, which is exactly what this
 * is — and they cost nothing but a canvas.
 *
 * Adapted from the Aurora hero concept, with the noise field made meaningful
 * rather than decorative:
 *
 *   - the ribbon count and amplitude follow SCROLL, so the field opens up as
 *     you move through the page instead of idling at one intensity;
 *   - the cursor bends the ribbons locally with a distance falloff, the same
 *     mechanic as the bubble cluster, so the whole UI responds to the pointer
 *     the same way;
 *   - each ribbon is a horizontal gradient that fades at both edges, which
 *     keeps the centre of the screen readable without a vignette on top.
 *
 * The simplex implementation is inlined deliberately: it is ~50 lines, and a
 * dependency for one noise function is worse than the function.
 */

const RIBBONS = 7
const STEP = 6            // px between sample points; lower = smoother, slower
const SPEED = 0.0055
const COMPLEXITY = 0.0016
const AMPLITUDE = 140

// One accent per act, matching the page's own palette.
const HUES = [
  [251, 184, 74],    // gold      — overture
  [140, 199, 255],   // cold blue — console
  [158, 249, 204],   // green     — chain
  [219, 169, 255],   // violet    — coda
]

function makeSimplex() {
  const F2 = 0.5 * (Math.sqrt(3) - 1)
  const G2 = (3 - Math.sqrt(3)) / 6
  const p = new Uint8Array(256)
  for (let i = 0; i < 256; i++) p[i] = i
  // Deterministic shuffle: the backdrop should look the same on every load,
  // so a random permutation would be a bug, not a feature.
  let seed = 1337
  const rnd = () => (seed = (seed * 1664525 + 1013904223) & 0x7fffffff) / 0x7fffffff
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    ;[p[i], p[j]] = [p[j], p[i]]
  }
  const perm = new Uint8Array(512)
  const perm12 = new Uint8Array(512)
  const grad3 = new Float32Array([
    1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1, 1, 0, -1,
    -1, 0, -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
  ])
  for (let i = 0; i < 512; i++) {
    perm[i] = p[i & 255]
    perm12[i] = perm[i] % 12
  }
  return (xin, yin) => {
    let n0 = 0, n1 = 0, n2 = 0
    const s = (xin + yin) * F2
    const i = Math.floor(xin + s)
    const j = Math.floor(yin + s)
    const t = (i + j) * G2
    const x0 = xin - (i - t)
    const y0 = yin - (j - t)
    const i1 = x0 > y0 ? 1 : 0
    const j1 = x0 > y0 ? 0 : 1
    const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2
    const x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2
    const ii = i & 255, jj = j & 255
    let t0 = 0.5 - x0 * x0 - y0 * y0
    if (t0 >= 0) {
      const gi = perm12[ii + perm[jj]]
      t0 *= t0
      n0 = t0 * t0 * (grad3[gi * 3] * x0 + grad3[gi * 3 + 1] * y0)
    }
    let t1 = 0.5 - x1 * x1 - y1 * y1
    if (t1 >= 0) {
      const gi = perm12[ii + i1 + perm[jj + j1]]
      t1 *= t1
      n1 = t1 * t1 * (grad3[gi * 3] * x1 + grad3[gi * 3 + 1] * y1)
    }
    let t2 = 0.5 - x2 * x2 - y2 * y2
    if (t2 >= 0) {
      const gi = perm12[ii + 1 + perm[jj + 1]]
      t2 *= t2
      n2 = t2 * t2 * (grad3[gi * 3] * x2 + grad3[gi * 3 + 1] * y2)
    }
    return 70 * (n0 + n1 + n2)
  }
}

function lerp(a, b, t) { return a + (b - a) * t }

export default function AuroraRibbons({ progressRef }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const noise = makeSimplex()
    let raf = 0, w = 0, h = 0, t = 0, visible = true
    const mouse = { x: -1e4, y: -1e4, tx: -1e4, ty: -1e4, on: false }
    let prog = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = canvas.clientWidth; h = canvas.clientHeight
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const onMove = (e) => { mouse.tx = e.clientX; mouse.ty = e.clientY; mouse.on = true }
    const onLeave = () => { mouse.on = false }
    window.addEventListener('pointermove', onMove, { passive: true })
    window.addEventListener('pointerleave', onLeave)

    const onVis = () => { visible = !document.hidden }
    document.addEventListener('visibilitychange', onVis)

    const frame = () => {
      raf = requestAnimationFrame(frame)
      if (!visible) return
      t += SPEED
      mouse.x += (mouse.tx - mouse.x) * 0.08
      mouse.y += (mouse.ty - mouse.y) * 0.08

      const target = progressRef?.current ?? 0
      prog += (target - prog) * 0.05

      const x = Math.max(0, Math.min(HUES.length - 1.001, prog * (HUES.length - 1)))
      const ci = Math.floor(x), cf = x - ci
      const c0 = HUES[ci], c1 = HUES[ci + 1] ?? HUES[ci]
      const R = Math.round(lerp(c0[0], c1[0], cf))
      const G = Math.round(lerp(c0[1], c1[1], cf))
      const B = Math.round(lerp(c0[2], c1[2], cf))

      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = 'lighter'

      // More ribbons and more travel as the page opens up.
      const n = Math.round(lerp(4, RIBBONS, prog))
      for (let r = 0; r < n; r++) {
        const u = n > 1 ? r / (n - 1) : 0
        const baseY = h * (0.22 + u * 0.56)
        const amp = AMPLITUDE * (0.45 + prog * 0.75) * (0.6 + 0.4 * Math.sin(r * 1.7))

        const grad = ctx.createLinearGradient(0, 0, w, 0)
        grad.addColorStop(0, 'rgba(0,0,0,0)')
        grad.addColorStop(0.5, `rgba(${R},${G},${B},${0.30 + 0.22 * (1 - u)})`)
        grad.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.strokeStyle = grad
        ctx.lineWidth = 1.1 + (1 - u) * 0.9

        ctx.beginPath()
        for (let px = 0; px <= w; px += STEP) {
          // Cursor bends the field locally — same falloff mechanic the
          // bubble cluster uses, so pointer response feels consistent.
          const d = Math.hypot(px - mouse.x, baseY - mouse.y)
          const pull = mouse.on ? 1 + (1 - Math.min(1, d / 420)) * 0.9 : 1
          const nz = noise(px * COMPLEXITY * pull, r * 12.3 + t)
          const y = baseY + nz * amp * pull
          if (px === 0) ctx.moveTo(px, y); else ctx.lineTo(px, y)
        }
        ctx.stroke()
      }
      ctx.globalCompositeOperation = 'source-over'
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerleave', onLeave)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [progressRef])

  return (
    <div aria-hidden style={{
      position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none',
      background: 'radial-gradient(ellipse 120% 85% at 50% 25%, #0a0c14 0%, #05060a 55%, #030407 100%)',
    }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  )
}
