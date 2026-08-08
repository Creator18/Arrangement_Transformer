import { useEffect, useRef } from 'react'

/**
 * ScrollStage — the living backdrop.
 *
 * One fixed canvas behind everything, whose *scene* is driven by scroll
 * position.  Each section of the page owns a palette and a motion
 * character, and the stage interpolates between them continuously rather
 * than switching, so scrolling reads as one machine changing state instead
 * of separate pages.
 *
 * Canvas rather than DOM elements: at ~90 drifting particles plus a
 * gradient field and a scan grid, animating DOM nodes would thrash layout
 * every frame.  One canvas is a single composite.
 *
 * Everything is driven by two smoothed values — scroll progress and scroll
 * VELOCITY.  Velocity is what makes it feel responsive: particles stretch
 * and the grid brightens while you move, then settle when you stop.
 */

// One entry per page section, in document order.  `hue` drives the field,
// `drift` the particle speed, `density` how busy the scene is.
const SCENES = [
  { hue: 38,  drift: 0.22, density: 0.55, name: 'hero'     },  // amber
  { hue: 265, drift: 0.40, density: 0.85, name: 'console'  },  // violet
  { hue: 190, drift: 0.30, density: 0.70, name: 'pipeline' },  // cyan
  { hue: 150, drift: 0.50, density: 1.00, name: 'engine'   },  // green
]

const N_PARTICLES = 90

function lerp(a, b, t) { return a + (b - a) * t }

/**
 * Continuous scene index from the ACTUAL positions of the sections, not
 * from a fraction of document height.
 *
 * Document-fraction was fine while every section was roughly a screen tall.
 * The pinned pipeline section is five viewports tall, which makes it ~60%
 * of the page — under the old mapping the backdrop would sprint through
 * three palettes while the reader was still inside a single section, and
 * then sit frozen for the rest.  Anchoring to section centres means each
 * section holds its own colour for exactly as long as it is on screen,
 * which is also what makes the backdrop agree with the rail.
 *
 * Returns a float: 2.4 means "40% of the way from section 2 to section 3".
 */
function sceneIndexFromSections(ids) {
  if (!ids?.length) return 0
  const focus = window.scrollY + window.innerHeight * 0.5
  const centres = ids.map((id) => {
    const el = document.getElementById(id)
    if (!el) return null
    const r = el.getBoundingClientRect()
    return r.top + window.scrollY + r.height * 0.5
  })
  const pts = centres.map((c, i) => [c, i]).filter(([c]) => c != null)
  if (!pts.length) return 0
  if (focus <= pts[0][0]) return pts[0][1]
  for (let k = 0; k < pts.length - 1; k++) {
    const [c0, i0] = pts[k], [c1, i1] = pts[k + 1]
    if (focus <= c1) {
      const t = (focus - c0) / Math.max(1, c1 - c0)
      return i0 + t * (i1 - i0)
    }
  }
  return pts[pts.length - 1][1]
}

function sceneAt(x0) {
  // x0 is a continuous scene index, not a document fraction
  const x = Math.max(0, Math.min(SCENES.length - 1.0001, x0))
  const i = Math.floor(x)
  const t = x - i
  const a = SCENES[i], b = SCENES[Math.min(i + 1, SCENES.length - 1)]
  return {
    hue:     lerp(a.hue, b.hue, t),
    drift:   lerp(a.drift, b.drift, t),
    density: lerp(a.density, b.density, t),
  }
}

export default function ScrollStage({ sections = [] }) {
  const canvasRef = useRef(null)
  // `scene`/`sceneTarget` are a scene INDEX (0..SCENES.length-1) driven by
  // section geometry; `progress` stays a plain 0..1 document fraction and
  // only drives the slow drift of the gradient field and grid.
  const state = useRef({ scene: 0, sceneTarget: 0, progress: 0, target: 0, velocity: 0, lastY: 0 })
  const idsRef = useRef(sections.map((s) => (typeof s === 'string' ? s : s.id)))
  idsRef.current = sections.map((s) => (typeof s === 'string' ? s : s.id))

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let raf = 0
    let w = 0, h = 0, dpr = 1

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = canvas.clientWidth
      h = canvas.clientHeight
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    // Deterministic particle field — no hydration flicker, same every load.
    let seed = 7
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) & 0xffffffff
      return (seed >>> 0) / 0xffffffff
    }
    const parts = Array.from({ length: N_PARTICLES }, () => ({
      x: rand(), y: rand(),
      r: rand() * 1.8 + 0.4,
      sp: rand() * 0.6 + 0.2,
      ph: rand() * Math.PI * 2,
    }))

    const onScroll = () => {
      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
      const y = window.scrollY
      state.current.target = y / max
      state.current.sceneTarget = sceneIndexFromSections(idsRef.current)
      state.current.velocity += Math.abs(y - state.current.lastY) * 0.02
      state.current.lastY = y
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    let t0 = performance.now()
    const frame = (now) => {
      const dt = Math.min(0.05, (now - t0) / 1000)
      t0 = now
      const s = state.current
      // Smooth both, so nothing snaps
      s.progress += (s.target - s.progress) * Math.min(1, dt * 4)
      s.scene    += (s.sceneTarget - s.scene) * Math.min(1, dt * 3)
      s.velocity *= Math.pow(0.02, dt)          // decay toward rest
      const vel = Math.min(1, s.velocity)

      const sc = sceneAt(s.scene)
      ctx.clearRect(0, 0, w, h)

      // ── field: two large soft radials that swap position with scroll ──
      const cx1 = w * (0.25 + 0.25 * Math.sin(s.progress * Math.PI * 1.4))
      const cy1 = h * (0.30 + 0.20 * Math.cos(s.progress * Math.PI * 1.1))
      const cx2 = w * (0.75 - 0.25 * Math.sin(s.progress * Math.PI * 1.7))
      const cy2 = h * (0.70 - 0.18 * Math.cos(s.progress * Math.PI * 1.3))
      for (const [cx, cy, hue, alpha, rad] of [
        [cx1, cy1, sc.hue,        0.16 + 0.06 * vel, Math.max(w, h) * 0.55],
        [cx2, cy2, (sc.hue + 45) % 360, 0.12 + 0.05 * vel, Math.max(w, h) * 0.45],
      ]) {
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad)
        g.addColorStop(0,   `hsla(${hue}, 85%, 55%, ${alpha})`)
        g.addColorStop(0.5, `hsla(${hue}, 85%, 45%, ${alpha * 0.35})`)
        g.addColorStop(1,   'hsla(0,0%,0%,0)')
        ctx.fillStyle = g
        ctx.fillRect(0, 0, w, h)
      }

      // ── scan grid: the workstation surface, brighter while moving ──
      const step = 46
      ctx.lineWidth = 1
      ctx.strokeStyle = `hsla(${sc.hue}, 70%, 60%, ${0.035 + 0.05 * vel})`
      ctx.beginPath()
      const off = (s.progress * step * 6) % step
      for (let x = -off; x < w; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, h) }
      for (let y = -off; y < h; y += step) { ctx.moveTo(0, y); ctx.lineTo(w, y) }
      ctx.stroke()

      // ── particles: drift up, stretch with velocity ──
      const n = Math.floor(parts.length * sc.density)
      for (let i = 0; i < n; i++) {
        const p = parts[i]
        p.y -= dt * sc.drift * p.sp * 0.06
        if (p.y < -0.05) { p.y = 1.05; p.x = rand() }
        const px = (p.x + 0.012 * Math.sin(now / 2600 + p.ph)) * w
        const py = p.y * h
        const len = 1 + vel * 26 * p.sp
        const hue = (sc.hue + (i % 3) * 22) % 360
        ctx.strokeStyle = `hsla(${hue}, 90%, 68%, ${0.10 + 0.30 * p.sp})`
        ctx.lineWidth = p.r
        ctx.beginPath()
        ctx.moveTo(px, py)
        ctx.lineTo(px, py + len)
        ctx.stroke()
      }

      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('scroll', onScroll)
    }
  }, [])

  return (
    <>
      <canvas
        ref={canvasRef}
        aria-hidden
        style={{
          position: 'fixed', inset: 0, width: '100%', height: '100%',
          pointerEvents: 'none', zIndex: 0,
        }}
      />
      {/* Vignette keeps text legible over the brightest field moments */}
      <div aria-hidden style={{
        position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        background:
          'radial-gradient(ellipse 90% 70% at 50% 40%, transparent 0%, rgba(6,6,10,0.72) 78%)',
      }} />
    </>
  )
}
