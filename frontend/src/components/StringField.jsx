import { useEffect, useRef } from 'react'

/**
 * StringField — the ribbon-mesh idea, rebuilt as piano strings.
 *
 * Adapted from the "Aether ribbon mesh" concept: superposed trigonometric
 * ribbons that react to cursor deflection and click impulse.  Three things
 * changed, and the reasons are the design:
 *
 * 1. THE MATH IS NOW REAL.  The source multiplied its waves by
 *    `sin((x/width) * PI)` and called it an edge envelope — that expression
 *    IS the fundamental mode shape of a string pinned at both ends.  It had
 *    arrived at string physics by eye.  So this draws the actual thing:
 *    displacement is a sum over partials p, each `sin(p*PI*x/W)` in space
 *    times `sin(w_p*t)` in time.  Endpoints stay pinned for free, higher
 *    partials get their own nodes, and the harmonic series is exactly the
 *    subject of this app.
 *
 * 2. STRUCK STRINGS DECAY.  A click excites the strings near it and that
 *    energy bleeds away, with higher partials dying faster than the
 *    fundamental — which is what makes a real piano note turn from bright
 *    to round as it rings.  The source's ripple expanded at constant
 *    amplitude, which reads as a shockwave rather than a note.
 *
 * 3. IT IS TRANSPARENT.  The source ran with `alpha: false` and painted an
 *    opaque background every frame.  Here it composites over ScrollStage,
 *    so the two layers read as one instrument.
 *
 * The "sparkles" burst is folded in here as struck-string dust rather than
 * pulling in tsparticles — we already own a particle loop, and this one
 * inherits the strike's colour and position for free.
 */

const N_STRINGS = 20
const PARTIALS = 3

// Bass strings glow amber, treble strings cyan — the same low→high colour
// language the rest of the app already uses for hands and registers.
const BASS = [245, 158, 11]
const TREB = [34, 211, 238]

function lerp(a, b, t) { return a + (b - a) * t }

export default function StringField({ height = 380 }) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return
    const ctx = canvas.getContext('2d')          // alpha ON — we composite
    let raf = 0, w = 0, h = 0, running = false
    let t = 0, last = performance.now()

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = canvas.clientWidth
      h = canvas.clientHeight
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    // Per-string state.  Index 0 is the lowest string.
    const strings = Array.from({ length: N_STRINGS }, (_, i) => {
      const u = i / (N_STRINGS - 1)
      return {
        u,
        // Shorter, tenser strings for the treble: more cycles, faster.
        cycles: 1 + u * 1.6,
        speed: 0.55 + u * 1.5,
        phase: i * 0.7,
        excite: 0,                 // energy from the last strike
        y: 0,                      // resting height, set on resize below
      }
    })

    const mouse = { x: -1, y: -1, tx: -1, ty: -1, on: false }
    const dust = []

    const toLocal = (cx, cy) => {
      const r = canvas.getBoundingClientRect()
      return { x: cx - r.left, y: cy - r.top, inside:
        cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom }
    }

    const onMove = (e) => {
      const p = 'touches' in e && e.touches[0]
        ? toLocal(e.touches[0].clientX, e.touches[0].clientY)
        : toLocal(e.clientX, e.clientY)
      mouse.tx = p.x; mouse.ty = p.y; mouse.on = p.inside
    }
    const onLeave = () => { mouse.on = false }

    // A strike: excite strings by proximity, and throw dust off the point.
    const onDown = (e) => {
      const src = 'touches' in e && e.touches[0] ? e.touches[0] : e
      const p = toLocal(src.clientX, src.clientY)
      if (!p.inside) return

      strings.forEach((s) => {
        const d = Math.abs(s.y - p.y)
        const reach = h * 0.30
        const hit = Math.exp(-Math.pow(d / reach, 2))
        if (hit > 0.02) s.excite = Math.min(1.6, s.excite + hit * 1.35)
      })

      const col = lerpColor(1 - p.y / Math.max(1, h))
      for (let i = 0; i < 26; i++) {
        const a = Math.random() * Math.PI * 2
        const sp = 0.6 + Math.random() * 2.6
        dust.push({
          x: p.x, y: p.y,
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 0.5,
          life: 1, decay: 0.012 + Math.random() * 0.02,
          r: 0.7 + Math.random() * 1.7, col,
        })
      }
    }

    function lerpColor(u) {
      return [
        Math.round(lerp(BASS[0], TREB[0], u)),
        Math.round(lerp(BASS[1], TREB[1], u)),
        Math.round(lerp(BASS[2], TREB[2], u)),
      ]
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('touchmove', onMove, { passive: true })
    window.addEventListener('mouseleave', onLeave)
    window.addEventListener('mousedown', onDown)
    window.addEventListener('touchstart', onDown, { passive: true })

    // Don't burn frames while the hero is scrolled away.  With three canvas
    // systems on this page now, each one has to pay for itself only while
    // it is actually visible.
    // start/stop are guarded by `running` because IntersectionObserver
    // fires once immediately on observe().  Without the guard, an element
    // that is already visible gets a second rAF loop on top of the one
    // started below, and every animation runs at double speed.
    const start = () => {
      if (running) return
      running = true
      last = performance.now()          // else dt spans the whole pause
      raf = requestAnimationFrame(frame)
    }
    const stop = () => {
      if (!running) return
      running = false
      cancelAnimationFrame(raf)
    }
    const io = new IntersectionObserver(
      ([e]) => (e.isIntersecting ? start() : stop()),
      { threshold: 0 },
    )
    io.observe(wrap)

    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      t += dt

      mouse.x += (mouse.tx - mouse.x) * Math.min(1, dt * 9)
      mouse.y += (mouse.ty - mouse.y) * Math.min(1, dt * 9)

      ctx.clearRect(0, 0, w, h)

      const pad = h * 0.10
      const span = h - pad * 2

      for (let i = 0; i < N_STRINGS; i++) {
        const s = strings[i]
        s.y = pad + (i / (N_STRINGS - 1)) * span
        // Strike energy decays; this is what makes it a note, not a pulse.
        s.excite *= Math.pow(0.28, dt)

        const [R, G, B] = lerpColor(s.u)
        const amp = (1.4 + s.excite * 16) * (1 - s.u * 0.45)

        ctx.beginPath()
        const STEP = 5
        for (let x = 0; x <= w; x += STEP) {
          const xn = x / w
          let d = 0

          // Standing wave: spatial mode shape x temporal oscillation.
          // Endpoints are pinned automatically because sin(p*PI*0)=0 and
          // sin(p*PI*1)=0 — no hand-rolled envelope needed.
          for (let p = 1; p <= PARTIALS; p++) {
            // Upper partials decay faster, so a struck string turns from
            // bright to round as it rings out.
            const pw = p === 1 ? 1 : (s.excite > 0.01 ? Math.pow(s.excite, p - 1) : 0) + 0.18 / p
            d += (Math.sin(p * Math.PI * xn) *
                  Math.sin(t * s.speed * p * 1.7 + s.phase * p)) * (amp / p) * pw
          }

          // Cursor deflection — the string bends toward the pointer as it
          // passes, like a finger resting on it.
          if (mouse.on) {
            const dx = Math.abs(x - mouse.x)
            const dy = s.y - mouse.y
            const pull = Math.exp(-Math.pow(dx / (w * 0.20), 2)) *
                         Math.exp(-Math.pow(dy / (h * 0.22), 2))
            d += -dy * pull * 0.34
          }

          const y = s.y + d
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
        }

        const glow = Math.min(1, s.excite)
        ctx.strokeStyle = `rgba(${R},${G},${B},${0.16 + glow * 0.55})`
        // Bass strings are physically thicker.  Struck strings thicken too,
        // which reads as brightness without needing a separate glow pass.
        ctx.lineWidth = (1.5 - s.u * 0.85) + glow * 1.5
        ctx.stroke()

        // Harmonic nodes of the 2nd partial — genuine still points on a
        // vibrating string, so the markers mean something.
        if (glow > 0.12) {
          ctx.fillStyle = `rgba(${R},${G},${B},${glow * 0.75})`
          for (let k = 1; k < 2; k++) {
            const nx = (k / 2) * w
            ctx.beginPath(); ctx.arc(nx, s.y, 1.6 + glow * 1.8, 0, Math.PI * 2); ctx.fill()
          }
        }
      }

      // Struck-string dust
      for (let i = dust.length - 1; i >= 0; i--) {
        const d = dust[i]
        d.x += d.vx; d.y += d.vy
        d.vy += 0.028                 // settles downward
        d.vx *= 0.985; d.vy *= 0.985
        d.life -= d.decay
        if (d.life <= 0) { dust.splice(i, 1); continue }
        ctx.fillStyle = `rgba(${d.col[0]},${d.col[1]},${d.col[2]},${d.life * 0.85})`
        ctx.beginPath(); ctx.arc(d.x, d.y, d.r * d.life, 0, Math.PI * 2); ctx.fill()
      }

      raf = requestAnimationFrame(frame)
    }
    start()

    return () => {
      stop()
      io.disconnect()
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('mouseleave', onLeave)
      window.removeEventListener('mousedown', onDown)
      window.removeEventListener('touchstart', onDown)
    }
  }, [])

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
      <canvas
        ref={canvasRef}
        aria-hidden
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  )
}
