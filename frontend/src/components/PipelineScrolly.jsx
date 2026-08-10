import { useEffect, useRef, useState } from 'react'
import { getLenis } from '../utils/smoothscroll'

/**
 * PipelineScrolly — the pipeline, told by scrolling.
 *
 * The architecture, because it is the whole idea
 * ----------------------------------------------
 * There is ONE particle system of N tokens that lives for the entire
 * section.  Each stage of the pipeline is not a different picture — it is a
 * different LAYOUT FUNCTION for those same tokens.  Scrolling interpolates
 * every token between the layout it has in stage i and the one it has in
 * stage i+1.
 *
 * That is what makes this worth building rather than cross-fading five
 * images: the tokens persist, so you watch the *same* material become a
 * waveform, then separated stems, then notes on a roll, then two hands,
 * then keys. Continuity of identity is the argument the section is making —
 * that nothing is invented along the way, it is the same audio being
 * progressively re-read.
 *
 * A token is deterministic in its index: layout(i) is pure. No physics, no
 * accumulated state, so scrubbing backwards is exact rather than
 * approximate, and there is no drift when the user scrolls fast.
 */

const STAGES = [
  { key: 'audio',      label: 'AUDIO IN',    color: '#f59e0b',
    blurb: 'A recording arrives as one dense stream — everything at once.' },
  { key: 'separate',   label: 'SEPARATE',    color: '#3b82f6',
    blurb: 'Demucs pulls the mix apart into sources, so the piano can be heard alone.' },
  { key: 'transcribe', label: 'TRANSCRIBE',  color: '#06b6d4',
    blurb: 'Onsets & Frames reads the audio as discrete notes — pitch, time, velocity.' },
  { key: 'arrange',    label: 'ARRANGE',     color: '#a78bfa',
    blurb: 'Notes are split between two hands and fitted to what a player can reach.' },
  { key: 'render',     label: 'RENDER',      color: '#34d399',
    blurb: 'The result is played back on a sampled grand, and written out as MIDI.' },
]

const N = 260

function lerp(a, b, t) { return a + (b - a) * t }

// Smoothstep — tokens should ease into each formation, not slide linearly.
// Linear interpolation between layouts looks mechanical; the ease is what
// makes it read as material settling into a new shape.
function ease(t) { return t * t * (3 - 2 * t) }

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}
const STAGE_RGB = STAGES.map((s) => hexToRgb(s.color))

/**
 * Where token `i` sits in stage `s`, in normalised 0..1 space.
 * Pure: same inputs, same output, every frame.
 */
function layout(s, i, n) {
  const u = i / n                       // position along the token's index
  const r1 = ((i * 9301 + 49297) % 233280) / 233280   // stable per-token noise
  const r2 = ((i * 4021 + 12983) % 233280) / 233280

  switch (s) {
    // ── 0 · one dense waveform ──────────────────────────────────────────
    case 0: {
      // Amplitude envelope of a plausible musical phrase, so it reads as a
      // recording rather than noise.
      const env = 0.30 + 0.26 * Math.abs(Math.sin(u * Math.PI * 3.1))
                       * (0.55 + 0.45 * Math.sin(u * Math.PI * 11))
      const side = i % 2 ? 1 : -1
      return { x: u, y: 0.5 + side * env * (0.35 + 0.65 * r1), sz: 1.0, a: 0.85 }
    }

    // ── 1 · four stems, stacked ─────────────────────────────────────────
    case 1: {
      const band = i % 4                     // drums / bass / other / piano
      const yc = 0.19 + band * 0.205
      const env = 0.055 * (0.4 + 0.6 * Math.abs(Math.sin(u * Math.PI * (4 + band * 3))))
      return { x: u, y: yc + (r1 - 0.5) * 2 * env, sz: 0.85, a: band === 3 ? 0.95 : 0.30 }
    }

    // ── 2 · piano roll: time on x, pitch on y, both quantised ───────────
    case 2: {
      const COLS = 32, ROWS = 22
      const col = Math.floor(u * COLS)
      // Pitch follows a slow contour so the roll looks like music, not hash
      const centre = 11 + Math.round(6.5 * Math.sin(col * 0.42) + 3 * Math.sin(col * 0.17))
      const row = Math.max(0, Math.min(ROWS - 1, centre + Math.round((r2 - 0.5) * 5)))
      return {
        x: 0.06 + (col / COLS) * 0.88 + 0.008,
        y: 0.88 - (row / ROWS) * 0.74,
        sz: 1.25, a: 0.9,
      }
    }

    // ── 3 · two hands, separated by the gap a real player has ───────────
    case 3: {
      const rh = i % 3 !== 0                 // more material in the right hand
      const col = Math.floor(u * 26)
      const contour = Math.sin(col * 0.5) * 0.5 + 0.5
      const y = rh
        ? 0.20 + contour * 0.16 + (r1 - 0.5) * 0.06     // melody, upper stave
        : 0.66 + contour * 0.13 + (r1 - 0.5) * 0.05     // accompaniment, lower
      return { x: 0.07 + (col / 26) * 0.86, y, sz: rh ? 1.3 : 1.1, a: rh ? 0.95 : 0.6 }
    }

    // ── 4 · settle onto the keyboard ────────────────────────────────────
    default: {
      const KEYS = 36
      const k = Math.floor(u * KEYS)
      const black = [1, 3, 6, 8, 10].includes(k % 12)
      return {
        x: 0.08 + (k / KEYS) * 0.84 + (r2 - 0.5) * 0.004,
        y: black ? 0.60 : 0.70,
        sz: black ? 1.0 : 1.45,
        a: 0.92,
      }
    }
  }
}

export default function PipelineScrolly({ id = 'pipeline' }) {
  const outerRef = useRef(null)
  const canvasRef = useRef(null)
  const progRef = useRef(0)          // 0..1 across the whole pinned run
  const [stage, setStage] = useState(0)

  useEffect(() => {
    const outer = outerRef.current
    const canvas = canvasRef.current
    if (!outer || !canvas) return
    const ctx = canvas.getContext('2d')
    let raf = 0, w = 0, h = 0

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

    // Progress through the pinned run: 0 when the sticky panel first locks
    // to the top, 1 when the section's bottom reaches the viewport bottom.
    const readProgress = () => {
      const r = outer.getBoundingClientRect()
      const runway = Math.max(1, r.height - window.innerHeight)
      progRef.current = Math.max(0, Math.min(1, -r.top / runway))
    }
    readProgress()

    // Lenis emits its own high-resolution scroll signal; using it rather
    // than the DOM event gives us the interpolated position on the same
    // frame Lenis computed it, so the canvas never lags the page by a tick.
    const lenis = getLenis()
    const onLenis = () => readProgress()
    if (lenis) lenis.on('scroll', onLenis)
    window.addEventListener('scroll', readProgress, { passive: true })

    let shown = -1
    const frame = () => {
      const p = progRef.current
      // Map progress across the segments BETWEEN stages
      const x = p * (STAGES.length - 1)
      const i = Math.min(STAGES.length - 2, Math.floor(x))
      const t = ease(Math.max(0, Math.min(1, x - i)))

      const near = t < 0.5 ? i : i + 1
      if (near !== shown) { shown = near; setStage(near) }

      ctx.clearRect(0, 0, w, h)

      const [r0, g0, b0] = STAGE_RGB[i]
      const [r1c, g1c, b1c] = STAGE_RGB[i + 1]
      const R = Math.round(lerp(r0, r1c, t))
      const G = Math.round(lerp(g0, g1c, t))
      const B = Math.round(lerp(b0, b1c, t))

      // Baseline that only exists at the keyboard end — the ledger the
      // keys sit on. Fades in over the last segment.
      if (i === STAGES.length - 2) {
        ctx.strokeStyle = `rgba(${R},${G},${B},${0.16 * t})`
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(w * 0.08, h * 0.78); ctx.lineTo(w * 0.92, h * 0.78)
        ctx.stroke()
      }

      for (let k = 0; k < N; k++) {
        const A = layout(i, k, N)
        const Bl = layout(i + 1, k, N)
        const px = lerp(A.x, Bl.x, t) * w
        const py = lerp(A.y, Bl.y, t) * h
        const sz = lerp(A.sz, Bl.sz, t)
        const al = lerp(A.a, Bl.a, t)

        // Tokens in flight glow slightly brighter — motion is legible as
        // energy, which is what sells "being transformed" over "being moved".
        const travel = Math.hypot(Bl.x - A.x, Bl.y - A.y)
        const heat = travel * 4 * Math.sin(t * Math.PI)

        ctx.fillStyle = `rgba(${R},${G},${B},${Math.min(1, al + heat * 0.3)})`
        ctx.beginPath()
        ctx.arc(px, py, Math.max(0.6, sz * 1.7), 0, Math.PI * 2)
        ctx.fill()

        if (heat > 0.08) {
          ctx.fillStyle = `rgba(${R},${G},${B},${Math.min(0.5, heat * 0.35)})`
          ctx.beginPath()
          ctx.arc(px, py, sz * 4.5, 0, Math.PI * 2)
          ctx.fill()
        }
      }
      raf = requestAnimationFrame(frame)
    }
    // The pipeline is five viewports tall but only worth drawing while some
    // of it is on screen.  With the hero's StringField and ScrollStage also
    // running, leaving this looping from the top of the page meant paying
    // for 260 tokens nobody could see.
    // Guarded by `running`: IntersectionObserver fires once immediately on
    // observe(), so an already-visible element would otherwise get a second
    // rAF loop stacked on the one started below.
    let running = false
    const start = () => { if (!running) { running = true; raf = requestAnimationFrame(frame) } }
    const stop  = () => { if (running)  { running = false; cancelAnimationFrame(raf) } }

    const io = new IntersectionObserver(
      ([e]) => (e.isIntersecting ? start() : stop()),
      { threshold: 0 },
    )
    io.observe(outer)
    start()

    return () => {
      stop()
      io.disconnect()
      window.removeEventListener('resize', resize)
      window.removeEventListener('scroll', readProgress)
      if (lenis) lenis.off('scroll', onLenis)
    }
  }, [])

  const active = STAGES[stage]

  return (
    <section
      id={id}
      ref={outerRef}
      // Five viewports of scroll-scrubbed narrative: the snap system must
      // step THROUGH this, not over it.
      data-free-scroll
      // Runway: one viewport per transition, plus one to read the last
      // stage. Less than this and the morphs go by too fast to follow.
      style={{ position: 'relative', height: `${STAGES.length * 100}vh` }}
    >
      <div style={{
        position: 'sticky', top: 0, height: '100vh',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 18,
      }}>
        {/* stage ticks */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {STAGES.map((s, i) => (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                width: i === stage ? 26 : 7, height: 3, borderRadius: 2,
                background: i <= stage ? s.color : 'rgba(255,255,255,0.12)',
                boxShadow: i === stage ? `0 0 10px ${s.color}` : 'none',
                transition: 'width .45s cubic-bezier(.4,0,.2,1), background .45s',
              }} />
            </div>
          ))}
        </div>

        <div style={{ textAlign: 'center', minHeight: 76, padding: '0 24px' }}>
          <div style={{
            fontFamily: 'ui-monospace,monospace', fontSize: 10, fontWeight: 700,
            letterSpacing: '0.24em', color: active.color, marginBottom: 8,
            transition: 'color .4s',
          }}>
            {String(stage + 1).padStart(2, '0')} · {active.label}
          </div>
          <p style={{
            fontSize: 14.5, color: 'rgba(255,255,255,0.50)', maxWidth: 460,
            margin: '0 auto', lineHeight: 1.55,
          }}>
            {active.blurb}
          </p>
        </div>

        <canvas
          ref={canvasRef}
          aria-hidden
          style={{ width: 'min(920px, 92vw)', height: 'min(420px, 46vh)' }}
        />
      </div>
    </section>
  )
}
