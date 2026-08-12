import { useEffect, useRef } from 'react'

/**
 * SonicWaveform — the landing backdrop, rebuilt as a field of waveforms.
 *
 * Replaces AuroraRibbons here.  Both draw displaced lines, but noise ribbons
 * read as weather and stacked waveforms read as SIGNAL, which is what this
 * product actually does: sixty traces of the same wave, each offset in phase,
 * so the field looks like an oscilloscope with persistence rather than an
 * aurora.
 *
 * Adapted from the Sonic Waveform hero rather than pasted.  Four things
 * changed, and each one is a bug the original would have shipped here:
 *
 *   - it sized the canvas to `window`, which on a page whose backdrop is
 *     `position: fixed` inside a scroll container gives the wrong height on
 *     mobile and never reacts to a container resize.  This measures its own
 *     parent through a ResizeObserver.
 *   - it ignored devicePixelRatio, so every line was soft on a retina screen.
 *     Capped at 2 because the trail effect redraws the whole surface each
 *     frame and 3x costs more than it shows.
 *   - the colour was hard-coded teal.  The landing page already moves through
 *     four accents as you scroll, and a backdrop that ignores that fights the
 *     page.  Accent is a prop and is interpolated, not snapped, so the field
 *     drifts between acts instead of flicking.
 *   - it ran forever.  This stops when the tab is hidden or the element
 *     scrolls out of view, and honours prefers-reduced-motion by drawing a
 *     single static frame.
 *
 * The trail is the whole look: each frame paints translucent black over the
 * last one rather than clearing it, so lines smear into their own history.
 * That requires an opaque backdrop underneath, which is why this sits on
 * --ct-ink and not on a transparent parent.
 */

const LINES = 60
const SEGMENTS = 80
const MAX_DPR = 2

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '')
  return m
    ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)]
    : [0, 255, 192]
}

export default function SonicWaveform({ accent = '#00ffc0', progressRef }) {
  const canvasRef = useRef(null)
  const accentRef = useRef(accent)
  accentRef.current = accent

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    let raf = 0
    let time = 0
    let visible = true
    let w = 0
    let h = 0
    const mouse = { x: -9999, y: -9999 }
    // Interpolated so an accent change eases in over ~1s rather than jumping.
    let rgb = hexToRgb(accent)

    const resize = () => {
      const parent = canvas.parentElement
      const rect = parent ? parent.getBoundingClientRect() : { width: window.innerWidth, height: window.innerHeight }
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR)
      w = Math.max(1, Math.floor(rect.width))
      h = Math.max(1, Math.floor(rect.height))
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      // A resize discards the trail, so repaint the backdrop rather than
      // leaving a stretched ghost of the previous size.
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, w, h)
    }

    const frame = () => {
      const target = hexToRgb(accentRef.current)
      rgb = rgb.map((c, i) => c + (target[i] - c) * 0.04)

      // Scroll opens the field up, matching what AuroraRibbons did, so the
      // page still gains intensity as you move through it.
      const p = progressRef?.current ?? 0
      const amp = 0.6 + p * 0.8

      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
      ctx.fillRect(0, 0, w, h)

      const mid = h / 2
      const [r, g, b] = rgb
      for (let i = 0; i < LINES; i++) {
        ctx.beginPath()
        const intensity = Math.sin((i / LINES) * Math.PI)
        // 0.26, not the 0.5 this was adapted from.  The backdrop is FIXED, so
        // the band sits at viewport centre permanently — which is exactly
        // where the preset grid and the drop zone live.  At 0.5 it washed out
        // "7 oct · full range" and the accepted-formats line.  A hero that
        // owns an empty screen can be bright; a backdrop behind controls
        // cannot.
        ctx.strokeStyle = `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${intensity * 0.26})`
        ctx.lineWidth = 1.5

        for (let j = 0; j <= SEGMENTS; j++) {
          const x = (j / SEGMENTS) * w
          const d = Math.hypot(x - mouse.x, mid - mouse.y)
          const pull = Math.max(0, 1 - d / 400)
          const noise = Math.sin(j * 0.1 + time + i * 0.2) * 20
          const spike =
            Math.cos(j * 0.2 + time + i * 0.1) * Math.sin(j * 0.05 + time) * 50
          const y = mid + (noise + spike * (1 + pull * 2)) * amp
          if (j === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }

      time += 0.02
      if (!reduced && visible) raf = requestAnimationFrame(frame)
    }

    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect()
      mouse.x = e.clientX - rect.left
      mouse.y = e.clientY - rect.top
    }
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999 }

    const start = () => {
      if (raf || reduced) return
      raf = requestAnimationFrame(frame)
    }
    const stop = () => { cancelAnimationFrame(raf); raf = 0 }

    const ro = new ResizeObserver(resize)
    if (canvas.parentElement) ro.observe(canvas.parentElement)

    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (visible) start(); else stop()
    }, { threshold: 0 })
    io.observe(canvas)

    const onVisibility = () => {
      if (document.hidden) { visible = false; stop() }
      else { visible = true; start() }
    }

    window.addEventListener('mousemove', onMove, { passive: true })
    window.addEventListener('mouseout', onLeave, { passive: true })
    document.addEventListener('visibilitychange', onVisibility)

    resize()
    if (reduced) frame()          // one static frame, no loop
    else start()

    return () => {
      stop()
      ro.disconnect()
      io.disconnect()
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseout', onLeave)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [progressRef])

  return (
    <div
      aria-hidden="true"
      style={{
        position: 'fixed', inset: 0, zIndex: 0,
        background: '#000', pointerEvents: 'none',
      }}
    >
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      {/* The band is brightest across the vertical middle, which is where all
          the controls sit, so the scrim is heaviest there rather than only at
          the bottom edge.  Text first, texture second. */}
      <div style={{
        position: 'absolute', inset: 0,
        background:
          'linear-gradient(to bottom, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.62) 40%, rgba(0,0,0,0.62) 60%, rgba(0,0,0,0.4) 80%, var(--ct-ink) 100%)',
      }} />
    </div>
  )
}
