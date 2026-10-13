import { useEffect, useRef } from 'react'
import { getLevels } from '../utils/tonechain'

/**
 * MusicReactiveField — the results-page backdrop: four filled, layered waves
 * whose height, motion and colour follow the track playing in the roll.
 *
 * Ported from the "music-reactive hero" (bassIntensity / midIntensity /
 * trebleIntensity / hue / saturation / lightness props kept), with changes so
 * it can sit behind a live piano roll:
 *
 *   - no <audio> or play button of its own: it listens through tonechain's
 *     analyser, i.e. to whatever the roll's player is playing;
 *   - the hero text, buttons and artist block are gone — this is a backdrop;
 *   - the three full-screen costs per frame are removed: chromatic aberration
 *     (copied the whole canvas into a fresh canvas every frame), per-pixel
 *     film grain regenerated in JS every second frame (now a small pre-made
 *     tile, offset each frame), and hundreds of scanline strokes (now one
 *     pre-rendered pattern);
 *   - sized to its parent with devicePixelRatio (capped), paused when hidden,
 *     a single static frame under prefers-reduced-motion, and a scrim so the
 *     controls on top stay readable.
 *
 * With nothing playing it runs the original's gentle demo motion.
 */
const MAX_DPR = 1.5
const WAVES = [
  { amplitude: 30, frequency: 0.003, speed: 0.02, offset: 0, opacity: 0.9 },
  { amplitude: 25, frequency: 0.004, speed: 0.015, offset: Math.PI * 0.5, opacity: 0.7 },
  { amplitude: 20, frequency: 0.005, speed: 0.025, offset: Math.PI, opacity: 0.5 },
  { amplitude: 35, frequency: 0.002, speed: 0.01, offset: Math.PI * 1.5, opacity: 0.6 },
]

function makeGrainTile(size = 128) {
  const c = document.createElement('canvas'); c.width = c.height = size
  const g = c.getContext('2d'); const img = g.createImageData(size, size)
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255
  }
  g.putImageData(img, 0, 0)
  return c
}

function makeScanlines() {
  const c = document.createElement('canvas'); c.width = 4; c.height = 3
  const g = c.getContext('2d'); g.fillStyle = 'rgba(0,0,0,0.02)'; g.fillRect(0, 0, 4, 1)
  return c
}

export default function MusicReactiveField({ bassIntensity = 3, midIntensity = 1, trebleIntensity = 1,
                                            hue = 0, saturation = 0, lightness = 0 }) {
  const canvasRef = useRef(null)
  const params = useRef({})
  params.current = { bassIntensity, midIntensity, trebleIntensity, hue, saturation, lightness }

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    const grain = ctx.createPattern(makeGrainTile(), 'repeat')
    const scan = ctx.createPattern(makeScanlines(), 'repeat')
    const waves = WAVES.map(w => ({ ...w }))
    const bassHist = new Array(20).fill(0)
    const color = { h: 30, s: 80, l: 50 }
    const st = { bass: 0, mid: 0, treble: 0, time: 0, grainK: 0.04 }
    let w = 1, h = 1, raf = 0, visible = true

    const resize = () => {
      const r = canvas.parentElement.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR)
      w = Math.max(1, Math.floor(r.width)); h = Math.max(1, Math.floor(r.height))
      canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h)
    }

    const frame = () => {
      const p = params.current
      const L = getLevels()
      let tgt
      if (L) {
        bassHist.shift(); bassHist.push(L.bass)
        const avgBass = bassHist.reduce((a, b) => a + b) / bassHist.length
        st.bass = avgBass * p.bassIntensity
        st.mid = L.mid * p.midIntensity
        st.treble = L.treble * p.trebleIntensity
        // colour follows the dominant band, as in the original
        tgt = L.bass > L.mid && L.bass > L.treble ? { h: L.bass * 30, s: 80 + L.bass * 20, l: 50 + L.bass * 10 }
          : L.mid > L.treble ? { h: 40 + L.mid * 80, s: 20 + L.mid * 30, l: 55 + L.mid * 15 }
          : { h: 200 + L.treble * 80, s: 20 + L.treble * 40, l: 60 + L.treble * 10 }
        st.grainK = 0.03 + L.bass * 0.2
      } else {
        const t = st.time
        st.bass = (0.4 + Math.sin(t * 0.01) * 0.3) * p.bassIntensity
        st.mid = (0.3 + Math.sin(t * 0.015) * 0.2) * p.midIntensity
        st.treble = (0.2 + Math.sin(t * 0.02) * 0.1) * p.trebleIntensity
        tgt = { h: 180 + Math.sin(t * 0.005) * 180, s: 70 + Math.sin(t * 0.01) * 30, l: 50 + Math.sin(t * 0.008) * 20 }
        st.grainK = 0.04
      }
      tgt.h += p.hue
      tgt.s = Math.max(0, Math.min(100, tgt.s + p.saturation))
      tgt.l = Math.max(0, Math.min(100, tgt.l + p.lightness))
      color.h += (tgt.h - color.h) * 0.5; color.s += (tgt.s - color.s) * 0.2; color.l += (tgt.l - color.l) * 0.1
      st.time++

      ctx.fillStyle = 'rgba(0,0,0,0.92)'; ctx.fillRect(0, 0, w, h)
      const cy = h / 2
      waves.forEach((wave, i) => {
        wave.offset += wave.speed * (1 + st.bass * 0.8)
        const amp = wave.amplitude * (1 + (i < 2 ? st.bass : st.mid) * 5)
        const wh = color.h + i * 15, ws = color.s - i * 5, wl = color.l + i * 5
        const g = ctx.createLinearGradient(0, cy - amp, 0, cy + amp)
        const a = wave.opacity * (0.5 + st.bass * 0.5)
        g.addColorStop(0, `hsla(${wh},${ws}%,${wl}%,0)`)
        g.addColorStop(0.5, `hsla(${wh},${ws}%,${wl + 10}%,${a})`)
        g.addColorStop(1, `hsla(${wh},${ws}%,${wl}%,0)`)
        ctx.beginPath()
        for (let x = -50; x <= w + 50; x += 4) {
          const y = cy + Math.sin(x * wave.frequency + wave.offset) * amp
            + Math.sin(x * wave.frequency * 2 + wave.offset * 1.5) * amp * 0.3 * st.mid
            + Math.sin(x * wave.frequency * 0.5 + wave.offset * 0.7) * amp * 0.5
          if (x === -50) ctx.moveTo(x, y); else ctx.lineTo(x, y)
        }
        ctx.lineTo(w + 50, h); ctx.lineTo(-50, h); ctx.closePath()
        ctx.fillStyle = g; ctx.fill()
      })

      // film grain: one pre-made tile, offset every frame so it crawls
      ctx.save()
      ctx.globalCompositeOperation = 'screen'; ctx.globalAlpha = st.grainK * 0.5
      ctx.translate((Math.random() * 128) | 0, (Math.random() * 128) | 0)
      ctx.fillStyle = grain; ctx.fillRect(-128, -128, w + 256, h + 256)
      ctx.restore()
      ctx.fillStyle = scan; ctx.fillRect(0, 0, w, h)
      // vignette
      const v = ctx.createRadialGradient(w / 2, h / 2, w * 0.2, w / 2, h / 2, w * 0.9)
      v.addColorStop(0, 'rgba(0,0,0,0)'); v.addColorStop(0.5, 'rgba(0,0,0,0.12)')
      v.addColorStop(0.8, 'rgba(0,0,0,0.24)'); v.addColorStop(1, 'rgba(0,0,0,0.4)')
      ctx.fillStyle = v; ctx.fillRect(0, 0, w, h)
      // occasional dust
      if (Math.random() < 0.02) {
        for (let i = 0, n = 1 + ((Math.random() * 5) | 0); i < n; i++) {
          ctx.fillStyle = `rgba(255,255,255,${Math.random() * 0.3})`
          ctx.beginPath(); ctx.arc(Math.random() * w, Math.random() * h, Math.random() * 2 + 0.5, 0, Math.PI * 2); ctx.fill()
        }
      }
      if (!reduced && visible) raf = requestAnimationFrame(frame)
    }

    const start = () => { if (!raf && !reduced) raf = requestAnimationFrame(frame) }
    const stop = () => { cancelAnimationFrame(raf); raf = 0 }
    const onVis = () => { visible = !document.hidden; visible ? start() : stop() }
    const ro = new ResizeObserver(resize); ro.observe(canvas.parentElement)
    document.addEventListener('visibilitychange', onVis)
    resize()
    if (reduced) frame(); else start()
    return () => { stop(); ro.disconnect(); document.removeEventListener('visibilitychange', onVis) }
  }, [])

  return (
    <div aria-hidden="true" style={{ position: 'fixed', inset: 0, zIndex: 0, background: '#000', pointerEvents: 'none' }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
      {/* scrim: the controls and the roll sit on top, text first */}
      <div style={{ position: 'absolute', inset: 0,
        background: 'linear-gradient(to bottom, rgba(0,0,0,0.35), rgba(0,0,0,0.55) 45%, rgba(0,0,0,0.55) 60%, rgba(0,0,0,0.35))' }} />
    </div>
  )
}
