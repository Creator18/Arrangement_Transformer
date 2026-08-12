import { useEffect, useRef } from 'react'

/**
 * ShaderField — interference rings on the GPU.  NOT CURRENTLY MOUNTED.
 *
 * This was the processing-page backdrop for one iteration.  It is kept but
 * unused, because three different backdrops across three pages is not a design
 * — SonicWaveform now runs on all of them, varying only by accent.  Swapping
 * this back in is one import and one tag in ProcessingPage; nothing else
 * depends on it, and three.js is lazy-imported so an unmounted ShaderField
 * costs the bundle nothing.
 *
 * Used only while the pipeline is running, and the choice is about meaning
 * rather than decoration.  The fragment shader draws rings radiating from the
 * centre with the three colour channels offset in phase, so the field reads as
 * something propagating outward — which is what the page is actually waiting
 * on.  It also costs the CPU nothing, and this page is rendered while the
 * backend is saturating the CPU with Demucs and NMF.  A canvas backdrop here
 * would be competing with the work the user is waiting for; a shader is not.
 *
 * Adapted rather than pasted:
 *
 *   - the original sized itself to `container.clientWidth/Height` once and on
 *     window resize.  A fixed backdrop inside a scroll container needs a
 *     ResizeObserver, and the renderer needs the DPR clamped or a retina
 *     screen renders four times the pixels for no visible gain.
 *   - it never stopped.  This pauses when the tab is hidden, which matters on
 *     a page people leave open for minutes while a track processes, and draws
 *     a single frame under prefers-reduced-motion.
 *   - WebGL failure was unhandled.  If the context cannot be created this
 *     renders a plain backdrop instead of throwing — a decorative layer must
 *     never be able to take down the page that reports pipeline progress.
 *
 * three.js is a real cost here (a few hundred KB) and it is loaded lazily, so
 * the landing page and the results page never pay for it.
 */

const MAX_DPR = 1.5     // rings are smooth gradients; extra pixels buy nothing

const VERT = `
  void main() { gl_Position = vec4(position, 1.0); }
`

const FRAG = `
  precision highp float;
  uniform vec2 resolution;
  uniform float time;
  uniform float intensity;

  void main(void) {
    vec2 uv = (gl_FragCoord.xy * 2.0 - resolution.xy) / min(resolution.x, resolution.y);
    float t = time * 0.05;
    float lineWidth = 0.002;

    vec3 color = vec3(0.0);
    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < 5; i++) {
        color[j] += lineWidth * float(i * i) /
          abs(fract(t - 0.01 * float(j) + float(i) * 0.01) * 5.0
              - length(uv) + mod(uv.x + uv.y, 0.2));
      }
    }
    gl_FragColor = vec4(color * intensity, 1.0);
  }
`

export default function ShaderField({ intensity = 1.7 }) {
  const hostRef = useRef(null)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    let disposed = false
    let cleanup = () => {}

    // Lazy import so three is not in the initial bundle for pages that never
    // show this.
    import('three').then((THREE) => {
      if (disposed || !hostRef.current) return

      let renderer
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
      } catch {
        return                         // no WebGL — leave the plain backdrop
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DPR))

      const camera = new THREE.Camera()
      camera.position.z = 1
      const scene = new THREE.Scene()
      const geometry = new THREE.PlaneGeometry(2, 2)
      const uniforms = {
        time: { value: 1.0 },
        resolution: { value: new THREE.Vector2() },
        intensity: { value: intensity },
      }
      const material = new THREE.ShaderMaterial({
        uniforms, vertexShader: VERT, fragmentShader: FRAG,
      })
      scene.add(new THREE.Mesh(geometry, material))

      const el = renderer.domElement
      el.style.display = 'block'
      el.style.width = '100%'
      el.style.height = '100%'
      host.appendChild(el)

      const resize = () => {
        const r = host.getBoundingClientRect()
        const w = Math.max(1, Math.floor(r.width))
        const h = Math.max(1, Math.floor(r.height))
        renderer.setSize(w, h, false)
        uniforms.resolution.value.set(el.width, el.height)
      }

      const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      let raf = 0
      let running = false

      const loop = () => {
        uniforms.time.value += 0.05
        renderer.render(scene, camera)
        if (running) raf = requestAnimationFrame(loop)
      }
      const start = () => {
        if (running || reduced) return
        running = true
        raf = requestAnimationFrame(loop)
      }
      const stop = () => { running = false; cancelAnimationFrame(raf); raf = 0 }

      const ro = new ResizeObserver(resize)
      ro.observe(host)
      const onVis = () => (document.hidden ? stop() : start())
      document.addEventListener('visibilitychange', onVis)

      resize()
      if (reduced) renderer.render(scene, camera)
      else start()

      cleanup = () => {
        stop()
        ro.disconnect()
        document.removeEventListener('visibilitychange', onVis)
        if (el.parentNode === host) host.removeChild(el)
        geometry.dispose()
        material.dispose()
        renderer.dispose()
      }
    }).catch(() => {})

    return () => { disposed = true; cleanup() }
  }, [intensity])

  return (
    <div
      aria-hidden="true"
      style={{
        position: 'fixed', inset: 0, zIndex: 0,
        background: '#000', pointerEvents: 'none',
      }}
    >
      <div ref={hostRef} style={{ position: 'absolute', inset: 0 }} />
      {/* Holds the middle back so the step list stays first, but only just.
          At 0.72 in the centre this buried the rings entirely — the shader
          was running, animating and invisible, which is the worst of both:
          full GPU cost, no visual. */}
      <div style={{
        position: 'absolute', inset: 0,
        background:
          'radial-gradient(ellipse 70% 55% at center, rgba(0,0,0,0.62) 0%, rgba(0,0,0,0.3) 45%, transparent 80%)',
      }} />
    </div>
  )
}
