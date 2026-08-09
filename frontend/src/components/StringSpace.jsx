import { useEffect, useRef } from 'react'

/**
 * StringSpace — the instrument you scroll through.
 *
 * A WebGL field of vibrating strings receding into depth, with the camera
 * travelling along them as the page scrolls.  This is the piece I argued
 * against when three.js first came up, on the grounds that a flat workstation
 * aesthetic did not need 3D.  That was true of the old design.  This one is
 * built around depth, so the objection no longer applies — the strings have
 * to recede for the scroll to feel like travelling rather than sliding.
 *
 * The physics is the same standing-wave model as the old hero StringField,
 * lifted into three dimensions: displacement is a sum over partials p of
 * sin(p·π·x/L) in space times sin(ω_p·t) in time, so endpoints stay pinned
 * and the harmonic series is visible in the shapes.  Struck strings decay,
 * with upper partials dying first — bright to round, like a real note.
 *
 * Deliberate constraints
 * ----------------------
 * - three is loaded by DYNAMIC IMPORT so its ~600KB never blocks first paint.
 *   That was the other half of my original objection and it is a real cost;
 *   code-splitting is the answer to it, not pretending it is free.
 * - The renderer is capped at devicePixelRatio 2 and pauses when the tab is
 *   hidden, because this runs for the entire page rather than one section.
 * - If WebGL is unavailable the component renders nothing and the page is
 *   still completely usable.  Nothing here carries meaning that is not also
 *   in the DOM.
 */

const N_STRINGS = 26
const SEGMENTS = 90
const PARTIALS = 3

// Scene accents, one per scroll act. Near-monochrome page, one luminous
// colour at a time — the opposite of the old three-accent workstation.
const ACCENTS = [
  [0.98, 0.72, 0.29],   // warm gold      — overture
  [0.55, 0.78, 1.00],   // cold blue      — console
  [0.62, 0.98, 0.80],   // pale green     — chain
  [0.86, 0.66, 1.00],   // violet         — coda
]

function lerp(a, b, t) { return a + (b - a) * t }

export default function StringSpace({ progressRef }) {
  const mountRef = useRef(null)
  const stateRef = useRef({ scroll: 0, target: 0, mx: 0, my: 0, strike: 0 })

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    let disposed = false
    let cleanup = () => {}

    // Dynamic import: three never lands in the initial bundle.
    import('three').then((THREE) => {
      if (disposed) return

      let renderer
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      } catch {
        return                       // no WebGL — page works without this
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(mount.clientWidth, mount.clientHeight)
      mount.appendChild(renderer.domElement)

      const scene = new THREE.Scene()
      scene.fog = new THREE.FogExp2(0x05060a, 0.055)

      const camera = new THREE.PerspectiveCamera(
        58, mount.clientWidth / mount.clientHeight, 0.1, 200,
      )
      camera.position.set(0, 0, 12)

      // ── strings ────────────────────────────────────────────────────────
      // Each string is a polyline along X, placed at its own Y and Z. Lower
      // strings sit lower and nearer, so the field opens out like a
      // soundboard seen from inside the instrument.
      const strings = []
      const stringGroup = new THREE.Group()
      for (let i = 0; i < N_STRINGS; i++) {
        const u = i / (N_STRINGS - 1)
        const positions = new Float32Array(SEGMENTS * 3)
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        const mat = new THREE.LineBasicMaterial({
          color: new THREE.Color(1, 1, 1),
          transparent: true,
          opacity: 0.18 + 0.5 * (1 - u),
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        })
        const line = new THREE.Line(geo, mat)
        stringGroup.add(line)
        strings.push({
          u,
          y: (u - 0.5) * 9,
          z: -u * 46,
          len: 26 - u * 8,
          cycles: 1 + u * 1.7,
          speed: 0.5 + u * 1.6,
          phase: i * 0.73,
          excite: Math.random() * 0.15,
          positions,
          geo,
          mat,
        })
      }
      scene.add(stringGroup)

      // ── dust ───────────────────────────────────────────────────────────
      // The "sparkles" idea, in depth: points scattered through the volume so
      // travelling forward produces real parallax rather than a flat drift.
      const N_DUST = 1400
      const dustPos = new Float32Array(N_DUST * 3)
      for (let i = 0; i < N_DUST; i++) {
        dustPos[i * 3] = (Math.random() - 0.5) * 48
        dustPos[i * 3 + 1] = (Math.random() - 0.5) * 24
        dustPos[i * 3 + 2] = -Math.random() * 60
      }
      const dustGeo = new THREE.BufferGeometry()
      dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3))
      const dustMat = new THREE.PointsMaterial({
        size: 0.055,
        transparent: true,
        opacity: 0.75,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
      const dust = new THREE.Points(dustGeo, dustMat)
      scene.add(dust)

      // ── interaction ────────────────────────────────────────────────────
      const st = stateRef.current
      const onMove = (e) => {
        st.mx = (e.clientX / window.innerWidth) * 2 - 1
        st.my = -((e.clientY / window.innerHeight) * 2 - 1)
      }
      const onDown = () => {
        st.strike = 1
        // Strike the strings nearest the pointer height.
        for (const s of strings) {
          const d = Math.abs(s.y / 4.5 - st.my)
          s.excite = Math.min(2.2, s.excite + Math.exp(-d * d * 2.2) * 1.6)
        }
      }
      window.addEventListener('pointermove', onMove, { passive: true })
      window.addEventListener('pointerdown', onDown, { passive: true })

      const onResize = () => {
        if (!mount.clientWidth) return
        camera.aspect = mount.clientWidth / mount.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, mount.clientHeight)
      }
      window.addEventListener('resize', onResize)

      const tmpColor = new THREE.Color()
      let raf = 0
      let last = performance.now()
      let t = 0
      let visible = true
      const onVis = () => { visible = !document.hidden; last = performance.now() }
      document.addEventListener('visibilitychange', onVis)

      const frame = (now) => {
        raf = requestAnimationFrame(frame)
        if (!visible) return
        const dt = Math.min(0.05, (now - last) / 1000)
        last = now
        t += dt

        st.target = progressRef?.current ?? 0
        st.scroll += (st.target - st.scroll) * Math.min(1, dt * 3.2)
        st.strike *= Math.pow(0.06, dt)

        // Accent for the current act, interpolated so colour never snaps.
        const x = Math.max(0, Math.min(ACCENTS.length - 1.001, st.scroll * (ACCENTS.length - 1)))
        const ai = Math.floor(x)
        const af = x - ai
        const c0 = ACCENTS[ai], c1 = ACCENTS[ai + 1] ?? ACCENTS[ai]
        const R = lerp(c0[0], c1[0], af)
        const G = lerp(c0[1], c1[1], af)
        const B = lerp(c0[2], c1[2], af)

        // Camera travels INTO the field; the small cursor lean gives the
        // volume parallax so it reads as a space rather than a backdrop.
        camera.position.z = 12 - st.scroll * 40
        camera.position.x += (st.mx * 1.7 - camera.position.x) * Math.min(1, dt * 2)
        camera.position.y += (st.my * 1.1 - camera.position.y) * Math.min(1, dt * 2)
        camera.lookAt(0, 0, camera.position.z - 14)

        for (const s of strings) {
          s.excite *= Math.pow(0.3, dt)
          const amp = 0.05 + s.excite * 1.05
          const half = s.len / 2
          for (let k = 0; k < SEGMENTS; k++) {
            const f = k / (SEGMENTS - 1)
            let d = 0
            for (let p = 1; p <= PARTIALS; p++) {
              const pw = p === 1 ? 1 : Math.pow(Math.max(s.excite, 0.14), p - 1)
              d += Math.sin(p * Math.PI * f) *
                   Math.sin(t * s.speed * p * 1.6 + s.phase * p) * (amp / p) * pw
            }
            s.positions[k * 3] = -half + f * s.len
            s.positions[k * 3 + 1] = s.y + d
            s.positions[k * 3 + 2] = s.z
          }
          s.geo.attributes.position.needsUpdate = true
          const glow = Math.min(1, s.excite)
          tmpColor.setRGB(
            lerp(R * 0.55, 1, glow),
            lerp(G * 0.55, 1, glow),
            lerp(B * 0.55, 1, glow),
          )
          s.mat.color.copy(tmpColor)
          s.mat.opacity = (0.14 + 0.42 * (1 - s.u)) + glow * 0.5
        }

        dustMat.color.setRGB(R, G, B)
        dustMat.opacity = 0.45 + st.strike * 0.4
        dust.rotation.z += dt * 0.012

        renderer.render(scene, camera)
      }
      raf = requestAnimationFrame(frame)

      cleanup = () => {
        cancelAnimationFrame(raf)
        window.removeEventListener('pointermove', onMove)
        window.removeEventListener('pointerdown', onDown)
        window.removeEventListener('resize', onResize)
        document.removeEventListener('visibilitychange', onVis)
        strings.forEach((s) => { s.geo.dispose(); s.mat.dispose() })
        dustGeo.dispose(); dustMat.dispose()
        renderer.dispose()
        if (renderer.domElement.parentNode === mount) {
          mount.removeChild(renderer.domElement)
        }
      }
    })

    return () => { disposed = true; cleanup() }
  }, [progressRef])

  return (
    <div
      ref={mountRef}
      aria-hidden
      style={{
        position: 'fixed', inset: 0, zIndex: 0,
        pointerEvents: 'none',
        background:
          'radial-gradient(ellipse 120% 80% at 50% 30%, #0b0d14 0%, #05060a 60%, #030407 100%)',
      }}
    />
  )
}
