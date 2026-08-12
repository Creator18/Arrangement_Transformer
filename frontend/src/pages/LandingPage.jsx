import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, ChevronDown, AudioWaveform } from 'lucide-react'
import SonicWaveform from '../components/SonicWaveform'
import Act from '../components/Act'
import PipelineScrolly from '../components/PipelineScrolly'
import { uploadAudio, getWarmupStatus } from '../utils/api'
import { scrollToTarget } from '../utils/smoothscroll'
import { initScrollSnap } from '../utils/scrollsnap'

/**
 * LandingPage — rebuilt as a scroll story.
 *
 * The previous page was a centred column of glass cards over a flat canvas.
 * This one is four full-viewport ACTS over a simplex-noise ribbon field,
 * with one accent colour at a time and type doing the work that decoration
 * used to.  Functionally identical: same eight presets, same drag-and-drop
 * upload, same progress, same warm-up lamp, same navigation payload.
 */

const ACTS = [
  { id: 'overture', label: 'Overture', accent: '#fbb84a' },
  { id: 'console',  label: 'Console',  accent: '#8cc7ff' },
  { id: 'pipeline', label: 'Chain',    accent: '#9ef9cc' },
  { id: 'coda',     label: 'Coda',     accent: '#dba9ff' },
]

const ACCEPTED = '.mp3,.wav,.m4a,.ogg,.flac,.aiff'

// Unchanged from the old page — these are real backend configs and the
// object travels with the session, so its shape must not drift.
const PRESETS = [
  { id: 'standard',  label: 'Standard',  octaves: 5, components: 16, color: '#fbb84a', desc: '5 oct · balanced'   },
  { id: 'wide',      label: 'Wide',      octaves: 6, components: 16, color: '#8cc7ff', desc: '6 oct · expressive' },
  { id: 'full',      label: 'Concert',   octaves: 7, components: 20, color: '#dba9ff', desc: '7 oct · full range' },
  { id: 'minimal',   label: 'Minimal',   octaves: 4, components: 8,  color: '#9ef9cc', desc: '4 oct · clean'      },
  { id: 'jazz',      label: 'Jazz',      octaves: 5, components: 24, color: '#ff9b8a', desc: '5 oct · rich NMF'   },
  { id: 'classical', label: 'Classical', octaves: 6, components: 20, color: '#7fe3e0', desc: '6 oct · detailed'   },
  { id: 'beginner',  label: 'Beginner',  octaves: 4, components: 12, color: '#c6e77f', desc: '4 oct · simplified' },
  { id: 'dense',     label: 'Dense',     octaves: 5, components: 32, color: '#f2a0e0', desc: '5 oct · max NMF'    },
]

function ActIndex({ acts, current }) {
  return (
    <nav className="ct-index" aria-label="Sections">
      {acts.map((a, i) => (
        <button
          key={a.id}
          data-on={i === current}
          onClick={() => scrollToTarget(a.id)}
          title={a.label}
        >
          <span className="tick" />
          {a.label}
        </button>
      ))}
    </nav>
  )
}

export default function LandingPage() {
  const [dragging, setDragging]     = useState(false)
  const [uploading, setUploading]   = useState(false)
  const [progress, setProgress]     = useState(0)
  const [error, setError]           = useState(null)
  const [preset, setPreset]         = useState(PRESETS[0])
  const [engineReady, setEngineReady] = useState(null)
  const [act, setAct]               = useState(0)

  const inputRef  = useRef(null)
  const navigate  = useNavigate()
  // Scroll progress for the ribbon field. A ref, not state: the canvas reads
  // it every frame and re-rendering React 60x a second for a number nothing
  // in the DOM displays would be pure waste.
  const progressRef = useRef(0)

  useEffect(() => {
    let ticking = false
    const read = () => {
      ticking = false
      const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight)
      const p = window.scrollY / max
      progressRef.current = p
      // Which act is on screen, from GEOMETRY not from a fraction of the
      // document.  Fractions assume the acts are evenly spaced; the pipeline
      // is five viewports where the others are one, so at 25% scrolled the
      // rail said CONSOLE while the pipeline filled the screen.  Same mistake
      // ScrollStage had, caught here by screenshotting the page.
      const mid = window.scrollY + window.innerHeight / 2
      let idx = 0, best = Infinity
      ACTS.forEach((a, i) => {
        const el = document.getElementById(a.id)
        if (!el) return
        const r = el.getBoundingClientRect()
        const top = r.top + window.scrollY
        const d = mid < top ? top - mid
                : mid > top + r.height ? mid - (top + r.height) : 0
        if (d < best) { best = d; idx = i }
      })
      setAct((prev) => (prev === idx ? prev : idx))
    }
    const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(read) } }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    read()
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])

  // One wheel gesture advances one act.  The pipeline section opts out via
  // data-free-scroll so its five-stage narrative still scrubs.
  useEffect(() => initScrollSnap(ACTS.map((a) => a.id)), [])

  // One accent for the whole page, following the act. Everything that uses
  // --ct-accent moves together, which is what keeps it feeling like one
  // instrument changing state rather than four differently-themed pages.
  useEffect(() => {
    document.documentElement.style.setProperty('--ct-accent', ACTS[act].accent)
  }, [act])

  // Warm the backend while the user reads — unchanged behaviour.
  useEffect(() => {
    let alive = true
    let timer = null
    const poll = async () => {
      try {
        const st = await getWarmupStatus()
        if (!alive) return
        setEngineReady(!!st.ready)
        if (!st.ready) timer = setTimeout(poll, 1500)
      } catch {
        if (alive) { setEngineReady(false); timer = setTimeout(poll, 4000) }
      }
    }
    poll()
    return () => { alive = false; if (timer) clearTimeout(timer) }
  }, [])

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setError(null); setUploading(true); setProgress(0)
    try {
      const result = await uploadAudio(file, setProgress)
      navigate(`/processing/${result.session_id}`, {
        state: { filename: file.name, duration: result.duration_seconds, preset },
      })
    } catch (e) {
      setError(e?.response?.data?.detail || 'Upload failed. Please try again.')
      setUploading(false)
    }
  }, [navigate, preset])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  return (
    <div style={{ position: 'relative', background: 'var(--ct-ink)' }}>
      {/* The backdrop takes the current act's accent, so scrolling through the
          page drifts the whole field from amber to blue to mint to violet
          instead of holding one colour behind four different sections. */}
      <SonicWaveform progressRef={progressRef} accent={ACTS[act]?.accent} />
      <ActIndex acts={ACTS} current={act} />

      {/* ─── I · Overture ─────────────────────────────────────────────── */}
      <Act id="overture" label="I · Overture" minH="100vh">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            padding: '6px 14px', marginBottom: 26, borderRadius: 999,
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.12)',
            backdropFilter: 'blur(6px)',
          }}
        >
          <AudioWaveform size={15} style={{ color: ACTS[0].accent }} />
          <span className="ct-mono" style={{ letterSpacing: '0.06em' }}>
            Audio → playable arrangement
          </span>
        </motion.div>

        <h1 className="ct-display" style={{ marginBottom: 32 }}>
          Any recording.<br />
          Any keyboard.<br />
          <em>A playable score.</em>
        </h1>
        <p className="ct-lead" style={{ marginBottom: 42 }}>
          Audio goes in. What comes out is an arrangement a human hand can
          actually play — every note traceable to something the recording
          really did.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
          <button className="ct-cta" onClick={() => scrollToTarget('console')}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              Begin <ArrowRight size={17} />
            </span>
          </button>
          <span className="ct-mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
            <ChevronDown size={14} className="ct-nudge" /> Scroll
          </span>
        </div>
      </Act>

      {/* ─── II · Console ─────────────────────────────────────────────── */}
      <Act id="console" label="II · Console" minH="100vh">
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(260px, 1fr) minmax(300px, 1.05fr)',
          gap: 'clamp(28px, 5vw, 76px)',
          alignItems: 'start',
        }}>
          {/* left — presets */}
          <div>
            <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 18 }}>
              <h2 className="ct-display" style={{ fontSize: 'clamp(26px, 3vw, 44px)' }}>Voicing</h2>
              <span className="ct-mono">{preset.octaves} oct · {preset.components} nmf</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  className="ct-key"
                  data-on={preset.id === p.id}
                  onClick={() => setPreset(p)}
                  style={preset.id === p.id ? { '--ct-accent': p.color } : undefined}
                >
                  <div style={{
                    fontSize: 12, fontWeight: 600, letterSpacing: '0.02em',
                    color: preset.id === p.id ? p.color : 'var(--ct-text)',
                    marginBottom: 4,
                  }}>
                    {p.label}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--ct-faint)', fontFamily: 'ui-monospace,monospace' }}>
                    {p.desc}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* right — upload */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
              <h2 className="ct-display" style={{ fontSize: 'clamp(26px, 3vw, 44px)' }}>Source</h2>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  width: 5, height: 5, borderRadius: '50%',
                  background: engineReady === true ? '#9ef9cc'
                            : engineReady === false ? 'var(--ct-accent)' : 'rgba(255,255,255,0.2)',
                  boxShadow: engineReady === true ? '0 0 8px #9ef9cc' : 'none',
                  transition: 'background .4s, box-shadow .4s',
                }} />
                <span className="ct-mono">
                  {engineReady === true ? 'Ready' : engineReady === false ? 'Warming' : '—'}
                </span>
              </span>
            </div>

            {!uploading ? (
              <div
                onDrop={onDrop}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                className="ct-panel"
                style={{
                  padding: 'clamp(28px, 4vw, 54px) 28px',
                  borderStyle: 'dashed',
                  borderColor: dragging ? 'var(--ct-accent)' : 'var(--ct-line)',
                  background: dragging ? 'color-mix(in srgb, var(--ct-accent) 7%, transparent)' : undefined,
                  textAlign: 'center',
                  transition: 'border-color .25s, background .25s',
                }}
              >
                <div className="ct-mono" style={{ marginBottom: 14 }}>Drop audio here</div>
                <div style={{ fontSize: 11, color: 'var(--ct-faint)', marginBottom: 26 }}>
                  MP3 · WAV · M4A · OGG · FLAC · AIFF
                </div>
                <button className="ct-cta" onClick={() => inputRef.current?.click()}>
                  <span>Choose file</span>
                </button>
                <input
                  ref={inputRef}
                  type="file"
                  accept={ACCEPTED}
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files[0])}
                />
              </div>
            ) : (
              <div className="ct-panel" style={{ padding: 34 }}>
                <div className="ct-mono" style={{ marginBottom: 16 }}>Uploading</div>
                <div style={{ height: 1, background: 'var(--ct-line)', position: 'relative', marginBottom: 12 }}>
                  <motion.div
                    style={{ position: 'absolute', inset: 0, transformOrigin: 'left', background: 'var(--ct-accent)' }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: progress / 100 }}
                    transition={{ ease: 'easeOut' }}
                  />
                </div>
                <div style={{
                  fontFamily: 'ui-monospace,monospace', fontSize: 32, fontWeight: 300,
                  color: 'var(--ct-accent)',
                }}>
                  {progress}<span style={{ fontSize: 14, color: 'var(--ct-faint)' }}>%</span>
                </div>
              </div>
            )}

            <AnimatePresence>
              {error && (
                <motion.p
                  initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  style={{ marginTop: 16, fontSize: 12, color: '#ff9b8a' }}
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      </Act>

      {/* ─── III · Chain — the pipeline scroll story ───────────────────── */}
      <PipelineScrolly id="pipeline" />

      {/* ─── IV · Coda ────────────────────────────────────────────────── */}
      <Act id="coda" label="IV · Coda" minH="100vh">
        <h2 className="ct-display" style={{ marginBottom: 30 }}>
          The reduction may<br />only <em>subtract</em>.
        </h2>
        <p className="ct-lead" style={{ marginBottom: 34 }}>
          A smaller keyboard means fewer notes, never different ones. Onsets
          stay where the performance put them, pitch classes stay what they
          were, and anything that cannot be reached is released rather than
          rewritten — held by the pedal, exactly as a player would.
        </p>
        <div style={{ display: 'flex', gap: 'clamp(20px, 4vw, 64px)', flexWrap: 'wrap' }}>
          {[
            ['0.00%', 'onsets moved'],
            ['0.00%', 'pitch added'],
            ['10', 'keys maximum'],
            ['14', 'semitone reach'],
          ].map(([v, k]) => (
            <div key={k}>
              <div style={{
                fontSize: 'clamp(26px, 3.4vw, 46px)', fontWeight: 300,
                letterSpacing: '-0.03em', color: 'var(--ct-accent)',
              }}>
                {v}
              </div>
              <div className="ct-mono" style={{ marginTop: 4 }}>{k}</div>
            </div>
          ))}
        </div>
      </Act>
    </div>
  )
}
