import { useEffect, useState, useRef } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import SonicWaveform from '../components/SonicWaveform'
import StageCarousel from '../components/StageCarousel'

// Stable ref identity on purpose.  StringSpace rebuilds its whole WebGL scene
// when this prop changes, and an inline {current: x} is a NEW object every
// render — on a page that re-renders on every progress tick that would tear
// down and recreate the renderer dozens of times.
const SCENE_POS = { current: 0.55 }

import AudioBars from '../components/AudioBars'
import { countUp, elasticPop } from '../utils/animefx'
import {
  transcribeAudio, separateStems, segmentAudio,
  transformArrangement, renderPreviews,
  getTranscriptionAudioUrl, getTransformedAudioUrl,
} from '../utils/api'

const STEPS = [
  { key: 'upload',     label: 'Upload',     detail: 'Audio received and converted to WAV',                     sym: '▲', color: '#f59e0b' },
  { key: 'separate',   label: 'Analyse',    detail: 'Checking whether this is solo piano or a full song…', sym: '⬡', color: '#06b6d4' },
  { key: 'transcribe', label: 'Transcribe', detail: 'Hearing the notes with our spectral transformer…',   sym: '◈', color: '#a78bfa' },
  { key: 'segment',    label: 'Chords',     detail: 'Finding the beat grid and chords…',           sym: '▪', color: '#34d399' },
  { key: 'transform',  label: 'Arrange',    detail: 'Fitting the arrangement to your keyboard and hands…',    sym: '▶', color: '#f87171' },
  { key: 'previews',   label: 'Previews',   detail: 'Rendering original + arranged audio on a sampled grand…', sym: '♫', color: '#fbbf24' },
]

// ── Step pad ───────────────────────────────────────────────────────────────────
function StepPad({ step, index, status }) {
  // status: 'done' | 'active' | 'pending'
  const isDone    = status === 'done'
  const isActive  = status === 'active'
  const isPending = status === 'pending'
  const padRef    = useRef(null)
  const wasDone   = useRef(false)

  // anime.js: one-shot elastic pop + green flash the moment this step
  // flips to done.  Guarded so a re-render of an already-done pad
  // doesn't re-fire it.
  useEffect(() => {
    if (isDone && !wasDone.current) {
      wasDone.current = true
      elasticPop(padRef.current, '#22c55e')
    }
  }, [isDone])

  const color = isDone ? '#22c55e' : isActive ? step.color : 'rgba(255,255,255,0.08)'
  const textColor = isDone ? '#22c55e' : isActive ? step.color : 'rgba(255,255,255,0.22)'

  return (
    <motion.div
      ref={padRef}
      className="relative flex items-center gap-4 p-4 rounded-2xl"
      style={{
        background: isActive
          ? `linear-gradient(135deg, ${step.color}12, ${step.color}05)`
          : isDone
          ? 'rgba(34,197,94,0.04)'
          : 'rgba(255,255,255,0.015)',
        border: `1px solid ${isActive ? step.color + '40' : isDone ? 'rgba(34,197,94,0.2)' : 'rgba(255,255,255,0.05)'}`,
        transition: 'all 0.35s ease',
      }}
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: isPending ? 0.4 : 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.06 }}
    >
      {/* Pad button visual */}
      <motion.div
        style={{
          width: 48, height: 48,
          borderRadius: 12,
          flexShrink: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: isDone
            ? 'rgba(34,197,94,0.12)'
            : isActive
            ? `${step.color}18`
            : 'rgba(255,255,255,0.03)',
          border: `1px solid ${isDone ? 'rgba(34,197,94,0.35)' : isActive ? step.color + '45' : 'rgba(255,255,255,0.06)'}`,
          boxShadow: isActive
            ? `0 0 16px ${step.color}30, 0 0 32px ${step.color}15`
            : isDone
            ? '0 0 12px rgba(34,197,94,0.2)'
            : 'none',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Shimmer on active */}
        {isActive && (
          <motion.div
            style={{
              position: 'absolute',
              inset: 0,
              background: `linear-gradient(90deg, transparent, ${step.color}20, transparent)`,
              backgroundSize: '200% 100%',
            }}
            animate={{ backgroundPosition: ['-200% center', '200% center'] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
          />
        )}

        {isDone ? (
          <motion.svg
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 500, damping: 28 }}
            style={{ width: 16, height: 16 }}
            fill="none" viewBox="0 0 24 24" stroke="#22c55e" strokeWidth={2.5}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </motion.svg>
        ) : isActive ? (
          <motion.span
            style={{ fontSize: 16, color: step.color, fontFamily: 'ui-monospace,monospace', lineHeight: 1, position: 'relative' }}
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          >
            {step.sym}
          </motion.span>
        ) : (
          <span style={{ fontSize: 14, color: 'rgba(255,255,255,0.15)', fontFamily: 'ui-monospace,monospace' }}>
            {step.sym}
          </span>
        )}
      </motion.div>

      {/* Text */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <p style={{
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: '0.04em',
            color: textColor,
            transition: 'color 0.3s',
          }}>
            {step.label.toUpperCase()}
          </p>
          {isDone && (
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              style={{ fontSize: 9, color: '#22c55e', fontFamily: 'ui-monospace,monospace', letterSpacing: '0.1em', flexShrink: 0 }}
            >
              DONE
            </motion.span>
          )}
        </div>

        <AnimatePresence>
          {isActive && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', marginTop: 4, letterSpacing: '0.02em' }}
            >
              {step.detail}
            </motion.p>
          )}
        </AnimatePresence>

        {/* Progress bar */}
        <div className="mt-2.5 h-0.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
          <motion.div
            className="h-full rounded-full"
            style={{
              background: isDone
                ? '#22c55e'
                : isActive
                ? `linear-gradient(90deg, ${step.color}, ${step.color}80)`
                : 'transparent',
              boxShadow: isDone
                ? '0 0 6px rgba(34,197,94,0.5)'
                : isActive
                ? `0 0 6px ${step.color}60`
                : 'none',
            }}
            animate={{ width: isDone ? '100%' : isActive ? '60%' : '0%' }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        </div>
      </div>
    </motion.div>
  )
}

// ── Main processing page ───────────────────────────────────────────────────────
export default function ProcessingPage() {
  const { sessionId } = useParams()
  const { state }     = useLocation()
  const navigate      = useNavigate()

  const preset = state?.preset || { octaves: 5, components: 16, color: '#f59e0b', label: 'Standard' }

  const [currentStep, setCurrentStep] = useState(0)
  const [error,       setError]       = useState(null)
  const ran    = useRef(false)
  const pctRef = useRef(null)

  useEffect(() => {
    if (ran.current) return
    ran.current = true
    runPipeline()
  }, [])

  async function runPipeline() {
    try {
      setCurrentStep(1); await separateStems(sessionId)
      setCurrentStep(2); await transcribeAudio(sessionId)
      setCurrentStep(3); await segmentAudio(sessionId)
      setCurrentStep(4); await transformArrangement(sessionId, preset.octaves, preset.components)

      // Render ONLY the original track before landing on Results — it is the
      // default view.  The Salamander pass is the single largest slice of the
      // wait (38% of upload-to-result, measured), and rendering both up front
      // made the user wait for a track they cannot listen to simultaneously.
      setCurrentStep(5); await renderPreviews(sessionId, false, 'original_arrangement')

      // Warm the browser cache for the one that exists, so first-click on
      // Listen is a cache hit rather than a download.  Failure is silent —
      // playback still works, it just streams.
      await fetch(getTranscriptionAudioUrl(sessionId), { cache: 'reload' }).catch(() => {})

      // The transformed arrangement renders in the background while the user
      // is already on Results.  Not awaited: by the time they switch to it
      // it is normally ready, and if not the endpoint serves it lazily.
      renderPreviews(sessionId, false, 'transformed')
        .then(() => fetch(getTransformedAudioUrl(sessionId), { cache: 'reload' }))
        .catch(() => {})

      // Fidelity scoring is off for now: it compared spectrograms of the song and
      // the piano, which is not how arrangements are judged at the moment, and
      // it added ~40 s before Results opened.  /score still exists.
      setCurrentStep(6)
      setTimeout(() => navigate(`/results/${sessionId}`), 900)
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Pipeline failed.')
    }
  }

  const overallPct = Math.round((currentStep / STEPS.length) * 100)

  // anime.js: tween the displayed % between step values instead of jumping
  useEffect(() => {
    countUp(pctRef.current, overallPct, { duration: 900, suffix: '%' })
  }, [overallPct])

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-4 py-24"
         style={{ background: 'var(--ct-ink)' }}>
      {/* Same 3D field as the landing page, parked deep in the run so the
          scene reads as continuous across the navigation rather than the
          user arriving somewhere else. */}
      <SonicWaveform accent="#f59e0b" progressRef={SCENE_POS} />

      <motion.div
        className="relative z-10 w-full"
        style={{ maxWidth: 480 }}
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
      >
        {/* Header glass panel */}
        <div
          className="p-6 mb-4"
          style={{
            // Borderless: the panel is a region of the space, not a card
            // sitting on top of it.  A single hairline UNDER the content
            // gives it a baseline without boxing it in.
            background: 'transparent',
            borderBottom: '1px solid var(--ct-line-soft)',
          }}
        >
          {/* Waveform + title row */}
          <div className="flex items-center gap-4 mb-5 min-w-0">
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2 mb-0.5 min-w-0">
                <h2 style={{ fontSize: 16, fontWeight: 700, color: '#fff', letterSpacing: '0.02em', flexShrink: 0 }}>
                  Processing
                </h2>
                <span style={{
                  fontSize: 10,
                  fontFamily: 'ui-monospace,monospace',
                  color: `${preset.color || '#f59e0b'}bb`,
                  letterSpacing: '0.1em',
                  flexShrink: 0,
                }}>
                  {preset.label?.toUpperCase()}
                </span>
              </div>
              {state?.filename && (
                <p style={{ fontSize: 11, color: 'rgba(255,255,255,0.28)', letterSpacing: '0.02em' }}
                  className="truncate">
                  {state.filename}
                </p>
              )}
            </div>
            <div className="flex-shrink-0">
              <AudioBars color={preset.color || '#22c55e'} height={28} />
            </div>
          </div>

          {/* Overall progress bar */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{
                  background: `linear-gradient(90deg, ${preset.color || '#f59e0b'}, ${preset.color || '#d97706'})`,
                  boxShadow: overallPct > 0 ? `0 0 8px ${preset.color || '#f59e0b'}60` : 'none',
                }}
                animate={{ width: `${overallPct}%` }}
                transition={{ ease: 'easeOut', duration: 0.55 }}
              />
            </div>
            <span ref={pctRef} style={{ fontSize: 12, fontWeight: 700, color: preset.color || '#f59e0b', fontFamily: 'ui-monospace,monospace', flexShrink: 0 }}>
              0%
            </span>
          </div>
        </div>

        {/* Stage cluster.  The stacked rectangular cards read as a build log;
            a cluster reads as one machine with parts lighting up.  The active
            stage's detail moves into a single spotlight line below rather than
            every row carrying its own copy. */}
        <StageCarousel
          items={STEPS.map((step, i) => ({
            key: step.key, label: step.label, done: i < currentStep,
          }))}
          activeIndex={Math.min(currentStep, STEPS.length - 1)}
          accent={STEPS[Math.min(currentStep, STEPS.length - 1)]?.color}
        />

        <div style={{ minHeight: 44, textAlign: 'center', marginTop: 22 }}>
          <p style={{
            fontSize: 12, color: 'rgba(255,255,255,0.42)', lineHeight: 1.5,
            maxWidth: 380, margin: '0 auto',
          }}>
            {STEPS[Math.min(currentStep, STEPS.length - 1)]?.detail}
          </p>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-5 p-4 rounded-2xl"
              style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.18)' }}
            >
              <p style={{ fontSize: 13, fontWeight: 600, color: '#f87171', marginBottom: 4 }}>
                Something went wrong
              </p>
              <p style={{ fontSize: 11, color: 'rgba(239,68,68,0.6)' }}>{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
