import { useEffect, useState, useRef } from 'react'
import { useParams, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import GradientOrbs from '../components/GradientOrbs'
import GlassCard from '../components/GlassCard'
import AudioBars from '../components/AudioBars'
import {
  transcribeAudio, separateStems, segmentAudio,
  transformArrangement, scoreFidelity
} from '../utils/api'

const STEPS = [
  { key: 'upload',     label: 'Uploading',               detail: 'Audio file received and converted to WAV' },
  { key: 'transcribe', label: 'Transcribing',             detail: 'Basic Pitch is detecting every note…' },
  { key: 'separate',   label: 'Separating Stems',         detail: 'Spleeter is isolating bass and melody…' },
  { key: 'segment',    label: 'Detecting Segments',       detail: 'Analysing measure structure and star parts…' },
  { key: 'transform',  label: 'Transforming Arrangement', detail: 'Applying priority hierarchy and ghost fills…' },
  { key: 'score',      label: 'Scoring Fidelity',         detail: 'Comparing spectrograms for fidelity…' },
]

function StepProgressBar({ active, done }) {
  if (done) {
    return (
      <div className="mt-2 w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(34,197,94,0.15)' }}>
        <div className="h-full w-full rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 6px rgba(34,197,94,0.5)' }} />
      </div>
    )
  }
  if (active) {
    return <AudioBars color="#22c55e" height={20} className="mt-2" />
  }
  return <div className="mt-2 w-full h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }} />
}

export default function ProcessingPage() {
  const { sessionId } = useParams()
  const { state } = useLocation()
  const navigate = useNavigate()

  const [currentStep, setCurrentStep] = useState(0)
  const [error, setError] = useState(null)
  const ran = useRef(false)

  useEffect(() => {
    if (ran.current) return
    ran.current = true
    runPipeline()
  }, [])

  async function runPipeline() {
    try {
      setCurrentStep(1)
      await transcribeAudio(sessionId)
      setCurrentStep(2)
      await separateStems(sessionId)
      setCurrentStep(3)
      await segmentAudio(sessionId)
      setCurrentStep(4)
      await transformArrangement(sessionId)
      setCurrentStep(5)
      await scoreFidelity(sessionId)
      setCurrentStep(6)
      setTimeout(() => navigate(`/results/${sessionId}`), 800)
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || 'Pipeline failed.')
    }
  }

  const overallPct = Math.round((currentStep / STEPS.length) * 100)

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden">
      <GradientOrbs />

      <motion.div
        className="relative z-10 w-full max-w-lg px-4"
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
      >
        <GlassCard className="p-10">
          {/* Header */}
          <div className="flex items-baseline justify-between mb-1">
            <h2 className="text-xl font-semibold text-gradient">Processing</h2>
            <span className="text-sm font-medium text-amber-400 tabular-nums">{overallPct}%</span>
          </div>
          {state?.filename && (
            <p className="text-sm text-zinc-500 mb-2 truncate">{state.filename}</p>
          )}

          {/* Overall progress bar — amber gradient */}
          <div className="w-full h-1.5 rounded-full overflow-hidden mb-8" style={{ background: 'rgba(255,255,255,0.08)' }}>
            <motion.div
              className="h-full rounded-full"
              style={{
                background: 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)',
                boxShadow: overallPct > 0 ? '0 0 8px rgba(245,158,11,0.5)' : 'none',
              }}
              animate={{ width: `${overallPct}%` }}
              transition={{ ease: 'easeOut', duration: 0.5 }}
            />
          </div>

          {/* Steps */}
          <div className="space-y-5">
            {STEPS.map((step, i) => {
              const done = i < currentStep
              const active = i === currentStep
              const pending = i > currentStep

              return (
                <motion.div
                  key={step.key}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: pending ? 0.35 : 1, x: 0 }}
                  transition={{ duration: 0.4, delay: i * 0.05 }}
                >
                  <div className="flex items-center gap-3">
                    {/* Status icon */}
                    <div className="flex-shrink-0">
                      {done ? (
                        <div
                          style={{
                            width: 20, height: 20,
                            borderRadius: '50%',
                            background: '#22c55e',
                            boxShadow: '0 0 8px rgba(34,197,94,0.5)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                          }}
                        >
                          <svg style={{ width: 12, height: 12 }} fill="none" viewBox="0 0 24 24" stroke="white" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      ) : active ? (
                        <div
                          style={{
                            width: 20, height: 20,
                            borderRadius: '50%',
                            border: '2px solid #f59e0b',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                          }}
                        >
                          <motion.div
                            style={{
                              width: 8, height: 8,
                              borderRadius: '50%',
                              background: '#f59e0b',
                              willChange: 'transform',
                              transformOrigin: '50% 50%',
                            }}
                            animate={{ scale: [1, 1.4, 1] }}
                            transition={{ duration: 1, repeat: Infinity, ease: 'easeInOut' }}
                          />
                        </div>
                      ) : (
                        <div style={{ width: 20, height: 20, borderRadius: '50%', border: '2px solid #3f3f46', flexShrink: 0 }} />
                      )}
                    </div>

                    {/* Label + detail */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline justify-between">
                        <p className={`text-sm font-medium ${
                          done   ? 'text-zinc-300'
                          : active ? 'text-amber-400'
                          : 'text-zinc-600'
                        }`}>
                          {step.label}
                        </p>
                        {done && (
                          <span className="text-[10px] text-green-500 font-medium ml-2 flex-shrink-0">Done</span>
                        )}
                      </div>

                      <AnimatePresence>
                        {active && (
                          <motion.p
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="text-xs text-zinc-500 mt-0.5"
                          >
                            {step.detail}
                          </motion.p>
                        )}
                      </AnimatePresence>

                      <StepProgressBar active={active} done={done} />
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 p-4 rounded-xl"
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
            >
              <p className="text-sm font-medium text-red-400 mb-1">Something went wrong</p>
              <p className="text-xs text-red-500">{error}</p>
            </motion.div>
          )}
        </GlassCard>
      </motion.div>
    </div>
  )
}
