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
      <div className="mt-2 w-full h-1 rounded-full bg-green-100 overflow-hidden">
        <div className="h-full w-full bg-green-400 rounded-full" />
      </div>
    )
  }
  if (active) {
    return <AudioBars color="#0071e3" height={20} className="mt-2" />
  }
  return <div className="mt-2 w-full h-1 rounded-full bg-[#f0f0f3]" />
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
            <span className="text-sm font-medium text-[#0071e3] tabular-nums">{overallPct}%</span>
          </div>
          {state?.filename && (
            <p className="text-sm text-[#6e6e73] mb-2 truncate">{state.filename}</p>
          )}

          {/* Overall bar */}
          <div className="w-full h-1.5 bg-[#e5e5ea] rounded-full overflow-hidden mb-8">
            <motion.div
              className="h-full bg-[#0071e3] rounded-full"
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
                  animate={{ opacity: pending ? 0.4 : 1, x: 0 }}
                  transition={{ duration: 0.4, delay: i * 0.05 }}
                >
                  <div className="flex items-center gap-3">
                    {/* Status icon */}
                    <div className="flex-shrink-0">
                      {done ? (
                        <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                          <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        </div>
                      ) : active ? (
                        <div className="w-5 h-5 rounded-full border-2 border-[#0071e3] flex items-center justify-center">
                          <motion.div
                            className="w-2 h-2 rounded-full bg-[#0071e3]"
                            animate={{ scale: [1, 1.5, 1] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          />
                        </div>
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-[#d2d2d7]" />
                      )}
                    </div>

                    {/* Label + detail */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline justify-between">
                        <p className={`text-sm font-medium ${done ? 'text-[#3a3a3c]' : active ? 'text-[#0071e3]' : 'text-[#6e6e73]'}`}>
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
                            className="text-xs text-[#6e6e73] mt-0.5"
                          >
                            {step.detail}
                          </motion.p>
                        )}
                      </AnimatePresence>

                      {/* Per-step bar */}
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
              className="mt-6 p-4 rounded-xl bg-red-50 border border-red-100"
            >
              <p className="text-sm font-medium text-red-600 mb-1">Something went wrong</p>
              <p className="text-xs text-red-500">{error}</p>
            </motion.div>
          )}
        </GlassCard>
      </motion.div>
    </div>
  )
}
