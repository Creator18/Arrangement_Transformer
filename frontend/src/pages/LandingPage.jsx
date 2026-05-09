import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import GradientOrbs from '../components/GradientOrbs'
import GlassCard from '../components/GlassCard'
import AudioBars from '../components/AudioBars'
import { uploadAudio } from '../utils/api'

const ACCEPTED = '.mp3,.wav,.m4a,.ogg,.flac,.aiff'

// Decorative MIDI pad grid — 3 rows × 4 cols
function SoundboardPads() {
  // Each pad has a dot color: green (active), amber (hot), or off
  const pads = [
    'green', 'off',   'amber', 'off',
    'off',   'green', 'off',   'green',
    'amber', 'off',   'green', 'amber',
  ]

  return (
    <div className="grid grid-cols-4 gap-2 w-fit mx-auto" aria-hidden>
      {pads.map((state, i) => (
        <motion.div
          key={i}
          className="w-12 h-12 rounded-lg flex items-end justify-end p-1.5"
          style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04), 0 2px 8px rgba(0,0,0,0.4)',
          }}
          whileHover={{ background: 'rgba(255,255,255,0.06)', scale: 1.04 }}
          transition={{ duration: 0.15 }}
        >
          {state !== 'off' && (
            <motion.span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: state === 'green' ? '#22c55e' : '#f59e0b',
                display: 'block',
                boxShadow: state === 'green'
                  ? '0 0 6px rgba(34,197,94,0.7)'
                  : '0 0 6px rgba(245,158,11,0.7)',
              }}
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 1.8 + i * 0.2, repeat: Infinity, ease: 'easeInOut', delay: i * 0.1 }}
            />
          )}
        </motion.div>
      ))}
    </div>
  )
}

export default function LandingPage() {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const navigate = useNavigate()

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setError(null)
    setUploading(true)
    setProgress(0)
    try {
      const result = await uploadAudio(file, setProgress)
      navigate(`/processing/${result.session_id}`, { state: { filename: file.name, duration: result.duration_seconds } })
    } catch (e) {
      setError(e?.response?.data?.detail || 'Upload failed. Please try again.')
      setUploading(false)
    }
  }, [navigate])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden pt-24">
      <GradientOrbs />

      {/* Hero */}
      <motion.div
        className="relative z-10 text-center mb-10"
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
      >
        <h1
          className="text-6xl md:text-8xl font-bold tracking-tight mb-4 leading-none text-gradient-amber"
        >
          Arrangement<br />Transformer
        </h1>
        <p className="text-xl md:text-2xl text-zinc-400 font-light">
          Any piano piece. Any keyboard.
        </p>
      </motion.div>

      {/* Decorative soundboard pads */}
      <motion.div
        className="relative z-10 mb-10"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1, ease: [0.4, 0, 0.2, 1] }}
      >
        <SoundboardPads />
      </motion.div>

      {/* Upload card */}
      <motion.div
        className="relative z-10 w-full max-w-lg px-4"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2, ease: [0.4, 0, 0.2, 1] }}
      >
        <GlassCard
          className={`p-10 text-center transition-all duration-300 ${
            dragging
              ? 'ring-2 ring-amber-400/60 scale-[1.02]'
              : ''
          }`}
          style={dragging ? { borderColor: 'rgba(245,158,11,0.4)', boxShadow: '0 0 32px rgba(245,158,11,0.15)' } : {}}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          {!uploading ? (
            <>
              {/* Piano icon in a dark knob-like circle */}
              <div className="mb-6 flex justify-center">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-3xl"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    boxShadow: '0 0 0 4px rgba(245,158,11,0.08), inset 0 1px 0 rgba(255,255,255,0.08)',
                  }}
                >
                  🎹
                </div>
              </div>

              {/* Drop zone dashed border hint */}
              <div
                className="rounded-xl px-6 py-5 mb-6"
                style={{
                  border: `2px dashed ${dragging ? 'rgba(245,158,11,0.5)' : 'rgba(255,255,255,0.1)'}`,
                  background: dragging ? 'rgba(245,158,11,0.04)' : 'transparent',
                  transition: 'all 0.2s ease',
                }}
              >
                <p className="text-base font-medium text-white mb-1">
                  Drop your piano recording here
                </p>
                <p className="text-sm text-zinc-500">
                  MP3, WAV, M4A, OGG, FLAC, AIFF supported
                </p>
              </div>

              <button
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full text-sm font-semibold transition-all duration-200 active:scale-95"
                style={{
                  background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                  color: '#0a0a0b',
                  boxShadow: '0 4px 16px rgba(245,158,11,0.35)',
                }}
              >
                Choose File
              </button>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED}
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />
              {error && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4 text-sm text-red-400"
                >
                  {error}
                </motion.p>
              )}
            </>
          ) : (
            <div className="py-4 flex flex-col items-center gap-4">
              <AudioBars color="#22c55e" height={36} />
              <p className="text-sm font-medium text-white">Uploading…</p>
              <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)' }}
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ ease: 'easeOut' }}
                />
              </div>
              <p className="text-xs text-zinc-500">{progress}%</p>
            </div>
          )}
        </GlassCard>
      </motion.div>

      {/* How it works */}
      <motion.section
        id="how-it-works"
        className="relative z-10 mt-24 w-full max-w-4xl px-4 pb-24"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.4, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-center text-3xl font-semibold tracking-tight mb-12 text-gradient">
          How it works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            {
              num: '01',
              icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              ),
              title: 'Upload',
              desc: 'Drop any piano recording — MP3, WAV, and more. We convert it to the ideal format automatically.',
            },
            {
              num: '02',
              icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                </svg>
              ),
              title: 'Analyse',
              desc: 'Basic Pitch transcribes every note. Spleeter separates stems. Librosa detects tempo and segments by measure.',
            },
            {
              num: '03',
              icon: (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                </svg>
              ),
              title: 'Transform',
              desc: 'Notes are rearranged into your target keyboard range using a smart priority hierarchy with ghost-note fills.',
            },
          ].map((step) => (
            <GlassCard key={step.title} className="p-6 glass-hover" hover>
              {/* Amber numbered circle */}
              <div className="flex items-center gap-3 mb-4">
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{
                    background: 'rgba(245,158,11,0.12)',
                    border: '1px solid rgba(245,158,11,0.3)',
                    boxShadow: '0 0 12px rgba(245,158,11,0.15)',
                  }}
                >
                  <span className="text-xs font-bold text-amber-400">{step.num}</span>
                </div>
                <div className="text-amber-400">{step.icon}</div>
              </div>
              <h3 className="text-base font-semibold mb-2 text-white">{step.title}</h3>
              <p className="text-sm text-zinc-400 leading-relaxed">{step.desc}</p>
            </GlassCard>
          ))}
        </div>
      </motion.section>
    </div>
  )
}
