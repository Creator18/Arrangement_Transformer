import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import GradientOrbs from '../components/GradientOrbs'
import GlassCard from '../components/GlassCard'
import AudioBars from '../components/AudioBars'
import { uploadAudio } from '../utils/api'

const ACCEPTED = '.mp3,.wav,.m4a,.ogg,.flac,.aiff'

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

      {/* Hero text */}
      <motion.div
        className="relative z-10 text-center mb-12"
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
      >
        <h1 className="text-6xl md:text-8xl font-bold tracking-tight text-gradient mb-4 leading-none">
          Arrangement<br />Transformer
        </h1>
        <p className="text-xl md:text-2xl text-[#6e6e73] font-light">
          Any piano piece. Any keyboard.
        </p>
      </motion.div>

      {/* Upload card */}
      <motion.div
        className="relative z-10 w-full max-w-lg px-4"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.15, ease: [0.4, 0, 0.2, 1] }}
      >
        <GlassCard
          className={`p-10 text-center transition-all duration-300 ${
            dragging ? 'ring-2 ring-blue-400 scale-[1.02] bg-white/85' : ''
          }`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          {!uploading ? (
            <>
              <div className="mb-6 flex justify-center">
                <div className="w-16 h-16 rounded-2xl glass flex items-center justify-center text-3xl shadow-glass">
                  🎹
                </div>
              </div>
              <p className="text-base font-medium text-[#1d1d1f] mb-1">
                Drop your piano recording here
              </p>
              <p className="text-sm text-[#6e6e73] mb-8">
                MP3, WAV, M4A, OGG, FLAC, AIFF supported
              </p>
              <button
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-[#0071e3] text-white text-sm font-medium hover:bg-[#0077ed] active:bg-[#006edb] transition-colors duration-200 shadow-sm"
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
                  className="mt-4 text-sm text-red-500"
                >
                  {error}
                </motion.p>
              )}
            </>
          ) : (
            <div className="py-4 flex flex-col items-center gap-4">
              <AudioBars color="#0071e3" height={36} />
              <p className="text-sm font-medium text-[#1d1d1f]">Uploading…</p>
              <div className="w-full h-1.5 bg-[#e5e5ea] rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-[#0071e3] rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ ease: 'easeOut' }}
                />
              </div>
              <p className="text-xs text-[#6e6e73]">{progress}%</p>
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
        transition={{ duration: 0.8, delay: 0.35, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-center text-3xl font-semibold tracking-tight mb-12 text-gradient">
          How it works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { icon: '🎵', title: 'Upload', desc: 'Drop any piano recording — MP3, WAV, and more. We convert it to the ideal format automatically.' },
            { icon: '🧠', title: 'Analyse', desc: 'Basic Pitch transcribes every note. Spleeter separates stems. Librosa detects tempo and segments by measure.' },
            { icon: '🎹', title: 'Transform', desc: 'Notes are rearranged into your target keyboard range using a smart priority hierarchy with ghost-note fills.' },
          ].map((step) => (
            <GlassCard key={step.title} className="p-6">
              <div className="text-3xl mb-3">{step.icon}</div>
              <h3 className="text-base font-semibold mb-2 text-[#1d1d1f]">{step.title}</h3>
              <p className="text-sm text-[#6e6e73] leading-relaxed">{step.desc}</p>
            </GlassCard>
          ))}
        </div>
      </motion.section>
    </div>
  )
}
