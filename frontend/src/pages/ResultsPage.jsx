import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import GradientOrbs from '../components/GradientOrbs'
import GlassCard from '../components/GlassCard'
import FidelityBadge from '../components/FidelityBadge'
import PianoRoll from '../components/PianoRoll'
import CdefView from '../components/CdefView'
import { getCdefText, getMidiUrl, getPdfUrl, getCdefUrl, getOriginalAudioUrl, getTranscriptionAudioUrl, getTransformedAudioUrl } from '../utils/api'
import AudioPlayer from '../components/AudioPlayer'
import AudioBars from '../components/AudioBars'

const TABS = ['Piano Roll', 'Sheet Music', 'CDEF']

export default function ResultsPage() {
  const { sessionId } = useParams()
  const [tab, setTab] = useState(0)
  const [scoreData, setScoreData] = useState(null)
  const [transformData, setTransformData] = useState(null)
  const [transcriptionData, setTranscriptionData] = useState(null)
  const [segmentData, setSegmentData] = useState(null)
  const [cdef, setCdef] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function load() {
      try {
        // Read cached JSON files via fetch (served by vite proxy or directly)
        const [scoreRes, transformRes, transcriptionRes, segmentRes] = await Promise.all([
          fetch(`/session-data/${sessionId}/score.json`).then(r => r.ok ? r.json() : null),
          fetch(`/session-data/${sessionId}/transformed.json`).then(r => r.ok ? r.json() : null),
          fetch(`/session-data/${sessionId}/transcription.json`).then(r => r.ok ? r.json() : null),
          fetch(`/session-data/${sessionId}/segments.json`).then(r => r.ok ? r.json() : null),
        ])
        setScoreData(scoreRes)
        setTransformData(transformRes)
        setTranscriptionData(transcriptionRes)
        setSegmentData(segmentRes)

        const cdefText = await getCdefText(sessionId)
        setCdef(cdefText)
      } catch (e) {
        console.error('Failed to load results', e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [sessionId])

  const originalNotes = transcriptionData?.notes || []
  const transformedNotes = transformData?.transformed_notes || []
  const segments = segmentData?.segments || []
  const flagged = scoreData?.flagged_segments || []

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <GradientOrbs />
        <AudioBars color="#0071e3" height={40} className="relative z-10" />
        <p className="relative z-10 text-[#6e6e73] text-sm">Loading results…</p>
      </div>
    )
  }

  return (
    <div className="relative min-h-screen pt-20 pb-24 px-4">
      <GradientOrbs />
      <div className="relative z-10 max-w-4xl mx-auto space-y-6">

        {/* Header + fidelity */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h1 className="text-3xl font-bold tracking-tight text-gradient mb-6">Your Arrangement</h1>
          {scoreData && (
            <div className="flex justify-center mb-6">
              <FidelityBadge score={scoreData.fidelity_score} tier={scoreData.tier} />
            </div>
          )}
          {scoreData?.manual_review_required && (
            <p className="text-sm text-[#6e6e73]">
              Some segments were flagged for manual review — see below.
            </p>
          )}
        </motion.div>

        {/* Audio player */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.08 }}
        >
          <GlassCard className="p-5">
            <p className="text-xs font-semibold text-[#6e6e73] uppercase tracking-wider mb-3">Listen</p>
            <AudioPlayer
              tracks={[
                { label: 'Original Recording', url: getOriginalAudioUrl(sessionId), downloadName: 'original.wav' },
                { label: 'Original Arrangement', url: getTranscriptionAudioUrl(sessionId), downloadName: 'transcription.wav' },
                { label: 'Transformed', url: getTransformedAudioUrl(sessionId), downloadName: 'transformed.wav' },
              ]}
            />
          </GlassCard>
        </motion.div>

        {/* Downloads */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <GlassCard className="p-5 flex flex-wrap gap-3 justify-center">
            {[
              { label: 'Download MIDI', url: getMidiUrl(sessionId), ext: '.mid' },
              { label: 'Download PDF', url: getPdfUrl(sessionId), ext: '.pdf' },
              { label: 'Download CDEF', url: getCdefUrl(sessionId), ext: '.cdef' },
            ].map(({ label, url, ext }) => (
              <a
                key={ext}
                href={url}
                download
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#1d1d1f] text-white text-sm font-medium hover:bg-[#3a3a3c] transition-colors duration-200"
              >
                {label}
              </a>
            ))}
          </GlassCard>
        </motion.div>

        {/* Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
        >
          <GlassCard className="overflow-hidden">
            {/* Tab bar */}
            <div className="flex border-b border-white/40">
              {TABS.map((name, i) => (
                <button
                  key={name}
                  onClick={() => setTab(i)}
                  className={`flex-1 py-3.5 text-sm font-medium transition-colors duration-200 relative ${
                    tab === i ? 'text-[#0071e3]' : 'text-[#6e6e73] hover:text-[#1d1d1f]'
                  }`}
                >
                  {name}
                  {tab === i && (
                    <motion.div
                      layoutId="tab-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#0071e3]"
                    />
                  )}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div className="p-6">
              <AnimatePresence mode="wait">
                {tab === 0 && (
                  <motion.div
                    key="piano-roll"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <PianoRoll originalNotes={originalNotes} transformedNotes={transformedNotes} />
                  </motion.div>
                )}

                {tab === 1 && (
                  <motion.div
                    key="sheet"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    className="text-center"
                  >
                    <iframe
                      src={getPdfUrl(sessionId)}
                      className="w-full rounded-xl border border-white/30"
                      style={{ height: 600 }}
                      title="Sheet Music"
                    />
                    <p className="mt-3 text-xs text-[#6e6e73]">
                      If the PDF doesn't render, use the Download PDF button above.
                    </p>
                  </motion.div>
                )}

                {tab === 2 && (
                  <motion.div
                    key="cdef"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <CdefView cdef={cdef} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </GlassCard>
        </motion.div>

        {/* Segment breakdown */}
        {segments.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <h2 className="text-lg font-semibold text-gradient mb-4">Segment Breakdown</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {segments.map((seg) => {
                const flagInfo = flagged.find(f => f.measure_index === seg.measure_index)
                const isFlagged = !!flagInfo

                return (
                  <GlassCard
                    key={seg.measure_index}
                    className={`p-4 ${isFlagged ? (flagInfo.flag === 'red' ? 'ring-1 ring-red-200' : 'ring-1 ring-yellow-200') : ''}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs font-semibold text-[#3a3a3c]">
                        Measure {seg.measure_index + 1}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                          seg.star === 'bass' ? 'bg-blue-50 text-blue-600' : 'bg-rose-50 text-rose-600'
                        }`}>
                          ★ {seg.star}
                        </span>
                        {isFlagged && (
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                            flagInfo.flag === 'red' ? 'bg-red-50 text-red-600' : 'bg-yellow-50 text-yellow-600'
                          }`}>
                            {flagInfo.flag === 'red' ? '⚠ Low fidelity' : '⚠ Review'}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-[#6e6e73]">
                      {seg.start_time.toFixed(2)}s – {seg.end_time.toFixed(2)}s &nbsp;·&nbsp; {seg.time_signature}
                    </p>
                    <div className="mt-2 flex gap-4">
                      <div>
                        <p className="text-[10px] text-[#aeaeb2] uppercase tracking-wide">Bass JS</p>
                        <p className="text-xs font-medium text-[#3a3a3c]">{seg.js_scores.bass.toFixed(3)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-[#aeaeb2] uppercase tracking-wide">Melody JS</p>
                        <p className="text-xs font-medium text-[#3a3a3c]">{seg.js_scores.melody.toFixed(3)}</p>
                      </div>
                      {flagInfo && (
                        <div>
                          <p className="text-[10px] text-[#aeaeb2] uppercase tracking-wide">Fidelity</p>
                          <p className={`text-xs font-medium ${flagInfo.flag === 'red' ? 'text-red-600' : 'text-yellow-600'}`}>
                            {Math.round(flagInfo.segment_fidelity * 100)}%
                          </p>
                        </div>
                      )}
                    </div>
                  </GlassCard>
                )
              })}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
