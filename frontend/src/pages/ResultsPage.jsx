import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import GradientOrbs from '../components/GradientOrbs'
import GlassCard from '../components/GlassCard'
import FidelityBadge from '../components/FidelityBadge'
import CdefView from '../components/CdefView'
import PianoRollPlayer from '../components/PianoRollPlayer'
import SheetMusicView from '../components/SheetMusicView'
import { getCdefText, getMidiUrl, getCdefUrl, getMusicXmlUrl, getPartAudioUrl, getOriginalAudioUrl, getTranscriptionAudioUrl, getTransformedAudioUrl } from '../utils/api'
import AudioPlayer from '../components/AudioPlayer'
import AudioBars from '../components/AudioBars'

const TABS = ['Piano Roll', 'Sheet Music', 'CDEF']

function DetectedKeyBadge({ detected_scale }) {
  if (!detected_scale) return null
  const { root, scale, notes } = detected_scale
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="flex justify-center"
    >
      <div
        className="inline-flex items-center gap-3 px-4 py-2.5 rounded-2xl"
        style={{
          background: 'rgba(245,158,11,0.08)',
          border: '1px solid rgba(245,158,11,0.3)',
          boxShadow: '0 0 16px rgba(245,158,11,0.12), inset 0 1px 0 rgba(255,255,255,0.04)',
        }}
      >
        {/* Musical note icon */}
        <svg
          className="w-4 h-4 flex-shrink-0"
          style={{ color: '#f59e0b' }}
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          <path d="M9 3v10.55A4 4 0 1011 17V7h4V3H9z" />
        </svg>
        <div className="flex items-baseline gap-2">
          <span className="text-xs font-bold tracking-widest text-amber-400 uppercase">Key</span>
          <span className="text-sm font-bold text-white">{root}</span>
          <span className="text-sm text-amber-300/80 capitalize">{scale}</span>
        </div>
        {notes && notes.length > 0 && (
          <>
            <div className="w-px h-4 bg-white/10" />
            <div className="flex gap-1">
              {notes.slice(0, 7).map((n) => (
                <span
                  key={n}
                  className="text-[10px] font-medium text-zinc-500 tabular-nums"
                >
                  {n}
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </motion.div>
  )
}

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

  const originalNotes  = transcriptionData?.notes || []
  const transformedNotes = transformData?.transformed_notes || []
  const segments = segmentData?.segments || []
  const flagged  = scoreData?.flagged_segments || []

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <GradientOrbs />
        <AudioBars color="#22c55e" height={40} className="relative z-10" />
        <p className="relative z-10 text-zinc-500 text-sm">Loading results…</p>
      </div>
    )
  }

  return (
    <div className="relative min-h-screen pt-20 pb-24 px-4">
      <GradientOrbs />
      <div className="relative z-10 max-w-4xl mx-auto space-y-6">

        {/* Header + fidelity + detected key */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center space-y-4"
        >
          <h1 className="text-3xl font-bold tracking-tight text-gradient mb-2">Your Arrangement</h1>
          {scoreData && (
            <div className="flex justify-center">
              <FidelityBadge score={scoreData.fidelity_score} tier={scoreData.tier} />
            </div>
          )}
          {/* Detected key badge — between fidelity and player */}
          <DetectedKeyBadge detected_scale={transformData?.detected_scale} />
          {scoreData?.manual_review_required && (
            <p className="text-sm text-zinc-500">
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
            <p
              className="text-xs font-bold tracking-widest mb-4 uppercase"
              style={{ color: '#f59e0b', letterSpacing: '0.12em' }}
            >
              Listen
            </p>
            <AudioPlayer
              tracks={[
                { label: 'Original Recording',    url: getOriginalAudioUrl(sessionId),      downloadName: 'original.wav'      },
                { label: 'Original Arrangement',  url: getTranscriptionAudioUrl(sessionId), downloadName: 'transcription.wav' },
                { label: 'Transformed',           url: getTransformedAudioUrl(sessionId),   downloadName: 'transformed.wav'   },
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
              { label: 'Download MIDI', url: getMidiUrl(sessionId),  ext: '.mid'  },
              { label: 'Download MusicXML', url: getMusicXmlUrl(sessionId), ext: '.xml' },
              { label: 'Download CDEF', url: getCdefUrl(sessionId),  ext: '.cdef' },
            ].map(({ label, url, ext }) => (
              <a
                key={ext}
                href={url}
                download
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 hover:scale-[1.03] active:scale-95"
                style={{
                  background: 'rgba(245,158,11,0.08)',
                  border: '1px solid rgba(245,158,11,0.3)',
                  color: '#f59e0b',
                  boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'rgba(245,158,11,0.14)'
                  e.currentTarget.style.boxShadow = '0 0 12px rgba(245,158,11,0.2), 0 2px 12px rgba(0,0,0,0.3)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'rgba(245,158,11,0.08)'
                  e.currentTarget.style.boxShadow = '0 2px 12px rgba(0,0,0,0.3)'
                }}
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
            <div className="flex" style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              {TABS.map((name, i) => (
                <button
                  key={name}
                  onClick={() => setTab(i)}
                  className={`flex-1 py-3.5 text-sm font-medium transition-colors duration-200 relative ${
                    tab === i ? 'text-amber-400' : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {name}
                  {tab === i && (
                    <motion.div
                      layoutId="tab-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 rounded-full"
                      style={{ background: 'linear-gradient(90deg, #f59e0b, #d97706)' }}
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
                    className="space-y-6"
                  >
                    {/* Original — 7-octave keyboard C1→B7 */}
                    <PianoRollPlayer
                      notes={originalNotes}
                      getPartUrl={(part) => getPartAudioUrl(sessionId, 'transcription', part)}
                      label="Original Arrangement"
                      tempo={segmentData?.tempo_bpm ?? 120}
                      keyboardStart={24}
                      keyboardOctaves={7}
                    />
                    <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '8px 0' }} />
                    {/* Transformed — 5-octave keyboard C2→B6 */}
                    <PianoRollPlayer
                      notes={transformedNotes}
                      getPartUrl={(part) => getPartAudioUrl(sessionId, 'transformed', part)}
                      label="Transformed Arrangement"
                      tempo={segmentData?.tempo_bpm ?? 120}
                      keyboardStart={36}
                      keyboardOctaves={5}
                    />
                  </motion.div>
                )}
                {tab === 1 && (
                  <motion.div
                    key="sheet"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <SheetMusicView xmlUrl={getMusicXmlUrl(sessionId)} />
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
                const isRedFlag = isFlagged && flagInfo.flag === 'red'

                return (
                  <GlassCard
                    key={seg.measure_index}
                    className="p-4"
                    style={isFlagged ? {
                      borderColor: isRedFlag ? 'rgba(239,68,68,0.35)' : 'rgba(234,179,8,0.35)',
                      boxShadow: isRedFlag
                        ? '0 0 16px rgba(239,68,68,0.1), 0 25px 60px rgba(0,0,0,0.5)'
                        : '0 0 16px rgba(234,179,8,0.1), 0 25px 60px rgba(0,0,0,0.5)',
                    } : {}}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs font-semibold text-zinc-300">
                        Measure {seg.measure_index + 1}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span
                          className="text-xs px-2 py-0.5 rounded-full font-medium"
                          style={
                            seg.star === 'bass'
                              ? { background: 'rgba(59,130,246,0.15)', color: '#60a5fa' }
                              : { background: 'rgba(244,63,94,0.15)', color: '#fb7185' }
                          }
                        >
                          ★ {seg.star}
                        </span>
                        {isFlagged && (
                          <span
                            className="text-xs px-2 py-0.5 rounded-full font-medium"
                            style={isRedFlag
                              ? { background: 'rgba(239,68,68,0.15)', color: '#f87171' }
                              : { background: 'rgba(234,179,8,0.15)', color: '#fde047' }
                            }
                          >
                            {isRedFlag ? '⚠ Low fidelity' : '⚠ Review'}
                          </span>
                        )}
                      </div>
                    </div>

                    <p className="text-xs text-zinc-600">
                      {seg.start_time.toFixed(2)}s – {seg.end_time.toFixed(2)}s &nbsp;·&nbsp; {seg.time_signature}
                    </p>

                    <div className="mt-2 flex gap-4">
                      <div>
                        <p className="text-[10px] text-zinc-600 uppercase tracking-wide">Bass JS</p>
                        <p className="text-xs font-medium text-zinc-300">{seg.js_scores.bass.toFixed(3)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] text-zinc-600 uppercase tracking-wide">Melody JS</p>
                        <p className="text-xs font-medium text-zinc-300">{seg.js_scores.melody.toFixed(3)}</p>
                      </div>
                      {flagInfo && (
                        <div>
                          <p className="text-[10px] text-zinc-600 uppercase tracking-wide">Fidelity</p>
                          <p className={`text-xs font-medium ${isRedFlag ? 'text-red-400' : 'text-yellow-400'}`}>
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
