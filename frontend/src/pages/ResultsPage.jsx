import { useEffect, useState, useRef, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import AuroraRibbons from '../components/AuroraRibbons'
import BubbleCluster from '../components/BubbleCluster'
import { DEFAULTS as TONE_DEFAULTS } from '../utils/tonechain'

// Stable ref identity on purpose.  StringSpace rebuilds its whole WebGL scene
// when this prop changes, and an inline {current: x} is a NEW object every
// render — on a page that re-renders on every progress tick that would tear
// down and recreate the renderer dozens of times.
const SCENE_POS = { current: 0.85 }

import GlassCard from '../components/GlassCard'
import FidelityBadge from '../components/FidelityBadge'
import CdefView from '../components/CdefView'
import PianoRollPlayer from '../components/PianoRollPlayer'
import SheetMusicView from '../components/SheetMusicView'
import { getCdefText, getMidiUrl, getCdefUrl, getMusicXmlUrl, getPartAudioUrl, getLayerAudioUrl, getOriginalAudioUrl, getTranscriptionAudioUrl, getTransformedAudioUrl, getPatternClusters, getPatternClusterAudioUrl } from '../utils/api'
import { letterReveal, gridRise } from '../utils/animefx'
import AudioPlayer from '../components/AudioPlayer'
import AudioBars from '../components/AudioBars'

const TABS = ['Piano Roll', 'Layers', 'Sheet Music', 'CDEF']

// Audio-level / classifier-level layers we expose as their own piano roll for
// reviewing what the pipeline actually captured per stem.  Order matches the
// natural register stack low→high.
// Four base layers from REPET-SIM-on-both-bands + pYIN-with-mask.
// 'bass' is shown as a fallback when REPET hasn't run on bass (no mask).
const LAYER_VIEWS = [
  { key: 'bass_pattern', label: 'Bass Pattern', desc: 'pYIN notes where REPET bass mask ≥ 0.5 (recurring bass figure)', color: '#1d4ed8', keyboardStart: 24, octaves: 4, hasClusters: true  },
  { key: 'bass_melody',  label: 'Bass Melody',  desc: 'pYIN notes where REPET bass mask < 0.5 (free bass line)',         color: '#60a5fa', keyboardStart: 24, octaves: 4, hasClusters: false },
  { key: 'pattern',      label: 'Pattern',      desc: 'Basic Pitch on pattern.wav (recurring upper-register figures)',   color: '#a78bfa', keyboardStart: 36, octaves: 5, hasClusters: true  },
  { key: 'melody',       label: 'Melody',       desc: 'Basic Pitch on melody.wav (free/lead voice)',                     color: '#f87171', keyboardStart: 36, octaves: 5, hasClusters: false },
]

// Cluster row palette — variations of the parent layer color
const CLUSTER_PALETTE = ['#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6']

// ── Static fallback parts (used when API returns no dynamic layers) ────────────
const STATIC_PARTS = [
  { key: 'all',    label: 'All',    sym: '◉', color: '#a1a1aa', desc: 'Full mix'      },
  { key: 'bass',   label: 'Bass',   sym: '▪', color: '#3b82f6', desc: 'Low register'  },
  { key: 'melody', label: 'Melody', sym: '▲', color: '#f87171', desc: 'Mid register'  },
  { key: 'voice',  label: 'Voice',  sym: '◈', color: '#a78bfa', desc: 'High register' },
]

// Layer sym glyphs per name keyword
function layerSym(name) {
  if (name.includes('Bass'))   return '▪'
  if (name.includes('Mid'))    return '◈'
  if (name.includes('High'))   return '⬡'
  if (name.includes('Melody')) return '▲'
  return '◉'
}

function PartPad({ part, selected, onSelect }) {
  const ref = useRef(null)

  const onMouseMove = (e) => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width  - 0.5
    const y = (e.clientY - r.top)  / r.height - 0.5
    el.style.transform = `perspective(300px) rotateX(${-y * 10}deg) rotateY(${x * 10}deg) scale(${selected ? 1.05 : 1.03})`
    el.style.transition = 'transform 0.07s ease'
  }

  const onMouseLeave = () => {
    const el = ref.current
    if (!el) return
    el.style.transform = selected
      ? 'perspective(300px) rotateX(0deg) rotateY(0deg) scale(1.03)'
      : 'perspective(300px) rotateX(0deg) rotateY(0deg) scale(1)'
    el.style.transition = 'transform 0.35s ease'
  }

  return (
    <motion.button
      ref={ref}
      onClick={() => onSelect(part.key)}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      className="pad-press flex-1 flex flex-col items-center justify-center gap-1.5 py-4 rounded-sm relative overflow-hidden"
      style={{
        background: selected
          ? `linear-gradient(145deg, ${part.color}18, ${part.color}08)`
          : 'rgba(255,255,255,0.02)',
        border: `1px solid ${selected ? part.color + '50' : 'rgba(255,255,255,0.06)'}`,
        cursor: 'pointer',
        boxShadow: selected
          ? `0 0 0 1px ${part.color}25, 0 0 20px ${part.color}20, 0 8px 32px rgba(0,0,0,0.5)`
          : '0 4px 16px rgba(0,0,0,0.4)',
        transition: 'border-color 0.2s, box-shadow 0.2s, background 0.2s',
        willChange: 'transform',
        minHeight: 80,
      }}
      whileTap={{ scale: 0.91 }}
      initial={false}
      animate={{ scale: selected ? 1.03 : 1 }}
      transition={{ type: 'spring', stiffness: 320, damping: 26 }}
    >
      {/* LED */}
      <motion.div
        style={{
          width: 5, height: 5,
          borderRadius: '50%',
          background: selected ? part.color : 'rgba(255,255,255,0.08)',
          boxShadow: selected ? `0 0 6px ${part.color}, 0 0 12px ${part.color}70` : 'none',
          marginBottom: 2,
        }}
        animate={selected ? { opacity: [1, 0.4, 1] } : { opacity: 1 }}
        transition={{ duration: 1.4, repeat: selected ? Infinity : 0 }}
      />

      <span style={{
        fontSize: 16,
        color: selected ? part.color : 'rgba(255,255,255,0.25)',
        fontFamily: 'ui-monospace,monospace',
        lineHeight: 1,
        transition: 'color 0.2s',
      }}>
        {part.sym}
      </span>

      <p style={{
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: '0.10em',
        color: selected ? part.color : 'rgba(255,255,255,0.35)',
        transition: 'color 0.2s',
      }}>
        {part.label.toUpperCase()}
      </p>

      <p style={{
        fontSize: 7.5,
        color: selected ? `${part.color}80` : 'rgba(255,255,255,0.15)',
        letterSpacing: '0.04em',
        transition: 'color 0.2s',
      }}>
        {part.desc}
      </p>
    </motion.button>
  )
}

// ── Detected key badge ────────────────────────────────────────────────────────
function DetectedKeyBadge({ detected_scale }) {
  if (!detected_scale) return null
  const { root, scale, notes } = detected_scale
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.88 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.45 }}
      className="flex justify-center"
    >
      <div
        className="inline-flex items-center gap-3 px-5 py-2.5 rounded-sm"
        style={{
          background: 'rgba(245,158,11,0.07)',
          border: '1px solid rgba(245,158,11,0.28)',
          boxShadow: '0 0 20px rgba(245,158,11,0.10), inset 0 1px 0 rgba(255,255,255,0.04)',
        }}
      >
        <span style={{ fontSize: 14, color: '#f59e0b', fontFamily: 'ui-monospace,monospace' }}>♪</span>
        <div className="flex items-baseline gap-2">
          <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.18em', color: '#f59e0b' }}>KEY</span>
          <span style={{ fontSize: 14, fontWeight: 800, color: '#fff' }}>{root}</span>
          <span style={{ fontSize: 13, color: 'rgba(251,191,36,0.75)', textTransform: 'capitalize' }}>{scale}</span>
        </div>
        {notes?.length > 0 && (
          <>
            <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)' }} />
            <div className="flex gap-1">
              {notes.slice(0, 7).map((n) => (
                <span key={n} style={{ fontSize: 9, color: 'rgba(255,255,255,0.28)', fontFamily: 'ui-monospace,monospace' }}>
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

// ── Main results page ─────────────────────────────────────────────────────────
export default function ResultsPage() {
  const { sessionId } = useParams()
  const [tab,              setTab]              = useState(0)
  const [scoreData,        setScoreData]        = useState(null)
  const [transformData,    setTransformData]    = useState(null)
  const [transcriptionData,setTranscriptionData]= useState(null)
  const [segmentData,      setSegmentData]      = useState(null)
  const [cdef,             setCdef]             = useState('')
  const [loading,          setLoading]          = useState(true)
  const [activePart,       setActivePart]       = useState('all')
  // Which arrangement the stage shows.  The two rolls used to be stacked, each
  // squeezed into 370px; only one can be listened to at a time, so showing one
  // at full height and switching is both clearer and bigger.
  const [range,            setRange]            = useState('transformed')
  const [stageH,           setStageH]           = useState(560)
  // Live monitoring tone.  Starts at the server's own settings so moving a
  // slider adjusts from where the render already sits rather than jumping.
  const [tone,             setTone]             = useState(TONE_DEFAULTS)
  const [patternClusters,  setPatternClusters]  = useState({})
  const kickerRef  = useRef(null)
  const padsRowRef = useRef(null)

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
        try {
          const clusters = await getPatternClusters(sessionId)
          setPatternClusters(clusters || {})
        } catch (e) {
          // Endpoint is best-effort — empty {} just means no clusters to show
          setPatternClusters({})
        }
      } catch (e) {
        console.error('Failed to load results', e)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [sessionId])

  // anime.js flourishes once data has loaded and the page is rendered
  useEffect(() => {
    if (loading) return
    letterReveal(kickerRef.current, { delay: 250, stagger: 26 })
    if (padsRowRef.current) {
      gridRise(Array.from(padsRowRef.current.children), { delay: 500 })
    }
  }, [loading])

  const originalNotes    = transcriptionData?.notes || []
  const transformedNotes = transformData?.transformed_notes || []
  const segments         = segmentData?.segments || []
  const flagged          = scoreData?.flagged_segments || []

  // Build layer pads from dynamic API layers, or fall back to static register parts
  const apiLayers = transformData?.layers || null
  const layerPads = useMemo(() => {
    if (!apiLayers || apiLayers.length === 0) return STATIC_PARTS
    // Always prepend an "All" pad
    const all = { key: 'all', label: 'All', sym: '◉', color: '#a1a1aa', desc: 'Full mix' }
    return [
      all,
      ...apiLayers.map(l => ({
        key:   l.id,
        label: l.name.length > 8 ? l.name.split(' ')[0] : l.name,
        sym:   layerSym(l.name),
        color: l.color,
        desc:  `${l.notes?.length ?? 0} notes`,
      })),
    ]
  }, [apiLayers])

  // The roll should fill whatever the viewport leaves after the transport and
  // padding, rather than being pinned to a constant.
  useEffect(() => {
    const fit = () => setStageH(Math.max(320, window.innerHeight - 250))
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])

  // Notes shown in the transformed piano roll — filtered by selected layer
  const displayedTransformedNotes = useMemo(() => {
    if (activePart === 'all' || !apiLayers) return transformedNotes
    const layer = apiLayers.find(l => l.id === activePart)
    return layer?.notes ?? transformedNotes
  }, [activePart, transformedNotes, apiLayers])

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <AuroraRibbons progressRef={SCENE_POS} />
        <AudioBars color="#a78bfa" height={40} className="relative z-10" />
        <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.28)', fontFamily: 'ui-monospace,monospace', letterSpacing: '0.12em' }}
          className="relative z-10">
          LOADING RESULTS
        </p>
      </div>
    )
  }

  return (
    <div style={{ position: 'fixed', inset: 0, display: 'flex',
                  background: 'var(--ct-ink)', paddingTop: 56 }}>
      <AuroraRibbons progressRef={SCENE_POS} />

      {/* ── Sidebar: everything you can switch, nothing you can read ──────
          The old page put the view tabs across the top of a 4xl column and
          gave the roll 370px of it.  The roll IS the artefact here, so it
          takes the stage and the controls move to an edge. */}
      <aside
        // Lenis hijacks the wheel globally to drive its smooth page scroll,
        // which leaves nested scrollers dead to the wheel — you could only
        // drag the bar.  data-lenis-prevent hands wheel events back to this
        // element.  The CSS for it shipped with Lenis; it was never applied.
        data-lenis-prevent
        style={{
        width: 232, flexShrink: 0, position: 'relative', zIndex: 3,
        borderRight: '1px solid var(--ct-line-soft)',
        padding: '22px 18px', overflowY: 'auto',
        display: 'flex', flexDirection: 'column', gap: 26,
      }}>
        {/* Status — read-only, so it sits above the switches. */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {scoreData && (
            <FidelityBadge score={scoreData.fidelity_score} tier={scoreData.tier} />
          )}
          <DetectedKeyBadge detected_scale={transformData?.detected_scale} />
        </div>

        <div>
          <div className="ct-mono" style={{ marginBottom: 10 }}>View</div>
          {TABS.map((name, i) => (
            <button key={name} onClick={() => setTab(i)}
              style={{
                display: 'block', width: '100%', textAlign: 'left',
                background: 'none', border: 'none', cursor: 'pointer',
                padding: '7px 0', fontSize: 12,
                color: tab === i ? 'var(--ct-accent)' : 'rgba(255,255,255,0.46)',
                transition: 'color .3s',
              }}>
              {tab === i ? '— ' : '  '}{name}
            </button>
          ))}
        </div>

        {tab === 0 && (
          <div>
            <div className="ct-mono" style={{ marginBottom: 10 }}>Range</div>
            {[['transformed', 'Transformed', '5 octaves'],
              ['original',    'Original',    '7 octaves']].map(([k, label, sub]) => (
              <button key={k} onClick={() => setRange(k)}
                style={{
                  display: 'block', width: '100%', textAlign: 'left',
                  background: 'none', border: 'none', cursor: 'pointer',
                  padding: '7px 0',
                  color: range === k ? 'var(--ct-accent)' : 'rgba(255,255,255,0.46)',
                  transition: 'color .3s',
                }}>
                <div style={{ fontSize: 12 }}>{range === k ? '— ' : '  '}{label}</div>
                <div style={{ fontSize: 9, opacity: 0.55, paddingLeft: 14,
                              fontFamily: 'ui-monospace,monospace' }}>{sub}</div>
              </button>
            ))}
          </div>
        )}

        <div>
          <div className="ct-mono" style={{ marginBottom: 10 }}>Part</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {layerPads.map(p => (
              <button key={p.key} className="sw-pill" data-on={activePart === p.key}
                onClick={() => setActivePart(p.key)}
                style={{ padding: '6px 12px', fontSize: 10,
                         ...(activePart === p.key ? { '--ct-accent': p.color } : {}) }}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="ct-mono" style={{ marginBottom: 10 }}>Tone</div>
          {[
            ['bassDb',   'Bass',   -6,  6,  0.5, 'dB'],
            ['trebleDb', 'Treble', -6,  6,  0.5, 'dB'],
            ['room',     'Room',    0,  0.9, 0.05, ''],
          ].map(([key, label, min, max, step, unit]) => (
            <div key={key} style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            fontSize: 10, color: 'var(--ct-faint)',
                            fontFamily: 'ui-monospace,monospace', marginBottom: 4 }}>
                <span>{label}</span>
                <span style={{ color: 'var(--ct-accent)' }}>
                  {key === 'room' ? `${Math.round(tone[key] * 100)}%`
                                  : `${tone[key] > 0 ? '+' : ''}${tone[key].toFixed(1)}${unit}`}
                </span>
              </div>
              <input
                type="range" min={min} max={max} step={step} value={tone[key]}
                onChange={(e) => setTone(t => ({ ...t, [key]: parseFloat(e.target.value) }))}
                style={{ width: '100%', accentColor: 'var(--ct-accent)', height: 2 }}
              />
            </div>
          ))}
          <button
            onClick={() => setTone(TONE_DEFAULTS)}
            style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                     fontSize: 10, color: 'var(--ct-faint)',
                     fontFamily: 'ui-monospace,monospace' }}>
            reset
          </button>
          <p style={{ fontSize: 9, color: 'rgba(255,255,255,0.20)', marginTop: 8,
                      lineHeight: 1.5 }}>
            Affects playback only — exports carry the server settings.
          </p>
        </div>

        <div>
          <div className="ct-mono" style={{ marginBottom: 10 }}>Export</div>
          {[['MIDI', getMidiUrl(sessionId)],
            ['MusicXML', getMusicXmlUrl(sessionId)],
            ['CDEF', getCdefUrl(sessionId)]].map(([label, href]) => (
            <a key={label} href={href} download
              style={{ display: 'block', padding: '6px 0', fontSize: 12,
                       color: 'rgba(255,255,255,0.46)', textDecoration: 'none' }}>
              ↓ {label}
            </a>
          ))}
        </div>
      </aside>

      {/* ── Stage ───────────────────────────────────────────────────────── */}
      <main style={{ flex: 1, minWidth: 0, position: 'relative', zIndex: 2,
                     overflowY: 'auto', padding: '20px 26px 40px' }}>

        {/* ── Tabs: Piano Roll / Sheet / CDEF ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.18 }}>
          <GlassCard className="overflow-hidden">
            {/* Tab content */}
            <div className="p-5">
              <AnimatePresence mode="wait">
                {tab === 0 && (
                  <motion.div
                    key="piano-roll"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="space-y-6"
                  >
                    {/* One roll, at whatever height the viewport allows.
                        Both used to be rendered stacked at 370px each, but only
                        one can be played at a time, so the other was costing
                        half the screen to sit idle. */}
                    {range === 'original' ? (
                      <PianoRollPlayer
                        key="roll-original"
                        notes={originalNotes}
                        getPartUrl={(part) => getPartAudioUrl(sessionId, 'transcription', part)}
                        label="Original"
                        tempo={segmentData?.tempo_bpm ?? 120}
                        keyboardStart={24}
                        keyboardOctaves={7}
                        defaultPart={activePart}
                        tone={tone}
                        rollHeight={stageH}
                      />
                    ) : (
                      <PianoRollPlayer
                        key="roll-transformed"
                        notes={displayedTransformedNotes}
                        getPartUrl={(part) => getPartAudioUrl(sessionId, 'transformed', part)}
                        label="Transformed"
                        tempo={segmentData?.tempo_bpm ?? 120}
                        keyboardStart={36}
                        keyboardOctaves={5}
                        defaultPart={apiLayers ? 'all' : activePart}
                        tone={tone}
                        rollHeight={stageH}
                      />
                    )}
                  </motion.div>
                )}
                {tab === 1 && (
                  <motion.div
                    key="layers"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="space-y-6"
                  >
                    <p style={{
                      fontSize: 11,
                      color: 'rgba(255,255,255,0.5)',
                      lineHeight: 1.5,
                    }}>
                      Each layer below is the slice of the 88-key transcription routed
                      to it by the pipeline.  REPET-SIM runs independently on bass.wav
                      and upper.wav, producing four layers: <em>bass pattern</em>,
                      <em> bass melody</em>, <em>pattern</em>, <em>melody</em>.  Pattern
                      layers are further broken into <em>Pattern 1…N</em> by agglomerative
                      clustering on chroma + onset-shape per bar — each is a distinct
                      recurring figure detected by the separator.
                    </p>
                    {LAYER_VIEWS.map((lv, i) => {
                      const layerNotes = originalNotes.filter(n => n.layer === lv.key)
                      const clusterMeta = lv.hasClusters ? (patternClusters?.[lv.key] ?? null) : null
                      const nClusters = clusterMeta?.n_clusters ?? 0
                      return (
                        <div key={lv.key}>
                          {i > 0 && (
                            <div style={{ height: 1, background: 'rgba(255,255,255,0.04)', marginBottom: 16 }} />
                          )}
                          <div className="flex items-baseline gap-3 mb-1.5 px-1">
                            <span style={{
                              fontSize: 11, fontWeight: 700, letterSpacing: '0.14em',
                              color: lv.color,
                            }}>
                              {lv.label.toUpperCase()}
                            </span>
                            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.30)' }}>
                              {layerNotes.length} note{layerNotes.length === 1 ? '' : 's'}
                            </span>
                            {nClusters > 0 && (
                              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.30)' }}>
                                · {nClusters} cluster{nClusters === 1 ? '' : 's'}
                                {clusterMeta?.granularity === 'half-bar' ? ' (½-bar)' : ''}
                              </span>
                            )}
                            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.20)', marginLeft: 'auto' }}>
                              {lv.desc}
                            </span>
                          </div>
                          <PianoRollPlayer
                            notes={layerNotes}
                            getPartUrl={() => getLayerAudioUrl(sessionId, lv.key)}
                            label={lv.label}
                            tempo={segmentData?.tempo_bpm ?? 120}
                            keyboardStart={lv.keyboardStart}
                            keyboardOctaves={lv.octaves}
                            defaultPart="all"
                          />
                          {/* Per-cluster sub-rows: Pattern 1, Pattern 2, … */}
                          {nClusters > 1 && Array.from({ length: nClusters }).map((_, ci) => {
                            const clusterNotes = layerNotes.filter(n => Number(n.pattern_idx) === ci)
                            const barCount = clusterMeta?.cluster_bar_counts?.[String(ci)] ?? 0
                            const accent = CLUSTER_PALETTE[ci % CLUSTER_PALETTE.length]
                            return (
                              <div key={`${lv.key}-${ci}`} style={{ marginTop: 12, paddingLeft: 16, borderLeft: `2px solid ${accent}40` }}>
                                <div className="flex items-baseline gap-3 mb-1.5 px-1">
                                  <span style={{
                                    fontSize: 10, fontWeight: 700, letterSpacing: '0.14em',
                                    color: accent,
                                  }}>
                                    {lv.label.toUpperCase()} {ci + 1}
                                  </span>
                                  <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.30)' }}>
                                    {clusterNotes.length} note{clusterNotes.length === 1 ? '' : 's'} · {barCount} bar{barCount === 1 ? '' : 's'}
                                  </span>
                                </div>
                                <PianoRollPlayer
                                  notes={clusterNotes}
                                  getPartUrl={() => getPatternClusterAudioUrl(sessionId, lv.key, ci)}
                                  label={`${lv.label} ${ci + 1}`}
                                  tempo={segmentData?.tempo_bpm ?? 120}
                                  keyboardStart={lv.keyboardStart}
                                  keyboardOctaves={lv.octaves}
                                  defaultPart="all"
                                />
                              </div>
                            )
                          })}
                        </div>
                      )
                    })}
                  </motion.div>
                )}
                {tab === 2 && (
                  <motion.div key="sheet" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                    <SheetMusicView xmlUrl={getMusicXmlUrl(sessionId)} />
                  </motion.div>
                )}
                {tab === 3 && (
                  <motion.div key="cdef" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
                    <CdefView cdef={cdef} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </GlassCard>
        </motion.div>

        {/* ── Listen ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.08 }}>
          <GlassCard className="p-6">
            <p style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.20em', color: 'rgba(245,158,11,0.7)', fontFamily: 'ui-monospace,monospace', marginBottom: 16 }}>
              LISTEN
            </p>
            <AudioPlayer
              tracks={[
                { label: 'Original Recording',   url: getOriginalAudioUrl(sessionId),      downloadName: 'original.wav'      },
                { label: 'Original Arrangement',  url: getTranscriptionAudioUrl(sessionId), downloadName: 'transcription.wav' },
                { label: 'Transformed',           url: getTransformedAudioUrl(sessionId),   downloadName: 'transformed.wav'   },
              ]}
            />
          </GlassCard>
        </motion.div>

        {/* ── Segment breakdown ── */}
        {segments.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.22 }}>
            <div className="flex items-center gap-3 mb-4">
              <div style={{ height: 1, flex: 1, background: 'rgba(255,255,255,0.05)' }} />
              <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.20em', color: 'rgba(255,255,255,0.20)', fontFamily: 'ui-monospace,monospace' }}>
                SEGMENT BREAKDOWN
              </span>
              <div style={{ height: 1, flex: 1, background: 'rgba(255,255,255,0.05)' }} />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {segments.map((seg) => {
                const flagInfo  = flagged.find(f => f.measure_index === seg.measure_index)
                const isFlagged = !!flagInfo
                const isRed     = isFlagged && flagInfo.flag === 'red'

                return (
                  <GlassCard
                    key={seg.measure_index}
                    className="p-4"
                    style={isFlagged ? {
                      borderColor: isRed ? 'rgba(239,68,68,0.3)' : 'rgba(234,179,8,0.3)',
                      boxShadow:   isRed ? '0 0 16px rgba(239,68,68,0.08)' : '0 0 16px rgba(234,179,8,0.08)',
                    } : {}}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span style={{ fontSize: 11, fontWeight: 600, color: 'rgba(255,255,255,0.6)', fontFamily: 'ui-monospace,monospace' }}>
                        M{seg.measure_index + 1}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span
                          className="text-[9px] px-2 py-0.5 rounded-full font-bold tracking-wide"
                          style={seg.star === 'bass'
                            ? { background: 'rgba(59,130,246,0.12)', color: '#60a5fa' }
                            : { background: 'rgba(244,63,94,0.12)', color: '#fb7185' }}
                        >
                          ★ {seg.star}
                        </span>
                        {isFlagged && (
                          <span
                            className="text-[9px] px-2 py-0.5 rounded-full font-bold"
                            style={isRed
                              ? { background: 'rgba(239,68,68,0.12)', color: '#f87171' }
                              : { background: 'rgba(234,179,8,0.12)', color: '#fde047' }}
                          >
                            {isRed ? '⚠ Low' : '⚠ Review'}
                          </span>
                        )}
                      </div>
                    </div>

                    <p style={{ fontSize: 10, color: 'rgba(255,255,255,0.22)', fontFamily: 'ui-monospace,monospace' }}>
                      {seg.start_time.toFixed(2)}s – {seg.end_time.toFixed(2)}s · {seg.time_signature}
                    </p>

                    <div className="mt-2.5 flex gap-5">
                      {[
                        { label: 'BASS JS',   val: seg.js_scores.bass.toFixed(3) },
                        { label: 'MELODY JS', val: seg.js_scores.melody.toFixed(3) },
                        ...(flagInfo ? [{ label: 'FIDELITY', val: `${Math.round(flagInfo.segment_fidelity * 100)}%`, color: isRed ? '#f87171' : '#fde047' }] : []),
                      ].map(({ label, val, color }) => (
                        <div key={label}>
                          <p style={{ fontSize: 8, color: 'rgba(255,255,255,0.22)', letterSpacing: '0.12em', fontFamily: 'ui-monospace,monospace', marginBottom: 2 }}>
                            {label}
                          </p>
                          <p style={{ fontSize: 11, fontWeight: 600, color: color || 'rgba(255,255,255,0.55)', fontFamily: 'ui-monospace,monospace' }}>
                            {val}
                          </p>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                )
              })}
            </div>
          </motion.div>
        )}
      </main>
    </div>
  )
}
