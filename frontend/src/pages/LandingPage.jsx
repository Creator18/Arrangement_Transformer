import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ScrollStage from '../components/ScrollStage'
import SectionRail from '../components/SectionRail'
import RackSection from '../components/RackSection'
import GlassCard from '../components/GlassCard'
import AudioBars from '../components/AudioBars'
import { uploadAudio } from '../utils/api'
import { letterReveal, gridRise } from '../utils/animefx'
import KeyboardAssembly from '../components/KeyboardAssembly'
import MasterDial from '../components/MasterDial'
import PipelineScrolly from '../components/PipelineScrolly'

// Channels on the rail, in document order.  Keep this list in step with
// ScrollStage's SCENES so the backdrop and the menu agree about where the
// user is — they read the same scroll position independently.
const SECTIONS = [
  { id: 'hero',     label: 'INTRO',    color: '#f59e0b' },
  { id: 'console',  label: 'CONSOLE',  color: '#a78bfa' },
  { id: 'pipeline', label: 'PIPELINE', color: '#06b6d4' },
  { id: 'engine',   label: 'ENGINE',   color: '#34d399' },
]

const ACCEPTED = '.mp3,.wav,.m4a,.ogg,.flac,.aiff'

// ── Transformation presets — the soundboard pads ─────────────────────────────
// Each pad is a real config that gets passed to the backend transform step.
const PRESETS = [
  { id: 'standard',  label: 'Standard',   sym: '◈',  octaves: 5, components: 16, color: '#f59e0b', desc: '5 oct · balanced'    },
  { id: 'wide',      label: 'Wide',        sym: '⬡',  octaves: 6, components: 16, color: '#3b82f6', desc: '6 oct · expressive'  },
  { id: 'full',      label: 'Concert',     sym: '◉',  octaves: 7, components: 20, color: '#a78bfa', desc: '7 oct · full range'  },
  { id: 'minimal',   label: 'Minimal',     sym: '▪',  octaves: 4, components: 8,  color: '#34d399', desc: '4 oct · clean'       },
  { id: 'jazz',      label: 'Jazz',        sym: '♩',  octaves: 5, components: 24, color: '#f87171', desc: '5 oct · rich NMF'    },
  { id: 'classical', label: 'Classical',   sym: '♪',  octaves: 6, components: 20, color: '#06b6d4', desc: '6 oct · detailed'    },
  { id: 'beginner',  label: 'Beginner',    sym: '▸',  octaves: 4, components: 12, color: '#84cc16', desc: '4 oct · simplified'  },
  { id: 'dense',     label: 'Dense',       sym: '⬟',  octaves: 5, components: 32, color: '#e879f9', desc: '5 oct · max NMF'     },
]

// ── Single preset pad ─────────────────────────────────────────────────────────
function PresetPad({ preset, selected, onSelect }) {
  const ref = useRef(null)

  const onMouseMove = (e) => {
    const el = ref.current
    if (!el) return
    const r  = el.getBoundingClientRect()
    const x  = (e.clientX - r.left) / r.width  - 0.5
    const y  = (e.clientY - r.top)  / r.height - 0.5
    el.style.transform = `perspective(360px) rotateX(${-y * 14}deg) rotateY(${x * 14}deg) scale(${selected ? 1.06 : 1.04}) translateZ(4px)`
    el.style.transition = 'transform 0.06s ease'
  }

  const onMouseLeave = () => {
    const el = ref.current
    if (!el) return
    el.style.transform = selected
      ? 'perspective(360px) rotateX(0deg) rotateY(0deg) scale(1.03) translateZ(2px)'
      : 'perspective(360px) rotateX(0deg) rotateY(0deg) scale(1) translateZ(0px)'
    el.style.transition = 'transform 0.35s cubic-bezier(0.4,0,0.2,1)'
  }

  return (
    <motion.button
      ref={ref}
      onClick={() => onSelect(preset)}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      className="pad-press relative flex flex-col items-start text-left"
      style={{
        background: selected
          ? `linear-gradient(145deg, ${preset.color}18, ${preset.color}08)`
          : 'rgba(255,255,255,0.025)',
        border: `1px solid ${selected ? preset.color + '55' : 'rgba(255,255,255,0.07)'}`,
        borderRadius: 14,
        padding: '14px 12px 12px',
        cursor: 'pointer',
        boxShadow: selected
          ? `0 0 0 1px ${preset.color}30, 0 0 24px ${preset.color}25, 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.10)`
          : 'inset 0 1px 0 rgba(255,255,255,0.04), 0 4px 16px rgba(0,0,0,0.5)',
        transition: 'border-color 0.2s, box-shadow 0.2s, background 0.2s',
        minHeight: 90,
        willChange: 'transform',
      }}
      initial={false}
      animate={{ scale: selected ? 1.03 : 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 24 }}
      whileTap={{ scale: 0.92 }}
    >
      {/* Status LED */}
      <motion.div
        style={{
          position: 'absolute',
          top: 8, right: 8,
          width: 6, height: 6,
          borderRadius: '50%',
          background: selected ? preset.color : 'rgba(255,255,255,0.08)',
          boxShadow: selected ? `0 0 8px ${preset.color}, 0 0 16px ${preset.color}80` : 'none',
        }}
        animate={selected ? { opacity: [1, 0.5, 1] } : { opacity: 1 }}
        transition={{ duration: 1.6, repeat: selected ? Infinity : 0 }}
      />

      {/* Symbol */}
      <span style={{
        fontSize: 18,
        color: selected ? preset.color : 'rgba(255,255,255,0.35)',
        display: 'block',
        marginBottom: 8,
        lineHeight: 1,
        transition: 'color 0.2s',
        fontFamily: 'ui-monospace, monospace',
      }}>
        {preset.sym}
      </span>

      {/* Label */}
      <p style={{
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: '0.08em',
        color: selected ? preset.color : 'rgba(255,255,255,0.65)',
        marginBottom: 3,
        transition: 'color 0.2s',
      }}>
        {preset.label.toUpperCase()}
      </p>

      {/* Description */}
      <p style={{
        fontSize: 8.5,
        color: selected ? `${preset.color}99` : 'rgba(255,255,255,0.25)',
        letterSpacing: '0.04em',
        transition: 'color 0.2s',
        lineHeight: 1.4,
      }}>
        {preset.desc}
      </p>
    </motion.button>
  )
}

// ── Soundboard pad section ─────────────────────────────────────────────────────
function PresetBoard({ selected, onSelect }) {
  const gridRef = useRef(null)

  // anime.js: cascade the 8 pads in as a 4x2 grid wave on mount
  useEffect(() => {
    if (gridRef.current) {
      gridRise(Array.from(gridRef.current.children),
               { grid: [4, 2], delay: 350 })
    }
  }, [])

  return (
    <div>
      {/* Section label row */}
      <div className="flex items-center gap-3 mb-4">
        <div style={{ height: 1, flex: 1, background: 'rgba(255,255,255,0.06)' }} />
        <span style={{
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: '0.18em',
          color: 'rgba(255,255,255,0.25)',
          fontFamily: 'ui-monospace, monospace',
        }}>
          SELECT PRESET
        </span>
        <div style={{ height: 1, flex: 1, background: 'rgba(255,255,255,0.06)' }} />
      </div>

      <div
        ref={gridRef}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 8,
        }}
      >
        {PRESETS.map((p) => (
          <PresetPad
            key={p.id}
            preset={p}
            selected={selected?.id === p.id}
            onSelect={onSelect}
          />
        ))}
      </div>

      {/* Octave + component hint */}
      <AnimatePresence mode="wait">
        {selected && (
          <motion.div
            key={selected.id}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="flex items-center justify-center gap-4 mt-4"
          >
            <span style={{ fontSize: 10, color: `${selected.color}bb`, fontFamily: 'ui-monospace,monospace', letterSpacing: '0.06em' }}>
              {selected.octaves} OCTAVES
            </span>
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.15)', fontFamily: 'ui-monospace,monospace' }}>·</span>
            <span style={{ fontSize: 10, color: `${selected.color}bb`, fontFamily: 'ui-monospace,monospace', letterSpacing: '0.06em' }}>
              {selected.components} NMF COMPONENTS
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Main landing page ─────────────────────────────────────────────────────────
export default function LandingPage() {
  const [dragging,  setDragging]  = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress,  setProgress]  = useState(0)
  const [error,     setError]     = useState(null)
  const [preset,    setPreset]    = useState(PRESETS[0])   // Standard is default

  const inputRef    = useRef(null)
  const kickerRef   = useRef(null)
  const subtitleRef = useRef(null)
  const navigate    = useNavigate()

  // anime.js: letter-by-letter reveal on the kicker + subtitle.  The h1
  // keeps its framer fade — background-clip:text gradients don't survive
  // per-letter span splitting.
  useEffect(() => {
    letterReveal(kickerRef.current,   { delay: 300, stagger: 22 })
    letterReveal(subtitleRef.current, { delay: 750, stagger: 12, translate: 10 })
  }, [])

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setError(null)
    setUploading(true)
    setProgress(0)
    try {
      const result = await uploadAudio(file, setProgress)
      navigate(`/processing/${result.session_id}`, {
        state: {
          filename: file.name,
          duration: result.duration_seconds,
          preset,           // ← the chosen preset travels with the session
        },
      })
    } catch (e) {
      setError(e?.response?.data?.detail || 'Upload failed. Please try again.')
      setUploading(false)
    }
  }, [navigate, preset])

  const onDrop      = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  return (
    <div className="relative min-h-screen flex flex-col items-center pt-24 pb-40">
      <ScrollStage sections={SECTIONS} />
      <SectionRail sections={SECTIONS} />

      {/* ── Hero ── */}
      <motion.div
        id="hero"
        className="relative z-10 text-center mb-14 px-4"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: [0.4, 0, 0.2, 1] }}
      >
        <motion.div
          className="float"
          style={{ display: 'inline-block', marginBottom: 8 }}
        >
          <span ref={kickerRef} style={{
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: '0.22em',
            color: 'rgba(245,158,11,0.6)',
            fontFamily: 'ui-monospace, monospace',
          }}>
            ◈ AI PIANO ARRANGEMENT ◈
          </span>
        </motion.div>

        <h1
          className="text-gradient-animated font-black tracking-tighter leading-none select-none"
          style={{
            fontSize: 'clamp(52px, 9vw, 100px)',
            lineHeight: 0.92,
            marginBottom: 20,
            letterSpacing: '-0.03em',
          }}
        >
          Arrangement<br />Transformer
        </h1>

        <p ref={subtitleRef} style={{
          fontSize: 18,
          color: 'rgba(255,255,255,0.38)',
          fontWeight: 300,
          letterSpacing: '0.01em',
        }}>
          Any song. Any keyboard. Full arrangement.
        </p>

        {/* Keys fly in scattered and assemble into a strip; they drift
            apart again ("exploded view") as the hero scrolls away */}
        <div style={{ marginTop: 36 }}>
          <KeyboardAssembly />
        </div>
      </motion.div>

      {/* ── Main card: preset board + upload ── */}
      <motion.div
        id="console"
        className="relative z-10 w-full px-4"
        style={{ maxWidth: 560 }}
        initial={{ opacity: 0, y: 48 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.15, ease: [0.4, 0, 0.2, 1] }}
      >
        <GlassCard tilt tiltStrength={3} className="p-7">

          {/* Preset board */}
          {!uploading && (
            <div className="mb-7">
              <PresetBoard selected={preset} onSelect={setPreset} />
            </div>
          )}

          {/* Divider */}
          {!uploading && (
            <div style={{ height: 1, background: 'rgba(255,255,255,0.05)', marginBottom: 24 }} />
          )}

          {/* Upload zone */}
          {!uploading ? (
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
            >
              {/* Drop area */}
              <motion.div
                className="rounded-xl flex flex-col items-center gap-3 py-8 px-6 mb-5"
                style={{
                  border: `1.5px dashed ${dragging ? (preset?.color || 'rgba(245,158,11,0.6)') : 'rgba(255,255,255,0.08)'}`,
                  background: dragging ? `${preset?.color || '#f59e0b'}06` : 'rgba(255,255,255,0.01)',
                  transition: 'all 0.2s ease',
                  cursor: 'default',
                }}
                animate={dragging ? { scale: 1.01 } : { scale: 1 }}
              >
                {/* Animated icon */}
                <motion.div
                  className="float"
                  style={{
                    width: 52, height: 52,
                    borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 22,
                    background: `linear-gradient(135deg, ${preset?.color || '#f59e0b'}20, ${preset?.color || '#f59e0b'}08)`,
                    border: `1px solid ${preset?.color || '#f59e0b'}30`,
                    boxShadow: `0 0 20px ${preset?.color || '#f59e0b'}20`,
                  }}
                >
                  🎹
                </motion.div>

                <div className="text-center">
                  <p className="text-sm font-medium text-white mb-1">
                    Drop your recording here
                  </p>
                  <p style={{ fontSize: 11, color: 'rgba(255,255,255,0.28)', letterSpacing: '0.04em' }}>
                    MP3 · WAV · M4A · OGG · FLAC · AIFF
                  </p>
                </div>
              </motion.div>

              {/* CTA */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => inputRef.current?.click()}
                  className="flex-1 py-3 rounded-xl text-sm font-bold tracking-wide transition-all duration-200 active:scale-95 relative overflow-hidden"
                  style={{
                    background: `linear-gradient(135deg, ${preset?.color || '#f59e0b'} 0%, ${preset?.color || '#d97706'}cc 100%)`,
                    color: '#0a0a0b',
                    boxShadow: `0 4px 24px ${preset?.color || '#f59e0b'}40, 0 0 0 1px ${preset?.color || '#f59e0b'}20`,
                  }}
                >
                  <span className="shimmer absolute inset-0 rounded-xl" />
                  <span className="relative">Choose File</span>
                </button>

                <div
                  style={{
                    fontSize: 10,
                    color: 'rgba(255,255,255,0.22)',
                    fontFamily: 'ui-monospace,monospace',
                    letterSpacing: '0.06em',
                    lineHeight: 1.5,
                    textAlign: 'center',
                    padding: '0 8px',
                  }}
                >
                  PRESET<br />
                  <span style={{ color: preset?.color || '#f59e0b' }}>{preset?.label.toUpperCase()}</span>
                </div>
              </div>

              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED}
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />

              <AnimatePresence>
                {error && (
                  <motion.p
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mt-4 text-sm text-red-400 text-center"
                  >
                    {error}
                  </motion.p>
                )}
              </AnimatePresence>
            </div>
          ) : (
            /* Uploading state */
            <div className="flex flex-col items-center gap-5 py-6">
              <AudioBars color={preset?.color || '#22c55e'} height={40} />
              <p className="text-sm font-semibold" style={{ color: preset?.color || '#f59e0b' }}>
                Uploading…
              </p>
              <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: `linear-gradient(90deg, ${preset?.color || '#f59e0b'}, ${preset?.color || '#d97706'})` }}
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ ease: 'easeOut' }}
                />
              </div>
              <p style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', fontFamily: 'ui-monospace,monospace' }}>
                {progress}%
              </p>
            </div>
          )}
        </GlassCard>
      </motion.div>

      {/* ── The pipeline, told by scrolling ──
          This replaces the old static SignalChain section: both told the
          same story, and one where the material visibly persists through
          every stage tells it far better.  Deliberately NOT wrapped in
          RackSection — RackSection applies a parallax transform, and a
          transformed ancestor becomes the containing block for the sticky
          panel inside, which breaks the pin. */}
      <div className="relative z-10 w-full mt-36">
        <PipelineScrolly id="pipeline" />
      </div>

      {/* ── The engine ── */}
      <div className="relative z-10 w-full px-4 mt-36"
           style={{ maxWidth: 900, marginLeft: 'auto', marginRight: 'auto' }}>
        <RackSection
          id="engine"
          label="THE ENGINE"
          color="#34d399"
          subtitle="Scroll drives the dial — the page position is the knob."
        >
          <div className="flex justify-center">
            <MasterDial size={400} />
          </div>
        </RackSection>
      </div>
    </div>
  )
}
