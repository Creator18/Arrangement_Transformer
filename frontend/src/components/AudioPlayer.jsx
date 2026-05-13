import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AudioBars from './AudioBars'

function fmt(s) {
  if (!isFinite(s)) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

// VU Meter: 8 bars + LED strip that animate as a spectrum analyzer when playing
function VUMeter({ playing }) {
  const bars = [
    { peak: 60,  color: '#22c55e' },
    { peak: 80,  color: '#22c55e' },
    { peak: 95,  color: '#22c55e' },
    { peak: 75,  color: '#22c55e' },
    { peak: 100, color: '#22c55e' },
    { peak: 85,  color: '#f59e0b' },
    { peak: 65,  color: '#f59e0b' },
    { peak: 50,  color: '#ef4444' },
  ]

  return (
    <div
      className="flex items-end gap-[3px]"
      style={{ height: 32, filter: playing ? 'drop-shadow(0 0 5px rgba(34,197,94,0.5))' : 'none' }}
      aria-hidden
    >
      {bars.map((b, i) =>
        playing ? (
          <motion.span
            key={i}
            style={{ width: 5, borderRadius: 2, background: b.color, display: 'block' }}
            animate={{ height: [4, b.peak * 0.32, b.peak * 0.18, b.peak * 0.28, 4] }}
            transition={{
              duration: 0.45 + i * 0.07,
              delay: i * 0.055,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />
        ) : (
          <span
            key={i}
            style={{ width: 5, height: 4, borderRadius: 2, background: '#27272a', display: 'block' }}
          />
        )
      )}
    </div>
  )
}

export default function AudioPlayer({ tracks }) {
  // tracks: [{ label, url, downloadName }]
  const [activeIdx, setActiveIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loading, setLoading] = useState(false)
  const audioRef = useRef((() => {
    // preload='auto' means: as soon as src is set, start downloading the full
    // file.  ProcessingPage pre-warms the HTTP cache for the main previews
    // before navigating here, so this request is usually served from cache —
    // first-click playback feels instant.
    const a = new Audio()
    a.preload = 'auto'
    return a
  })())
  const seekRef = useRef(null)

  const track = tracks[activeIdx]

  // Wire up audio events once
  useEffect(() => {
    const audio = audioRef.current
    const onTimeUpdate = () => setCurrentTime(audio.currentTime)
    const onDurationChange = () => setDuration(audio.duration)
    const onEnded = () => setPlaying(false)
    const onCanPlay = () => setLoading(false)
    const onWaiting = () => setLoading(true)

    audio.addEventListener('timeupdate', onTimeUpdate)
    audio.addEventListener('durationchange', onDurationChange)
    audio.addEventListener('ended', onEnded)
    audio.addEventListener('canplay', onCanPlay)
    audio.addEventListener('waiting', onWaiting)
    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate)
      audio.removeEventListener('durationchange', onDurationChange)
      audio.removeEventListener('ended', onEnded)
      audio.removeEventListener('canplay', onCanPlay)
      audio.removeEventListener('waiting', onWaiting)
      audio.pause()
    }
  }, [])

  // Load new track whenever active index changes
  useEffect(() => {
    const audio = audioRef.current
    const wasPlaying = !audio.paused
    audio.pause()
    setPlaying(false)
    setCurrentTime(0)
    setDuration(0)
    setLoading(true)
    audio.src = track.url
    audio.load()
    if (wasPlaying) {
      audio.play().then(() => setPlaying(true)).catch(() => {})
    }
  }, [activeIdx])

  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (playing) {
      audio.pause()
      setPlaying(false)
    } else {
      setLoading(true)
      audio.play()
        .then(() => { setPlaying(true); setLoading(false) })
        .catch(() => setLoading(false))
    }
  }, [playing])

  const onSeek = (e) => {
    const audio = audioRef.current
    const val = parseFloat(e.target.value)
    audio.currentTime = val
    setCurrentTime(val)
  }

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="space-y-4">
      {/* Track selector tabs */}
      <div className="flex gap-2 flex-wrap">
        {tracks.map((t, i) => (
          <button
            key={t.label}
            onClick={() => setActiveIdx(i)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 border ${
              activeIdx === i
                ? 'bg-amber-500/15 text-amber-400 border-amber-500/40 amber-glow'
                : 'bg-white/5 text-zinc-400 border-white/10 hover:text-zinc-200 hover:bg-white/8'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* VU Meter strip */}
      <div
        className="flex items-center gap-3 px-3 py-2 rounded-xl"
        style={{ background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        <span className="text-[9px] font-bold tracking-widest text-zinc-600 uppercase select-none">VU</span>
        <VUMeter playing={playing} />
        <div className="flex-1" />
        {/* Level LED indicators */}
        <div className="flex gap-1" aria-hidden>
          {['#22c55e', '#22c55e', '#22c55e', '#f59e0b', '#f59e0b', '#ef4444'].map((c, i) => (
            <motion.span
              key={i}
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: playing ? c : '#27272a',
                display: 'block',
              }}
              animate={playing ? {
                opacity: [0.35, 1, 0.35],
                boxShadow: [`0 0 4px ${c}40`, `0 0 8px ${c}80`, `0 0 4px ${c}40`],
              } : { opacity: 0.25, boxShadow: 'none' }}
              transition={{ duration: 0.3 + i * 0.08, delay: i * 0.04, repeat: Infinity, ease: 'easeInOut' }}
            />
          ))}
        </div>
        <span className="text-[9px] font-bold tracking-widest text-zinc-600 uppercase select-none">OUT</span>
      </div>

      {/* Transport */}
      <div className="flex items-center gap-4">
        {/* Play / pause button */}
        <button
          onClick={togglePlay}
          disabled={loading}
          className="w-11 h-11 rounded-full flex items-center justify-center flex-shrink-0 active:scale-95 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            boxShadow: playing
              ? '0 0 16px rgba(245,158,11,0.5), 0 4px 12px rgba(0,0,0,0.4)'
              : '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.span key="spin" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : playing ? (
              <motion.svg key="pause" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </motion.svg>
            ) : (
              <motion.svg key="play" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="w-4 h-4 text-white ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </motion.svg>
            )}
          </AnimatePresence>
        </button>

        {/* Waveform + seek */}
        <div className="flex-1 min-w-0 space-y-1.5">
          <div className="flex items-end justify-between gap-3">
            <div className="flex-1">
              {playing
                ? <AudioBars color="#22c55e" height={20} />
                : <div className="flex items-end gap-[3px] h-5">
                    {Array.from({ length: 14 }, (_, i) => (
                      <span key={i} style={{
                        width: 4, borderRadius: 2, display: 'block', background: '#3f3f46',
                        height: [4, 7, 5, 10, 6, 14, 8, 12, 5, 9, 6, 11, 4, 8][i] ?? 5
                      }} />
                    ))}
                  </div>
              }
            </div>
            <span className="text-[11px] text-zinc-400 tabular-nums flex-shrink-0">
              {fmt(currentTime)} / {fmt(duration)}
            </span>
          </div>

          {/* Seek bar — amber */}
          <div className="relative w-full h-1.5 group">
            <div className="absolute inset-0 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
              <div
                className="h-full rounded-full transition-none"
                style={{
                  width: `${pct}%`,
                  background: 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)',
                  boxShadow: pct > 0 ? '0 0 6px rgba(245,158,11,0.5)' : 'none',
                }}
              />
            </div>
            <input
              ref={seekRef}
              type="range"
              min={0}
              max={duration || 0}
              step={0.01}
              value={currentTime}
              onChange={onSeek}
              className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
            />
          </div>
        </div>

        {/* Download */}
        <a
          href={track.url}
          download={track.downloadName}
          className="text-[11px] text-amber-400 hover:text-amber-300 hover:underline flex-shrink-0 transition-colors"
        >
          ↓ WAV
        </a>
      </div>
    </div>
  )
}
