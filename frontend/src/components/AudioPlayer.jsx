import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AudioBars from './AudioBars'

function fmt(s) {
  if (!isFinite(s)) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function AudioPlayer({ tracks }) {
  // tracks: [{ label, url, downloadName }]
  const [activeIdx, setActiveIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loading, setLoading] = useState(false)
  const audioRef = useRef(new Audio())
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
    // If the user switched tracks while playing, auto-play the new one
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
    <div className="space-y-3">
      {/* Track selector tabs */}
      <div className="flex gap-2 flex-wrap">
        {tracks.map((t, i) => (
          <button
            key={t.label}
            onClick={() => setActiveIdx(i)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-200 ${
              activeIdx === i
                ? 'bg-[#0071e3] text-white shadow-sm'
                : 'bg-white/60 text-[#6e6e73] hover:bg-white/90 hover:text-[#1d1d1f]'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Player body */}
      <div className="flex items-center gap-4">
        {/* Play / pause button */}
        <button
          onClick={togglePlay}
          disabled={loading}
          className="w-11 h-11 rounded-full bg-[#0071e3] flex items-center justify-center flex-shrink-0
                     hover:bg-[#0077ed] active:scale-95 transition-all duration-150 shadow-sm
                     disabled:opacity-50 disabled:cursor-not-allowed"
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

        {/* Waveform / seek area */}
        <div className="flex-1 min-w-0 space-y-1.5">
          <div className="flex items-end justify-between gap-3">
            <div className="flex-1">
              {playing
                ? <AudioBars color="#0071e3" height={20} />
                : <div className="flex items-end gap-[3px] h-5">
                    {Array.from({ length: 14 }, (_, i) => (
                      <span key={i} style={{
                        width: 3, borderRadius: 2, display: 'block', background: '#d2d2d7',
                        height: [4, 7, 5, 10, 6, 14, 8, 12, 5, 9, 6, 11, 4, 8][i] ?? 5
                      }} />
                    ))}
                  </div>
              }
            </div>
            <span className="text-[11px] text-[#6e6e73] tabular-nums flex-shrink-0">
              {fmt(currentTime)} / {fmt(duration)}
            </span>
          </div>

          {/* Seek bar */}
          <div className="relative w-full h-1.5 group">
            <div className="absolute inset-0 rounded-full bg-[#e5e5ea] overflow-hidden">
              <div
                className="h-full bg-[#0071e3] rounded-full transition-none"
                style={{ width: `${pct}%` }}
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
          className="text-[11px] text-[#0071e3] hover:underline flex-shrink-0"
        >
          ↓ WAV
        </a>
      </div>
    </div>
  )
}
