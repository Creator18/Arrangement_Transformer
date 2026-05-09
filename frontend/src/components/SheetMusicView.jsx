import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// Lazy-load OSMD (it's ~2 MB) so it doesn't bloat the initial bundle
let _OSMD = null
async function getOSMD() {
  if (_OSMD) return _OSMD
  const mod = await import('opensheetmusicdisplay')
  _OSMD = mod.OpenSheetMusicDisplay
  return _OSMD
}

export default function SheetMusicView({ xmlUrl }) {
  const containerRef = useRef(null)
  const osmdRef      = useRef(null)
  const [state, setState] = useState('idle')   // idle | loading | ready | error
  const [errMsg, setErrMsg] = useState('')

  useEffect(() => {
    if (!xmlUrl) return
    let cancelled = false
    setState('loading')
    setErrMsg('')

    async function render() {
      try {
        const OpenSheetMusicDisplay = await getOSMD()
        if (cancelled) return

        // Clear previous render
        if (containerRef.current) containerRef.current.innerHTML = ''

        const osmd = new OpenSheetMusicDisplay(containerRef.current, {
          backend:               'svg',
          drawTitle:             true,
          drawSubtitle:          false,
          drawComposer:          false,
          drawCredits:           false,
          drawMetronomeMarks:    true,
          drawPartNames:         true,
          followCursor:          false,
          autoResize:            true,
          pageFormat:            'Endless',    // single scrollable page
          defaultFontFamily:     'Times New Roman, serif',
          // Keep notation on white (OSMD SVG colours are hardcoded)
        })

        osmdRef.current = osmd
        await osmd.load(xmlUrl)
        if (cancelled) return
        osmd.render()
        setState('ready')
      } catch (e) {
        if (!cancelled) {
          setErrMsg(e?.message || 'Failed to render sheet music.')
          setState('error')
        }
      }
    }

    render()
    return () => { cancelled = true }
  }, [xmlUrl])

  return (
    <div className="relative">
      {/* Loading overlay */}
      <AnimatePresence>
        {state === 'loading' && (
          <motion.div
            key="loader"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center gap-3 py-16"
          >
            <div className="w-8 h-8 rounded-full border-2 border-amber-500 border-t-transparent animate-spin" />
            <p className="text-sm text-zinc-500">Rendering sheet music…</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      {state === 'error' && (
        <div
          className="rounded-xl p-6 text-center"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <p className="text-sm font-medium text-red-400 mb-1">Sheet music unavailable</p>
          <p className="text-xs text-red-500 mb-3">{errMsg}</p>
          <a
            href={xmlUrl}
            download="sheet_music.xml"
            className="text-xs text-amber-400 hover:text-amber-300 underline"
          >
            Download MusicXML instead
          </a>
        </div>
      )}

      {/* OSMD renders into this div — keep it on white like real sheet paper */}
      <div
        ref={containerRef}
        style={{
          display: state === 'ready' ? 'block' : 'none',
          background: '#ffffff',
          borderRadius: 12,
          padding: '24px 16px',
          minHeight: 200,
          overflowX: 'auto',
        }}
      />

      {/* Download link below */}
      {state === 'ready' && (
        <div className="mt-3 text-center">
          <a
            href={xmlUrl}
            download="sheet_music.xml"
            className="text-xs text-zinc-600 hover:text-amber-400 transition-colors"
          >
            ↓ Download MusicXML
          </a>
        </div>
      )}
    </div>
  )
}
