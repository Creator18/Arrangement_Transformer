import { useEffect, useRef, useState } from 'react'
import GlassCard from './GlassCard'
import anime from '../utils/animefx'
import { onScrollScrub, whenInView } from '../utils/scrollfx'

/**
 * SignalChain — "How it works" as a scroll-driven rack.
 *
 * A patch-cable SVG path runs down the left rail and draws itself as you
 * scroll.  Each module card lights up (border glow + elastic nudge) when
 * it enters the viewport, and its little VU meter starts running — the
 * rack powers on section by section as the signal reaches it.
 */

const MODULES = [
  {
    num: '01',
    color: '#f59e0b',
    sym: '▲',
    title: 'Upload',
    desc: 'Drop a piano recording or a full song, in any format. We convert and normalise it automatically.',
  },
  {
    num: '02',
    color: '#a78bfa',
    sym: '◈',
    title: 'Hear',
    desc: 'A pitch-aligned spectral transformer, trained by us, hears the notes. Solo piano is transcribed directly (≈0.955 note F1 on real recordings); a song goes to the same model fine-tuned on human piano covers, which keeps what a pianist would play.',
  },
  {
    num: '03',
    color: '#06b6d4',
    sym: '▶',
    title: 'Fit',
    desc: 'The arrangement is fitted to your keyboard preset and split between two hands. Next: a model trained on thousands of arrangements rendered as solo piano and as full band mixes, so any song points to one clear piano part.',
  },
]

const RAIL_W = 56           // left rail width holding the cable
const NODE_R = 7

function VUMeter({ color, running }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current || !running) return
    const bars = Array.from(ref.current.children)
    const anims = bars.map((bar, i) => anime({
      targets: bar,
      height: () => [4 + anime.random(2, 6), 6 + anime.random(10, 22)],
      duration: () => anime.random(260, 520),
      direction: 'alternate',
      loop: true,
      easing: 'easeInOutSine',
      delay: i * 60,
    }))
    return () => anims.forEach(a => a.pause())
  }, [running])

  return (
    <div ref={ref} className="flex items-end gap-[3px]" style={{ height: 26 }}>
      {Array.from({ length: 7 }, (_, i) => (
        <span key={i} style={{
          width: 3.5, height: 5, borderRadius: 2,
          background: color,
          boxShadow: `0 0 5px ${color}90`,
          display: 'block',
        }} />
      ))}
    </div>
  )
}

function ModuleCard({ mod, index, onVisible, visible }) {
  const cardRef = useRef(null)

  useEffect(() => {
    const el = cardRef.current
    if (!el) return
    return whenInView(el, () => {
      onVisible(index)
      anime({
        targets: el,
        translateX: [24, 0],
        opacity: [0, 1],
        duration: 750,
        easing: 'easeOutCubic',
        complete: () => { el.style.transform = ''; el.style.opacity = '' },
      })
      // Power-on border flash in the module's colour
      anime({
        targets: el,
        boxShadow: [
          `0 0 0px ${mod.color}00`,
          `0 0 34px ${mod.color}45`,
          `0 0 18px ${mod.color}20`,
        ],
        duration: 1200,
        easing: 'easeOutQuad',
      })
    }, { threshold: 0.45 })
  }, [index, mod.color, onVisible])

  return (
    <div ref={cardRef} style={{ opacity: 0 }}>
      <GlassCard className="p-6 relative overflow-hidden" style={{ cursor: 'default' }}>
        {/* Background number */}
        <span style={{
          position: 'absolute', right: 16, top: 12,
          fontSize: 64, fontWeight: 900,
          color: `${mod.color}0a`,
          fontFamily: 'ui-monospace,monospace',
          userSelect: 'none', lineHeight: 1,
        }}>
          {mod.num}
        </span>

        <div className="flex items-start justify-between gap-4">
          <div>
            <div style={{
              width: 40, height: 40, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: `${mod.color}12`,
              border: `1px solid ${mod.color}30`,
              boxShadow: `0 0 16px ${mod.color}15`,
              marginBottom: 14,
              fontSize: 16, color: mod.color,
              fontFamily: 'ui-monospace,monospace',
            }}>
              {mod.sym}
            </div>
            <h3 className="text-base font-bold mb-2" style={{ color: '#ffffff' }}>
              {mod.title}
            </h3>
            <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.38)', lineHeight: 1.6, maxWidth: 460 }}>
              {mod.desc}
            </p>
          </div>
          {/* Live VU once the module has powered on */}
          <div style={{ paddingTop: 4, opacity: visible ? 1 : 0.15, transition: 'opacity 0.6s' }}>
            <VUMeter color={mod.color} running={visible} />
          </div>
        </div>
      </GlassCard>
    </div>
  )
}

export default function SignalChain() {
  const rootRef  = useRef(null)
  const cableRef = useRef(null)
  const nodesRef = useRef([])
  const [vis, setVis] = useState([false, false, false])

  const setNodeLit = (i, color) => {
    const node = nodesRef.current[i]
    if (!node || node.dataset.lit) return
    node.dataset.lit = '1'
    node.style.fill = color
    node.style.filter = `drop-shadow(0 0 8px ${color})`
    anime({
      targets: node,
      scale: [1, 1.8, 1],
      duration: 700,
      easing: 'easeOutElastic(1, .6)',
    })
  }

  useEffect(() => {
    const root  = rootRef.current
    const cable = cableRef.current
    if (!root || !cable) return
    const len = cable.getTotalLength()
    cable.style.strokeDasharray  = `${len}`
    cable.style.strokeDashoffset = `${len}`

    // The cable draws with scroll — signal flowing down the rack.
    const stop = onScrollScrub(root, (p) => {
      const drawn = Math.max(0, Math.min(1, (p - 0.12) * 1.7))
      cable.style.strokeDashoffset = `${len * (1 - drawn)}`
    })
    return stop
  }, [])

  const onVisible = (i) => {
    setVis(v => {
      if (v[i]) return v
      const next = [...v]
      next[i] = true
      return next
    })
    setNodeLit(i, MODULES[i].color)
  }

  // Node vertical positions as fractions of the rail height
  const nodeYs = ['16%', '50%', '84%']

  return (
    <div ref={rootRef} className="relative" style={{ display: 'flex', gap: 0 }}>
      {/* Left rail: patch cable + nodes */}
      <div style={{ width: RAIL_W, position: 'relative', flexShrink: 0 }}>
        <svg width={RAIL_W} height="100%"
             style={{ position: 'absolute', inset: 0, overflow: 'visible' }}
             preserveAspectRatio="none" viewBox="0 0 56 300">
          {/* Dim guide track */}
          <path d="M 28 0 V 300" stroke="rgba(255,255,255,0.05)" strokeWidth="2" fill="none" />
          {/* Live cable — drawn by scroll */}
          <path ref={cableRef} d="M 28 0 V 300"
                stroke="url(#chain-grad)" strokeWidth="2.5" fill="none"
                style={{ filter: 'drop-shadow(0 0 4px rgba(245,158,11,0.5))' }} />
          <defs>
            <linearGradient id="chain-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"  stopColor="#f59e0b" />
              <stop offset="50%" stopColor="#a78bfa" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>
          {/* Connection nodes */}
          {nodeYs.map((y, i) => (
            <circle key={i}
                    ref={el => { nodesRef.current[i] = el }}
                    cx="28" cy={y} r={NODE_R}
                    fill="rgba(255,255,255,0.10)"
                    stroke="rgba(255,255,255,0.18)" strokeWidth="1.5"
                    style={{ transformBox: 'fill-box', transformOrigin: 'center',
                             transition: 'fill 0.4s' }} />
          ))}
        </svg>
      </div>

      {/* Module cards */}
      <div className="flex-1 flex flex-col gap-10 py-2">
        {MODULES.map((m, i) => (
          <ModuleCard key={m.title} mod={m} index={i}
                      onVisible={onVisible} visible={vis[i]} />
        ))}
      </div>
    </div>
  )
}
