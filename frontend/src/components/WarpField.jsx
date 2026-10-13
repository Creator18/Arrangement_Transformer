import { useMemo } from 'react'

/**
 * WarpField — the results-page backdrop: a perspective tunnel of four gridded
 * walls with light beams travelling along them toward the viewer.
 *
 * Inspired by Magic UI's WarpBackground, rebuilt rather than imported (the
 * project has no shadcn/ui setup, and it needed changes to sit behind a live
 * piano roll):
 *
 *   - every moving part is a CSS transform animation, so it composites on the
 *     GPU and never competes with the roll's per-frame work (the roll fps rig
 *     is why that matters);
 *   - beams take the roll's register colours (bass / melody / voice) at low
 *     opacity, so the room is lit by the same palette as the notes;
 *   - it is dimmed toward the centre where the roll sits, so it frames the
 *     content rather than competing with it;
 *   - prefers-reduced-motion freezes the beams (the grid stays).
 */
const COLORS = ['#3b82f6', '#f87171', '#a78bfa']

function Beams({ count, duration, side }) {
  const beams = useMemo(() => Array.from({ length: count }, (_, i) => {
    const r = Math.sin((i + 1) * 9973.1 + side.length * 31.7) * 43758.5453
    const f = r - Math.floor(r)
    return {
      x: Math.round(((i + 0.5) / count) * 100 + (f - 0.5) * (60 / count)),
      delay: -(f * duration),
      dur: duration * (0.75 + f * 0.6),
      color: COLORS[i % COLORS.length],
      w: 1 + Math.round(f * 2),
    }
  }), [count, duration, side])
  return beams.map((b, i) => (
    <span key={i} className="warp-beam" style={{
      left: `${b.x}%`, width: `${b.w}%`, animationDuration: `${b.dur}s`, animationDelay: `${b.delay}s`,
      background: `linear-gradient(to top, transparent, ${b.color}cc 45%, ${b.color} 50%, transparent)`,
    }} />
  ))
}

export default function WarpField({ beamsPerSide = 4, duration = 6 }) {
  const sides = ['top', 'bottom', 'left', 'right']
  return (
    <div aria-hidden className="warp-field">
      <style>{`
        .warp-field { position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:0;
          perspective: 110px; container-type: size; }
        .warp-room { position:absolute; inset:0; transform-style: preserve-3d; }
        .warp-wall { position:absolute; overflow:hidden; transform-style: preserve-3d;
          background-image:
            linear-gradient(rgba(167,139,250,0.22) 1px, transparent 1px),
            linear-gradient(90deg, rgba(167,139,250,0.22) 1px, transparent 1px);
          background-size: 5% 5%; }
        .warp-top    { left:0; top:0;    width:100cqw; height:100cqmax; transform-origin: 50% 0%;   transform: rotateX(-90deg); }
        .warp-bottom { left:0; bottom:0; width:100cqw; height:100cqmax; transform-origin: 50% 100%; transform: rotateX(90deg); }
        .warp-left   { left:0; top:0;    width:100cqmax; height:100cqh; transform-origin: 0% 50%;   transform: rotateY(90deg); }
        .warp-right  { right:0; top:0;   width:100cqmax; height:100cqh; transform-origin: 100% 50%; transform: rotateY(-90deg); }
        .warp-beam { position:absolute; top:0; aspect-ratio: 1 / 8; opacity:0.85;
          animation-name: warp-run; animation-timing-function: linear; animation-iteration-count: infinite;
          will-change: transform; }
        .warp-left .warp-beam, .warp-right .warp-beam { animation-name: warp-run-x; top:auto;
          aspect-ratio: auto; height: 1.5%; width: 12% !important; left:auto !important; }
        @keyframes warp-run   { from { transform: translateY(100cqmax) } to { transform: translateY(-100%) } }
        @keyframes warp-run-x { from { transform: translateX(100cqmax) } to { transform: translateX(-100%) } }
        .warp-vignette { position:absolute; inset:0;
          background: radial-gradient(ellipse 60% 55% at 50% 50%, rgba(10,10,12,0.70), rgba(10,10,12,0.30) 60%, rgba(10,10,12,0))); }
        @media (prefers-reduced-motion: reduce) { .warp-beam { animation-play-state: paused } }
      `}</style>
      <div className="warp-room">
        {sides.map(s => (
          <div key={s} className={`warp-wall warp-${s}`}>
            {s === 'left' || s === 'right'
              ? Array.from({ length: beamsPerSide }, (_, i) => (
                  <span key={i} className="warp-beam" style={{
                    top: `${((i + 0.5) / beamsPerSide) * 100}%`, animationDuration: `${duration * (0.8 + (i % 3) * 0.2)}s`,
                    animationDelay: `${-(i * 1.7) % duration}s`,
                    background: `linear-gradient(to left, transparent, ${COLORS[i % 3]} 50%, transparent)`,
                  }} />
                ))
              : <Beams count={beamsPerSide} duration={duration} side={s} />}
          </div>
        ))}
      </div>
      <div className="warp-vignette" />
    </div>
  )
}
