import { useRef, useState, useCallback } from 'react'

/**
 * BubbleCluster — round nodes that magnify toward the pointer.
 *
 * The Apple Watch honeycomb / macOS Dock mechanic: every bubble's scale is a
 * function of its DISTANCE from the pointer, not a binary hover state.  That
 * distinction is the whole feel — with a plain :hover the thing under the
 * cursor pops and everything else sits still, which reads as a button grid.
 * With a distance falloff the cluster deforms continuously, so neighbours
 * swell slightly as you approach and the field behaves like a surface being
 * pushed rather than a set of targets being hit.
 *
 * The focused bubble also rises (z-index and shadow) and desaturates its
 * neighbours, which is the "spotlight" half: attention is taken from the
 * others rather than merely added to one.
 *
 * Layout is an offset honeycomb — each row shifted half a cell — because a
 * square grid re-reads as a toolbar the moment the bubbles stop moving.
 */

const FALLOFF = 132      // px; how far the magnification reaches
const MAX_SCALE = 1.42   // scale directly under the pointer
const MIN_SCALE = 0.9    // scale for bubbles far from it, when any is near

export default function BubbleCluster({
  items,                 // [{ key, label, sub, symbol, color, done, active }]
  perRow = 4,
  size = 92,
  gap = 16,
  onSelect,
  selectedKey,
}) {
  const wrapRef = useRef(null)
  const [pointer, setPointer] = useState(null)
  const [focus, setFocus] = useState(null)

  const onMove = useCallback((e) => {
    const el = wrapRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPointer({ x: e.clientX - r.left, y: e.clientY - r.top })
  }, [])

  // Balanced rows, widest in the middle, every row centred.
  //
  // The old offset honeycomb split 7 items into 3/3/1 and shifted alternate
  // rows half a cell, which left a lone bubble hanging off one side.  A
  // cluster only reads as deliberate if it is symmetric about its own axis,
  // so rows are now sized as evenly as possible and then re-ordered into a
  // diamond — 7 becomes 2/3/2 rather than 3/3/1.
  const rows = (() => {
    const n = items.length
    if (n <= perRow) return [items]
    const nRows = Math.ceil(n / perRow)
    const base = Math.floor(n / nRows)
    const extra = n % nRows
    const sizes = Array.from({ length: nRows }, (_, i) => base + (i < extra ? 1 : 0))
    sizes.sort((a, b) => a - b)              // ascending
    // Re-lay ascending sizes outward from the centre so the widest row is
    // in the middle and the shape mirrors top-to-bottom.
    const arranged = new Array(nRows)
    let lo = 0, hi = nRows - 1
    for (let i = 0; i < nRows; i++) {
      if (i % 2 === 0) arranged[lo++] = sizes[i]
      else arranged[hi--] = sizes[i]
    }
    const out = []
    let k = 0
    for (const sz of arranged) { out.push(items.slice(k, k + sz)); k += sz }
    return out
  })()

  return (
    <div
      ref={wrapRef}
      onMouseMove={onMove}
      onMouseLeave={() => { setPointer(null); setFocus(null) }}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap }}
    >
      {rows.map((row, ri) => (
        <div
          key={ri}
          style={{
            display: 'flex',
            gap,
            justifyContent: 'center',
          }}
        >
          {row.map((it) => {
            const isSel = selectedKey != null && it.key === selectedKey
            const accent = it.color || 'var(--ct-accent)'
            return (
              <Bubble
                key={it.key}
                item={it}
                size={size}
                accent={accent}
                selected={isSel}
                pointer={pointer}
                dimmed={focus != null && focus !== it.key}
                onEnter={() => setFocus(it.key)}
                onSelect={onSelect}
              />
            )
          })}
        </div>
      ))}
    </div>
  )
}

function Bubble({ item, size, accent, selected, pointer, dimmed, onEnter, onSelect }) {
  const ref = useRef(null)

  // Distance-based magnification, recomputed on the parent's pointer move.
  let scale = 1
  if (pointer && ref.current) {
    const el = ref.current
    const cx = el.offsetLeft + el.offsetWidth / 2
    const cy = el.offsetTop + el.offsetHeight / 2
    const parent = el.offsetParent
    const ox = parent ? parent.offsetLeft : 0
    const oy = parent ? parent.offsetTop : 0
    const d = Math.hypot(pointer.x - (cx + ox), pointer.y - (cy + oy))
    const t = Math.max(0, 1 - d / FALLOFF)
    // Smoothstep so the swell eases in rather than tracking distance linearly.
    const e = t * t * (3 - 2 * t)
    scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * e
  }

  const lit = selected || item.active
  return (
    <button
      ref={ref}
      onMouseEnter={onEnter}
      onClick={() => onSelect?.(item)}
      title={item.label}
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        flexShrink: 0,
        border: 'none',
        // No outline at all.  A hard ring competed with the strings behind it
        // and made every bubble read as a button; a soft radial body reads as
        // a light source sitting in the space instead.
        background: lit
          ? `radial-gradient(circle at 50% 34%, ${accent}3d, ${accent}14 55%, rgba(255,255,255,0.02) 100%)`
          : 'radial-gradient(circle at 50% 34%, rgba(255,255,255,0.075), rgba(255,255,255,0.022) 70%, transparent)',
        color: lit ? accent : 'rgba(255,255,255,0.82)',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 3,
        cursor: 'pointer',
        transform: `scale(${scale})`,
        // No transition on transform: it is already driven continuously by
        // the pointer, and easing it as well makes the cluster feel laggy.
        transition: 'border-color .35s, background .35s, color .35s, opacity .35s, box-shadow .35s',
        opacity: dimmed ? 0.42 : 1,
        boxShadow: lit
          ? `0 0 34px ${accent}42, 0 0 90px ${accent}1c`
          : '0 6px 24px rgba(0,0,0,0.45)',
        zIndex: Math.round(scale * 10),
        position: 'relative',
        backdropFilter: 'blur(6px)',
      }}
    >
      {item.symbol && (
        <span style={{
          fontSize: size * 0.30, lineHeight: 1, fontFamily: 'ui-monospace,monospace',
          fontWeight: 700,
          // The 3D field behind is bright and moving, so a thin glyph
          // disappears into it.  Bigger, heavier, and carrying its own
          // shadow so it holds an edge wherever it lands.
          textShadow: lit
            ? `0 0 18px ${accent}, 0 2px 6px rgba(0,0,0,0.8)`
            : '0 2px 8px rgba(0,0,0,0.9)',
          marginBottom: 2,
        }}>
          {item.done ? '✓' : item.symbol}
        </span>
      )}
      <span style={{
        fontSize: Math.max(8, size * 0.105),
        fontWeight: 700, letterSpacing: '0.14em',
        fontFamily: 'ui-monospace,monospace', textTransform: 'uppercase',
        textShadow: '0 2px 8px rgba(0,0,0,0.9)',
        opacity: lit ? 1 : 0.72,
      }}>
        {item.label}
      </span>
      {item.sub && (
        <span style={{
          fontSize: Math.max(7, size * 0.085),
          color: 'rgba(255,255,255,0.30)', letterSpacing: '0.06em',
        }}>
          {item.sub}
        </span>
      )}
    </button>
  )
}
