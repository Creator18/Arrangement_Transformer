export default function CdefView({ cdef = '' }) {
  if (!cdef) return <p className="text-sm text-zinc-500">No CDEF data available.</p>

  const lines = cdef.split('\n')

  return (
    <div className="overflow-x-auto">
      <div
        className="inline-block min-w-full rounded-xl p-4"
        style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)' }}
      >
        {lines.map((line, i) => {
          const isRH = line.startsWith('RH:')
          const isLH = line.startsWith('LH:')
          const prefix = line.slice(0, 3)
          const content = line.slice(3)

          // Tokenise: beats, rests, note clusters, measure bars
          const tokens = content.trim().split(/\s+/)

          return (
            <div key={i} className="flex items-baseline gap-0 font-mono text-sm leading-8 whitespace-nowrap mb-1">
              <span className={`mr-3 text-xs font-semibold ${isRH ? 'text-[#f87171]' : 'text-[#3b82f6]'}`}>
                {prefix}
              </span>
              {tokens.map((tok, j) => {
                if (tok === '|') {
                  return (
                    <span key={j} className="mx-2 text-zinc-600 select-none">│</span>
                  )
                }
                if (tok === '.') {
                  return (
                    <span key={j} className="w-8 text-center text-zinc-600">·</span>
                  )
                }
                // Note cluster: C4-E4-G4
                const noteTokens = tok.split('-')
                return (
                  <span
                    key={j}
                    className={`w-8 text-center text-xs font-medium ${
                      isRH ? 'text-rose-400' : 'text-blue-400'
                    }`}
                    title={tok}
                  >
                    {noteTokens.length > 1 ? `${noteTokens[0]}…` : tok}
                  </span>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
