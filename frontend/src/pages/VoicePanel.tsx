import * as React from "react";
import { Mic } from "lucide-react";

// ---- shape of /voice-data (backend/app/routers/progress.py: eval/voice_tests.json + live cluster status) ----
type PRF = { name: string; p: number; r: number; f1: number };
type Cond = { name: string; solo: number; fill: number; fill_p?: number; fill_r?: number };
type ValSet = { ap: number | null; p: number | null; r: number | null; f1: number | null; pos: number };
type VoiceVal = { step: number; score: number; [set: string]: ValSet | number };
type Job = { id: string; name: string; state: string; elapsed: string; limit: string; reason: string };
type VoiceData = {
  updated: string;
  aah: { note: string; rows: PRF[]; songs: [string, number][] };
  fill: { note: string; conditions: Cond[]; steps: { note: string; rows: PRF[] }; models: { name: string; solo: number; fill: number }[] };
  detect_rules: { note: string; rows: { name: string; plain_p: number; plain_r: number; pad_p: number; pad_r: number }[] };
  musdb: { note: string; duration: { margin: number; share: number }[]; starts: { margin: number; share: number }[] };
  live: { curves: number | null; songs: number | null; jobs: Job[]; vals: VoiceVal[]; trace: [number, number, number][]; info: string[]; finished: boolean; error?: string;
          run?: string; gpu?: string | null; ref?: Record<string, VoiceVal[]>; stale?: string };
};

// categorical order (validated for dark surfaces and colour-blind separation): before, after, third
const C1 = "#5b8def", C2 = "#22a374", C3 = "#b8822a";
const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

/** Grouped horizontal bars on a 0-1 (or 0-max) scale: one row per item, one bar per series, value at the bar end. */
export function HBars({ title, rows, series, max = 1, fmt = f3 }: {
  title: string; rows: { label: string; vals: number[] }[]; series: { name: string; color: string }[];
  max?: number; fmt?: (v: number) => string;
}) {
  const labelW = 150, w = 460, barH = 9, gap = 2, rowGap = 10, padT = 4;
  const rowH = series.length * barH + (series.length - 1) * gap;
  const h = padT + rows.length * (rowH + rowGap);
  const x = (v: number) => labelW + (Math.max(0, v) / max) * (w - labelW - 44);
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline gap-x-3 text-[11px]">
        <span className="text-white">{title}</span>
        {series.length > 1 && series.map((s) => (
          <span key={s.name} className="flex items-center gap-1 text-muted-foreground">
            <span className="inline-block size-2 rounded-sm" style={{ background: s.color }} />{s.name}
          </span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={title}>
        {[0.5, 1].filter((t) => t <= max).map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={h - rowGap / 2} stroke="currentColor" strokeDasharray="2 3" className="text-border" />
        ))}
        {rows.map((r, i) => {
          const y0 = padT + i * (rowH + rowGap);
          return (
            <g key={r.label}>
              <text x={labelW - 8} y={y0 + rowH / 2} dy="0.35em" textAnchor="end" fontSize={10} className="fill-muted-foreground">{r.label}</text>
              {r.vals.map((v, k) => {
                const y = y0 + k * (barH + gap);
                return (
                  <g key={k}>
                    <rect x={labelW} y={y} width={Math.max(2, x(v) - labelW)} height={barH} rx={2} fill={series[k].color}>
                      <title>{`${r.label} · ${series[k].name}: ${fmt(v)}`}</title>
                    </rect>
                    <text x={x(v) + 4} y={y + barH / 2} dy="0.35em" fontSize={9} className="fill-white/80 tabular-nums">{fmt(v)}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

type Series = { name: string; color: string; pts: [number, number][] };
/** Multi-series line chart over steps (same look as the page's other charts). */
export function Lines({ label, series, fmt, lo, hi }: { label: string; series: Series[]; fmt: (v: number) => string; lo?: number; hi?: number }) {
  const all = series.flatMap((s) => s.pts);
  if (all.length < 2) return <div className="text-[11px] text-muted-foreground">{label}: not enough points yet</div>;
  const w = 420, h = 150, padL = 40, padR = 8, padT = 10, padB = 16;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = lo ?? Math.min(...ys), y1 = hi ?? Math.max(...ys);
  const x = (v: number) => padL + ((v - x0) * (w - padL - padR)) / Math.max(1, x1 - x0);
  const y = (v: number) => padT + (1 - (v - y0) / Math.max(1e-12, y1 - y0)) * (h - padT - padB);
  return (
    <figure className="min-w-0">
      <figcaption className="flex flex-wrap items-baseline gap-x-3 text-[11px]">
        <span className="text-white">{label}</span>
        {series.length > 1 && series.map((s) => <span key={s.name} className="text-muted-foreground"><span style={{ color: s.color }}>●</span> {s.name}</span>)}
      </figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={label}>
        <text x={padL - 4} y={y(y1)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{fmt(y1)}</text>
        <text x={padL - 4} y={y(y0)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{fmt(y0)}</text>
        <line x1={padL} x2={w - padR} y1={h - padB} y2={h - padB} stroke="currentColor" className="text-border" />
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} points={s.pts.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")} />
            {s.pts.length < 40 && s.pts.map((p) => (
              <circle key={p[0]} cx={x(p[0])} cy={y(p[1])} r={4} fill={s.color} stroke="#0b0b10" strokeWidth={2}>
                <title>{`${s.name} · step ${p[0].toLocaleString()}: ${fmt(p[1])}`}</title>
              </circle>
            ))}
          </g>
        ))}
        <text x={w - padR} y={h - 3} textAnchor="end" fontSize={9} className="fill-muted-foreground">step {x1.toLocaleString()}</text>
      </svg>
    </figure>
  );
}

export function Tile({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-white/[0.02] px-3 py-2.5">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-xl font-semibold tabular-nums text-white">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

function Card({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border/70 p-3">
      <div className="text-xs font-medium text-white">{title}</div>
      {note && <div className="mb-2 mt-0.5 text-[11px] text-muted-foreground">{note}</div>}
      {children}
    </div>
  );
}

const TOTAL = 30000;
const C4 = "#9b7bea";   // speed chart only, always on its own

function DetectorHealth({ L, sets, best, detState }: {
  L: VoiceData["live"]; sets: [string, string, string][]; best: VoiceVal | null; detState: string;
}) {
  const last = L.trace.at(-1);
  const sps = L.trace.length ? L.trace.slice(-5).reduce((a, t) => a + t[2], 0) / Math.min(5, L.trace.length) : 0;
  const left = last && sps > 0 && !L.finished ? (TOTAL - last[0]) / sps / 60 : null;
  const vs = (v?: ValSet | number) => (v && typeof v === "object" ? v : undefined);
  const ap = (k: string) => L.vals.filter((v) => vs(v[k])?.ap != null).map((v) => [v.step, vs(v[k])!.ap!] as [number, number]);
  const pr = (k: string, f: "p" | "r") => L.vals.filter((v) => vs(v[k])?.[f] != null).map((v) => [v.step, vs(v[k])![f]!] as [number, number]);
  const refBest = (r: VoiceVal[] = []) => (r.length ? r.reduce((a, b) => (b.score > a.score ? b : a)) : null);
  const refs = Object.entries(L.ref ?? {}).map(([k, r]) => [k, refBest(r)] as const).filter(([, b]) => b);
  return (
    <div className="mb-4 rounded-lg border border-primary/50 bg-primary/[0.04] p-3">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <div className="text-sm font-semibold text-white">Solo-voice detector v3: training health</div>
        <div className="text-[11px] text-muted-foreground">{L.run ?? "v3"}{L.gpu ? ` · ${L.gpu}` : ""}</div>
        {L.stale && <div className="text-[11px] text-amber-200">cluster didn't answer just now: showing its last reading, from {L.stale}</div>}
      </div>
      <div className="mb-3 mt-0.5 text-[11px] text-muted-foreground">
        Reads the full mix and marks each 16 ms frame where the voice sounds over a band that is silent (below the song's
        softest instrumental level) or light (in the bottom 30% of that song's own range, the threshold chosen from the base's
        recall per level). Labels: Demucs on Music2MIDI (offline) and MUSDB18-HQ's true stems, band turned down in random stretches.
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Tile label="Progress" value={last ? `${((100 * last[0]) / TOTAL).toFixed(0)}%` : "—"} hint={last ? `step ${last[0].toLocaleString()} / ${TOTAL.toLocaleString()}` : detState} />
        <Tile label="Speed" value={sps ? `${sps.toFixed(1)}/s` : "—"} hint={L.finished ? "finished" : left != null ? `~${Math.round(left)} min left` : "steps per second"} />
        <Tile label="Best validation AP" value={best ? f3(best.score) : "—"} hint={best ? `step ${best.step.toLocaleString()} (mean of 3 sets)` : "first check at step 1,000"} />
        <Tile label="Latest loss" value={last ? f3(last[1]) : "—"} hint="cross-entropy, solo classes weighted x4" />
      </div>
      {last && (
        <div className="mb-3 max-w-xl">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full" style={{ width: `${(100 * last[0]) / TOTAL}%`, background: C2 }} />
          </div>
        </div>
      )}
      {L.jobs.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2 text-[11px]">
          {L.jobs.map((j) => (
            <span key={j.id} className="rounded-md border border-border px-2 py-0.5 text-muted-foreground">
              <span className="text-white">{j.name}</span> · {j.state.toLowerCase()} {j.state === "RUNNING" ? `${j.elapsed} / ${j.limit}` : j.reason}
            </span>
          ))}
        </div>
      )}

      {L.vals.length || L.trace.length ? (
        <div className="grid gap-x-6 gap-y-4 md:grid-cols-2 2xl:grid-cols-3">
          <Lines label="validation AP (silent or light), by set" lo={0} hi={1} fmt={f3}
                 series={sets.map(([k, nm, c]) => ({ name: nm, color: c, pts: ap(k) }))} />
          <Lines label="Music2MIDI val AP, by class" lo={0} hi={1} fmt={f3}
                 series={[{ name: "silent or light", color: C1, pts: ap("m2m_val") }, { name: "silent", color: C2, pts: ap("m2m_val_silent") },
                          { name: "light", color: C3, pts: ap("m2m_val_quiet") }]} />
          <Lines label="Music2MIDI val precision / recall at 0.5" lo={0} hi={1} fmt={f3}
                 series={[{ name: "precision", color: C1, pts: pr("m2m_val", "p") }, { name: "recall", color: C2, pts: pr("m2m_val", "r") }]} />
          <Lines label="MUSDB natural AP, by class" lo={0} hi={1} fmt={f3}
                 series={[{ name: "silent or light", color: C1, pts: ap("musdb_nat") }, { name: "silent", color: C2, pts: ap("musdb_nat_silent") },
                          { name: "light", color: C3, pts: ap("musdb_nat_quiet") }]} />
          <Lines label="training loss (per 100 steps)" fmt={f3} series={[{ name: "loss", color: C1, pts: L.trace.map((t) => [t[0], t[1]] as [number, number]) }]} />
          <Lines label="speed (steps / s)" lo={0} fmt={(v) => v.toFixed(1)} series={[{ name: "steps/s", color: C4, pts: L.trace.map((t) => [t[0], t[2]] as [number, number]) }]} />
        </div>
      ) : <div className="text-[11px] text-muted-foreground">{detState}</div>}

      {best && (
        <table className="mt-4 w-full max-w-2xl text-left text-xs">
          <thead className="text-muted-foreground"><tr><th className="pb-1 font-normal">best so far (step {best.step.toLocaleString()})</th><th className="font-normal">AP</th><th className="font-normal">precision</th><th className="font-normal">recall</th><th className="font-normal">F1</th><th className="font-normal">share of frames</th></tr></thead>
          <tbody>
            {sets.flatMap(([k, nm]) => [[k, nm], [`${k}_silent`, "band silent"], [`${k}_quiet`, "band light (bottom 30%)"]])
              .filter(([k]) => vs(best[k])).map(([k, nm]) => { const v = vs(best[k])!; const sub = /_(silent|quiet)$/.test(k); return (
              <tr key={k} className="border-t border-border/60 tabular-nums"><td className={sub ? "py-1 pl-4 pr-3 text-muted-foreground" : "py-1 pr-3 text-white"}>{nm}</td><td>{v.ap == null ? "—" : f3(v.ap)}</td><td>{v.p == null ? "—" : f3(v.p)}</td><td>{v.r == null ? "—" : f3(v.r)}</td><td>{v.f1 == null ? "—" : f3(v.f1)}</td><td>{pct(v.pos)}</td></tr>
            ); })}
          </tbody>
        </table>
      )}
      {refs.length > 0 && (
        <>
          <table className="mt-3 w-full max-w-2xl text-left text-xs">
            <thead className="text-muted-foreground"><tr><th className="pb-1 font-normal">earlier detectors, best (their own labels)</th><th className="font-normal">Music2MIDI val AP</th><th className="font-normal">F1</th><th className="font-normal">MUSDB natural AP</th><th className="font-normal">solo share</th></tr></thead>
            <tbody>
              {refs.map(([k, b]) => { const mv = vs(b!["m2m_val"]); const mn = vs(b!["musdb_nat"]); return (
                <tr key={k} className="border-t border-border/60 tabular-nums"><td className="py-1 pr-3 text-white">{k === "v1" ? "v1 (voice 12 dB+ over the band)" : "v2 (voice 8 dB+ over the band)"}</td><td>{mv?.ap != null ? f3(mv.ap) : "—"}</td><td>{mv?.f1 != null ? f3(mv.f1) : "—"}</td><td>{mn?.ap != null ? f3(mn.ap) : "—"}</td><td>{mv ? pct(mv.pos) : "—"}</td></tr>
              ); })}
            </tbody>
          </table>
          <div className="mt-1 text-[11px] text-muted-foreground">Each detector is scored against its own definition of solo, so these are context, not a ranking; the real comparison is fill-and-remove on the 106 test songs.</div>
        </>
      )}
    </div>
  );
}

export default function VoicePanel() {
  const [d, setD] = React.useState<VoiceData | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  React.useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch("/voice-data");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (alive) { setD(j); setErr(null); }
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    };
    load();
    const id = setInterval(load, 30000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  if (err && !d) return <div className="mb-8 text-xs text-rose-200">Voice work: couldn't load ({err})</div>;
  if (!d) return null;

  const L = d.live;
  const best = L.vals.length ? L.vals.reduce((a, b) => (b.score > a.score ? b : a)) : null;
  const lastStep = L.trace.at(-1)?.[0];
  const detState = L.finished ? "trained" : lastStep ? `training · step ${lastStep.toLocaleString()}` :
    L.jobs.some((j) => j.name.startsWith("solo-train")) ? "queued" : "waits for the Demucs labels";
  const sepDone = L.curves != null && L.songs ? Math.min(L.curves, L.songs) : null;
  const base = d.fill.conditions[0], hard = d.fill.conditions.at(-1)!;
  const sets: [string, string, string][] = [["m2m_val", "Music2MIDI val (Demucs labels)", C1], ["musdb_nat", "MUSDB test, natural", C2], ["musdb_aug", "MUSDB test, made solo", C3]];

  return (
    <section className="mb-10 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Mic className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Voice: the solo-voice fix</h2>
        <span className="text-xs text-muted-foreground">
          the base reads a voice well inside a mix but drops notes when it is alone: fill the solo bars with the song's own
          accompaniment, transcribe, remove what was added. Results {d.updated}.
        </span>
      </div>

      <DetectorHealth L={L} sets={sets} best={best} detState={detState} />
      <div className="mb-4 grid grid-cols-2 gap-2 lg:grid-cols-4">
        <Tile label="Clean 'aah' melody, F1" value={<>{f3(base.solo)} → {f3(base.fill)}</>} hint="solo → fill and remove (50 songs)" />
        <Tile label="Singer-like voice + bleed, F1" value={<>{f3(hard.solo)} → {f3(hard.fill)}</>} hint="with monophony cleanup" />
        <Tile label="Demucs labels (cluster)" value={sepDone == null ? "—" : `${sepDone} / ${L.songs}`} hint="Music2MIDI songs separated" />
        <Tile label="Solo detector" value={best ? `AP ${f3(best.score)}` : "—"} hint={detState} />
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card title="1 · Does a neutral 'aah' transcribe?" note={d.aah.note}>
          <HBars title="base on the solo 'aah'" series={[{ name: "precision", color: C1 }, { name: "recall", color: C2 }, { name: "F1", color: C3 }]}
                 rows={d.aah.rows.map((r) => ({ label: r.name, vals: [r.p, r.r, r.f1] }))} />
          <div className="mt-1 text-[11px] text-muted-foreground">At 0.5 it hears 95% of the notes but isn't sure enough to clear 0.9: missing notes, not wrong ones.</div>
        </Card>

        <Card title="2 · Fill and remove, by condition" note={d.fill.note}>
          <HBars title="note F1" series={[{ name: "solo voice", color: C1 }, { name: "fill and remove", color: C2 }]}
                 rows={d.fill.conditions.map((c) => ({ label: c.name, vals: [c.solo, c.fill] }))} />
        </Card>

        <Card title="3 · What each cleanup step adds" note={d.fill.steps.note}>
          <HBars title="singer-like voice + bleed" series={[{ name: "precision", color: C1 }, { name: "recall", color: C2 }, { name: "F1", color: C3 }]}
                 rows={d.fill.steps.rows.map((r) => ({ label: r.name, vals: [r.p, r.r, r.f1] }))} />
          <div className="mt-1 text-[11px] text-muted-foreground">The voice filter (pitch track) cuts bleed but checked the scoop at each note start; a retune is pending.</div>
        </Card>

        <Card title="4 · Is it the base? (clean 'aah')" note="C and D read a solo voice almost perfectly on their own; the base is the one that needs the fill">
          <HBars title="note F1" series={[{ name: "solo voice", color: C1 }, { name: "fill and remove", color: C2 }]}
                 rows={d.fill.models.map((m) => ({ label: m.name, vals: [m.solo, m.fill] }))} />
        </Card>

        <Card title="5 · Finding solo bars with rules" note={d.detect_rules.note}>
          <HBars title="recall of solo bars (precision stays 0.80-0.99)" series={[{ name: "voice alone", color: C1 }, { name: "voice + quiet pad", color: C3 }]}
                 rows={d.detect_rules.rows.map((r) => ({ label: r.name, vals: [r.plain_r, r.pad_r] }))} />
          <div className="mt-1 text-[11px] text-muted-foreground">Rules break as soon as anything sounds under the voice, hence the trained detector.</div>
        </Card>

        <Card title="6 · How much solo voice do real songs have?" note={d.musdb.note}>
          <HBars title="share in the clear (rest of the band at least … below the voice)" max={0.06} fmt={pct}
                 series={[{ name: "by duration (runs ≥ 1 s)", color: C1 }, { name: "by note start (250 ms around)", color: C2 }]}
                 rows={d.musdb.duration.map((r, i) => ({ label: `${r.margin} dB`, vals: [r.share, d.musdb.starts[i].share] }))} />
        </Card>
      </div>

    </section>
  );
}
