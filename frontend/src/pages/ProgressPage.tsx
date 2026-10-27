import * as React from "react";
import { Activity, Check, Cpu, Loader2, RefreshCw, TriangleAlert } from "lucide-react";

import {
  Timeline,
  TimelineContent,
  TimelineDate,
  TimelineHeader,
  TimelineIndicator,
  TimelineItem,
  TimelineSeparator,
  TimelineTitle,
} from "@/components/ui/timeline";
import { GridHeatmap } from "@/components/ui/grid-heatmap";
import { cn } from "@/lib/utils";
import VoicePanel from "./VoicePanel";
import CleanupPanel from "./CleanupPanel";

// ---- shape of /progress-data (backend/eval/progress_dump.py) ----
type Val = { time: string; step: number; f1: number; p: number; r: number; frame: number };
type Run = {
  started: string;
  vals: Val[];
  last: { time: string; step: number; loss: number; speed: number; gpu: number | null } | null;
  stopped: string | null;
  trace?: [number, number, number | null, number, number][];   // step, loss, GPU %, data wait %, steps/s
};
type Test = Partial<{ recall: number; precision: number; f1: number; dbl: number; start_median: number; start_abs: number; within8: number }>;
type ShardInfo = { s: string; songs: number; of: number | null; clips: number; el: number; rss: number; node: string | null } | null;
type Render = {
  done: number; running: number; pending: number; failed: number; total: number; grid?: string;
  songs?: number; songs_total?: number | null; clips?: number; first?: string | null; wall_min?: number | null;
  rate?: number | null; eta_min?: number | null; shard_min_mean?: number | null; shard_min_max?: number | null;
  rss_max?: number; nodes?: number; tasks?: ShardInfo[]; active?: boolean;
};
type Job = { id: string; name: string; state: string; elapsed: string; limit: string; reason: string; partition: string; gpu?: string | null };
type JointVal = { time: string; step: number; score: number; combo: number; modules: Record<string, number> };
type Joint = {
  vals: JointVal[];
  trace: [number, number, number, number][];          // step, loss, drift, data wait %
  last: { time: string; step: number; loss: number; drift: number; speed: number; wait: number } | null;
  stopped: string | null;
};
type GateVal = { time: string; step: number; detect: number; gate: number; known: number; none: number; modules: Record<string, [number, number]> };
type GatePhase = {
  vals: GateVal[];
  trace: [number, number, number, number][];          // step, bce, task, data wait %
  last: { time: string; step: number; bce: number; task: number; speed: number; wait: number } | null;
  stopped: string | null;
};
type Snapshot = {
  time: string;
  jobs: Job[];
  renders: Record<string, Render>;
  plain: { train: Run | null; test: Test | null; merged: boolean };
  modules: Record<string, { lane: number; train: Run | null; test: Test | null; baseline: Test | null }>;
  lanes: string[][];
  alerts?: { key: string; text: string }[];
  notices?: { key: string; text: string }[];
  joint?: Joint | null;
  newbase?: NewBase;
  gate?: { classify: GatePhase | null; e2e: GatePhase | null };
  schedule_stale?: string;
};

const TOTAL_STEPS = 20000;
const POLL_MS = 30000;

type ModState = "rendering" | "queued" | "preparing" | "training" | "testing" | "done";

function moduleState(s: Snapshot, m: string): ModState {
  const mod = s.modules[m];
  const r = s.renders[m];
  if (mod?.test?.f1 !== undefined) return "done";
  if (mod?.train) {
    const lane = s.lanes[mod.lane];
    const next = lane[lane.indexOf(m) + 1];
    const moved = next !== undefined && s.modules[next]?.train;
    return mod.train.stopped || moved ? "testing" : "training";
  }
  if (!r || r.done < r.total) return "rendering";
  // the lane is running and the module before this one has finished training (or there is none):
  // this one is mixing its data
  const lane = mod ? s.lanes[mod.lane] : [];
  const laneRunning = s.jobs.some((j) => j.name === `mod-lane${mod?.lane}` && j.state === "RUNNING");
  const prev = lane[lane.indexOf(m) - 1];
  const prevTrain = prev ? s.modules[prev]?.train : null;
  const prevFinished = !prev || !!prevTrain?.stopped || s.modules[prev]?.test?.f1 !== undefined;
  return laneRunning && prevFinished && !s.modules[m]?.train ? "preparing" : "queued";
}

// ---- small pieces ----
function Bar({ value, total, className }: { value: number; total: number; className?: string }) {
  const pct = total ? Math.min(100, (100 * value) / total) : 0;
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div className="h-full rounded-full bg-primary transition-all duration-700" style={{ width: `${pct}%` }} />
    </div>
  );
}

function Sparkline({ vals }: { vals: Val[] }) {
  if (vals.length < 2) return null;
  const w = 280, h = 64, pad = 4;
  const ys = vals.map((v) => v.f1);
  const lo = Math.min(...ys) - 0.01, hi = Math.max(...ys) + 0.01;
  const x = (i: number) => pad + (i * (w - 2 * pad)) / (vals.length - 1);
  const y = (v: number) => h - pad - ((v - lo) * (h - 2 * pad)) / (hi - lo);
  const best = ys.indexOf(Math.max(...ys));
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-16 w-full max-w-[280px]" role="img" aria-label="validation note F1 by step">
      <polyline fill="none" stroke="oklch(0.6 0.2 250)" strokeWidth="2" points={ys.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
      {ys.map((v, i) => (
        <circle key={i} cx={x(i)} cy={y(v)} r={i === best ? 4 : 2.5} fill={i === best ? "oklch(0.8 0.15 150)" : "oklch(0.6 0.2 250)"}>
          <title>{`step ${vals[i].step}: F1 ${v.toFixed(3)}`}</title>
        </circle>
      ))}
    </svg>
  );
}

function Stat({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="truncate text-lg font-semibold tabular-nums text-white">{value}</div>
      {hint && <div className="text-[11px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

function LiveRun({ title, run, job, total = TOTAL_STEPS, valEvery = 1000, note = "Stops early after 3 checks without a new best." }:
  { title: string; run: Run; job?: Job; total?: number; valEvery?: number; note?: string }) {
  const last = run.last;
  const best = run.vals.reduce<Val | null>((b, v) => (!b || v.f1 > b.f1 ? v : b), null);
  const etaMin = last && last.speed > 0 ? (total - last.step) / last.speed / 60 : null;
  return (
    <div className="rounded-xl border border-border bg-white/[0.03] p-4">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
        <Activity className="size-4 text-primary" />
        {title}
        {job && <span className="ml-auto text-xs font-normal text-muted-foreground">{job.gpu ?? job.partition} · {job.reason} · {job.elapsed}</span>}
      </div>
      {last ? (
        <>
          <Bar value={last.step} total={total} />
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
            <Stat label="step" value={`${last.step.toLocaleString()}`} hint={`of ${total.toLocaleString()} max`} />
            <Stat label="loss" value={last.loss.toFixed(4)} />
            <Stat label="speed" value={`${last.speed.toFixed(2)}/s`} hint={etaMin ? `≤ ${fmtMin(etaMin)} left` : undefined} />
            <Stat label="GPU" value={last.gpu === null ? "—" : `${last.gpu}%`} hint={last.gpu === null ? "node doesn't report it" : undefined} />
            <Stat label="best val F1" value={best ? best.f1.toFixed(3) : "—"} hint={best ? `step ${best.step}` : `first check at ${valEvery.toLocaleString()}`} />
          </div>
          <div className="mt-3 flex flex-wrap items-end gap-4">
            <Sparkline vals={run.vals} />
            <div className="text-[11px] text-muted-foreground">
              {note}<br />Last update {last.time}.
            </div>
          </div>
        </>
      ) : (
        <div className="text-sm text-muted-foreground">Loading data…</div>
      )}
    </div>
  );
}

function TestRow({ name, t }: { name: string; t: Test | null }) {
  const f = (v?: number, d = 3) => (v === undefined ? "—" : v.toFixed(d));
  return (
    <tr className="border-t border-border/60">
      <td className="py-1.5 pr-3 text-white">{name}</td>
      <td className="pr-3 tabular-nums">{f(t?.recall)}</td>
      <td className="pr-3 tabular-nums">{f(t?.precision)}</td>
      <td className="pr-3 tabular-nums font-semibold text-white">{f(t?.f1)}</td>
      <td className="pr-3 tabular-nums">{f(t?.dbl, 1)}</td>
      <td className="tabular-nums">{t?.start_abs === undefined ? "—" : `${t.start_abs.toFixed(1)} ms`}</td>
    </tr>
  );
}

const STATE_STYLE: Record<ModState, string> = {
  rendering: "text-muted-foreground",
  queued: "text-muted-foreground",
  preparing: "text-primary",
  training: "text-primary",
  testing: "text-amber-300",
  done: "text-emerald-300",
};

function ModuleCard({ s, m }: { s: Snapshot; m: string }) {
  const st = moduleState(s, m);
  const mod = s.modules[m];
  const r = s.renders[m];
  const d = (a?: number, b?: number) => (a === undefined || b === undefined ? null : a - b);
  const df1 = d(mod?.test?.f1, mod?.baseline?.f1);
  return (
    <div className={cn("rounded-lg border border-border bg-white/[0.02] p-3", st === "training" && "border-primary/60")}>
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-white">{m}</span>
        <span className={cn("ml-auto flex items-center gap-1 text-[11px] uppercase tracking-wider", STATE_STYLE[st])}>
          {(st === "training" || st === "preparing") && <Loader2 className="size-3 animate-spin" />}
          {st === "done" && <Check className="size-3" />}
          {st}
        </span>
      </div>
      {st === "rendering" && r && (
        <div className="mt-2">
          <Bar value={r.done} total={r.total} />
          <div className="mt-1 text-[11px] text-muted-foreground">
            {r.done}/{r.total} shards{r.running ? ` · ${r.running} running` : ""}{r.failed ? ` · ${r.failed} failed (rerun)` : ""}
          </div>
        </div>
      )}
      {st === "training" && mod?.train?.last && (
        <div className="mt-2">
          <Bar value={mod.train.last.step} total={TOTAL_STEPS} />
          <div className="mt-1 text-[11px] text-muted-foreground">
            step {mod.train.last.step.toLocaleString()} · val F1 {mod.train.vals.at(-1)?.f1.toFixed(3) ?? "—"}
          </div>
        </div>
      )}
      {st === "done" && (
        <div className="mt-1 text-[11px] text-muted-foreground">
          F1 {mod.test?.f1?.toFixed(3)} vs base {mod.baseline?.f1?.toFixed(3) ?? "—"}
          {df1 !== null && (
            <span className={cn("ml-1 font-semibold", df1 >= 0 ? "text-emerald-300" : "text-rose-300")}>
              {df1 >= 0 ? "+" : ""}{df1.toFixed(3)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ---- training health ----

// sequential: the demo's cool pair, dark navy → mint
function f1Color(v: number) {
  const t = Math.max(0, Math.min(1, (v - 0.6) / 0.35));
  return `color-mix(in oklch, #b4fbde ${Math.round(t * 100)}%, #122549)`;
}
// diverging, for module − plain_base: red worse, green better, ±`span` saturates
function deltaColor(d: number, span: number) {
  const t = Math.max(-1, Math.min(1, d / span));
  return t >= 0 ? `color-mix(in oklch, #34d399 ${Math.round(t * 100)}%, #26262c)` : `color-mix(in oklch, #f33d15 ${Math.round(-t * 100)}%, #26262c)`;
}

type Series = { name: string; color: string; pts: [number, number][] };

/** A small multi-series line chart over training steps (same look as the health charts). */
function MiniChart({ label, series, fmt, marks = [], lo, hi, height = 150, skipStart = 0 }: {
  label: string; series: Series[]; fmt: (v: number) => string; marks?: number[]; lo?: number; hi?: number; height?: number;
  skipStart?: number;   // scale the y axis without the first share of points (loss starts very high)
}) {
  const all = series.flatMap((s_) => s_.pts);
  if (all.length < 2) return <div className="text-[11px] text-muted-foreground">{label}: not enough points yet</div>;
  const w = 420, h = height, padL = 40, padR = 8, padT = 22, padB = 16;
  const xs = all.map((p_) => p_[0]), ys = all.map((p_) => p_[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const scaleYs = skipStart ? series.flatMap((s_) => s_.pts.slice(Math.floor(s_.pts.length * skipStart)).map((p_) => p_[1])) : ys;
  const y0 = lo ?? Math.min(...scaleYs), y1 = hi ?? Math.max(...scaleYs);
  const x = (v: number) => padL + ((v - x0) * (w - padL - padR)) / Math.max(1, x1 - x0);
  const y = (v: number) => padT + (1 - (v - y0) / Math.max(1e-12, y1 - y0)) * (h - padT - padB);
  return (
    <div>
    <div className="flex flex-wrap items-baseline gap-x-3 text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      {series.length > 1 && series.map((s_) => <span key={s_.name} style={{ color: s_.color }}>● {s_.name}</span>)}
    </div>
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={label}>
      <text x={padL - 4} y={y(y1)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{fmt(y1)}</text>
      <text x={padL - 4} y={y(y0)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{fmt(y0)}</text>
      <line x1={padL} x2={w - padR} y1={h - padB} y2={h - padB} stroke="currentColor" className="text-border" />
      {marks.map((m) => <line key={m} x1={x(m)} x2={x(m)} y1={padT} y2={h - padB} stroke="currentColor" strokeDasharray="2 3" className="text-border" />)}
      {series.map((s_) => (
        <g key={s_.name}>
          <polyline fill="none" stroke={s_.color} strokeWidth={1.5} points={s_.pts.map((p_) => `${x(p_[0])},${y(Math.min(p_[1], y1))}`).join(" ")} />
          {s_.pts.length < 30 && s_.pts.map((p_) => (
            <circle key={p_[0]} cx={x(p_[0])} cy={y(p_[1])} r={2.5} fill={s_.color}><title>{`${s_.name} · step ${p_[0]}: ${fmt(p_[1])}`}</title></circle>
          ))}
        </g>
      ))}
      <text x={w - padR} y={h - 3} textAnchor="end" fontSize={9} className="fill-muted-foreground">step {x1.toLocaleString()}</text>
    </svg>
    </div>
  );
}

/** Stage 5 in full: training curves, validation curves, and every module against its step-0 self. */
function JointDetail({ j, modules, job }: { j: Joint; modules: string[]; job?: Job }) {
  const ref = j.vals[0], cur = j.vals.at(-1);
  const best = j.vals.reduce<JointVal | null>((b, v) => (!b || v.score > b.score ? v : b), null);
  const marks = j.vals.map((v) => v.step);
  const mean = (v: JointVal) => modules.reduce((a, m) => a + (v.modules[m] ?? 0), 0) / modules.length;
  const etaMin = j.last && j.last.speed > 0 && !j.stopped ? (15000 - j.last.step) / j.last.speed / 60 : null;
  const rows = ref && cur ? modules.map((m) => {
    const hist = j.vals.map((v) => v.modules[m]).filter((v) => v !== undefined) as number[];
    const bi = j.vals.reduce((b_, v, i) => ((v.modules[m] ?? -1) > (j.vals[b_].modules[m] ?? -1) ? i : b_), 0);
    return { m, alone: ref.modules[m], now: cur.modules[m], d: (cur.modules[m] ?? 0) - (ref.modules[m] ?? 0),
             best: j.vals[bi].modules[m], bestStep: j.vals[bi].step, hist };
  }).sort((a_, b_) => a_.d - b_.d) : [];
  const hurt = rows.filter((r) => r.d < -0.005).length, helped = rows.filter((r) => r.d > 0.005).length;
  const sign = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
  return (
    <section className="mb-14 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Activity className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Stage 5 · joint refinement</h2>
        <span className="ml-auto text-xs text-muted-foreground">
          {j.stopped ? j.stopped : job ? `${job.gpu ?? job.partition} · ${job.reason} · ${job.elapsed}` : "waiting for a GPU"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 2xl:grid-cols-8">
        <Stat label="step" value={j.last ? j.last.step.toLocaleString() : "0"} hint="of 15,000 max" />
        <Stat label="speed" value={j.last ? `${j.last.speed.toFixed(2)}/s` : "—"} hint={etaMin ? `≤ ${Math.round(etaMin)} min left` : undefined} />
        <Stat label="loss" value={j.last ? j.last.loss.toFixed(4) : "—"} />
        <Stat label="drift" value={j.last ? j.last.drift.toExponential(1) : "—"} hint="from trained-alone weights" />
        <Stat label="combination F1" value={cur ? cur.combo.toFixed(3) : "—"}
              hint={cur && ref ? `alone ${ref.combo.toFixed(3)}, ${sign(cur.combo - ref.combo)}` : "step 0 pending"} />
        <Stat label="mean module F1" value={cur ? mean(cur).toFixed(3) : "—"}
              hint={cur && ref ? `alone ${mean(ref).toFixed(3)}, ${sign(mean(cur) - mean(ref))}` : undefined} />
        <Stat label="modules helped / hurt" value={rows.length > 1 && j.vals.length > 1 ? `${helped} / ${hurt}` : "—"} hint="by more than 0.005" />
        <Stat label="best score" value={best ? best.score.toFixed(4) : "—"} hint={best ? `step ${best.step} · ${j.vals.length} checks` : undefined} />
      </div>
      <div className="mt-4 grid gap-x-6 gap-y-2 md:grid-cols-2 2xl:grid-cols-4">
        <MiniChart label="loss" marks={marks} fmt={(v) => v.toFixed(3)}
                   series={[{ name: "loss", color: "oklch(0.6 0.2 250)", pts: j.trace.map((t) => [t[0], t[1]] as [number, number]) }]} />
        <MiniChart label="drift from trained-alone weights" marks={marks} lo={0} fmt={(v) => v.toExponential(0)}
                   series={[{ name: "drift", color: "#c4b5fd", pts: j.trace.map((t) => [t[0], t[2]] as [number, number]) }]} />
        <MiniChart label="data wait %" marks={marks} lo={0} hi={Math.max(10, ...j.trace.map((t) => t[3]))} fmt={(v) => `${Math.round(v)}`}
                   series={[{ name: "wait", color: "#f33d15", pts: j.trace.map((t) => [t[0], t[3]] as [number, number]) }]} />
        <MiniChart label="validation F1" fmt={(v) => v.toFixed(3)} series={[
          { name: "combination", color: "#b4fbde", pts: j.vals.map((v) => [v.step, v.combo] as [number, number]) },
          { name: "mean module", color: "oklch(0.6 0.2 250)", pts: j.vals.map((v) => [v.step, mean(v)] as [number, number]) },
        ]} />
      </div>
      {rows.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-xs text-muted-foreground">
            <thead>
              <tr>
                <th className="pb-1 font-normal">module (worst change first)</th><th className="font-normal">alone (step 0)</th>
                <th className="font-normal">now</th><th className="font-normal">change</th><th className="font-normal">best</th>
                <th className="font-normal">over the checks</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.m} className="border-t border-border/60">
                  <td className="py-1 pr-3 text-white">{r.m}</td>
                  <td className="pr-3 tabular-nums">{r.alone?.toFixed(3) ?? "—"}</td>
                  <td className="pr-3 tabular-nums text-white">{r.now?.toFixed(3) ?? "—"}</td>
                  <td className={cn("pr-3 tabular-nums font-semibold", r.d > 0.005 ? "text-emerald-300" : r.d < -0.005 ? "text-rose-300" : "")}>{sign(r.d)}</td>
                  <td className="pr-3 tabular-nums">{r.best?.toFixed(3)} <span className="text-[10px]">@{r.bestStep / 1000}k</span></td>
                  <td className="w-40">
                    {r.hist.length > 1 && (
                      <svg viewBox="0 0 120 18" className="h-4 w-32" aria-hidden>
                        <polyline fill="none" strokeWidth={1.5} stroke={r.d >= 0 ? "#34d399" : "#f33d15"}
                          points={r.hist.map((v, i) => `${(i * 118) / (r.hist.length - 1) + 1},${16 - ((v - Math.min(...r.hist)) / Math.max(1e-6, Math.max(...r.hist) - Math.min(...r.hist))) * 14}`).join(" ")} />
                      </svg>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function HealthLines({ run }: { run: Run }) {
  const tr = run.trace ?? [];
  if (tr.length < 2) return <div className="text-xs text-muted-foreground">Not enough steps logged yet.</div>;
  const w = 420, h = 150, padL = 34, padR = 8, padT = 22, padB = 16;
  const steps = tr.map((t) => t[0]);
  const x = (s: number) => padL + ((s - steps[0]) * (w - padL - padR)) / Math.max(1, steps.at(-1)! - steps[0]);
  const panels: { label: string; idx: number; lo: number; hi: number; fmt: (v: number) => string; color: string }[] = [
    { label: "loss", idx: 1, lo: Math.min(...tr.map((t) => t[1])), hi: Math.max(...tr.slice(Math.floor(tr.length / 10)).map((t) => t[1])), fmt: (v) => v.toFixed(3), color: "oklch(0.6 0.2 250)" },
    { label: "GPU %", idx: 2, lo: 0, hi: 100, fmt: (v) => `${v}`, color: "#b4fbde" },
    { label: "data wait %", idx: 3, lo: 0, hi: Math.max(10, ...tr.map((t) => t[3])), fmt: (v) => `${v}`, color: "#f33d15" },
  ];
  return (
    <div className="grid gap-x-6 gap-y-2 md:grid-cols-3">
      {panels.map((p) => {
        const y = (v: number) => padT + (1 - (Math.min(v, p.hi) - p.lo) / Math.max(1e-9, p.hi - p.lo)) * (h - padT - padB);
        return (
          <svg key={p.label} viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label={`${p.label} by step`}>
            <text x={padL} y={12} fontSize={11} className="fill-muted-foreground">{p.label}</text>
            <text x={padL - 4} y={y(p.hi)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{p.fmt(p.hi)}</text>
            <text x={padL - 4} y={y(p.lo)} dy="0.35em" textAnchor="end" fontSize={9} className="fill-muted-foreground">{p.fmt(p.lo)}</text>
            <line x1={padL} x2={w - padR} y1={h - padB} y2={h - padB} stroke="currentColor" className="text-border" />
            <polyline fill="none" stroke={p.color} strokeWidth={1.5} points={tr.filter((t) => t[p.idx] !== null).map((t) => `${x(t[0])},${y(t[p.idx] as number)}`).join(" ")} />
            {p.label === "loss" && run.vals.map((v) => (
              <line key={v.step} x1={x(v.step)} x2={x(v.step)} y1={padT} y2={h - padB} stroke="currentColor" strokeDasharray="2 3" className="text-border" />
            ))}
            <text x={w - padR} y={h - 3} textAnchor="end" fontSize={9} className="fill-muted-foreground">step {steps.at(-1)!.toLocaleString()}</text>
          </svg>
        );
      })}
    </div>
  );
}

function TrainingHealth({ s }: { s: Snapshot }) {
  const modules = s.lanes.flat();
  // validation F1 by check
  const runs: [string, Run][] = [];
  if (s.plain.train) runs.push(["plain", s.plain.train]);
  modules.forEach((m) => s.modules[m]?.train && runs.push([m, s.modules[m].train!]));
  const valCols = Array.from({ length: 20 }, (_, i) => `${i + 1}k`);
  // module − plain_base, each on its own effect
  const metrics: { label: string; get: (t: Test) => number | undefined; better: 1 | -1; span: number }[] = [
    { label: "recall", get: (t) => t.recall, better: 1, span: 0.05 },
    { label: "precision", get: (t) => t.precision, better: 1, span: 0.05 },
    { label: "F1", get: (t) => t.f1, better: 1, span: 0.05 },
    { label: "double/1k", get: (t) => t.dbl, better: -1, span: 20 },
    { label: "start err", get: (t) => t.start_abs, better: -1, span: 2 },
  ];
  const tested = modules.filter((m) => s.modules[m]?.test?.f1 !== undefined && s.modules[m]?.baseline);
  const [pick, setPick] = React.useState<string | null>(null);
  const live = runs.find(([, r]) => !r.stopped)?.[0];
  const shown = runs.find(([n]) => n === (pick ?? live ?? runs.at(-1)?.[0]));

  return (
    <section className="mb-14 grid gap-6 2xl:grid-cols-2">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground 2xl:col-span-2">Training health</h2>

      {shown && (
        <div className="rounded-xl border border-border bg-white/[0.02] p-4 2xl:col-span-2">
          <div className="mb-2 flex flex-wrap items-center gap-1.5">
            <span className="mr-2 text-sm font-medium text-white">Loss, GPU and data wait by step</span>
            {runs.map(([n, r]) => (
              <button key={n} onClick={() => setPick(n)}
                className={cn("rounded-md border px-2 py-0.5 text-[11px]",
                  n === shown[0] ? "border-primary text-white" : "border-border text-muted-foreground hover:text-white")}>
                {n}{!r.stopped && " ●"}
              </button>
            ))}
          </div>
          <HealthLines run={shown[1]} />
          <p className="mt-1 text-[11px] text-muted-foreground">Dashed lines mark validation checks. Healthy: GPU near 100%, data wait near 0, loss flattening.</p>
        </div>
      )}

      <div className="rounded-xl border border-border bg-white/[0.02] p-4">
        <div className="mb-2 text-sm font-medium text-white">Validation note F1 at each check</div>
        {runs.length ? (
          <GridHeatmap
            rows={runs.map(([n]) => n)}
            columns={valCols}
            cellHeight={18}
            value={(r, c) => runs[r][1].vals.find((v) => v.step === (c + 1) * 1000)?.f1 ?? null}
            color={f1Color}
            title={(r, c, v) => `${runs[r][0]} · step ${(c + 1) * 1000}: ${v === null ? "not reached" : `F1 ${v.toFixed(3)}`}`}
          />
        ) : <div className="text-xs text-muted-foreground">No validation yet.</div>}
        <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
          0.60 <span className="h-2 w-24 rounded" style={{ background: "linear-gradient(90deg, #122549, #b4fbde)" }} /> 0.95
          <span className="ml-2">Columns stop where early stopping ended the run.</span>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-white/[0.02] p-4">
        <div className="mb-2 text-sm font-medium text-white">Each module vs plain_base, on its own effect</div>
        {tested.length ? (
          <GridHeatmap
            rows={tested}
            columns={metrics.map((m) => m.label)}
            cellHeight={20}
            value={(r, c) => {
              const mt = metrics[c], a = mt.get(s.modules[tested[r]].test!), b = mt.get(s.modules[tested[r]].baseline!);
              return a === undefined || b === undefined ? null : (a - b) * mt.better;
            }}
            color={(v, c) => deltaColor(v, metrics[c].span)}
            title={(r, c, v) => `${tested[r]} · ${metrics[c].label}: ${v === null ? "—" : `${v >= 0 ? "better" : "worse"} by ${Math.abs(v).toFixed(3)}`}`}
          />
        ) : (
          <div className="text-xs text-muted-foreground">Fills in as modules finish their tests. Green means the module beats plain_base on its effect, red means worse.</div>
        )}
      </div>

    </section>
  );
}

const fmtMin = (m?: number | null) => (m == null ? "—" : m < 60 ? `${Math.round(m)} min` : `${Math.floor(m / 60)} h ${Math.round(m % 60)} min`);

/** Every render job: shards, songs, clips, speed, time left, shard times, memory, nodes, and a per-shard
 *  heatmap where a running shard is shaded by how far through its songs it is. */
function RenderDetail({ s }: { s: Snapshot }) {
  const order = ["plain", ...s.lanes.flat(), "combo"];
  const keys = Object.keys(s.renders).sort((a_, b_) => {
    const ra = s.renders[a_], rb = s.renders[b_];
    if (!!ra.active !== !!rb.active) return ra.active ? -1 : 1;          // running jobs first
    return order.indexOf(a_) - order.indexOf(b_);
  });
  const tot = keys.reduce((a, k) => ({ songs: a.songs + (s.renders[k].songs ?? 0), clips: a.clips + (s.renders[k].clips ?? 0),
    run: a.run + s.renders[k].running, fail: a.fail + s.renders[k].failed }), { songs: 0, clips: 0, run: 0, fail: 0 });
  const active = keys.filter((k) => s.renders[k].active);
  const nCols = Math.max(100, ...keys.map((k) => s.renders[k].total));   // jobs split 100 / 200 / 400 ways
  const cols = Array.from({ length: nCols }, (_, i) => `${i}`);
  // value: -1 failed, 0 waiting, 1..2 running (1 + share of its songs done), 3 done
  const val = (k: string, i: number) => {
    if (i >= s.renders[k].total) return null;                            // this job has fewer shards
    const t = s.renders[k].tasks?.[i];
    if (!t) return 0;
    if (t.s === "d") return 3;
    if (t.s === "f") return -1;
    if (t.s === "r") return 1 + (t.of ? Math.min(1, t.songs / t.of) : 0);
    return 0;
  };
  const color = (v: number) => v === 3 ? "#b4fbde" : v === -1 ? "#f33d15" : v >= 1
    ? `color-mix(in oklch, oklch(0.75 0.17 250) ${Math.round((v - 1) * 100)}%, oklch(0.45 0.15 250))` : "rgba(255,255,255,0.13)";
  return (
    <section className="mb-14 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Cpu className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Renders</h2>
        <span className="ml-auto text-xs text-muted-foreground">
          {active.length ? `${active.length} job${active.length > 1 ? "s" : ""} rendering: ${active.join(", ")}` : "nothing rendering now"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="render jobs" value={keys.length} hint={`${keys.length - active.length} finished`} />
        <Stat label="shards running" value={tot.run} hint="1 CPU core each" />
        <Stat label="songs rendered" value={tot.songs.toLocaleString()} hint="all jobs, counting each render pass" />
        <Stat label="clips written" value={tot.clips.toLocaleString()} hint={tot.fail ? `${tot.fail} shard(s) failed` : "no failed shards"} />
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[900px] text-left text-xs text-muted-foreground">
          <thead>
            <tr className="[&>th]:pb-1 [&>th]:pr-3 [&>th]:font-normal">
              <th>job</th><th className="w-40">shards</th><th>done / run / wait / failed</th><th>songs</th><th>clips</th>
              <th>took</th><th>songs/min</th><th>time left</th><th>shard time avg / max</th><th>peak memory</th><th>nodes</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const r = s.renders[k]; const W = (x: number) => `${(100 * x) / r.total}%`;
              return (
                <tr key={k} className={cn("border-t border-border/60 [&>td]:py-1 [&>td]:pr-3 tabular-nums", r.active && "text-white")}>
                  <td className="text-white">{k}{r.active && <span className="ml-1 text-primary">●</span>}</td>
                  <td>
                    <div className="flex h-1.5 w-36 overflow-hidden rounded-full bg-muted">
                      <div style={{ width: W(r.done), background: "#b4fbde" }} />
                      <div style={{ width: W(r.running), background: "oklch(0.6 0.2 250)" }} />
                      <div style={{ width: W(r.failed), background: "#f33d15" }} />
                    </div>
                  </td>
                  <td>{r.done} / {r.running} / {r.pending} / <span className={r.failed ? "text-rose-300" : ""}>{r.failed}</span></td>
                  <td>{r.songs ?? "—"}{r.songs_total ? ` / ${r.songs_total}` : ""}</td>
                  <td>{r.clips?.toLocaleString() ?? "—"}</td>
                  <td>{fmtMin(r.wall_min)}</td>
                  <td>{r.rate?.toFixed(1) ?? "—"}</td>
                  <td>{r.active ? fmtMin(r.eta_min) : "done"}</td>
                  <td>{r.shard_min_mean ?? "—"} / {r.shard_min_max ?? "—"} min</td>
                  <td>{r.rss_max ? `${(r.rss_max / 1024).toFixed(2)} GB` : "—"}</td>
                  <td>{r.nodes ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-5 mb-2 text-sm font-medium text-white">Every shard</div>
      <GridHeatmap
        rows={keys}
        columns={cols}
        labelEvery={nCols / 10}
        cellHeight={11}
        gap={nCols > 150 ? 0 : 1}
        value={(r, c) => val(keys[r], c)}
        color={color}
        title={(r, c) => {
          const t = s.renders[keys[r]].tasks?.[c];
          if (!t) return `${keys[r]} shard ${c}: waiting`;
          const st = { d: "done", r: "running", f: "FAILED", p: "waiting", c: "cancelled" }[t.s] ?? t.s;
          return `${keys[r]} shard ${c}: ${st}` + (t.s === "p" ? "" : ` · ${t.songs}${t.of ? `/${t.of}` : ""} songs · ${t.clips} clips · `
            + `${Math.round(t.el / 60)} min · ${(t.rss / 1024).toFixed(2)} GB${t.node ? ` · ${t.node}` : ""}`);
        }}
      />
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: "rgba(255,255,255,0.13)" }} />waiting</span>
        <span className="flex items-center gap-1"><span className="h-2.5 w-6 rounded-sm" style={{ background: "linear-gradient(90deg, oklch(0.45 0.15 250), oklch(0.75 0.17 250))" }} />running (darker = just started, lighter = nearly done)</span>
        <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: "#b4fbde" }} />done</span>
        <span className="flex items-center gap-1"><span className="size-2.5 rounded-sm" style={{ background: "#f33d15" }} />FAILED</span>
        <span>Hover a cell for its songs, clips, time, memory and node.</span>
      </div>
    </section>
  );
}

/** Stage 6 in full: both phases' curves, the three kinds of switches, and per-effect detection. */
function GateDetail({ gate, modules, jobs }: { gate: { classify: GatePhase | null; e2e: GatePhase | null }; modules: string[]; jobs: Job[] }) {
  const phases = (["classify", "e2e"] as const).filter((ph) => gate[ph]);
  const cur = gate.e2e?.vals.at(-1) ?? gate.classify?.vals.at(-1);
  return (
    <section className="mb-14 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Activity className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Stage 6 · gate</h2>
        <span className="ml-auto text-xs text-muted-foreground">
          {(["classify", "e2e"] as const).map((ph) => {
            const j = jobs.find((x) => x.name === `gate-${ph}`); const g = gate[ph];
            return `${ph}: ${g?.stopped ? "done" : j ? (j.state === "RUNNING" ? `${j.gpu ?? j.partition} · ${j.elapsed}` : "queued") : g ? "stopped" : "waiting"}`;
          }).join("   ·   ")}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="detection (balanced)" value={cur ? cur.detect.toFixed(3) : "—"} hint="0.5 = guessing, 1 = perfect" />
        <Stat label="notes, gate's switches" value={cur ? cur.gate.toFixed(3) : "—"} hint="note F1 on combination clips" />
        <Stat label="notes, known switches" value={cur ? cur.known.toFixed(3) : "—"} hint="what stage 5 trained with" />
        <Stat label="notes, no adapters" value={cur ? cur.none.toFixed(3) : "—"} hint="= plain_base" />
      </div>
      {phases.map((ph) => {
        const g = gate[ph]!; const marks = g.vals.map((v) => v.step);
        return (
          <div key={ph} className="mt-4">
            <div className="mb-1 text-xs font-medium text-white">{ph === "classify" ? "Phase 1 · naming the effects" : "Phase 2 · end to end (best notes)"}
              <span className="ml-2 font-normal text-muted-foreground">{g.stopped ?? (g.last ? `step ${g.last.step} · ${g.last.speed.toFixed(2)}/s · wait ${g.last.wait}%` : "starting")}</span></div>
            <div className="grid gap-x-6 gap-y-2 md:grid-cols-2 2xl:grid-cols-3">
              <MiniChart label="effect-label loss" marks={marks} fmt={(v) => v.toFixed(3)}
                         series={[{ name: "bce", color: "#c4b5fd", pts: g.trace.map((t) => [t[0], t[1]] as [number, number]) }]} />
              {ph === "e2e" && <MiniChart label="transcription loss" marks={marks} fmt={(v) => v.toFixed(3)}
                         series={[{ name: "task", color: "oklch(0.6 0.2 250)", pts: g.trace.map((t) => [t[0], t[2]] as [number, number]) }]} />}
              <MiniChart label="note F1 on combination clips" fmt={(v) => v.toFixed(3)} series={[
                { name: "gate", color: "#b4fbde", pts: g.vals.map((v) => [v.step, v.gate] as [number, number]) },
                { name: "known", color: "oklch(0.6 0.2 250)", pts: g.vals.map((v) => [v.step, v.known] as [number, number]) },
                { name: "none", color: "#f33d15", pts: g.vals.map((v) => [v.step, v.none] as [number, number]) },
              ]} />
              {ph === "classify" && <MiniChart label="detection (balanced)" lo={0.5} hi={1} fmt={(v) => v.toFixed(2)}
                         series={[{ name: "detect", color: "#fbbf24", pts: g.vals.map((v) => [v.step, v.detect] as [number, number]) }]} />}
            </div>
          </div>
        );
      })}
      {cur && Object.keys(cur.modules).length > 0 && (
        <div className="mt-4">
          <div className="mb-1 text-[11px] text-muted-foreground">How well each effect is detected at each check (balanced accuracy; dark = guessing, bright = perfect)</div>
          <GridHeatmap
            rows={modules}
            columns={[...(gate.classify?.vals ?? []).map((v) => `c${v.step / 1000}k`), ...(gate.e2e?.vals ?? []).map((v) => `e${v.step / 1000}k`)]}
            cellHeight={13}
            value={(r, c) => {
              const all = [...(gate.classify?.vals ?? []), ...(gate.e2e?.vals ?? [])];
              return all[c]?.modules[modules[r]]?.[0] ?? null;
            }}
            color={(v) => f1Color(0.6 + (v - 0.5) * 0.7)}
            title={(r, c, v) => `${modules[r]}: ${v === null ? "—" : v.toFixed(2)}`}
          />
        </div>
      )}
    </section>
  );
}

function JointPanel({ j, modules, job }: { j: Joint; modules: string[]; job?: Job }) {
  const ref = j.vals[0];
  const cur = j.vals.at(-1);
  const best = j.vals.reduce<JointVal | null>((b, v) => (!b || v.score > b.score ? v : b), null);
  const meanDelta = (v: JointVal) => modules.reduce((a, m) => a + ((v.modules[m] ?? 0) - (ref?.modules[m] ?? 0)), 0) / modules.length;
  const steps = j.vals.map((v) => `${v.step / 1000}k`);
  return (
    <div className="mt-3 space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="state" value={j.stopped ? "done" : job ? "training" : "waiting"} hint={job ? `${job.gpu ?? job.partition} · ${job.elapsed}` : undefined} />
        <Stat label="step" value={j.last ? j.last.step.toLocaleString() : "0"} hint={j.last ? `${j.last.speed.toFixed(2)}/s · wait ${j.last.wait}%` : "validating step 0"} />
        <Stat label="combination F1" value={cur ? cur.combo.toFixed(3) : "—"} hint={cur && ref ? `${cur.combo - ref.combo >= 0 ? "+" : ""}${(cur.combo - ref.combo).toFixed(3)} vs alone` : undefined} />
        <Stat label="modules, mean change" value={cur && ref ? `${meanDelta(cur) >= 0 ? "+" : ""}${meanDelta(cur).toFixed(3)}` : "—"} hint={best ? `best at step ${best.step}` : undefined} />
      </div>
      {j.vals.length > 0 && (
        <div>
          <div className="mb-1 text-[11px] text-muted-foreground">Each module on its own clips: change from step 0 (trained alone). Green better, red worse; ±0.02 saturates.</div>
          <GridHeatmap
            rows={["COMBINATION", ...modules]}
            columns={steps}
            cellHeight={13}
            labelWidth={96}
            value={(r, c) => {
              const v = j.vals[c];
              if (r === 0) return v.combo - ref.combo;
              const m = modules[r - 1];
              return v.modules[m] === undefined || ref.modules[m] === undefined ? null : v.modules[m] - ref.modules[m];
            }}
            color={(v) => deltaColor(v, 0.02)}
            title={(r, c, v) => `${r === 0 ? "combination" : modules[r - 1]} at step ${j.vals[c].step}: ${v === null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(3)}`}`}
          />
        </div>
      )}
      {j.stopped && <div className="text-[11px]">{j.stopped}</div>}
    </div>
  );
}

const STAGES = ["Plain renders", "Plain adapter", "Module renders", "Module adapters", "Joint refinement", "Gate", "Final comparison"];

function Kpis({ s, active, rendersDone, adaptersDone, total }: { s: Snapshot; active: number; rendersDone: number; adaptersDone: number; total: number }) {
  const gpu = s.jobs.filter((j) => /^(mod-|gate-|s5-|win|mr-base)/.test(j.name));
  const gpuRun = gpu.filter((j) => j.state === "RUNNING").length;
  const shards = Object.values(s.renders);
  const shardRun = shards.reduce((a, r) => a + r.running, 0);
  const shardDone = shards.reduce((a, r) => a + r.done, 0);
  const shardAll = shards.reduce((a, r) => a + r.total, 0);
  const tested = s.lanes.flat().filter((m) => s.modules[m]?.test?.f1 !== undefined && s.modules[m]?.baseline?.f1 !== undefined);
  const better = tested.filter((m) => s.modules[m].test!.f1! > s.modules[m].baseline!.f1!).length;
  const ageMin = Math.max(0, Math.round((Date.now() - new Date(s.time.replace(" ", "T")).getTime()) / 60000));
  const alerts = s.alerts?.length ?? 0;
  const items: { label: string; value: React.ReactNode; hint: string; tone?: string }[] = [
    { label: "stage", value: `${active}/7`, hint: STAGES[active - 1] },
    { label: "module renders", value: `${rendersDone}/${total}`, hint: `${shardDone.toLocaleString()}/${shardAll.toLocaleString()} shards` },
    { label: "CPU shards running", value: shardRun, hint: "1 core each" },
    { label: "adapters tested", value: `${adaptersDone}/${total}`, hint: tested.length ? `${better} beat plain_base` : "none yet" },
    { label: "GPU lanes running", value: `${gpuRun}/${gpu.length}`, hint: gpu.filter((j) => j.state === "RUNNING").map((j) => j.gpu ?? j.partition).join(", ") || "waiting" },
    { label: "alerts", value: alerts, hint: alerts ? "see the red box" : "all clear", tone: alerts ? "text-rose-300" : "text-emerald-300" },
    { label: "snapshot", value: `${ageMin} min ago`, hint: s.time.slice(11), tone: ageMin > 12 ? "text-amber-300" : undefined },
  ];
  return (
    <div className="mb-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-4 xl:grid-cols-7">
      {items.map((it) => (
        <div key={it.label} className="min-w-0 bg-[#0b0b10] px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{it.label}</div>
          <div className={cn("text-xl font-semibold tabular-nums text-white", it.tone)}>{it.value}</div>
          <div className="truncate text-[11px] text-muted-foreground">{it.hint}</div>
        </div>
      ))}
    </div>
  );
}

// ---- page ----
// ---- the NEW BASE plan (NEW_BASE_PLAN.md): renders -> base -> instrument adapters -> combining them
type SetTest = Partial<{ recall: number; precision: number; f1: number; dbl: number; start_abs: number; quick: number[] }>;
type ModelTests = Partial<Record<"mod_plain" | "mod_combo" | "mod_fast", SetTest>> & {
  real?: { strict: number; precision: number; recall: number; tolerant: number; pc: number; notes_per_s: number };
};
type Experiment = {
  label: string; tests: ModelTests;
  train: { last: Run["last"]; stopped: string | null; checks: number; best: number | null;
           vals?: [number, number, number, number][];      // step, F1, precision, recall
           trace?: [number, number, number][] };            // step, loss, steps/s
};
const EXP_COLOR: Record<string, string> = { A_global: "#94a3b8", B_layers: "#fbbf24", C_soft: "oklch(0.6 0.2 250)", D_halfoct: "#b4fbde" };
type NewBase = {
  run: string; base: Run | null; base_tests: ModelTests; families: string[];
  adapters: Record<string, Run | null>; adapter_tests: Record<string, ModelTests>;
  fusion: Run | null; fusion_tests: ModelTests; experiments: Record<string, Experiment>;
};

const NEW_STAGES = ["Renders", "Base", "Instrument adapters", "Combining", "Final comparison"];
const SETS: [keyof ModelTests, string][] = [["mod_plain", "plain"], ["mod_combo", "combination"], ["mod_fast", "fast"]];
const fmt3 = (v?: number | null) => (v == null ? "—" : v.toFixed(3));
const famName = (f: string) => f.replace(/_/g, " ");

function newPlanState(s: Snapshot) {
  const nb = s.newbase;
  const rb = s.renders["base"], rt = s.renders["basetempo"];
  const rendersDone = !!rb && !!rt && !rb.active && !rt.active && rb.done > 0 && rt.done > 0;
  const baseDone = !!nb?.base?.stopped && !!nb?.base_tests?.real;
  const adaptersDone = nb ? nb.families.filter((f) => nb.adapter_tests[f]?.mod_combo?.f1 !== undefined).length : 0;
  const done = [rendersDone, baseDone, !!nb && adaptersDone === nb.families.length, !!nb?.fusion?.stopped && !!nb?.fusion_tests?.real, false];
  return { nb, rb, rt, done, active: done.findIndex((d) => !d) + 1 || done.length, adaptersDone };
}

function NewKpis({ s }: { s: Snapshot }) {
  const { nb, rb, rt, active, adaptersDone } = newPlanState(s);
  const shards = (rb?.done ?? 0) + (rt?.done ?? 0), shardsAll = (rb?.total ?? 0) + (rt?.total ?? 0);
  const clips = (rb?.clips ?? 0) + (rt?.clips ?? 0);
  const gpu = s.jobs.filter((j) => j.state === "RUNNING" && /^(base|inst|fusion|win|basetest)/.test(j.name));
  const last = nb?.base?.last; const best = nb?.base?.vals.reduce((b, v) => Math.max(b, v.f1), 0);
  const ageMin = Math.max(0, Math.round((Date.now() - new Date(s.time.replace(" ", "T")).getTime()) / 60000));
  const alerts = s.alerts?.length ?? 0;
  const items: { label: string; value: React.ReactNode; hint: string; tone?: string }[] = [
    { label: "stage", value: `${active}/5`, hint: NEW_STAGES[active - 1] },
    { label: "render shards", value: `${shards}/${shardsAll || "—"}`, hint: `${clips.toLocaleString()} clips written` },
    { label: "CPU shards running", value: (rb?.running ?? 0) + (rt?.running ?? 0), hint: rb?.eta_min ? `main pass ~${fmtMin(rb.eta_min)} left` : "1 core each" },
    { label: "base", value: last ? `step ${last.step.toLocaleString()}` : "not started", hint: best ? `best val F1 ${best.toFixed(3)}` : last ? "training, first check at 2,000" : "starts after the renders" },
    { label: "instrument adapters", value: `${adaptersDone}/${nb?.families.length ?? 14}`, hint: "tested" },
    { label: "GPU jobs running", value: gpu.length, hint: gpu.map((j) => `${j.name} (${j.gpu ?? j.partition})`).join(", ") || "none" },
    { label: "alerts", value: alerts, hint: alerts ? "see the red box" : "all clear", tone: alerts ? "text-rose-300" : "text-emerald-300" },
    { label: "snapshot", value: `${ageMin} min ago`, hint: s.time.slice(11), tone: ageMin > 12 ? "text-amber-300" : undefined },
  ];
  return (
    <div className="mb-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-4 2xl:grid-cols-8">
      {items.map((it) => (
        <div key={it.label} className="min-w-0 bg-[#0b0b10] px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{it.label}</div>
          <div className={cn("text-xl font-semibold tabular-nums text-white", it.tone)}>{it.value}</div>
          <div className="truncate text-[11px] text-muted-foreground">{it.hint}</div>
        </div>
      ))}
    </div>
  );
}

/** One model's tests at the app's 0.9 bar: per render set and on real songs, optionally beside a reference. */
function TestTable({ rows }: { rows: { name: string; t: ModelTests }[] }) {
  const has = rows.some((r) => SETS.some(([k]) => r.t[k]) || r.t.real);
  if (!has) return <div className="text-[11px] text-muted-foreground">Tests run automatically when training finishes.</div>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-left text-xs text-muted-foreground">
        <thead>
          <tr className="[&>th]:pb-1 [&>th]:pr-3 [&>th]:font-normal">
            <th>model</th>
            {SETS.map(([k, n]) => <th key={k}>{n}: F1 · double /1000</th>)}
            <th>quick notes 32-96 ms (combination)</th><th>start error</th>
            <th>real songs: strict · P · R</th><th>notes/s (human 8.0)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ name, t }) => {
            const q = t.mod_combo?.quick; const qm = q ? (q[1] + q[2] + q[3]) / 3 : undefined;
            return (
              <tr key={name} className="border-t border-border/60 tabular-nums [&>td]:py-1 [&>td]:pr-3">
                <td className="text-white">{name}</td>
                {SETS.map(([k]) => <td key={k}>{fmt3(t[k]?.f1)} · {t[k]?.dbl?.toFixed(1) ?? "—"}</td>)}
                <td>{fmt3(qm)}</td>
                <td>{t.mod_combo?.start_abs != null ? `${t.mod_combo.start_abs.toFixed(1)} ms` : "—"}</td>
                <td className="text-white">{t.real ? `${t.real.strict.toFixed(3)} · ${t.real.precision.toFixed(3)} · ${t.real.recall.toFixed(3)}` : "—"}</td>
                <td>{t.real?.notes_per_s?.toFixed(1) ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BaseDetail({ s }: { s: Snapshot }) {
  const nb = s.newbase!; const b = nb.base;
  const job = s.jobs.find((j) => j.name.startsWith("base") && j.name !== "basetempo" && j.state === "RUNNING");
  const ref = nb.experiments["C_soft"];
  return (
    <section className="mb-10 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Activity className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Base · {nb.run}</h2>
        <span className="text-xs text-muted-foreground">~20M parameters, learned soft windows, soft onset target, trained on the new renders</span>
        <span className="ml-auto text-xs text-muted-foreground">
          {b?.stopped ?? (job ? `${job.gpu ?? job.partition} · ${job.elapsed}` : b ? "stopped" : "starts when the renders finish")}
        </span>
      </div>
      {b && b.last ? (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
            <Stat label="step" value={b.last.step.toLocaleString()} hint="of 120,000 max" />
            <Stat label="speed" value={`${b.last.speed.toFixed(2)}/s`} hint={!b.stopped ? `≤ ${fmtMin((120000 - b.last.step) / b.last.speed / 60)} left` : undefined} />
            <Stat label="loss" value={b.last.loss.toFixed(4)} />
            <Stat label="best val F1" value={b.vals.length ? Math.max(...b.vals.map((v) => v.f1)).toFixed(3) : "—"} hint={`${b.vals.length} checks`} />
            <Stat label="C reached" value={ref?.train.best?.toFixed(3) ?? "—"} hint="on its (older) renders" />
          </div>
          <div className="mt-4 grid gap-x-6 gap-y-2 md:grid-cols-2 2xl:grid-cols-3">
            <MiniChart label="speed (steps/s)" lo={0} fmt={(v) => v.toFixed(1)}
                       series={[{ name: "speed", color: "#c4b5fd", pts: b.trace.map((t) => [t[0], t[4]] as [number, number]) }]} />
            <MiniChart label="loss" skipStart={0.05} marks={b.vals.map((v) => v.step)} fmt={(v) => v.toFixed(3)}
                       series={[{ name: "loss", color: "oklch(0.6 0.2 250)", pts: b.trace.map((t) => [t[0], t[1]] as [number, number]) }]} />
            <MiniChart label="validation" fmt={(v) => v.toFixed(3)} series={[
              { name: "note F1", color: "#b4fbde", pts: b.vals.map((v) => [v.step, v.f1] as [number, number]) },
              { name: "recall", color: "oklch(0.6 0.2 250)", pts: b.vals.map((v) => [v.step, v.r] as [number, number]) },
              { name: "precision", color: "#fbbf24", pts: b.vals.map((v) => [v.step, v.p] as [number, number]) },
            ]} />
          </div>
        </>
      ) : (
        <div className="text-xs text-muted-foreground">Not started yet: it trains on the new renders once both render passes finish.</div>
      )}
      <div className="mt-4 mb-1 text-xs font-medium text-white">Tests (0.9 bar), against C, the best model so far</div>
      <TestTable rows={[{ name: `${nb.run} (new base)`, t: nb.base_tests }, ...(ref ? [{ name: "C (reference)", t: ref.tests }] : [])]} />
    </section>
  );
}

function ExperimentsPanel({ s }: { s: Snapshot }) {
  const ex = s.newbase?.experiments ?? {};
  const ids = Object.keys(ex);
  if (!ids.length) return null;
  return (
    <section className="mb-10 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Activity className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Architecture tests</h2>
        <span className="text-xs text-muted-foreground">same data (arr_scratch's renders), 60k steps; only the attention or tokens differ</span>
      </div>
      <div className="mb-3 flex flex-wrap gap-2 text-[11px]">
        {ids.map((n) => {
          const e = ex[n]; const st = e.train.stopped ? "trained" : e.train.last ? `training · step ${e.train.last.step.toLocaleString()}` : "queued";
          return <span key={n} className="rounded-md border border-border px-2 py-0.5 text-muted-foreground"><span className="text-white">{e.label}</span> · {st}{e.train.best ? ` · best val ${e.train.best.toFixed(3)}` : ""}</span>;
        })}
      </div>
      <div className="mb-4 grid gap-x-6 gap-y-2 md:grid-cols-2 2xl:grid-cols-4">
        <MiniChart label="validation note F1" fmt={(v) => v.toFixed(3)}
                   series={ids.map((n) => ({ name: n[0], color: EXP_COLOR[n] ?? "#fff", pts: (ex[n].train.vals ?? []).map((v) => [v[0], v[1]] as [number, number]) }))} />
        <MiniChart label="validation recall / precision (D vs A)" fmt={(v) => v.toFixed(3)}
                   series={[["D_halfoct", 3, "D recall"], ["A_global", 3, "A recall"], ["D_halfoct", 2, "D prec"], ["A_global", 2, "A prec"]]
                     .filter(([n]) => ex[n as string]).map(([n, k, nm], i) => ({ name: nm as string, color: ["#b4fbde", "#94a3b8", "#34d399", "#64748b"][i],
                       pts: (ex[n as string].train.vals ?? []).map((v) => [v[0], v[k as number]] as [number, number]) }))} />
        <MiniChart label="training loss" skipStart={0.03} fmt={(v) => v.toFixed(3)}
                   series={ids.map((n) => ({ name: n[0], color: EXP_COLOR[n] ?? "#fff", pts: (ex[n].train.trace ?? []).map((t) => [t[0], t[1]] as [number, number]) }))} />
        <MiniChart label="speed (steps/s)" lo={0} fmt={(v) => v.toFixed(1)}
                   series={ids.map((n) => ({ name: n[0], color: EXP_COLOR[n] ?? "#fff", pts: (ex[n].train.trace ?? []).map((t) => [t[0], t[2]] as [number, number]) }))} />
      </div>
      <TestTable rows={ids.map((n) => ({ name: ex[n].label, t: ex[n].tests }))} />
    </section>
  );
}

function AdaptersPanel({ s }: { s: Snapshot }) {
  const nb = s.newbase!;
  return (
    <section className="mb-10 rounded-xl border border-border bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Activity className="size-4 text-primary" />
        <h2 className="text-sm font-semibold text-white">Instrument adapters</h2>
        <span className="text-xs text-muted-foreground">one per family on the frozen base; (a) the whole arrangement on that family, (b) its notes inside mixes</span>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 2xl:grid-cols-7">
        {nb.families.map((f) => {
          const r = nb.adapters[f]; const t = nb.adapter_tests[f];
          const st = t?.mod_combo?.f1 !== undefined ? "done" : r?.stopped ? "testing" : r?.last ? "training" : "waiting";
          return (
            <div key={f} className={cn("rounded-lg border border-border p-2.5", st === "training" && "border-primary/60")}>
              <div className="flex items-center gap-1.5 text-xs">
                <span className="font-medium text-white">{famName(f)}</span>
                <span className={cn("ml-auto text-[10px] uppercase", st === "done" ? "text-emerald-300" : st === "training" ? "text-primary" : "text-muted-foreground")}>{st}</span>
              </div>
              {st === "training" && r?.last && <Bar value={r.last.step} total={20000} className="mt-1.5" />}
              {st === "done" && <div className="mt-1 text-[11px] text-muted-foreground">F1 {fmt3(t.mod_combo?.f1)}</div>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function NewTimeline({ s }: { s: Snapshot }) {
  const { nb, rb, rt, done, active, adaptersDone } = newPlanState(s);
  const bar = (r?: Render) => r && (
    <div className="mt-1 max-w-sm">
      <Bar value={r.done} total={r.total} />
      <div className="mt-0.5 text-[11px]">{r.done}/{r.total} shards · {r.running} running · {(r.clips ?? 0).toLocaleString()} clips{r.failed ? ` · ${r.failed} failed` : ""}{r.active && r.eta_min ? ` · ~${fmtMin(r.eta_min)} left` : ""}</div>
    </div>
  );
  const stage = (i: number, date: string, title: string, body: React.ReactNode) => (
    <TimelineItem step={i} key={i}>
      <TimelineIndicator>{done[i - 1] && <Check className="size-3.5" />}</TimelineIndicator>
      {i < 5 && <TimelineSeparator />}
      <TimelineHeader><TimelineDate>{date}</TimelineDate><TimelineTitle className="text-white">{title}</TimelineTitle></TimelineHeader>
      <TimelineContent>{body}</TimelineContent>
    </TimelineItem>
  );
  const b = nb?.base;
  return (
    <Timeline value={active}>
      {stage(1, "Stage 1 · CPU", "Renders", <>
        POP909 + Music2MIDI training songs: the arrangement plus extra parts and drums, 14 instrument families from 3
        soundfonts, synth-style tone (no pitch effects), loud mastering, melodies on lead instruments; targets at the written times.
        <div className="mt-2 text-[11px] text-white">Main pass (1 piano + 8 multi per song)</div>{bar(rb)}
        <div className="mt-2 text-[11px] text-white">Sped-up pass (1.2-2x, 5 multi per song)</div>{bar(rt)}
      </>)}
      {stage(2, "Stage 2 · GPU", "Base", <>
        ~20M parameters, learned soft time windows, soft onset target, front end calibrated on the new renders.
        <div className="mt-1 text-[11px]">{b?.stopped ?? (b?.last ? `training · step ${b.last.step.toLocaleString()}` : "waits for the renders")}</div>
      </>)}
      {stage(3, "Stage 3 · GPU", `Instrument adapters · ${adaptersDone}/${nb?.families.length ?? 14}`,
        "One LoRA per family on the frozen base: the whole arrangement on that family, and that family's notes inside mixes (extra parts counted).")}
      {stage(4, "Stage 4 · GPU", "Combining the adapters",
        "Attention over the adapters' outputs, trained on mixed renders with the adapters frozen. No bundle adjustment.")}
      {stage(5, "Stage 5", "Final comparison", "Real songs (evaluation only) and your listening, against C, arr_scratch and the attack adapter.")}
    </Timeline>
  );
}

export default function ProgressPage() {
  const [snap, setSnap] = React.useState<Snapshot | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);

  const load = React.useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/progress-data");
      if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`);
      setSnap(await res.json());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  const modules = snap ? snap.lanes.flat() : [];
  const states = snap ? modules.map((m) => moduleState(snap, m)) : [];
  const rendersDone = snap ? modules.filter((m) => (snap.renders[m]?.done ?? 0) >= 100).length : 0;
  const adaptersDone = states.filter((x) => x === "done").length;
  const plainRender = snap?.renders.plain;
  const plainDone = !!snap?.plain.merged && snap.plain.test?.f1 !== undefined;

  // stage the timeline is on (1-based); later stages are not built yet
  const done = [
    !!plainRender && plainRender.done >= plainRender.total,
    plainDone,
    snap ? rendersDone === modules.length : false,
    snap ? adaptersDone === modules.length : false,
    !!snap?.joint?.stopped, !!snap?.gate?.e2e?.stopped, false,
  ];
  const active = done.findIndex((d) => !d) + 1 || done.length;

  const jobByName = (n: string) => snap?.jobs.find((j) => j.name === n && j.state === "RUNNING");
  const live: { title: string; run: Run; job?: Job }[] = [];
  if (snap?.plain.train && !snap.plain.train.stopped && !snap.plain.merged)
    live.push({ title: "Plain adapter", run: snap.plain.train, job: jobByName("mod-plain") });
  if (snap)
    modules.forEach((m, i) => {
      const t = snap.modules[m]?.train;
      if (states[i] === "training" && t) live.push({ title: `Module adapter: ${m}`, run: t, job: jobByName(`mod-lane${snap.modules[m].lane}`) });
    });
  const gpuJobs = snap?.jobs.filter((j) => /^(mod-|gate-|s5-|win|mr-base|base|inst|fusion)/.test(j.name)) ?? [];

  return (
    <main className="mx-auto max-w-[2560px] px-4 pb-24 pt-28 text-white sm:px-6 xl:px-10">
      <header className="mb-10 flex flex-wrap items-end gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Training progress</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            New base: renders → base → instrument adapters → combining them (NEW_BASE_PLAN.md).
          </p>
        </div>
        <button
          onClick={load}
          className="ml-auto flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-white"
        >
          <RefreshCw className={cn("size-3.5", loading && "animate-spin")} />
          {snap ? `Cluster at ${snap.time.slice(11)}` : "Loading"}
        </button>
      </header>

      <CleanupPanel />

      <VoicePanel />

      {snap && <NewKpis s={snap} />}

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-rose-400/40 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
          <TriangleAlert className="size-4" /> Couldn't reach the cluster: {error}
        </div>
      )}

      {snap?.schedule_stale && (
        <div className="mb-6 flex items-center gap-2 rounded-lg border border-amber-300/40 bg-amber-300/10 px-3 py-2 text-sm text-amber-100">
          <TriangleAlert className="size-4 shrink-0" />
          The cluster's 5-minute snapshot stopped (last one: {snap.schedule_stale}), so email alerts are off. Showing a live read instead.
        </div>
      )}

      {!!snap?.notices?.length && (
        <section className="mb-4 rounded-lg border border-primary/40 bg-primary/10 px-4 py-3 text-sm text-white">
          <div className="mb-1 flex items-center gap-2 font-semibold"><Activity className="size-4 text-primary" /> Next step needed</div>
          <ul className="list-disc space-y-0.5 pl-6 text-xs">{snap.notices.map((n) => <li key={n.key}>{n.text}</li>)}</ul>
        </section>
      )}

      {!!snap?.alerts?.length && (
        <section className="mb-8 rounded-lg border border-rose-400/40 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
          <div className="mb-1 flex items-center gap-2 font-semibold">
            <TriangleAlert className="size-4" /> {snap.alerts.length} need{snap.alerts.length === 1 ? "s" : ""} attention
          </div>
          <ul className="list-disc space-y-0.5 pl-6 text-xs">
            {snap.alerts.map((a) => <li key={a.key}>{a.text}</li>)}
          </ul>
        </section>
      )}

      {/* the new plan: training and health on the left, timeline and jobs on the right */}
      {snap && (
        <div className="xl:grid xl:grid-cols-12 xl:gap-10">
          <div className="min-w-0 xl:col-span-7 2xl:col-span-8">
            {snap.newbase?.base?.last && !snap.newbase.base.stopped && (
              <section className="mb-10">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Training now</h2>
                <LiveRun title={`Base · ${snap.newbase.run}`} run={snap.newbase.base} total={120000} valEvery={2000}
                         note="Validates every 2,000 steps; keeps the best checkpoint; runs to 120,000."
                         job={snap.jobs.find((j) => j.name.startsWith("base") && j.name !== "basetempo" && j.state === "RUNNING")} />
              </section>
            )}
            {snap.newbase && <BaseDetail s={snap} />}
            <ExperimentsPanel s={snap} />
            {snap.newbase && <AdaptersPanel s={snap} />}
            <RenderDetail s={snap} />
          </div>
          <div className="min-w-0 xl:col-span-5 2xl:col-span-4">
            <NewTimeline s={snap} />
            {gpuJobs.length > 0 && (
              <section className="mt-12">
                <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <Cpu className="size-3.5" /> Cluster jobs
                </h2>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[420px] text-left text-xs text-muted-foreground">
                    <tbody>
                      {gpuJobs.map((j) => (
                        <tr key={j.id} className="border-t border-border/60">
                          <td className="py-1.5 pr-3 text-white">{j.name}</td>
                          <td className={cn("pr-3", j.state === "RUNNING" && "text-primary")}>{j.state.toLowerCase()}</td>
                          <td className="pr-3">{j.partition}{j.gpu && <span className="text-white"> · {j.gpu}</span>}</td>
                          <td className="tabular-nums">{j.state === "RUNNING" ? `${j.elapsed} / ${j.limit}` : j.reason.replace(/[()]/g, "")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
          </div>
        </div>
      )}

      {!snap && !error && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Asking the cluster…
        </div>
      )}

      {/* the previous line, kept for reference */}
      {snap && (
        <details className="mt-16 rounded-xl border border-border bg-white/[0.01] p-4">
          <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-white">
            Previous line: modular adapters (plain base, 22 effect adapters, joint refinement, gate) · finished / paused
          </summary>
          <div className="mt-6">
            <Kpis s={snap} active={active} rendersDone={rendersDone} adaptersDone={adaptersDone} total={modules.length} />
      {/* wide screens: health on the left, timeline and jobs on the right */}
      <div className="xl:grid xl:grid-cols-12 xl:gap-10">
      <div className="min-w-0 xl:col-span-7 2xl:col-span-8">
      {live.length > 0 && (
        <section className="mb-12 grid gap-3 2xl:grid-cols-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground 2xl:col-span-2">Training now</h2>
          {live.map((l) => <LiveRun key={l.title} {...l} />)}
        </section>
      )}

      {snap?.gate && (snap.gate.classify || snap.gate.e2e) && <GateDetail gate={snap.gate} modules={modules} jobs={snap.jobs} />}
      {snap?.joint && <JointDetail j={snap.joint} modules={modules} job={jobByName("mod-joint")} />}
      {snap && <TrainingHealth s={snap} />}
      </div>
      <div className="min-w-0 xl:col-span-5 2xl:col-span-4">

      {!snap && !error && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Asking the cluster…
        </div>
      )}

      {snap && (
        <Timeline value={active}>
          <TimelineItem step={1}>
            <TimelineIndicator>{done[0] && <Check className="size-3.5" />}</TimelineIndicator>
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 1 · CPU</TimelineDate>
              <TimelineTitle className="text-white">Plain renders</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              POP909 arrangements played by stock instruments, with no effects and notes exactly as written.
              {plainRender && (
                <div className="mt-2 max-w-sm">
                  <Bar value={plainRender.done} total={plainRender.total} />
                  <div className="mt-1 text-[11px]">{plainRender.done}/{plainRender.total} shards</div>
                </div>
              )}
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={2}>
            <TimelineIndicator>{done[1] && <Check className="size-3.5" />}</TimelineIndicator>
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 2 · GPU</TimelineDate>
              <TimelineTitle className="text-white">Plain adapter → plain_base</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              A LoRA on the frozen piano transcriber, then folded in to make the base for every module.
              {snap.plain.test && (
                <table className="mt-3 w-full max-w-lg text-left text-xs">
                  <thead className="text-muted-foreground">
                    <tr><th className="pb-1 font-normal">at 0.9</th><th className="font-normal">recall</th><th className="font-normal">precision</th><th className="font-normal">F1</th><th className="font-normal">double /1000</th><th className="font-normal">start err</th></tr>
                  </thead>
                  <tbody><TestRow name="plain_base" t={snap.plain.test} /></tbody>
                </table>
              )}
              {snap.plain.train?.stopped && <div className="mt-2 text-[11px]">{snap.plain.train.stopped}</div>}
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={3}>
            <TimelineIndicator>{done[2] && <Check className="size-3.5" />}</TimelineIndicator>
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 3 · CPU</TimelineDate>
              <TimelineTitle className="text-white">Module renders · {rendersDone}/{modules.length}</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              Plain plus exactly one effect each. No note is ever added or removed.
              <Bar value={rendersDone} total={modules.length} className="mt-2 max-w-sm" />
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={4}>
            <TimelineIndicator>{done[3] && <Check className="size-3.5" />}</TimelineIndicator>
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 4 · GPU · 7 lanes</TimelineDate>
              <TimelineTitle className="text-white">Module adapters · {adaptersDone}/{modules.length}</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              One thin LoRA per effect on plain_base, each tested against plain_base on its own effect.
              <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-2 2xl:grid-cols-3">
                {modules.map((m) => <ModuleCard key={m} s={snap} m={m} />)}
              </div>
              {adaptersDone > 0 && (
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full min-w-[480px] text-left text-xs">
                    <thead className="text-muted-foreground">
                      <tr><th className="pb-1 font-normal">at 0.9, own effect</th><th className="font-normal">recall</th><th className="font-normal">precision</th><th className="font-normal">F1</th><th className="font-normal">double /1000</th><th className="font-normal">start err</th></tr>
                    </thead>
                    <tbody>
                      {modules.filter((m) => snap.modules[m]?.test?.f1 !== undefined).flatMap((m) => [
                        <TestRow key={m} name={m} t={snap.modules[m].test} />,
                        <TestRow key={`${m}-b`} name={`  plain_base`} t={snap.modules[m].baseline} />,
                      ])}
                    </tbody>
                  </table>
                </div>
              )}
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={5}>
            <TimelineIndicator />
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 5 · GPU</TimelineDate>
              <TimelineTitle className="text-white">Joint refinement (bundle adjustment)</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              All 22 adapters side by side on plain_base, refined together on clips with 2-5 effects at once (plus each
              module's own clips), each pulled toward its own weights.
              {snap.joint && <JointPanel j={snap.joint} modules={modules} job={jobByName("mod-joint")} />}
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={6}>
            <TimelineIndicator />
            <TimelineSeparator />
            <TimelineHeader>
              <TimelineDate>Stage 6 · GPU</TimelineDate>
              <TimelineTitle className="text-white">Gate over adapters</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>
              Listens and sets each adapter's switch, with alternating local (±2 s) and global attention. First it
              learns to name the effects; then, end to end, the switches that give the best notes.
              {snap.gate && (snap.gate.classify || snap.gate.e2e) && (
                <div className="mt-2 text-[11px]">
                  {(["classify", "e2e"] as const).map((ph) => {
                    const g = snap.gate![ph]; const v = g?.vals.at(-1);
                    return <div key={ph}>{ph}: {!g ? "waiting" : g.stopped ? "done" : `step ${g.last?.step ?? 0}`}
                      {v && ` · detection ${v.detect.toFixed(3)} · notes with gate ${v.gate.toFixed(3)} (known ${v.known.toFixed(3)}, none ${v.none.toFixed(3)})`}</div>;
                  })}
                </div>
              )}
            </TimelineContent>
          </TimelineItem>

          <TimelineItem step={7}>
            <TimelineIndicator />
            <TimelineHeader>
              <TimelineDate>Stage 7</TimelineDate>
              <TimelineTitle className="text-white">Final comparison</TimelineTitle>
            </TimelineHeader>
            <TimelineContent>Real songs, against the attack adapter, then your listening test.</TimelineContent>
          </TimelineItem>
        </Timeline>
      )}

      {gpuJobs.length > 0 && (
        <section className="mt-16 xl:mt-10">
          <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <Cpu className="size-3.5" /> Cluster jobs
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-left text-xs text-muted-foreground">
              <tbody>
                {gpuJobs.map((j) => (
                  <tr key={j.id} className="border-t border-border/60">
                    <td className="py-1.5 pr-3 text-white">{j.name}</td>
                    <td className="pr-3">{snap!.lanes[Number(j.name.replace("mod-lane", ""))]?.join(", ") ?? ""}</td>
                    <td className={cn("pr-3", j.state === "RUNNING" && "text-primary")}>{j.state.toLowerCase()}</td>
                    <td className="pr-3">{j.partition}{j.gpu && <span className="text-white"> · {j.gpu}</span>}</td>
                    <td className="tabular-nums">{j.state === "RUNNING" ? `${j.elapsed} / ${j.limit}` : j.reason.replace(/[()]/g, "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      </div>
      </div>
          </div>
        </details>
      )}
    </main>
  );
}
