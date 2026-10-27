import * as React from "react";
import { Lines, Tile } from "./VoicePanel";

// ---- shape of /cleanup-data (backend/app/routers/progress.py: the cutrain log + squeue) ----
type PRF = { R: number; P: number; F1: number; kept: number };
type Pos = { notes: number; acc: number; off_share: number; err_ms_median: Record<string, number>;
             within_10ms: Record<string, number>; worse_than_50ms: Record<string, number> };
type Val = { step: number; position?: Pos; [k: string]: PRF | Pos | number | undefined };
type Job = { id: string; name: string; state: string; elapsed: string; limit: string; node: string; reason: string };
type CleanupData = {
  data: number | null; clips: number | null; jobs: Job[]; vals: Val[]; trace: [number, number, number | null, number | null][];
  info: string[]; finished: boolean; log: string | null; error?: string; stale?: string; version?: number; total?: number;
  data2?: { clips: number; of: number; beats: number; beats_of: number } | null;
};

const C1 = "#5b8def", C2 = "#22a374", C3 = "#b8822a", C4 = "#9b7bea";
const BARS = ["keep 0.3", "keep 0.4", "keep 0.5", "keep 0.6", "keep 0.7"];
const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${Math.round(v * 100)}%`;
const prf = (v: Val, k: string) => (k !== "position" && v[k] && typeof v[k] === "object" ? (v[k] as PRF) : undefined);
const bestBar = (v: Val) => BARS.filter((k) => prf(v, k)).reduce((a, k) => (prf(v, k)!.F1 > prf(v, a)!.F1 ? k : a), BARS[0]);

// the onset-bar review that started this (eval/thr_review.py, 2026-10-27): fixed numbers
const REVIEW: [string, string, string, string, string][] = [
  ["0.8", "0.692 / 0.820", "0.856", "0.477", "15.1 / 78.7%"],
  ["0.85", "0.714 / 0.791", "0.869", "0.496", "13.1 / 74.7%"],
  ["0.9", "0.719 / 0.722", "0.871", "0.510", "10.7 / 67.1%"],
];

export default function CleanupPanel() {
  const [d, setD] = React.useState<CleanupData | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  React.useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const r = await fetch("/cleanup-data");
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
  if (err && !d) return <div className="mb-8 text-xs text-rose-200">Clean-up model: couldn't load ({err})</div>;
  if (!d) return null;

  const TOTAL = d.total ?? 30000;
  const v2 = (d.version ?? 1) === 2;
  const last = d.trace.at(-1);
  const speeds = d.trace.map((t) => t[2]).filter((s): s is number => s != null);
  const sps = speeds.length ? speeds.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, speeds.length) : 0;
  const left = last && sps > 0 && !d.finished ? (TOTAL - last[0]) / sps / 60 : null;
  const lastVal = d.vals.at(-1);
  const best = d.vals.length ? d.vals.reduce((a, b) => (prf(b, bestBar(b))!.F1 > prf(a, bestBar(a))!.F1 ? b : a)) : null;
  const b9 = lastVal ? prf(lastVal, "base 0.9") : undefined;
  const b8 = lastVal ? prf(lastVal, "base 0.8") : undefined;
  const dataDone = d.data != null && d.clips != null && d.data >= d.clips - 400;
  const state = d.finished ? "finished" : last ? `training · step ${last[0].toLocaleString()}` :
    d.data2 && (d.data2.clips < d.data2.of - 400 || d.data2.beats < d.data2.beats_of - 400)
      ? `v2 data: base notes ${d.data2.clips.toLocaleString()} / ${d.data2.of.toLocaleString()} training clips, beats ${d.data2.beats.toLocaleString()} / ${d.data2.beats_of.toLocaleString()}` :
    !dataDone ? `building data: ${d.data?.toLocaleString() ?? "?"} / ${d.clips?.toLocaleString() ?? "?"} clips` :
    d.jobs.some((j) => j.name.startsWith("cutrain")) ? "queued" : "not running";
  const pts = (k: string, f: keyof PRF) => d.vals.filter((v) => prf(v, k)).map((v) => [v.step, prf(v, k)![f]] as [number, number]);
  const bestPts = d.vals.map((v) => [v.step, prf(v, bestBar(v))!.F1] as [number, number]);

  return (
    <section className="mb-8">
      <div className="mb-4 rounded-lg border border-primary/50 bg-primary/[0.04] p-3">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <div className="text-sm font-semibold text-white">Clean-up model: training health</div>
          <div className="text-[11px] text-muted-foreground">{v2 ? "cleanup2 (+1,522 training songs, same-key + beat features, learned quantizing)" : "cleanup1 · 4.9M params"}{d.jobs.find((j) => j.name.startsWith("cutrain") && j.state === "RUNNING") ? ` · ${d.jobs.find((j) => j.name.startsWith("cutrain") && j.state === "RUNNING")!.node}` : ""}</div>
          {d.stale && <div className="text-[11px] text-amber-200">cluster didn't answer just now: showing its last reading, from {d.stale}</div>}
        </div>
        <div className="mb-3 mt-0.5 text-[11px] text-muted-foreground">
          The base at the 0.8 onset bar hears the notes you want (voice included) but adds mistakes. This model reads the base's
          0.8 notes of a 20 s clip (pitch, timing, length, loudness, confidence, and the base's probabilities around each note:
          its own key, neighbours, octaves, twelfths) and keeps or drops each one. It never writes a note. Trained on the base's
          output on the 34k validation renders it never trained on; scored on 22 songs held out by song.
          {v2 && " v2 adds 40 clips of each of the 1,522 training songs, the same key's last strike, madmom's beat grid, and a second output: which slot of the beat each note belongs in (beat, 1/4, 1/3, 1/2, 2/3, 3/4) or leave it, learned from POP909's on-grid arrangements instead of snapping to the nearest slot."}
        </div>
        {v2 && <div className="mb-3 text-[11px] text-white">{state}</div>}

        <div className="mb-3 grid grid-cols-2 gap-2 lg:grid-cols-4">
          <Tile label="Progress" value={last ? pct(last[0] / TOTAL) : "—"} hint={last ? `step ${last[0].toLocaleString()} / ${TOTAL.toLocaleString()}` : state} />
          <Tile label="Speed" value={sps ? `${sps.toFixed(1)}/s` : "—"} hint={d.finished ? "finished" : left != null ? `~${Math.round(left)} min left` : "steps per second"} />
          <Tile label="Best F1 (clean-up)" value={best ? f3(prf(best, bestBar(best))!.F1) : "—"}
                hint={best ? `step ${best.step.toLocaleString()}, ${bestBar(best)} · base 0.9: ${b9 ? f3(b9.F1) : "—"}, 0.8: ${b8 ? f3(b8.F1) : "—"}` : "first check at step 2,000"} />
          <Tile label="Latest loss" value={last ? f3(last[1]) : "—"} hint="keep / drop cross-entropy" />
        </div>
        {last && (
          <div className="mb-3 max-w-xl">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full" style={{ width: `${(100 * last[0]) / TOTAL}%`, background: C2 }} />
            </div>
          </div>
        )}
        {d.jobs.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2 text-[11px]">
            {d.jobs.map((j) => (
              <span key={j.id} className="rounded-md border border-border px-2 py-0.5 text-muted-foreground">
                <span className="text-white">{j.name}</span> · {j.state.toLowerCase()} {j.state === "RUNNING" ? `${j.elapsed} / ${j.limit}` : j.reason}
              </span>
            ))}
          </div>
        )}

        {d.vals.length || d.trace.length ? (
          <div className="grid gap-x-6 gap-y-4 md:grid-cols-2 2xl:grid-cols-3">
            <Lines label="note F1 on held-out songs" fmt={f3}
                   series={[{ name: "clean-up (best keep bar)", color: C2, pts: bestPts }, { name: "base 0.9", color: C1, pts: pts("base 0.9", "F1") },
                            { name: "base 0.8", color: C3, pts: pts("base 0.8", "F1") }]} />
            <Lines label="recall (written notes found)" fmt={f3}
                   series={[{ name: "clean-up keep 0.3", color: C2, pts: pts("keep 0.3", "R") }, { name: "keep 0.5", color: C4, pts: pts("keep 0.5", "R") },
                            { name: "base 0.9", color: C1, pts: pts("base 0.9", "R") }, { name: "base 0.8", color: C3, pts: pts("base 0.8", "R") }]} />
            <Lines label="precision (notes kept that are right)" fmt={f3}
                   series={[{ name: "clean-up keep 0.3", color: C2, pts: pts("keep 0.3", "P") }, { name: "keep 0.5", color: C4, pts: pts("keep 0.5", "P") },
                            { name: "base 0.9", color: C1, pts: pts("base 0.9", "P") }, { name: "base 0.8", color: C3, pts: pts("base 0.8", "P") }]} />
            <Lines label="training loss (per 200 steps)" fmt={f3}
                   series={[{ name: "keep / drop", color: C1, pts: d.trace.map((t) => [t[0], t[1]] as [number, number]) },
                            ...(v2 ? [{ name: "beat position", color: C3, pts: d.trace.filter((t) => t[3] != null).map((t) => [t[0], t[3]!] as [number, number]) }] : [])]} />
            {v2 && <Lines label="start error of right notes, median ms" lo={0} fmt={(v) => v.toFixed(1)}
                   series={[{ name: "learned slot", color: C2, pts: d.vals.filter((v) => v.position).map((v) => [v.step, v.position!.err_ms_median.learned] as [number, number]) },
                            { name: "nearest slot (naive)", color: C4, pts: d.vals.filter((v) => v.position).map((v) => [v.step, v.position!.err_ms_median["nearest slot"]] as [number, number]) },
                            { name: "base (unsnapped)", color: C3, pts: d.vals.filter((v) => v.position).map((v) => [v.step, v.position!.err_ms_median.base] as [number, number]) }]} />}
            <Lines label="speed (steps / s)" lo={0} fmt={(v) => v.toFixed(1)}
                   series={[{ name: "steps/s", color: C4, pts: d.trace.filter((t) => t[2] != null).map((t) => [t[0], t[2]!] as [number, number]) }]} />
          </div>
        ) : <div className="text-[11px] text-muted-foreground">{state}</div>}

        {lastVal && (
          <table className="mt-4 w-full max-w-2xl text-left text-xs">
            <thead className="text-muted-foreground"><tr><th className="pb-1 font-normal">latest check (step {lastVal.step.toLocaleString()})</th><th className="font-normal">recall</th><th className="font-normal">precision</th><th className="font-normal">F1</th><th className="font-normal">notes kept</th></tr></thead>
            <tbody>
              {["base 0.8", "base 0.9", ...BARS].filter((k) => prf(lastVal, k)).map((k) => { const v = prf(lastVal, k)!; const base = k.startsWith("base"); return (
                <tr key={k} className="border-t border-border/60 tabular-nums">
                  <td className={base ? "py-1 pr-3 text-muted-foreground" : "py-1 pr-3 text-white"}>{base ? `${k} (no clean-up)` : `0.8 + clean-up, ${k}`}</td>
                  <td>{f3(v.R)}</td><td>{f3(v.P)}</td><td className={k === bestBar(lastVal) ? "font-semibold text-white" : ""}>{f3(v.F1)}</td><td>{pct(v.kept)}</td>
                </tr>
              ); })}
            </tbody>
          </table>
        )}

        {lastVal?.position && (() => { const q = lastVal.position!; return (
          <table className="mt-4 w-full max-w-2xl text-left text-xs">
            <thead className="text-muted-foreground"><tr><th className="pb-1 font-normal">quantizing, right notes on POP909 clips ({q.notes.toLocaleString()}; slot accuracy {pct(q.acc)}, {pct(q.off_share)} best left alone)</th><th className="font-normal">median error</th><th className="font-normal">within 10 ms</th><th className="font-normal">off by 50 ms+</th></tr></thead>
            <tbody>
              {[["base", "as the base places them"], ["nearest slot", "nearest grid slot (naive snap)"], ["learned", "learned slot (v2)"]].map(([k, nm]) => (
                <tr key={k} className="border-t border-border/60 tabular-nums"><td className={k === "learned" ? "py-1 pr-3 text-white" : "py-1 pr-3 text-muted-foreground"}>{nm}</td>
                  <td>{q.err_ms_median[k].toFixed(1)} ms</td><td>{pct(q.within_10ms[k])}</td><td>{(100 * q.worse_than_50ms[k]).toFixed(1)}%</td></tr>
              ))}
            </tbody>
          </table>
        ); })()}

        <table className="mt-4 w-full max-w-2xl text-left text-xs">
          <thead className="text-muted-foreground"><tr><th className="pb-1 font-normal">why: the onset bar alone</th><th className="font-normal">new renders F1 / sung recall</th><th className="font-normal">dense mixes F1</th><th className="font-normal">real songs pc-F1</th><th className="font-normal">real notes/s / sung line</th></tr></thead>
          <tbody>
            {REVIEW.map((r) => (
              <tr key={r[0]} className="border-t border-border/60 tabular-nums"><td className="py-1 pr-3 text-white">bar {r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td></tr>
            ))}
          </tbody>
        </table>
        <div className="mt-1 text-[11px] text-muted-foreground">Lowering the bar brings the voice back (+12 pts of the sung line on real songs) but adds ~4 notes/s; the extra notes are 28-44% right on renders, the rest double presses, octave / overtone ghosts and neighbours. Human arrangements: 8.0 notes/s.</div>
      </div>
    </section>
  );
}
