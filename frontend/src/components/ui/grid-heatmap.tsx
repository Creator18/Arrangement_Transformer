import * as React from "react";
import { Group } from "@visx/group";
import { scaleLinear } from "@visx/scale";
import { HeatmapRect } from "@visx/heatmap";

/** A labelled rows × columns heatmap on visx's HeatmapRect, for real data (heatmaps.tsx is the visx demo
 *  on generated data).  Each cell's colour comes from `color(value)`; null cells are drawn empty. */
export type GridHeatmapProps = {
  rows: string[];
  columns: string[];                       // column labels (only every `labelEvery`-th is drawn)
  value: (row: number, col: number) => number | null;
  color: (v: number, col: number) => string;
  title?: (row: number, col: number, v: number | null) => string;
  cellHeight?: number;
  labelWidth?: number;
  labelEvery?: number;
  gap?: number;
  emptyColor?: string;
};

type Cell = { row: number; col: number; v: number | null };
type Column = { bins: Cell[] };

function useWidth<T extends HTMLElement>() {
  const ref = React.useRef<T>(null);
  const [width, setWidth] = React.useState(0);
  React.useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function GridHeatmap({
  rows, columns, value, color, title,
  cellHeight = 14, labelWidth = 92, labelEvery = 1, gap = 1, emptyColor = "rgba(255,255,255,0.04)",
}: GridHeatmapProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const axis = 16;
  const plotW = Math.max(0, width - labelWidth);
  const binWidth = columns.length ? plotW / columns.length : 0;
  const height = rows.length * cellHeight + axis;

  const data: Column[] = React.useMemo(
    () => columns.map((_, col) => ({ bins: rows.map((_, row) => ({ row, col, v: value(row, col) })) })),
    [rows, columns, value],
  );
  const xScale = scaleLinear<number>({ domain: [0, columns.length], range: [0, plotW] });
  const yScale = scaleLinear<number>({ domain: [0, rows.length], range: [0, rows.length * cellHeight] });

  return (
    <div ref={ref} className="w-full">
      {width > 0 && (
        <svg width={width} height={height} role="img">
          {rows.map((r, i) => (
            <text key={r} x={labelWidth - 8} y={i * cellHeight + cellHeight / 2} dy="0.35em" textAnchor="end"
                  fontSize={10.5} fill="currentColor" className="text-muted-foreground">{r}</text>
          ))}
          <Group left={labelWidth}>
            <HeatmapRect<Column, Cell>
              data={data}
              bins={(d) => d.bins}
              count={(d) => d.v ?? 0}
              xScale={(i) => xScale(i) ?? 0}
              yScale={(i) => yScale(i) ?? 0}
              binWidth={binWidth}
              binHeight={cellHeight}
              gap={gap}
            >
              {(heatmap) =>
                heatmap.map((col) =>
                  col.map((bin) => {
                    const c = bin.bin as Cell;
                    return (
                      <rect key={`${c.row}-${c.col}`} x={bin.x} y={c.row * cellHeight + gap / 2}
                            width={Math.max(0.5, bin.width)} height={cellHeight - gap} rx={Math.min(2, binWidth / 4)}
                            fill={c.v === null ? emptyColor : color(c.v, c.col)}>
                        {title && <title>{title(c.row, c.col, c.v)}</title>}
                      </rect>
                    );
                  }),
                )
              }
            </HeatmapRect>
            {columns.map((label, i) =>
              i % labelEvery === 0 ? (
                <text key={i} x={i * binWidth + binWidth / 2} y={rows.length * cellHeight + 11} textAnchor="middle"
                      fontSize={9.5} fill="currentColor" className="text-muted-foreground">{label}</text>
              ) : null,
            )}
          </Group>
        </svg>
      )}
    </div>
  );
}
