"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { AnimatedNumber, Dots, t } from "@/lib/motion";
import type { DashboardSpec, Widget } from "../../../shared/dashboard_spec/dashboard-spec";
import { readMetric, type FrameSample } from "./session";
import "./dashboard.css";

type Props = {
  spec: DashboardSpec;
  classes: readonly string[];
  samples: readonly FrameSample[];
  preview: string | null;
  loading: boolean;
  error: string | null;
  sampleMode: boolean;
};

const label = (value: string) => value.replaceAll("_", " ");
const format = (value: number) => new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);

/** Empty / waiting state: dots while inference is genuinely running, otherwise a quiet placeholder icon. */
function Empty({ waiting = false, children }: { waiting?: boolean; children: string }) {
  return (
    <div className="dash-empty" role="status">
      {waiting ? <Dots /> : (
        <svg className="dash-empty-ic" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3 3" aria-hidden>
          <rect x="3.5" y="5.5" width="17" height="13" rx="3" />
        </svg>
      )}
      <span>{waiting ? "Waiting for inference…" : children}</span>
    </div>
  );
}

function Trend({ points, windowSeconds, now }: { points: { at: number; value: number | null }[]; windowSeconds: number; now: number }) {
  const visible = points.filter((point) => point.at >= now - windowSeconds * 1000);
  const values = visible.flatMap((point) => point.value === null ? [] : [point.value]);
  if (!values.length) return <Empty>No readings in this window.</Empty>;
  const max = Math.max(1, ...values);
  const marks = visible.map(({ at, value }) => value === null ? null : {
    x: 12 + 336 * (1 - Math.max(0, now - at) / (windowSeconds * 1000)),
    y: 86 - (70 * value) / max,
  });
  const lastMark = [...marks].reverse().find((mark) => mark !== null) ?? null;
  const lines = marks.flatMap((mark, i) => mark && marks[i - 1] ? [[marks[i - 1]!, mark]] : []);
  return (
    <figure className="dash-trend">
      <svg viewBox="0 0 360 100" role="img" aria-label={`Metric readings over the last ${windowSeconds} seconds`}>
        {[16, 51, 86].map((y) => <line key={y} x1="12" x2="348" y1={y} y2={y} stroke="var(--color-line)" />)}
        {lines.map(([a, b], i) => <line key={i} x1={a.x} x2={b.x} y1={a.y} y2={b.y} stroke="var(--viz-1, var(--color-brand))" strokeWidth="2" strokeLinecap="round" />)}
        {lastMark && <motion.circle initial={false} animate={{ cx: lastMark.x, cy: lastMark.y }} transition={t("fast")} r="4" fill="var(--viz-1, var(--color-brand))" stroke="var(--color-surface)" strokeWidth="2" />}
      </svg>
      <figcaption>Last {windowSeconds}s <span>Range 0–{format(max)}</span></figcaption>
    </figure>
  );
}

function Content({ widget, spec, classes, samples, preview, loading, sampleMode, now }: Props & { widget: Widget; now: number }) {
  const latest = samples.at(-1);
  const result = latest?.result;
  if (widget.type === "live_preview") {
    return preview || (sampleMode && result) ? (
      <div className="dash-preview" style={{ aspectRatio: result ? result.width / result.height : 16 / 9 }}>
        {/* eslint-disable-next-line @next/next/no-img-element -- blob: frame preview, next/image cannot optimise it */}
        {preview && <img src={preview} alt="Frame submitted for inference" />}
        {sampleMode && !preview && <span className="dash-sample-frame">Simulated frame</span>}
        {widget.config.show_boxes && result && (
          <svg viewBox={`0 0 ${result.width} ${result.height}`} aria-label="Detection boxes" role="img">
            {result.detections.map(({ box: [x, y, x2, y2], cls }, i) => (
              <g key={i}>
                <rect x={x} y={y} width={x2 - x} height={y2 - y} fill="none" stroke="#8fb0ff" strokeWidth="3" />
                <text x={x + 4} y={Math.max(18, y - 6)} fill="white" stroke="#0e1727" paintOrder="stroke" strokeWidth="4" fontSize={result.width / 40}>{label(cls)}</text>
              </g>
            ))}
          </svg>
        )}
      </div>
    ) : <Empty waiting={loading}>Start the camera or choose an image.</Empty>;
  }
  if (widget.type === "kpi" || widget.type === "time_series") {
    const id = widget.config.metric_id;
    const metric = spec.metrics.find((item) => item.id === id && item.source_id === widget.source_id);
    if (!metric) return <Empty>Metric unavailable.</Empty>;
    if (widget.type === "time_series") {
      return <Trend now={now} windowSeconds={widget.config.window_seconds} points={samples.map((sample) => ({
        at: sample.at, value: readMetric(metric, sample.result, classes).value,
      }))} />;
    }
    const reading = readMetric(metric, result, classes);
    return reading.status === "available"
      ? <div className="dash-kpi"><AnimatedNumber value={reading.value} format={format} duration={0.15} />{metric.operation === "inference_latency_ms" && <small> ms</small>}</div>
      : <Empty waiting={loading}>Metric unavailable.</Empty>;
  }
  if (!result) return <Empty waiting={loading}>No inference results yet.</Empty>;
  if (widget.type === "detection_table") {
    if (widget.config.rows === "events") {
      if (!latest.eventsEnabled) return <Empty>Event detection is off.</Empty>;
      const events = result.events.slice(0, widget.config.max_rows);
      return events.length ? <ul className="dash-events">{events.map((event, i) => <li key={i}>{label(event)}</li>)}</ul>
        : <Empty>No events in the latest frame.</Empty>;
    }
    const rows = result.detections.slice(0, widget.config.max_rows);
    return rows.length ? (
      <div className="dash-table"><table><thead><tr><th>Class</th><th>Confidence</th></tr></thead><tbody>
        {rows.map((detection, i) => <tr key={i}><td>{label(detection.cls)}</td><td>{format(100 * detection.score)}%</td></tr>)}
      </tbody></table></div>
    ) : <Empty>No detections in the latest frame.</Empty>;
  }
  const counts = new Map<string, number>();
  for (const { cls } of result.detections) counts.set(cls, (counts.get(cls) ?? 0) + 1);
  if (!counts.size) return <Empty>No classes detected in the latest frame.</Empty>;
  return (
    <ul className="dash-bars">{[...counts].sort(([a], [b]) => a.localeCompare(b)).map(([cls, count]) => (
      <li key={cls}><span>{label(cls)}</span><meter min={0} max={result.detections.length} value={count} aria-label={label(cls)} /><strong>{count}</strong></li>
    ))}</ul>
  );
}

export function DashboardRenderer(props: Props) {
  const { spec, samples, loading, error, sampleMode } = props;
  const [now, setNow] = useState(0);
  const hasSeries = spec.widgets.some((widget) => widget.type === "time_series");
  useEffect(() => {
    if (!hasSeries) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [hasSeries]);
  return (
    <section className="dash" aria-label={spec.title}>
      <header className="dash-heading">
        <div><h1>{spec.title}</h1><p>{spec.description}</p></div>
        <span className={`dash-status ${sampleMode ? "sample" : ""} ${samples.length && !error ? "on" : ""}`}>
          <span className="dash-dot" aria-hidden />{sampleMode ? "Sample data · simulated" : "Live inference"}
        </span>
      </header>
      {error && <p className="dash-error" role="alert">{error}</p>}
      <AnimatePresence initial={false}>
        {loading && !samples.length && (
          <motion.p key="pending" className="dash-pending" role="status" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={t("fast")}>
            Running inference…
          </motion.p>
        )}
      </AnimatePresence>
      <div className="dash-grid" style={{ gridTemplateColumns: `repeat(${spec.layout.columns}, minmax(0, 1fr))` }}>
        <AnimatePresence initial={false}>
          {spec.widgets.map((widget) => (
            <motion.section key={widget.id} layout="position" className="dash-card" style={{ gridColumn: `span ${widget.span}` }} data-widget={widget.type}
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.98 }} transition={t("base")}>
              <h2>{widget.title}</h2><Content {...props} widget={widget} now={Math.max(now, samples.at(-1)?.at ?? 0)} />
            </motion.section>
          ))}
        </AnimatePresence>
      </div>
    </section>
  );
}
