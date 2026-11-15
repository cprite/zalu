"use client";
import { motion } from "motion/react";
import { SPRING } from "@/lib/motion";
import { useState } from "react";
import { REASONS } from "@/lib/explain";
import type { Failure, ModelEvaluation } from "@/lib/types";

/** Failure Gallery: the actual test frames the model got wrong, with the reason in plain language. */
export function FailureGallery({ e, imageSize }: { e: ModelEvaluation; imageSize?: [number, number] }) {
  const [filter, setFilter] = useState<"all" | Failure["kind"]>("all");
  const items = e.failures.filter((f) => filter === "all" || f.kind === filter);
  const nfn = e.failures.filter((f) => f.kind === "false_negative").length;

  return (
    <section className="card col fade-in" aria-labelledby="gallery-title">
      <div className="row">
        <h3 id="gallery-title">Failure gallery</h3>
        {e.sample_data && <span className="small muted">Demo frames</span>}
        <div className="spacer" />
        <div className="tabs has-pill" role="group" aria-label="Filter failures">
          {([["all", `All (${e.failures.length})`], ["false_negative", `Missed (${nfn})`], ["false_positive", `False alarms (${e.failures.length - nfn})`]] as const).map(([key, text]) => (
            <button key={key} aria-pressed={filter === key} className={filter === key ? "on" : ""} onClick={() => setFilter(key)}>
              {filter === key && <motion.span layoutId="failure-filter" className="tab-pill" transition={SPRING} aria-hidden />}
              <span className="tab-label">{text}</span>
            </button>
          ))}
        </div>
      </div>
      <p className="small ink2">Real test frames the model got wrong, with the likely cause and what to do about it.</p>
      <div className="legend">
        <span><i style={{ borderColor: "var(--color-fn)" }} />solid: object the model missed</span>
        <span><i style={{ borderColor: "var(--color-fp)", borderStyle: "dashed" }} />dashed: the model&apos;s wrong box</span>
      </div>
      {items.length === 0 && <p className="small muted">No failures of this type on the test footage.</p>}
      <ul className="gallery" style={{ listStyle: "none", padding: 0, margin: 0 }}>
        {items.map((f, i) => <li key={i}><FailureTile f={f} size={imageSize} /></li>)}
      </ul>
      {e.failures_total > e.failures.length && (
        <p className="tiny muted">Showing {e.failures.length} of {e.failures_total} failures: misses first, then the most confident false alarms.</p>
      )}
    </section>
  );
}

function FailureTile({ f, size = [1280, 720] }: { f: Failure; size?: [number, number] }) {
  const r = REASONS[f.kind][f.reason];
  const [w, h] = size;
  const miss = f.kind === "false_negative";
  const what = `${miss ? "Missed" : "False alarm"}: ${f.cls}`;
  return (
    <article className="fail">
      <div className="img">
        {f.thumbnail ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={f.thumbnail} alt={`${what} on test frame ${f.image}`} />
        ) : (
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={`${what} (sample, no image)`}>
            <rect width={w} height={h} fill="#1B2333" />
            <text x={20} y={40} fill="#AEB7C6" style={{ font: "600 26px system-ui" }}>no image (sample)</text>
            <Rect b={f.box} c={miss ? "var(--color-fn)" : "var(--color-fp)"} w={w} dashed={!miss} />
            {f.predicted_box && <Rect b={f.predicted_box} c="var(--color-fp)" w={w} dashed />}
          </svg>
        )}
      </div>
      <div className="meta">
        <div className="row" style={{ gap: 8 }}>
          <span className={`badge ${miss ? "warn" : "bad"}`}>{miss ? "▢ Missed" : "✕ False alarm"}</span>
          <b className="small">{f.cls}</b>
          {f.score != null && <span className="tiny muted num">confidence {Math.round(f.score * 100)}%</span>}
        </div>
        <p className="small">{r?.title ?? f.reason}{f.predicted_cls && f.predicted_cls !== f.cls ? ` (said “${f.predicted_cls}”)` : ""}</p>
        <p className="tiny muted">→ {r?.fix}</p>
        <p className="tiny muted mono" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={f.image}>{f.image}</p>
      </div>
    </article>
  );
}

function Rect({ b, c, w, dashed }: { b: number[]; c: string; w: number; dashed?: boolean }) {
  const sw = w / 220;
  return <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none" stroke={c} strokeWidth={sw} strokeDasharray={dashed ? `${sw * 4} ${sw * 3}` : undefined} />;
}
