"use client";
import { VERDICT, pct, variantLabel } from "@/lib/explain";
import type { EvaluationReport } from "@/lib/types";

/** Baseline vs fine-tuned, and the synthetic A/B, on the same independent test footage. */
export function Comparison({ rep, selected, onSelect }: { rep: EvaluationReport; selected: string; onSelect: (id: string) => void }) {
  const entries = Object.entries(rep.per_model);
  const sample = entries.some(([, e]) => e.sample_data);
  const real = entries.find(([, e]) => e.model?.kind === "trained" && e.model.variant === "real_only");
  const syn = entries.find(([, e]) => e.model?.kind === "trained" && e.model.variant === "real_plus_synthetic");
  // ratio deltas are shown in percentage points, count deltas as plain numbers
  const d = (v: number | null | undefined, count = false) => {
    if (v == null) return null;
    const good = count ? v < 0 : v > 0;
    const txt = count ? String(v) : `${(v * 100).toFixed(0)} pts`;
    return <span className={`tiny ${v === 0 ? "muted" : good ? "pos" : "negv"}`}> {v > 0 ? "+" : ""}{txt}</span>;
  };

  return (
    <div className="card col fade-in">
      <div className="row"><h2>Model comparison</h2>{sample && <span className="small muted">Demo results</span>}</div>
      <div className="tbl-wrap"><table className="tbl">
        <caption className="sr-only">Models compared on the same independent test footage. Select a model to open its report card.</caption>
        <thead><tr><th>Model</th><th>Recall</th><th>Precision</th><th>F1</th><th>False alarms</th><th>Missed</th><th>Speed</th><th>Verdict</th></tr></thead>
        <tbody>
          {entries.map(([id, e]) => {
            const dv = e.delta_vs_baseline;
            return (
              <tr key={id} className={id === selected ? "sel" : ""} onClick={() => onSelect(id)} style={{ cursor: "pointer" }}>
                <td>
                  <button className="text-left font-semibold small cursor-pointer bg-transparent border-0 p-0 text-ink underline-offset-2 hover:underline" aria-pressed={id === selected}
                    onClick={(ev) => { ev.stopPropagation(); onSelect(id); }}>
                    {variantLabel(e.model?.kind, e.model?.variant)}{id === selected && <span className="sr-only"> (report shown below)</span>}
                  </button>
                  <div className="tiny muted mono">{id}</div>
                </td>
                <td className="num">{pct(e.overall.recall)}{d(dv?.recall)}</td>
                <td className="num">{pct(e.overall.precision)}{d(dv?.precision)}</td>
                <td className="num">{pct(e.overall.f1)}{d(dv?.f1)}</td>
                <td className="num">{e.overall.fp}{d(dv?.fp, true)}</td>
                <td className="num">{e.overall.fn}{d(dv?.fn, true)}</td>
                <td className="num">{e.performance?.fps ? `${e.performance.fps} FPS` : "—"}</td>
                <td><span className={`badge ${e.readiness.verdict === "ready" ? "good" : e.readiness.verdict === "insufficient_data" ? "warn" : "bad"}`}>{VERDICT[e.readiness.verdict]?.icon} {VERDICT[e.readiness.verdict]?.label}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table></div>
      <div className="tiny muted">Changes are against the zero-shot baseline. A change is called clear only when the 95% ranges do not overlap. Select a model to open its report card.</div>
      {real && syn && <SyntheticAB real={real[1].overall} syn={syn[1].overall} />}
    </div>
  );
}

function SyntheticAB({ real, syn }: { real: { f1: number | null; recall_ci95: [number, number] | null }; syn: { f1: number | null; recall_ci95: [number, number] | null } }) {
  if (real.f1 == null || syn.f1 == null) return null;
  const helped = syn.f1 > real.f1;
  const sep = real.recall_ci95 && syn.recall_ci95 && (syn.recall_ci95[0] > real.recall_ci95[1] || real.recall_ci95[0] > syn.recall_ci95[1]);
  const kind = helped && sep ? "good" : helped ? "info" : "warn";
  const msg = helped && sep ? "Synthetic data clearly improved results on real footage."
    : helped ? "Synthetic data gave a small gain on real footage, within measurement noise."
    : "Synthetic data did NOT improve results on real footage. The real-only model is the safer choice.";
  return (
    <div className={`banner ${kind}`}>
      <div className="col" style={{ gap: 4 }}>
        <b>Synthetic A/B test</b>
        <span className="small">{msg} F1: real-only {pct(real.f1, 1)} vs real + synthetic {pct(syn.f1, 1)}.</span>
      </div>
    </div>
  );
}
