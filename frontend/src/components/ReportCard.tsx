"use client";
import { METRIC_HELP, REASONS, VERDICT, pct, variantLabel } from "@/lib/explain";
import type { Check, ModelEvaluation } from "@/lib/types";
import { CiBar, Kpi, SampleBadge } from "./ui";

/** Model Report Card: everything needed to decide "deploy or not", with limitations stated next to the numbers. */
export function ReportCard({ e, onDownload }: { e: ModelEvaluation; onDownload?: () => void }) {
  const o = e.overall;
  const r = e.readiness;
  const target = (n: string) => r.criteria.find((c) => c.name.startsWith(n))?.target;
  const v = VERDICT[r.verdict] ?? VERDICT.not_ready;
  const ci = (c: [number, number] | null) => (c ? `95% range ${pct(c[0])}–${pct(c[1])}` : "not measurable");

  return (
    <div className="card col fade-in" style={{ gap: 18 }}>
      <div className="row">
        <div className="col" style={{ gap: 2 }}>
          <div className="row"><h3>Model report card</h3><SampleBadge show={e.sample_data} /></div>
          <div className="small ink2">{variantLabel(e.model?.kind, e.model?.variant)} · base <code>{e.model?.base_model}</code></div>
          <div className="tiny muted mono">version {e.model?.id} · tested on {e.test.dataset_id} ({e.test.split ?? "test"} split)</div>
        </div>
        <div className="spacer" />
        {onDownload && <button className="btn sm" onClick={onDownload}>Download JSON</button>}
      </div>

      <div className={`verdict ${r.verdict}`}>
        <div className="icon">{v.icon}</div>
        <div className="col" style={{ gap: 2 }}>
          <b>{v.label}</b>
          <span className="small">{r.summary}</span>
        </div>
      </div>

      <div className="kpis">
        <Kpi k="Recall · objects found" v={pct(o.recall)} ci={ci(o.recall_ci95)} help={METRIC_HELP.recall} />
        <Kpi k="Precision · alerts that are real" v={pct(o.precision)} ci={ci(o.precision_ci95)} help={METRIC_HELP.precision} />
        <Kpi k="F1 · balance" v={pct(o.f1)} help={METRIC_HELP.f1} />
        <Kpi k="Speed" v={e.performance?.fps ? `${e.performance.fps} FPS` : "—"} ci={e.performance ? `${e.performance.latency_ms_median} ms per frame` : "not measured"} />
      </div>

      <div className="grid2">
        <div className="col">
          <h4>Against your acceptance criteria</h4>
          {[["Recall", o.recall, o.recall_ci95], ["Precision", o.precision, o.precision_ci95]].map(([n, val, c]) => (
            <div key={n as string} className="col" style={{ gap: 2 }}>
              <div className="row small"><span>{n as string}</span><div className="spacer" /><span className="num">{pct(val as number)} <span className="muted">target {pct(target(n as string))}</span></span></div>
              <CiBar value={val as number} ci={c as [number, number]} target={target(n as string)} />
            </div>
          ))}
          <div className="tiny muted">Dot = measured value · shaded band = 95% range · line = target</div>
          <Checklist items={[...r.criteria, ...r.data_criteria]} />
        </div>
        <div className="col">
          <h4>Errors on the test footage</h4>
          <div className="grid2">
            <Kpi k="False alarms" v={o.fp} help={METRIC_HELP.fp} />
            <Kpi k="Missed objects" v={o.fn} help={METRIC_HELP.fn} />
          </div>
          <ReasonList title="Why it raised false alarms" kind="false_positive" counts={e.failure_counts.false_positive} />
          <ReasonList title="Why it missed objects" kind="false_negative" counts={e.failure_counts.false_negative} />
        </div>
      </div>

      <div className="grid2">
        <div className="col">
          <h4>Per class</h4>
          <table className="tbl">
            <thead><tr><th>Class</th><th>Test objects</th><th>Recall</th><th>Precision</th><th>Frames flagged correctly</th></tr></thead>
            <tbody>
              {Object.entries(e.per_class).map(([c, m]) => (
                <tr key={c}><td>{c}</td><td className="num">{e.test.gt_per_class[c]}</td><td className="num">{pct(m.recall)}</td><td className="num">{pct(m.precision)}</td>
                  <td className="num">{pct(m.image_level.recall)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="col">
          <h4>Test data and limitations</h4>
          <div className="small ink2">
            {e.test.n_images} independent frames ({e.test.n_images_without_objects} with nothing to detect) from {e.test.n_sessions ?? "an unknown number of"} recording session(s) never used for training.
            Box counted correct at overlap ≥ {e.settings.iou_threshold}, confidence ≥ {e.settings.conf_threshold}.
          </div>
          {e.warnings.map((w) => <div key={w} className="small" style={{ color: "#6b4000" }}>⚠ {w}</div>)}
          <div className="small muted">A small test set shows how the model behaves here; it does not prove production safety. Keep monitoring after deployment.</div>
          {e.performance && <div className="tiny muted">Speed: {e.performance.note}; median of {e.performance.n} frames, p95 {e.performance.latency_ms_p95} ms.</div>}
          {e.model?.val_metrics && Object.keys(e.model.val_metrics).length > 0 && (
            <div className="tiny muted">Validation during training (used for model selection only, not the score above): recall {pct(e.model.val_metrics.recall)}, precision {pct(e.model.val_metrics.precision)}.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Checklist({ items }: { items: Check[] }) {
  return (
    <div className="col" style={{ gap: 4 }}>
      {items.map((c) => (
        <div key={c.name} className="row small" style={{ gap: 8 }}>
          <span className={c.passed ? "pos" : "negv"}>{c.passed ? "✓" : "✕"}</span>
          <span>{c.name}</span><div className="spacer" />
          <span className="num muted">{fmt(c.value)} {c.op} {fmt(c.target)}</span>
        </div>
      ))}
    </div>
  );
}

const fmt = (v: number | null) => (v == null ? "—" : v <= 1 && !Number.isInteger(v) ? pct(v) : String(v));

function ReasonList({ title, kind, counts }: { title: string; kind: "false_positive" | "false_negative"; counts: Record<string, number> }) {
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!rows.length) return <div className="small muted">{title}: none 🎉</div>;
  return (
    <div className="col" style={{ gap: 6 }}>
      <div className="small" style={{ fontWeight: 600 }}>{title}</div>
      {rows.map(([k, n]) => (
        <div key={k} className="small">
          <div className="row"><span>{REASONS[kind][k]?.title ?? k}</span><div className="spacer" /><b className="num">{n}</b></div>
          <div className="tiny muted">→ {REASONS[kind][k]?.fix}</div>
        </div>
      ))}
    </div>
  );
}
