"use client";
import { useEffect, useState } from "react";
import { Comparison } from "@/components/Comparison";
import { FailureGallery } from "@/components/FailureGallery";
import { MediaInput } from "@/components/MediaInput";
import { ReportCard } from "@/components/ReportCard";
import { Banner, ErrorBox, SectionHead, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { makeSampleFrames } from "@/lib/sampleFrames";
import { pct } from "@/lib/explain";
import type { DatasetManifest, EvaluationReport } from "@/lib/types";

export function Verify() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState<"" | "upload" | "eval">("");
  const [selected, setSelected] = useState<string>(() => bestTrained(s.report));
  const [serverSets, setServerSets] = useState<DatasetManifest[]>([]);
  const trainIds = new Set(s.models.map((m) => m.dataset_id).filter(Boolean));
  const trainSessions = new Set([...(s.trainDataset?.source_ids ?? [])]);
  const sameSession = trainSessions.has(s.testSession);

  useEffect(() => {
    if (api.mode === "live") api.listDatasets().then((all) => setServerSets(all.filter((d) => !trainIds.has(d.id) && d.id !== s.trainDataset?.id))).catch(() => setServerSets([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  async function uploadTest() {
    if (!s.task) return;
    setBusy("upload");
    setErr(null);
    try {
      const testDataset = await api.createDataset(s.task.id, s.testFiles, { session_id: s.testSession, role: "test" });
      set({ testDataset, report: null });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy("");
    }
  }

  async function evaluate() {
    if (!s.testDataset) return;
    setBusy("eval");
    setErr(null);
    try {
      const report = await api.createEvaluation({ model_ids: s.models.map((m) => m.id), test_dataset_id: s.testDataset.id });
      set({ report });
      setSelected(bestTrained(report));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy("");
    }
  }

  const download = () => {
    const b = new Blob([JSON.stringify(s.report, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `${s.report!.id}.json`;
    a.click();
  };

  const sel = s.report?.per_model[selected];
  const sample = !!s.report && Object.values(s.report.per_model).some((e) => e.sample_data);
  const leaks = s.report?.notes.filter((n) => n.startsWith("LEAKAGE")) ?? [];

  return (
    <div className="fade-in">
      <SectionHead title="Verify on footage the model has never seen"
        sub="Record a few minutes at another time of day or from another angle. This independent test is the score that counts." />
      <div className="split">
        <div className="card col">
          <label className="field">Test recording name
            <input className="input" value={s.testSession} aria-invalid={sameSession} aria-describedby={sameSession ? "leak-msg" : undefined}
              onChange={(e) => set({ testSession: e.target.value.replace(/\s+/g, "_"), testDataset: null })} />
          </label>
          {sameSession && <Banner kind="bad" id="leak-msg">This name matches a training recording. Test footage must come from a <b>different</b> session, otherwise the score is inflated.</Banner>}
          <MediaInput files={s.testFiles} onFiles={(testFiles) => set({ testFiles, testDataset: null })} sessionId={s.testSession} />
          {api.mode === "mock" && !s.testFiles.length && (
            <button className="btn sm self-start" onClick={async () => set({ testFiles: await makeSampleFrames(24, s.testSession) })}>Use generated sample test frames</button>
          )}
          {api.mode === "live" && serverSets.length > 0 && (
            <label className="field">Or choose a test dataset on the server
              <select className="select" defaultValue="" onChange={(e) => set({ testDataset: serverSets.find((d) => d.id === e.target.value) ?? null })}>
                <option value="" disabled>Choose…</option>
                {serverSets.map((d) => <option key={d.id} value={d.id}>{d.id} · {d.variant} · sessions {d.source_ids.join(", ") || "unknown"}</option>)}
              </select>
            </label>
          )}
          <ErrorBox error={err} />
          <div className="row">
            <button className="btn" onClick={() => set({ step: 3 })}>← Back</button>
            <div className="spacer" />
            {!s.testDataset ? (
              <button className="btn primary" disabled={!s.testFiles.length || sameSession || busy === "upload"} onClick={uploadTest}>
                {busy === "upload" && <Spinner />}Use as test footage
              </button>
            ) : (
              <button className="btn primary" disabled={busy === "eval"} onClick={evaluate}>
                {busy === "eval" && <Spinner />}{busy === "eval" ? "Testing all models…" : s.report ? "Run the test again" : "Run independent test"}
              </button>
            )}
          </div>
          {s.testDataset && <p className="tiny muted">✓ Test footage ready: {s.testDataset.id} (recording {s.testDataset.source_ids.join(", ")})</p>}
        </div>
        <aside className="col small ink2" style={{ padding: "8px 4px" }}>
          <h4 className="text-ink">Why a separate recording?</h4>
          <p>Frames from the same video look almost identical. Testing on them is like grading an exam with the answers the student memorised.</p>
          <ul className="col" style={{ gap: 4, paddingLeft: 18, margin: 0 }}>
            <li>Synthetic images are never used for testing.</li>
            <li>The confidence threshold is fixed before the test.</li>
            <li>Run the test once per model version; don&apos;t tune on it.</li>
          </ul>
        </aside>
      </div>

      {s.report && (
        <div className="col" style={{ gap: 16, marginTop: 24 }}>
          {leaks.length > 0 && <Banner kind="bad"><div>{leaks.map((n) => <div key={n}>{n}</div>)}</div></Banner>}
          {sample && <Banner kind="sample"><span><b>Sample results.</b> Produced by the real evaluation engine on simulated predictions. They are not measurements of your model.</span></Banner>}
          {isMockReport(s.report) ? (
            <MockReport rep={s.report} />
          ) : (
            <>
              <Comparison rep={s.report} selected={selected} onSelect={setSelected} />
              {sel && <ReportCard e={sel} onDownload={download} />}
              {sel && <FailureGallery e={sel} imageSize={s.report.image_size} />}
            </>
          )}
          <div className="row">
            <div className="spacer" />
            <button className="btn primary" onClick={() => set({ step: 5, deployModelId: selected })}>Deploy this model →</button>
          </div>
        </div>
      )}
    </div>
  );
}

/** The backend answers with mock=true (val-split metrics only) when the Evaluation Engine is not installed. */
function isMockReport(r: EvaluationReport) {
  return !!r.mock || Object.values(r.per_model).some((e) => !(e as { overall?: unknown }).overall);
}

function MockReport({ rep }: { rep: EvaluationReport }) {
  return (
    <section className="card col">
      <Banner kind="warn"><b>Not an independent test.</b> The backend has no Evaluation Engine yet, so it returned the training run&apos;s validation metrics. Do not use these numbers to decide on deployment.</Banner>
      <div className="tbl-wrap"><table className="tbl">
        <thead><tr><th>Model</th><th>Validation recall</th><th>Validation precision</th><th>mAP50</th></tr></thead>
        <tbody>
          {Object.entries(rep.per_model).map(([id, e]) => {
            const v = (e as unknown as { val_metrics?: Record<string, number> }).val_metrics ?? {};
            return <tr key={id}><td className="mono small">{id}</td><td>{pct(v.recall)}</td><td>{pct(v.precision)}</td><td>{pct(v.map50)}</td></tr>;
          })}
        </tbody>
      </table></div>
      {rep.notes.map((n) => <p key={n} className="tiny muted">{n}</p>)}
    </section>
  );
}

function bestTrained(report: import("@/lib/types").EvaluationReport | null): string {
  if (!report) return "";
  const best = Object.entries(report.per_model).filter(([, e]) => e.model?.kind === "trained").sort((a, b) => (b[1].overall.f1 ?? 0) - (a[1].overall.f1 ?? 0))[0];
  return best?.[0] ?? Object.keys(report.per_model)[0];
}
