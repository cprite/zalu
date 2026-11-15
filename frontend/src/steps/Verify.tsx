"use client";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { Comparison } from "@/components/Comparison";
import { FailureGallery } from "@/components/FailureGallery";
import { Composer } from "@/components/Composer";
import { ReportCard } from "@/components/ReportCard";
import { Banner, ErrorBox, SampleBadge } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { makeSampleFrames } from "@/lib/sampleFrames";
import { pct } from "@/lib/explain";
import { Disclosure, EXIT, StatusLine, reveal, t } from "@/lib/motion";
import type { DatasetManifest, EvaluationReport } from "@/lib/types";

export function Verify() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState<"" | "upload" | "eval">("");
  const [selected, setSelected] = useState<string>(() => bestTrained(s.report));
  const [more, setMore] = useState<"" | "compare" | "mistakes">("");
  const [serverSets, setServerSets] = useState<DatasetManifest[]>([]);
  const trainIds = new Set(s.models.map((m) => m.dataset_id).filter(Boolean));
  const trainSessions = new Set([...(s.trainDataset?.source_ids ?? [])]);
  const sameSession = trainSessions.has(s.testSession);

  useEffect(() => {
    if (api.mode === "live") api.listDatasets().then((all) => setServerSets(all.filter((d) => !trainIds.has(d.id) && d.id !== s.trainDataset?.id))).catch(() => setServerSets([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

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

  /** One action: register the new footage as an independent test set (if needed), then evaluate all models. */
  async function runTest() {
    if (!s.task) return;
    setErr(null);
    try {
      let test = s.testDataset;
      if (!test) {
        setBusy("upload");
        test = await api.createDataset(s.task.id, s.testFiles, { session_id: s.testSession, role: "test" });
        set({ testDataset: test, report: null });
      }
      setBusy("eval");
      const report = await api.createEvaluation({ model_ids: s.models.map((m) => m.id), test_dataset_id: test.id });
      set({ report });
      setSelected(bestTrained(report));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="center-col" style={s.report ? { maxWidth: 1040 } : undefined}>
      {!s.report ? (
        <>
          <div className="hero compact">
            <h1>Test it on new footage</h1>
            <p>Record a few minutes the model has never seen, at another time or from another angle. This is the score that counts.</p>
          </div>
          <Composer textless files={s.testFiles} onFiles={(testFiles) => set({ testFiles, testDataset: null })} onSubmit={runTest}
            canSubmit={(s.testFiles.length > 0 || !!s.testDataset) && !sameSession} busy={!!busy} submitLabel="Run the test" />
          <AnimatePresence initial={false}>
            {busy && (
              <motion.p key="busy" className="hint" role="status" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={t("fast")}>
                <StatusLine state="working">{busy === "upload" ? "Uploading test footage…" : "Testing every model on your new footage…"}</StatusLine>
              </motion.p>
            )}
          </AnimatePresence>
          {sameSession && <Banner kind="bad">This footage has the same recording name as the training footage. Use a different recording, otherwise the score is inflated.</Banner>}
          {!busy && (
            <div className="actions-center">
              {api.mode === "mock" && !s.testFiles.length && (
                <button className="toggle" onClick={async () => set({ testFiles: await makeSampleFrames(24, s.testSession) })}>Use sample test frames</button>
              )}
              {api.mode === "live" && serverSets.length > 0 && (
                <label className="small ink2">Or use a test set on the server:{" "}
                  <select className="select" style={{ width: "auto", display: "inline-block", minHeight: 32 }} defaultValue=""
                    onChange={(e) => set({ testDataset: serverSets.find((d) => d.id === e.target.value) ?? null })}>
                    <option value="" disabled>Choose…</option>
                    {serverSets.map((d) => <option key={d.id} value={d.id}>{d.id} · {d.variant}</option>)}
                  </select>
                </label>
              )}
            </div>
          )}
          <ErrorBox error={err} />
          <div className="actions-center"><button className="btn ghost sm" onClick={() => set({ step: 3 })}>← Training</button></div>
        </>
      ) : (
        <>
          <div className="hero compact">
            <h1>Results on new footage</h1>
            {sample && isMockReport(s.report) && <SampleBadge label="Sample results" />}
          </div>
          {leaks.length > 0 && <Banner kind="bad"><div>{leaks.map((n) => <div key={n}>{n}</div>)}</div></Banner>}
          {isMockReport(s.report) ? (
            <MockReport rep={s.report} />
          ) : (
            <>
              {sel && <ReportCard e={sel} onDownload={download} />}
              <div className="actions-center">
                <Disclosure open={more === "compare"} onToggle={() => setMore(more === "compare" ? "" : "compare")} controls="verify-more">Compare models</Disclosure>
                <Disclosure open={more === "mistakes"} onToggle={() => setMore(more === "mistakes" ? "" : "mistakes")} controls="verify-more">See mistakes</Disclosure>
              </div>
              <AnimatePresence mode="wait" initial={false}>
                {more === "compare" && (
                  <motion.div key="compare" id="verify-more" {...reveal} exit={{ ...reveal.exit, transition: EXIT }} style={{ overflow: "hidden" }}><Comparison rep={s.report} selected={selected} onSelect={setSelected} /></motion.div>
                )}
                {more === "mistakes" && sel && (
                  <motion.div key="mistakes" id="verify-more" {...reveal} exit={{ ...reveal.exit, transition: EXIT }} style={{ overflow: "hidden" }}><FailureGallery e={sel} imageSize={s.report.image_size} /></motion.div>
                )}
              </AnimatePresence>
            </>
          )}
          <div className="actions-center">
            <button className="btn ghost" onClick={() => set({ step: 3 })}>← Training</button>
            <button className="btn ghost" onClick={() => set({ report: null, testDataset: null })}>Test other footage</button>
            <button className="btn primary" onClick={() => set({ step: 5, deployModelId: selected })}>Deploy this model →</button>
          </div>
        </>
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
