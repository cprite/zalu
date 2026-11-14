"use client";
import { useEffect, useState } from "react";
import { ErrorBox, Progress, SectionHead, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { pct, variantLabel } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import type { DatasetManifest } from "@/lib/types";

const PRESETS = [
  { label: "Quick", epochs: 10, note: "about 1 min" },
  { label: "Standard", epochs: 30, note: "about 3 min" },
  { label: "Thorough", epochs: 60, note: "about 6 min" },
];

export function Train() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [datasets, setDatasets] = useState<DatasetManifest[]>([]);
  const [pick, setPick] = useState<Record<string, boolean>>({});
  const [zeroShot, setZeroShot] = useState(true);
  const [epochs, setEpochs] = useState(30);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const ds = s.trainDataset!;
  const classes = s.task?.classes ?? ds.classes;

  useEffect(() => {
    api.listDatasets().then((all) => {
      // variants of this upload; independent TEST datasets are never offered for training
      const mine = all.filter((d) => d.variant !== "test" &&
        (d.id === ds.id || (ds.task_id && d.task_id === ds.task_id && d.source_ids.join() === ds.source_ids.join())));
      const list = mine.length ? mine : [ds];
      setDatasets(list);
      setPick(Object.fromEntries(list.map((d) => [d.id, true])));
    }).catch(() => { setDatasets([ds]); setPick({ [ds.id]: true }); });
  }, [api, ds]);

  const running = s.jobs.some((j) => j.status === "queued" || j.status === "running");
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      const jobs = await Promise.all(s.jobs.map((j) => (j.status === "succeeded" || j.status === "failed" ? j : api.getJob(j.id).catch(() => j))));
      const newModels = await Promise.all(jobs.filter((j) => j.status === "succeeded" && j.result_id && !s.models.some((m) => m.id === j.result_id))
        .map((j) => api.getModel(j.result_id!)));
      set((p) => ({ jobs, models: [...p.models, ...newModels.filter((m) => !p.models.some((x) => x.id === m.id))] }));
    }, 1500);
    return () => clearInterval(t);
  }, [running, s.jobs, s.models, api, set]);

  async function train() {
    setBusy(true);
    setErr(null);
    try {
      const jobs = await Promise.all(datasets.filter((d) => pick[d.id]).map((d) => api.startTraining({ dataset_id: d.id, epochs, base_model: "yolo11n.pt" })));
      const zs = zeroShot && !s.models.some((m) => m.kind === "zero_shot") ? [await api.createZeroShot(classes)] : [];
      set((p) => ({ jobs: [...p.jobs, ...jobs], models: [...p.models, ...zs], report: null }));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  const done = s.models.some((m) => m.kind === "trained") && !running;
  const nothingPicked = !zeroShot && !datasets.some((d) => pick[d.id]);

  return (
    <div className="fade-in">
      <SectionHead title="Train a small, fast model" sub="We fine-tune a compact detector on your labelled frames. It runs on a GPU and takes a few minutes." />
      <div className="card col" style={{ gap: 24 }}>
        <div className="grid2">
          <fieldset className="col" style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="text-[15px] font-semibold" style={{ marginBottom: 8 }}>What to train</legend>
            {datasets.map((d) => (
              <label key={d.id} className="flex gap-2 items-start small">
                <input type="checkbox" checked={!!pick[d.id]} onChange={(e) => setPick({ ...pick, [d.id]: e.target.checked })} />
                <span><b>{variantLabel("trained", d.variant)}</b> <span className="muted">· {Object.entries(d.counts).map(([k, v]) => `${v} ${k}`).join(", ")}</span></span>
              </label>
            ))}
            <label className="flex gap-2 items-start small">
              <input type="checkbox" checked={zeroShot} onChange={(e) => setZeroShot(e.target.checked)} />
              <span><b>Zero-shot baseline</b> <span className="muted">· the large open-vocabulary model without training, to compare against</span></span>
            </label>
            {datasets.length > 1 && <p className="tiny muted">Training both variants lets ZALU check honestly whether synthetic examples help.</p>}
          </fieldset>
          <div className="col">
            <h4 id="time-label">Training time</h4>
            <div className="tabs" role="radiogroup" aria-labelledby="time-label">
              {PRESETS.map((p) => (
                <button key={p.epochs} role="radio" aria-checked={epochs === p.epochs} className={epochs === p.epochs ? "on" : ""} onClick={() => setEpochs(p.epochs)}>
                  {p.label} <span className="muted">{p.note}</span>
                </button>
              ))}
            </div>
            <details className="small ink2">
              <summary className="cursor-pointer">Advanced settings</summary>
              <div className="col" style={{ gap: 4, marginTop: 8 }}>
                <span>Base model: <code>yolo11n.pt</code> (compact, exported to ONNX)</span>
                <span>Epochs: {epochs} · image size 640 · batch 16</span>
              </div>
            </details>
          </div>
        </div>
        <ErrorBox error={err} />
        <div className="row">
          <button className="btn" onClick={() => set({ step: 2 })}>← Back</button>
          <div className="spacer" />
          {done ? (
            <>
              <button className="btn" disabled={busy || nothingPicked} onClick={train}>Train again</button>
              <button className="btn primary" onClick={() => set({ step: 4 })}>Verify on new footage →</button>
            </>
          ) : (
            <button className="btn primary" disabled={busy || running || nothingPicked} onClick={train}>
              {(busy || running) && <Spinner />}{running ? "Training…" : "Start training"}
            </button>
          )}
        </div>
      </div>

      {(s.jobs.length > 0 || s.models.length > 0) && (
        <section className="card col" style={{ marginTop: 16 }} aria-live="polite">
          <h3>Progress</h3>
          {s.jobs.map((j) => {
            const d = datasets.find((x) => x.id === j.request.dataset_id);
            const m = s.models.find((x) => x.id === j.result_id);
            const icon = j.status === "succeeded" ? "✓" : j.status === "failed" ? "✕" : "…";
            return (
              <div key={j.id} className="col" style={{ gap: 4 }}>
                <div className="row small">
                  <b>{variantLabel("trained", d?.variant)}</b>
                  <span className={`badge ${j.status === "succeeded" ? "good" : j.status === "failed" ? "bad" : "exp"}`}>{icon} {j.status}</span>
                  <span className="muted">{j.message.split("\n")[0]}</span>
                  <div className="spacer" />
                  {m && <span className="tiny muted">trained in {m.train_seconds}s · validation recall {pct(m.metrics.recall)}, precision {pct(m.metrics.precision)}</span>}
                </div>
                {j.status !== "failed"
                  ? <Progress value={j.progress} label={`${variantLabel("trained", d?.variant)} progress`} />
                  : <p className="small negv">{j.error} <button className="btn sm" onClick={train}>Retry</button></p>}
              </div>
            );
          })}
          {s.models.filter((m) => m.kind === "zero_shot").map((m) => (
            <div key={m.id} className="row small"><b>Zero-shot baseline</b> <span className="badge good">✓ ready</span> <span className="muted">{m.base_model}, no training needed</span></div>
          ))}
          <p className="tiny muted">Validation numbers come from frames of the same recording. They help pick a model but are optimistic. The real score comes from the next step.</p>
        </section>
      )}
    </div>
  );
}
