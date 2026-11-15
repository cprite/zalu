"use client";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { ErrorBox } from "@/components/ui";
import { ScreenActivity, TrainingComplete, TrainingProgress } from "@/components/ScreenActivity";
import styles from "./train.module.css";
import { useApi } from "@/lib/api";
import { variantLabel } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import { Disclosure, EXIT, SPRING, reveal, t } from "@/lib/motion";
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
  const [showOptions, setShowOptions] = useState(false);
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
    let active = true;
    const t = setTimeout(async () => {
      try {
        const jobs = await Promise.all(s.jobs.map((job) =>
          job.status === "succeeded" || job.status === "failed" ? job : api.getJob(job.id)));
        const newModels = await Promise.all(jobs.filter((job) => job.status === "succeeded" && job.result_id
          && !s.models.some((model) => model.id === job.result_id)).map((job) => api.getModel(job.result_id!)));
        if (active) {
          setErr(null);
          set((current) => ({ jobs, models: [...current.models,
            ...newModels.filter((model) => !current.models.some((item) => item.id === model.id))] }));
        }
      } catch (error) {
        if (active) {
          setErr(error);
          set((current) => ({ jobs: [...current.jobs] }));
        }
      }
    }, 1500);
    return () => { active = false; clearTimeout(t); };
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
  // real training time reported by the backend (slowest trained model), shown only when present
  const seconds = s.models.filter((m) => m.kind === "trained" && m.train_seconds != null).map((m) => m.train_seconds as number);
  const trainedIn = seconds.length ? (Math.max(...seconds) >= 90 ? `${Math.round(Math.max(...seconds) / 60)} min` : `${Math.round(Math.max(...seconds))} s`) : null;
  const nothingPicked = !zeroShot && !datasets.some((d) => pick[d.id]);

  const overall = s.jobs.length ? s.jobs.reduce((t, j) => t + (j.status === "failed" ? 0 : j.progress), 0) / s.jobs.length : 0;

  return (
    <div className="center-col">
      <div className="hero compact">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={done ? "done" : running ? "running" : "idle"} className="col" style={{ alignItems: "center", gap: 8 }}
            initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={t("base")}>
            {done && <TrainingComplete />}
            <h1>{done ? "Training complete" : s.jobs.some((job) => job.status === "running") ? "Training your model" : running ? "Training queued" : "Train your model"}</h1>
            <p>{done ? `${trainedIn ? `Trained in ${trainedIn}. ` : ""}Test it on fresh footage.` : !running ? "Use your labelled footage to train a detector." : null}</p>
            {api.mode === "mock" && <span className="small muted">Demo training</span>}
          </motion.div>
        </AnimatePresence>
      </div>

      <ScreenActivity message={busy ? "Submitting training" : undefined} />

      {(s.jobs.length > 0) && (
        <motion.section className={`card ${styles.jobs}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={t("base")}>
          {s.jobs.length > 1 && <TrainingProgress value={overall} active={s.jobs.some((job) => job.status === "running")} label="Overall training progress" />}
          {s.jobs.map((j) => {
            const d = datasets.find((x) => x.id === j.request.dataset_id);
            const epoch = /epoch\s+(\d+)\s*\/\s*(\d+)/i.exec(j.message ?? "");   // only when the backend reports it
            const name = variantLabel("trained", d?.variant);
            const repeated = s.jobs.filter((job) => job.request.dataset_id === j.request.dataset_id);
            const attempt = repeated.length > 1 ? ` · Run ${repeated.indexOf(j) + 1}` : "";
            const active = j.status === "running";
            const epochText = epoch ? `Epoch ${epoch[1]} of ${epoch[2]}` : undefined;
            const message = j.message?.split("\n")[0].replace(/\s*\(sample\)$/i, "").trim();
            return (
              <article key={j.id} className={styles.job} data-job-status={j.status}>
                <div className={styles.jobHeader}>
                  <b>{name}{attempt}</b>
                  {active && <span className="small muted num">{Math.round(j.progress * 100)}%</span>}
                </div>
                {active ? <ScreenActivity announce={!epochText} message={epochText ?? (message || "Training model")} />
                  : <span className={`small ${j.status === "failed" ? "negv" : "muted"}`}>
                    {j.status === "queued" ? "Queued" : j.status === "succeeded" ? "Complete" : "Failed"}
                  </span>}
                {j.status !== "failed"
                  ? <TrainingProgress value={j.progress} active={active} label={`${name}${attempt} progress`} valueText={active ? epochText : undefined} />
                  : <p className="small negv">{j.error} <button className="btn sm" onClick={train}>Retry</button></p>}
              </article>
            );
          })}
        </motion.section>
      )}

      <ErrorBox error={err} />

      {!running && (
        <div className="actions-center">
          {done ? (
            <>
              <button className="btn ghost" onClick={() => set({ step: 2 })}>← Labels</button>
              <button className="btn ghost" disabled={busy || nothingPicked} onClick={train}>Train again</button>
              <button className="btn primary" onClick={() => set({ step: 4 })}>Test on new footage →</button>
            </>
          ) : (
            <>
              <button className="btn ghost" onClick={() => set({ step: 2 })}>← Labels</button>
              <button className="btn primary" disabled={busy || nothingPicked} onClick={train}>Start training</button>
            </>
          )}
        </div>
      )}

      {!running && (
        <div className="actions-center">
          <Disclosure open={showOptions} onToggle={() => setShowOptions(!showOptions)} controls="train-options">Options</Disclosure>
        </div>
      )}
      <AnimatePresence initial={false}>
      {showOptions && !running && (
        <motion.section key="options" id="train-options" className="card col" {...reveal} exit={{ ...reveal.exit, transition: EXIT }} style={{ overflow: "hidden" }}>
          <fieldset className="col" style={{ border: 0, padding: 0, margin: 0, gap: 8 }}>
            <legend className="text-[15px] font-semibold" style={{ marginBottom: 8 }}>Models to compare</legend>
            {datasets.map((d) => (
              <label key={d.id} className="flex gap-2 items-start small">
                <input type="checkbox" checked={!!pick[d.id]} onChange={(e) => setPick({ ...pick, [d.id]: e.target.checked })} />
                <span>{variantLabel("trained", d.variant)}</span>
              </label>
            ))}
            <label className="flex gap-2 items-start small">
              <input type="checkbox" checked={zeroShot} onChange={(e) => setZeroShot(e.target.checked)} />
              <span>Pretrained baseline for comparison</span>
            </label>
          </fieldset>
          <div className="tabs has-pill" role="radiogroup" aria-label="Training time">
            {PRESETS.map((p) => (
              <button key={p.epochs} role="radio" aria-checked={epochs === p.epochs} className={epochs === p.epochs ? "on" : ""} onClick={() => setEpochs(p.epochs)}>
                {epochs === p.epochs && <motion.span layoutId="train-preset" className="tab-pill" transition={SPRING} aria-hidden />}
                <span className="tab-label">{p.label} <span className="muted">{p.note}</span></span>
              </button>
            ))}
          </div>
        </motion.section>
      )}
      </AnimatePresence>
    </div>
  );
}
