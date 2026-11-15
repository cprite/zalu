"use client";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { ErrorBox, Progress, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { variantLabel } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import { AnimatedNumber, CheckDraw, Disclosure, EXIT, SPRING, StatusLine, reveal, t } from "@/lib/motion";
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
            {done && (
              <span className="done-wrap" aria-hidden>
                <motion.span className="done-ring" initial={{ opacity: 0.5, scale: 1 }} animate={{ opacity: 0, scale: 1.8 }} transition={{ duration: 0.9, ease: [0.2, 0, 0, 1], delay: 0.2 }} />
                <motion.span className="done-mark" initial={{ scale: 0.7 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 420, damping: 26 }}>
                  <CheckDraw size={22} delay={0.08} />
                </motion.span>
              </span>
            )}
            <h1>{done ? "Your model is ready" : running ? "Training your model…" : "Train your model"}</h1>
            <p>{done
              ? (trainedIn ? `Trained in ${trainedIn}. Next, ZALU checks it on footage it has never seen.` : "Next, ZALU checks it on footage it has never seen.")
              : "A small, fast detector trained on your labelled frames. It takes a few minutes."}</p>
          </motion.div>
        </AnimatePresence>
      </div>

      {(s.jobs.length > 0) && (
        <motion.section className="card col" aria-live="polite" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={t("base")}>
          {s.jobs.length > 1 && <Progress value={overall} label="Overall training progress" />}
          {s.jobs.map((j) => {
            const d = datasets.find((x) => x.id === j.request.dataset_id);
            const epoch = /epoch\s+(\d+)\s*\/\s*(\d+)/i.exec(j.message ?? "");   // only when the backend reports it
            const state = j.status === "succeeded" ? "done" : j.status === "failed" ? "error" : "working";
            const label = j.status === "queued" ? "Waiting for the GPU…" : j.status === "running" ? (epoch ? "Training…" : (j.message.split("\n")[0] || "Training…")) : j.status === "succeeded" ? "Done" : "Failed";
            return (
              <div key={j.id} className="col" style={{ gap: 4 }}>
                <div className="row small">
                  <b>{variantLabel("trained", d?.variant)}</b>
                  <div className="spacer" />
                  {epoch && j.status === "running" && (
                    <span className="epoch small">epoch <AnimatedNumber value={Number(epoch[1])} format={(x) => String(Math.round(x))} duration={0.15} /> of {epoch[2]}</span>
                  )}
                  {j.status === "running" && (
                    <span className="small num muted" style={{ minWidth: 40, textAlign: "right" }}>
                      <AnimatedNumber value={Math.round(j.progress * 100)} format={(x) => `${Math.round(x)}%`} duration={0.22} />
                    </span>
                  )}
                  <StatusLine state={state} className="small">{label}</StatusLine>
                </div>
                {j.status !== "failed"
                  ? <Progress value={j.progress} label={`${variantLabel("trained", d?.variant)} progress`} />
                  : <p className="small negv">{j.error} <button className="btn sm" onClick={train}>Retry</button></p>}
              </div>
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
              <button className="btn primary" disabled={busy || nothingPicked} onClick={train}>{busy && <Spinner />}Start training</button>
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
              <span>Zero-shot baseline (no training, for comparison)</span>
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
