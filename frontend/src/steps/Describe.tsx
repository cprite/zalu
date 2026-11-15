"use client";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { ScreenActivity } from "@/components/ScreenActivity";
import { Composer } from "@/components/Composer";
import { Banner, ErrorBox } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { reveal, rise, stagger, t } from "@/lib/motion";

const EXAMPLES = [
  "Find people without safety goggles",
  "Which restaurant tables are free",
  "Alert me when someone isn't wearing a hi-vis vest",
];

/** Stage 1 · Describe: one message box for the task; footage can be attached right away. */
export function Describe() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [prompt, setPrompt] = useState(s.task?.prompt ?? "");
  const [busy, setBusy] = useState<"" | "task" | "upload">("");
  const [err, setErr] = useState<unknown>(null);
  const [clarify, setClarify] = useState<string[] | null>(null);

  async function go() {
    setBusy("task");
    setErr(null);
    setClarify(null);
    try {
      if (s.task?.prompt === prompt.trim() && (!s.task.status || s.task.status === "ok")) {
        if (s.trainDataset || !s.trainFiles.length) {
          set({ step: s.trainDataset ? 2 : 1 });
        } else {
          setBusy("upload");
          const ds = await api.createDataset(s.task.id, s.trainFiles, { session_id: s.trainSession, role: "train" });
          set({ trainDataset: ds, step: 2 });
        }
        return;
      }
      const task = await api.createTask(prompt.trim());
      set({ task, trainDataset: null, labels: [], frames: [], labelsSaved: false, annotationsLoaded: false, annotationsDirty: false, jobs: [], models: [],
        report: null, testDataset: null, deployModelId: null, dashboardPlans: {} });
      if (task.status && task.status !== "ok") {
        setClarify([...(task.issues ?? []), ...(task.clarification_questions ?? [])]);
        return;
      }
      if (s.trainFiles.length) {
        setBusy("upload");
        const ds = await api.createDataset(task.id, s.trainFiles, { session_id: s.trainSession, role: "train" });
        set({ trainDataset: ds, step: 2 });
      } else {
        set({ step: 1 });
      }
    } catch (e) {
      setErr(e);
    } finally {
      setBusy("");
    }
  }

  return (
    <motion.div className="center-col" variants={stagger} initial="hidden" animate="show">
      <div className="hero">
        <motion.h1 variants={rise}>What would you like to detect?</motion.h1>
        <motion.p variants={rise}>Describe what to look for and attach footage from your camera.</motion.p>
      </div>
      <motion.div variants={rise}>
      <Composer value={prompt} onChange={setPrompt} files={s.trainFiles} onFiles={(trainFiles) => set({ trainFiles, trainDataset: null, frames: [], labels: [], labelsSaved: false, annotationsLoaded: false, annotationsDirty: false,
          jobs: [], models: [], report: null, deployModelId: null, dashboardPlans: {} })}
        onSubmit={go} canSubmit={!!prompt.trim()} busy={!!busy} autoFocus
        placeholder="e.g. Find people without safety goggles" submitLabel="Create task" />
      </motion.div>
      <motion.div variants={rise} style={{ minHeight: 40 }}>
        <AnimatePresence mode="wait" initial={false}>
          {busy ? (
            <ScreenActivity key="busy" message={busy === "task" ? "Reading your request" : "Preparing footage"} />
          ) : !prompt ? (
            <motion.div key="examples" className="suggestions" aria-label="Examples" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={t("fast")}>
              {EXAMPLES.map((x) => (
                <motion.button key={x} className="chip" whileTap={{ scale: 0.97 }} transition={t("fast")}
                  onClick={() => { setPrompt(x); document.getElementById("composer-text")?.focus(); }}>{x}</motion.button>
              ))}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </motion.div>
      <AnimatePresence>
        {clarify && (
          <motion.div key="clarify" {...reveal} style={{ overflow: "hidden" }}>
            <Banner kind="warn">
              <b>Could you be a bit more specific?</b>
              {clarify.map((q) => <div key={q}>• {q}</div>)}
            </Banner>
          </motion.div>
        )}
      </AnimatePresence>
      <ErrorBox error={err} />
    </motion.div>
  );
}
