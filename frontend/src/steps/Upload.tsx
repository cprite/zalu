"use client";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { Composer } from "@/components/Composer";
import { CameraCapture } from "@/components/MediaInput";
import { ErrorBox } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { makeSampleFrames } from "@/lib/sampleFrames";
import { TextShimmer, reveal, t } from "@/lib/motion";

/** Stage 1 · Describe → footage. Same message-box feel, files only. */
export function Upload() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [camera, setCamera] = useState(false);

  async function submit() {
    if (!s.task) return;
    if (s.trainDataset) {
      set({ step: 2 });
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const ds = await api.createDataset(s.task.id, s.trainFiles, { session_id: s.trainSession, role: "train" });
      set({ trainDataset: ds, frames: [], labels: [], labelsSaved: false, annotationsLoaded: false, annotationsDirty: false, step: 2 });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="center-col">
      <div className="hero compact">
        <h1>Add footage from your camera</h1>
        <p>A few minutes from the place where the camera will run, including moments where there is nothing to detect.</p>
      </div>
      <Composer textless files={s.trainFiles} onFiles={(trainFiles) => set({ trainFiles, trainDataset: null, frames: [], labels: [], labelsSaved: false, annotationsLoaded: false, annotationsDirty: false,
          jobs: [], models: [], report: null, deployModelId: null, dashboardPlans: {} })} onSubmit={submit}
        canSubmit={s.trainFiles.length > 0} busy={busy} submitLabel="Upload and label" />
      <AnimatePresence initial={false}>
        {busy && (
          <motion.p key="busy" className="hint" role="status" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={t("fast")}>
            <TextShimmer>Uploading and labelling your footage. Longer videos take a few minutes.</TextShimmer>
          </motion.p>
        )}
      </AnimatePresence>
      {!busy && (
        <div className="actions-center">
          <button className="toggle" onClick={() => setCamera(!camera)} aria-expanded={camera}>{camera ? "Hide camera" : "Record with my camera"}</button>
          {api.mode === "mock" && !s.trainFiles.length && (
            <button className="toggle" onClick={async () => set({ trainFiles: await makeSampleFrames(36, s.trainSession) })}>Use sample frames</button>
          )}
        </div>
      )}
      <AnimatePresence initial={false}>
        {camera && (
          <motion.div key="camera" {...reveal} style={{ overflow: "hidden" }}>
            <CameraCapture sessionId={s.trainSession} onFrames={(fs) => set({ trainFiles: [...s.trainFiles, ...fs], trainDataset: null, frames: [], labels: [],
              labelsSaved: false, annotationsLoaded: false, annotationsDirty: false, jobs: [], models: [], report: null, deployModelId: null, dashboardPlans: {} })} />
          </motion.div>
        )}
      </AnimatePresence>
      <ErrorBox error={err} />
      <div className="actions-center"><button className="btn ghost sm" onClick={() => set({ step: 0 })}>← Change the task</button></div>
    </div>
  );
}
