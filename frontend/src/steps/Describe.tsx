"use client";
import { useState } from "react";
import { Composer } from "@/components/Composer";
import { Banner, ErrorBox } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";

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
      const task = await api.createTask(prompt.trim());
      set({ task, trainDataset: null, labels: [], frames: [], jobs: [], models: [], report: null, testDataset: null });
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
    <div className="center-col fade-in">
      <div className="hero">
        <h1>What should your camera detect?</h1>
        <p>Describe it in one sentence and add a few minutes of footage. ZALU labels it, trains a small model and tests it for you.</p>
      </div>
      <Composer value={prompt} onChange={setPrompt} files={s.trainFiles} onFiles={(trainFiles) => set({ trainFiles })}
        onSubmit={go} canSubmit={!!prompt.trim()} busy={!!busy} autoFocus
        placeholder="e.g. Find people without safety goggles" submitLabel="Create task" />
      {busy && <p className="hint" role="status">{busy === "task" ? "Understanding the task…" : "Uploading and labelling your footage. Longer videos take a few minutes."}</p>}
      {!busy && !prompt && (
        <div className="suggestions" aria-label="Examples">
          {EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => { setPrompt(x); document.getElementById("composer-text")?.focus(); }}>{x}</button>)}
        </div>
      )}
      {clarify && (
        <Banner kind="warn">
          <b>Could you be a bit more specific?</b>
          {clarify.map((q) => <div key={q}>• {q}</div>)}
        </Banner>
      )}
      <ErrorBox error={err} />
    </div>
  );
}
