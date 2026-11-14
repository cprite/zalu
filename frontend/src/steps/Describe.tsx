"use client";
import { useState } from "react";
import { Banner, ErrorBox, SampleBadge, SectionHead, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";

const EXAMPLES = [
  "Find people without safety goggles at the lab bench",
  "Which restaurant tables are free",
  "Alert me when someone on site is not wearing a hi-vis vest",
];

export function Describe() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [prompt, setPrompt] = useState(s.task?.prompt ?? "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const t = s.task;
  const isSample = api.mode === "mock" || !!t?.notes?.toUpperCase().includes("MOCK") || !!t?.notes?.toUpperCase().includes("SAMPLE");

  async function go() {
    setBusy(true);
    setErr(null);
    try {
      const task = await api.createTask(prompt.trim());
      set({ task, trainDataset: null, labels: [], jobs: [], models: [], report: null, testDataset: null });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  const edit = (patch: Partial<NonNullable<typeof t>>) => t && set({ task: { ...t, ...patch } });

  return (
    <div className="fade-in">
      <SectionHead title="What should your camera detect?" sub="Describe it in your own words. We turn it into a detection task you can check before anything is trained." />
      <div className="card col">
        <label htmlFor="task-prompt" className="sr-only">Describe what the camera should detect</label>
        <textarea id="task-prompt" className="textarea" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="e.g. Find people without safety goggles"
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && prompt.trim()) go(); }} />
        <div className="row">
          {EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => setPrompt(x)}>{x}</button>)}
          <div className="spacer" />
          <button className={`btn ${t ? "" : "primary"}`} disabled={!prompt.trim() || busy} onClick={go}>{busy ? <Spinner /> : null}{t ? "Update task" : "Create task"}</button>
        </div>
        <ErrorBox error={err} />
      </div>

      {t && (
        <div className="card fade-in" style={{ marginTop: 16 }}>
          <div className="card-title"><h3>Task summary</h3><SampleBadge show={isSample} label="Sample planner" /><div className="spacer" /><span className="mono muted tiny">{t.id}</span></div>
          <div className="grid2">
            <div className="col">
              <div>
                <div className="small ink2" style={{ fontWeight: 600 }}>The model will look for</div>
                <ChipEditor items={t.classes} onChange={(classes) => edit({ classes, target_event: classes.includes(t.target_event ?? "") ? t.target_event : classes[0] ?? null })} kind="accent" />
              </div>
              {(t.negative_classes.length > 0 || !t.compiler) && (
                <div>
                  <div className="small ink2" style={{ fontWeight: 600 }}>Look-alikes it must NOT flag</div>
                  <ChipEditor items={t.negative_classes} onChange={(negative_classes) => edit({ negative_classes })} kind="neg" />
                </div>
              )}
              {t.visual_conditions && t.visual_conditions.length > 0 && (
                <div className="small">
                  <div className="ink2" style={{ fontWeight: 600 }}>Footage should cover</div>
                  <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{t.visual_conditions.map((v) => <li key={v}>{v}</li>)}</ul>
                </div>
              )}
            </div>
            <div className="col small">
              {t.summary && <div><b>{t.summary}</b></div>}
              {t.status && t.status !== "ok" && (
                <Banner kind="warn">
                  <b>{t.status === "unsupported" ? "This task is not supported yet." : "The planner needs clarification."}</b>
                  {[...(t.issues ?? []), ...(t.clarification_questions ?? [])].map((q) => <div key={q}>• {q}</div>)}
                </Banner>
              )}
              <div><span className="ink2">Alert when:</span> <b>{t.target_event ?? "—"}</b> is detected</div>
              <div><span className="ink2">Most important:</span> <b>{t.primary_metric === "recall" ? "don't miss any (recall)" : t.primary_metric}</b></div>
              <div><span className="ink2">Follows objects over time:</span> <b>{t.requires_tracking ? "yes" : "no"}</b></div>
              {t.notes && <div className="muted tiny">{t.notes}</div>}
            </div>
          </div>
          <div className="row" style={{ marginTop: 16 }}>
            <span className="small muted">Class edits are kept in this browser for the next steps.</span>
            <div className="spacer" />
            <button className="btn primary" disabled={!t.classes.length} onClick={() => set({ step: 1 })}>Looks right → Upload footage</button>
          </div>
        </div>
      )}
    </div>
  );
}

function ChipEditor({ items, onChange, kind }: { items: string[]; onChange: (x: string[]) => void; kind: "accent" | "neg" }) {
  const [v, setV] = useState("");
  const add = () => {
    const x = v.trim().toLowerCase().replace(/\s+/g, "_");
    if (x && !items.includes(x)) onChange([...items, x]);
    setV("");
  };
  return (
    <div className="row" style={{ marginTop: 6, gap: 6 }}>
      {items.map((c) => (
        <span key={c} className={`chip ${kind}`}>{c}<button aria-label={`remove ${c}`} onClick={() => onChange(items.filter((x) => x !== c))}>×</button></span>
      ))}
      <input className="input" style={{ width: 140, padding: "4px 10px" }} placeholder="+ add" value={v} onChange={(e) => setV(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && add()} onBlur={add} />
    </div>
  );
}
