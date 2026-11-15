"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState, type FormEvent } from "react";
import { ScreenActivity } from "@/components/ScreenActivity";
import { DashboardRenderer } from "@/dashboard/DashboardRenderer";
import { samplePlan, type DashboardPlan } from "@/dashboard/samplePlan";
import { useInferenceSession } from "@/dashboard/useInferenceSession";
import { ErrorBox } from "@/components/ui";
import { useApi } from "@/lib/api";
import { pct, variantLabel } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import { t } from "@/lib/motion";

export function Deploy() {
  const { s, set } = useFlow();
  const model = s.models.find((item) => item.id === s.deployModelId)
    ?? s.models.find((item) => item.kind === "trained") ?? s.models[0];
  if (!model || !s.task) return <p className="hint">Train a detection model first.</p>;
  return (
    <div className="wide-col" style={{ maxWidth: 1120 }}>
      <div className="row deploy-tools">
        <button className="btn ghost sm" onClick={() => set({ step: 4 })}>← Evaluation</button>
        {s.models.length > 1 && <label>Model <select className="select" value={model.id}
          onChange={(event) => set({ deployModelId: event.target.value })}>
          {s.models.map((item) => <option key={item.id} value={item.id}>{variantLabel(item.kind, item.variant)}</option>)}
        </select></label>}
      </div>
      <DashboardStage key={model.id} modelId={model.id} classes={model.classes} taskId={s.task.id}
        prompt={s.task.prompt} target={s.task.target_event ?? ""} />
    </div>
  );
}

function DashboardStage({ modelId, classes, taskId, prompt, target }: {
  modelId: string; classes: string[]; taskId: string; prompt: string; target: string;
}) {
  const { api } = useApi();
  const { s, set } = useFlow();
  const sampleMode = api.mode === "mock";
  const activeTarget = classes.includes(target) ? target : "";
  const evalEntry = s.report?.per_model[modelId];
  const evaluationInvalid = evalEntry?.readiness?.verdict === "invalid_test_set"
    || (evalEntry?.test?.split != null && evalEntry.test.split !== "test");
  const measured = s.report?.mock || evaluationInvalid ? undefined : evalEntry?.overall;
  const plan = s.dashboardPlans[modelId];
  const [createError, setCreateError] = useState<unknown>(null);   // dashboard generation failed
  const [planError, setPlanError] = useState<unknown>(null);       // refine / export failed
  const planning = !plan && !createError;                          // derived, so the effect never sets state synchronously
  const [refining, setRefining] = useState(false);
  const [instruction, setInstruction] = useState("");
  const [attempt, setAttempt] = useState(0);
  const session = useInferenceSession((frame) => api.inference(modelId, frame, {
    conf: evalEntry?.settings?.conf_threshold ?? 0.25, targetEvent: activeTarget || null,
  }), sampleMode, classes, activeTarget);
  const savePlan = (next: DashboardPlan) => set((current) => ({
    dashboardPlans: { ...current.dashboardPlans, [modelId]: next },
  }));

  useEffect(() => {
    if (plan) return;
    let active = true;
    if (sampleMode) {
      savePlan(samplePlan(prompt, classes, activeTarget));
      return;
    }
    api.createDashboard(prompt, taskId, modelId).then((next) => {
      if (active) savePlan(next);
    }).catch((error) => {
      if (active) setCreateError(error);
    });
    return () => { active = false; };
    // classes / activeTarget / savePlan are re-created every render; the plan only depends on the ids below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, attempt, modelId, plan, prompt, sampleMode, taskId]);

  async function refine(event: FormEvent) {
    event.preventDefault();
    if (!plan || !instruction.trim() || refining) return;
    setRefining(true);
    setPlanError(null);
    try {
      if (sampleMode) {
        const changes = instruction.trim();
        const rename = /^(?:rename(?: dashboard)?(?: to)?|title)\s*[: ]\s*(.+)$/i.exec(changes);
        const preview = /^(hide|show) (?:the )?(?:camera|preview)$/i.exec(changes);
        if (!rename && !preview) throw new Error('In Sample mode, try “rename to Loading dock” or “hide camera”.');
        const spec = structuredClone(plan.spec);
        if (rename) spec.title = rename[1].slice(0, 100);
        if (preview?.[1].toLowerCase() === "hide") {
          if (spec.widgets.length === 1) throw new Error("Keep at least one dashboard widget.");
          spec.widgets = spec.widgets.filter((widget) => widget.type !== "live_preview");
        } else if (preview && !spec.widgets.some((widget) => widget.type === "live_preview")) {
          spec.widgets.unshift({ id: "camera", title: "Camera preview", type: "live_preview",
            source_id: spec.data_sources[0].id, span: spec.layout.columns, config: { show_boxes: true } });
        }
        savePlan({ ...plan, spec });
      } else savePlan(await api.refineDashboard(plan.id, instruction.trim()));
      setInstruction("");
    } catch (error) {
      setPlanError(error);
    } finally {
      setRefining(false);
    }
  }

  return <>
    <AnimatePresence initial={false}>
      {planning && (
        <motion.div key="planning" className="hero compact" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={t("base")}>
          <ScreenActivity message="Preparing dashboard" />
        </motion.div>
      )}
    </AnimatePresence>
    <ErrorBox error={createError} />
    <ErrorBox error={planError} />
    {!plan && !planning && <button className="btn" onClick={() => { setCreateError(null); setAttempt((value) => value + 1); }}>Retry dashboard</button>}
    {plan && <>
      <DashboardRenderer spec={plan.spec} classes={classes} samples={session.samples} preview={session.preview}
        loading={session.loading} error={session.error} sampleMode={sampleMode} />
      <div className="row deploy-tools">
        <button className="btn primary" onClick={session.running ? session.stop : session.start} aria-pressed={session.running}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={session.running ? "stop" : "play"} aria-hidden style={{ display: "inline-grid" }}
              initial={{ opacity: 0, scale: 0.6, rotate: -45 }} animate={{ opacity: 1, scale: 1, rotate: 0 }} exit={{ opacity: 0, scale: 0.6 }} transition={t("fast")}>
              {session.running ? "■" : "▶"}
            </motion.span>
          </AnimatePresence>
          {session.running ? "Stop inference" : sampleMode ? "Run sample inference" : "Start camera"}
        </button>
        {!sampleMode && <label className="btn">Choose image
          <input type="file" className="sr-only" accept="image/*" onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void session.useImage(file);
            event.target.value = "";
          }} />
        </label>}
        <button className="btn" onClick={async () => {
          try {
            const { downloadDashboard } = await import("@/dashboard/exportDashboard");
            downloadDashboard(plan.spec, classes, activeTarget, evalEntry?.settings?.conf_threshold ?? 0.25);
          } catch (error) {
            setPlanError(error);
          }
        }}>Export dashboard</button>
        {!sampleMode && !plan.ai_generated && <span className="small muted">Default layout</span>}
        {measured && <span className="small muted">
          {sampleMode ? "Sample evaluation" : "Test evaluation"} · precision {pct(measured.precision)} · recall {pct(measured.recall)}
        </span>}
      </div>
      {[...plan.limitations, ...plan.clarification_questions].length > 0 && (
        <ul className="small muted">{[...plan.limitations, ...plan.clarification_questions].map((message, i) => <li key={i}>{message}</li>)}</ul>
      )}
      <form className="dash-refine" aria-busy={refining} onSubmit={refine}>
        <label htmlFor="dashboard-refinement">Refine dashboard</label>
        <div className="row">
          <input id="dashboard-refinement" className="input" value={instruction} maxLength={500}
            placeholder={sampleMode ? "e.g. Rename to Loading dock" : "e.g. Put the detection table first"}
            onChange={(event) => setInstruction(event.target.value)} />
          <button className="btn" disabled={refining || !instruction.trim()}>Apply</button>
        </div>
        <ScreenActivity slot message={refining ? "Updating dashboard" : undefined} />
      </form>
    </>}
  </>;
}
