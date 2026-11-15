"use client";
// Shared state of the Describe -> Upload -> Prepare -> Train -> Verify -> Deploy journey.
import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import type { DashboardPlan } from "../dashboard/samplePlan";
import type { Frame } from "./frames";
import type { Annotation, DatasetManifest, EvaluationReport, ModelArtifact, TaskSpec, TrainingJob } from "./types";

export interface FlowState {
  step: number;
  task: TaskSpec | null;
  trainSession: string;
  trainFiles: File[];
  frames: Frame[];
  trainDataset: DatasetManifest | null;
  labels: Annotation[];
  labelsSaved: boolean;
  jobs: TrainingJob[];
  models: ModelArtifact[];
  testSession: string;
  testFiles: File[];
  testDataset: DatasetManifest | null;
  report: EvaluationReport | null;
  deployModelId: string | null;
  dashboardPlans: Record<string, DashboardPlan>;
  annotationsLoaded: boolean;
  annotationsDirty: boolean;
}

const initial: FlowState = {
  step: 0, task: null, trainSession: "session_1", trainFiles: [], frames: [], trainDataset: null, labels: [],
  labelsSaved: false, jobs: [], models: [], testSession: "test_session_1", testFiles: [], testDataset: null,
  report: null, deployModelId: null, dashboardPlans: {}, annotationsLoaded: false, annotationsDirty: false,
};

export const STEPS = [
  { key: "describe", label: "Describe", sub: "What to detect" },
  { key: "upload", label: "Upload", sub: "Your camera footage" },
  { key: "prepare", label: "Prepare", sub: "Check auto-labels" },
  { key: "train", label: "Train", sub: "Small, fast model" },
  { key: "verify", label: "Verify", sub: "Test on new footage" },
  { key: "deploy", label: "Deploy", sub: "Live camera" },
] as const;

/** The three user-facing stages. Internal steps (above) are screens inside a stage:
 *  Describe = describe + upload + prepare/verify labels; Train = train + test on new footage; Deploy = live camera. */
export const STAGES = [
  { label: "Describe", steps: [0, 1, 2] },
  { label: "Train", steps: [3, 4] },
  { label: "Deploy", steps: [5] },
] as const;

export function stageOf(step: number): number {
  return STAGES.findIndex((st) => (st.steps as readonly number[]).includes(step));
}

/** 0..1 fill of each stage segment for the current step (Stories-style progress). */
export function stageFill(step: number): number[] {
  const cur = stageOf(step);
  return STAGES.map((st, i) => (i < cur ? 1 : i > cur ? 0 : ((st.steps as readonly number[]).indexOf(step) + 1) / st.steps.length));
}

export function unlocked(s: FlowState): boolean[] {
  const trained = s.models.some((m) => m.kind === "trained");
  return [true, !!s.task, !!s.trainDataset, !!s.trainDataset, trained, trained];
}

const Ctx = createContext<{ s: FlowState; set: (p: Partial<FlowState> | ((s: FlowState) => Partial<FlowState>)) => void; reset: () => void } | null>(null);

export function FlowProvider({ children }: { children: ReactNode }) {
  const [s, setS] = useState(initial);
  const set = useCallback((p: Partial<FlowState> | ((s: FlowState) => Partial<FlowState>)) =>
    setS((prev) => ({ ...prev, ...(typeof p === "function" ? p(prev) : p) })), []);
  const reset = useCallback(() => setS(initial), []);
  return <Ctx.Provider value={{ s, set, reset }}>{children}</Ctx.Provider>;
}

export function useFlow() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useFlow outside FlowProvider");
  return c;
}
