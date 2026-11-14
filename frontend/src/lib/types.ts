// UI model of the shared contract. The live adapter (lib/api/live.ts) converts the backend's objects into these:
// TaskSpec / DatasetManifest / Annotation come from Kirill's engines (task_id, dataset_id, ClassDef, BoxLabel),
// TrainingJob / ModelArtifact / InferenceResult from Kolyan's backend/schemas.py, EvaluationReport from Damir's engine.

export interface TaskSpec {
  id: string;
  prompt: string;
  task_type: "object_detection";
  classes: string[];
  target_event: string | null;
  negative_classes: string[];
  requires_tracking: boolean;
  primary_metric: string;
  notes: string | null;
  created_at: number;
  // from Kirill's Task Engine (live backend); absent in sample mode
  summary?: string;
  status?: "ok" | "needs_clarification" | "unsupported";
  issues?: string[];
  clarification_questions?: string[];
  visual_conditions?: string[];
  test_scenarios?: string[];
  class_descriptions?: Record<string, string>;
  compiler?: string;
}

export interface Annotation {
  image: string;
  cls: string;
  box: [number, number, number, number];
  score: number | null;
  source: "auto" | "human" | "synthetic";
}

export type DatasetVariant = "real_only" | "real_plus_synthetic" | "test";

export interface DatasetManifest {
  id: string;
  task_id: string | null;
  root: string;
  data_yaml: string;
  classes: string[];
  variant: DatasetVariant;
  counts: Record<string, number>;
  source_ids: string[];
  created_at: number;
  warnings?: string[];
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface TrainingRequest {
  dataset_id: string;
  base_model?: string;
  epochs?: number;
  imgsz?: number;
  batch?: number;
  export_onnx?: boolean;
}

export interface TrainingJob {
  id: string;
  kind: "training" | "evaluation";
  status: JobStatus;
  request: Record<string, unknown>;
  progress: number;
  message: string;
  error: string | null;
  result_id: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface ModelArtifact {
  id: string;
  kind: "trained" | "zero_shot";
  dataset_id: string | null;
  variant: string | null;
  base_model: string;
  classes: string[];
  weights: string | null;
  onnx: string | null;
  metrics: Record<string, number>;
  train_seconds: number | null;
  epochs: number | null;
  imgsz: number | null;
  device: string | null;
  created_at: number;
}

export interface EvaluationRequest {
  model_ids: string[];
  test_dataset_id: string;
}

// ---- shape of EvaluationReport.per_model[model_id], produced by backend/evaluation_engine/core.py ----
export interface PRF {
  tp: number;
  fp: number;
  fn: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  precision_ci95: [number, number] | null;
  recall_ci95: [number, number] | null;
}

export interface Check {
  name: string;
  op: string;
  target: number;
  value: number | null;
  passed: boolean;
}

export type Verdict = "ready" | "not_ready" | "insufficient_data" | "invalid_test_set";

export interface Failure {
  image: string;
  kind: "false_positive" | "false_negative";
  reason: string;
  cls: string;
  box: [number, number, number, number];
  score: number | null;
  iou: number;
  predicted_cls?: string | null;
  predicted_box?: [number, number, number, number] | null;
  thumbnail?: string | null;
  image_path?: string | null;   // absolute path on the GPU host, fetchable with GET /api/files?path=
}

export interface ModelEvaluation extends Omit<PRF, "precision_ci95" | "recall_ci95" | "tp"> {
  settings: { iou_threshold: number; conf_threshold: number; matching: string };
  test: {
    n_images: number;
    n_images_without_objects: number;
    n_sessions: number | null;
    gt_per_class: Record<string, number>;
    classes: string[];
    dataset_id?: string;
    split?: string;
  };
  overall: PRF;
  per_class: Record<string, PRF & { image_level: PRF & { tn: number } }>;
  failure_counts: { false_positive: Record<string, number>; false_negative: Record<string, number> };
  failures: Failure[];
  failures_total: number;
  performance: { latency_ms_median: number; latency_ms_p95: number; fps: number | null; n: number; note: string } | null;
  warnings: string[];
  readiness: { verdict: Verdict; summary: string; criteria: Check[]; data_criteria: Check[] };
  delta_vs_baseline?: { baseline_id: string; precision: number | null; recall: number | null; f1: number | null; fp: number; fn: number; recall_clear: boolean; precision_clear: boolean };
  model?: { id: string; kind: string; variant: string | null; base_model: string; dataset_id: string | null; epochs: number | null; created_at: number; val_metrics: Record<string, number> };
  fps: number | null;
  sample_data: boolean;
}

export interface EvaluationReport {
  id: string;
  request: EvaluationRequest;
  per_model: Record<string, ModelEvaluation>;
  notes: string[];
  created_at: number;
  image_size?: [number, number];
  mock?: boolean;               // true when the backend has no Evaluation Engine and returned val metrics only
}

export interface Detection {
  cls: string;
  score: number;
  box: [number, number, number, number];
}

export interface InferenceResult {
  id: string;
  model_id: string;
  width: number;
  height: number;
  detections: Detection[];
  latency_ms: number;
  events: string[];
}

export interface Health {
  ok: boolean;
  host?: string;
  engines?: Record<string, string>;                 // e.g. {"evaluation_engine": "Damir" | "MOCK", ...}
  gpu_worker?: { ok: boolean; error?: string; cuda?: boolean; device?: string | null; [k: string]: unknown };
}

/** Frontend-only metadata for an upload batch (one recording session). */
export interface UploadMeta {
  session_id: string;
  role: "train" | "test";
}
