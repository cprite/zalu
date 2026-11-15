import { frameFromBlob, type Frame } from "../frames";
import type { Annotation, DatasetManifest, TaskSpec } from "../types";
import type { ZaluApi } from "./types";
import { ApiError } from "./types";

/**
 * Real FastAPI backend (backend/main.py on kolyan/backend-p0: API on nikopi5, GPU worker on the GX10).
 * Backend objects are converted to the UI model here, so components never see contract details.
 */

// ---- backend shapes (Kirill's backend/task_engine/schema.py and backend/data_engine/schema.py) ----
interface RawTask {
  task_id: string; prompt: string; task_type: string; summary: string; status: TaskSpec["status"];
  issues: string[]; clarification_questions: string[]; classes: { name: string; description: string }[];
  target_event: string | null; requires_tracking: boolean; primary_metric: string;
  visual_conditions: string[]; test_scenarios: string[]; compiler: string;
}
interface RawDataset {
  dataset_id: string; created_at: string; task_id: string; variant: DatasetManifest["variant"]; root: string;
  data_yaml: string; class_names: string[]; warnings?: string[];
  splits: Record<string, { images: number; background_images: number; boxes: number; sessions: string[]; synthetic_images: number }>;
  images: { frame_id: string; split: string; session_id: string; synthetic: boolean }[];
}
interface RawBox { class_name: string; xyxy: number[]; confidence: number | null; source: "auto" | "manual" }
interface RawAnnotation { frame_id: string; image_path: string; width: number; height: number; session_id: string; synthetic: boolean; boxes: RawBox[]; status: string }

function toTask(r: RawTask): TaskSpec {
  const mock = (r.compiler || "").toUpperCase().startsWith("MOCK");
  return {
    id: r.task_id, prompt: r.prompt, task_type: "object_detection", classes: r.classes.map((c) => c.name),
    target_event: r.target_event, negative_classes: [], requires_tracking: r.requires_tracking, primary_metric: r.primary_metric,
    notes: mock ? `MOCK task spec from the backend (${r.compiler}).` : `Compiled by ${r.compiler}.`, created_at: Date.now() / 1000,
    summary: r.summary, status: r.status, issues: r.issues, clarification_questions: r.clarification_questions,
    visual_conditions: r.visual_conditions, test_scenarios: r.test_scenarios, compiler: r.compiler,
    class_descriptions: Object.fromEntries(r.classes.map((c) => [c.name, c.description])),
  };
}

function toDataset(r: RawDataset): DatasetManifest {
  const counts: Record<string, number> = Object.fromEntries(Object.entries(r.splits || {}).map(([k, v]) => [k, v.images]));
  const syn = Object.values(r.splits || {}).reduce((a, v) => a + (v.synthetic_images || 0), 0);
  if (syn) counts.synthetic = syn;
  const sessions = new Set<string>((r.images || []).map((i) => i.session_id).filter(Boolean));
  Object.values(r.splits || {}).forEach((v) => (v.sessions || []).forEach((s) => sessions.add(s)));
  return {
    id: r.dataset_id, task_id: r.task_id, root: r.root, data_yaml: r.data_yaml, classes: r.class_names, variant: r.variant,
    counts, source_ids: [...sessions], created_at: Date.parse(r.created_at) / 1000 || Date.now() / 1000, warnings: r.warnings,
  };
}

export function createLiveApi(baseUrl: string, token: string): ZaluApi {
  const base = baseUrl.replace(/\/+$/, "");
  const annotationCache = new Map<string, RawAnnotation[]>();

  async function raw(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    if (token) headers.set("Authorization", `Bearer ${token}`);
    if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
    let res: Response;
    try {
      res = await fetch(base + path, { ...init, headers });
    } catch (e) {
      throw new ApiError(0, `Backend unreachable at ${base} (${(e as Error).message})`, path);
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
      } catch { /* not JSON */ }
      throw new ApiError(res.status, detail, path);
    }
    return res;
  }
  const call = async <T,>(path: string, init: RequestInit = {}): Promise<T> => (await raw(path, init)).json() as Promise<T>;
  const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

  async function annotations(dsId: string): Promise<RawAnnotation[]> {
    if (!annotationCache.has(dsId)) {
      const set = await call<{ annotations: RawAnnotation[] }>(`/api/datasets/${dsId}/annotations`);
      annotationCache.set(dsId, set.annotations.filter((a) => a.status !== "rejected"));
    }
    return annotationCache.get(dsId)!;
  }

  return {
    mode: "live",
    health: () => call("/api/health"),
    createTask: async (prompt) => toTask(await call<RawTask>("/api/tasks", json({ prompt }))),
    createDataset: async (taskId, files, meta) => {
      // Recording sessions are derived from the uploaded file names by the Data Engine.
      const fd = new FormData();
      fd.append("task_id", taskId);
      fd.append("kind", meta.role === "test" ? "test" : "train");
      files.forEach((f) => fd.append("files", f, f.name));
      return toDataset(await call<RawDataset>("/api/datasets", { method: "POST", body: fd }));
    },
    listDatasets: async () => (await call<RawDataset[]>("/api/datasets")).map(toDataset),
    getLabels: async (dsId) =>
      (await annotations(dsId)).flatMap((a) => a.boxes.map((b): Annotation => ({
        image: a.frame_id, cls: b.class_name, box: b.xyxy as Annotation["box"], score: b.confidence,
        source: b.source === "manual" ? "human" : "auto",
      }))),
    listFrames: async (dsId, limit = 60) => {
      const anns = (await annotations(dsId)).slice(0, limit);
      const frames = await Promise.all(anns.map(async (a): Promise<Frame | null> => {
        try {
          const blob = await (await raw(`/api/files?path=${encodeURIComponent(a.image_path)}`)).blob();
          return await frameFromBlob(a.frame_id, blob, a.session_id);
        } catch {
          return null;
        }
      }));
      return frames.filter((f): f is Frame => !!f);
    },
    saveLabels: async (dsId, labels) => {
      // Send corrections only for frames whose boxes changed (LabelCorrection, Kirill's contract).
      const original = await annotations(dsId);
      const key = (bs: { cls: string; box: number[] }[]) => JSON.stringify(bs.map((b) => [b.cls, b.box.map((v) => Math.round(v))]).sort());
      const corrections = original.flatMap((a) => {
        const mine = labels.filter((l) => l.image === a.frame_id);
        const before = a.boxes.map((b) => ({ cls: b.class_name, box: b.xyxy }));
        const touched = mine.some((l) => l.source === "human");
        if (!touched && key(mine) === key(before)) return [];
        return [{ frame_id: a.frame_id, status: "reviewed", boxes: mine.map((l) => ({
          class_name: l.cls, xyxy: l.box, confidence: l.source === "human" ? null : l.score, source: l.source === "human" ? "manual" : "auto" })) }];
      });
      if (!corrections.length) return { applied: 0 };
      const res = await call(`/api/datasets/${dsId}/labels`, json(corrections));
      annotationCache.delete(dsId);
      return res;
    },
    startTraining: (req) => call("/api/training", json(req)),
    getJob: (id) => call(`/api/jobs/${id}`),
    createZeroShot: (classes) => call("/api/models/zero-shot", json({ classes })),
    listModels: () => call("/api/models"),
    getModel: (id) => call(`/api/models/${id}`),
    createEvaluation: (req) => call("/api/evaluations", json(req)),
    getReport: (id) => call(`/api/reports/${id}`),
    createDashboard: (prompt, taskId, modelId) => call("/api/dashboards", json({ prompt, task_id: taskId, model_id: modelId })),
    refineDashboard: (id, instruction) => call(`/api/dashboards/${encodeURIComponent(id)}/refine`, json({ instruction })),
    inference: (modelId, image, opts = {}) => {
      const fd = new FormData();
      fd.append("model_id", modelId);
      fd.append("image", image, "frame.jpg");
      fd.append("conf", String(opts.conf ?? 0.25));
      if (opts.targetEvent) fd.append("target_event", opts.targetEvent);
      return call("/api/inference", { method: "POST", body: fd });
    },
  };
}
