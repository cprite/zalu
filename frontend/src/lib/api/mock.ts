// SAMPLE DATA ONLY. In-memory mock of the backend so the UI works with no server.
// Every object it returns is marked as sample (notes / ids / sample_data) and the UI shows a SAMPLE badge.
import sampleReport from "@/mocks/sample-report.json";
import { extractFrames, type Frame } from "../frames";
import { SAMPLE_TRUTH } from "../sampleFrames";
import type {
  Annotation, DatasetManifest, EvaluationReport, ModelArtifact, ModelEvaluation, TaskSpec, TrainingJob,
} from "../types";
import type { ZaluApi } from "./types";
import { ApiError } from "./types";

const TRAIN_SECONDS = 10;
const now = () => Date.now() / 1000;
const id = (p: string) => `${p}_sample${Math.random().toString(16).slice(2, 8)}`;
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

const db = {
  tasks: new Map<string, TaskSpec>(),
  datasets: new Map<string, DatasetManifest>(),
  frames: new Map<string, Frame[]>(),
  labels: new Map<string, Annotation[]>(),
  jobs: new Map<string, TrainingJob & { _t0: number }>(),
  models: new Map<string, ModelArtifact>(),
  reports: new Map<string, EvaluationReport>(),
};

/** Very small rule-based "planner" standing in for Kirill's Task Engine (fine-tuned with Adaption). */
export function mockPlan(prompt: string): Pick<TaskSpec, "classes" | "negative_classes" | "target_event" | "requires_tracking" | "primary_metric"> {
  const p = prompt.toLowerCase();
  const rules: [RegExp, string[], string[], boolean][] = [
    [/goggle|glasses|eye protection/, ["no_goggles", "goggles"], ["sunglasses", "face_shield"], false],
    [/vest|hi-?vis/, ["no_vest", "vest"], ["orange_jacket", "traffic_cone"], false],
    [/helmet|hard ?hat/, ["no_helmet", "helmet"], ["cap", "hood"], false],
    [/table/, ["free_table", "occupied_table"], ["chair", "lamp"], true],
    [/parking|car|space/, ["free_space", "occupied_space"], ["shadow", "road_marking"], true],
    [/queue|line/, ["person_in_queue"], ["mannequin", "poster_person"], true],
  ];
  for (const [re, classes, neg, track] of rules) {
    if (re.test(p)) return { classes, negative_classes: neg, target_event: classes[0], requires_tracking: track, primary_metric: "recall" };
  }
  const noun = p.match(/(?:find|detect|count|spot|track)\s+(?:all\s+|the\s+|any\s+)?([a-z]+)/)?.[1] ?? "object";
  return { classes: [noun], negative_classes: [], target_event: noun, requires_tracking: false, primary_metric: "recall" };
}

function hash(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return () => ((h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0) / 4294967296);
}

function jobView(j: TrainingJob & { _t0: number }): TrainingJob {
  const { _t0, ...job } = j;
  if (job.status === "succeeded" || job.status === "failed") return job;
  const el = now() - _t0;
  if (el < 1) return { ...job, status: "queued", message: "waiting for the GPU (sample)" };
  const p = Math.min(1, (el - 1) / TRAIN_SECONDS);
  const epochs = Number(job.request.epochs ?? 30);
  if (p < 1) {
    return { ...job, status: "running", started_at: _t0 + 1, progress: Math.round(p * 0.95 * 1000) / 1000,
      message: p > 0.93 ? "exporting ONNX (sample)" : `epoch ${Math.max(1, Math.ceil(p * epochs))}/${epochs} (sample)` };
  }
  const ds = db.datasets.get(String(job.request.dataset_id))!;
  const art: ModelArtifact = {
    id: id("model"), kind: "trained", dataset_id: ds.id, variant: ds.variant, base_model: String(job.request.base_model ?? "yolo11n.pt"),
    classes: ds.classes, weights: "/data/runs/SAMPLE/weights/best.pt", onnx: "/data/runs/SAMPLE/weights/best.onnx",
    metrics: ds.variant === "real_only" ? { precision: 0.91, recall: 0.9, map50: 0.88 } : { precision: 0.93, recall: 0.89, map50: 0.89 },
    train_seconds: TRAIN_SECONDS, epochs, imgsz: 640, device: "SAMPLE (no GPU used)", created_at: now(),
  };
  db.models.set(art.id, art);
  Object.assign(j, { status: "succeeded", progress: 1, result_id: art.id, finished_at: now(), message: `done in ${TRAIN_SECONDS}s (sample)` });
  return jobView(j);
}

function remapClasses(e: ModelEvaluation, classes: string[]): ModelEvaluation {
  const from = ["no_goggles", "goggles"];
  const map = new Map(from.map((c, i) => [c, classes[i] ?? c]));
  const m = (c: string) => map.get(c) ?? c;
  const rekey = <T,>(o: Record<string, T>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [m(k), v]));
  return {
    ...e,
    test: { ...e.test, classes: e.test.classes.map(m), gt_per_class: rekey(e.test.gt_per_class) },
    per_class: rekey(e.per_class),
    failures: e.failures.map((f) => ({ ...f, cls: m(f.cls), predicted_cls: f.predicted_cls ? m(f.predicted_cls) : f.predicted_cls })),
    readiness: { ...e.readiness, data_criteria: e.readiness.data_criteria.map((c) => ({ ...c, name: c.name.replace(/'([^']+)'/, (_, k) => `'${m(k)}'`) })) },
  };
}

export function createMockApi(): ZaluApi {
  const notFound = (kind: string, x: string) => new ApiError(404, `${kind} ${x} not found (sample store)`, kind);
  return {
    mode: "mock",
    async health() {
      return { ok: true, cuda: false, device: "SAMPLE mock backend", hooks: { task_engine: false, data_engine: false, evaluation_engine: true } };
    },
    async createTask(prompt) {
      await wait(600);
      const t: TaskSpec = { id: id("task"), prompt, task_type: "object_detection", ...mockPlan(prompt),
        notes: "SAMPLE task spec from a rule-based mock planner (Task Engine not connected).", created_at: now() };
      db.tasks.set(t.id, t);
      return t;
    },
    async createDataset(taskId, files, meta) {
      const task = db.tasks.get(taskId);
      if (!task) throw notFound("task", taskId);
      const frames = await extractFrames(files);
      await wait(500);
      const base = { task_id: taskId, classes: task.classes, source_ids: [meta.session_id], created_at: now() };
      const ds: DatasetManifest = { id: id("ds"), root: `/data/datasets/SAMPLE/${meta.session_id}`, data_yaml: "/data/datasets/SAMPLE/data.yaml",
        variant: "real_only", counts: meta.role === "test" ? { test: frames.length } : { train: Math.round(frames.length * 0.8), val: frames.length - Math.round(frames.length * 0.8) }, ...base };
      db.datasets.set(ds.id, ds);
      db.frames.set(ds.id, frames);
      if (meta.role === "train") {  // Kirill's fal step would add rare synthetic cases to a sibling dataset
        const syn: DatasetManifest = { ...ds, id: id("ds"), variant: "real_plus_synthetic", counts: { ...ds.counts, synthetic: 24 } };
        db.datasets.set(syn.id, syn);
      }
      return ds;
    },
    async listDatasets() {
      return [...db.datasets.values()].sort((a, b) => b.created_at - a.created_at);
    },
    async listFrames(dsId, limit = 60) {
      if (!db.datasets.has(dsId)) throw notFound("dataset", dsId);
      return (db.frames.get(dsId) ?? []).slice(0, limit);
    },
    async getLabels(dsId) {
      const ds = db.datasets.get(dsId);
      if (!ds) throw notFound("dataset", dsId);
      if (!db.labels.has(dsId)) {
        const out: Annotation[] = [];
        for (const f of db.frames.get(dsId) ?? []) {
          const r = hash(f.name);
          const truth = SAMPLE_TRUTH.get(f.name);
          if (truth) {  // generated sample frames: label the drawn figures, with a few deliberate auto-label mistakes
            truth.forEach((t) => {
              const roll = r();
              if (roll < 0.08) return;  // missed by the auto-labeller
              const cls = roll < 0.18 ? ds.classes[(t.clsIndex + 1) % ds.classes.length] : ds.classes[t.clsIndex % ds.classes.length];
              const j = () => (r() - 0.5) * 24;
              out.push({ image: f.name, cls, box: [t.box[0] + j(), t.box[1] + j(), t.box[2] + j(), t.box[3] + j()],
                score: Math.round((roll < 0.18 ? 0.3 + r() * 0.2 : 0.55 + r() * 0.4) * 100) / 100, source: "auto" });
            });
            continue;
          }
          const n = r() < 0.2 ? 0 : 1 + Math.floor(r() * 2);
          for (let k = 0; k < n; k++) {
            const s = (0.18 + r() * 0.22) * Math.min(f.width, f.height);
            const x = r() * (f.width - s), y = r() * (f.height - s * 1.2);
            out.push({ image: f.name, cls: ds.classes[Math.floor(r() * ds.classes.length)], box: [x, y, x + s, y + s * 1.2],
              score: Math.round((0.35 + r() * 0.6) * 100) / 100, source: "auto" });
          }
        }
        db.labels.set(dsId, out);
      }
      await wait(300);
      return structuredClone(db.labels.get(dsId)!);
    },
    async saveLabels(dsId, labels) {
      db.labels.set(dsId, structuredClone(labels));
      return { saved: labels.length, sample: true };
    },
    async startTraining(req) {
      if (!db.datasets.has(req.dataset_id)) throw notFound("dataset", req.dataset_id);
      const j = { id: id("job"), kind: "training" as const, status: "queued" as const, request: { base_model: "yolo11n.pt", epochs: 30, ...req },
        progress: 0, message: "queued (sample)", error: null, result_id: null, created_at: now(), started_at: null, finished_at: null, _t0: now() };
      db.jobs.set(j.id, j);
      return jobView(j);
    },
    async getJob(jobId) {
      const j = db.jobs.get(jobId);
      if (!j) throw notFound("job", jobId);
      return jobView(j);
    },
    async createZeroShot(classes) {
      const art: ModelArtifact = { id: id("model"), kind: "zero_shot", dataset_id: null, variant: null, base_model: "yoloe-11l-seg.pt",
        classes, weights: null, onnx: null, metrics: {}, train_seconds: null, epochs: null, imgsz: null, device: "SAMPLE", created_at: now() };
      db.models.set(art.id, art);
      return art;
    },
    async listModels() {
      return [...db.models.values()];
    },
    async getModel(mid) {
      const m = db.models.get(mid);
      if (!m) throw notFound("model", mid);
      return m;
    },
    async createEvaluation(req) {
      await wait(1200);
      const sample = sampleReport as unknown as EvaluationReport;
      const entries = Object.values(sample.per_model);
      const pick = (m?: ModelArtifact) =>
        entries.find((e) => m?.kind === "zero_shot" ? e.model?.kind === "zero_shot" : e.model?.variant === (m?.variant ?? "real_only")) ?? entries[1];
      const per_model: Record<string, ModelEvaluation> = {};
      for (const mid of req.model_ids) {
        const m = db.models.get(mid);
        const e = remapClasses(structuredClone(pick(m)), m?.classes ?? []);
        e.model = { ...e.model!, id: mid };
        if (e.delta_vs_baseline) e.delta_vs_baseline.baseline_id = req.model_ids.find((x) => db.models.get(x)?.kind === "zero_shot") ?? req.model_ids[0];
        per_model[mid] = e;
      }
      const rep: EvaluationReport = { ...structuredClone(sample), id: id("rep"), request: req, per_model, created_at: now() };
      db.reports.set(rep.id, rep);
      return rep;
    },
    async getReport(rid) {
      const r = db.reports.get(rid);
      if (!r) throw notFound("report", rid);
      return r;
    },
    async inference(modelId, _image, opts = {}) {
      const m = db.models.get(modelId);
      const classes = m?.classes ?? ["object"];
      const t = Date.now() / 1000;
      const w = 1280, h = 720;
      const dets = [0, 1].slice(0, 1 + (Math.floor(t / 4) % 2)).map((k) => {
        const cx = w * (0.3 + 0.4 * k) + Math.sin(t * 0.8 + k) * 120, cy = h * 0.5 + Math.cos(t * 0.6 + k) * 60;
        return { cls: classes[k % classes.length], score: Math.round((0.7 + 0.25 * Math.sin(t + k)) * 100) / 100,
          box: [cx - 110, cy - 140, cx + 110, cy + 140] as [number, number, number, number] };
      }).filter((d) => d.score >= (opts.conf ?? 0.25));
      await wait(30);
      return { id: id("inf"), model_id: modelId, width: w, height: h, detections: dets, latency_ms: 4 + Math.round(Math.random() * 15) / 10,
        events: opts.targetEvent && dets.some((d) => d.cls === opts.targetEvent) ? [opts.targetEvent] : [] };
    },
  };
}
