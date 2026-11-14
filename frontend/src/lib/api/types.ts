import type { Frame } from "../frames";
import type {
  Annotation, DatasetManifest, EvaluationReport, EvaluationRequest, Health, InferenceResult, ModelArtifact,
  TaskSpec, TrainingJob, TrainingRequest, UploadMeta,
} from "../types";

/** The only way UI components talk to the backend. Mock and live implementations are interchangeable. */
export interface ZaluApi {
  readonly mode: "mock" | "live";
  health(): Promise<Health>;
  createTask(prompt: string): Promise<TaskSpec>;
  createDataset(taskId: string, files: File[], meta: UploadMeta): Promise<DatasetManifest>;
  listDatasets(): Promise<DatasetManifest[]>;
  /** Auto-labels for the review screen (live: GET /api/datasets/{id}/annotations). Annotation.image = frame id. */
  getLabels(datasetId: string): Promise<Annotation[]>;
  /** Frames to review (live: frames the server extracted, loaded through GET /api/files). */
  listFrames(datasetId: string, limit?: number): Promise<Frame[]>;
  saveLabels(datasetId: string, labels: Annotation[]): Promise<unknown>;
  startTraining(req: TrainingRequest): Promise<TrainingJob>;
  getJob(jobId: string): Promise<TrainingJob>;
  createZeroShot(classes: string[]): Promise<ModelArtifact>;
  listModels(): Promise<ModelArtifact[]>;
  getModel(modelId: string): Promise<ModelArtifact>;
  createEvaluation(req: EvaluationRequest): Promise<EvaluationReport>;
  getReport(reportId: string): Promise<EvaluationReport>;
  inference(modelId: string, image: Blob, opts?: { conf?: number; targetEvent?: string | null }): Promise<InferenceResult>;
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public path: string) {
    super(message);
  }
  /** 501 = a teammate's module is not merged yet; the UI explains this instead of failing. */
  get notImplemented() {
    return this.status === 501 || (this.status === 404 && this.path.endsWith("/annotations"));
  }
}
