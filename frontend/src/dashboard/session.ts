import type { MetricDefinition, MetricReading } from "../../../shared/dashboard_spec/dashboard-spec";

export interface FrameResult {
  width: number;
  height: number;
  detections: { cls: string; box: number[]; score: number }[];
  events: string[];
  latency_ms: number;
}

export interface FrameSample {
  at: number;
  result: FrameResult;
  eventsEnabled: boolean;
}

export function readMetric(metric: MetricDefinition, frame: FrameResult | undefined, classes: readonly string[]): MetricReading {
  if (!frame) return { status: "unavailable", value: null };
  if (metric.operation === "inference_latency_ms") {
    return Number.isFinite(frame.latency_ms) && frame.latency_ms >= 0
      ? { status: "available", value: frame.latency_ms } : { status: "unavailable", value: null };
  }
  if (!Array.isArray(frame.detections)) return { status: "unavailable", value: null };
  if (metric.operation === "class_count" && !classes.includes(metric.class_name)) {
    return { status: "unavailable", value: null };
  }
  const value = metric.operation === "class_count"
    ? frame.detections.filter((detection) => detection.cls === metric.class_name).length
    : frame.detections.length;
  return { status: "available", value };
}

export function sampleResult(classes: readonly string[], index: number, target?: string): FrameResult {
  const detections = index % 5 === 4 ? [] : Array.from({ length: 1 + index % 3 }, (_, i) => ({
    cls: classes[(index + i) % classes.length], score: 0.92 - i * 0.08,
    box: [90 + i * 180, 105 + i * 16, 235 + i * 180, 355 - i * 10],
  })).filter((detection) => !!detection.cls);
  return {
    width: 800, height: 450, detections,
    events: target && detections.some((detection) => detection.cls === target) ? [target] : [],
    latency_ms: 18 + index % 8,
  };
}
