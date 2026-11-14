/** DashboardSpec v1. Runtime validation lives in dashboard-spec.v1.schema.json and validation.py. */

export interface InferenceSessionSource {
  id: string;
  type: "inference_session";
}

interface Binding {
  id: string;
  title: string;
  source_id: string;
}

export type MetricDefinition = Binding & (
  | { operation: "class_count"; class_name: string }
  | { operation: "detection_count" | "inference_latency_ms"; class_name?: never }
);

interface WidgetBinding extends Binding {
  span: number;
}

export type Widget = WidgetBinding & (
  | { type: "live_preview"; config: { show_boxes: boolean } }
  | { type: "kpi"; config: { metric_id: string } }
  | { type: "time_series"; config: { metric_id: string; window_seconds: number } }
  | { type: "detection_table"; config: { rows: "detections" | "events"; max_rows: number } }
  | { type: "class_distribution"; config: Record<string, never> }
);

export interface DashboardSpec {
  version: "1";
  title: string;
  description: string;
  layout: { columns: number };
  data_sources: [InferenceSessionSource];
  metrics: MetricDefinition[];
  widgets: Widget[];
}

export type MetricReading =
  | { status: "available"; value: number }
  | { status: "unavailable"; value: null };
