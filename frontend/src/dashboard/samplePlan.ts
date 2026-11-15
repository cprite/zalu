import type { DashboardSpec } from "../../../shared/dashboard_spec/dashboard-spec";
import parking from "../../../shared/dashboard_spec/examples/parking-vehicles.json";
import safety from "../../../shared/dashboard_spec/examples/workplace-safety.json";

export interface DashboardPlan {
  id: string;
  spec: DashboardSpec;
  ai_generated: boolean;
  generator: string;
  limitations: string[];
  clarification_questions: string[];
}

export function samplePlan(prompt: string, classes: readonly string[], target?: string | null): DashboardPlan {
  const spec = structuredClone(/park|vehicle|traffic|\bcars?\b/i.test(prompt) ? parking : safety) as DashboardSpec;
  spec.title = prompt.trim().replace(/\s+/g, " ").slice(0, 100) || "Camera dashboard";
  spec.description = "Explore a simulated inference session for your task.";
  const primary = classes.includes(target ?? "") ? target : classes[0];
  for (const metric of spec.metrics) {
    if (metric.operation !== "class_count") continue;
    if (primary) {
      metric.class_name = primary;
      metric.title = `${primary.replaceAll("_", " ")} per frame`;
    } else {
      Object.assign(metric, { operation: "detection_count", title: "Detections per frame" });
      delete (metric as { class_name?: string }).class_name;
    }
  }
  for (const widget of spec.widgets) {
    if (widget.type === "kpi" || widget.type === "time_series") {
      const id = widget.config.metric_id;
      widget.title = spec.metrics.find((metric) => metric.id === id)!.title;
    } else if (widget.type === "live_preview") widget.title = "Camera preview";
    else if (widget.type === "class_distribution") widget.title = "Detections by class";
    else if (widget.type === "detection_table" && widget.config.rows === "detections") widget.title = "Latest detections";
  }
  return { id: "sample", spec, ai_generated: false, generator: "sample:fixture", limitations: [], clarification_questions: [] };
}
