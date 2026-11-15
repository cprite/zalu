import { strToU8, zipSync } from "fflate";
import type { DashboardSpec } from "../../../shared/dashboard_spec/dashboard-spec";
import files from "./generated/export-files.json";

export function downloadDashboard(spec: DashboardSpec, classes: readonly string[], targetEvent: string, confidence: number) {
  const project = {
    ...files,
    "src/dashboard/dashboard.json": JSON.stringify(spec, null, 2),
    "src/dashboard/sample-data.json": JSON.stringify({ classes, target_event: targetEvent, confidence }, null, 2),
  };
  const zip = zipSync(Object.fromEntries(Object.entries(project).map(([name, content]) => [name, strToU8(content)])));
  const blob = new Blob([new Uint8Array(zip)], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "zalu-dashboard.zip";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
