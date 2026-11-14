// Smart Data Coach: rule-based guidance computed from the user's actual frames and labels.
// Stands in until Kirill's Data Engine returns suggestions; then merge both lists (source tells them apart).
import type { Frame } from "./frames";
import type { Annotation, TaskSpec } from "./types";

export interface Tip {
  level: "warn" | "tip" | "ok";
  title: string;
  detail: string;
  source: "rules" | "data_engine";
}

export const LOW_LIGHT = 70;      // mean luma below this = low-light frame
export const MIN_FRAMES = 40;
export const MIN_PER_CLASS = 15;
export const MIN_NEGATIVE_SHARE = 0.15;

export function coach(task: TaskSpec, frames: Frame[], labels: Annotation[], sessions: string[]): Tip[] {
  const tips: Tip[] = [];
  const r = (level: Tip["level"], title: string, detail: string) => tips.push({ level, title, detail, source: "rules" });
  if (!frames.length) return tips;

  if (frames.length < MIN_FRAMES) {
    r("warn", "Add more footage", `${frames.length} frames so far; aim for at least ${MIN_FRAMES} varied frames.`);
  }
  const labelled = new Set(labels.map((l) => l.image));
  const empty = frames.filter((f) => !labelled.has(f.name)).length;
  if (empty / frames.length < MIN_NEGATIVE_SHARE) {
    const look = task.negative_classes.length ? ` and look-alikes (${task.negative_classes.join(", ")})` : "";
    r("warn", "Need more negative examples", `Only ${empty} of ${frames.length} frames have nothing to detect. Record empty scenes${look} so the model learns what NOT to flag.`);
  }
  for (const c of task.classes) {
    const n = labels.filter((l) => l.cls === c).length;
    if (n < MIN_PER_CLASS) r("warn", `Few examples of “${c}”`, `${n} labelled so far; aim for ${MIN_PER_CLASS}+.`);
  }
  const dark = frames.filter((f) => f.brightness < LOW_LIGHT).length;
  if (dark === 0) {
    r("tip", "Include low-light conditions", "All frames are well lit. If the camera runs in the evening or with lights off, record a short clip in those conditions.");
  }
  if (sessions.length < 2) {
    r("tip", "Capture another angle", "All footage comes from one recording. A second position or angle makes the model far more robust.");
  }
  const lowConf = labels.filter((l) => l.source === "auto" && (l.score ?? 1) < 0.5).length;
  if (lowConf > 0) r("tip", "Check uncertain labels", `${lowConf} auto-labels have low confidence (dashed boxes). Confirm or fix them.`);
  if (!tips.some((t) => t.level === "warn")) r("ok", "Dataset looks balanced", "Good to train. Keep a separate recording for the final test.");
  return tips;
}
