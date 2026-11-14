// Plain-language explanations shown to non-ML users. Keys match backend/evaluation_engine/matching.py.
export const BRAND = "ZALU";

export const REASONS: Record<string, Record<string, { title: string; fix: string }>> = {
  false_positive: {
    wrong_class: { title: "Confused with another class", fix: "Add more examples of both classes side by side." },
    duplicate: { title: "Counted the same object twice", fix: "Usually harmless for alerts; raise the confidence threshold if it matters." },
    poor_localization: { title: "Right object, box in the wrong place", fix: "Check that labels tightly fit the objects." },
    background: { title: "False alarm on the background", fix: "Add negative examples: empty scenes and look-alikes." },
  },
  false_negative: {
    low_confidence: { title: "Seen, but not confident enough", fix: "Add similar examples, or lower the threshold if misses are costly." },
    wrong_class: { title: "Found, but called the wrong class", fix: "Add more examples that show the difference between classes." },
    poor_localization: { title: "Found, but the box was too far off", fix: "Check labels for this kind of view." },
    missed: { title: "Not detected at all", fix: "Capture more footage in this condition (angle, distance, lighting)." },
  },
};

export const VERDICT: Record<string, { label: string; icon: string }> = {
  ready: { label: "Ready to deploy", icon: "✓" },
  not_ready: { label: "Not ready yet", icon: "✕" },
  insufficient_data: { label: "Promising, but not proven", icon: "!" },
  invalid_test_set: { label: "Test invalid: data leakage", icon: "⚠" },
};

export const METRIC_HELP = {
  precision: "Of all the alerts the model raised, how many were real.",
  recall: "Of all the real objects, how many the model found.",
  f1: "One number balancing precision and recall.",
  fp: "False alarms: the model flagged something that was not there.",
  fn: "Misses: a real object the model did not flag.",
};

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
export const variantLabel = (kind?: string, variant?: string | null) =>
  kind === "zero_shot" ? "Zero-shot baseline (no training)" : variant === "real_plus_synthetic" ? "Trained · real + synthetic" : "Trained · real footage only";
