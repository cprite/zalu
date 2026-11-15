// Screen-reader text for Video Search results. Pure functions, no React, no imports from other ZALU files.
// Field names mirror backend/video_search/schemas.py (SearchResult / Search). Times are milliseconds from the
// start of the recording. The score is a CLIP cosine similarity, NOT a probability and NOT an identity match,
// so the wording never says "the same person" unless the backend explicitly says so.

export type CrossCameraIdentityStatus = "not_assessed" | "possible" | "confirmed" | "rejected" | (string & {});

export interface VideoSearchResultLike {
  tracklet_id: string;
  camera_id: string;
  recording_id?: string;
  start_ms: number;
  end_ms: number;
  visible_duration_ms?: number;
  start_utc?: string | null;
  query_score?: number;
  score_type?: string;
  cross_camera_identity_status?: CrossCameraIdentityStatus;
}

export interface VideoSearchResponseLike {
  status?: string; // "completed" | "running" | "failed" ...
  query?: string;
  total_matches?: number;
  results: VideoSearchResultLike[];
}

const SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"];

/** "three" for 3, "21" for 21. Words read more naturally for small counts. */
export function countWord(n: number): string {
  return Number.isInteger(n) && n >= 0 && n < SMALL.length ? SMALL[n] : String(n);
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** "camera_01" -> "Camera 1", "cam-2" -> "Camera 2", "Loading bay" -> "Loading bay". */
export function cameraLabel(cameraId: string): string {
  const m = /^(?:camera|cam)[\s_-]*0*(\d+)$/i.exec(cameraId.trim());
  if (m) return `Camera ${m[1]}`;
  return cap(cameraId.replace(/[_-]+/g, " ").trim());
}

/** 45000 -> "00:45", 3725000 -> "1:02:05". Visible clock format (also what a sighted user sees). */
export function clockTimestamp(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  const pad = (v: number) => String(v).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

/** 45000 -> "45 seconds into the recording", 84000 -> "1 minute 24 seconds into the recording". */
export function spokenOffset(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  const parts: string[] = [];
  if (h) parts.push(`${h} hour${h === 1 ? "" : "s"}`);
  if (m) parts.push(`${m} minute${m === 1 ? "" : "s"}`);
  if (s || parts.length === 0) parts.push(`${s} second${s === 1 ? "" : "s"}`);
  return `${parts.join(" ")} into the recording`;
}

/** Rounded, explicitly approximate duration. Sampling is ~2 fps, so sub-second precision would be false. */
export function approxDuration(ms: number): string {
  if (!(ms > 0)) return "visible in a single sampled frame";
  const s = Math.round(ms / 1000);
  if (s < 1) return "visible for less than a second";
  if (s < 60) return `visible for approximately ${s} second${s === 1 ? "" : "s"}`;
  const m = Math.round(s / 60);
  return `visible for approximately ${m} minute${m === 1 ? "" : "s"}`;
}

export function identityText(status: CrossCameraIdentityStatus | undefined): string {
  switch (status) {
    case "confirmed": return "Same person across cameras: confirmed by the backend.";
    case "possible": return "Same person across cameras: possible, not confirmed.";
    case "rejected": return "Same person across cameras: unlikely.";
    default: return "Same person across cameras: not assessed.";
  }
}

export interface ResultTextOptions {
  /** Include the raw similarity score, labelled as such. Off by default: it is easy to mistake for confidence. */
  includeScore?: boolean;
}

/** One result, e.g. "Result one: Camera 2, timestamp 00:45, visible for approximately 12 seconds." */
export function describeResult(r: VideoSearchResultLike, index: number, opts: ResultTextOptions = {}): string {
  const dur = r.visible_duration_ms ?? Math.max(0, r.end_ms - r.start_ms);
  let text = `Result ${countWord(index + 1)}: ${cameraLabel(r.camera_id)}, timestamp ${clockTimestamp(r.start_ms)}, ${approxDuration(dur)}.`;
  if (opts.includeScore && typeof r.query_score === "number") {
    text += ` Similarity score ${r.query_score.toFixed(2)} (${r.score_type === "clip_cosine_similarity" ? "text-to-image similarity, not a probability" : r.score_type ?? "relative"}).`;
  }
  return text;
}

/** Accessible name for a "seek to result" button: says what will happen, in words. */
export function seekLabel(r: VideoSearchResultLike, index: number): string {
  return `Play result ${index + 1}: ${cameraLabel(r.camera_id)} from ${spokenOffset(r.start_ms)}`;
}

export interface SummaryOptions extends ResultTextOptions {
  /** How many results to spell out after the count. Default 3; the rest stay in the navigable list. */
  maxResults?: number;
}

/**
 * Whole-search summary for a polite live region. Never claims completion unless status is "completed"
 * (or absent, when the caller only has a finished response).
 */
export function describeSearch(res: VideoSearchResponseLike, opts: SummaryOptions = {}): string {
  const status = res.status ?? "completed";
  if (status === "failed") return "Search failed. No results to show.";
  if (status !== "completed") return "Searching recordings.";
  const n = res.results.length;
  if (n === 0) return "Search complete. No matching appearances.";
  const total = res.total_matches ?? n;
  const head = `Search complete. ${cap(countWord(total))} matching appearance${total === 1 ? "" : "s"}${total > n ? `, showing ${countWord(n)}` : ""}.`;
  const shown = res.results.slice(0, opts.maxResults ?? 3).map((r, i) => describeResult(r, i, opts));
  const statuses = new Set(res.results.map((r) => r.cross_camera_identity_status ?? "not_assessed"));
  const identity = statuses.size === 1 ? identityText([...statuses][0]) : "Same person across cameras: varies by result.";
  return [head, ...shown, identity].join(" ");
}
