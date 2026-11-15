"use client";
// Polite / assertive live-region announcer. Self-contained: mount <LiveAnnouncerProvider> once near the root
// and call useAnnounce() anywhere below it.
//
// Rules it enforces so screen readers are not flooded:
//  - debounced: rapid calls within `debounceMs` collapse into the LAST message (no per-frame / per-% spam);
//  - de-duplicated: the same text is not re-announced within `repeatMs` (e.g. a re-render of "Training complete");
//  - the region is always in the DOM (live regions added at the same time as their text are often missed);
//  - text is cleared then set, so an identical message after `repeatMs` is announced again.
// Only announce what the backend has confirmed. Callers own that rule; this helper only delivers text.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

export type Politeness = "polite" | "assertive";
export type Announce = (message: string, politeness?: Politeness) => void;

const noop: Announce = () => {};
const AnnounceContext = createContext<Announce>(noop);

/** Visually hidden but readable by assistive technology (the standard "sr-only" pattern). */
export const visuallyHidden: React.CSSProperties = {
  position: "absolute", width: 1, height: 1, padding: 0, margin: -1, overflow: "hidden",
  clip: "rect(0 0 0 0)", clipPath: "inset(50%)", whiteSpace: "nowrap", border: 0,
};

export interface LiveAnnouncerProps {
  children?: ReactNode;
  debounceMs?: number;
  repeatMs?: number;
}

export function LiveAnnouncerProvider({ children, debounceMs = 700, repeatMs = 4000 }: LiveAnnouncerProps) {
  const [polite, setPolite] = useState("");
  const [assertive, setAssertive] = useState("");
  const pending = useRef<{ text: string; politeness: Politeness } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const last = useRef<{ text: string; at: number }>({ text: "", at: 0 });

  const flush = useCallback(() => {
    timer.current = null;
    const p = pending.current;
    pending.current = null;
    if (!p) return;
    const now = Date.now();
    if (p.text === last.current.text && now - last.current.at < repeatMs) return;
    last.current = { text: p.text, at: now };
    const set = p.politeness === "assertive" ? setAssertive : setPolite;
    set("");
    // next frame: changing from "" to text is what triggers the announcement
    requestAnimationFrame(() => set(p.text));
  }, [repeatMs]);

  const announce = useCallback<Announce>((message, politeness = "polite") => {
    const text = message.trim();
    if (!text) return;
    // an assertive message is never downgraded by a later polite one inside the same window
    if (pending.current?.politeness === "assertive" && politeness === "polite") return;
    pending.current = { text, politeness };
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(flush, debounceMs);
  }, [debounceMs, flush]);

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  return (
    <AnnounceContext.Provider value={announce}>
      {children}
      <div style={visuallyHidden} role="status" aria-live="polite" aria-atomic="true" data-zalu-announcer="polite">{polite}</div>
      <div style={visuallyHidden} role="alert" aria-live="assertive" aria-atomic="true" data-zalu-announcer="assertive">{assertive}</div>
    </AnnounceContext.Provider>
  );
}

/** Returns announce(message, politeness?). A no-op outside the provider, so components never crash. */
export function useAnnounce(): Announce {
  return useContext(AnnounceContext);
}

/**
 * Announce a value only when it changes to something meaningful, e.g. a job status:
 *   useAnnounceOnChange(job.status === "completed" ? "Training completed." : null)
 * Pass null/undefined to stay silent. Progress percentages should be bucketed by the caller (see progressMilestone).
 */
export function useAnnounceOnChange(message: string | null | undefined, politeness: Politeness = "polite") {
  const announce = useAnnounce();
  const prev = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    if (message && message !== prev.current) announce(message, politeness);
    prev.current = message;
  }, [message, politeness, announce]);
}

/** Bucket a 0-100 progress value so announcements happen at most every `step` percent (default 25). */
export function progressMilestone(percent: number, step = 25): number {
  if (!Number.isFinite(percent)) return 0;
  return Math.min(100, Math.max(0, Math.floor(percent / step) * step));
}

/** Convenience: memoised label for a progressbar's aria-valuetext, e.g. "Epoch 12 of 30, 40 percent". */
export function useProgressText(percent: number, detail?: string): string {
  return useMemo(() => `${detail ? detail + ", " : ""}${Math.round(percent)} percent`, [percent, detail]);
}
