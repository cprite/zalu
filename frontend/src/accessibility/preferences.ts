// Integration contract types for the Accessibility section of ZALU Settings (Settings UI is owned by the
// design-system agent; this file only defines the shape + OS-preference resolution, it renders nothing).
// Every preference must change real CSS/behaviour when on. No cosmetic toggles; no "screen reader mode".

export type TriState = "system" | "on" | "off";

export interface AccessibilityPreferences {
  /** Follows `prefers-contrast: more` / `forced-colors` when "system". */
  highContrast: TriState;
  /** Follows `prefers-reduced-motion: reduce` when "system". */
  reducedMotion: TriState;
  /** Thicker, always-visible focus ring (3px, >=3:1 against both adjacent colours). No OS equivalent. */
  strongFocus: boolean;
  /** Show caption tracks by default WHEN a video has one. Never implies captions exist. */
  captionsWhenAvailable: boolean;
  // Larger Text is intentionally NOT here: it belongs to the global font-scale setting (do not duplicate).
}

export const DEFAULT_A11Y_PREFERENCES: AccessibilityPreferences = {
  highContrast: "system", reducedMotion: "system", strongFocus: false, captionsWhenAvailable: true,
};

export interface ResolvedAccessibility {
  highContrast: boolean;
  reducedMotion: boolean;
  strongFocus: boolean;
  captionsWhenAvailable: boolean;
}

function media(query: string): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(query).matches;
}

function tri(v: TriState, system: () => boolean): boolean {
  return v === "on" ? true : v === "off" ? false : system();
}

/** Resolve "system" against the OS media queries. Safe during SSR (system => false). */
export function resolveAccessibility(p: AccessibilityPreferences = DEFAULT_A11Y_PREFERENCES): ResolvedAccessibility {
  return {
    highContrast: tri(p.highContrast, () => media("(prefers-contrast: more)") || media("(forced-colors: active)")),
    reducedMotion: tri(p.reducedMotion, () => media("(prefers-reduced-motion: reduce)")),
    strongFocus: p.strongFocus,
    captionsWhenAvailable: p.captionsWhenAvailable,
  };
}

/** Data attributes for <html>, so CSS can target `[data-a11y-contrast="more"]` etc. */
export function accessibilityDataAttributes(r: ResolvedAccessibility): Record<string, string> {
  return {
    "data-a11y-contrast": r.highContrast ? "more" : "normal",
    "data-a11y-motion": r.reducedMotion ? "reduce" : "full",
    "data-a11y-focus": r.strongFocus ? "strong" : "normal",
  };
}
