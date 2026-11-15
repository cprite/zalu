"use client";
// Display preferences: theme, motion intensity and UI scale. One small external store, persisted in localStorage,
// mirrored onto <html> as data attributes so CSS tokens switch without React re-rendering the tree.
// The same logic runs before first paint as an inline script (PREFS_SCRIPT, injected by layout.tsx), so there is no
// flash of the wrong theme or scale. Approach follows next-themes (inline script, system listener, transitions paused
// while the palette swaps) without its client-side <script> render, which React 19 warns about.
import { useSyncExternalStore } from "react";
import { PREFS_KEY as KEY } from "./prefs-script";

export type ThemePref = "light" | "dark" | "system";
export type ScalePref = "compact" | "default" | "large";
/** 0 Reduced · 1 Subtle · 2 Balanced · 3 Expressive */
export type MotionLevel = 0 | 1 | 2 | 3;
export interface Prefs { theme: ThemePref; motion: MotionLevel; scale: ScalePref }

export const MOTION_LABELS = ["Reduced", "Subtle", "Balanced", "Expressive"] as const;
export const DEFAULT_PREFS: Prefs = { theme: "system", motion: 2, scale: "default" };

// ---------------------------------------------------------------- store

let prefs: Prefs = DEFAULT_PREFS;
let loaded = false;
const listeners = new Set<() => void>();

function sanitize(p: Partial<Prefs> | null | undefined): Prefs {
  const theme = p?.theme === "light" || p?.theme === "dark" ? p.theme : "system";
  const scale = p?.scale === "compact" || p?.scale === "large" ? p.scale : "default";
  const motion = ([0, 1, 2, 3] as const).includes(p?.motion as MotionLevel) ? (p!.motion as MotionLevel) : 2;
  return { theme, scale, motion };
}

function read(): Prefs {
  try { return sanitize(JSON.parse(localStorage.getItem(KEY) ?? "{}")); } catch { return DEFAULT_PREFS; }
}

const mq = (q: string) => (typeof window === "undefined" ? null : window.matchMedia(q));
export const systemDark = () => !!mq("(prefers-color-scheme: dark)")?.matches;
export const systemReduced = () => !!mq("(prefers-reduced-motion: reduce)")?.matches;

/** The level actually in force: the OS reduced-motion setting always wins. */
export function effectiveMotion(p: Prefs = prefs): MotionLevel {
  return systemReduced() ? 0 : p.motion;
}

/** Swap the palette with transitions paused for one frame so 150 ms colour tweens don't smear across the page. */
function pauseTransitions() {
  const css = document.createElement("style");
  css.appendChild(document.createTextNode("*,*::before,*::after{transition:none!important}"));
  document.head.appendChild(css);
  return () => { void window.getComputedStyle(document.body).opacity; setTimeout(() => css.remove(), 1); };
}

function apply(p: Prefs, prev?: Prefs) {
  const d = document.documentElement;
  const dark = p.theme === "dark" || (p.theme === "system" && systemDark());
  const theme = dark ? "dark" : "light";
  const resume = prev && d.dataset.theme !== theme ? pauseTransitions() : null;
  d.dataset.theme = theme;
  d.dataset.themePref = p.theme;
  d.dataset.scale = p.scale;
  d.dataset.motion = String(effectiveMotion(p));
  d.style.colorScheme = theme;
  resume?.();
}

function ensureLoaded() {
  if (loaded || typeof window === "undefined") return;
  loaded = true;
  prefs = read();
  const sync = () => { apply(prefs, prefs); listeners.forEach((l) => l()); };
  mq("(prefers-color-scheme: dark)")?.addEventListener("change", sync);
  mq("(prefers-reduced-motion: reduce)")?.addEventListener("change", sync);
  // another tab changed the preferences
  window.addEventListener("storage", (e) => { if (e.key === KEY) { const prev = prefs; prefs = read(); apply(prefs, prev); listeners.forEach((l) => l()); } });
}

export function setPrefs(patch: Partial<Prefs>) {
  ensureLoaded();
  const prev = prefs;
  prefs = sanitize({ ...prefs, ...patch });
  try { localStorage.setItem(KEY, JSON.stringify(prefs)); } catch { /* private mode: keep in memory */ }
  apply(prefs, prev);
  listeners.forEach((l) => l());
}

/** Current preferences without subscribing (used by motion tokens at animation time). */
export function getPrefs(): Prefs { ensureLoaded(); return prefs; }

function subscribe(cb: () => void) { ensureLoaded(); listeners.add(cb); return () => { listeners.delete(cb); }; }

// Snapshot includes the system state so components re-render when the OS setting flips.
let snap: { prefs: Prefs; motion: MotionLevel; dark: boolean; sysReduced: boolean } | null = null;
function snapshot() {
  ensureLoaded();
  const motion = effectiveMotion(prefs), dark = prefs.theme === "dark" || (prefs.theme === "system" && systemDark()), sysReduced = systemReduced();
  if (!snap || snap.prefs !== prefs || snap.motion !== motion || snap.dark !== dark || snap.sysReduced !== sysReduced) snap = { prefs, motion, dark, sysReduced };
  return snap;
}
const SERVER = { prefs: DEFAULT_PREFS, motion: 2 as MotionLevel, dark: false, sysReduced: false };

/** Preferences plus derived state: the effective motion level, whether dark is showing, whether the OS forces reduced motion. */
export function usePrefs() {
  return useSyncExternalStore(subscribe, snapshot, () => SERVER);
}
