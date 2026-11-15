"use client";
// Motion tokens for ZALU. One easing curve and three durations (150 / 220 / 320 ms at the default level) everywhere.
// Every token is read lazily (getters), so the global Motion Intensity preference (lib/prefs) changes durations, travel,
// springs and loading loops the next time anything animates, without touching the screens that use the tokens.
// <MotionConfig> in app/providers.tsx turns transforms into instant changes for Reduced and for people whose OS asks
// for less motion; the OS setting always wins over the in-app slider.
import { AnimatePresence, animate, motion, useInView, useMotionValue, useReducedMotion, useTransform, type Transition } from "motion/react";
import { useEffect, useRef, useSyncExternalStore, type ReactNode } from "react";
import { effectiveMotion, getPrefs, usePrefs, type MotionLevel } from "./prefs";

export const EASE = [0.2, 0, 0, 1] as const;           // standard "emphasized decelerate"

/**
 * Per-level dials. Reduced keeps short opacity fades only (no travel, no springs, no loops); Expressive adds a little
 * travel and a spring with a hint of follow-through, never bounce.
 */
export const LEVELS: Record<MotionLevel, { time: number; travel: number; stiffness: number; damping: number; press: number; loop: number }> = {
  0: { time: 0.6, travel: 0, stiffness: 900, damping: 80, press: 1, loop: 0 },
  1: { time: 0.8, travel: 0.5, stiffness: 600, damping: 52, press: 0.99, loop: 0.6 },
  2: { time: 1, travel: 1, stiffness: 520, damping: 44, press: 0.98, loop: 1 },
  3: { time: 1.2, travel: 1.5, stiffness: 400, damping: 30, press: 0.97, loop: 1.3 },
};
// Until hydration finishes the tokens stay at Balanced, so server HTML and the first client render agree
// (initial styles such as translateY are serialised). Providers flips this right after mount.
let hydrated = false;
export function markMotionHydrated() { hydrated = true; }
/** The dials in force right now. */
export const level = () => LEVELS[!hydrated ? 2 : effectiveMotion(getPrefs())];

const BASE = { fast: 0.15, base: 0.22, slow: 0.32 } as const;
type DurKey = keyof typeof BASE;
/** Durations in seconds, scaled by the motion level when read. */
export const DUR = {
  get fast() { return BASE.fast * level().time; },
  get base() { return BASE.base * level().time; },
  get slow() { return BASE.slow * level().time; },
};
/** Vertical travel in px for entrances, scaled by the motion level. */
const travel = (px: number) => px * level().travel;

/** Tween on the shared curve. The duration is a getter, so it resolves when the animation starts, not at import. */
export const t = (d: DurKey = "base", extra: Transition = {}): Transition => {
  const o = { ease: EASE, ...extra } as Transition;
  if (!("duration" in extra)) Object.defineProperty(o, "duration", { get: () => DUR[d], enumerable: true });
  return o;
};

/** Screen / section entrance: fade with a short rise. Spread it ({...enter}) at render time. */
export const enter = {
  initial: { opacity: 0, get y() { return travel(8); } },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, get y() { return travel(-4); } },
  transition: t("base"),
};

/** Container + child variants for the one orchestrated load moment (Describe hero). */
export const stagger = {
  hidden: {},
  show: { transition: { get staggerChildren() { return 0.05 * level().time; }, get delayChildren() { return 0.04 * level().time; } } },
};
export const rise = {
  hidden: { opacity: 0, get y() { return travel(8); } },
  show: { opacity: 1, y: 0, transition: t("slow") },
};

/** true when the effective level is Reduced (in-app choice or the OS setting). Prefer it to useReducedMotion(). */
export function useLessMotion(): boolean {
  const { motion: m } = usePrefs();
  const os = useReducedMotion();
  return m === 0 || !!os;
}

/** Height reveal for disclosures (Options, Compare models, See mistakes). */
export const reveal = {
  initial: { opacity: 0, height: 0 },
  animate: { opacity: 1, height: "auto" },
  exit: { opacity: 0, height: 0 },
  transition: t("base"),
};

/**
 * Animated number (adapted from motion-primitives `animated-number`, MIT). On first view it counts up from 0 to the
 * measured value; afterwards it tweens from the previous real value to the next one. Screen readers get the final
 * value immediately, and with reduced motion the number simply jumps. It never displays a value that was not passed in
 * except during that short tween.
 */
export function AnimatedNumber({ value, format, className, duration }: { value: number; format: (v: number) => string; className?: string; duration?: number }) {
  const reduce = useLessMotion();
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => format(v));
  const first = useRef(true);
  useEffect(() => {
    if (reduce) { mv.set(value); return; }
    // first appearance counts up from 0 to the measured value once it is visible; later updates tween from the last value
    if (first.current) {
      if (!inView) return;
      first.current = false;
      mv.set(0);
    }
    const c = animate(mv, value, { duration: duration ?? DUR.slow * 2, ease: EASE });
    return () => c.stop();
  }, [value, reduce, inView, mv, duration]);
  return (
    <span ref={ref} className={className}>
      <motion.span aria-hidden>{text}</motion.span>
      <span className="sr-only">{format(value)}</span>
    </span>
  );
}

/**
 * Status text with a moving sheen (adapted from motion-primitives `text-shimmer`, MIT) for long-running work.
 * Uses ZALU ink tokens; static text when reduced motion is on.
 */
export function TextShimmer({ children, className }: { children: string; className?: string }) {
  const reduce = useLessMotion();
  if (reduce) return <span className={className}>{children}</span>;
  return (
    <motion.span className={`shimmer ${className ?? ""}`}
      initial={{ backgroundPosition: "100% center" }} animate={{ backgroundPosition: "0% center" }}
      transition={{ repeat: Infinity, duration: 1.8 / Math.max(level().loop, 0.6), ease: "linear" }}>
      {children}
    </motion.span>
  );
}

// ---------------------------------------------------------------- Organic pass (see docs/motion-polish/ORGANIC-MOTION.md)

/**
 * Shared-element spring: selection indicators, progress smoothing, the settings slider. Near-critically damped at
 * Subtle and Balanced (no overshoot); Expressive allows a few percent of follow-through. Resolves at animation start.
 */
export const SPRING = {
  type: "spring" as const,
  mass: 0.9,
  get stiffness() { return level().stiffness; },
  get damping() { return level().damping; },
};
/**
 * whileTap targets that follow the motion level, for motion.* buttons that animate their own press
 * (CSS handles plain buttons). Reduced resolves to scale 1; the CSS tint then carries the feedback.
 * Usage: <motion.button whileTap={press}> or whileTap={pressIcon} for small round icon buttons.
 */
export const press = { get scale() { return level().press; } };
export const pressIcon = { get scale() { return 1 - (1 - level().press) * 3; } };
/** Exits are always faster than entrances. */
export const EXIT = t("fast");

/** Three-dot activity indicator. Loops only while mounted (i.e. while work is running); static under reduced motion. */
export function Dots({ label, className }: { label?: string; className?: string }) {
  const reduce = useLessMotion();
  const k = Math.max(level().loop, 0.6);
  return (
    <span className={`dots ${className ?? ""}`} role={label ? "status" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {[0, 1, 2].map((i) => (
        <motion.span key={i} className="dot-i"
          animate={reduce ? { opacity: 0.6 } : { opacity: [0.25, 1, 0.25], y: [0, -2 * k, 0] }}
          transition={reduce ? undefined : { duration: 1.05 / k, repeat: Infinity, ease: "easeInOut", delay: i * 0.14 }} />
      ))}
    </span>
  );
}

/** Check mark that draws itself once. */
export function CheckDraw({ size = 18, delay = 0 }: { size?: number; delay?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <motion.path d="M5 12.5l4.5 4.5L19 7.5" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={t("slow", { delay })} />
    </svg>
  );
}

export type WorkState = "working" | "done" | "error";

/** One status vocabulary for every long-running step: dots while working, a drawn check when done, a cross on error. */
export function StatusLine({ state, children, className }: { state: WorkState; children: string; className?: string }) {
  return (
    <span className={`status-line ${state} ${className ?? ""}`} role={state === "error" ? "alert" : "status"} aria-live="polite">
      <span className="status-ic" aria-hidden>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span key={state} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.6, transition: EXIT }} transition={t("base")}
            style={{ display: "grid", placeItems: "center" }}>
            {state === "working" ? <Dots /> : state === "done" ? <CheckDraw size={16} /> : <span className="status-x">✕</span>}
          </motion.span>
        </AnimatePresence>
      </span>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={children} initial={{ opacity: 0, y: 3 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -3, transition: EXIT }} transition={t("base")}>
          {children}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

/** Text button that opens a panel; the chevron turns to show the state (aria-expanded carries it for AT). */
export function Disclosure({ open, onToggle, children, controls }: { open: boolean; onToggle: () => void; children: ReactNode; controls?: string }) {
  return (
    <button type="button" className="toggle disclosure" aria-expanded={open} aria-controls={controls} onClick={onToggle}>
      <span>{children}</span>
      <motion.svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"
        animate={{ rotate: open ? 180 : 0 }} transition={SPRING} aria-hidden>
        <path d="M6 9l6 6 6-6" />
      </motion.svg>
    </button>
  );
}

/** true on mouse / trackpad devices. Used to keep autoFocus off phones (no surprise keyboard). SSR-safe. */
export function usePointerFine(): boolean {
  return useSyncExternalStore(
    (cb) => { const m = window.matchMedia("(pointer: fine)"); m.addEventListener("change", cb); return () => m.removeEventListener("change", cb); },
    () => window.matchMedia("(pointer: fine)").matches,
    () => false,
  );
}
