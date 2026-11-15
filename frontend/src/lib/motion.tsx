"use client";
// Motion tokens for ZALU. One easing curve and three durations (150 / 220 / 320 ms) everywhere.
// <MotionConfig reducedMotion="user"> in page.tsx turns transforms into instant changes for people who ask for less motion.
import { animate, motion, useInView, useMotionValue, useReducedMotion, useTransform, type Transition } from "motion/react";
import { useEffect, useRef } from "react";

export const EASE = [0.2, 0, 0, 1] as const;           // standard "emphasized decelerate"
export const DUR = { fast: 0.15, base: 0.22, slow: 0.32 } as const;
export const t = (d: keyof typeof DUR = "base", extra: Transition = {}): Transition => ({ duration: DUR[d], ease: EASE, ...extra });

/** Screen / section entrance: fade with a short rise. */
export const enter = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -4 },
  transition: t("base"),
};

/** Container + child variants for the one orchestrated load moment (Describe hero). */
export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05, delayChildren: 0.04 } },
};
export const rise = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: t("slow") },
};

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
export function AnimatedNumber({ value, format, className, duration = DUR.slow * 2 }: { value: number; format: (v: number) => string; className?: string; duration?: number }) {
  const reduce = useReducedMotion();
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
    const c = animate(mv, value, { duration, ease: EASE });
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
  const reduce = useReducedMotion();
  if (reduce) return <span className={className}>{children}</span>;
  return (
    <motion.span className={`shimmer ${className ?? ""}`}
      initial={{ backgroundPosition: "100% center" }} animate={{ backgroundPosition: "0% center" }}
      transition={{ repeat: Infinity, duration: 1.8, ease: "linear" }}>
      {children}
    </motion.span>
  );
}
