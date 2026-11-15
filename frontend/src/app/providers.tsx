"use client";
import { MotionConfig } from "motion/react";
import { useLayoutEffect, type ReactNode } from "react";
import { enter, markMotionHydrated } from "@/lib/motion";
import { usePrefs } from "@/lib/prefs";

/**
 * App-wide providers. Theme, scale and the CSS side of motion live on <html> data attributes (set before paint by
 * PREFS_SCRIPT in layout.tsx); this adds the Motion side: Reduced (or the OS setting) disables transform and layout
 * animation for every motion component, the other levels flow through the lazily-read tokens in lib/motion.
 */
export function Providers({ children }: { children: ReactNode }) {
  const { motion } = usePrefs();
  useLayoutEffect(() => { markMotionHydrated(); }, []);
  return (
    <MotionConfig reducedMotion={motion === 0 ? "always" : "user"} transition={enter.transition}>
      {children}
    </MotionConfig>
  );
}
