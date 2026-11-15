"use client";

import { motion, useReducedMotion } from "motion/react";
import { CheckDraw, Dots, t } from "@/lib/motion";
import styles from "./screenActivity.module.css";

export function ScreenActivity({ message, slot = false }: { message?: string; slot?: boolean }) {
  const reduce = useReducedMotion();
  if (!message && !slot) return null;
  return <div className={slot ? styles.slot : undefined} role="status">
    {message && <span className={styles.activity}>
      <span className={styles.activityText}>
        <motion.span key={message} initial={reduce ? false : { clipPath: "inset(0 100% 0 0)" }}
          animate={{ clipPath: "inset(0 0% 0 0)" }} transition={t("slow")}>{message}</motion.span>
      </span>
      <Dots />
    </span>}
  </div>;
}

export function TrainingProgress({ value, active, label }: { value: number; active: boolean; label: string }) {
  const reduce = useReducedMotion();
  const fraction = Math.max(0, Math.min(1, value));
  const percent = Math.round(fraction * 100);
  return <div className={styles.track} role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100}
    aria-valuenow={percent}>
    <motion.div className={styles.fill} initial={false} animate={{ width: `${fraction * 100}%` }}
      transition={reduce ? { duration: 0 } : t("slow")}>
      {active && value > 0 && !reduce && <span className={styles.flow} data-training-flow />}
    </motion.div>
  </div>;
}

export function TrainingComplete() {
  const reduce = useReducedMotion();
  return <motion.span className={styles.complete} aria-hidden data-training-complete
    initial={reduce ? false : { opacity: 0, scale: 0.88 }} animate={{ opacity: 1, scale: 1 }} transition={t("slow")}>
    {reduce ? <span>✓</span> : <CheckDraw size={22} />}
  </motion.span>;
}
