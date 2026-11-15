"use client";
import { motion } from "motion/react";
import type { ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { Dots, SPRING, t } from "@/lib/motion";

export function SampleBadge({ show = true, label = "Sample data" }: { show?: boolean; label?: string }) {
  if (!show) return null;
  return <span className="badge sample" title="Generated for development. Not a real experimental result.">◆ {label}</span>;
}

type Kind = "info" | "warn" | "bad" | "good" | "sample";
const ICON: Record<Kind, string> = { info: "i", warn: "!", bad: "✕", good: "✓", sample: "◆" };
const LABEL: Record<Kind, string> = { info: "Note", warn: "Warning", bad: "Error", good: "Success", sample: "Sample data" };

/** Status is always conveyed by icon + hidden label + text, never by colour alone. */
export function Banner({ kind = "info", children, id }: { kind?: Kind; children: ReactNode; id?: string }) {
  return (
    <motion.div className={`banner ${kind}`} id={id} role={kind === "bad" ? "alert" : "status"}
      initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} transition={t("base")}>
      <span className="ic" aria-hidden>{ICON[kind]}</span>
      <span className="sr-only">{LABEL[kind]}:</span>
      <div className="small" style={{ flex: 1 }}>{children}</div>
    </motion.div>
  );
}

export function Spinner() {
  return <Dots label="Loading…" />;
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  if (error instanceof ApiError && error.notImplemented) {
    return (
      <Banner kind="warn">
        <b>This step is not connected on the backend yet.</b> {error.message}
        <div>Switch to <b>Sample mode</b> (top right) to continue the walkthrough, or pick an existing dataset.</div>
      </Banner>
    );
  }
  const msg = error instanceof Error ? error.message : String(error);
  return <Banner kind="bad"><b>Something went wrong.</b> {msg} <div>Check the connection and try again.</div></Banner>;
}

export function Kpi({ k, v, ci, help }: { k: string; v: ReactNode; ci?: ReactNode; help?: string }) {
  return (
    <div className="kpi">
      <div className="k">{k}</div>
      <div className="v num">{v}</div>
      {ci && <div className="ci num">{ci}</div>}
      {help && <div className="tiny muted" style={{ marginTop: 4 }}>{help}</div>}
    </div>
  );
}

/** Point estimate, 95% interval and the acceptance target on one 0..100% track. Text equivalent is given next to it. */
export function CiBar({ value, ci, target }: { value: number | null; ci: [number, number] | null; target?: number }) {
  return (
    <div className="ci-track" aria-hidden>
      <div className="rail" />
      {ci && <div className="range" style={{ left: `${ci[0] * 100}%`, width: `${(ci[1] - ci[0]) * 100}%` }} />}
      {target != null && <div className="target" style={{ left: `${target * 100}%` }} />}
      {value != null && <div className="pt" style={{ left: `${value * 100}%` }} />}
    </div>
  );
}

export function Progress({ value, label }: { value: number; label: string }) {
  const pctv = Math.round(value * 100);
  return (
    <div className="bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pctv}>
      {/* scaleX (compositor-only), spring-smoothed toward the real backend value */}
      <motion.span className="bar-fill" initial={false} animate={{ scaleX: Math.max(0, Math.min(1, value)) }} transition={SPRING} />
    </div>
  );
}
