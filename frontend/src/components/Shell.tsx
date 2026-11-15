"use client";
import { motion } from "motion/react";
import { BRAND } from "@/lib/explain";
import { STAGES, stageFill, stageOf, unlocked, useFlow } from "@/lib/flow";
import { SPRING } from "@/lib/motion";
import { SettingsMenu } from "./Settings";

/** Header with the brand and Settings. */
export function Topbar() {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="#fff" strokeWidth="2" strokeDasharray="4 3" /><circle cx="9" cy="9" r="3" fill="#fff" /></svg>
        </div>
        {BRAND}
      </div>
      <div className="spacer" />
      <SettingsMenu />
    </header>
  );
}

/** Stories-style progress: one thin segment per stage, filled by progress inside the stage. */
export function StoryProgress() {
  const { s, set } = useFlow();
  const open = unlocked(s);
  const cur = stageOf(s.step);
  const fill = stageFill(s.step);
  return (
    <nav aria-label="Progress" className="story">
      <ol className="story-bar">
        {STAGES.map((st, i) => {
          const first = st.steps[0];
          const reachable = open[first] && i !== cur;
          const state = i === cur ? "current stage" : i < cur ? "completed" : open[first] ? "available" : "not started";
          return (
            <li key={st.label}>
              <button className={`story-seg ${i === cur ? "active" : ""}`} disabled={!reachable} onClick={() => set({ step: first })}
                aria-current={i === cur ? "step" : undefined}>
                <span className="track" aria-hidden>
                  <motion.span initial={false} animate={{ scaleX: fill[i] }} transition={SPRING} />
                </span>
                <span className="story-label">{st.label}</span>
                <span className="sr-only">, {state}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
