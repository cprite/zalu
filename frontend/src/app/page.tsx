"use client";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef } from "react";
import { StoryProgress, Topbar } from "@/components/Shell";
import { ApiProvider } from "@/lib/api";
import { FlowProvider, STAGES, STEPS, stageOf, useFlow } from "@/lib/flow";
import { enter } from "@/lib/motion";
import { Deploy } from "@/steps/Deploy";
import { Describe } from "@/steps/Describe";
import { Prepare } from "@/steps/Prepare";
import { Train } from "@/steps/Train";
import { Upload } from "@/steps/Upload";
import { Verify } from "@/steps/Verify";

const VIEWS = [Describe, Upload, Prepare, Train, Verify, Deploy];

function Journey() {
  const { s } = useFlow();
  const Step = VIEWS[s.step];
  const main = useRef<HTMLElement>(null);
  const first = useRef(true);
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (first.current) { first.current = false; return; }
    main.current?.focus({ preventScroll: true }); // move keyboard / screen-reader focus to the new step
  }, [s.step]);
  return (
    <main id="main" ref={main} tabIndex={-1} aria-label={`${STAGES[stageOf(s.step)].label}: ${STEPS[s.step].label}`} style={{ outline: "none" }}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={s.step} {...enter}>
          <Step />
        </motion.div>
      </AnimatePresence>
    </main>
  );
}

export default function Page() {
  return (
    <ApiProvider>
      <FlowProvider>
        <a href="#main" className="skip">Skip to content</a>
        <div className="shell">
          <Topbar />
          <StoryProgress />
          <Journey />
        </div>
      </FlowProvider>
    </ApiProvider>
  );
}
