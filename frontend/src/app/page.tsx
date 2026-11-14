"use client";
import { useEffect, useRef } from "react";
import { StatusFooter, Stepper, Topbar } from "@/components/Shell";
import { ApiProvider } from "@/lib/api";
import { FlowProvider, STEPS, useFlow } from "@/lib/flow";
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
    <main id="main" ref={main} tabIndex={-1} aria-label={`Step ${s.step + 1} of ${STEPS.length}: ${STEPS[s.step].label}`} style={{ outline: "none" }}>
      <Step key={s.step} />
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
          <Stepper />
          <Journey />
          <StatusFooter />
        </div>
      </FlowProvider>
    </ApiProvider>
  );
}
