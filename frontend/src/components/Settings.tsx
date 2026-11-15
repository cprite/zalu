"use client";
import * as Slider from "@radix-ui/react-slider";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { EXIT, SPRING, level, t } from "@/lib/motion";
import { MOTION_LABELS, setPrefs, usePrefs, type MotionLevel, type ScalePref, type ThemePref } from "@/lib/prefs";

const MOTION_HINT = [
  "Fades only. No movement, springs or looping indicators.",
  "Shorter, smaller movements.",
  "The default: calm movement that explains what changed.",
  "A little more travel and spring. Still no bounce.",
] as const;

export function SettingsMenu() {
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const titleId = useId();

  const close = useCallback((refocus = true) => { setOpen(false); if (refocus) trigger.current?.focus(); }, []);

  // outside click closes; focus moves into the panel on open
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!panel.current?.contains(e.target as Node) && !trigger.current?.contains(e.target as Node)) close(false);
    };
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented) close();
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    const f = requestAnimationFrame(() => panel.current?.querySelector<HTMLElement>('[role="radio"][aria-checked="true"], [role="slider"]')?.focus());
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
      cancelAnimationFrame(f);
    };
  }, [open, close]);

  return (
    <div className="settings-anchor">
      <button ref={trigger} type="button" className="settings-trigger" onClick={() => setOpen(!open)}
        aria-haspopup="dialog" aria-expanded={open} aria-controls="settings-pop" aria-label="Settings">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M9.6 3h4.8l.5 2.3 2 .9 2.1-1.1 2.4 4.1-1.7 1.6v2.4l1.7 1.6-2.4 4.1-2.1-1.1-2 .9-.5 2.3H9.6l-.5-2.3-2-.9L5 18.9l-2.4-4.1 1.7-1.6v-2.4L2.6 9.2 5 5.1l2.1 1.1 2-.9z" /><circle cx="12" cy="12" r="3" />
        </svg>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div ref={panel} id="settings-pop" className="settings-pop" role="dialog" aria-modal="false" aria-labelledby={titleId}
            initial={{ opacity: 0, y: -6 * level().travel, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4 * level().travel, scale: 0.98, transition: EXIT }} transition={t("base")}>
            <div className="settings-head">
              <h2 id={titleId}>Settings</h2>
              <button type="button" className="icon-btn ghost" onClick={() => close()} aria-label="Close settings">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M6 6l12 12M18 6L6 18" /></svg>
              </button>
            </div>
            <Appearance />
            <Accessibility />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------- appearance

function Appearance() {
  const { prefs, sysReduced } = usePrefs();
  return (
    <section className="settings-group" aria-labelledby="settings-appearance">
      <h3 id="settings-appearance">Appearance</h3>
      <Section title="Theme">
        <Segmented<ThemePref> label="Theme" value={prefs.theme} onChange={(theme) => setPrefs({ theme })} options={[
          { value: "light", label: "Light", icon: <SunIcon /> },
          { value: "dark", label: "Dark", icon: <MoonIcon /> },
          { value: "system", label: "System", icon: <AutoIcon /> },
        ]} />
      </Section>
      <Section title="Motion" aside={MOTION_LABELS[prefs.motion]}>
        <MotionSlider value={prefs.motion} onChange={(motion) => setPrefs({ motion })} />
        <p className="settings-hint" aria-live="polite">
          {sysReduced || prefs.reducedMotion ? "Reduced motion is active." : MOTION_HINT[prefs.motion]}
        </p>
      </Section>
      <Section title="Text size">
        <Segmented<ScalePref> label="Text size" value={prefs.scale} onChange={(scale) => setPrefs({ scale })} options={[
          { value: "compact", label: "Compact", icon: <span className="aa sm" aria-hidden>Aa</span> },
          { value: "default", label: "Default", icon: <span className="aa" aria-hidden>Aa</span> },
          { value: "large", label: "Large", icon: <span className="aa lg" aria-hidden>Aa</span> },
        ]} />
      </Section>
    </section>
  );
}

function Section({ title, aside, children }: { title: string; aside?: string; children: ReactNode }) {
  return (
    <section className="settings-sec">
      <div className="settings-sec-head"><h4>{title}</h4>{aside && <span className="settings-aside">{aside}</span>}</div>
      {children}
    </section>
  );
}

function Accessibility() {
  const { prefs, sysReduced } = usePrefs();
  return (
    <section className="settings-group" aria-labelledby="settings-accessibility">
      <h3 id="settings-accessibility">Accessibility</h3>
      <div className="settings-options">
        <label><input type="checkbox" checked={prefs.highContrast}
          onChange={(event) => setPrefs({ highContrast: event.target.checked })} /> High contrast</label>
        <label><input type="checkbox" checked={prefs.reducedMotion || sysReduced} disabled={sysReduced}
          onChange={(event) => setPrefs({ reducedMotion: event.target.checked })} /> Reduced motion</label>
        {sysReduced && <p className="settings-hint">Enabled by your device.</p>}
        <label><input type="checkbox" checked={prefs.strongFocus}
          onChange={(event) => setPrefs({ strongFocus: event.target.checked })} /> Strong focus indicators</label>
        <label><input type="checkbox" checked={prefs.keyboardShortcuts}
          onChange={(event) => setPrefs({ keyboardShortcuts: event.target.checked })} /> Keyboard shortcuts</label>
        <p className="settings-hint">Arrow keys browse frames. Delete removes a selected box.</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- primitives

type Opt<T extends string> = { value: T; label: string; icon?: ReactNode };

/** Radio group drawn as a segmented control; the selection pill glides between options on the shared spring. */
function Segmented<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: Opt<T>[]; onChange: (v: T) => void }) {
  const id = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, i: number) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + options.length) % options.length;
    onChange(options[n].value);
    refs.current[n]?.focus();
  };
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o, i) => {
        const on = o.value === value;
        return (
          <button key={o.value} ref={(el) => { refs.current[i] = el; }} type="button" role="radio" aria-checked={on} tabIndex={on ? 0 : -1}
            className={on ? "on" : ""} onClick={() => onChange(o.value)} onKeyDown={(e) => onKey(e, i)}>
            {on && <motion.span layoutId={`${id}-pill`} className="segmented-pill" transition={SPRING} aria-hidden />}
            <span className="segmented-label">{o.icon}{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Four-stop slider (Radix: pointer, touch and full keyboard support). The thumb glides between stops on a tactile curve. */
function MotionSlider({ value, onChange }: { value: MotionLevel; onChange: (v: MotionLevel) => void }) {
  return (
    <div className="mslider">
      <Slider.Root className="mslider-root" min={0} max={3} step={1} value={[value]} onValueChange={([v]) => onChange(v as MotionLevel)}>
        <Slider.Track className="mslider-track">
          <Slider.Range className="mslider-range" />
        </Slider.Track>
        <span className="mslider-ticks" aria-hidden>{[0, 1, 2, 3].map((i) => <i key={i} className={i <= value ? "on" : ""} />)}</span>
        <Slider.Thumb className="mslider-thumb" aria-label="Motion intensity" aria-valuetext={MOTION_LABELS[value]} />
      </Slider.Root>
      <div className="mslider-labels" aria-hidden>
        {MOTION_LABELS.map((l, i) => (
          <button key={l} type="button" tabIndex={-1} className={i === value ? "on" : ""} onClick={() => onChange(i as MotionLevel)}>{l}</button>
        ))}
      </div>
    </div>
  );
}

const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
    <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);
const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
  </svg>
);
const AutoIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" />
  </svg>
);
