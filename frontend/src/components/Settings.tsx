"use client";
// Settings popover: Theme, Motion intensity, UI scale and Data source. Replaces the header "Sample mode" badge; the
// trigger keeps a quiet, always-visible indication of the current data source (text + dot, never colour alone).
import * as Slider from "@radix-ui/react-slider";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useApi, type ZaluApi } from "@/lib/api";
import { BRAND } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import { EXIT, SPRING, level, t } from "@/lib/motion";
import { MOTION_LABELS, setPrefs, usePrefs, type MotionLevel, type ScalePref, type ThemePref } from "@/lib/prefs";
import type { Health } from "@/lib/types";

const MOTION_HINT = [
  "Fades only. No movement, springs or looping indicators.",
  "Shorter, smaller movements.",
  "The default: calm movement that explains what changed.",
  "A little more travel and spring. Still no bounce.",
] as const;

export function SettingsMenu() {
  const { api, config, setConfig } = useApi();
  const { s, reset } = useFlow();
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const titleId = useId();

  // health probe, tagged with the api it belongs to (a new source shows "checking" without resetting state in an effect)
  const [probe, setProbe] = useState<{ api: ZaluApi; health?: Health; err?: string } | null>(null);
  const current = probe?.api === api ? probe : null;
  useEffect(() => {
    let live = true;
    api.health().then((h) => live && setProbe({ api, health: h })).catch((e) => live && setProbe({ api, err: e.message }));
    return () => { live = false; };
  }, [api]);

  const close = useCallback((refocus = true) => { setOpen(false); if (refocus) trigger.current?.focus(); }, []);

  // outside click closes; focus moves into the panel on open
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!panel.current?.contains(e.target as Node) && !trigger.current?.contains(e.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", onDown);
    const f = requestAnimationFrame(() => panel.current?.querySelector<HTMLElement>('[role="radio"][aria-checked="true"], [role="slider"]')?.focus());
    return () => { document.removeEventListener("pointerdown", onDown); cancelAnimationFrame(f); };
  }, [open, close]);

  const source = api.mode === "mock" ? { cls: "sample", text: "Sample data" }
    : current?.err ? { cls: "bad", text: "Offline" }
    : current?.health ? { cls: "live", text: "Live" }
    : { cls: "checking", text: "Connecting…" };

  return (
    <div className="settings-anchor">
      <button ref={trigger} type="button" className="settings-trigger" onClick={() => setOpen(!open)}
        aria-haspopup="dialog" aria-expanded={open} aria-controls="settings-pop" aria-label={`Settings. Data source: ${source.text}`}>
        <span className={`source-ind ${source.cls}`} aria-hidden><i />{source.text}</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M4 7h10M18 7h2M4 17h4M12 17h8" /><circle cx="16" cy="7" r="2" /><circle cx="10" cy="17" r="2" />
        </svg>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div ref={panel} id="settings-pop" className="settings-pop" role="dialog" aria-modal="false" aria-labelledby={titleId}
            onKeyDown={(e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } }}
            initial={{ opacity: 0, y: -6 * level().travel, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4 * level().travel, scale: 0.98, transition: EXIT }} transition={t("base")}>
            <div className="settings-head">
              <h2 id={titleId}>Settings</h2>
              <button type="button" className="icon-btn ghost" onClick={() => close()} aria-label="Close settings">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M6 6l12 12M18 6L6 18" /></svg>
              </button>
            </div>
            <Appearance />
            <DataSource api={api} config={config} health={current?.health ?? null} err={current?.err ?? null}
              onSwitch={(c) => {
                if ((s.task || s.trainFiles.length) && !window.confirm("Switch data source and start a new session?")) return false;
                setConfig(c); reset(); return true;
              }} />
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
    <>
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
          {sysReduced ? "Your device asks for reduced motion, so animations stay minimal whatever this is set to." : MOTION_HINT[prefs.motion]}
        </p>
      </Section>
      <Section title="Text size">
        <Segmented<ScalePref> label="Text size" value={prefs.scale} onChange={(scale) => setPrefs({ scale })} options={[
          { value: "compact", label: "Compact", icon: <span className="aa sm">Aa</span> },
          { value: "default", label: "Default", icon: <span className="aa">Aa</span> },
          { value: "large", label: "Large", icon: <span className="aa lg">Aa</span> },
        ]} />
      </Section>
    </>
  );
}

function Section({ title, aside, children }: { title: string; aside?: string; children: ReactNode }) {
  return (
    <section className="settings-sec">
      <div className="settings-sec-head"><h3>{title}</h3>{aside && <span className="settings-aside">{aside}</span>}</div>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------- data source

function DataSource({ api, config, health, err, onSwitch }: {
  api: ZaluApi; config: { baseUrl: string; token: string }; health: Health | null; err: string | null;
  onSwitch: (c: { baseUrl: string; token: string }) => boolean;
}) {
  const [want, setWant] = useState<"mock" | "live">(api.mode);
  const [draft, setDraft] = useState(() => {
    if (config.baseUrl) return config;
    let remembered = "";
    try { remembered = JSON.parse(localStorage.getItem("zalu.api") ?? "{}").baseUrl ?? ""; } catch { /* ignore */ }
    return { ...config, baseUrl: remembered || process.env.NEXT_PUBLIC_ZALU_API || "" };
  });
  const pick = (m: "mock" | "live") => {
    if (m === "mock" && api.mode === "live") { if (!onSwitch({ baseUrl: "", token: "" })) return; }
    setWant(m);
  };
  return (
    <Section title="Data source">
      <Segmented<"mock" | "live"> label="Data source" value={want} onChange={pick} options={[
        { value: "mock", label: "Sample", icon: <span className="src-glyph sample" /> },
        { value: "live", label: "Live backend", icon: <span className="src-glyph live" /> },
      ]} />
      <AnimatePresence initial={false} mode="wait">
        {want === "mock" ? (
          <motion.p key="mock" className="settings-hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: EXIT }} transition={t("fast")}>
            Runs entirely in your browser on generated sample frames. Results are labelled as sample data and are not real measurements.
          </motion.p>
        ) : (
          <motion.div key="live" className="col settings-live" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0, transition: EXIT }} transition={t("base")} style={{ overflow: "hidden" }}>
            <label className="field">Backend URL
              <input className="input" placeholder="https://…" value={draft.baseUrl} onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                inputMode="url" autoComplete="url" spellCheck={false} />
            </label>
            <label className="field">Access token
              <input className="input" type="password" autoComplete="off" value={draft.token} onChange={(e) => setDraft({ ...draft, token: e.target.value })} />
            </label>
            {api.mode === "live" && health && (
              <div className="small ink2">
                GPU worker: {health.gpu_worker?.ok ? "✓ reachable" : `✕ ${health.gpu_worker?.error ?? "unreachable"}`} · engines:{" "}
                {Object.entries(health.engines ?? {}).map(([k, v]) => {
                  const mockEngine = /mock/i.test(v);
                  return <span key={k} title={v} className={`badge ${mockEngine ? "warn" : "good"}`} style={{ marginRight: 4 }}>{mockEngine ? "! " : "✓ "}{k.replace("_engine", "")}</span>;
                })}
              </div>
            )}
            {api.mode === "live" && err && <div className="small negv" role="alert">✕ {err}</div>}
            <div className="row">
              <button type="button" className="btn primary sm" disabled={!draft.baseUrl} onClick={() => onSwitch(draft)}>
                {api.mode === "live" && config.baseUrl === draft.baseUrl ? "Reconnect" : "Connect"}
              </button>
              <span className="tiny muted">Calls the {BRAND} backend with your footage.</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Section>
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
