"use client";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { useApi, type ZaluApi } from "@/lib/api";
import { BRAND } from "@/lib/explain";
import { STAGES, stageFill, stageOf, unlocked, useFlow } from "@/lib/flow";
import { t } from "@/lib/motion";
import type { Health } from "@/lib/types";

export function Topbar() {
  const { api, config, setConfig } = useApi();
  const { reset } = useFlow();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(config);
  const { s } = useFlow();
  // health probe result is tagged with the api it belongs to, so a new data source shows "checking" without a reset in the effect
  const [probe, setProbe] = useState<{ api: ZaluApi; health?: Health; err?: string } | null>(null);
  const current = probe?.api === api ? probe : null;
  const health = current?.health ?? null;
  const err = current?.err ?? null;

  /** Prefill the source form when it opens (remembered URL, then the build-time default). */
  function openSource() {
    try {
      const remembered = JSON.parse(localStorage.getItem("zalu.api") ?? "{}").baseUrl;
      setDraft(config.baseUrl ? config : { ...config, baseUrl: remembered || process.env.NEXT_PUBLIC_ZALU_API || "" });
    } catch { setDraft(config); }
    setOpen(true);
  }
  useEffect(() => {
    let live = true;
    api.health().then((h) => live && setProbe({ api, health: h })).catch((e) => live && setProbe({ api, err: e.message }));
    return () => { live = false; };
  }, [api]);

  const switchTo = (c: typeof config) => {
    if ((s.task || s.trainFiles.length) && !window.confirm("Switch data source and start a new session?")) return;
    setConfig(c);
    reset();
    setOpen(false);
  };

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="#fff" strokeWidth="2" strokeDasharray="4 3" /><circle cx="9" cy="9" r="3" fill="#fff" /></svg>
        </div>
        {BRAND}
      </div>
      <div className="spacer" />
      <div style={{ position: "relative" }}>
        <button className="btn sm" onClick={() => (open ? setOpen(false) : openSource())} aria-expanded={open} aria-controls="source-pop"
          aria-label={`Data source: ${api.mode === "mock" ? "sample mode" : "live backend"}. Change`}>
          {api.mode === "mock" ? <span className="badge sample">◆ Sample mode</span> : <span className="badge live">● Live backend</span>}
          {api.mode === "live" && (err ? <span className="badge bad">✕ offline</span> : health ? <span className="badge good">✓ connected</span> : <span className="spin" />)}
        </button>
        {open && (
          <div id="source-pop" className="card popover fade-in col" role="dialog" aria-label="Data source"
            onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}>
            <h4>Data source</h4>
            <p className="small ink2">Sample mode runs entirely in your browser on generated sample data. Live mode calls the {BRAND} backend.</p>
            <label className="field">Backend URL
              <input className="input" placeholder="http://localhost:9100" value={draft.baseUrl} onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })} />
            </label>
            <label className="field">Access token
              <input className="input" type="password" value={draft.token} onChange={(e) => setDraft({ ...draft, token: e.target.value })} />
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
            {err && <div className="small negv">{err}</div>}
            <div className="row">
              <button className="btn primary sm" disabled={!draft.baseUrl} onClick={() => switchTo(draft)}>Connect live</button>
              <button className="btn sm" onClick={() => switchTo({ baseUrl: "", token: "" })}>Use sample mode</button>
            </div>
          </div>
        )}
      </div>
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
                  <motion.span initial={false} animate={{ width: `${Math.round(fill[i] * 100)}%` }} transition={t("slow")} />
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
