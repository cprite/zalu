"use client";
import { useEffect, useState } from "react";
import { useApi } from "@/lib/api";
import { BRAND } from "@/lib/explain";
import { STEPS, unlocked, useFlow } from "@/lib/flow";
import type { Health } from "@/lib/types";

export function Topbar() {
  const { api, config, setConfig } = useApi();
  const { reset } = useFlow();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(config);
  const [health, setHealth] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { setDraft(config); }, [config]);
  useEffect(() => {
    let live = true;
    setHealth(null);
    setErr(null);
    api.health().then((h) => live && setHealth(h)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [api]);

  const switchTo = (c: typeof config) => { setConfig(c); reset(); setOpen(false); };

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="#fff" strokeWidth="2" strokeDasharray="4 3" /><circle cx="9" cy="9" r="3" fill="#fff" /></svg>
        </div>
        {BRAND}
      </div>
      <span className="muted small hidden md:inline">Describe it. Train it. Prove it works.</span>
      <div className="spacer" />
      <div style={{ position: "relative" }}>
        <button className="btn sm" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="source-pop"
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

export function Stepper() {
  const { s, set } = useFlow();
  const open = unlocked(s);
  return (
    <nav aria-label="Progress">
      <ol className="stepper">
        {STEPS.map((st, i) => {
          const done = i < s.step && open[i];
          const state = i === s.step ? "current step" : done ? "completed" : open[i] ? "available" : "locked";
          return (
            <li key={st.key}>
              <button className={`step ${i === s.step ? "active" : ""} ${done ? "done" : ""}`} disabled={!open[i]}
                onClick={() => set({ step: i })} aria-current={i === s.step ? "step" : undefined} title={`${st.label}: ${st.sub}`}>
                <span className="num" aria-hidden>{done ? "✓" : i + 1}</span>
                <span><span className="lbl block">{st.label}</span><span className="sub block">{st.sub}</span></span>
                <span className="sr-only">, {state}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/** P2: make clear which parts are working, experimental or mocked. */
export function StatusFooter() {
  const { api } = useApi();
  const [engines, setEngines] = useState<Record<string, string>>({});
  const [gpuOk, setGpuOk] = useState(false);
  useEffect(() => {
    api.health().then((h) => { setEngines(h.engines ?? {}); setGpuOk(!!h.gpu_worker?.ok); }).catch(() => { setEngines({}); setGpuOk(false); });
  }, [api]);
  const mock = api.mode === "mock";
  // live: an engine string containing "MOCK" means the backend answers with clearly labelled mock data
  const eng = (k: string): [string, string] => {
    const v = engines[k];
    if (!v) return ["warn", "unknown"];
    return /mock/i.test(v) ? ["sample", v] : ["live", v];
  };
  const gpu: [string, string] = gpuOk ? ["live", "GX10 worker"] : ["warn", "GPU worker unreachable"];
  const pick = (sample: string, live: [string, string]): [string, string] => (mock ? ["sample", sample] : live);
  const rows: [string, string, string][] = [
    ["Task planner", ...pick("rule-based sample", eng("task_engine"))],
    ["Upload & auto-label", ...pick("sample labels", eng("data_engine"))],
    ["Training (GPU)", ...pick("simulated progress", gpu)],
    ["Evaluation engine", ...pick("real engine, sample predictions", eng("evaluation_engine"))],
    ["Live inference", ...pick("simulated boxes", gpu)],
    ["Data coach", "exp", "rule-based on your frames"],
  ];
  const label: Record<string, string> = { sample: "Sample", live: "Working", exp: "Experimental", warn: "Not connected" };
  return (
    <footer className="status card flat">
      <div className="row" style={{ marginBottom: 8 }}><h4>What is real in this demo</h4><span className="muted small">updates from the backend health check</span></div>
      <div className="row" style={{ gap: 18 }}>
        {rows.map(([n, k, d]) => (
          <div key={n} className="small"><span className={`badge ${k === "warn" ? "warn" : k}`}>{label[k]}</span> <b>{n}</b> <span className="muted">· {d}</span></div>
        ))}
      </div>
    </footer>
  );
}
