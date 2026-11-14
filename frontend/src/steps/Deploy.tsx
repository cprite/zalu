"use client";
import { useEffect, useRef, useState } from "react";
import { classColor } from "@/components/Annotator";
import { Banner, SampleBadge } from "@/components/ui";
import { useApi } from "@/lib/api";
import { VERDICT, pct, variantLabel } from "@/lib/explain";
import { useFlow } from "@/lib/flow";
import { grabJpeg } from "@/lib/frames";
import type { InferenceResult } from "@/lib/types";

export function Deploy() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const modelId = s.deployModelId ?? s.models.find((m) => m.kind === "trained")?.id ?? s.models[0]?.id;
  const model = s.models.find((m) => m.id === modelId);
  const evalEntry = modelId ? s.report?.per_model[modelId] : undefined;
  const classes = model?.classes ?? s.task?.classes ?? [];
  const [target, setTarget] = useState<string>(s.task?.target_event ?? classes[0] ?? "");
  const [conf, setConf] = useState(evalEntry?.settings.conf_threshold ?? 0.25);

  const video = useRef<HTMLVideoElement>(null);
  const [on, setOn] = useState(false);
  const [camErr, setCamErr] = useState<string | null>(null);
  const [res, setRes] = useState<InferenceResult | null>(null);
  const [fps, setFps] = useState(0);
  const [alerts, setAlerts] = useState(0);
  const [loopErr, setLoopErr] = useState<string | null>(null);
  const runId = useRef(0);

  async function start() {
    setCamErr(null);
    setLoopErr(null);
    try {
      const st = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      video.current!.srcObject = st;
      await video.current!.play();
    } catch (e) {
      if (api.mode !== "mock") { setCamErr(`Camera not available: ${(e as Error).message}`); return; }
      setCamErr("No camera found: showing simulated sample detections.");
    }
    setOn(true);
  }
  function stop() {
    runId.current++;
    setOn(false);
    const st = video.current?.srcObject as MediaStream | null;
    st?.getTracks().forEach((t) => t.stop());
    if (video.current) video.current.srcObject = null;
  }
  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!on || !modelId) return;
    const my = ++runId.current;
    let last = performance.now();
    (async () => {
      while (runId.current === my) {
        const v = video.current;
        const blob = (v && v.videoWidth ? await grabJpeg(v) : null) ?? new Blob();
        try {
          const r = await api.inference(modelId, blob, { conf, targetEvent: target || null });
          if (runId.current !== my) break;
          setRes(r);
          if (r.events.length) setAlerts((a) => a + 1);
          const t = performance.now();
          const inst = 1000 / Math.max(1, t - last);  // computed now: the state updater runs later, after `last` changes
          last = t;
          setFps((f) => (f === 0 ? inst : 0.85 * f + 0.15 * inst));
        } catch (e) {
          setLoopErr((e as Error).message);
          await new Promise((r) => setTimeout(r, 1000));
        }
        if (api.mode === "mock") await new Promise((r) => setTimeout(r, 60));
      }
    })();
  }, [on, modelId, conf, target, api]);

  const firing = !!res?.events.length;
  const readiness = evalEntry?.readiness;

  return (
    <div className="center-col fade-in" style={{ maxWidth: 880 }}>
      <div className="hero compact">
        <h1>Run it on your camera</h1>
        <p>Live detections from your model{readiness ? "" : ". It has not been tested on new footage yet"}.</p>
        {readiness && (
          <span className={`badge ${readiness.verdict === "ready" ? "good" : readiness.verdict === "insufficient_data" ? "warn" : "bad"}`}>
            {VERDICT[readiness.verdict].icon} {VERDICT[readiness.verdict].label} · recall {pct(evalEntry!.overall.recall)} · precision {pct(evalEntry!.overall.precision)}
          </span>
        )}
      </div>

      <div className="stage" style={{ aspectRatio: "16/9", outline: firing ? "4px solid var(--color-error)" : undefined }}>
        <video ref={video} muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        {res && on && (
          <svg className="overlay" viewBox={`0 0 ${res.width} ${res.height}`} preserveAspectRatio="xMidYMid slice">
            {res.detections.map((d, i) => (
              <g key={i}>
                <rect x={d.box[0]} y={d.box[1]} width={d.box[2] - d.box[0]} height={d.box[3] - d.box[1]} fill="none" stroke={classColor(classes, d.cls)} strokeWidth={res.width / 260} />
                <text x={d.box[0] + 6} y={d.box[1] - 8} className="box-label" style={{ fontSize: res.width / 45 }}>{d.cls} {Math.round(d.score * 100)}%</text>
              </g>
            ))}
          </svg>
        )}
        {!on && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}><button className="btn primary" disabled={!modelId} onClick={start}>▶ Start camera</button></div>}
        {on && <div style={{ position: "absolute", left: 12, top: 12 }} className="row">
          <span className="badge live">● LIVE</span>
          <span className="badge" style={{ background: "rgba(0,0,0,.6)", color: "#fff" }}>{fps.toFixed(1)} FPS</span>
          <SampleBadge show={api.mode === "mock"} label="Simulated" />
        </div>}
      </div>
      {firing && <Banner kind="bad"><b>Alert: {res!.events.join(", ")} detected</b></Banner>}
      {camErr && <Banner kind={api.mode === "mock" ? "sample" : "bad"}>{camErr}</Banner>}
      {loopErr && <Banner kind="bad">Inference error: {loopErr}</Banner>}

      <div className="row" style={{ justifyContent: "center", gap: 24 }}>
        <label className="field" style={{ minWidth: 200 }}>Alert on
          <select className="select" value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">No alerts</option>
            {classes.map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>
        <label className="field" style={{ minWidth: 220 }}>Sensitivity: {Math.round(conf * 100)}% confidence
          <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
        </label>
        {s.models.length > 1 && (
          <label className="field" style={{ minWidth: 220 }}>Model
            <select className="select" value={modelId ?? ""} onChange={(e) => { stop(); set({ deployModelId: e.target.value }); }}>
              {s.models.map((m) => <option key={m.id} value={m.id}>{variantLabel(m.kind, m.variant)}</option>)}
            </select>
          </label>
        )}
      </div>
      <div className="actions-center">
        {on && <button className="btn" onClick={stop}>■ Stop</button>}
        {alerts > 0 && <span className="small muted">{alerts} alert(s) this session</span>}
      </div>
      {model?.onnx && <p className="hint">Works offline too: <code>python tools/webcam_client.py --onnx best.onnx --names {classes.join(",")}</code></p>}
    </div>
  );
}
