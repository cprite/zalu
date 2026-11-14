"use client";
import { useEffect, useRef, useState } from "react";
import { classColor } from "@/components/Annotator";
import { Banner, SampleBadge, SectionHead } from "@/components/ui";
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
    <div className="fade-in">
      <SectionHead title="Run it on your camera" sub="Live detections from the trained model. The same model is exported to ONNX to run fully offline on a laptop or edge device." />
      <div className="split">
        <div className="col">
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
            {!on && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}><button className="btn primary" disabled={!modelId} onClick={start}>▶ Start live detection</button></div>}
            {on && <div style={{ position: "absolute", left: 12, top: 12 }} className="row">
              <span className="badge live">● LIVE</span>
              <span className="badge" style={{ background: "rgba(0,0,0,.6)", color: "#fff" }}>{fps.toFixed(1)} FPS end-to-end</span>
              {res && <span className="badge" style={{ background: "rgba(0,0,0,.6)", color: "#fff" }}>model {res.latency_ms} ms</span>}
              <SampleBadge show={api.mode === "mock"} label="Simulated" />
            </div>}
          </div>
          {firing && <Banner kind="bad"><b>Alert: {res!.events.join(", ")} detected</b></Banner>}
          {camErr && <Banner kind={api.mode === "mock" ? "sample" : "bad"}><span className="small">{camErr}</span></Banner>}
          {loopErr && <Banner kind="bad"><span className="small">Inference error: {loopErr}</span></Banner>}
          <div className="card row">
            <label className="field" style={{ flex: 1 }}>Alert on
              <select className="select" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">No alerts</option>
                {classes.map((c) => <option key={c}>{c}</option>)}
              </select>
            </label>
            <label className="field" style={{ flex: 1 }}>Confidence threshold: {Math.round(conf * 100)}%
              <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
            </label>
            {on && <button className="btn" onClick={stop}>■ Stop</button>}
          </div>
          {evalEntry && conf !== evalEntry.settings.conf_threshold && (
            <div className="tiny muted">The report card was measured at {Math.round(evalEntry.settings.conf_threshold * 100)}%. Changing the threshold changes precision and recall.</div>
          )}
        </div>

        <div className="col">
          <div className="card col">
            <h4>Model</h4>
            <select className="select" value={modelId ?? ""} onChange={(e) => { stop(); set({ deployModelId: e.target.value }); }}>
              {s.models.map((m) => <option key={m.id} value={m.id}>{variantLabel(m.kind, m.variant)} · {m.id}</option>)}
            </select>
            {model && (
              <div className="small ink2 col" style={{ gap: 2 }}>
                <div>Version <code>{model.id}</code></div>
                <div>Base <code>{model.base_model}</code>{model.epochs ? ` · ${model.epochs} epochs` : ""}{model.train_seconds ? ` · trained in ${model.train_seconds}s` : ""}</div>
                <div>Classes: {model.classes.join(", ")}</div>
                {model.onnx && <div>ONNX: <code className="tiny">{model.onnx}</code></div>}
                {model.device && <div className="tiny muted">Trained on {model.device}</div>}
              </div>
            )}
            {readiness ? (
              <div className={`verdict ${readiness.verdict}`} style={{ padding: 10 }}>
                <div className="icon" style={{ width: 32, height: 32, fontSize: 16 }}>{VERDICT[readiness.verdict].icon}</div>
                <div className="small"><b>{VERDICT[readiness.verdict].label}</b><div>recall {pct(evalEntry!.overall.recall)} · precision {pct(evalEntry!.overall.precision)} on independent test</div></div>
              </div>
            ) : (
              <Banner kind="warn"><span className="small">Not verified yet. Run the independent test before relying on this model.</span></Banner>
            )}
            <div className="small muted">Alerts this session: <b>{alerts}</b></div>
          </div>
          <div className="card flat col small">
            <h4>Run offline on a laptop</h4>
            <div className="ink2">Download the ONNX file and run the camera client, no network needed:</div>
            <code className="tiny" style={{ background: "var(--color-surface-2)", padding: 8, borderRadius: 8, wordBreak: "break-all" }}>
              python tools/webcam_client.py --onnx best.onnx --names {classes.join(",")}
            </code>
          </div>
        </div>
      </div>
    </div>
  );
}
