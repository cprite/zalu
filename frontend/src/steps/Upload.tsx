"use client";
import { useEffect, useState } from "react";
import { MediaInput } from "@/components/MediaInput";
import { Banner, ErrorBox, SectionHead, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { makeSampleFrames } from "@/lib/sampleFrames";
import type { DatasetManifest } from "@/lib/types";

export function Upload() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [existing, setExisting] = useState<DatasetManifest[]>([]);

  useEffect(() => {
    if (api.mode === "live") api.listDatasets().then(setExisting).catch(() => setExisting([]));
  }, [api]);

  async function submit() {
    if (!s.task) return;
    setBusy(true);
    setErr(null);
    try {
      const ds = await api.createDataset(s.task.id, s.trainFiles, { session_id: s.trainSession, role: "train" });
      set({ trainDataset: ds, frames: [], labels: [], labelsSaved: false, step: 2 });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fade-in">
      <SectionHead title="Upload footage from your camera"
        sub="Use footage from the place where the camera will actually run. A few minutes is enough: we pick about 40 varied frames." />
      <div className="split">
        <div className="card col">
          <label className="field">Recording name
            <input className="input" value={s.trainSession} onChange={(e) => set({ trainSession: e.target.value.replace(/\s+/g, "_") })} />
            <span className="tiny muted" style={{ fontWeight: 400 }}>Each recording is kept together, so training and testing never share footage from the same session.</span>
          </label>
          <MediaInput files={s.trainFiles} onFiles={(trainFiles) => set({ trainFiles })} sessionId={s.trainSession} />
          {api.mode === "mock" && !s.trainFiles.length && (
            <button className="btn sm" style={{ alignSelf: "flex-start" }} onClick={async () => set({ trainFiles: await makeSampleFrames(36, s.trainSession) })}>
              No footage at hand? Use generated sample frames
            </button>
          )}
          <ErrorBox error={err} />
          {busy && api.mode === "live" && (
            <Banner kind="info">Uploading and processing on the GPU server: frame extraction and auto-labelling can take a few minutes for longer videos. Keep this tab open.</Banner>
          )}
          <div className="row">
            <button className="btn" onClick={() => set({ step: 0 })}>← Back</button>
            <div className="spacer" />
            <button className="btn primary" disabled={!s.trainFiles.length || busy} onClick={submit}>{busy && <Spinner />}{busy ? "Processing…" : "Prepare dataset →"}</button>
          </div>
        </div>
        <div className="col">
          <div className="card flat col small">
            <h4>Tips for good footage</h4>
            <div>• Show the situations you care about <b>and</b> ones you don&apos;t (empty scene, look-alikes).</div>
            <div>• Vary distance, angle and lighting.</div>
            <div>• Keep one recording aside: you&apos;ll use <b>new</b> footage to test the model honestly.</div>
          </div>
          {api.mode === "live" && existing.length > 0 && (
            <div className="card flat col small">
              <h4>Or use a dataset already on the server</h4>
              <select className="select" defaultValue="" onChange={(e) => {
                const ds = existing.find((d) => d.id === e.target.value);
                if (ds) set({ trainDataset: ds, frames: [], labels: [], step: 2 });
              }}>
                <option value="" disabled>Choose…</option>
                {existing.map((d) => <option key={d.id} value={d.id}>{d.id} · {d.variant} · {Object.entries(d.counts).map(([k, v]) => `${k} ${v}`).join(", ")}</option>)}
              </select>
            </div>
          )}
          {api.mode === "mock" && <Banner kind="sample"><span className="small">Sample mode: files stay in your browser. Auto-labels in the next step are generated samples.</span></Banner>}
        </div>
      </div>
    </div>
  );
}
