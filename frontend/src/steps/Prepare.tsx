"use client";
import { useEffect, useMemo, useState } from "react";
import { Annotator, classColor } from "@/components/Annotator";
import { Banner, ErrorBox, SampleBadge, SectionHead, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { coach } from "@/lib/coach";
import { useFlow } from "@/lib/flow";
import type { Annotation } from "@/lib/types";

export function Prepare() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [saveErr, setSaveErr] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [cur, setCur] = useState(0);
  const ds = s.trainDataset!;
  const classes = s.task?.classes.length ? s.task.classes : ds.classes;

  useEffect(() => {
    if (!ds || (s.labels.length && s.frames.length)) return;
    setLoading(true);
    Promise.all([api.getLabels(ds.id), api.listFrames(ds.id)])
      .then(([labels, frames]) => set({ labels, frames }))
      .catch(setErr)
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds?.id]);

  const byImage = useMemo(() => {
    const m = new Map<string, Annotation[]>();
    s.labels.forEach((l) => m.set(l.image, [...(m.get(l.image) ?? []), l]));
    return m;
  }, [s.labels]);
  const frame = s.frames[cur];
  const tips = s.task ? coach({ ...s.task, classes }, s.frames, s.labels, ds.source_ids) : [];
  const human = s.labels.filter((l) => l.source === "human").length;

  const setImageLabels = (img: string, next: Annotation[]) =>
    set((p) => ({ labels: [...p.labels.filter((l) => l.image !== img), ...next], labelsSaved: false }));

  async function save() {
    setSaving(true);
    setSaveErr(null);
    try {
      await api.saveLabels(ds.id, s.labels);
      set({ labelsSaved: true });
    } catch (e) {
      setSaveErr(e);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fade-in">
      <SectionHead title="Check the automatic labels"
        sub="An open-vocabulary detector labelled every frame from your description. Fix anything wrong: these labels are what the model learns from."
        right={<SampleBadge show={api.mode === "mock"} label="Sample auto-labels" />} />
      <div className="split">
        <div className="col">
          {loading && <div className="card row"><Spinner /> Loading frames and auto-labels…</div>}
          <ErrorBox error={err} />
          {!s.frames.length && !loading && (
            <Banner kind="info">No frames to preview for this dataset (for example a dataset registered directly on the server). You can go straight to training.</Banner>
          )}
          {frame && (
            <div className="card col">
              <div className="row small">
                <b>{frame.name}</b><span className="muted">frame {cur + 1} of {s.frames.length}</span>
                <div className="spacer" />
                <button className="btn sm" aria-label="Previous frame" disabled={cur === 0} onClick={() => setCur(cur - 1)}>←</button>
                <button className="btn sm" aria-label="Next frame" disabled={cur >= s.frames.length - 1} onClick={() => setCur(cur + 1)}>→</button>
              </div>
              <Annotator frame={frame} labels={byImage.get(frame.name) ?? []} classes={classes} onChange={(n) => setImageLabels(frame.name, n)} />
            </div>
          )}
          {s.frames.length > 0 && (
            <div className="thumbs">
              {s.frames.map((f, i) => {
                const ls = byImage.get(f.name) ?? [];
                return (
                  <button key={f.name} className={`thumb ${i === cur ? "sel" : ""}`} onClick={() => setCur(i)} title={f.name}
                    aria-label={`Frame ${i + 1}: ${ls.length ? `${ls.length} box(es)` : "nothing labelled"}`} aria-current={i === cur ? "true" : undefined}>
                    <svg viewBox={`0 0 ${f.width} ${f.height}`} preserveAspectRatio="xMidYMid slice">
                      <image href={f.url} width={f.width} height={f.height} />
                      {ls.map((l, k) => <rect key={k} x={l.box[0]} y={l.box[1]} width={l.box[2] - l.box[0]} height={l.box[3] - l.box[1]} fill="none" stroke={classColor(classes, l.cls)} strokeWidth={f.width / 120} />)}
                    </svg>
                    <span className="count">{ls.length || "empty"}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="col">
          <div className="card col">
            <h4>Dataset</h4>
            <div className="small ink2">
              <div>{s.frames.length || Object.values(ds.counts).reduce((a, b) => a + b, 0)} frames · {s.labels.length} boxes · {human} checked by you</div>
              {classes.map((c) => <div key={c}><span className="dot" style={{ background: classColor(classes, c) }} /> {c}: {s.labels.filter((l) => l.cls === c).length}</div>)}
              <div className="tiny muted" style={{ marginTop: 4 }}>Recording: {ds.source_ids.join(", ") || "—"} · {ds.id}</div>
            </div>
            <ErrorBox error={saveErr} />
            <button className="btn primary" onClick={() => set({ step: 3 })}>Continue to training →</button>
            <button className="btn" disabled={saving || !s.labels.length} onClick={save}>{saving && <Spinner />}{s.labelsSaved ? "✓ Corrections saved" : "Save corrections"}</button>
          </div>
          <div className="card col">
            <div className="row"><h4>Data coach</h4><span className="badge exp">Rule-based</span></div>
            {tips.length === 0 && <div className="small muted">Tips appear once frames are loaded.</div>}
            {tips.map((t) => (
              <div key={t.title} className="small">
                <div><span aria-hidden className={t.level === "warn" ? "text-warning-ink font-bold" : t.level === "ok" ? "text-success-ink font-bold" : "text-brand-ink font-bold"}>{t.level === "warn" ? "!" : t.level === "ok" ? "✓" : "i"}</span> <span className="sr-only">{t.level === "warn" ? "Needs attention:" : t.level === "ok" ? "Good:" : "Tip:"}</span> <b>{t.title}</b></div>
                <div className="ink2" style={{ paddingLeft: 14 }}>{t.detail}</div>
              </div>
            ))}
            <div className="tiny muted">Based on your frames (lighting measured from pixels) and labels. Suggestions from the Data Engine will appear here too.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
