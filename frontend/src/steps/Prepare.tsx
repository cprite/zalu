"use client";
import { useEffect, useMemo, useState } from "react";
import { Annotator, classColor } from "@/components/Annotator";
import { Banner, ErrorBox, SampleBadge, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import type { Annotation } from "@/lib/types";

/** Stage 1 · Describe → prepare / verify the auto-labels. One primary action: continue to training. */
export function Prepare() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [cur, setCur] = useState(0);
  const ds = s.trainDataset!;
  const classes = s.task?.classes.length ? s.task.classes : ds.classes;

  useEffect(() => {
    if (!ds || (s.labels.length && s.frames.length)) return;
    setLoading(true);
    Promise.all([api.getLabels(ds.id), api.listFrames(ds.id)])
      .then(([labels, frames]) => {
        set({ labels, frames });
        const firstLabelled = frames.findIndex((f) => labels.some((l) => l.image === f.name));
        if (firstLabelled > 0) setCur(firstLabelled);  // open on a frame that has something to check
      })
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
  const edited = s.labels.some((l) => l.source === "human");

  const setImageLabels = (img: string, next: Annotation[]) =>
    set((p) => ({ labels: [...p.labels.filter((l) => l.image !== img), ...next], labelsSaved: false }));

  /** Corrections are saved automatically when the user moves on. */
  async function next() {
    setErr(null);
    if (edited && !s.labelsSaved) {
      setSaving(true);
      try {
        await api.saveLabels(ds.id, s.labels);
        set({ labelsSaved: true });
      } catch (e) {
        setErr(e);
        setSaving(false);
        return;
      }
      setSaving(false);
    }
    set({ step: 3 });
  }

  return (
    <div className="wide-col fade-in" style={{ maxWidth: 880 }}>
      <div className="hero compact">
        <h1>Check the labels</h1>
        <p>ZALU marked every frame. Fix anything that looks wrong: the model learns from these boxes.</p>
        <SampleBadge show={api.mode === "mock"} label="Sample labels" />
      </div>

      {loading && <p className="hint" role="status"><Spinner /> Loading frames…</p>}
      <ErrorBox error={err} />
      {!loading && !s.frames.length && !err && (
        <Banner kind="info">No frames to preview for this dataset. You can continue to training.</Banner>
      )}

      {frame && (
        <div className="card col">
          <div className="row small">
            <span className="muted">Frame {cur + 1} of {s.frames.length}</span>
            <div className="spacer" />
            <button className="btn sm" aria-label="Previous frame" disabled={cur === 0} onClick={() => setCur(cur - 1)}>←</button>
            <button className="btn sm" aria-label="Next frame" disabled={cur >= s.frames.length - 1} onClick={() => setCur(cur + 1)}>→</button>
          </div>
          <Annotator frame={frame} labels={byImage.get(frame.name) ?? []} classes={classes} onChange={(n) => setImageLabels(frame.name, n)} />
        </div>
      )}

      {s.frames.length > 1 && (
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

      <div className="actions-center" style={{ marginTop: 8 }}>
        <button className="btn ghost" onClick={() => set({ step: 1 })}>← Footage</button>
        <button className="btn primary" disabled={saving || loading} onClick={next}>{saving && <Spinner />}{edited && !s.labelsSaved ? "Save and train →" : "Looks good, train →"}</button>
      </div>
    </div>
  );
}
