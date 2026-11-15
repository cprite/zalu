"use client";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Annotator, classColor } from "@/components/Annotator";
import { ScreenActivity } from "@/components/ScreenActivity";
import { Banner, ErrorBox, Spinner } from "@/components/ui";
import { useApi } from "@/lib/api";
import { useFlow } from "@/lib/flow";
import { getPrefs } from "@/lib/prefs";
import { t, useLessMotion } from "@/lib/motion";
import type { Annotation } from "@/lib/types";

/** Stage 1 · Describe → prepare / verify the auto-labels. One primary action: continue to training. */
export function Prepare() {
  const { api } = useApi();
  const { s, set } = useFlow();
  const [err, setErr] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  const [cur, setCur] = useState(0);
  const ds = s.trainDataset!;
  const loading = !!ds && !s.annotationsLoaded && !err;  // derived: no setState inside the loading effect
  const reduce = useLessMotion();
  const strip = useRef<HTMLDivElement>(null);
  const classes = s.task?.classes.length ? s.task.classes : ds.classes;

  useEffect(() => {
    if (!ds || s.annotationsLoaded) return;
    let active = true;
    Promise.all([api.getLabels(ds.id), api.listFrames(ds.id)])
      .then(([labels, frames]) => {
        if (!active) return;
        set({ labels, frames, annotationsLoaded: true });
        const firstLabelled = frames.findIndex((f) => labels.some((l) => l.image === f.name));
        if (firstLabelled > 0) setCur(firstLabelled);  // open on a frame that has something to check
      })
      .catch((e) => { if (active) setErr(e); });
    return () => { active = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds?.id]);

  const byImage = useMemo(() => {
    const m = new Map<string, Annotation[]>();
    s.labels.forEach((l) => m.set(l.image, [...(m.get(l.image) ?? []), l]));
    return m;
  }, [s.labels]);
  const frame = s.frames[cur];
  const total = s.frames.length;

  // ← / → move between frames (ignored while typing or choosing a class)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!total || !getPrefs().keyboardShortcuts || (e.target as HTMLElement).closest?.("header, [role=dialog], [contenteditable]")) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.altKey || e.metaKey || e.ctrlKey) return;
      if (e.key === "ArrowRight") { e.preventDefault(); setCur((c) => Math.min(total - 1, c + 1)); }
      if (e.key === "ArrowLeft") { e.preventDefault(); setCur((c) => Math.max(0, c - 1)); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [total]);
  useEffect(() => {
    strip.current?.querySelector<HTMLElement>(`[data-i="${cur}"]`)?.scrollIntoView({ block: "nearest", inline: "center", behavior: reduce ? "auto" : "smooth" });
  }, [cur, reduce]);
  const edited = s.annotationsDirty;

  const setImageLabels = (img: string, next: Annotation[]) =>
    set((p) => ({ labels: [...p.labels.filter((l) => l.image !== img), ...next], labelsSaved: false, annotationsDirty: true }));

  /** Corrections are saved automatically when the user moves on. */
  async function next() {
    setErr(null);
    if (edited && !s.labelsSaved) {
      setSaving(true);
      try {
        await api.saveLabels(ds.id, s.labels);
        set({ labelsSaved: true, annotationsDirty: false });
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
    <div className="wide-col" style={{ maxWidth: 880 }}>
      <div className="hero compact">
        <h1>Check the labels</h1>
        <p>Review the boxes and correct any mistakes before training.</p>
        {api.mode === "mock" && <span className="small muted">Demo labels</span>}
      </div>

      <ScreenActivity message={loading ? "Loading frames and labels" : saving ? "Saving labels" : undefined} />
      <ErrorBox error={err} />
      {!loading && !s.frames.length && !err && (
        <Banner kind="info">No frames to preview for this dataset. You can continue to training.</Banner>
      )}

      {frame && (
        <div className="card col">
          <div className="row small">
            <span className="muted" aria-live="polite">Frame {cur + 1} of {s.frames.length}</span>
            <div className="spacer" />
            <span className="tiny muted" aria-hidden>← → to browse</span>
            <motion.button className="btn sm" aria-label="Previous frame" aria-keyshortcuts="ArrowLeft" disabled={cur === 0} onClick={() => setCur(cur - 1)}
              whileTap={reduce ? undefined : { x: -2 }} transition={t("fast")}>←</motion.button>
            <motion.button className="btn sm" aria-label="Next frame" aria-keyshortcuts="ArrowRight" disabled={cur >= s.frames.length - 1} onClick={() => setCur(cur + 1)}
              whileTap={reduce ? undefined : { x: 2 }} transition={t("fast")}>→</motion.button>
          </div>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={frame.name} initial={{ opacity: 0.35 }} animate={{ opacity: 1 }} exit={{ opacity: 0.35 }} transition={t("fast")}>
              <Annotator frame={frame} labels={byImage.get(frame.name) ?? []} classes={classes} onChange={(n) => setImageLabels(frame.name, n)} />
            </motion.div>
          </AnimatePresence>
        </div>
      )}

      {s.frames.length > 1 && (
        <div className="thumbs" ref={strip}>
          {s.frames.map((f, i) => {
            const ls = byImage.get(f.name) ?? [];
            return (
              <button key={f.name} data-i={i} className={`thumb ${i === cur ? "sel" : ""}`} onClick={() => setCur(i)} title={f.name}
                aria-label={`Frame ${i + 1}: ${ls.length ? `${ls.length} box(es)` : "empty"}`} aria-current={i === cur ? "true" : undefined}>
                <svg viewBox={`0 0 ${f.width} ${f.height}`} preserveAspectRatio="xMidYMid slice">
                  <image href={f.url} width={f.width} height={f.height} />
                  {ls.map((l, k) => <rect key={k} x={l.box[0]} y={l.box[1]} width={l.box[2] - l.box[0]} height={l.box[3] - l.box[1]} fill="none" stroke={classColor(classes, l.cls)} strokeWidth={f.width / 120} />)}
                </svg>
                <span className="count">{ls.length || "empty"}</span>
                {i === cur && <motion.span layoutId="thumb-ring" className="thumb-ring" transition={t("base")} aria-hidden />}
              </button>
            );
          })}
        </div>
      )}

      <div className="actions-center" style={{ marginTop: 8 }}>
        <button className="btn ghost" onClick={() => set({ step: 1 })}>← Change footage</button>
        <button className="btn primary" disabled={saving || loading} onClick={next}>{saving && <Spinner />}{edited && !s.labelsSaved ? "Save and train →" : "Looks good, train →"}</button>
      </div>
    </div>
  );
}
