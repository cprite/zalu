"use client";
// Box editor: drag to draw, click to select, change class, delete, confirm. Boxes are in image pixels.
import { motion } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { getPrefs } from "@/lib/prefs";
import { t } from "@/lib/motion";
import type { Frame } from "@/lib/frames";
import type { Annotation } from "@/lib/types";

const PALETTE = ["#3651f5", "#e5484d", "#2fbf71", "#f5a623", "#8b5cf6", "#0ea5e9"];
export const classColor = (classes: string[], c: string) => PALETTE[Math.max(0, classes.indexOf(c)) % PALETTE.length];

export function Annotator({ frame, labels, classes, onChange }: {
  frame: Frame; labels: Annotation[]; classes: string[]; onChange: (next: Annotation[]) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  // selection belongs to a frame: switching frames clears it without an effect
  const [selState, setSelState] = useState<{ frame: string; i: number } | null>(null);
  const sel = selState && selState.frame === frame.name ? selState.i : null;
  const setSel = (i: number | null) => setSelState(i == null ? null : { frame: frame.name, i });
  const [draw, setDraw] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [active, setActive] = useState(classes[0]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!getPrefs().keyboardShortcuts || (e.target as HTMLElement).closest?.("header, [role=dialog], [contenteditable]")) return;
      if (sel == null || (e.target as HTMLElement).closest?.("input, textarea, select")) return;
      if (e.key === "Delete" || e.key === "Backspace") { onChange(labels.filter((_, i) => i !== sel)); setSelState(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sel, labels, onChange]);

  const pt = (cx: number, cy: number) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: ((cx - r.left) / r.width) * frame.width, y: ((cy - r.top) / r.height) * frame.height };
  };

  // Window listeners are attached synchronously on pointerdown, so even a very fast drag keeps its pointerup.
  const drawRef = useRef<typeof draw>(null);
  const latest = useRef({ labels, onChange, active, frame });
  useLayoutEffect(() => { latest.current = { labels, onChange, active, frame }; });
  function beginDraw(cx: number, cy: number) {
    const p = pt(cx, cy);
    drawRef.current = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    setDraw(drawRef.current);
    const move = (e: PointerEvent) => {
      const d = drawRef.current;
      if (!d) return;
      const q = pt(e.clientX, e.clientY);
      const f = latest.current.frame;
      drawRef.current = { ...d, x1: Math.max(0, Math.min(f.width, q.x)), y1: Math.max(0, Math.min(f.height, q.y)) };
      setDraw(drawRef.current);
    };
    const up = (e: PointerEvent) => {
      move(e);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const d = drawRef.current;
      drawRef.current = null;
      setDraw(null);
      const { labels: ls, onChange: oc, active: cls, frame: f } = latest.current;
      if (d && Math.abs(d.x1 - d.x0) > 8 && Math.abs(d.y1 - d.y0) > 8) {
        const box: Annotation["box"] = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
        oc([...ls, { image: f.name, cls, box, score: null, source: "human" }]);
        setSel(ls.length);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  const update = (i: number, p: Partial<Annotation>) => onChange(labels.map((l, k) => (k === i ? { ...l, ...p } : l)));
  const sw = Math.max(2, frame.width / 320);

  return (
    <div className="col">
      <div className="stage">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={frame.url} alt={`Frame ${frame.name} with ${labels.length} labelled box(es)`} draggable={false} />
        <svg ref={svg} className="overlay" viewBox={`0 0 ${frame.width} ${frame.height}`} preserveAspectRatio="none"
          style={{ cursor: "crosshair" }}
          onPointerDown={(e) => {
            if (e.target !== svg.current) return;
            setSel(null);
            beginDraw(e.clientX, e.clientY);
          }}>
          {labels.map((l, i) => {
            const [x1, y1, x2, y2] = l.box;
            const col = classColor(classes, l.cls);
            const unsure = l.source === "auto" && (l.score ?? 1) < 0.5;
            return (
              <g key={i} onPointerDown={(e) => { e.stopPropagation(); setSel(i); }} style={{ cursor: "pointer" }}>
                {/* boxes settle in from their own centre; the selected one thickens smoothly */}
                <motion.rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={i === sel ? `${col}33` : "transparent"} stroke={col}
                  strokeDasharray={unsure ? `${sw * 4} ${sw * 3}` : undefined}
                  style={{ transformBox: "fill-box", transformOrigin: "center" }}
                  initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1, strokeWidth: i === sel ? sw * 1.8 : sw }} transition={t("base")} />
                <text x={x1 + 4} y={Math.max(14, y1 - 6)} className="box-label" style={{ fontSize: frame.width / 55 }}>
                  {l.cls}{l.source === "auto" && l.score != null ? ` ${Math.round(l.score * 100)}%` : l.source === "human" ? " ✓" : ""}
                </text>
              </g>
            );
          })}
          {draw && <rect x={Math.min(draw.x0, draw.x1)} y={Math.min(draw.y0, draw.y1)} width={Math.abs(draw.x1 - draw.x0)} height={Math.abs(draw.y1 - draw.y0)}
            fill="transparent" stroke={classColor(classes, active)} strokeWidth={sw} strokeDasharray={`${sw * 3} ${sw * 2}`} />}
        </svg>
      </div>
      <div className="row small" role="group" aria-label="Class for new boxes">
        <span className="ink2">New box class:</span>
        {classes.map((c) => (
          <button key={c} className={`chip ${c === active ? "accent" : ""}`} aria-pressed={c === active} onClick={() => setActive(c)}>
            <span className="dot" style={{ background: classColor(classes, c) }} aria-hidden />{c}
          </button>
        ))}
      </div>
      {/* Keyboard / screen-reader access to every box on this frame */}
      <ul className="col" style={{ gap: 4, listStyle: "none", padding: 0, margin: 0 }} aria-label={`Boxes on ${frame.name}`}>
        {labels.length === 0 && <li className="small muted">No boxes on this frame. If something should be detected here, drag on the image to add it.</li>}
        {labels.map((l, i) => {
          const unsure = l.source === "auto" && (l.score ?? 1) < 0.5;
          return (
            <motion.li key={i} layout="position" className="row small" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={t("fast")}
              style={{ gap: 8, padding: "4px 8px", borderRadius: 8, background: i === sel ? "var(--color-brand-soft)" : "transparent",
                transition: "background-color .15s cubic-bezier(.2,0,0,1)" }}>
              <button className="chip" aria-pressed={i === sel} onClick={() => setSel(i === sel ? null : i)}>
                <span className="dot" style={{ background: classColor(classes, l.cls) }} aria-hidden />
                Box {i + 1}: {l.cls}
              </button>
              <span className="muted">
                {l.source === "human" ? "✓ checked by you" : `auto, ${Math.round((l.score ?? 0) * 100)}% sure${unsure ? " (unsure)" : ""}`}
              </span>
              {i === sel && (
                <>
                  <div className="spacer" />
                  <label className="sr-only" htmlFor={`cls-${i}`}>Class of box {i + 1}</label>
                  <select id={`cls-${i}`} className="select" style={{ width: 160, minHeight: 32, padding: "4px 8px" }} value={l.cls}
                    onChange={(e) => update(i, { cls: e.target.value, source: "human", score: null })}>
                    {classes.map((c) => <option key={c}>{c}</option>)}
                  </select>
                  <button className="btn sm" onClick={() => update(i, { source: "human", score: null })}>✓ Correct</button>
                  <button className="btn sm danger" onClick={() => { onChange(labels.filter((_, k) => k !== i)); setSel(null); }}>Delete</button>
                </>
              )}
            </motion.li>
          );
        })}
      </ul>
      <p className="tiny muted">Drag on the image to add a box · select a box to fix or delete it</p>
    </div>
  );
}
