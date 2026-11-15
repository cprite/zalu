"use client";
// Message-box style input: text plus image/video attachments, one send button.
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { Dots, pressIcon, t, usePointerFine } from "@/lib/motion";
import { AttachmentPreview } from "./AttachmentPreview";

const MAX_PREVIEWS = 7;

export function Composer({
  value, onChange, files, onFiles, onSubmit, placeholder, busy = false, canSubmit, submitLabel = "Send", textless = false, autoFocus = false,
}: {
  value?: string;
  onChange?: (v: string) => void;
  files: File[];
  onFiles: (f: File[]) => void;
  onSubmit: () => void;
  placeholder?: string;
  busy?: boolean;
  canSubmit: boolean;
  submitLabel?: string;
  textless?: boolean;
  autoFocus?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const [over, setOver] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const pointerFine = usePointerFine();

  const add = (list: FileList | File[] | null) => {
    if (!list || busy) return;
    const ok = [...list].filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"));
    if (ok.length) onFiles([...files, ...ok]);
  };

  useEffect(() => {  // auto-grow like a chat box
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [value]);

  const previews = expanded ? files : files.slice(0, MAX_PREVIEWS);

  return (
    <div className={`composer ${over ? "over" : ""} ${busy ? "busy" : ""}`} style={{ position: "relative" }} aria-busy={busy}
      onDragOver={(e) => { e.preventDefault(); if (!busy) setOver(true); }}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOver(false); }}
      onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
      <AnimatePresence>
        {over && (
          <motion.div className="drop-hint" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: t("fast") }} transition={t("base")} aria-hidden>
            <span className="row" style={{ gap: 8 }}>
              <motion.svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
                initial={{ y: -6, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={t("slow")}>
                <path d="M12 4v12M7 11l5 5 5-5M5 20h14" />
              </motion.svg>
              Drop to attach
            </span>
          </motion.div>
        )}
      </AnimatePresence>
      {files.length > 0 && (
        <ul className="attachments" aria-label="Attached footage" style={{ listStyle: "none", margin: 0, padding: 0 }}>
          <AnimatePresence initial={false}>
            {previews.map((f, index) => (
              <motion.li key={`${index}-${f.name}-${f.size}-${f.lastModified}`} layout className="attachment" title={f.name}
                initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.92 }} transition={t("base")}>
                <AttachmentPreview file={f} busy={busy} onRemove={() => onFiles(files.filter((_, i) => i !== index))} />
              </motion.li>
            ))}
          </AnimatePresence>
          {files.length > MAX_PREVIEWS && <li><button type="button" className="btn sm" aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}>{expanded ? "Show fewer files" : `Show all ${files.length} files`}</button></li>}
        </ul>
      )}
      {!textless && (
        <>
          <label htmlFor="composer-text" className="sr-only">{placeholder}</label>
          <textarea id="composer-text" ref={area} rows={1} disabled={busy} value={value} placeholder={placeholder} autoFocus={autoFocus && pointerFine}
            onChange={(e) => onChange?.(e.target.value)}
            onPaste={(e) => add(e.clipboardData.files)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && canSubmit && !busy) { e.preventDefault(); onSubmit(); } }} />
        </>
      )}
      <div className="composer-bar">
        <motion.button type="button" whileTap={pressIcon} transition={t("fast")} className="icon-btn" aria-label="Attach images or video" title="Attach images or video" disabled={busy} onClick={() => input.current?.click()}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M12 5v14M5 12h14" />
          </svg>
        </motion.button>
        <input ref={input} type="file" multiple accept="image/*,video/*" hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
        <span className="small muted" style={{ flex: 1 }} aria-live="polite">
          {files.length ? `${files.length} file${files.length > 1 ? "s" : ""} attached` : textless ? "Attach images or video" : "Attach footage"}
        </span>
        <motion.button type="button" whileTap={canSubmit && !busy ? pressIcon : undefined} transition={t("fast")}
          className="icon-btn send" aria-label={submitLabel} title={submitLabel} disabled={!canSubmit || busy} onClick={onSubmit}>
          <AnimatePresence mode="wait" initial={false}>
            {busy ? (
              <motion.span key="busy" initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.6 }} transition={t("fast")} style={{ display: "grid" }}><Dots /></motion.span>
            ) : (
              <motion.svg key="send" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={t("fast")}
                width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M12 19V5M5 12l7-7 7 7" />
              </motion.svg>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
    </div>
  );
}
