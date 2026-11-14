"use client";
// Message-box style input: text plus image/video attachments, one send button.
import { useEffect, useRef, useState } from "react";
import { Spinner } from "./ui";

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

  const add = (list: FileList | File[] | null) => {
    if (!list || busy) return;
    const ok = [...list].filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"));
    if (ok.length) onFiles([...files, ...ok]);
  };

  useEffect(() => {  // auto-grow like a chat box
    const t = area.current;
    if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(200, t.scrollHeight)}px`;
  }, [value]);

  const [expanded, setExpanded] = useState(false);
  const [previews, setPreviews] = useState<{ f: File; url: string | null }[]>([]);
  useEffect(() => {
    const next = (expanded ? files : files.slice(0, MAX_PREVIEWS)).map((f) => ({
      f, url: f.type.startsWith("image/") ? URL.createObjectURL(f) : null,
    }));
    setPreviews(next);
    return () => next.forEach((preview) => preview.url && URL.revokeObjectURL(preview.url));
  }, [expanded, files]);

  return (
    <div className={`composer ${over ? "over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
      {files.length > 0 && (
        <ul className="attachments" aria-label="Attached footage" style={{ listStyle: "none", margin: 0, padding: 0 }}>
          {previews.map(({ f, url }, i) => (
            <li key={f.name + i} className="attachment" title={f.name}>
              {url ? <img src={url} alt="" /> : <span>🎬<br />{f.name.slice(0, 14)}</span>}
              <button className="x" aria-label={`Remove ${f.name}`} disabled={busy} onClick={() => onFiles(files.filter((x) => x !== f))}>×</button>
            </li>
          ))}
          {files.length > MAX_PREVIEWS && <li><button type="button" className="btn sm" aria-expanded={expanded}
            onClick={() => setExpanded(!expanded)}>{expanded ? "Show fewer files" : `Show all ${files.length} files`}</button></li>}
        </ul>
      )}
      {!textless && (
        <>
          <label htmlFor="composer-text" className="sr-only">{placeholder}</label>
          <textarea id="composer-text" ref={area} rows={1} disabled={busy} value={value} placeholder={placeholder} autoFocus={autoFocus}
            onChange={(e) => onChange?.(e.target.value)}
            onPaste={(e) => add(e.clipboardData.files)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && canSubmit && !busy) { e.preventDefault(); onSubmit(); } }} />
        </>
      )}
      <div className="composer-bar">
        <button type="button" className="icon-btn" aria-label="Attach images or video" title="Attach images or video" disabled={busy} onClick={() => input.current?.click()}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
        <input ref={input} type="file" multiple accept="image/*,video/*" hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
        <span className="small muted" style={{ flex: 1 }}>
          {files.length ? `${files.length} file${files.length > 1 ? "s" : ""} attached` : textless ? "Drop images or a video, or press +" : "Add footage now or in the next step"}
        </span>
        <button type="button" className="icon-btn send" aria-label={submitLabel} title={submitLabel} disabled={!canSubmit || busy} onClick={onSubmit}>
          {busy ? <Spinner /> : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
