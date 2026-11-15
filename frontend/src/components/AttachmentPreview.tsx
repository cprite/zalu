"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./attachmentPreview.module.css";

type Media = { url: string; thumbnail?: string; detail?: string };

function useLocalMedia(file: File, isVideo: boolean) {
  const [media, setMedia] = useState<Media | null>(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    if (!isVideo) {
      const img = new Image();
      img.onload = () => setMedia({ url, thumbnail: url, detail: `${img.naturalWidth} × ${img.naturalHeight}` });
      img.onerror = () => setMedia({ url });
      img.src = url;
      return () => {
        img.onload = img.onerror = null;
        img.removeAttribute("src");
        URL.revokeObjectURL(url);
      };
    }

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    let settled = false;
    const releaseDecoder = () => {
      video.onloadeddata = video.onerror = null;
      video.removeAttribute("src");
      video.load();
    };
    const finish = (thumbnail?: string, detail?: string) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      releaseDecoder();
      setMedia({ url, thumbnail, detail });
    };
    const timeout = window.setTimeout(() => finish(), 4000);
    video.onerror = () => finish();
    video.onloadeddata = () => {
      const { videoWidth, videoHeight, duration } = video;
      if (!videoWidth || !videoHeight) { finish(); return; }
      const canvas = document.createElement("canvas");
      canvas.width = Math.min(320, videoWidth);
      canvas.height = Math.max(1, Math.round(canvas.width * videoHeight / videoWidth));
      const context = canvas.getContext("2d");
      const detail = `${videoWidth} × ${videoHeight}${Number.isFinite(duration) ? ` · ${duration.toFixed(1)} s` : ""}`;
      try {
        context?.drawImage(video, 0, 0, canvas.width, canvas.height);
        finish(context ? canvas.toDataURL("image/jpeg", 0.82) : undefined, detail);
      } catch {
        finish(undefined, detail);
      }
    };
    video.src = url;
    return () => {
      clearTimeout(timeout);
      if (!settled) releaseDecoder();
      URL.revokeObjectURL(url);
    };
  }, [file, isVideo]);

  return media;
}

export function AttachmentPreview({ file, busy, onRemove }: { file: File; busy: boolean; onRemove: () => void }) {
  const isVideo = file.type.startsWith("video/");
  const media = useLocalMedia(file, isVideo);
  const dialog = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const metadata = [media?.detail, `${Math.max(1, Math.round(file.size / 1024))} KB`].filter(Boolean).join(" · ");

  function preview() {
    dialog.current?.showModal();
    setFailed(false);
    setOpen(true);
  }

  return <>
    <button type="button" className={styles.tile} onClick={preview} disabled={busy || !media}
      aria-label={`Preview ${file.name}`} title={`${file.name} · ${metadata}`}>
      {media?.thumbnail ? (
        // eslint-disable-next-line @next/next/no-img-element -- browser-generated local thumbnail
        <img className={styles.thumbnail} src={media.thumbnail} alt="" />
      ) : <span className={styles.fallback} aria-hidden>
        {isVideo ? "▷" : "▧"}<span>{file.name}</span>
      </span>}
      {isVideo && media?.thumbnail && <span className={styles.play} aria-hidden>▶</span>}
    </button>
    <button type="button" className={styles.remove} aria-label={`Remove ${file.name}`} disabled={busy} onClick={onRemove}>×</button>
    <dialog ref={dialog} className={styles.dialog} aria-label={`Preview ${file.name}`} onClose={() => setOpen(false)}>
      {open && media && <>
        <div className={styles.heading}>
          <div><strong>{file.name}</strong><p className="small muted">{metadata}</p></div>
          <button type="button" autoFocus className="btn sm" onClick={() => dialog.current?.close()}>Close</button>
        </div>
        {failed || (!isVideo && !media.thumbnail) ? <p className="muted">Preview unavailable. The file is still attached.</p>
          : isVideo ? (
            <video className={styles.media} src={media.url} controls playsInline onError={() => setFailed(true)} />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element -- local blob preview
            <img className={styles.media} src={media.url} alt={file.name} />
          )}
      </>}
    </dialog>
  </>;
}
