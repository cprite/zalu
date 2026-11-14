"use client";
// File drop + camera capture. Camera capture records JPEG frames at a fixed rate and returns them as Files.
import { useEffect, useRef, useState } from "react";

export function MediaInput({ files, onFiles, sessionId }: { files: File[]; onFiles: (f: File[]) => void; sessionId: string }) {
  const [tab, setTab] = useState<"files" | "camera">("files");
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const add = (list: FileList | null) => {
    if (!list) return;
    const ok = [...list].filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"));
    onFiles([...files, ...ok]);
  };
  const vids = files.filter((f) => f.type.startsWith("video/")).length;

  return (
    <div className="col">
      <div className="tabs" role="group" aria-label="Source">
        <button aria-pressed={tab === "files"} className={tab === "files" ? "on" : ""} onClick={() => setTab("files")}>Images or video</button>
        <button aria-pressed={tab === "camera"} className={tab === "camera" ? "on" : ""} onClick={() => setTab("camera")}>Use my camera</button>
      </div>
      {tab === "files" ? (
        <button type="button" className={`drop ${over ? "over" : ""}`} onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}>
          <span style={{ fontSize: 28, display: "block" }} aria-hidden>⇪</span>
          <b>Drop images or a video here</b>
          <span className="small muted" style={{ display: "block" }}>or press to choose files · JPG, PNG, MP4, MOV · a few minutes of footage is enough</span>
          <input ref={inputRef} type="file" multiple accept="image/*,video/*" hidden onChange={(e) => add(e.target.files)} />
        </button>
      ) : (
        <CameraCapture sessionId={sessionId} onFrames={(fs) => onFiles([...files, ...fs])} />
      )}
      {files.length > 0 && (
        <div className="row small" role="status">
          <span>✓ <b>{files.length}</b> file(s) added{vids ? ` · ${vids} video(s), frames are sampled automatically` : ""}</span>
          <div className="spacer" />
          <button className="btn ghost sm danger" onClick={() => onFiles([])}>Clear</button>
        </div>
      )}
    </div>
  );
}

function CameraCapture({ onFrames, sessionId }: { onFrames: (f: File[]) => void; sessionId: string }) {
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [rec, setRec] = useState(0);
  const timer = useRef<number | null>(null);
  const buf = useRef<File[]>([]);

  useEffect(() => () => { stream?.getTracks().forEach((t) => t.stop()); if (timer.current) clearInterval(timer.current); }, [stream]);

  async function start() {
    setErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
      setStream(s);
      if (video.current) { video.current.srcObject = s; await video.current.play(); }
    } catch (e) {
      setErr(`Camera not available: ${(e as Error).message}`);
    }
  }
  function grab() {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    c.toBlob((b) => {
      if (!b) return;
      buf.current.push(new File([b], `${sessionId}_cam_${Date.now()}.jpg`, { type: "image/jpeg" }));
      setRec(buf.current.length);
    }, "image/jpeg", 0.85);
  }
  function toggleRec() {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
      onFrames(buf.current);
      buf.current = [];
      setRec(0);
    } else {
      timer.current = window.setInterval(grab, 500);
      setRec(0.0001);
    }
  }
  return (
    <div className="col">
      <div className="stage" style={{ aspectRatio: "16/9" }}>
        <video ref={video} muted playsInline />
        {!stream && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}><button className="btn primary" onClick={start}>Turn on camera</button></div>}
      </div>
      {err && <div className="small negv">{err}</div>}
      {stream && (
        <div className="row">
          <button className="btn sm" onClick={() => { grab(); setTimeout(() => { onFrames(buf.current); buf.current = []; setRec(0); }, 200); }}>Snap one frame</button>
          <button className={`btn sm ${timer.current ? "danger" : "primary"}`} onClick={toggleRec}>{timer.current ? `■ Stop (${Math.floor(rec)} frames)` : "● Record frames (2 per second)"}</button>
          <span className="small muted">Move around: different distances and angles make a better model.</span>
        </div>
      )}
    </div>
  );
}
