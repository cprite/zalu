// Client-side frame extraction for previews and the Data Coach. Real measurements, not mocked:
// brightness is the mean luma of a downscaled copy of each frame.

export interface Frame {
  name: string;        // image key used in Annotation.image (file name, or "<video>#t=<sec>")
  url: string;         // object/data URL for display
  width: number;
  height: number;
  brightness: number;  // 0..255
  source: string;      // original file name
}

const cache = new WeakMap<File, Promise<Frame[]>>();

export function extractFrames(files: File[], perVideo = 8): Promise<Frame[]> {
  return Promise.all(files.map((f) => {
    let p = cache.get(f);
    if (!p) {
      p = f.type.startsWith("video/") ? fromVideo(f, perVideo) : fromImage(f);
      cache.set(f, p);
    }
    return p;
  })).then((xs) => xs.flat());
}

function luma(src: CanvasImageSource, w: number, h: number): number {
  const c = document.createElement("canvas");
  c.width = 32;
  c.height = Math.max(1, Math.round((32 * h) / w));
  const ctx = c.getContext("2d")!;
  ctx.drawImage(src, 0, 0, c.width, c.height);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 4) s += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  return s / (d.length / 4);
}

async function fromImage(f: File): Promise<Frame[]> {
  const url = URL.createObjectURL(f);
  try {
    const bmp = await createImageBitmap(f);
    const fr = { name: f.name, url, width: bmp.width, height: bmp.height, brightness: luma(bmp, bmp.width, bmp.height), source: f.name };
    bmp.close();
    return [fr];
  } catch {
    return [];
  }
}

function fromVideo(f: File, n: number): Promise<Frame[]> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.muted = true;
    v.preload = "auto";
    v.src = URL.createObjectURL(f);
    const out: Frame[] = [];
    v.onerror = () => resolve(out);
    v.onloadedmetadata = async () => {
      const dur = v.duration || 1;
      const c = document.createElement("canvas");
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d")!;
      for (let k = 0; k < n; k++) {
        const t = (dur * (k + 0.5)) / n;
        await new Promise<void>((r) => { v.onseeked = () => r(); v.currentTime = t; });
        ctx.drawImage(v, 0, 0);
        out.push({
          name: `${f.name}#t=${t.toFixed(1)}`, url: c.toDataURL("image/jpeg", 0.7),
          width: c.width, height: c.height, brightness: luma(c, c.width, c.height), source: f.name,
        });
      }
      URL.revokeObjectURL(v.src);
      resolve(out);
    };
  });
}

/** A frame served by the backend (GET /api/files), with measured brightness. */
export async function frameFromBlob(name: string, blob: Blob, source: string): Promise<Frame> {
  const bmp = await createImageBitmap(blob);
  const fr = { name, url: URL.createObjectURL(blob), width: bmp.width, height: bmp.height, brightness: luma(bmp, bmp.width, bmp.height), source };
  bmp.close();
  return fr;
}

/** Grab one JPEG frame from a playing <video> (webcam). */
export function grabJpeg(video: HTMLVideoElement, quality = 0.8): Promise<Blob | null> {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d")!.drawImage(video, 0, 0);
  return new Promise((r) => c.toBlob(r, "image/jpeg", quality));
}
