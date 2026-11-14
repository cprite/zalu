// Generated placeholder frames so the walkthrough works without footage. Clearly labelled SAMPLE on the image.
// Where each figure was drawn is remembered so the mock auto-labeller can put boxes in plausible places.
export const SAMPLE_TRUTH = new Map<string, { clsIndex: number; box: [number, number, number, number] }[]>();

export async function makeSampleFrames(n = 36, session = "sample"): Promise<File[]> {
  const out: File[] = [];
  for (let i = 0; i < n; i++) {
    const c = document.createElement("canvas");
    c.width = 960; c.height = 540;
    const g = c.getContext("2d")!;
    const hue = 200 + ((i * 37) % 60);
    const grd = g.createLinearGradient(0, 0, 960, 540);
    grd.addColorStop(0, `hsl(${hue} 25% 78%)`);
    grd.addColorStop(1, `hsl(${hue + 20} 20% 58%)`);
    g.fillStyle = grd; g.fillRect(0, 0, 960, 540);
    g.fillStyle = "rgba(40,40,50,.25)"; g.fillRect(0, 380, 960, 160); // bench
    const people = i % 6 === 0 ? 0 : 1 + (i % 3 === 0 ? 1 : 0);
    const name = `${session}_frame_${String(i + 1).padStart(3, "0")}.jpg`;
    const truth: { clsIndex: number; box: [number, number, number, number] }[] = [];
    for (let k = 0; k < people; k++) {
      const x = 160 + ((i * 97 + k * 330) % 600), y = 140 + ((i * 13) % 60);
      g.fillStyle = `hsl(${(i * 53 + k * 90) % 360} 35% 45%)`;
      g.beginPath(); g.roundRect(x - 60, y + 70, 120, 200, 30); g.fill();
      g.fillStyle = "#e9c6a6"; g.beginPath(); g.arc(x, y + 30, 44, 0, Math.PI * 2); g.fill();
      const goggles = (i + k) % 2 === 0;
      if (goggles) { g.fillStyle = "rgba(120,200,255,.8)"; g.fillRect(x - 36, y + 18, 72, 18); }
      truth.push({ clsIndex: goggles ? 1 : 0, box: [x - 64, y - 18, x + 64, y + 274] });
    }
    g.fillStyle = "rgba(124,58,237,.9)"; g.fillRect(0, 0, 150, 30);
    g.fillStyle = "#fff"; g.font = "600 15px system-ui"; g.fillText(`SAMPLE ${String(i + 1).padStart(2, "0")}`, 12, 20);
    // toDataURL is synchronous (toBlob can be throttled to ~1 s per call in background tabs)
    const bin = atob(c.toDataURL("image/jpeg", 0.85).split(",")[1]);
    const bytes = Uint8Array.from(bin, (ch) => ch.charCodeAt(0));
    SAMPLE_TRUTH.set(name, truth);
    out.push(new File([bytes], name, { type: "image/jpeg" }));
  }
  return out;
}
