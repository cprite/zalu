# ZALU frontend (owner: Damir)

Next.js 16 + React 19 + TypeScript + Tailwind CSS v4. Three stages: **Describe** (task, footage, check labels) → **Train** (train, test on new footage) → **Deploy**.
Design tokens live in `src/app/globals.css` (`@theme`), shared with Figma via `design/` (see `design/README.md`).

```bash
cd frontend
npm ci
npm run dev          # http://localhost:3000, Sample mode by default (no backend needed)
```

Connecting the real backend and what each screen calls: see **[INTEGRATION.md](INTEGRATION.md)**.

## Sample mode vs live backend
- **Sample mode** (default): an in-browser mock of the API (`src/lib/api/mock.ts`). Every sample object is badged
  `◆ Sample`; sample metrics come from `src/mocks/sample-report.json`, produced by the **real** Evaluation Engine on simulated
  predictions (`python -m tools.make_sample_report`). They are never presented as experimental results.
- **Live**: top-right switch → backend URL + token (or `NEXT_PUBLIC_ZALU_API`, see `.env.example`).
  For the GX10: `ssh -L 9100:127.0.0.1:9100 gx10`, then use `http://localhost:9100`.

UI components never call `fetch`: they use `useApi()` (`src/lib/api/`), which returns either implementation of the same `ZaluApi` interface.
Routes that a teammate has not merged yet (HTTP 501) show a clear "not connected yet" message instead of failing.

## Layout
| Path | What |
|---|---|
| `src/lib/types.ts` | TypeScript mirror of `backend/schemas.py` + the evaluation report shape |
| `src/lib/api/` | `ZaluApi` interface, `live.ts` (FastAPI), `mock.ts` (sample data), provider |
| `src/lib/frames.ts` | client-side frame extraction from images/videos, brightness, webcam JPEG grab |
| `src/steps/*` | one component per step (Describe, Upload, Prepare, Train, Verify, Deploy) |
| `src/components/ReportCard.tsx` | Model Report Card (metrics + 95% ranges vs targets, errors, per class, limitations, verdict) |
| `src/components/FailureGallery.tsx` | missed / false-alarm test frames with plain-language reasons and fixes |
| `src/components/Comparison.tsx` | baseline vs fine-tuned table + synthetic A/B verdict |
| `src/components/Shell.tsx` | top bar, data-source switch, Stories-style 3-stage progress |
| `src/components/Composer.tsx` | message-box input: text + image/video attachments + send |
| `src/app/globals.css` | design tokens (`@theme` → Tailwind utilities) and semantic component classes |

## Personalized dashboard and export

After evaluation, open Deploy. The live adapter requests a DashboardSpec for the current prompt,
task and selected model. Refine it in place using the planner. In Sample mode, ZALU selects and
adapts a shared fixture to your prompt and model classes. The sample refinement input supports
renaming and showing or hiding the camera widget.

Start the camera or submit a single image in Live mode. The dashboard derives counts and charts
from the inference response and the current browser session. Precision and recall come from the
evaluation report when available.

Choose **Export dashboard** to download the current configuration as a standalone React project.
Unzip it, run `npm install` and `npm run dev`, then open `http://localhost:3000`. The project defaults
to sample data and includes an optional Live connection form. Export packages the shared renderer,
styles and configuration. It contains no uploaded frames, inference history or credentials.
