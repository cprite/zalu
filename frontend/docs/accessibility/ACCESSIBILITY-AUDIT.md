# ZALU accessibility audit (Phase A)

Date: 2026-11-15. Branch `damir/accessibility-v1`, based on `ec8b0db` (PR #7, screen-ux-v3 incl. organic-motion).
Target build: the same commit served at `http://127.0.0.1:3000/`, SAMPLE (mock) data path, no live backend, no token.

**What this is and is not.** These are findings from automated axe-core 4.14.0 scans plus scripted keyboard checks
that were actually run (evidence below). Automated tools catch only part of WCAG. **WCAG 2.2 AA compliance is NOT
claimed.** **NVDA, VoiceOver (macOS/iOS), Narrator and TalkBack were NOT tested**: none is available on this host
(Raspberry Pi, headless Chromium). Manual zoom/reflow at 200%/400%, the dark theme and the large-text setting were also
not tested in this phase (the theme/scale system is still in progress on `damir/global-design-v3`).

## How to run

```
cd frontend && npm ci && python3 tests/accessibility/audit.py   # needs a running build on :3000 (ZALU_TEST_URL to override)
```

Python Playwright + system Chromium (`ZALU_CHROMIUM`, default `/usr/bin/chromium`); axe-core is injected from
`node_modules/axe-core/axe.min.js` (already a transitive dev dependency, so no `package.json` change). Modes:
`desktop` (1280x850), `mobile` (390x844, touch), `reduced` (1280x850, `prefers-reduced-motion: reduce`); choose with
`ZALU_A11Y_MODES`. Rule tags: wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa, best-practice. Exits 1 on any
serious/critical violation or a detected keyboard trap, so it can gate CI once the findings below are fixed.

Screens scanned in each mode (12 states): Describe, Upload, Upload preview dialog (desktop), Prepare, Train, Train
(running), Train (complete), Verify, Verify results, Verify details, Deploy dashboard, Deploy (sample inference).
Video Search has no frontend yet, so it was not scanned.

## Evidence

All under `frontend/tests/accessibility/results/` (run 2026-11-15 ~10:49 UTC):

- `axe-desktop.json`, `axe-mobile.json`, `axe-reduced.json`: per-screen violations and "incomplete" (needs manual review) items
- `keyboard-desktop.json`: Tab order per screen, focus indicator style, file chooser by keyboard, dialog focus,
  live-region inventory; `keyboard-reduced.json` / `keyboard-mobile.json`: running-animation counts
- `summary.json`: rules violated and where; `run.log`: console output
- Screenshots: `{desktop,mobile,reduced}-{describe,upload,prepare,train-complete,verify-results,deploy-dashboard}.png`,
  `desktop-describe-composer-focused.png`

## Result summary

| Area | Result |
|---|---|
| axe violations | 2 rules: `label-content-name-mismatch` (serious, Prepare, all 3 modes), `heading-order` (moderate, Verify, all 3 modes). All other 10 states: 0 violations. |
| axe incomplete (manual review needed) | `color-contrast` on Prepare (39 nodes), Verify results/details, Deploy inference (text over images/gradients axe cannot resolve); `video-caption` on the preview dialog; `aria-prohibited-attr` on Describe; `label-content-name-mismatch` on Verify (7) |
| Keyboard traps | None found (Describe, Upload, Prepare, Verify results, Deploy dashboard walked with Tab until focus cycled) |
| Skip link | Present, first Tab stop ("Skip to content") |
| Upload without drag-and-drop | PASS: "Attach images or video" is a Tab stop; Enter opens the native file chooser (multiple files), file attached |
| Preview dialog | PASS: native modal `<dialog>`; Tab never reached page content behind it; Escape closes; focus returned to the tile |
| Reduced motion | PASS (Train running): 0 running animations with `reduce` vs 17 (15 infinite) without |
| Mobile | The full flow completed at 390x844 with the same 2 axe rules; horizontal overflow was not measured by this script |

## Findings (prioritised)

Owner: **Codex** = screen components (`damir/screen-ux-v3`), **Claude** = global design system / shell / settings
(`damir/global-design-v3`), **A11y** = this branch. Status for all: **open** (no fixes applied to files owned by
others, per the ownership rules).

### F1. Adding a box on Prepare needs a drag (serious)
- WCAG 2.2: 2.5.7 Dragging Movements (AA); 2.1.1 Keyboard (A).
- Screen: Prepare. User impact: keyboard, switch and many motor-impaired users cannot add a missing box; the only
  instruction is "Drag on the image to add a box". Inspecting, re-classing, confirming and deleting existing boxes IS
  possible by keyboard through the "Boxes on <frame>" list (good).
- Repro: Prepare, Tab through the page: no control creates a box; `src/components/Annotator.tsx` only creates boxes in
  the `pointerdown/pointermove/pointerup` handlers.
- Fix (`frontend/src/components/Annotator.tsx`): add an "Add box" button that inserts a default box (e.g. centred,
  25% of the frame) for the active class, selects it, and exposes four numeric inputs (left, top, right, bottom, in px or
  %) plus arrow-key nudging on the selected box. Change the hint to "Drag on the image, or use Add box".
- Owner: Codex. Evidence: `keyboard-desktop.json` (Prepare walk), `desktop-prepare.png`. Note: this is static code
  review + Tab walk; fixed annotation must be re-tested manually before claiming it is accessible.

### F2. Frame thumbnails: accessible name hides the visible count (serious, axe)
- WCAG 2.2: 2.5.3 Label in Name (A). axe `label-content-name-mismatch`, 9 nodes in every mode.
- Screen: Prepare. Impact: speech-input users who say what they see ("click 1", "click empty") do not hit the button.
- Repro: `button.thumb` has `aria-label="Frame 7: nothing labelled"` but visible text "empty" (or "1", "2").
- Fix (`frontend/src/steps/Prepare.tsx`, thumbnail button ~line 123): either drop `aria-label` and put a visually
  hidden "Frame 7:" before the visible count, or make the visible text part of the name, e.g.
  `aria-label={\`Frame ${i + 1}: ${ls.length ? \`${ls.length} box(es)\` : "empty"}\`}`.
- Owner: Codex. Evidence: `axe-*.json` screen "Prepare".

### F3. Training progress announces every epoch, for every model in parallel (moderate)
- WCAG 2.2: 4.1.3 Status Messages (AA) is met, but the volume is a usability failure for screen-reader users (and the
  brief's "no per-percentage spam" rule).
- Screen: Train (running). Each model card has a `role="status"` region with "Epoch N of 30"
  (`src/components/ScreenActivity.tsx:10`, used at `src/steps/Train.tsx:123`). Two models train at once, so a screen
  reader can be asked to read ~60 updates. The `role="progressbar"` elements (`src/components/ui.tsx:72`,
  `ScreenActivity.tsx:25`) have no `aria-valuetext`.
- Fix: keep the visible text, but move the live region out of the per-epoch text: announce only milestones
  (started, 25/50/75%, completed, failed) via `useAnnounceOnChange` + `progressMilestone` from
  `src/accessibility/LiveAnnouncer.tsx`; add `aria-valuetext={\`Epoch ${e} of ${n}\`}` to the progressbars.
- Owner: Codex (Train), Claude (mount `LiveAnnouncerProvider` once in the shell, `src/app/layout.tsx` or the providers file).
- Evidence: `keyboard-desktop.json` "Train-running ... live regions".

### F4. Verify results skips a heading level (moderate, axe)
- WCAG 2.2: 1.3.1 Info and Relationships (A) (axe `heading-order`, best practice).
- Screen: Verify results / details. `h1 "Test results"` is followed by `h3 "Trained · real footage only"` with no h2, so
  heading navigation (H key) loses the structure.
- Fix: `frontend/src/components/ReportCard.tsx:23` (and `Comparison.tsx:21`, `FailureGallery.tsx:17`) use `h2`, or
  accept a `headingLevel` prop. Owner: Codex. Evidence: `axe-*.json` "Verify-results".

### F5. Text contrast over images and gradients is unverified (needs manual check; potentially serious)
- WCAG 2.2: 1.4.3 Contrast (Minimum) (AA), 1.4.11 Non-text Contrast (AA).
- Screens: Prepare (39 nodes: box labels over video frames, "empty"/count badges on thumbnails), Verify results (1),
  Verify details (7), Deploy inference (4). axe returned "incomplete", not pass: it cannot compute contrast over
  images. Box labels drawn on frames depend on the footage behind them.
- Fix: give on-image text a solid backing (e.g. label chip with an opaque background) in
  `src/components/Annotator.tsx` (`.box-label`) and the thumbnail `.count` in `src/app/globals.css`; then measure with
  a colour picker in both themes. Owner: Codex (components) + Claude (tokens/themes).

### F6. Composer focus ring is low-contrast (minor, needs measurement)
- WCAG 2.2: 2.4.7 Focus Visible (AA) is met (`.composer:focus-within` changes border + halo,
  `desktop-describe-composer-focused.png`), but 2.4.13 Focus Appearance (AAA) is not targeted and the halo
  (`--color-brand-soft`) is pale. The textarea itself sets `outline: none`.
- Fix: in `src/app/globals.css` `.composer:focus-within`, add a 2px solid `--color-brand` outline/box-shadow ring (the
  global `:focus-visible` style already uses 3px brand). Owner: Claude.

### F7. `aria-label` on a plain div (minor, axe incomplete)
- WCAG 2.2: 4.1.2 Name, Role, Value (A). `src/steps/Describe.tsx:79` `<div className="suggestions" aria-label="Examples">`
  has no role, so the label is ignored by most screen readers. Fix: `role="group"` (or a `<ul>`/`<nav>` with the label).
  Owner: Codex.

### F8. No caption track on video previews (informational)
- WCAG 2.2: 1.2.1 / 1.2.2. axe `video-caption` (incomplete) on the Upload preview dialog
  (`src/components/AttachmentPreview.tsx:104`). User footage normally has no speech, and fake captions must not be
  added. Fix: if a `.vtt` is attached alongside a video, render `<track kind="captions">`; otherwise show a visible
  "No captions available" note in the dialog. ZALU's own demo video (with speech) needs real captions. Owner: Codex.

### F9. Many Tab stops on Prepare (minor)
- WCAG 2.2: 2.4.1 Bypass Blocks / efficiency. 36 thumbnails are 36 Tab stops (48 total on the screen) before the
  "Looks good, train" action. Arrow keys already browse frames. Fix: roving tabindex on the thumbnail strip
  (`tabIndex={i === cur ? 0 : -1}`, Left/Right/Home/End move). Owner: Codex.

## Checklist (in order)

1. [ ] F1 keyboard way to add/adjust a box (Annotator) - Codex
2. [ ] F2 thumbnail label-in-name - Codex (one-line fix)
3. [ ] F3 milestone-only training announcements + `aria-valuetext` - Codex/Claude
4. [ ] F4 heading levels on Verify - Codex (one-line fix)
5. [ ] F5 measure + fix on-image text contrast in light and dark themes - Codex/Claude
6. [ ] F7 `role="group"` on Describe examples - Codex (one-line fix)
7. [ ] F6 stronger composer focus ring - Claude
8. [ ] F8 caption availability state - Codex
9. [ ] F9 roving tabindex on thumbnails - Codex
10. [ ] Manual: NVDA + Firefox/Chrome, VoiceOver Safari (macOS + iOS), TalkBack Chrome; 200% text, 400% zoom reflow; dark theme; large-text setting - A11y agent / a human tester
11. [ ] Re-run `tests/accessibility/audit.py` after each merge; add regression asserts for each fixed item

## Accessibility Settings integration contract (for the design-system owner, Claude)

Types and the OS-preference resolver are in `frontend/src/accessibility/preferences.ts`
(`AccessibilityPreferences`, `DEFAULT_A11Y_PREFERENCES`, `resolveAccessibility()`, `accessibilityDataAttributes()`).
They render nothing; the Settings UI stays yours. Rules:

| Setting | Values | OS preference | What it must actually do |
|---|---|---|---|
| High contrast | system / on / off | `prefers-contrast: more`, `forced-colors: active` | Swap to a token set with >= 7:1 body text, >= 3:1 borders/icons, solid backgrounds behind on-image text; never only a filter |
| Reduced motion | system / on / off | `prefers-reduced-motion: reduce` | Same path the app already uses for `reduce` (verified: 0 running animations on Train). "off" must not override an OS `reduce` silently; show the OS state |
| Strong focus | on / off | none | 3px solid ring + 2px offset on every focusable, >= 3:1 against both neighbours, including `.composer`, chips, thumbnails |
| Larger text | (not here) | browser zoom / font size | Use the existing global UI-scale setting; do not add a second one. Must reflow without horizontal scroll at 200% |
| Captions when available | on / off | none | Default `<track>` to `showing` when a caption track exists; never implies captions exist |
| Keyboard help | link | n/a | Opens a dialog listing real shortcuts only (today: Left/Right on Prepare, Escape closes dialogs) |

Apply via `document.documentElement` data attributes (`data-a11y-contrast`, `data-a11y-motion`, `data-a11y-focus`).
No "screen reader mode" toggle: the default UI must work with screen readers. Each setting needs a test that asserts
its CSS effect (computed style), not just that the toggle exists.

Live announcements: mount `LiveAnnouncerProvider` (`src/accessibility/LiveAnnouncer.tsx`) once in the shell. It keeps
two always-present visually hidden regions (polite `role=status`, assertive `role=alert`), debounces bursts (700 ms,
last message wins), and drops repeats within 4 s. Callers announce only backend-confirmed states ("Video uploaded",
"Training completed"), never optimistic ones.

## Video Search accessible UI contract (pending integration)

Video Search has no frontend yet, so nothing here has been demonstrated in the UI. **Demo status: pending
integration.** The text layer exists and is type-checked: `frontend/src/accessibility/videoSearchText.ts`, using the
backend fields from `backend/video_search/schemas.py` (`camera_id`, `start_ms`, `end_ms`, `visible_duration_ms`,
`query_score`, `score_type`, `cross_camera_identity_status`, `total_matches`, `status`).

Checked against the real demo response (`deliverables/demo/vs/view/results.json`, 5 results), `describeSearch()` returns:

> Search complete. Five matching appearances. Result one: Camera 1, timestamp 00:48, visible for approximately 6
> seconds. Result two: Camera 1, timestamp 01:24, visible for approximately 12 seconds. Result three: Camera 1,
> timestamp 00:48, visible for approximately 6 seconds. Same person across cameras: not assessed.

`seekLabel(result, 1)` -> "Play result 2: Camera 1 from 1 minute 24 seconds into the recording". Non-completed status
returns "Searching recordings." / "Search failed. No results to show." (never claims completion early).

Required UI behaviour for whoever builds the screen:

1. Search box is a labelled `<input type="search">` inside a `<form role="search">`; Enter submits.
2. While running: visible "Searching recordings..." text; announce once (polite). On completion announce
   `describeSearch(response)` once; on failure announce the error (assertive) and show it visibly.
3. Results are an ordered list (`<ol aria-label="Search results">`), one `<li>` per result with the visible
   `describeResult()` text, the thumbnail `alt` = "Camera N at mm:ss", and one "Play" button named by `seekLabel()`.
4. The similarity score is shown as "similarity 0.32", never as a percentage or "confidence"; it is a CLIP cosine
   similarity, not a probability. Hidden by default in announcements (`includeScore: false`).
5. Identity: show `identityText(status)` on every result. Visual similarity is never described as "the same person".
6. Playback: native `<video controls>` (keyboard play/pause/seek) is enough for the demo; activating a result seeks to
   `start_ms`, moves focus to the player, and announces "Playing Camera N from mm:ss". Escape or a "Back to results"
   button returns focus to the same result.
7. No captions exist for CCTV; the player shows "No audio, no captions" instead of a captions button.
8. Re-run `tests/accessibility/audit.py` with a Video Search state added before the demo.

## Not done in Phase A

- Screen readers (NVDA, VoiceOver, Narrator, TalkBack): not tested; none available on this host.
- Dark theme, large-text setting, 200% text / 400% zoom reflow: not tested (theme/scale work not merged).
- Settings popover/dialog: not scanned (not reachable on this commit beyond the "Data source" button).
- Charts in the dashboard: not individually checked for text alternatives beyond axe.
- No fixes applied to screen/shell files (owned by other agents); `LiveAnnouncerProvider` is not mounted yet.
