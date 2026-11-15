# ZALU · Global Design System v3

Global display preferences on top of the Organic Minimalism pass (PR #6): dark mode, motion intensity, universal
press feedback, UI scale, and one Settings panel that replaces the header's "Sample mode" badge. Screens were not
redesigned. Everything reaches them through tokens, `lib/motion` and scoped global styles.

## What changed

| Area | How |
|---|---|
| **Preferences store** | `src/lib/prefs.tsx`: `{ theme: light/dark/system, motion: 0–3, scale: compact/default/large }` in `localStorage["zalu.prefs"]`, held in a `useSyncExternalStore` store. It is mirrored onto `<html>` as `data-theme`, `data-theme-pref`, `data-motion` and `data-scale`, plus `color-scheme`. It follows the OS colour scheme and reduced-motion setting live, and syncs across tabs through the `storage` event. |
| **No flash** | `src/lib/prefs-script.ts` is inlined into `<head>` by `layout.tsx` and sets the attributes before the body paints. This is the next-themes technique, but without rendering a `<script>` from a client component, which React 19 warns about. `next-themes` was evaluated and not installed. |
| **Dark theme** | `globals.css` overrides `html[data-theme="dark"]` for the same `--color-*` tokens. Surfaces are charcoal-navy (`#0E131C` / `#151B26` / `#1D2533`), never pure black. `--color-brand` brightens to `#7090FF` for rings, indicators and links. A new `--color-brand-solid` (`#315CFD` light, `#3F66FF` dark) sits behind white text: 5.2:1 and 4.6:1. Hardcoded colours in owned CSS became tokens: `--color-brand-line`, `--color-warning-line`, `--color-rail`, `--color-brand-range`, `--color-sample-ink`, `--color-on-solid`, `--color-media`, `--color-overlay`, and `--shadow-sm/md/lg/brand`. Theme switches pause CSS transitions for one frame so colours do not smear. |
| **Motion intensity** | Reduced / Subtle / **Balanced** (default) / Expressive. `lib/motion` tokens (`DUR`, `t()`, `enter`, `rise`, `stagger`, `SPRING`, `EXIT`, `Dots`, `TextShimmer`, `AnimatedNumber`) are getters that read the level when an animation starts. Each level sets a time factor (0.6 / 0.8 / 1 / 1.2), travel (0 / 0.5 / 1 / 1.5×), a spring (Expressive: k=400, c=30, a few percent of follow-through, no bounce), press depth and loop speed. CSS transitions use `--dur-fast/base/slow` and `--ease`, scaled by `--k`. Reduced sets `MotionConfig reducedMotion="always"` and turns off CSS animations and transitions. **The OS `prefers-reduced-motion` setting always wins**: the effective level is 0 and Settings says so. |
| **Universal press** | One scoped rule in `globals.css` covers `button`, `[role=button/tab/radio/switch]`, `summary` and `a.btn`. Pressing scales to `var(--press)` (0.99 / 0.98 / 0.97) and the element springs back on release. It is transform-only, so there is no layout shift. It skips disabled controls, anything inside SVG (annotator boxes), draggables, the slider thumb and `[data-no-press]`. Motion's inline transforms take precedence, so `layoutId` pills are never disturbed. Reduced and OS reduced motion use an inset tint instead of movement. |
| **UI scale** | `html { font-size: 100% }`, compact 93.75%, large 112.5%. All 36 `font-size: px` declarations in `globals.css`, plus control heights and padding (buttons, inputs, chips, tabs, cards, KPIs, composer, icon buttons), are now in rem. Percentages keep browser zoom and the user's browser font setting working on top. Dashboard typography is moved to rem from `globals.css` without editing `dashboard.css`. |
| **Settings** | `src/components/Settings.tsx`, opened from the header. Contents: Theme (segmented radio group), Motion (Radix slider with four stops and a thumb that glides on the motion curve; Expressive uses a slight follow-through), Text size, and Data source (Sample or Live backend, with URL, token, health and Connect). It is a dialog with focus moved in on open; Escape and outside clicks close it and return focus to the trigger. Arrow keys work in every group. On phones it becomes a sheet. |
| **Header** | Shows only the brand and the Settings button. The purple "◆ Sample mode" badge is gone. The button keeps a quiet indicator: a hollow dot with "Sample data", a filled dot with "Live", or "Offline" on error, plus `aria-label="Settings. Data source: …"`. Sample mode still works as before, and the screens' own sample disclosures are unchanged. |
| **Next.js dev badge** | `devIndicators: false` in `next.config.mjs`, the supported config. Runtime errors still surface. |

## Validation (see `checks/v3-checks.txt`)

`npm run typecheck`, `npm run lint` and `npm run build` are all clean. 48 automated checks pass, run with Playwright and Chrome against the dev server:

- The dark preference is applied before the body exists (no flash). The scale is applied before paint.
- System follows OS changes live. Preferences persist across reload.
- Settings: reachable with Tab; focus moves in; arrow keys work on the radios; Home, End and arrows work on the slider; labels are clickable; Escape and outside clicks close it and focus returns to the trigger.
- Press depth measures 0.99, 0.98 and 0.97 at the three levels. Reduced gives scale 1 plus an inset tint. No layout shift at any level.
- The OS reduced-motion setting overrides Expressive.
- No horizontal overflow on any of the six screens at Compact, Default or Large, at 1280 px and 390 px (Large runs were in dark mode).
- Text contrast is ≥ 4.5:1 (3:1 for large text) on every screen in **both** themes. This was audited from computed colours.
- No console errors, apart from the existing `/favicon.ico` 404.

Screenshots: `before/` and `after/` in `light/` and `dark/`. The "before dark" shots show the old build ignoring a dark OS. `checks/` holds Settings in light and dark, the OS-reduced note, and scale samples. Videos are in `video/`: theme switching across screens, the slider stops, press at Expressive versus Reduced, and text size (desktop and mobile).

## Integration notes for Codex (files I did not edit)

1. **Fixed `whileTap` values bypass the motion level.** These are in `Describe.tsx:82` (`{ scale: 0.97 }`), `Composer.tsx:100,109` (`{ scale: 0.92 }`) and `Prepare.tsx:105,107` (`{ x: ±2 }`). Motion applies them instantly, even under Reduced. Replace them with the level-aware tokens from `@/lib/motion`:
   `whileTap={press}` for chips and buttons, and `whileTap={pressIcon}` for round icon buttons. Under Reduced both resolve to scale 1 and the CSS tint gives feedback. Alternatively, remove `whileTap` and let the universal CSS press handle it.
2. **`useReducedMotion()` in `Prepare.tsx:20`** sees only the OS setting. Use `useLessMotion()` from `@/lib/motion` so the in-app Reduced choice is honoured too.
3. **`ReportCard.tsx:97`** colours warnings with inline `#6b4000`, which is unreadable on dark. A scoped override in `globals.css` keeps it legible for now. Please switch to `className="small"` with `style={{ color: "var(--color-warning-ink)" }}`, or a class, and the override can go.
4. **`FailureGallery.tsx:59–60`** (`#1B2333` / `#AEB7C6`) and **`DashboardRenderer.tsx:73–74`** (`#8fb0ff` / `#0e1727`) draw on media backgrounds that are dark in both themes, so they are fine as they are. If you touch them, `var(--color-media-2)` and `var(--color-muted)` are the matching tokens.
5. **px sizes inside screens** (`MediaInput.tsx:26` `fontSize: 28`) do not follow UI scale. Prefer `rem` or a class.
6. **New tokens available:** `--color-brand-solid` (fill behind white text), `--color-on-solid`, `--color-media`, `--color-overlay`, `--shadow-sm/md/lg`, and `--dur-fast/base/slow` with `--ease` for CSS transitions. Use `var(--color-brand-solid)` rather than `--color-brand` for any new filled button with white text.
7. The header no longer says "Sample mode". `ErrorBox` now points to "Settings (top right)". `Deploy.tsx:86` still says "In Sample mode, …", which reads fine as a mode name.
