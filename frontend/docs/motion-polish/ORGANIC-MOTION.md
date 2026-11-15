# ZALU · Organic Minimalism — motion spec

Builds on PR #5 (`src/lib/motion.tsx`). One system, applied everywhere; no screen gets its own effect vocabulary.

## Audit of PR #5 (what this pass fixes)
| Area | Finding | Fix |
|---|---|---|
| Loading | Spinner in buttons, shimmer text elsewhere: two vocabularies for "working" | One `Dots` indicator + `StatusLine` (working → done → error) |
| Progress | Bars and Stories segments animate `width` (layout, not compositor) | `scaleX` from the left, spring-smoothed toward the **real** value |
| Train | Job message text only; epoch not visible as a number | Epoch parsed from the backend message (`epoch 7/30`), shown only when present |
| Completion | Check mark only | Check draws, one soft ring expands once, real train time shown |
| Selection | Pills only on training presets | Shared indicators (`layoutId`) on every segmented control; spring, no overshoot |
| Disclosure | Text-only toggles | `Disclosure` button: chevron rotates, panel opens from the trigger |
| Press | No tactile response on regular buttons | `scale(.98)` on press (CSS, 150 ms), hover raises contrast, not position |
| Verify | Whole card appears at once | Progressive reveal: verdict, then each KPI (60 ms apart), numbers count to measured values |
| Deploy | Empty states are bare text; trend end marker jumps | Empty states with a quiet icon; end marker glides to the next real reading |
| Touch | 40 px targets | 44 px on coarse pointers, `touch-action: manipulation` |

## Tokens
- Easing: `[0.2, 0, 0, 1]` (enter), exits use `fast` — **exit is always faster than enter**.
- Durations: `fast 150` (state swaps, press, exits) · `base 220` (reveals, cards) · `slow 320` (progress, completion, first appearance).
- Spring for shared indicators and progress smoothing: `stiffness 520, damping 44, mass 0.9` (critically damped: settles without overshoot).
- Loops: only while something is genuinely working (dots, live dot). They stop under `prefers-reduced-motion` and when the work ends.

## Principles
1. Motion explains a change caused by the person or by the system; nothing moves on its own.
2. Progress only ever reflects backend numbers. No fake percentages, no indeterminate bars pretending to be determinate.
3. Animate `transform` and `opacity`. The only height animation is the disclosure panel, because opening is the meaning.
4. Every animation is interruptible (Motion), keyboard-reachable, and has a reduced-motion final state.
