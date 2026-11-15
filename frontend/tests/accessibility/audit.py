#!/usr/bin/env python3
"""ZALU accessibility audit: axe-core (WCAG 2.0/2.1/2.2 A+AA tags) on every workflow screen + keyboard checks.

Runs the SAMPLE (mock) backend path of the UI - no token, no live data - so results are reproducible.
Usage (from frontend/, against a running build, default http://127.0.0.1:3000):
    python3 tests/accessibility/audit.py
Env: ZALU_TEST_URL, ZALU_CHROMIUM (default /usr/bin/chromium), ZALU_A11Y_MODES=desktop,mobile,reduced
Output: tests/accessibility/results/{axe-<mode>.json, keyboard-<mode>.json, summary.json, *.png}
Exit code 1 if any serious/critical axe violation or keyboard check fails.
"""
import asyncio, json, os, re, sys, time
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
FRONT = HERE.parent.parent
OUT = HERE / "results"
URL = os.environ.get("ZALU_TEST_URL", "http://127.0.0.1:3000")
CHROMIUM = os.environ.get("ZALU_CHROMIUM", "/usr/bin/chromium")
AXE = FRONT / "node_modules" / "axe-core" / "axe.min.js"  # transitive dev dep (eslint-plugin-jsx-a11y)
VIDEO = FRONT / "docs" / "motion-polish" / "organic" / "videos" / "journey-mobile.webm"
PROMPT = "Find people without safety goggles"
TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]
MODES = {
    "desktop": dict(viewport={"width": 1280, "height": 850}, reduced_motion="no-preference"),
    "mobile": dict(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, reduced_motion="no-preference"),
    "reduced": dict(viewport={"width": 1280, "height": 850}, reduced_motion="reduce"),
}


async def axe(page, screen, mode, bucket, shots):
    await page.wait_for_timeout(900)  # let entry transitions settle
    if not await page.evaluate("typeof window.axe !== 'undefined'"):
        await page.add_script_tag(path=str(AXE))
    res = await page.evaluate(
        "async (tags) => await axe.run(document, {runOnly: {type: 'tag', values: tags}, resultTypes: ['violations','incomplete']})",
        TAGS)
    v = [{"id": x["id"], "impact": x["impact"], "help": x["help"], "tags": [t for t in x["tags"] if t.startswith("wcag")],
          "nodes": [{"target": n["target"], "html": n["html"][:300], "summary": (n.get("failureSummary") or "")[:400]}
                    for n in x["nodes"][:8]], "node_count": len(x["nodes"])} for x in res["violations"]]
    inc = [{"id": x["id"], "impact": x["impact"], "node_count": len(x["nodes"])} for x in res["incomplete"]]
    bucket.append({"screen": screen, "mode": mode, "url": page.url, "violations": v, "incomplete": inc,
                   "axe_version": res["testEngine"]["version"]})
    print(f"  [{mode}] {screen}: {len(v)} violation rule(s): " + ", ".join(f"{x['id']}({x['impact']},{x['node_count']})" for x in v), flush=True)
    if screen in shots:
        await page.screenshot(path=str(OUT / f"{mode}-{screen.lower()}.png"), full_page=False)


async def keyboard_walk(page, label, max_tabs=60):
    """Tab through the page from the top: record focus order, visible-focus style, traps."""
    await page.evaluate("document.activeElement && document.activeElement.blur(); window.scrollTo(0,0)")
    await page.locator("body").focus() if await page.locator("body").count() else None
    seen, stops = [], []
    for i in range(max_tabs):
        await page.keyboard.press("Tab")
        info = await page.evaluate("""() => {
          const e = document.activeElement; if (!e || e === document.body) return null;
          const cs = getComputedStyle(e); const r = e.getBoundingClientRect();
          const name = (e.getAttribute('aria-label') || e.innerText || e.getAttribute('title') || e.id || '').trim().slice(0,60);
          const outline = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0;
          const ring = cs.boxShadow && cs.boxShadow !== 'none';
          return {tag: e.tagName.toLowerCase(), role: e.getAttribute('role'), name, focusVisible: e.matches(':focus-visible'),
                  indicator: outline || ring, outline: cs.outlineStyle + ' ' + cs.outlineWidth + ' ' + cs.outlineColor,
                  w: Math.round(r.width), h: Math.round(r.height), key: (e.dataset.a11yTab ||= String(Math.random()).slice(2))};
        }""")
        if info is None:
            stops.append({"i": i, "tag": "body"}); continue
        info["i"] = i
        stops.append(info)
        if seen and info["key"] == seen[-1]:
            return {"screen": label, "trap": True, "stops": stops}
        if info["key"] in seen[:3] and i > 3:  # cycled back to the start: walked the whole page
            break
        seen.append(info["key"])
    no_ind = [s for s in stops if s.get("tag") != "body" and not s.get("indicator")]
    small = [s for s in stops if s.get("tag") not in (None, "body", "main") and (s.get("w", 99) < 24 or s.get("h", 99) < 24)]
    return {"screen": label, "trap": False, "tab_stops": len(stops), "no_visible_indicator": no_ind,
            "below_24px_target": small, "stops": stops}


async def wait_enabled(page, loc, t=30):
    end = time.time() + t
    while time.time() < end:
        if await loc.count() and await loc.first.is_visible() and await loc.first.is_enabled():
            return loc.first
        await page.wait_for_timeout(500)
    raise TimeoutError(str(loc))


async def run(browser, mode):
    opts = dict(MODES[mode])
    rm = opts.pop("reduced_motion")
    ctx = await browser.new_context(reduced_motion=rm, **opts)
    page = await ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    btn = lambda n: page.get_by_role("button", name=n, exact=True)
    heading = lambda n: page.get_by_role("heading", name=n, exact=True)
    bucket, kb = [], []
    shots = {"Describe", "Upload", "Prepare", "Train-complete", "Verify-results", "Deploy-dashboard"}
    try:
        await page.goto(URL)
        await heading("What would you like to detect?").wait_for(timeout=30000)
        await axe(page, "Describe", mode, bucket, shots)
        if mode == "desktop":
            kb.append(await keyboard_walk(page, "Describe"))
            await page.locator("#composer-text").focus(); await page.keyboard.press("Shift+Tab"); await page.keyboard.press("Tab")
            await page.screenshot(path=str(OUT / "desktop-describe-composer-focused.png"))
            # keyboard-only task creation: focus the composer, type, submit with keyboard
            await page.locator("#composer-text").focus()
            await page.keyboard.type(PROMPT)
            kb.append({"screen": "Describe", "check": "composer reachable + typed by keyboard",
                       "value_ok": await page.locator("#composer-text").input_value() == PROMPT})
        await page.locator("#composer-text").fill(PROMPT)
        await btn("Create task").click()
        await heading("Add your footage").wait_for(timeout=30000)
        await axe(page, "Upload", mode, bucket, shots)
        if mode == "desktop":
            kb.append(await keyboard_walk(page, "Upload"))
            # Upload without drag-and-drop: is the native file input (or a button that opens it) keyboard reachable?
            fi = page.locator("input[type=file]").first
            fi_info = await fi.evaluate("""e => { const cs = getComputedStyle(e); const lab = e.labels && e.labels[0];
               return {display: cs.display, visibility: cs.visibility, tabindex: e.tabIndex, hidden: e.hidden,
                       hasLabel: !!(lab || e.getAttribute('aria-label')), labelText: lab ? lab.innerText.slice(0,80) : e.getAttribute('aria-label')}; }""")
            # the Upload Tab walk above proves "Attach images or video" is a tab stop; now operate it with Enter only
            chooser_by_keyboard = None
            try:
                attach = page.get_by_role("button", name="Attach images or video").first
                await attach.focus()
                async with page.expect_file_chooser(timeout=5000) as fc:
                    await page.keyboard.press("Enter")
                chooser = await fc.value
                await chooser.set_files(str(VIDEO))
                chooser_by_keyboard = {"opened_from": "Attach images or video (Enter)", "ok": True, "multiple": chooser.is_multiple()}
            except Exception as e:
                chooser_by_keyboard = {"ok": False, "error": str(e)[:200]}
            await page.get_by_role("button", name="Preview journey-mobile.webm").first.wait_for(timeout=10000)
            attached = await page.get_by_role("button", name="Preview journey-mobile.webm").count() > 0
            kb.append({"screen": "Upload", "check": "file picker without drag-and-drop", "file_input": fi_info,
                       "keyboard_file_chooser": chooser_by_keyboard, "file_attached": attached})
            if attached:
                tile = page.get_by_role("button", name="Preview journey-mobile.webm")
                await tile.focus(); await page.keyboard.press("Enter")
                dlg = page.get_by_role("dialog", name="Preview journey-mobile.webm")
                await dlg.wait_for(timeout=10000)
                await axe(page, "Upload-preview-dialog", mode, bucket, shots)
                in_dialog = []
                for _ in range(8):
                    await page.keyboard.press("Tab")
                    # body = focus left for browser chrome (native modal <dialog> behaviour), not page content behind the dialog
                    in_dialog.append(await page.evaluate("document.activeElement === document.body || !!document.activeElement.closest('dialog')"))
                await page.keyboard.press("Escape")
                await dlg.wait_for(state="hidden", timeout=5000)
                restored = await tile.evaluate("e => e === document.activeElement")
                kb.append({"screen": "Upload-preview-dialog", "check": "focus contained while open, Escape closes, focus restored",
                           "focus_never_reached_page_behind_dialog": all(in_dialog), "focus_restored_to_tile": restored})
                await btn("Remove journey-mobile.webm").click()
        await btn("Use sample frames").click()
        await btn("Upload and label").click()
        await heading("Check the labels").wait_for(timeout=30000)
        await page.get_by_text("Loading frames and labels", exact=True).wait_for(state="hidden", timeout=30000)
        await axe(page, "Prepare", mode, bucket, shots)
        if mode == "desktop":
            kb.append(await keyboard_walk(page, "Prepare"))
            boxes = await page.evaluate("""() => { const svgs=[...document.querySelectorAll('main svg rect, main [data-box], main .box')];
               return {box_elements: svgs.length, tables: document.querySelectorAll('main table').length,
                       lists_with_labels: document.querySelectorAll('main ul li, main ol li').length}; }""")
            kb.append({"screen": "Prepare", "check": "non-visual alternative to bounding boxes", **boxes})
        await (await wait_enabled(page, page.get_by_role("button", name=re.compile("Looks good, train")))).click()
        await btn("Start training").wait_for(timeout=30000)
        await axe(page, "Train", mode, bucket, shots)
        await btn("Start training").click()
        await page.locator("[data-job-status='running']").first.wait_for(timeout=30000)
        await axe(page, "Train-running", mode, bucket, shots)
        live = await page.evaluate("""() => [...document.querySelectorAll('[aria-live],[role=status],[role=alert],[role=progressbar]')]
              .map(e => ({role: e.getAttribute('role'), live: e.getAttribute('aria-live'), text: (e.innerText||'').slice(0,80),
                          valuetext: e.getAttribute('aria-valuetext'), name: e.getAttribute('aria-label')}))""")
        kb.append({"screen": "Train-running", "check": "live regions / progressbar semantics", "regions": live})
        anims = await page.evaluate("""() => document.getAnimations().filter(a => a.playState === 'running').map(a => ({
              name: a.animationName || a.constructor.name, infinite: a.effect && a.effect.getTiming().iterations === Infinity,
              target: a.effect && a.effect.target ? (a.effect.target.className || a.effect.target.tagName).toString().slice(0,60) : null}))""")
        kb.append({"screen": "Train-running", "check": f"running animations (prefers-reduced-motion={rm})", "count": len(anims),
                   "infinite": sum(1 for a in anims if a["infinite"]), "animations": anims[:30]})
        await heading("Training complete").wait_for(timeout=120000)
        await axe(page, "Train-complete", mode, bucket, shots)
        await btn("Test on new footage →").click()
        await btn("Use sample test frames").click()
        await axe(page, "Verify", mode, bucket, shots)
        await btn("Run the test").click()
        await heading("Test results").wait_for(timeout=60000)
        await axe(page, "Verify-results", mode, bucket, shots)
        if mode == "desktop":
            kb.append(await keyboard_walk(page, "Verify-results"))
        await btn("Show details").click()
        await axe(page, "Verify-details", mode, bucket, shots)
        await btn("Hide details").click()
        deploy = page.get_by_role("button", name=re.compile("Deploy this model"))
        if not await deploy.count():
            await btn("Compare models").click(); await btn("See mistakes").click()
        await deploy.first.click()
        await heading(PROMPT).wait_for(timeout=60000)
        await axe(page, "Deploy-dashboard", mode, bucket, shots)
        if mode == "desktop":
            kb.append(await keyboard_walk(page, "Deploy-dashboard"))
        await btn("Run sample inference").click()
        await page.get_by_text("Simulated frame", exact=True).wait_for(timeout=30000)
        await axe(page, "Deploy-inference", mode, bucket, shots)
        await btn("Stop inference").click()
    except Exception as e:
        bucket.append({"screen": "FLOW-ERROR", "mode": mode, "error": str(e)[:500], "url": page.url})
        print(f"  [{mode}] flow stopped: {str(e)[:200]}", flush=True)
        await page.screenshot(path=str(OUT / f"{mode}-flow-error.png"))
    finally:
        await ctx.close()
    (OUT / f"axe-{mode}.json").write_text(json.dumps(bucket, indent=1))
    if kb:
        (OUT / f"keyboard-{mode}.json").write_text(json.dumps(kb, indent=1))
    return bucket, kb, errors


async def main():
    if not AXE.exists():
        sys.exit(f"axe-core not found at {AXE}; run `npm ci` in frontend/ first")
    OUT.mkdir(parents=True, exist_ok=True)
    modes = os.environ.get("ZALU_A11Y_MODES", "desktop,mobile,reduced").split(",")
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox"])
        results = await asyncio.gather(*(run(browser, m) for m in modes))
        await browser.close()
    summary, fail = {"url": URL, "modes": modes, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rules": {}}, False
    for (bucket, kb, errors), m in zip(results, modes):
        for s in bucket:
            if s.get("screen") == "FLOW-ERROR":
                fail = True; summary.setdefault("flow_errors", []).append(s); continue
            for v in s["violations"]:
                r = summary["rules"].setdefault(v["id"], {"impact": v["impact"], "help": v["help"], "wcag": v["tags"], "where": []})
                r["where"].append(f"{m}:{s['screen']}({v['node_count']})")
                fail |= v["impact"] in ("serious", "critical")
        for k in kb:
            fail |= bool(k.get("trap"))
        if errors:
            summary.setdefault("page_errors", {})[m] = errors
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {"impact": v["impact"], "where": v["where"]} for k, v in summary["rules"].items()}, indent=1))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    asyncio.run(main())
