// Run against the production server with Playwright available on NODE_PATH.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const loadFixtureDependency = createRequire(import.meta.url);
const { chromium } = loadFixtureDependency("playwright");
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const url = process.env.ZALU_TEST_URL ?? "http://127.0.0.1:3000";
const videoPath = path.join(__dirname, "../docs/motion-polish/organic/videos/journey-mobile.webm");
const output = process.env.ZALU_CAPTURE_DIR;
const prompt = "Find people without safety goggles";
const invalidVideo = { name: "unsupported.webm", mimeType: "video/webm", buffer: Buffer.from("not a video") };
const png = { name: "replacement.png", mimeType: "image/png",
  buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/X1sAAAAASUVORK5CYII=", "base64") };

async function run(browser, mode) {
  const mobile = mode === "mobile";
  const reduced = mode === "reduced";
  const viewport = mobile ? { width: 390, height: 844 } : { width: 1280, height: 850 };
  const capture = output && !reduced;
  const context = await browser.newContext({
    viewport, isMobile: mobile, hasTouch: mobile,
    reducedMotion: reduced ? "reduce" : "no-preference",
    ...(capture ? { recordVideo: { dir: output, size: viewport } } : {}),
  });
  const page = await context.newPage();
  const video = page.video();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const button = (name) => page.getByRole("button", { name, exact: true });
  const heading = (name) => page.getByRole("heading", { name, exact: true });
  const upload = () => page.locator("input[type=file]").first();
  const checkWidth = async () => assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  async function shot(name) {
    if (capture && (!mobile || name.startsWith("06"))) await page.screenshot({ path: path.join(output, name + ".png"), animations: "allow" });
  }

  try {
    await page.goto(url);
    await heading("What would you like to detect?").waitFor();
    assert.equal(await page.getByRole("navigation", { name: "Progress" }).getByRole("button").count(), 3);
    if (!mobile) await shot("01-describe");
    await page.locator("#composer-text").fill(prompt);
    await button("Create task").click();
    await heading("Add your footage").waitFor();
    await upload().setInputFiles(videoPath);
    const tile = button("Preview journey-mobile.webm");
    await tile.locator("img[src^='data:image/jpeg']").waitFor();
    await checkWidth();
    if (!mobile) await shot("02-upload-video");
    if (mobile) await tile.tap();
    else { await tile.focus(); await page.keyboard.press("Enter"); }
    const dialog = page.getByRole("dialog", { name: "Preview journey-mobile.webm" });
    await dialog.waitFor();
    await dialog.locator("video").evaluate((element) => element.play());
    await page.waitForFunction(() => document.querySelector("dialog video")?.currentTime > 0);
    if (mobile) await shot("06-mobile-preview");
    if (mobile) await button("Close").tap();
    else await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "hidden" });
    assert.equal(await tile.evaluate((element) => element === document.activeElement), true);
    await button("Remove journey-mobile.webm").click();
    await upload().setInputFiles(invalidVideo);
    await button("Preview unsupported.webm").click();
    await page.getByText("Preview unavailable. The file is still attached.").waitFor();
    await button("Close").click();
    await button("Remove unsupported.webm").click();
    await upload().setInputFiles(png);
    await button("Preview replacement.png").locator("img[src^='blob:']").waitFor();
    await button("Remove replacement.png").click();
    const localPreview = await page.evaluate(() => document.querySelector("dialog[open]") === null);
    assert(localPreview);

    await button("Use sample frames").click();
    await button("Upload and label").click();
    await heading("Check the labels").waitFor();
    await checkWidth();
    await page.getByText("Loading frames and labels", { exact: true }).waitFor({ state: "hidden" });
    await button("← Change footage").click();
    await heading("Add your footage").waitFor();
    await page.getByText("36 files attached").waitFor();
    await button("Upload and label").click();
    await page.getByRole("button", { name: /Looks good, train/ }).click();
    await button("Options").click();
    await page.getByRole("radio", { name: /Standard/ }).waitFor();
    await button("Options").click();
    await button("Start training").click();
    const job = page.locator("[data-job-status='running']").first();
    await job.waitFor();
    await job.getByText(/Epoch \d+ of 30/).waitFor();
    const bar = job.getByRole("progressbar");
    const observed = await bar.getAttribute("aria-valuenow");
    assert(await job.getByText(observed + "%", { exact: true }).isVisible());
    const shimmer = job.locator("[data-training-flow]");
    if (reduced) assert.equal(await shimmer.count(), 0);
    else await shimmer.waitFor();
    await checkWidth();
    await shot("03-training-progress");
    await heading("Training complete").waitFor();
    assert.equal(await page.locator("[data-training-flow]").count(), 0);
    assert.equal(await page.getByRole("progressbar").first().getAttribute("aria-valuenow"), "100");
    const circle = page.locator("[data-training-complete]");
    assert.equal(await circle.evaluate((element) => getComputedStyle(element).borderRadius), "50%");
    if (reduced) assert.equal(await circle.locator("svg").count(), 0);
    await shot("04-training-complete");
    await button("Test on new footage →").click();
    await button("Use sample test frames").click();
    await button("Run the test").click();
    await heading("Test results").waitFor();
    await page.getByText("Demo results", { exact: true }).first().waitFor();
    assert.equal(await page.locator("main .badge.sample").count(), 0);
    await checkWidth();
    await shot("05-test-results");
    await button("Show details").click();
    await page.getByRole("region", { name: "Per-class results" }).waitFor();
    const download = page.waitForEvent("download");
    await button("Download JSON").click();
    const report = JSON.parse(await fs.readFile(await (await download).path(), "utf8"));
    const model = Object.values(report.per_model).filter((entry) => entry.model.kind === "trained")
      .sort((a, b) => b.overall.f1 - a.overall.f1)[0];
    for (const [name, value] of [
      ["Recall", (model.overall.recall * 100).toFixed(0) + "%"],
      ["Precision", (model.overall.precision * 100).toFixed(0) + "%"],
      ["Missed objects", String(model.overall.fn)],
      ["Inference speed", model.performance.fps + " FPS"],
    ]) {
      const kpi = page.locator(".kpi").filter({ has: page.getByText(name, { exact: true }) }).first();
      assert.equal(await kpi.locator(".v").innerText(), value);
    }
    assert(await page.getByText(model.overall.fp + " false alarms", { exact: true }).isVisible());
    await button("Hide details").click();
    await button("Compare models").click();
    await heading("Model comparison").waitFor();
    await button("See mistakes").click();
    await page.getByText("Demo frames", { exact: true }).waitFor();
    await checkWidth();
    await page.getByRole("button", { name: /Deploy this model/ }).click();
    await heading(prompt).waitFor();
    await button("Run sample inference").click();
    await page.getByText("Simulated frame", { exact: true }).waitFor();
    await button("Stop inference").click();
    await checkWidth();
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
    if (capture) {
      const saved = path.join(output, mode === "desktop" ? "desktop-journey.webm" : "mobile-journey.webm");
      await video.saveAs(saved);
      await video.delete();
    }
  }
}

(async () => {
  if (output) await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({ channel: process.env.ZALU_BROWSER ?? "msedge" });
  try {
    const modes = process.env.ZALU_TEST_MODES?.split(",") ?? ["desktop", "mobile", "reduced"];
    await Promise.all(modes.map(async (mode) => {
      await run(browser, mode);
      console.log(`Passed browser scenario: ${mode}`);
    }));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
