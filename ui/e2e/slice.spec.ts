/**
 * The vertical slice, end to end, against the MOCK backend (vite --mode mock):
 * connect → pick a session → chat → loom with live progress → wear → unwear.
 * Screenshots go to $UI_SHOTS_DIR (default test-results/shots).
 */
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { expect, type Page, test } from "@playwright/test";

const SHOTS = process.env.UI_SHOTS_DIR || resolve("test-results", "shots");
mkdirSync(SHOTS, { recursive: true });

async function shot(page: Page, name: string): Promise<void> {
  const project = test.info().project.name;
  await page.screenshot({ path: resolve(SHOTS, `${project}-${name}.png`), fullPage: true });
}

async function noHorizontalScroll(page: Page): Promise<void> {
  const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(sw, "page scrolls horizontally").toBeLessThanOrEqual(iw + 1);
}

test.beforeEach(async ({ page, request }) => {
  // the mock backend is shared by both projects: start each test from the recording
  expect((await request.post("/__mock/reset")).status()).toBe(204);
  // every test starts from a clean browser: no saved session or settings
  await page.addInitScript(() => {
    try {
      localStorage.clear();
    } catch {
      // ignore
    }
  });
});

test("connect and pick a recorded session", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("conn-status")).toHaveText("same origin");
  const table = page.getByTestId("session-table");
  await expect(table).toContainText("uifix-phaseb");
  await expect(table).toContainText("on disk only");
  await noHorizontalScroll(page);
  await shot(page, "01-picker");

  await table
    .getByRole("row", { name: /uifix-phaseb/ })
    .getByRole("button", { name: "open" })
    .click();
  await expect(page.getByTestId("session-name")).toHaveText("uifix-phaseb");
  // the recorded session: two turns on each branch, code #3 worn
  await expect(page.getByTestId("turn-count")).toHaveText("turns loom 2 · base 2");
  await expect(page.getByTestId("wearbar")).toContainText("WEARING member code #3");
  await expect(page.getByTestId("future-card-3")).toContainText("worn");
  await expect(page.getByTestId("codespace")).toHaveAttribute("data-dims", "3");
  await expect(page.getByTestId("codespace-canvas")).toBeVisible();
  await noHorizontalScroll(page);
  await shot(page, "02-recorded-worn-session");

  await page.getByRole("button", { name: "split" }).click();
  await expect(page.getByTestId("transcript-base")).toBeVisible();
  await expect(page.getByTestId("transcript-loom")).toBeVisible();
});

test("restore an on-disk session", async ({ page }) => {
  await page.goto("/");
  const row = page.getByTestId("session-table").getByRole("row", { name: /archived-8b-demo/ });
  await row.getByRole("button", { name: "restore" }).click();
  await expect(page.getByTestId("session-name")).toHaveText("archived-8b-demo");
  await expect(page.getByTestId("turn-count")).toHaveText("turns loom 2 · base 2");
});

test("the slice: new session → chat → loom with progress → wear → unwear", async ({ page }) => {
  const name = `e2e-${test.info().project.name}-${Date.now()}`;
  await page.goto("/");
  await page.getByTestId("new-session-input").fill(name);
  await page.getByTestId("new-session-input").press("Enter");
  await expect(page.getByTestId("session-name")).toHaveText(name);
  await expect(page.getByTestId("transcript-loom")).toContainText("a new session");

  // SEND commits a turn
  await page.getByTestId("composer").fill("Tell me about the weather on a small island.");
  await page.getByTestId("send-btn").click();
  await expect(page.getByTestId("transcript-loom")).toContainText("no model ran");
  await expect(page.getByTestId("turn-count")).toHaveText("turns loom 1 · base 0");
  await expect(page.getByTestId("composer")).toHaveValue("");

  // LOOM contemplates one: k futures, live progress, nothing committed
  await page.getByTestId("composer").fill("And what would you do there first?");
  await page.getByTestId("k-input").fill("6");
  await page.getByTestId("loom-btn").click();
  const progress = page.getByTestId("loom-progress");
  await expect(progress).toBeVisible();
  await expect(page.getByTestId("loom-stage")).toContainText("harvesting codes");
  await shot(page, "03-loom-progress");
  await expect(progress).toBeHidden({ timeout: 20_000 });
  await expect(page.getByTestId("cards").locator("article")).toHaveCount(6);
  await expect(page.getByTestId("deck-stat")).toContainText("k=6");
  await expect(page.getByTestId("turn-count")).toHaveText("turns loom 1 · base 0");
  await expect(page.getByTestId("composer")).toHaveValue("And what would you do there first?");
  await expect(page.getByTestId("codespace")).toHaveAttribute("data-dims", "3");

  // WEAR one at a chosen alpha
  await page.getByTestId("select-2").click();
  await expect(page.getByTestId("future-card-2")).toHaveClass(/selected/);
  await page.getByTestId("alpha-slider").fill("0.4");
  await expect(page.getByTestId("alpha-readout")).toHaveText("0.400");
  await expect(page.getByTestId("zone")).toHaveText("threshold");
  await page.getByTestId("wear-btn").click();
  await expect(page.getByTestId("wearbar")).toContainText("WEARING member code #2");
  await expect(page.getByTestId("wearbar")).toContainText("0.400");
  await expect(page.getByTestId("future-card-2")).toContainText("worn");
  await expect(page.getByTestId("wear-receipt")).toContainText("receipt");
  await expect(page.getByTestId("transcript-loom")).toContainText("wore member code #2");
  await noHorizontalScroll(page);
  await shot(page, "04-worn");

  // the next SEND runs under the wear
  await page.getByTestId("send-btn").click();
  await expect(page.getByTestId("transcript-loom")).toContainText("under code #2");
  await expect(page.getByTestId("transcript-loom")).toContainText("bent · #2");

  // UNWEAR
  await page.getByTestId("unwear-btn").click();
  await expect(page.getByTestId("wearbar")).toContainText("NO CODE WORN");
  await expect(page.getByTestId("transcript-loom")).toContainText("unwore code #2");
});

test("light theme, and the code-space hover", async ({ page }) => {
  await page.goto("/");
  await page
    .getByTestId("session-table")
    .getByRole("row", { name: /lx-speak-k16/ })
    .getByRole("button", { name: "open" })
    .click();
  await expect(page.getByTestId("cards").locator("article")).toHaveCount(16);
  await page.getByRole("button", { name: "auto", exact: true }).click(); // → dark
  await page.getByRole("button", { name: "dark", exact: true }).click(); // → light
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  // hover the canvas until a body answers (positions depend on the fan)
  const canvas = page.getByTestId("codespace-canvas");
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (box && test.info().project.name === "desktop") {
    let found = false;
    for (let y = 0.1; y < 0.9 && !found; y += 0.05) {
      for (let x = 0.05; x < 0.95 && !found; x += 0.03) {
        await page.mouse.move(box.x + box.width * x, box.y + box.height * y);
        found = await page.getByTestId("codespace-tip").isVisible();
      }
    }
    expect(found, "no body answered a hover").toBe(true);
    await expect(page.getByTestId("codespace-tip")).toContainText("pull ×");
  }
  await shot(page, "05-light-k16");
});

test("an unreachable server is reported, not guessed at", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "connection settings" }).click();
  await page.locator("#base-url").fill("http://127.0.0.1:9");
  await page.getByTestId("connect-btn").click();
  await expect(page.getByTestId("connect-error")).toContainText("could not reach the server");
  await shot(page, "06-connect-error");
});
