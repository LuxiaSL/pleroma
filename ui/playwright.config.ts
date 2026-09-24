/**
 * E2E against the MOCK backend (vite --mode mock): no GPU, no loom server.
 * Screenshots land in $UI_SHOTS_DIR (default test-results/shots).
 */
import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.E2E_PORT || 5199);

export default defineConfig({
  testDir: "e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 }, colorScheme: "dark" },
    },
    {
      name: "narrow",
      use: { ...devices["Desktop Chrome"], viewport: { width: 400, height: 860 }, colorScheme: "light" },
    },
  ],
  webServer: {
    command: `npx vite --mode mock --port ${port} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${port}/`,
    reuseExistingServer: false,
    timeout: 60_000,
    env: { MOCK_LOOM_SECONDS: "2.5" },
  },
});
