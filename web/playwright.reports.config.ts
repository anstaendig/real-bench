import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-reports",
  use: { ...devices["Desktop Chrome"], baseURL: "http://127.0.0.1:4174" },
  webServer: {
    cwd: "..",
    command: "uv run python scripts/report_browser_server.py",
    url: "http://127.0.0.1:4174/reports",
    reuseExistingServer: false,
  },
});
