import { expect, test, type Page } from "@playwright/test";

const speedId = "c0743393eafd43ec81c4cac3679148d5";
const completeId = "13713713713713713713713713713713";

test("run cards distinguish configuration variants and search recorded settings", async ({
  page,
}) => {
  const report = await retainedRuns(page);
  const first = structuredClone(
    report.runs.find((r: { run_id: string }) => r.run_id === speedId),
  );
  const second = structuredClone(first);
  second.run_id = "ac".repeat(16);
  first.setup.declared.serving.environment = { cache_mode: "Q4" };
  second.setup.declared.serving.environment = {
    cache_mode: "8,4",
    scheduler: "custom-scheduler",
  };
  first.setup.declared.requests.reasoning = {
    chat_template_kwargs: { reasoning_effort: "medium" },
  };
  second.setup.declared.requests.reasoning = {
    chat_template_kwargs: { reasoning_effort: "xhigh" },
  };
  await page.route("**/api/reports?*", (route) =>
    route.fulfill({ json: { ...report, runs: [first, second] } }),
  );
  await page.goto("/reports");
  const rows = page.locator("#runs .run-row");
  await expect(rows).toHaveCount(2);
  await expect(
    rows
      .filter({
        has: page.getByLabel(`Compare ${first.run_id}`, { exact: true }),
      })
      .locator(".configuration-chips"),
  ).toContainText("medium");
  await expect(
    rows
      .filter({
        has: page.getByLabel(`Compare ${first.run_id}`, { exact: true }),
      })
      .locator(".configuration-chips"),
  ).toContainText("Q4");
  await expect(
    rows
      .filter({
        has: page.getByLabel(`Compare ${second.run_id}`, { exact: true }),
      })
      .locator(".configuration-chips"),
  ).toContainText("xhigh");
  await expect(
    rows
      .filter({
        has: page.getByLabel(`Compare ${second.run_id}`, { exact: true }),
      })
      .locator(".configuration-chips"),
  ).toContainText("8,4");
  await page.getByLabel("Search runs").fill("custom-scheduler");
  await expect(rows).toHaveCount(1);
  await rows.first().getByText("All settings", { exact: true }).click();
  await expect(rows.first().locator(".configuration-fields")).toContainText(
    "custom-scheduler",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(rows.first().locator(".configuration-chips")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth + 1,
    ),
  ).toBe(false);
});

async function retainedRuns(page: Page) {
  const response = await page.request.get("/api/reports");
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test("live monitor refreshes verified progress and observed metrics", async ({
  page,
}) => {
  const report = await retainedRuns(page);
  const retained = report.runs.find(
    (run: { run_id: string }) => run.run_id === speedId,
  );
  expect(retained).toBeDefined();
  const observed = retained.outcomes.slice(0, 4);
  const phases = [
    {
      name: "fresh_context",
      status: "running",
      expected_units: 60,
      accounted_units: 4,
    },
    {
      name: "growing_session",
      status: "pending",
      expected_units: 60,
      accounted_units: 0,
    },
    {
      name: "coding",
      status: "pending",
      expected_units: 164,
      accounted_units: 0,
    },
    {
      name: "long_context",
      status: "pending",
      expected_units: 117,
      accounted_units: 0,
    },
  ];
  const active = {
    ...retained,
    status: "partial",
    workload: { ...retained.workload, profile: "default-suite" },
    lifecycle: { state: "active", verified: true },
    observed_outcomes: observed.map((row: object) => ({
      phase: "fresh_context",
      ...row,
    })),
    phase_progress: {
      current: "fresh_context",
      completed: 0,
      total: 4,
      phases,
    },
    coverage: { fresh_context: { expected: 60, accounted: 4, successful: 4 } },
  };
  let snapshot = {
    generated_at: "2026-09-26T10:00:00Z",
    runs: [active],
    errors: [],
  };
  await page.route("**/api/dashboard", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(snapshot),
    }),
  );
  await page.goto("/reports");
  await page.locator('[data-view="live"]').click();
  await expect(page.locator("#live-content")).toContainText(speedId);
  await expect(page.locator("#live-content .phase-row").first()).toContainText(
    "4 / 60",
  );
  await expect(
    page
      .locator("#live-content .metric-card")
      .filter({ hasText: "Observed outcomes" }),
  ).toContainText("4");
  await expect(page.locator("#live-content")).toContainText("930.28 ms");

  const growing = retained.outcomes
    .slice(0, 5)
    .map((row: { metrics: Record<string, object> }, index: number) => ({
      ...row,
      phase: "growing_session",
      metrics: {
        ...row.metrics,
        ttft: { ...row.metrics.ttft, value: 2000 },
        completion_latency: {
          value: [60000, 185634, 3600000, 3661000, 59999][index],
          unit: "ms",
        },
      },
    }));
  snapshot = {
    ...snapshot,
    generated_at: "2026-09-26T10:01:00Z",
    runs: [
      {
        ...active,
        phase_progress: {
          current: "growing_session",
          completed: 1,
          total: 4,
          phases: [
            { ...phases[0], status: "completed", accounted_units: 60 },
            { ...phases[1], status: "running", accounted_units: 5 },
            ...phases.slice(2),
          ],
        },
        observed_outcomes: [
          ...retained.outcomes.map((row: object) => ({
            phase: "fresh_context",
            ...row,
          })),
          ...growing,
        ],
        coverage: {
          fresh_context: { expected: 60, accounted: 60, successful: 60 },
          growing_session: { expected: 60, accounted: 5, successful: 5 },
        },
      },
    ],
  };
  await page.locator("#live-refresh").click();
  await expect(page.locator("#live-content .phase-row").nth(0)).toContainText(
    "60 / 60",
  );
  await expect(page.locator("#live-content .phase-row").nth(0)).toContainText(
    "completed",
  );
  await expect(page.locator("#live-content .phase-row").nth(1)).toContainText(
    "5 / 60",
  );
  await expect(page.locator("#live-content .phase-row").nth(1)).toContainText(
    "running",
  );
  await expect(page.locator("#live-content")).toContainText("2 s");
  for (const duration of [
    "1 min",
    "3 min 6 s",
    "1 h",
    "1 h 1 min 1 s",
    "60 s",
  ]) {
    await expect(
      page.locator("#live-content td").getByText(duration, { exact: true }),
    ).toBeVisible();
  }
});

test("catalog filters completed runs and direct links reveal every result family", async ({
  page,
}) => {
  await page.goto("/reports");
  await page.locator(".filter-panel summary").click();
  await page.locator("#field").selectOption("status");
  await page.locator("#value").selectOption(JSON.stringify("completed"));
  await page.locator("#add-filter").click();
  await expect(page.locator("#runs")).toContainText(completeId);
  await page.goto(`/runs/${completeId}`);
  await expect(page.locator("#run-detail")).toContainText(completeId);
  await expect(page.locator("#run-detail")).toContainText(/performance/i);
  await expect(page.locator("#run-detail")).toContainText(/coding/i);
  await expect(page.locator("#run-detail")).toContainText(
    /long.context|RULER/i,
  );
  await expect(page.locator("#run-detail")).toContainText(/coverage/i);
  await page.goto(`/runs/${speedId}`);
  await expect(page.locator("#run-detail")).toContainText(speedId);
  await expect(page.locator("#run-detail")).toContainText(
    /time to first token|ttft/i,
  );
  await expect(page.locator("#run-detail")).toContainText(/decode rate/i);
  await expect(page.locator("#run-detail")).toContainText(
    /completion latency/i,
  );
  await expect(page.locator("#run-detail")).toContainText(
    /prefill throughput/i,
  );
});

test("empty and failed live reads recover on refresh without presenting stale data as current", async ({
  page,
}) => {
  const report = await retainedRuns(page);
  const retained = report.runs.find(
    (run: { run_id: string }) => run.run_id === speedId,
  );
  let mode: "empty" | "error" | "recovered" = "empty";
  await page.route("**/api/dashboard", (route) => {
    if (mode === "error")
      return route.fulfill({ status: 503, body: "unavailable" });
    const runs =
      mode === "recovered"
        ? [
            {
              ...retained,
              lifecycle: { state: "unverified", verified: false },
              observed_outcomes: retained.outcomes,
            },
          ]
        : [];
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        generated_at: "2026-09-26T10:00:00Z",
        runs,
        errors: [{ run_id: "dead".repeat(8), diagnostic: "Unreadable run" }],
      }),
    });
  });
  await page.goto("/reports");
  await page.locator('[data-view="live"]').click();
  await expect(page.locator("#live-content")).toContainText(
    /no.*run|no.*active|nothing/i,
  );
  mode = "error";
  await page.locator("#live-refresh").click();
  await expect(page.locator("#live-banner")).toContainText(
    /error|unavailable|failed|stale/i,
  );
  mode = "recovered";
  await page.locator("#live-refresh").click();
  await expect(page.locator("#live-content")).toContainText(speedId);
  await expect(page.locator("#live-content")).toContainText(
    /unverified|stale|not active/i,
  );
  mode = "error";
  await page.locator("#live-refresh").click();
  await expect(page.locator("#live-banner")).toContainText(
    "Showing retained snapshot",
  );
  await expect(page.locator("#live-banner")).toContainText("unreadable");
  await expect(page.locator("#live-content")).toContainText(speedId);
});

test("comparison keeps scenario and output target separate at narrow width", async ({
  page,
}) => {
  const report = await retainedRuns(page);
  const fresh = report.runs.find(
    (run: { run_id: string }) => run.run_id === speedId,
  );
  const growingId = "abababababababababababababababab";
  const growing = structuredClone(fresh);
  growing.run_id = growingId;
  growing.workload = { ...growing.workload, scenario: "growing-session" };
  growing.growing_session_curve = growing.fresh_context_curve.map(
    (point: Record<string, unknown>) => ({
      ...point,
      scenario: "growing-session",
    }),
  );
  delete growing.fresh_context_curve;
  await page.route("**/api/reports?*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...report, runs: [fresh, growing] }),
    }),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/reports");
  await page.getByLabel(`Compare ${speedId}`, { exact: true }).check();
  await page.getByLabel(`Compare ${growingId}`, { exact: true }).check();
  await page.locator('[data-view="compare"]').click();
  await expect(page.locator("#comparison")).toContainText(/fresh.context/i);
  await expect(page.locator("#comparison")).toContainText(/growing.session/i);
  await page.locator("#target").selectOption("1024");
  await expect(page.locator("#comparison")).toContainText("1024");
  const horizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth + 1,
  );
  expect(horizontalOverflow).toBe(false);
  await page.screenshot({
    path: "test-results/reports-dashboard-responsive.png",
    fullPage: true,
  });
});
