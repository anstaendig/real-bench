import { test, expect } from "@playwright/test";

test("filter and inspect retained live data alongside an unsupported different setup", async ({
  page,
  request,
}) => {
  const initial = await (await request.get("/api/reports")).json();
  const live = initial.runs.find(
    (run: { run_id: string }) =>
      run.run_id === "c0743393eafd43ec81c4cac3679148d5",
  );
  expect(live).toBeDefined();
  expect(live.coverage.fresh_context.successful).toBe(60);
  const setup = structuredClone(live.setup);
  setup.declared.model.source = "browser-test/unsupported-model";
  setup.declared.model.quantized_artifact = "browser-test-q8";
  setup.declared.requests.context_window = 1;
  // Inject an unsupported fixture for comparison without launching a benchmark.
  const unsupported = {
    ...structuredClone(live),
    run_id: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    setup,
    workload: {
      ...live.workload,
      profile: "growing-session-speed",
      scenario: "growing-session",
    },
    status: "partial",
    outcomes: live.outcomes.map((row: object) => ({
      ...row,
      outcome: "unsupported",
      metrics: {},
    })),
    coverage: {
      growing_session: { expected: 60, accounted: 60, unsupported: 60 },
    },
    parameters: {
      ...live.parameters,
      "setup.declared.model.source": "browser-test/unsupported-model",
      "setup.declared.model.quantized_artifact": "browser-test-q8",
      "setup.declared.requests.context_window": 1,
      "workload.profile": "growing-session-speed",
      "workload.scenario": "growing-session",
    },
  };
  delete unsupported.fresh_context_curve;
  const mocked = structuredClone(initial);
  mocked.runs.push(unsupported);
  mocked.parameters["setup.declared.model.source"].push(
    "browser-test/unsupported-model",
  );
  await page.route("**/api/reports?*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mocked),
    }),
  );
  expect(
    unsupported.outcomes.every(
      (row: { outcome: string }) => row.outcome === "unsupported",
    ),
  ).toBe(true);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/reports");
  await expect(page.getByRole("status")).toContainText("4 retained runs");
  await page.getByLabel(`Compare ${live.run_id}`, { exact: true }).check();
  await page
    .getByLabel(`Compare ${unsupported.run_id}`, { exact: true })
    .check();
  await page.locator('[data-view="compare"]').click();
  const cards = page.locator("#comparison .chart-card");
  await expect(cards).toHaveCount(2);
  await expect(
    page.getByRole("img", { name: /fresh-context Time to first token/ }),
  ).toBeVisible();
  await expect(cards.locator("table")).toHaveCount(0);
  const graph = cards.first().locator("svg");
  await expect(graph.locator(".chart-x-tick")).toHaveText([
    "4,096",
    "16,384",
    "32,768",
    "65,536",
    "131,072",
    "196,608",
  ]);
  const positions = await graph
    .locator(".chart-x-tick")
    .evaluateAll((ticks) =>
      ticks.map((tick) => Number(tick.getAttribute("x"))),
    );
  for (let i = 2; i < positions.length; i++) {
    expect(positions[i] - positions[i - 1]).toBeCloseTo(
      positions[1] - positions[0],
    );
  }
  const yTicks = (await graph.locator(".chart-y-tick").allTextContents()).map(
    (text) => Number(text.replaceAll(",", "")),
  );
  expect(yTicks[0]).toBe(0);
  expect(yTicks.every(Number.isInteger)).toBe(true);
  for (let i = 2; i < yTicks.length; i++) {
    expect(yTicks[i] - yTicks[i - 1]).toBe(yTicks[1] - yTicks[0]);
  }
  await expect(page.locator("#comparison")).toContainText("unsupported");
  await expect(page.locator("#parameters")).toContainText(
    "workload.profile · differs",
  );
  await page.getByLabel("Curve metric").selectOption("decode_rate");
  await page
    .getByRole("combobox", { name: "Output target" })
    .selectOption("1024");
  await expect(
    page.getByRole("img", {
      name: /fresh-context Decode rate .*output target 1024/,
    }),
  ).toBeVisible();
  await page.locator('[data-view="runs"]').click();
  await page.locator(".filter-panel summary").click();
  await page.locator("#field").selectOption("setup.declared.model.source");
  await page
    .locator("#value")
    .selectOption(JSON.stringify("browser-test/unsupported-model"));
  await page.getByRole("button", { name: "Add filter", exact: true }).click();
  await expect(page.locator("#runs .run-row")).toHaveCount(1);
  await page.locator('[data-view="compare"]').click();
  await expect(cards).toHaveCount(2);
  await page.locator('[data-view="runs"]').click();
  await page
    .getByRole("button", { name: "Clear filters", exact: true })
    .click();
  await expect(page.locator("#runs .run-row")).toHaveCount(4);
  const before = await (await request.get("/api/reports")).json();
  await page.locator("#rebuild").click();
  await expect(page.getByRole("status")).toContainText("3 retained runs");
  const after = await (await request.get("/api/reports")).json();
  expect(after).toEqual(before);
  await page.locator('[data-view="compare"]').click();
  await expect(cards).toHaveCount(2);
  expect(errors).toEqual([]);
  await page.screenshot({
    path: "test-results/reports-comparison.png",
    fullPage: true,
  });
});

test("inspect corrected quality data alongside retained partial coverage", async ({
  page,
}) => {
  await page.goto("/reports");
  const ready = "13713713713713713713713713713713";
  const partial = "13413413413413413413413413413413";
  await page.getByLabel(`Compare ${ready}`, { exact: true }).check();
  await page.getByLabel(`Compare ${partial}`, { exact: true }).check();
  await page.locator('[data-view="compare"]').click();

  await expect(page.locator("#comparison")).toContainText("50%");
  await expect(page.locator("#comparison")).toContainText("90.24%");
  await expect(page.locator("#comparison")).toContainText("117 scored");
  await expect(page.locator("#comparison")).toContainText("110 scored");
  await expect(page.locator("#comparison")).toContainText(
    "not a full official RULER score",
  );
  await page.goto(`/runs/${ready}`);
  const readyRaw = page
    .locator("#run-detail .raw-details")
    .filter({ hasText: "Retained summary and setup" });
  await readyRaw.locator("summary").click();
  await expect(readyRaw).toContainText('"status": "verified"');
  await expect(readyRaw).toContainText("endpoint-authoritative");
  await expect(readyRaw).toContainText('"minimum": 32026');
  await expect(readyRaw).toContainText('"maximum": 194731');
  await page.goto(`/runs/${partial}`);
  const partialRaw = page
    .locator("#run-detail .raw-details")
    .filter({ hasText: "Retained summary and setup" });
  await partialRaw.locator("summary").click();
  await expect(partialRaw).toContainText('"scored": 110');
  await expect(partialRaw).toContainText('"failed": 7');
});

test("grouped bars show absolute values without overlapping", async ({
  page,
  request,
}) => {
  const report = await (await request.get("/api/reports")).json();
  const baseline = structuredClone(report.runs[0]);
  baseline.run_id = "aa".repeat(16);
  baseline.fresh_context_curve = [4096, 16384, 32768, 65536, 131072].map(
    (input_tokens, i) => ({
      input_tokens,
      output_target: 256,
      metrics: { ttft: { mean: i === 4 ? 0 : 1000, unit: "ms" } },
    }),
  );
  const compared = structuredClone(baseline);
  compared.run_id = "bb".repeat(16);
  compared.fresh_context_curve.forEach(
    (point: { metrics: { ttft: { mean: number | null } } }, i: number) => {
      point.metrics.ttft.mean = [1000, 1001, 999, null, 1000][i];
    },
  );
  await page.route("**/api/reports?*", (route) =>
    route.fulfill({ json: { ...report, runs: [baseline, compared] } }),
  );
  await page.goto("/reports");
  await page.getByLabel(`Compare ${baseline.run_id}`, { exact: true }).check();
  await page.getByLabel(`Compare ${compared.run_id}`, { exact: true }).check();
  await page.locator('[data-view="compare"]').click();
  const card = page.locator("#comparison .chart-card").first();
  await expect(card.locator(".comparison-bar")).toHaveCount(9);
  await expect(card.locator(".bar-value")).toHaveText([
    "1",
    "1",
    "1",
    "1.001",
    "1",
    "0.999",
    "1",
    "N/A",
    "0",
    "1",
  ]);
  const bars = await card
    .locator('.comparison-bar[data-context="4096"]')
    .evaluateAll((elements) =>
      elements.map((e) => ({
        x: Number(e.getAttribute("x")),
        y: Number(e.getAttribute("y")),
        width: Number(e.getAttribute("width")),
        height: Number(e.getAttribute("height")),
      })),
    );
  expect(bars[0].y).toBe(bars[1].y);
  expect(bars[0].x + bars[0].width).toBeLessThan(bars[1].x);
  expect(bars[0].y + bars[0].height).toBe(215);
  await expect(card.locator(".comparison-bar title").first()).toContainText(
    "1 s",
  );
  await expect(card.locator(".difference-view")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("three nearly identical runs have compact non-overlapping absolute labels", async ({
  page,
  request,
}) => {
  const report = await (await request.get("/api/reports")).json();
  const runs = ["aa", "bb", "cc"].map((prefix, index) => ({
    ...structuredClone(report.runs[0]),
    run_id: prefix.repeat(16),
    fresh_context_curve: [4096, 16384, 32768, 65536, 131072, 196608].map(
      (input_tokens) => ({
        input_tokens,
        output_target: 256,
        metrics: { ttft: { mean: 1000 - index * 20, unit: "ms" } },
      }),
    ),
  }));
  runs.forEach((run, index) => {
    run.setup.declared.serving.environment = {
      cache_mode: ["Q4", "Q8", "FP16"][index],
      draft_mode: "mtp",
      draft_num_tokens: index + 1,
    };
    run.setup.declared.requests.reasoning = {
      reasoning_effort: ["low", "medium", "high"][index],
    };
  });
  await page.route("**/api/reports?*", (route) =>
    route.fulfill({ json: { ...report, runs } }),
  );
  await page.goto("/reports");
  for (const run of runs)
    await page.getByLabel(`Compare ${run.run_id}`, { exact: true }).check();
  await page.locator('[data-view="compare"]').click();
  for (const [index, cache] of ["Q4", "Q8", "FP16"].entries()) {
    const label = `Run ${String.fromCharCode(65 + index)}`;
    await expect(page.locator(".compare-run-head").nth(index)).toContainText(
      `KV cache: ${cache}`,
    );
    for (const cardIndex of [index * 2, index * 2 + 1]) {
      const card = page.locator("#comparison .quality-card").nth(cardIndex);
      await expect(card.locator(".comparison-identity")).toContainText(label);
      await expect(card.locator(".comparison-config")).toContainText(
        `KV cache: ${cache}`,
      );
    }
    await expect(
      page
        .locator("#comparison .chart-card")
        .first()
        .locator(".comparison-identity")
        .nth(index),
    ).toContainText(`KV cache: ${cache}`);
  }
  const svg = page.locator("#comparison .chart-card").first().locator("svg");
  await expect(svg).toHaveAttribute("data-labels-laid-out", "true");
  for (const width of [1929, 900, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect
      .poll(() =>
        svg.locator(".bar-value").evaluateAll((labels) => {
          const boxes = labels.map((label) => label.getBoundingClientRect());
          return boxes.every(
            (box, i) =>
              box.height <= 17 &&
              boxes.every(
                (other, j) =>
                  i === j ||
                  box.right <= other.left ||
                  other.right <= box.left ||
                  box.bottom <= other.top ||
                  other.bottom <= box.top,
              ),
          );
        }),
      )
      .toBe(true);
  }
});

test("exports performance, coding and RULER comparisons as PNG", async ({
  page,
}, testInfo) => {
  await page.goto("/reports");
  await page
    .getByLabel("Compare c0743393eafd43ec81c4cac3679148d5", { exact: true })
    .check();
  await page
    .getByLabel("Compare 13713713713713713713713713713713", { exact: true })
    .check();
  await page.locator('[data-view="compare"]').click();
  const buttons = [
    page
      .locator("#comparison .chart-card")
      .first()
      .getByRole("button", { name: "Export PNG", exact: true }),
    page.getByRole("button", { name: "Export Coding PNG", exact: true }),
    page.getByRole("button", { name: "Export RULER PNG", exact: true }),
  ];
  const { readFile } = await import("node:fs/promises");
  for (const [index, button] of buttons.entries()) {
    const pending = page.waitForEvent("download");
    await button.click();
    const download = await pending;
    const path = testInfo.outputPath(`export-${index}.png`);
    await download.saveAs(path);
    const png = await readFile(path);
    expect(png.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
    expect(png.readUInt32BE(16)).toBe(1800);
    expect(png.readUInt32BE(20)).toBeGreaterThan(400);
    await expect(button).toBeEnabled();
  }
  await page.goto("/runs/c0743393eafd43ec81c4cac3679148d5");
  const pending = page.waitForEvent("download");
  await page
    .locator("#run-detail .chart-card")
    .first()
    .getByRole("button", { name: "Export PNG", exact: true })
    .click();
  expect((await pending).suggestedFilename()).toMatch(/\.png$/);
});
