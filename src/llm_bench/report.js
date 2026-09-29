"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    report: { runs: [], parameters: {}, errors: [] },
    selected: new Set(),
    filters: [],
    query: "",
    detail: null,
    view: "runs",
    live: null,
    liveRun: null,
    reportBusy: false,
    liveBusy: false,
    liveFailure: false,
    lastLiveOk: null,
    timer: null,
  };
  const colors = [
    "#ff9a52",
    "#70d8d0",
    "#bba2ff",
    "#79baff",
    "#e3cf78",
    "#f58eb5",
  ];
  const metricInfo = {
    ttft: [
      "Time to first token",
      "Endpoint timing from request start to first token",
    ],
    decode_rate: [
      "Decode rate",
      "Generated tokens per second after the first token",
    ],
    completion_latency: [
      "Completion latency",
      "End-to-end completion time for the request",
    ],
    effective_endpoint_observed_prefill_throughput: [
      "Effective endpoint-observed prefill throughput",
      "Input tokens divided by TTFT; includes endpoint and transport effects",
    ],
  };
  const node = (tag, text = "", className = "") => {
    const n = document.createElement(tag);
    n.textContent = text;
    if (className) n.className = className;
    return n;
  };
  const append = (parent, tag, text = "", className = "") => {
    const n = node(tag, text, className);
    parent.append(n);
    return n;
  };
  const badge = (value) => {
    const n = node(
      "span",
      value == null ? "unavailable" : String(value).replaceAll("_", " "),
      `badge ${value || "missing"}`,
    );
    return n;
  };
  const str = (value) =>
    value == null
      ? "Unavailable"
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
  const fmt = (value) =>
    typeof value === "number" && Number.isFinite(value)
      ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(
          value,
        )
      : "Unavailable";
  const duration = (milliseconds) => {
    if (!Number.isFinite(milliseconds)) return "Unavailable";
    if (milliseconds < 1000) return `${fmt(milliseconds)} ms`;
    if (milliseconds < 60000) return `${fmt(milliseconds / 1000)} s`;
    const seconds = Math.round(milliseconds / 1000);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    return [
      hours && `${fmt(hours)} h`,
      minutes && `${minutes} min`,
      remainder && `${remainder} s`,
    ]
      .filter(Boolean)
      .join(" ");
  };
  const timeFactor = (unit) => ({ ms: 1, s: 1000, seconds: 1000 })[unit];
  const measurement = (value, unit) => {
    if (!Number.isFinite(value)) return "Unavailable";
    const factor = timeFactor(unit);
    return factor ? duration(value * factor) : `${fmt(value)} ${unit || ""}`;
  };
  // Every series in an overlay uses the same scale; retained values stay unchanged.
  const axisScale = (unit, maximum) => {
    const factor = timeFactor(unit);
    if (!factor) return { divisor: 1, unit };
    const ms = maximum * factor;
    const [divisor, label] =
      ms >= 3600000
        ? [3600000, "hours"]
        : ms >= 60000
          ? [60000, "minutes"]
          : ms >= 1000
            ? [1000, "seconds"]
            : [1, "ms"];
    return { divisor: divisor / factor, unit: label };
  };
  const pct = (value) =>
    typeof value === "number" && Number.isFinite(value)
      ? `${fmt(value * 100)}%`
      : "Unavailable";
  const short = (id) => (id ? id.slice(0, 8) + "…" : "Unavailable");
  const profile = (r) =>
    r?.workload?.profile || r?.workload?.scenario || "unspecified";
  const model = (r) =>
    r?.setup?.declared?.model?.source ||
    r?.setup?.observed?.model?.source ||
    short(r?.run_id);
  const artifact = (r) =>
    r?.setup?.declared?.model?.quantized_artifact || "Unspecified artifact";
  const hardware = (r) =>
    r?.setup?.declared?.hardware?.accelerator || "Unspecified hardware";
  const engine = (r) =>
    r?.setup?.declared?.serving?.engine || "Unspecified engine";
  const reasoning = (r) => {
    const requests = r.setup?.declared?.requests || {};
    const settings = requests.reasoning || {};
    const template = settings.chat_template_kwargs || {};
    if (
      template.enable_thinking === false ||
      settings.enable_thinking === false
    )
      return "off";
    return (
      settings.reasoning_effort ??
      template.reasoning_effort ??
      requests.reasoning_effort ??
      "Not recorded"
    );
  };
  function recordedSettings(r) {
    const fields = [];
    const walk = (value, path) => {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        for (const [key, item] of Object.entries(value))
          walk(item, path ? `${path}.${key}` : key);
      } else fields.push([path, value]);
    };
    walk(r.setup?.declared || {}, "setup");
    walk(r.workload || {}, "workload");
    return fields;
  }
  const runLetter = (index) => `Run ${String.fromCharCode(65 + index)}`;
  function configurationSummary(r) {
    const env = r.setup?.declared?.serving?.environment || {};
    const cache =
      env.cache_mode ?? env.kv_cache_dtype ?? env.cache_dtype ?? "Not recorded";
    const draft =
      env.draft_mode == null
        ? "Not recorded"
        : `${env.draft_mode}${env.draft_num_tokens == null ? "" : ` / ${env.draft_num_tokens} tokens`}`;
    return `${artifact(r)} · ${engine(r)} · Effort: ${reasoning(r)} · KV cache: ${cache} · Draft: ${draft}`;
  }
  function comparisonIdentity(parent, r, index) {
    const identity = append(parent, "div", "", "comparison-identity");
    const label = append(identity, "div", "", "comparison-identity-label");
    const swatch = append(label, "span", "", "swatch");
    swatch.style.backgroundColor = colors[index % colors.length];
    append(label, "strong", runLetter(index));
    append(identity, "div", configurationSummary(r), "comparison-config");
    return identity;
  }
  function renderRunSettings(parent, r) {
    const requests = r.setup?.declared?.requests || {};
    const environment = r.setup?.declared?.serving?.environment || {};
    const generation = requests.generation || {};
    const block = append(parent, "div", "", "run-settings");
    const chips = append(block, "div", "", "configuration-chips");
    const settings = [
      ["Effort", reasoning(r)],
      [
        "KV cache",
        environment.cache_mode ??
          environment.kv_cache_dtype ??
          environment.cache_dtype ??
          "Not recorded",
      ],
      [
        "Context",
        requests.context_window == null
          ? "Not recorded"
          : `${fmt(requests.context_window)} tokens`,
      ],
    ];
    if (environment.draft_mode != null)
      settings.push([
        "Draft",
        `${environment.draft_mode}${environment.draft_num_tokens == null ? "" : ` · ${environment.draft_num_tokens} tokens`}`,
      ]);
    if (generation.temperature != null)
      settings.push(["Temperature", generation.temperature]);
    if (generation.top_p != null) settings.push(["Top p", generation.top_p]);
    if (generation.top_k != null) settings.push(["Top k", generation.top_k]);
    for (const [label, value] of settings) {
      const chip = append(chips, "span", "", "configuration-chip");
      append(chip, "span", label);
      append(chip, "strong", str(value));
    }
    const details = append(block, "details", "", "configuration-details");
    append(details, "summary", "All settings");
    append(
      details,
      "p",
      "Recorded setup and workload for this run. Missing settings are not inferred.",
    );
    const list = append(details, "dl", "", "configuration-fields");
    for (const [path, value] of recordedSettings(r)) {
      const item = append(list, "div");
      append(item, "dt", path.replaceAll("_", " ").replaceAll(".", " / "));
      append(item, "dd", value === null ? "Not specified (null)" : str(value));
    }
  }
  const option = (select, value, label) => {
    const o = node("option", label);
    o.value = value;
    select.append(o);
  };
  const empty = (parent, title, body) => {
    const wrap = append(parent, "div", "", "empty-state panel");
    append(wrap, "strong", title);
    append(wrap, "span", body);
    return wrap;
  };
  const section = (parent, title, description = "") => {
    const wrap = append(parent, "section", "", "section-block");
    const head = append(wrap, "div", "", "section-heading");
    append(head, "h2", title);
    if (description) append(head, "p", description);
    return wrap;
  };
  const table = (parent, headings, rows) => {
    const scroll = append(parent, "div", "", "table-scroll");
    const t = append(scroll, "table", "", "data-table");
    const thead = append(t, "thead");
    const hr = append(thead, "tr");
    headings.forEach((h) => append(hr, "th", h));
    const body = append(t, "tbody");
    rows.forEach((row) => {
      const tr = append(body, "tr");
      row.forEach((value) => append(tr, "td", str(value)));
    });
    return t;
  };
  const raw = (parent, title, value) => {
    const d = append(parent, "details", "", "raw-details");
    append(d, "summary", title);
    append(d, "pre", JSON.stringify(value, null, 2));
  };
  const stat = (parent, label, value, note = "", emphasize = false) => {
    const card = append(
      parent,
      "div",
      "",
      `metric-card${emphasize ? " emphasis" : ""}`,
    );
    append(card, "span", label, "label");
    append(card, "strong", value);
    if (note) append(card, "small", note);
    return card;
  };
  const chooseView = (view) => {
    if (!["live", "runs", "compare"].includes(view)) view = "runs";
    const previousView = state.view;
    state.view = view;
    for (const name of ["live", "runs", "compare"]) {
      $(`view-${name}`).hidden = name !== view;
      const b = document.querySelector(`[data-view="${name}"]`);
      b.classList.toggle("active", name === view);
      b.setAttribute("aria-current", name === view ? "page" : "false");
    }
    $("breadcrumb-view").textContent = {
      live: "LIVE MONITOR",
      runs: "RUN LIBRARY",
      compare: "COMPARE",
    }[view];
    if (location.hash !== `#${view}`)
      history.replaceState(null, "", `#${view}`);
    if (view === "live") loadLive();
    if (view === "runs" && previousView !== "runs" && state.report.runs.length)
      loadReport();
    if (view === "compare") renderComparison();
  };
  function fillFacets() {
    const field = $("field"),
      sort = $("sort"),
      priorField = field.value,
      priorSort = sort.value;
    field.replaceChildren();
    sort.replaceChildren();
    option(sort, "added_at", "Date added");
    option(sort, "run_id", "Run ID");
    Object.keys(state.report.parameters || {})
      .sort()
      .forEach((key) => {
        option(field, key, key);
        option(sort, key, key);
      });
    if (priorField in (state.report.parameters || {})) field.value = priorField;
    if (
      ["run_id", "added_at"].includes(priorSort) ||
      priorSort in (state.report.parameters || {})
    )
      sort.value = priorSort;
    fillValues();
  }
  function fillValues() {
    const v = $("value");
    v.replaceChildren();
    (state.report.parameters[$("field").value] || []).forEach((item) =>
      option(v, JSON.stringify(item), str(item)),
    );
  }
  function matches(r) {
    const q = state.query.trim().toLowerCase();
    if (
      q &&
      ![
        r.run_id,
        model(r),
        artifact(r),
        hardware(r),
        engine(r),
        profile(r),
        r.status,
        reasoning(r),
        ...recordedSettings(r).map(([path, value]) => `${path} ${str(value)}`),
      ].some((v) =>
        String(v || "")
          .toLowerCase()
          .includes(q),
      )
    )
      return false;
    return state.filters.every(
      ([key, value]) =>
        key in (r.parameters || {}) &&
        JSON.stringify(r.parameters[key]) === value,
    );
  }
  function sortedRuns() {
    const key = $("sort").value || "added_at",
      desc = $("descending").checked;
    const rows = state.report.runs.filter(matches);
    rows.sort((a, b) => {
      const value = (run) =>
        key === "added_at"
          ? state.report.added_at?.[run.run_id]
          : key === "run_id"
            ? run.run_id
            : run.parameters?.[key];
      const x = value(a);
      const y = value(b);
      if (x == null && y != null) return 1;
      if (y == null && x != null) return -1;
      if (typeof x === "number" && typeof y === "number")
        return desc ? y - x : x - y;
      return desc
        ? JSON.stringify(y ?? null).localeCompare(JSON.stringify(x ?? null))
        : JSON.stringify(x ?? null).localeCompare(JSON.stringify(y ?? null));
    });
    return rows;
  }
  function renderRuns() {
    const rows = sortedRuns();
    $("runs").replaceChildren();
    $("run-total").textContent =
      `${state.report.runs.length} retained run${state.report.runs.length === 1 ? "" : "s"}`;
    $("filter-count").textContent = state.filters.length
      ? `${state.filters.length} active`
      : "";
    $("filters").replaceChildren();
    state.filters.forEach(([key, value], i) => {
      const c = append($("filters"), "button", `${key} = ${value} ×`, "chip");
      c.type = "button";
      c.setAttribute("aria-label", `Remove filter ${key} ${value}`);
      c.onclick = () => {
        state.filters.splice(i, 1);
        renderRuns();
      };
    });
    if (!rows.length) {
      empty(
        $("runs"),
        state.report.runs.length ? "No matching runs" : "No retained runs yet",
        state.report.runs.length
          ? "Try another search or clear the filters."
          : "Runs appear here after their first retained result is available.",
      );
      return;
    }
    for (const r of rows) {
      const row = append(
        $("runs"),
        "article",
        "",
        `run-row${state.selected.has(r.run_id) ? " selected" : ""}`,
      );
      const check = append(row, "input");
      check.type = "checkbox";
      check.checked = state.selected.has(r.run_id);
      check.setAttribute("aria-label", `Compare ${r.run_id}`);
      check.onchange = () => {
        check.checked
          ? state.selected.add(r.run_id)
          : state.selected.delete(r.run_id);
        updateSelected();
        renderRuns();
      };
      const name = append(row, "div");
      const link = append(name, "a", model(r), "run-name");
      link.href = `/runs/${encodeURIComponent(r.run_id)}`;
      link.onclick = (event) => {
        event.preventDefault();
        showDetail(r.run_id);
      };
      append(name, "span", r.run_id, "run-id");
      const art = append(row, "div", "", "run-extra");
      append(art, "span", "Artifact / engine", "run-col-label");
      append(art, "span", `${artifact(r)} · ${engine(r)}`, "run-col-value");
      const hard = append(row, "div", "", "run-extra hide-medium");
      append(hard, "span", "Hardware", "run-col-label");
      append(hard, "span", hardware(r), "run-col-value");
      const status = append(row, "div", "", "run-extra");
      append(status, "span", "Status", "run-col-label");
      status.append(badge(r.status));
      append(status, "span", profile(r), "run-sub");
      const open = append(row, "button", "›", "row-arrow");
      open.type = "button";
      open.setAttribute("aria-label", `Inspect ${r.run_id}`);
      open.onclick = () => showDetail(r.run_id);
      renderRunSettings(row, r);
    }
    for (const e of state.report.errors || [])
      append($("runs"), "p", `${e.run_id}: ${e.error}`, "inline-error");
  }
  async function showDetail(id, navigate = true) {
    state.detail = id;
    if (navigate) {
      chooseView("runs");
      history.replaceState(null, "", `/runs/${encodeURIComponent(id)}#runs`);
    }
    let r = state.report.runs.find((item) => item.run_id === id);
    if (!r) {
      try {
        r = await requestJson(
          `/api/runs/${encodeURIComponent(id)}?compact=true`,
          { cache: "no-store" },
          "Run request",
        );
      } catch (_) {
        /* Explicit missing state below. */
      }
    }
    if (state.detail !== id) return;
    const root = $("run-detail");
    root.replaceChildren();
    root.hidden = false;
    if (!r) {
      empty(root, "Run unavailable", "The retained run could not be loaded.");
      return;
    }
    const panel = append(root, "div", "", "panel");
    const head = append(panel, "div", "", "detail-header");
    append(head, "h2", model(r));
    head.append(badge(r.status));
    const back = append(head, "button", "Close detail", "button subtle");
    back.type = "button";
    back.onclick = () => {
      state.detail = null;
      root.hidden = true;
      history.replaceState(null, "", "/reports#runs");
    };
    append(
      panel,
      "p",
      `${r.run_id} · ${profile(r)} · ${str(r.evidence)}`,
      "outcome-note",
    );
    const summary = append(panel, "div", "", "metric-grid");
    stat(summary, "Run status", str(r.status));
    stat(
      summary,
      "Wall duration",
      r.wall_seconds == null ? "Unavailable" : duration(r.wall_seconds * 1000),
    );
    stat(
      summary,
      "Coding accuracy",
      pct(r.coding?.accuracy),
      coverageLabel(r.coverage?.coding),
    );
    stat(summary, "RULER scored", rulerScored(r), "Sampled diagnostic subset");
    renderPhases(panel, r);
    renderCoverage(panel, r);
    renderQuality(panel, [r]);
    renderSingleCurves(panel, r);
    renderOutcomes(panel, r);
    renderMetadata(panel, r);
    raw(panel, "Retained summary and setup", r);
    if (navigate) root.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function coverageLabel(c) {
    return c
      ? `${fmt(c.accounted)} / ${fmt(c.expected)} accounted${c.graded == null ? "" : ` · ${fmt(c.graded)} graded`}`
      : "Coverage unavailable";
  }
  function rulerScored(r) {
    const vals = Object.values(r.ruler?.quality_by_context || {});
    if (!vals.length) return "Unavailable";
    return `${fmt(vals.reduce((s, x) => s + (x.scored || 0), 0))} scored`;
  }
  function renderPhases(parent, r) {
    const phases = r.phase_progress?.phases;
    if (!Array.isArray(phases) || !phases.length) return;
    const block = section(
      parent,
      "Phase progress",
      "Accounted work is retained; running state may be unverified after interruption.",
    );
    const list = append(block, "div", "", "phase-list");
    phases.forEach((p) => {
      const row = append(list, "div", "", "phase-row");
      append(
        row,
        "strong",
        String(p.name || p.phase || "Phase").replaceAll("_", " "),
      );
      row.append(badge(p.status));
      const expected =
        p.expected_units ?? p.total_units ?? p.expected ?? p.planned_units;
      const accounted = p.accounted_units ?? p.completed_units ?? p.accounted;
      const track = append(row, "div", "", "progress-track");
      const bar = append(track, "span");
      const ratio =
        Number.isFinite(expected) && expected > 0 && Number.isFinite(accounted)
          ? accounted / expected
          : 0;
      bar.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;
      append(
        row,
        "small",
        expected == null || accounted == null
          ? p.elapsed_seconds == null
            ? "Coverage unavailable"
            : duration(p.elapsed_seconds * 1000)
          : `${fmt(accounted)} / ${fmt(expected)}`,
      );
    });
  }
  function renderCoverage(parent, r) {
    const entries = Object.entries(r.coverage || {});
    if (!entries.length) return;
    const block = section(
      parent,
      "Coverage",
      "Missing, failed, and unsupported work remains explicit.",
    );
    table(
      block,
      ["Workload", "Accounted / expected", "Successful", "Graded"],
      entries.map(([key, val]) => [
        key.replaceAll("_", " "),
        coverageLabel(val),
        fmt(val.successful),
        fmt(val.graded),
      ]),
    );
  }
  function exportButton(parent, label, options) {
    const button = append(parent, "button", label, "button subtle export-png");
    button.type = "button";
    button.onclick = async () => {
      button.disabled = true;
      button.textContent = "Preparing PNG…";
      try {
        await exportPng(options());
      } catch (error) {
        $("status").textContent = `PNG export failed: ${error.message}`;
      } finally {
        button.disabled = false;
        button.textContent = label;
      }
    };
  }
  async function exportPng({ title, subtitle, runs, charts = [], quality }) {
    await document.fonts.ready;
    const canvas = document.createElement("canvas");
    const width = 1800,
      margin = 64;
    canvas.width = width;
    const ctx = canvas.getContext("2d");
    const commands = [];
    let y = margin;
    const text = (
      value,
      size = 24,
      color = "#f3f0eb",
      x = margin,
      maxWidth = width - margin * 2,
    ) => {
      const font = `${size >= 32 ? 650 : 400} ${size}px system-ui, sans-serif`;
      ctx.font = font;
      let line = "";
      // Wrap at characters too, so long model names and configuration values fit.
      for (const character of String(value)) {
        if (
          character === "\n" ||
          ctx.measureText(line + character).width > maxWidth
        ) {
          const drawn = line,
            top = y;
          commands.push(() => {
            ctx.font = font;
            ctx.fillStyle = color;
            ctx.fillText(drawn, x, top);
          });
          y += size * 1.45;
          line = character === "\n" ? "" : character;
        } else line += character;
      }
      const drawn = line,
        top = y;
      commands.push(() => {
        ctx.font = font;
        ctx.fillStyle = color;
        ctx.fillText(drawn, x, top);
      });
      y += size * 1.45;
    };
    text("real-bench  /  RETAINED RUN RESULTS", 20, "#ff9a52");
    text(title, 40);
    text(subtitle, 24, "#b7b4af");
    y += 16;
    runs.forEach((run, i) => {
      const top = y;
      commands.push(() => {
        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(margin, top + 3, 16, 16);
      });
      text(
        `${runLetter(i)} · ${model(run)}`,
        24,
        "#f3f0eb",
        margin + 30,
        width - margin * 2 - 30,
      );
      text(
        configurationSummary(run),
        22,
        "#d0ccc6",
        margin + 30,
        width - margin * 2 - 30,
      );
      text(
        `${hardware(run)} · ${profile(run)} · ${run.status || "status unavailable"} · ${short(run.run_id)}`,
        20,
        "#aaa8a4",
        margin + 30,
        width - margin * 2 - 30,
      );
      y += 16;
    });
    for (const card of charts) {
      text(card.querySelector("h3").textContent, 30);
      const caption = [...card.children].find((child) => child.tagName === "P");
      if (caption) text(caption.textContent, 22, "#b7b4af");
      const svgs = [...card.querySelectorAll("svg")];
      if (!svgs.length)
        text("No measured values for this scenario and output target.");
      for (const original of svgs) {
        const copy = original.cloneNode(true);
        const originals = [original, ...original.querySelectorAll("*")];
        const copies = [copy, ...copy.querySelectorAll("*")];
        originals.forEach((element, index) => {
          const style = getComputedStyle(element);
          for (const property of [
            "fill",
            "stroke",
            "stroke-width",
            "stroke-dasharray",
            "font-size",
            "font-family",
            "font-weight",
            "text-anchor",
          ])
            copies[index].style.setProperty(
              property,
              style.getPropertyValue(property),
            );
        });
        copy.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        const box = original.viewBox.baseVal;
        const drawWidth = width - margin * 2;
        const drawHeight = (drawWidth * box.height) / box.width;
        copy.style.minWidth = "0";
        copy.style.width = `${drawWidth}px`;
        copy.style.height = `${drawHeight}px`;
        copy.setAttribute("width", drawWidth);
        copy.setAttribute("height", drawHeight);
        // Export text stays legible independently of the current browser viewport.
        copy.querySelectorAll(".bar-value").forEach((label) => {
          const available = Number(label.getAttribute("data-bar-width"));
          const fontSize = Math.min(
            (16 * box.width) / drawWidth,
            available / (label.textContent.length * 0.65),
          );
          label.style.fontSize = `${fontSize}px`;
        });
        const url = URL.createObjectURL(
          new Blob([new XMLSerializer().serializeToString(copy)], {
            type: "image/svg+xml;charset=utf-8",
          }),
        );
        try {
          const img = new Image();
          img.src = url;
          await img.decode();
          const top = y;
          commands.push(() =>
            ctx.drawImage(img, margin, top, drawWidth, drawHeight),
          );
          y += drawHeight + 12;
        } finally {
          URL.revokeObjectURL(url);
        }
      }
      for (const note of card.querySelectorAll(".chart-note"))
        text(note.textContent, 20, "#aaa8a4");
      y += 28;
    }
    const scoreBar = (run, i, score, detail) => {
      const valid = Number.isFinite(score);
      text(
        `${runLetter(i)} · ${valid ? `${fmt(score)}${quality === "ruler" ? " / 100" : "%"}` : "Unavailable"} · ${detail}`,
        24,
      );
      const top = y,
        barWidth = width - margin * 2;
      commands.push(() => {
        ctx.fillStyle = "#292d32";
        ctx.fillRect(margin, top, barWidth, 18);
        if (valid) {
          ctx.fillStyle = colors[i % colors.length];
          ctx.fillRect(
            margin,
            top,
            (barWidth * Math.max(0, Math.min(100, score))) / 100,
            18,
          );
        }
      });
      y += 40;
    };
    if (quality === "coding") {
      text("HumanEval+ accuracy · 0–100% · higher is better", 28);
      runs.forEach((run, i) => {
        scoreBar(
          run,
          i,
          run.coding?.accuracy == null ? null : run.coding.accuracy * 100,
          coverageLabel(run.coverage?.coding),
        );
        const u = run.coding?.uncertainty;
        if (u)
          text(`${u.method}: ${pct(u.lower)}–${pct(u.upper)}`, 20, "#aaa8a4");
      });
      text(
        "Standalone coding quality. Missing grades are not counted as correct.",
        20,
        "#aaa8a4",
      );
    }
    if (quality === "ruler") {
      text("Sampled RULER · score 0–100 · higher is better", 28);
      const contexts = [
        ...new Set(
          runs.flatMap((run) =>
            Object.keys(run.ruler?.quality_by_context || {}),
          ),
        ),
      ].sort((a, b) => Number(a) - Number(b));
      if (!contexts.length) text("No scored context measurements available.");
      for (const context of contexts) {
        text(`${fmt(Number(context))} nominal input-token ceiling`, 28);
        runs.forEach((run, i) => {
          const result = run.ruler?.quality_by_context?.[context];
          scoreBar(
            run,
            i,
            result?.mean_score,
            result?.scored == null
              ? "Scored count unavailable"
              : `${result.scored} scored samples`,
          );
        });
      }
      text(
        "Diagnostic subset with limited precision; descriptive sample means, not a full official RULER score or ranking.",
        20,
        "#aaa8a4",
      );
    }
    y += 16;
    text(
      "Exported from retained measurements · missing results remain unavailable",
      18,
      "#aaa8a4",
    );
    canvas.height = Math.ceil(y + margin);
    ctx.fillStyle = "#111315";
    ctx.fillRect(0, 0, width, canvas.height);
    ctx.textBaseline = "top";
    commands.forEach((draw) => draw());
    const blob = await new Promise((resolve, reject) =>
      canvas.toBlob(
        (value) =>
          value ? resolve(value) : reject(new Error("Could not create image")),
        "image/png",
      ),
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `real-bench-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  function renderQuality(parent, runs) {
    const block = section(
      parent,
      "Quality",
      "Coding and sampled long-context results remain separate from speed.",
    );
    const exports = append(block, "div", "", "export-actions");
    for (const [quality, label] of [
      ["coding", "Coding"],
      ["ruler", "RULER"],
    ])
      exportButton(exports, `Export ${label} PNG`, () => ({
        title: `${label} ${runs.length > 1 ? "comparison" : "results"}`,
        subtitle: "Quality measurements · higher is better",
        runs,
        quality,
      }));
    const grid = append(block, "div", "", "quality-grid");
    for (const [index, r] of runs.entries()) {
      const coding = append(grid, "div", "", "quality-card");
      if (runs.length > 1) comparisonIdentity(coding, r, index);
      append(
        coding,
        "h3",
        runs.length > 1 ? "HumanEval+ · Coding" : `${model(r)} · HumanEval+`,
      );
      append(coding, "strong", pct(r.coding?.accuracy));
      append(
        coding,
        "p",
        `${coverageLabel(r.coverage?.coding)}. Standalone coding quality; unavailable grades are not counted as correct.`,
      );
      if (r.coding?.uncertainty)
        append(
          coding,
          "p",
          `${r.coding.uncertainty.method}: ${pct(r.coding.uncertainty.lower)}–${pct(r.coding.uncertainty.upper)}.`,
        );
      if (r.coding?.identity)
        raw(coding, "Coding identity and outcome coverage", {
          identity: r.coding.identity,
          outcome_counts: r.coding.outcome_counts,
          coverage: r.coverage?.coding,
        });
      const ruler = append(grid, "div", "", "quality-card");
      if (runs.length > 1) comparisonIdentity(ruler, r, index);
      append(
        ruler,
        "h3",
        runs.length > 1
          ? "Sampled RULER · Long context"
          : `${model(r)} · sampled RULER`,
      );
      append(ruler, "strong", rulerScored(r));
      append(
        ruler,
        "p",
        r.ruler?.limited_precision ||
          "Diagnostic subset with limited precision; score unavailable.",
      );
      const by = r.ruler?.quality_by_context || {};
      if (Object.keys(by).length) {
        const bars = append(ruler, "div", "", "quality-bars");
        const rulerScale = 100;
        for (const [k, v] of Object.entries(by).sort(
          (a, b) => Number(a[0]) - Number(b[0]),
        )) {
          const line = append(bars, "div", "", "quality-bar");
          append(line, "span", `${fmt(Number(k))} input tokens`);
          const track = append(line, "div", "", "progress-track");
          const fill = append(track, "span");
          if (typeof v.mean_score === "number" && Number.isFinite(v.mean_score))
            fill.style.width = `${Math.max(0, Math.min(100, (v.mean_score / rulerScale) * 100))}%`;
          append(
            line,
            "strong",
            v.mean_score == null
              ? "Unavailable"
              : `${fmt(v.mean_score)} / ${rulerScale}`,
          );
        }
        table(
          ruler,
          [
            "Nominal input ceiling",
            "Mean score",
            "Scored",
            "Actual input range",
          ],
          Object.entries(by).map(([k, v]) => [
            k,
            v.mean_score == null ? "Unavailable" : fmt(v.mean_score),
            fmt(v.scored),
            str(r.ruler?.actual_input_ranges?.[k]),
          ]),
        );
      }
    }
  }
  function renderOutcomes(parent, r) {
    const outcomes = r.outcomes || [];
    const block = section(
      parent,
      "Request outcomes",
      "Actual output lengths and failures remain visible per request.",
    );
    if (!outcomes.length) {
      empty(
        block,
        "No request outcomes",
        "No request result has been retained.",
      );
      return;
    }
    table(
      block,
      [
        "Phase",
        "Condition / task",
        "Outcome",
        "Input tokens",
        "Output target",
        "Actual output",
        "TTFT",
        "Decode",
        "Completion",
        "Effective prefill",
      ],
      outcomes.map((o) => [
        str(o.phase),
        str(
          o.condition ||
            o.task ||
            o.task_id ||
            o.problem_id ||
            o.task_type ||
            o.request_id ||
            "—",
        ),
        str(o.outcome),
        fmt(o.input_tokens),
        fmt(o.output_target),
        fmt(
          o.actual_output_tokens ??
            o.output_tokens ??
            o.completion_tokens ??
            o.generated_tokens,
        ),
        metricText(o.metrics?.ttft),
        metricText(o.metrics?.decode_rate),
        metricText(o.metrics?.completion_latency),
        metricText(o.metrics?.effective_endpoint_observed_prefill_throughput),
      ]),
    );
  }
  function renderMetadata(parent, r) {
    const block = section(
      parent,
      "Recorded setup",
      "Declared and observed facts keep their original provenance.",
    );
    const dl = append(block, "dl", "", "metadata-grid");
    const pairs = [
      ["Model source", model(r)],
      ["Quantized artifact", artifact(r)],
      ["Serving engine", engine(r)],
      ["Hardware", hardware(r)],
      ["Reasoning effort", reasoning(r)],
      ["Workload", profile(r)],
      ["Evidence", r.evidence],
      ["Run ID", r.run_id],
    ];
    pairs.forEach(([k, v]) => {
      const wrap = append(dl, "div");
      append(wrap, "dt", k);
      append(wrap, "dd", str(v));
    });
  }
  function curve(r, scenario) {
    const saved =
      r[
        scenario === "fresh-context"
          ? "fresh_context_curve"
          : "growing_session_curve"
      ];
    if (Array.isArray(saved) && saved.length) return saved;
    if (r.workload?.scenario !== scenario) return [];
    return (r.outcomes || [])
      .filter((o) => o.metrics && Object.keys(o.metrics).length)
      .map((o) => ({
        input_tokens: o.input_tokens ?? r.workload.input_tokens,
        output_target: o.output_target ?? r.workload.output_target,
        outcomes: { [o.outcome]: 1 },
        metrics: Object.fromEntries(
          Object.entries(o.metrics).map(([k, m]) => [
            k,
            { mean: m.value, minimum: m.value, maximum: m.value, unit: m.unit },
          ]),
        ),
      }));
  }
  function chart(parent, runs, scenario, metric, target) {
    const card = append(parent, "div", "", "chart-card");
    const title =
      scenario === "fresh-context" ? "Fresh context" : "Growing session";
    append(card, "h3", title);
    exportButton(card, "Export PNG", () => ({
      title: `${title} · ${metricInfo[metric][0]}`,
      subtitle: `Output target ${fmt(target)} tokens · ${["ttft", "completion_latency"].includes(metric) ? "Lower is faster" : "Higher is faster"}`,
      runs,
      charts: [card],
    }));
    if (runs.length > 1) {
      const legend = append(card, "div", "", "chart-config-legend");
      runs.forEach((r, i) => comparisonIdentity(legend, r, i));
    }
    const values = runs.flatMap((r, i) =>
      curve(r, scenario)
        .filter((p) => p.output_target === target)
        .map((p) => ({ r, i, p, m: p.metrics?.[metric] })),
    );
    const valid = values.filter(
      (v) =>
        Number.isFinite(v.p.input_tokens) &&
        Number.isFinite(v.m?.mean) &&
        v.m.unit,
    );
    const units = [...new Set(valid.map((v) => v.m.unit))];
    append(
      card,
      "p",
      `${metricInfo[metric][0]} · output target ${target || "unavailable"} tokens · ${units.map((unit) => axisScale(unit, Math.max(...valid.filter((v) => v.m.unit === unit).map((v) => v.m.mean))).unit).join(" / ") || "unit unavailable"}`,
    );
    if (!valid.length)
      append(
        card,
        "div",
        "No measured values for this scenario and output target.",
        "empty-state",
      );
    if (units.length > 1) {
      append(
        card,
        "p",
        "Mixed metric units; each unit is drawn separately below.",
        "inline-error",
      );
    }
    for (const unit of units)
      drawSvg(
        card,
        valid.filter((v) => v.m.unit === unit),
        values,
        runs,
        scenario,
        metric,
        target,
        unit,
      );
    append(card, "p", metricInfo[metric][1], "chart-note");
    if (runs.length > 1)
      append(
        card,
        "p",
        `Bars follow the legend order. Values above bars use the Y-axis unit. ${["ttft", "completion_latency"].includes(metric) ? "Lower is faster." : "Higher is faster."} N/A means the measurement is unavailable.`,
        "chart-note",
      );
  }
  function drawSvg(
    parent,
    points,
    allPoints,
    runs,
    scenario,
    metric,
    target,
    unit,
  ) {
    const contexts = [...new Set(allPoints.map((v) => v.p.input_tokens))]
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const grouped = runs.length > 1;
    const right = grouped ? 780 : 520;
    const maxY = Math.max(1, ...points.map((v) => v.m.mean));
    const scale = axisScale(unit, maxY);
    const roughStep = Math.max(1, maxY / scale.divisor / 4);
    const magnitude = 10 ** Math.floor(Math.log10(roughStep));
    const step =
      [1, 2, 5, 10].find((n) => n * magnitude >= roughStep) * magnitude;
    const tickCount = Math.ceil(maxY / scale.divisor / step);
    const axisMaximum = tickCount * step * scale.divisor;
    const left = grouped
      ? Math.max(24, fmt(tickCount * step).length * 7 + 10)
      : 85;
    const pitch = (right - left) / contexts.length;
    const xPosition = (context) =>
      grouped
        ? left + pitch * (contexts.indexOf(context) + 0.5)
        : contexts.length === 1
          ? (left + right) / 2
          : left +
            ((right - left) * contexts.indexOf(context)) /
              (contexts.length - 1);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${right + 20} 280`);
    svg.setAttribute("role", "img");
    svg.setAttribute(
      "aria-label",
      `${scenario} ${metricInfo[metric][0]} in ${scale.unit}, output target ${target}, ${grouped ? "grouped bars" : "curve"} for ${runs.length} runs`,
    );
    parent.append(svg);
    const shape = (tag, attrs, txt) => {
      const n = document.createElementNS(ns, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, String(v)));
      if (txt != null) n.textContent = txt;
      svg.append(n);
      return n;
    };
    for (let i = 0; i <= tickCount; i++) {
      const y = 215 - (i / tickCount) * 180;
      shape("line", {
        x1: left,
        y1: y,
        x2: right,
        y2: y,
        class: "chart-gridline",
      });
      shape(
        "text",
        {
          x: left - 7,
          y: y + 3,
          "text-anchor": "end",
          class: "chart-label chart-y-tick",
        },
        fmt(i * step),
      );
    }
    shape("line", { x1: left, y1: 35, x2: left, y2: 215, class: "chart-axis" });
    shape("line", {
      x1: left,
      y1: 215,
      x2: right,
      y2: 215,
      class: "chart-axis",
    });
    contexts.forEach((context, index) => {
      const x = xPosition(context);
      shape("line", { x1: x, y1: 215, x2: x, y2: 221, class: "chart-axis" });
      shape(
        "text",
        {
          x,
          y: 239,
          "text-anchor":
            grouped || contexts.length === 1
              ? "middle"
              : index === 0
                ? "start"
                : index === contexts.length - 1
                  ? "end"
                  : "middle",
          class: "chart-label chart-x-tick",
        },
        fmt(context),
      );
    });
    shape(
      "text",
      {
        x: (left + right) / 2,
        y: 263,
        "text-anchor": "middle",
        class: "chart-label",
      },
      "Context size · input tokens",
    );
    shape("text", { x: left, y: 18, class: "chart-label" }, scale.unit);
    if (grouped) {
      const gap = 4;
      const width = Math.min(
        36,
        (pitch * 0.9 - gap * (runs.length - 1)) / runs.length,
      );
      contexts.forEach((context) => {
        const groupWidth = width * runs.length + gap * (runs.length - 1);
        runs.forEach((r, i) => {
          const m = allPoints.find(
            (v) => v.r.run_id === r.run_id && v.p.input_tokens === context,
          )?.m;
          const x = xPosition(context) - groupWidth / 2 + i * (width + gap);
          const valid = Number.isFinite(m?.mean) && m.unit === unit;
          const y = valid ? 215 - (180 * m.mean) / axisMaximum : 215;
          if (valid) {
            const bar = shape("rect", {
              x,
              y,
              width,
              height: 215 - y,
              fill: colors[i % colors.length],
              class: "comparison-bar",
              "data-run": r.run_id,
              "data-context": context,
            });
            const title = document.createElementNS(ns, "title");
            title.textContent = `${runLetter(i)} · ${configurationSummary(r)} · ${fmt(context)} input tokens · ${measurement(m.mean, unit)}`;
            bar.append(title);
            if (m.mean === 0)
              shape("line", {
                x1: x,
                x2: x + width,
                y1: 215,
                y2: 215,
                stroke: colors[i % colors.length],
                "stroke-width": 3,
              });
          }
          shape(
            "text",
            {
              x: x + width / 2,
              y: y - 9,
              "data-bar-top": y,
              "data-bar-width": width,
              "text-anchor": "middle",
              class: "chart-label bar-value",
              "data-run": r.run_id,
              "data-context": context,
            },
            valid
              ? new Intl.NumberFormat(undefined, {
                  maximumFractionDigits: 3,
                }).format(m.mean / scale.divisor)
              : "N/A",
          );
        });
      });
      // Keep values small in screen pixels and directly above their bars.
      // Narrow screens scroll the chart rather than shrinking labels to illegibility.
      const layoutLabels = () => {
        if (!svg.isConnected || !svg.getBoundingClientRect().width) return;
        const screenScale = svg.getBoundingClientRect().width / (right + 20);
        let minimumWidth = 640;
        let top = 0;
        svg.querySelectorAll(".bar-value").forEach((label) => {
          label.style.fontSize = `${10 / screenScale}px`;
          const barTop = Number(label.getAttribute("data-bar-top"));
          const barWidth = Number(label.getAttribute("data-bar-width"));
          label.setAttribute("y", barTop - 3 / screenScale);
          const box = label.getBBox();
          minimumWidth = Math.max(
            minimumWidth,
            ((box.width * screenScale + 4) * (right + 20)) / barWidth,
          );
          top = Math.min(top, box.y - 4 / screenScale);
        });
        svg.style.minWidth = `${Math.ceil(minimumWidth)}px`;
        svg.setAttribute("viewBox", `0 ${top} ${right + 20} ${280 - top}`);
        svg.setAttribute("data-labels-laid-out", "true");
      };
      const resize = new ResizeObserver(() => {
        if (!svg.isConnected) resize.disconnect();
        else layoutLabels();
      });
      resize.observe(svg);
      layoutLabels();
      return;
    }
    runs.forEach((r, i) => {
      const own = allPoints
        .filter((v) => v.r.run_id === r.run_id)
        .sort((a, b) => a.p.input_tokens - b.p.input_tokens);
      let prev = null;
      for (const v of own) {
        if (
          !Number.isFinite(v.p.input_tokens) ||
          !Number.isFinite(v.m?.mean) ||
          v.m.unit !== unit
        ) {
          prev = null;
          continue;
        }
        const contextIndex = contexts.indexOf(v.p.input_tokens);
        const x = xPosition(v.p.input_tokens),
          y = 215 - (180 * v.m.mean) / axisMaximum;
        if (prev && contextIndex === prev[2] + 1)
          shape("line", {
            x1: prev[0],
            y1: prev[1],
            x2: x,
            y2: y,
            stroke: colors[i % colors.length],
            "stroke-width": 2.5,
            "stroke-dasharray": ["none", "8 6", "2 5", "12 5 2 5"][i % 4],
          });
        const c = shape(
          i % 2 ? "rect" : "circle",
          i % 2
            ? {
                x: x - 6,
                y: y - 6,
                width: 12,
                height: 12,
                fill: "none",
                stroke: colors[i % colors.length],
                "stroke-width": 2,
              }
            : {
                cx: x,
                cy: y,
                r: 4,
                fill: colors[i % colors.length],
              },
        );
        const t = document.createElementNS(ns, "title");
        t.textContent = `${model(r)} · ${short(r.run_id)} · ${fmt(v.p.input_tokens)} input tokens · ${measurement(v.m.mean, unit)}`;
        c.append(t);
        prev = [x, y, contextIndex];
      }
    });
  }
  function renderSingleCurves(parent, r) {
    const block = section(
      parent,
      "Performance curves",
      "Speed input checkpoints are exact targets; output targets are requested lengths, not guaranteed completions.",
    );
    const targets = [
      ...new Set(
        ["fresh-context", "growing-session"].flatMap((s) =>
          curve(r, s).map((p) => p.output_target),
        ),
      ),
    ]
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    if (!targets.length) {
      empty(
        block,
        "No performance curve",
        "No measured speed condition is retained for this run.",
      );
      return;
    }
    const controls = append(block, "div", "", "compare-controls");
    const metricLabel = append(controls, "label", "Metric");
    const metricSelect = append(metricLabel, "select");
    Object.entries(metricInfo).forEach(([key, val]) =>
      option(metricSelect, key, val[0]),
    );
    const targetLabel = append(controls, "label", "Output target");
    const targetSelect = append(targetLabel, "select");
    targets.forEach((t) => option(targetSelect, String(t), `${fmt(t)} tokens`));
    const grid = append(block, "div", "", "chart-grid");
    const draw = () => {
      grid.replaceChildren();
      chart(
        grid,
        [r],
        "fresh-context",
        metricSelect.value,
        Number(targetSelect.value),
      );
      chart(
        grid,
        [r],
        "growing-session",
        metricSelect.value,
        Number(targetSelect.value),
      );
    };
    metricSelect.onchange = draw;
    targetSelect.onchange = draw;
    draw();
  }
  function updateSelected() {
    $("selection-count").textContent = state.selected.size;
    $("compare-total").textContent = `${state.selected.size} selected`;
    renderComparison();
  }
  function renderComparison() {
    const runs = state.report.runs.filter((r) => state.selected.has(r.run_id));
    $("comparison").replaceChildren();
    $("parameters").replaceChildren();
    $("compare-empty").hidden = !!runs.length;
    const targetEl = $("target"),
      prior = targetEl.value;
    targetEl.replaceChildren();
    const targets = [
      ...new Set(
        runs.flatMap((r) =>
          ["fresh-context", "growing-session"].flatMap((s) =>
            curve(r, s).map((p) => p.output_target),
          ),
        ),
      ),
    ]
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    targets.forEach((t) => option(targetEl, String(t), `${fmt(t)} tokens`));
    if (targets.some((t) => String(t) === prior)) targetEl.value = prior;
    if (!runs.length) return;
    const heads = append($("comparison"), "div", "", "compare-run-heads");
    runs.forEach((r, i) => {
      const h = append(heads, "div", "", "compare-run-head");
      comparisonIdentity(h, r, i);
      h.append(badge(r.status));
      append(
        h,
        "div",
        `${model(r)} · ${hardware(r)} · ${profile(r)}`,
        "comparison-model",
      );
      renderRunSettings(h, r);
      const inspect = append(
        h,
        "a",
        `Inspect ${runLetter(i)} · ${short(r.run_id)}`,
        "run-link",
      );
      inspect.href = `/runs/${r.run_id}`;
    });
    const metric = $("metric").value,
      target = Number(targetEl.value);
    const block = section(
      $("comparison"),
      "Performance",
      "Grouped bars start at zero within each scenario and metric unit. Missing measurements remain gaps.",
    );
    const grid = append(block, "div", "", "chart-grid");
    chart(grid, runs, "fresh-context", metric, target);
    chart(grid, runs, "growing-session", metric, target);
    exportButton(
      block.querySelector(".section-heading"),
      "Export performance PNG",
      () => ({
        title: `Performance · ${metricInfo[metric][0]}`,
        subtitle: `Output target ${fmt(target)} tokens`,
        runs,
        charts: [...grid.querySelectorAll(".chart-card")],
      }),
    );

    renderQuality($("comparison"), runs);
    const keys = [
      ...new Set(runs.flatMap((r) => Object.keys(r.parameters || {}))),
    ].sort();
    if (keys.length) {
      const t = table(
        $("parameters"),
        [
          "Parameter",
          ...runs.map((r, i) => `${runLetter(i)} · ${artifact(r)}`),
        ],
        keys.map((k) => [
          k,
          ...runs.map((r) =>
            k in (r.parameters || {}) ? str(r.parameters[k]) : "Unavailable",
          ),
        ]),
      );
      [...t.tBodies[0].rows].forEach((row, i) => {
        const vals = runs.map((r) => kValue(r, keys[i]));
        if (new Set(vals).size > 1) {
          row.classList.add("differs");
          row.cells[0].textContent += " · differs";
        }
      });
    }
  }
  const requestJson = async (url, options = {}, label = "Request") => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      if (!response.ok) throw Error(`${label} failed (${response.status})`);
      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  };
  const kValue = (r, k) =>
    k in (r.parameters || {}) ? JSON.stringify(r.parameters[k]) : "<missing>";
  async function loadReport(rebuild = false) {
    if (state.reportBusy) return;
    state.reportBusy = true;
    $("status").textContent = "Loading runs…";
    try {
      const data = await requestJson(
        rebuild
          ? "/api/reports/rebuild?compact=true"
          : "/api/reports?compact=true",
        { method: rebuild ? "POST" : "GET", cache: "no-store" },
        "Report request",
      );
      if (!Array.isArray(data.runs)) throw Error("Invalid report response");
      state.report = data;
      state.selected = new Set(
        [...state.selected].filter((id) =>
          data.runs.some((r) => r.run_id === id),
        ),
      );
      fillFacets();
      renderRuns();
      updateSelected();
      if (state.detail && state.view === "runs")
        showDetail(state.detail, false);
      $("status").textContent =
        `${data.runs.length} retained run${data.runs.length === 1 ? "" : "s"}`;
      if (data.errors?.length)
        $("status").textContent += ` · ${data.errors.length} unreadable`;
    } catch (error) {
      $("status").textContent = String(error);
      if (!state.report.runs.length) {
        $("runs").replaceChildren();
        empty(
          $("runs"),
          "Unable to load runs",
          "Use Rebuild index to try again. Previous selections will be retained.",
        );
      }
    } finally {
      state.reportBusy = false;
    }
  }
  function liveRows(r) {
    return Array.isArray(r.observed_outcomes)
      ? r.observed_outcomes
      : Array.isArray(r.outcomes)
        ? r.outcomes
        : [];
  }
  function liveMetric(rows, key) {
    const values = rows.flatMap((o) => {
      const m = o.metrics?.[key];
      return typeof m?.value === "number" && Number.isFinite(m.value)
        ? [{ value: m.value, unit: m.unit }]
        : [];
    });
    if (!values.length) return null;
    const unit = values[0].unit;
    if (values.some((v) => v.unit !== unit)) return null;
    return {
      value: values.reduce((a, b) => a + b.value, 0) / values.length,
      unit,
      count: values.length,
    };
  }
  function metricText(m) {
    return m ? measurement(m.value, m.unit) : "Unavailable";
  }
  function liveActive(r) {
    return r.lifecycle?.state === "active";
  }
  function renderLive() {
    const root = $("live-content"),
      banner = $("live-banner");
    root.replaceChildren();
    banner.textContent = "";
    banner.className = "notice";
    const data = state.live,
      all = data?.runs || [];
    const active = all.filter(liveActive);
    const candidates = active.length ? active : all;
    $("live-connection").textContent = state.liveFailure
      ? "Update failed"
      : state.lastLiveOk
        ? "Connected"
        : "Connecting";
    $("live-connection").className =
      `connection${state.liveFailure ? " error" : ""}`;
    if (state.liveFailure) {
      banner.textContent = state.lastLiveOk
        ? `Latest update failed. Showing retained snapshot from ${state.lastLiveOk.toLocaleTimeString()}.`
        : `Live data unavailable. Retry with Refresh now.`;
      banner.classList.add("warning");
    }
    if (!data) {
      empty(
        root,
        "Live data unavailable",
        "Refresh to load retained execution data.",
      );
      return;
    }
    if (data.errors?.length) {
      banner.textContent += `${banner.textContent ? " " : ""}${data.errors.length} retained run director${data.errors.length === 1 ? "y is" : "ies are"} unreadable.`;
      banner.classList.add("warning");
    }
    if (!active.length) {
      if (!state.liveFailure)
        banner.textContent += `${banner.textContent ? " " : ""}No active run is verified. Retained runs are available in the library.`;
      const go = append(root, "button", "Open run library", "button");
      go.type = "button";
      go.onclick = () => chooseView("runs");
    }
    if (!candidates.length) {
      empty(
        root,
        "No runs yet",
        "The live monitor will show a run once its first evidence is retained.",
      );
      return;
    }
    const selectWrap = append(root, "div", "", "live-selector");
    const lab = append(selectWrap, "label", "Run");
    const select = append(lab, "select");
    candidates.forEach((r) =>
      option(
        select,
        r.run_id,
        `${model(r)} · ${short(r.run_id)} · ${r.lifecycle?.state || r.status}`,
      ),
    );
    if (!candidates.some((r) => r.run_id === state.liveRun))
      state.liveRun = candidates[0].run_id;
    select.value = state.liveRun;
    select.onchange = () => {
      state.liveRun = select.value;
      renderLive();
    };
    const r =
      candidates.find((x) => x.run_id === state.liveRun) || candidates[0];
    const meta = append(root, "div", "", "live-meta");
    meta.append(badge(r.lifecycle?.state || r.status));
    append(meta, "span", r.run_id);
    append(meta, "span", `${artifact(r)} · ${engine(r)} · ${hardware(r)}`);
    append(
      meta,
      "span",
      `Current phase: ${str(r.phase_progress?.current || (r.lifecycle?.state === "active" ? profile(r) : "none"))}`,
    );
    if (data.generated_at)
      append(
        meta,
        "span",
        `Snapshot ${new Date(data.generated_at).toLocaleString()}`,
      );
    if (r.lifecycle?.state === "unverified") {
      const note = append(
        root,
        "div",
        "Recorded progress is unverified; the process may no longer be active.",
        "notice warning",
      );
      note.setAttribute("role", "status");
    }
    const rows = liveRows(r),
      speedRows = rows.filter(
        (o) =>
          o.phase === "fresh_context" ||
          o.phase === "growing_session" ||
          o.metrics?.ttft,
      );
    const total = rows.length;
    const metrics = append(root, "div", "", "metric-grid");
    stat(metrics, "Observed outcomes", fmt(total), "Retained requests", true);
    stat(
      metrics,
      "TTFT",
      metricText(liveMetric(speedRows, "ttft")),
      "Mean of available request values",
    );
    stat(
      metrics,
      "Decode rate",
      metricText(liveMetric(speedRows, "decode_rate")),
      "Mean of available request values",
    );
    stat(
      metrics,
      "Completion latency",
      metricText(liveMetric(speedRows, "completion_latency")),
      "End-to-end request latency",
    );
    const more = append(root, "div", "", "metric-grid");
    stat(
      more,
      "Effective observed prefill",
      metricText(
        liveMetric(speedRows, "effective_endpoint_observed_prefill_throughput"),
      ),
      "Input tokens / TTFT; endpoint observed",
    );
    stat(
      more,
      "Coding accuracy",
      pct(r.coding?.accuracy),
      coverageLabel(r.coverage?.coding),
    );
    stat(more, "RULER scored", rulerScored(r), "Sampled diagnostic subset");
    stat(
      more,
      "Run status",
      str(r.status),
      r.lifecycle?.state === "active" ? "Execution active" : "Retained summary",
    );
    renderPhases(root, r);
    renderCoverage(root, r);
    const perf = section(
      root,
      "Observed performance",
      "Only request metrics retained so far; no values are inferred for missing conditions.",
    );
    const conditions = rows.filter(
      (o) =>
        o.metrics &&
        Object.keys(o.metrics).length &&
        ((o.phase || "").includes("context") ||
          (o.phase || "").includes("session") ||
          r.workload?.scenario),
    );
    if (conditions.length)
      table(
        perf,
        [
          "Scenario",
          "Input tokens",
          "Output target",
          "Outcome",
          "TTFT",
          "Decode",
          "Completion",
          "Effective prefill",
        ],
        conditions
          .slice(-30)
          .map((o) => [
            str(o.phase || r.workload?.scenario),
            fmt(o.input_tokens ?? r.workload?.input_tokens),
            fmt(o.output_target ?? r.workload?.output_target),
            str(o.outcome),
            metricText(o.metrics?.ttft),
            metricText(o.metrics?.decode_rate),
            metricText(o.metrics?.completion_latency),
            metricText(
              o.metrics?.effective_endpoint_observed_prefill_throughput,
            ),
          ]),
      );
    else
      empty(
        perf,
        "No speed measurements yet",
        "Awaiting retained request measurements.",
      );
    const link = append(root, "a", "Open retained run detail →", "run-link");
    link.href = `/runs/${encodeURIComponent(r.run_id)}`;
    link.onclick = (e) => {
      e.preventDefault();
      showDetail(r.run_id);
    };
  }
  async function loadLive() {
    if (state.liveBusy) return;
    state.liveBusy = true;
    try {
      const data = await requestJson(
        "/api/dashboard",
        { cache: "no-store" },
        "Live request",
      );
      if (!Array.isArray(data.runs)) throw Error("Invalid live response");
      state.live = data;
      state.lastLiveOk = new Date();
      state.liveFailure = false;
    } catch (error) {
      state.liveFailure = true;
    } finally {
      state.liveBusy = false;
      renderLive();
    }
  }
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => (b.onclick = () => chooseView(b.dataset.view)));
  $("search").oninput = (e) => {
    state.query = e.target.value;
    renderRuns();
  };
  $("field").onchange = fillValues;
  $("sort").onchange = renderRuns;
  $("descending").onchange = renderRuns;
  $("add-filter").onclick = () => {
    if ($("field").value && $("value").value) {
      const f = [$("field").value, $("value").value];
      if (!state.filters.some((v) => v[0] === f[0] && v[1] === f[1]))
        state.filters.push(f);
      renderRuns();
    }
  };
  $("clear").onclick = () => {
    state.filters = [];
    renderRuns();
  };
  $("metric").onchange = renderComparison;
  $("target").onchange = renderComparison;
  $("rebuild").onclick = () => loadReport(true);
  $("live-refresh").onclick = loadLive;
  window.addEventListener("hashchange", () =>
    chooseView(location.hash.slice(1)),
  );
  const match = location.pathname.match(/^\/runs\/([0-9a-f]{32})$/);
  if (match) state.detail = match[1];
  chooseView(location.hash.slice(1) || "runs");
  loadReport();
  state.timer = setInterval(() => {
    if (state.view === "live" && document.visibilityState === "visible")
      loadLive();
  }, 10000);
})();
