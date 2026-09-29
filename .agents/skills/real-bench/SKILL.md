---
name: real-bench
description: "Run a real-bench benchmark: guide the guided setup of a remote model instance, obtain the user's explicit per-run choices, launch the prepared default suite, cancel it, and read its results. Use when the user wants to set up, launch, stop, or inspect a real-bench run."
---

Guide the user through a real-bench run using the same `bench` commands terminal
users run. The commands own preparation and validation; you help discover the
instance facts, obtain the user's explicit per-launch choices, and report what the
commands produced. Do not implement a separate benchmark workflow, duplicate the
preparation or validation logic, or hand-write submission JSON.

Work from the repository root. Prefix commands with `uv run`.

## 1. Setup

Run `uv run bench setup --no-interactive --answers '<json>'`, accumulating answers
across calls — the saved state resumes and keeps earlier answers. For each
execution-critical detail the user has not given, ask: the endpoint base URL, the
served model ID, the actual context window, the tokenizer/template location, and
the generation/reasoning settings. Descriptive hardware, model, and serving
metadata may be left `unknown`; record what the user actually knows. Offer
`--discover` (needs the endpoint reachable) to read the served model ID, serving
version, and context window read-only and confirm the findings with the user.

Keep going until the setup report shows `default_suite_ready: true`. If a step
needs a human-only action (a gated tokenizer's `HF_TOKEN`, a missing Docker
daemon), name it and stop there. Never persist secrets: the API key comes from
the `REAL_BENCH_API_KEY` environment variable, never from setup answers.

Setup ends without starting the benchmark. Take the prepared suite submission
path from the report's `prepared_submissions["default-suite"]`, the summary, and
the exact launch command from `launch_commands`.

## 2. Launch choices

For every launch, obtain two things from the user:

- **Reasoning effort**: ask the user at every launch. `off` is a valid explicit
  choice that disables thinking; never inherit or pre-fill it from saved setup
  or a previous run.
- **Sampling and penalties** (temperature, top_p, top_k, min_p,
  presence_penalty, repetition_penalty): present the most recent run's values as
  recommendations for acceptance or editing, and let the user confirm or change
  each. `null` records an explicit no-override. Never invent settings or fill
  them in silently.

Build `--run-config` from the user's answers. Confirm the full launch command with
the user before running it.

## 3. Launch

Run the command from step 1 with `--setup-directory` pointing at the setup
directory and the step 2 `--run-config`. Suggest generating a run ID first
(`uv run python -c 'import uuid; print(uuid.uuid4().hex)')` so the run can be
cancelled from another terminal, and pass it as `--run-id`. The launch
automatically finalizes or revalidates configuration-dependent preparation and
endpoint compatibility, then runs the four suite phases through the ordinary
lifecycle. If the launch is refused before work starts, report the reason the
command gave and have the user re-run setup or fix the configuration.

## 4. Cancel

To stop a running suite: `uv run bench cancel <run_id> --results <results_dir>`.
Finished phase work is retained, the interrupted phase is partial, and later
phases are skipped. Report what the retained result shows actually completed.

## 5. Results

Read without executing work:

- `uv run bench result <run_id> --results <results_dir>` for the JSON result,
- `uv run bench serve --results <results_dir>` for the local comparison page.

Report the run's status, per-phase progress, and coverage before any metrics.
Missing measurements are not zero scores, and a partial run is still evidence for
the work that completed. Coding and long-context quality are separate from both
speed curves; the long-context scores are diagnostic sampled coverage, not a full
official RULER score.

## Completion criteria

- Setup reached `default_suite_ready: true` (or the user stopped early), with the
  prepared suite submission saved and its launch command reported.
- The launch ran with the user's explicit reasoning choice and reviewed sampling,
  or the user stopped before it.
- The retained result was read and reported honestly, including partial phases and
  their reasons.
