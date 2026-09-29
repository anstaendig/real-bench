# real-bench

real-bench benchmarks a remote, already-running model endpoint using existing
tools, and keeps speed, coding quality, and long-context quality as separate
results. The default suite runs four workloads in one launch:

| Workload | What runs | Coverage |
|---|---|---:|
| `fresh-context-speed` | Independent requests across the declared input/output conditions (AIPerf), plus warm-ups. Measures latency and generation speed as context grows. | 60 |
| `growing-session-speed` | Multi-turn sessions that accumulate conversation and tool output (AIPerf), plus warm-ups. Measures prefix reuse and session growth. | 60 |
| `humaneval-plus` | One generated solution per HumanEval+ problem, graded in the pinned EvalPlus Docker container. | 164 |
| `sampled-ruler` | Sampled RULER requests: all 13 task types, three samples each, at 32K, 128K, and 192K nominal input. A diagnostic sample, not a full official RULER score. | 117 |

Warm-ups are excluded from coverage. Quality workloads use a fixed 65,536-token
output budget. Individual workload submissions remain available when the full
suite is not appropriate.

The methodology lives in the [specification](docs/SPEC.md); see
[docs/](docs/README.md) for the index. For workload objects, per-launch
configuration, preparation and launch gates, and the HTTP API, see the
[run interface](docs/run-interface.md).

Results are local. Run directories, setup state, and reports are written under
`results/` and `.real-bench/` and are git-ignored; nothing is uploaded anywhere.

**Platforms:** Linux and macOS. On Apple Silicon, HumanEval+ grading runs the
`linux/amd64` image under emulation and is slower.

## Quick start

```sh
git clone https://github.com/anstaendig/real-bench && cd real-bench
uv sync
uv tool install --editable .           # puts `bench` on your PATH (once)
bench setup                     # 1. describe the endpoint, prepare resources
bench run <submission> ...      # 2. launch (full command printed by setup)
bench serve --results results   # 3. open http://127.0.0.1:8000/
```

## Prerequisites

- **Python 3.12+ and `uv`** on the machine running real-bench. Install the CLI
  once with `uv tool install --editable .` so `bench` runs without `uv run`. Keep
  it editable and keep the checkout: `bench` reads helper scripts and pinned
  sources from it, and reports a clear error when run from a plain wheel install.
  Without the install, prefix every command with `uv run`.
- **A running endpoint** with streaming OpenAI-compatible chat completions at
  `/v1/chat/completions`, and its served model ID. real-bench does not start or
  configure the model server; it uses endpoint-reported token counts.
- **The matching tokenizer and chat template** (tokenizer files only, no weights),
  including any template override used by the server.
- **Docker** with `linux/amd64` support, for HumanEval+ grading.
- **Network access** on first use for tool dependencies and pinned resources, plus
  disk space for prompts, responses, native tool output, and results. AIPerf 0.12.0
  with Transformers 5.17.0 is invoked through `uv` automatically.

For an authenticated endpoint, set the bearer key in `REAL_BENCH_API_KEY` before
launching. It is used for validation and every workload and never persisted: setup
records, submissions, run evidence, native exports, and command records are
scrubbed of it. Do not put credentials in the URL or setup record.

## Setup

```sh
bench setup
```

Setup interactively records the instance details (endpoint, served model ID,
context window, tokenizer/template, reproduction metadata). Answers save after
each prompt, so an interrupted session resumes without re-asking. Descriptive
hardware, model, and serving metadata may be `unknown`; execution-critical details
are required and reported by name, never guessed.

- `--discover` reads the endpoint's model ID, serving version, and context window
  with read-only `GET` requests and presents them for review.
- For scripted or agent-driven use, apply answers without prompts:

  ```sh
  bench setup --answers '{"requests.base_url": "http://host:8080", "model.tokenizer": "/path"}' --no-interactive
  ```

Once the record is complete, setup prepares reusable resources (and reuses them on
later runs): the pinned performance tools, the tokenizer with a sha256 manifest,
the HumanEval+ v0.1.10 dataset and EvalPlus container, and the pinned RULER source
with hydrated data. It explains what it downloads, resumes if interrupted, and
names any human-only step, such as a gated tokenizer's `HF_TOKEN` or a missing
Docker daemon.

Setup ends with a prepared submission for each workload and for the complete
default suite, a configuration summary, and the exact launch commands. It does not
start the benchmark.

## Run

Launch the command setup printed, supplying the per-launch choices:

```sh
bench run .real-bench/setup/submissions/default-suite.json \
  --setup-directory .real-bench/setup \
  --run-config '{"reasoning_effort": "high", "temperature": 0.5, "top_p": null, "top_k": null, "min_p": null, "presence_penalty": null, "repetition_penalty": null}'
```

- `reasoning_effort` is a required explicit choice (never inherited); the six
  sampling parameters record explicit no-override as `null`.
- Without `--run-config`, an interactive terminal prompts for these, offering the
  latest run's sampling values as recommendations only. With `--no-interactive`
  the runner names the missing choices; the API returns `422`. See
  [per-launch configuration](docs/run-interface.md#per-launch-configuration-141).
- Launch first revalidates the configuration-dependent preparation and endpoint,
  and is refused before any work if prepared resources no longer verify.
- The four suite phases run in order; individual failures are retained and later
  phases continue, except that cancellation or endpoint unreachability stops
  further endpoint work.
- Progress is reported per phase with measured request, phase, and run durations.
- Results are written to `results/<run_id>/`.

To identify or cancel a run from another terminal, generate a 32-character
lowercase hex run ID first and pass it with `--run-id` (omit it to let the runner
generate one):

```sh
run_id=$(uv run python -c 'import uuid; print(uuid.uuid4().hex)')
bench run <submission> --run-id "$run_id" ...
```

### Cancel

```sh
bench cancel YOUR_RUN_ID --results results
```

Finished phase work is retained, the interrupted phase is reported partial, and
later phases are skipped. A new submission starts a new run; runs are not resumed.

## Dashboard

The dashboard is a local web app served from the results directory. It shows live
progress, the run catalog, and visual comparisons.

```sh
bench serve --results results
```

Then open <http://127.0.0.1:8000/>, or <http://127.0.0.1:8000/runs/YOUR_RUN_ID>
for a single run.

- Start it in a second terminal at any time, before, during, or after a run. Use
  the same `--results` directory that `bench run` writes to.
- Options: `--port` (default `8000`) and `--host` (default `127.0.0.1`, local
  only).
- **Live monitor:** polls every ten seconds while the page is visible. Metrics
  appear once the benchmark tool exports them; in-flight requests show no invented
  timing. A failed refresh keeps the previous snapshot with a stale-data notice.
  Runs started by current versions expose activity through an OS lock held by the
  runner; older partial runs show unverified activity rather than being assumed
  running.
- **Run details:** speed curves, quality coverage, individual outcomes, and
  reproduction settings. Partial runs are inspectable alongside completed ones.
- **Comparison:** select runs in the catalog to compare context curves and recorded
  parameters. Scenarios and output targets stay separate.
- Outcome rows omit prompt and response transcripts to keep responses compact; full
  transcripts stay in the run directory and the ordinary result API.

## Results

```sh
bench result YOUR_RUN_ID --results results
```

Each run retains `setup.json`, `workload.json`, `outcomes.json`, and
`summary.json`, plus workload-specific inputs, native outputs, commands, and
execution logs. Default-suite phases have their own directories under `phases/`.
Rebuild the report index from these files with
`bench rebuild-index --results results`; historical result directories
remain readable.

Interpreting results:

- Check completion status, coverage, and per-request outcomes first. Missing
  measurements are not zero scores, and a partial run is still evidence for the
  work that completed.
- TTFT, decode throughput, effective endpoint-observed prefill throughput (prompt
  tokens per second), and completion latency are reported separately.
- Coding and long-context quality are scored separately from speed, each with its
  own coverage and uncertainty limits. Sampled RULER scores are diagnostic, not a
  full official RULER score.

### HTTP API

The server exposes `GET /api/dashboard`, `GET /api/reports?compact=true`,
`POST /api/runs`, and `GET /api/runs/{run_id}`, with the schema at `/docs`. API
submissions execute synchronously, so client timeouts must cover the entire
workload and tool startup.

## Companion agent skill

The [real-bench companion skill](.agents/skills/real-bench/SKILL.md) guides the
same setup-to-run journey with the same `bench` commands. The commands own
preparation and validation; the skill only helps discover instance facts and the
user's per-launch choices.

## Security

`bench serve` binds to `127.0.0.1`, has no authentication, and rejects requests
with a non-loopback `Host` or a cross-site `Origin`. Binding elsewhere requires
`--allow-remote`; only do that on a trusted network. See [SECURITY.md](SECURITY.md)
for reporting vulnerabilities.

## Environment variables

| Variable | Purpose |
|---|---|
| `REAL_BENCH_API_KEY` | Bearer key for an authenticated endpoint; never persisted |
| `HF_TOKEN` | Access to a gated tokenizer during setup |

## Verification

```sh
make check                              # ruff, format check, mypy --strict
make test                               # backend tests
make web-e2e                            # dashboard browser tests (needs `npm --prefix web ci`)
```

`make verify` runs the full gate. The journey test drives the public
setup-to-run workflow (resumed setup, unknown metadata, prepared launch,
authentication, cancellation, endpoint outage, independent result reads) with
controlled tool output. A green gate proves deterministic software behavior, not
live model quality; see [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). real-bench is licensed under the
[Apache License 2.0](LICENSE); bundled and external third-party components are
listed in [NOTICE](NOTICE).
