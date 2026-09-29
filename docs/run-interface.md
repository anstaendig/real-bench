# Ordinary run interface

`bench run submission.json --results results` invokes pinned AIPerf 0.12.0 with
Transformers 5.17.0 for one streaming chat request, the complete fresh-context speed
curve, or the complete growing-session speed curve, all at concurrency one. No endpoint
process is launched or changed. The installed CLI and `bench serve` are the public
workflow. The workflow depends only on ordinary run directories and the
rebuildable report index.

A submission has `setup`, `workload`, and `run_config`. The readable setup separates
`declared` operator facts from `observed` facts (include a source/date with supplied
observations). Empty objects explicitly record no template override, environment,
generation, or reasoning override. Do not omit required categories. Hardware and serving
fields are descriptive, with no engine-specific admission or exhaustive weight hashes.

```json
{
  "setup": {
    "declared": {
      "hardware": {
        "accelerator": "operator-provided GPU model", "count": 1,
        "memory": "32 GiB per accelerator", "cpu": "operator-provided CPU",
        "ram": "64 GiB", "os": "Linux distribution/version", "driver": "driver version"
      },
      "model": {
        "source": "org/model", "revision": "pinned revision",
        "quantized_artifact": "exact artifact identifier",
        "tokenizer": "/absolute/path/to/pinned/tokenizer", "template_overrides": {}
      },
      "serving": {
        "engine": "operator-provided engine", "version": "version or image digest",
        "launch_command": "the actual serving command", "environment": {}
      },
      "requests": {
        "base_url": "http://localhost:8080", "model_id": "served-model",
        "context_window": 262144,
        "generation": {"temperature": 0.5, "seed": 121},
        "reasoning": {"chat_template_kwargs": {"enable_thinking": true}}
      },
      "benchmark": {
        "tool": "aiperf", "version": "0.12.0", "transformers_version": "5.17.0",
        "seed": 121, "settings": {"concurrency": 1}
      }
    },
    "observed": {}
  },
  "run_config": {
    "reasoning_effort": "medium", "temperature": 0.5, "top_p": null,
    "top_k": null, "min_p": null, "presence_penalty": null,
    "repetition_penalty": null
  },
  "workload": {
    "scenario": "fresh-context", "input_tokens": 4096, "output_target": 256,
    "messages": [{"role": "user", "content": "Replace with prepared exact-size input"}],
    "repetitions": 1, "request_timeout_seconds": 900
  }
}
```

For the fixed fresh-context curve, replace `workload` with:

```json
{
  "profile": "fresh-context-speed",
  "scenario": "fresh-context",
  "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
  "output_targets": [256, 1024],
  "repetitions": 5,
  "warmup_repetitions": 1,
  "request_timeout_seconds": 900,
  "cache_configuration": {"prefix_caching": false},
  "cache_observations": []
}
```

For the fixed growing-session speed curve, replace `workload` with:

```json
{
  "profile": "growing-session-speed",
  "scenario": "growing-session",
  "checkpoints": [4096, 16384, 32768, 65536, 131072, 196608],
  "output_targets": [256, 1024],
  "repetitions": 5,
  "warmup_repetitions": 1,
  "request_timeout_seconds": 900,
  "cache_configuration": {"prefix_caching": false},
  "cache_observations": []
}
```

The growing-session profile builds each measured turn's full prompt locally with the
submitted tokenizer and executable chat-template kwargs. For every `(output target,
repetition)` pair it runs one separate session: a standalone warm-up at the smallest
supported checkpoint (excluded, response discarded) followed by one measured turn per
checkpoint. Each measured turn appends its own assistant response (content plus
reasoning) to that session's history, so the sent prompt grows across turns; histories
are independent across sessions and never share turns. A checkpoint that exceeds the
declared context window for its target is recorded `unsupported` without invoking a
tool and does not advance the session history. The summary exposes all 60 measured
turns and a 12-point `growing_session_curve` (checkpoint by output target) with each
native series, minimum, maximum, mean, and unavailable count.

The fixed profile prepares each request with the submitted tokenizer and executable
chat-template kwargs. Each of the 12 speed conditions runs one separately retained
warm-up followed by five measured requests. All 72 inputs have distinct leading
prefixes to discourage reuse. This does not control, flush, or claim knowledge of
endpoint cache state. Record known cache configuration and observations in the
workload fields; empty values explicitly mean none were supplied.

The placeholder message is not a prepared 4K input; submit already-sized messages for
the single-condition workload. The model's tokenizer field selects AIPerf's tokenizer.
overrides describe the externally applied template; executable request-side template
kwargs belong in `requests.reasoning` or `requests.generation`. These two dictionaries
are passed to AIPerf unchanged; do not duplicate keys between them. Routing,
messages, streaming, output limit and number of choices are controlled by the
workload and cannot be overridden. No warm-up or repeated scored requests are
requested by the single-condition workload.

Credential-bearing URLs are rejected. Named credentials in declared setup, command
arguments, JSON exports and logs are redacted; declared credential values are also
scrubbed from unlabelled echoes. Native records retain their upstream shape except
for redaction.

## Endpoint authentication and compatibility

The bearer key for an authenticated endpoint is supplied at runtime through the
`REAL_BENCH_API_KEY` environment variable, never through the setup record, a
submission, or the command line. The runner sends it as a `Bearer` header to the
endpoint for the compatibility check and to AIPerf through its `--api-key` option
for every workload (speed, coding, long-context). The key is scrubbed from every
retained artifact: the `command.json` record redacts it by position, the live value
joins the run's scrub set so unlabelled echoes in tool logs and externally echoed
error bodies are removed, and the retained compatibility evidence carries the
server's own messages without the key.

`bench check` (state directory default `.real-bench/setup`, `--settings` for the
full set) and the launch gate before every vLLM run use only two endpoint routes:
`GET /v1/models` and `POST /v1/chat/completions`. They never launch, stop, flush,
tune, or reconfigure the externally operated service. The check is a small,
clearly identified setup-compatibility request, separate from scored work and from
any runtime-estimation workflow: one minimal completion plus, with `--settings`,
one request per final setting so a `400` response names the exact unsupported
parameter. Version floors are advisory only and recorded under `advisory`; the
live probe is the per-instance signal, because an HTTP success alone is not proof
that a parameter was applied (the probe records a named `400` for a rejected
setting and an `unsupported` list in the retained evidence).

The launch gate runs on the resolved final settings (the `run_config` values over
the declared generation, `reasoning_effort` over the template kwargs) before any
scored work. An unsupported setting or a rejected key stops the launch with an
actionable `ValueError` naming the setting or the `REAL_BENCH_API_KEY` contract;
nothing is silently dropped and no work directory is created. On success the
evidence is retained as `compatibility.json` in the run directory. Non-vLLM
engines keep their existing pass-through.

## Per-launch configuration

An executable submission carries a `run_config` beside `setup` and `workload` with
exactly the seven per-run choices: `reasoning_effort` plus `temperature`, `top_p`,
`top_k`, `min_p`, `presence_penalty`, and `repetition_penalty`. It does not accept
frequency penalty or seed; the declared setup seed still applies when a run leaves
the seed unset. Every sampling key must be present, but its value may be `null` to
record an explicit no-override for that parameter. The choices resolve into the final
setup record before execution: non-null values override `setup.requests.generation`,
`null` values remove the parameter, and `reasoning_effort` joins
`setup.requests.reasoning.chat_template_kwargs` (preserving other kwargs such as
`enable_thinking`). The retained `setup.json` is the resolved record as executed, and
prepared inputs are built from it at each launch.

`reasoning_effort` is required at every launch and must not be empty or `off`
(`none` and `null` are rejected); a saved setup or retained history never pre-selects
it. A submission without `run_config` is `422` through `POST /api/runs`.

`bench run` is the same launch boundary:

- With a complete `run_config`, the CLI and API run the supplied choices as-is;
  this is the agent-driven path and needs no terminal input.
- Without one, an interactive terminal prompts for the seven choices. The six
  sampling parameters show the most recent retained run's values as
  recommendations: blank accepts, a number edits, and `none` records an explicit
  no-override (also represented by `null`). A parameter the newest run left absent
  is not recommended, so a later removal is never resurrected from an older run.
  `reasoning_effort` is always prompted with no recommendation.
- `--run-config` supplies the seven choices as a JSON object (the six sampling
  keys and `reasoning_effort`), so an unattended launch of a prepared submission
  that carries no `run_config` does not need a terminal. `--no-interactive`,
  like absent terminal input, fails with the missing choices named instead of
  guessing; unattended launches cannot bypass the requirement.

## Resumable setup state

`bench setup` guides the setup record without JSON authoring. It prompts for the
five `setup.declared` categories in order, saves the nonsecret answers and
preparation status to `setup-state.json` in its state directory (default
`.real-bench/setup`) after every answer, and reports the result as JSON. The
agent-driven form applies the same state without prompts:

```sh
uv run bench setup --no-interactive \
  --answers '{"requests.base_url": "http://localhost:8080"}'
```

`POST /api/setup` applies the same answers and reports the same state;
`GET /api/setup` reads it back without executing any work. Both use the
server's results directory (`results/setup` by default).

- Execution-critical fields are required before any run: the endpoint
  (`requests.base_url`), served model ID (`requests.model_id`), context window
  (`requests.context_window`), tokenizer and template overrides
  (`model.tokenizer`, `model.template_overrides`), and the request settings
  (`requests.generation`, `requests.reasoning`). They cannot be `unknown`.
- Unavailable descriptive hardware, model, and serving metadata is preserved
  explicitly as `unknown`. The value flows through the saved setup, retained
  run `setup.json`, and every public result read, keeping declarations distinct
  from sourced observations in `setup.observed`.
- An interrupted or partial session persists: resuming keeps every valid answer
  and prompts only for the gaps. Preparation entries report honest status;
  nothing unimplemented or unfinished is ever reported as complete.
- A changed fully known configuration marks exactly the preparations that
  depend on it for revalidation, without touching the rest. This is setup
  revalidation only; it does not resume an interrupted benchmark run.

- A setup record is not executable by itself: it carries no `run_config`, because a
  reasoning choice cannot be pre-selected. Launching one still requires the per-run
  configuration described above, completed interactively or supplied in the
  submission.

## Automatic instance detail discovery

`bench setup --discover` and `POST /api/setup` with `"discover": true` offer,
within the same guided flow, to read the declared endpoint's instance details.
Discovery is a choice, never a requirement: a session without it behaves
exactly as before, and a session with it can fall back to manual entry at any
point.

Discovery is read-only. It uses only two endpoint routes, `GET /v1/models` and
`GET /v1/loads`, with the same bearer key contract as the compatibility check
(`REAL_BENCH_API_KEY` is never persisted). It never infers deployment facts the
endpoint does not report and never launches, stops, or reconfigures the
service. A single listed model yields a served model ID; `/v1/loads` yields
the serving version and context window. A missing, partial, or unreadable
route leaves the corresponding field for guided entry; nothing is fabricated.

Findings are presented for review with their provenance before any missing
field is prompted. The operator accepts the discovered value, keeps or
replaces the declaration (a contradiction shows both the declared and the
observed value and is never resolved silently), or leaves the field for
guided entry. Accepted values flow through the same answer boundary as every
other answer; the raw observation is always stored under `setup.observed`
with its source and date, keeping declarations distinct from sourced
observations. Unchanged findings are re-observed silently on a repeat run and
never re-asked.

Because an accepted value is evidence about the endpoint it came from, a
changed `requests.base_url` clears the discovery-accepted declarations and
marks the dependent preparations (compatibility) for revalidation. A value
the operator has since set to something else is operator-owned and survives
the change; only what still matches the endpoint's report is cleared. The
retained observations remain as history of what the old endpoint reported,
and the review record keeps the endpoint identity its values came from.
A review that still offers a value for a missing execution-critical field
does not block the session: the value is pending review, not fabricated.

## Speed workload preparation and launch

`bench setup` prepares the two reusable resources the speed workloads need,
once the setup state is critical-complete: the pinned performance tools
(`aiperf==0.12.0`, `transformers==5.17.0`, verified by loading them in the
pinned environment and reading the versions back, evidence retained under
`speed/performance-tools.json`) and a local copy of the declared tokenizer
(under `speed/tokenizer/` with a `manifest.json` recording the source,
relative file names, per-file sha256, and total bytes). The tokenizer fetch
runs in the pinned environment so the speed workloads size prompts with the
exact tokenizer the performance tools load; no model weights reach the runner,
only the pinned packages and the tokenizer files. A gated tokenizer names the
human-only step (create and export an `HF_TOKEN`, then re-run `bench setup`)
and leaves the entry `pending` instead of failing the session.

Each preparation persists as it goes, moves to `in-progress` before the long
subprocess starts, and reuses a verified download on a later setup: the saved
manifest is accepted when its source matches and every file still hashes to
its recorded value, and the tools probe is re-run on every reuse. An
interruption during the fetch leaves a resumable `in-progress` entry.

When both speed preparations pass, `bench setup` writes two prepared
submissions under the state directory, `submissions/fresh-context.json` and
`submissions/growing-session.json`, each carrying the fixed speed workload
profile and the redacted setup record. They exist only while both preparations
pass, and the setup report lists them under `prepared_submissions` with exact
`launch_commands`. A launch against a critical-complete setup state runs the
speed launch gate before any scored work: it re-runs the tools probe,
re-verifies the tokenizer manifest on disk, and compares each preparation's
hash against the resolved record, so a submission prepared under a different
configuration (a changed tokenizer, template, or benchmark pin) is refused
with no work started. On pass the declared tokenizer is swapped to the local
prepared copy and the evidence is retained as `speed-preparation.json` in the
run directory. Without a complete setup state the launch keeps the existing
live-sizing path, so single-condition and manual submissions are unaffected.

`speed_ready` is true only when both speed preparations are complete against
the current configuration and the tokenizer still verifies on disk.
`default_suite_ready` additionally requires the quality workloads'
preparations (datasets, grading container, and long context), so the full
default suite is never advertised ready before every workload is prepared.
The `bench run` boundary accepts `--setup-directory` (default
`.real-bench/setup`), whose prepared speed and quality inputs gate the launch,
and, for a prepared submission that carries no `run_config`, a `--run-config`
JSON object supplying the per-run choices.

## Quality workload preparation and launch

`bench setup` prepares the two reusable resources the HumanEval+ coding
workload needs, once the setup state is critical-complete: a verified copy of
the pinned dataset (the bundled
`src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz`, copied to
`quality/HumanEvalPlus-v0.1.10.jsonl.gz` and verified against the pinned
decompressed SHA-256, the 164 task rows, and the pinned task ids, evidence
retained under `quality/problems.json`) and the pinned EvalPlus grading
container (`quality/grading-container.json`). The container preparation uses
the Docker CLI directly: the daemon must answer, the pinned image
(`ganler/evalplus@sha256:26b1…`) is pulled once, for `linux/amd64`, when
absent, and running it must report `x86_64`. A missing daemon or a host that
cannot run the platform leaves the entry `pending` with actionable Docker
guidance instead of failing the session.

Each preparation persists as it goes, moves to `in-progress` before the
long subprocess starts, and reuses a verified resource on a later setup:
the dataset copy is re-verified (not re-copied) and the image is never
re-pulled when it is still present. An interruption during the pull leaves a
resumable `in-progress` entry.

When both quality preparations pass, `bench setup` writes the prepared coding
submission `submissions/humaneval-plus.json` under the state directory,
carrying the fixed coding workload profile (the prepared dataset path, one
solution per task, the fixed timeouts, and no `max_tokens`, so the workload's
fixed 65,536-token budget applies) and the redacted setup record. The coding
and speed prepared submissions exist only while their own preparations pass,
and the setup report lists them under `prepared_submissions` with exact
`launch_commands`. A coding launch against a critical-complete setup state
runs the quality launch gate before any scored work: it verifies the prepared
dataset copy still decompresses to the pinned SHA-256 with the pinned tasks,
that the grading image is still present locally, and that it still runs as
`x86_64` under `linux/amd64` (no pull), so a tampered or missing resource is
refused with no work started. On pass the launch grades against the prepared
copy and the evidence is retained as `quality-preparation.json` in the run
directory. Without a complete setup state the launch keeps the submitted
dataset path, so a manual coding submission is unaffected.

## Long-context (sampled RULER) preparation and launch

`bench setup` prepares the reusable resource the sampled-RULER long-context
workload needs, once the setup state is critical-complete: a local checkout of
the pinned upstream RULER source (the `NVIDIA/RULER` revision from
`third_party/lock.yaml`) under `long-context/source/`, hydrated with the
synthetic-data JSON inputs the generators read (SQuAD, HotpotQA, the Paul
Graham essays via the repository's own download helpers, plus the committed
word list). The checkout carries a sha256 data manifest under
`long-context/manifest.json`, with the source evidence retained under
`long-context/source.json`. The fetch runs in the pinned RULER runtime
environment and involves a small git checkout plus a few public data files —
never model weights. A source that cannot be reached names the human-only step
(restore outbound network access to the RULER source and re-run `bench setup`)
and leaves the entry `pending` instead of failing the session.

Each preparation persists as it goes, moves to `in-progress` before the clone
starts, and reuses a verified checkout on a later setup: the manifest is
accepted when the checkout is still at the pinned revision and every data file
still hashes to its recorded value. An interruption during the fetch leaves a
resumable `in-progress` entry.

When the long-context preparation passes, `bench setup` writes the prepared
long-context submission `submissions/sampled-ruler.json` under the state
directory, carrying the fixed long-context workload profile (the prepared
source path, the fixed request timeout, and no `max_tokens`, so the workload's
fixed 65,536-token RULER budget applies) and the redacted setup record. The
long-context, coding, and speed prepared submissions exist only while their
own preparations pass, and the setup report lists them under
`prepared_submissions` with exact `launch_commands`. A long-context launch
against a critical-complete setup state runs the launch gate before any scored
work: it verifies the prepared checkout is still at the pinned revision with
its data manifest intact, so a tampered or missing resource is refused with no
work started. On pass the launch generates from the prepared checkout (and the
tokenizer gate swaps in the prepared tokenizer) and the evidence is retained
as `long-context-preparation.json` in the run directory. Without a complete
setup state the launch keeps the submitted source path, so a manual RULER
submission is unaffected by this change.

The 117 prompts are prepared per launch by the long-context runner with the
upstream generators using the final resolved tokenizer, chat template, and
reasoning choice, so a changed tokenizer, template, or reasoning choice
re-prepares the inputs before scored execution. The template that sizes the
prompts is the declared effective template — the
`model.template_overrides.chat_template` override when one is declared,
otherwise the prepared tokenizer's own `chat_template.jinja` — and the
preparation metadata binds its sha256, so inputs sized under a different
template fail the binding check and are never scored. A generation that runs
the full fixed 65,536-token RULER budget stops with a `length` finish reason;
the request is still scored and the finish reason is retained in the outcome
so the truncation stays visible. The upstream inputs are preserved unmodified
below their nominal ceilings, with measured token counts, the output
reservation, and explicit unsupported sizes retained.

`default_suite_ready` additionally requires the long-context preparation to be
complete and still verify on disk, so the full default suite is ready only once
every workload is prepared.

## Retained contract

Each generated run ID names a new directory. Public reads use only that directory:

- `setup.json`: declared and supplied observed reproduction facts.
- `workload.json`: scenario, exact input checkpoint, output target, prepared history,
  repetition count and timeout, with `evidence: validation`.
- `input.jsonl`: AIPerf input including generation/reasoning settings, redacted after execution.
- `command.json`, `execution.json`: effective invocation and exit/log output, when invoked.
- `native/`: external-tool artifacts, including request payloads, streamed responses
  and measurement exports where exposed.
- `outcomes.json`, `summary.json`: request outcome and normalized metrics; the public
  result adds the separately retained setup and workload.

Fresh-context profile runs put each condition under
`fresh-context-INPUT-OUTPUT/`. Its prepared measured inputs remain at the condition root. Each
measured request has its own `repetition-000/` through `repetition-004/` directory,
containing input, command, execution, and native outputs. Excluded warm-up evidence
is under `warmup/`.
`preparation.json` retains local tokenizer counts, content counts, template overhead,
and unique request nonces. The summary exposes all 60 measured outcomes and a
12-point `fresh_context_curve` with each native series, minimum, maximum, mean, and
unavailable count. Warm-ups never enter those series.

Growing-session profile runs put each session under
`growing-session-TARGET-REP/`. Each measured turn lives under `turn-CHECKPOINT/` with
its locally sized `input.jsonl`, sizing spec and command, effective invocation,
execution, and `native/` outputs; excluded warm-up evidence is under `warmup/` and is
discarded from every measured series. The summary exposes all 60 measured turns and a
12-point `growing_session_curve` (checkpoint by output target).

`schema_version: 1` identifies this shared ordinary-result contract. Single-condition
runs use `evidence: validation`; fresh-context and growing-session profile runs use
`evidence: run`.
`default_workload_complete` is false for a single speed profile, because the
other workloads are separate runs.
`families` identifies performance coverage and explicitly missing coding/long-context
work. `status: completed` means every observation in the declared workload succeeded
or was accounted for as unsupported and its requested warm-up completed; otherwise
it is `partial`.

Outcomes include `success`, `unsupported`, `failed`, `missing`, and `cancelled`.
Missing native request or metric output is explicit. Nonzero tool exits remain
visible in execution evidence; valid completed measurements remain available.
Malformed, mismatched or invalid measurement output is failed. For performance
workloads, an available endpoint input count that differs from the locally exact
context checkpoint fails validation. Sampled RULER retains local preparation and
endpoint usage counts separately under the compatibility policy below;
unavailable endpoint usage stays unavailable and does not discard native timings. Failed
observations have no normalized timing metrics. Raw evidence remains inspectable.
Requested output targets are separate from actual endpoint completion-token counts
and target attainment; early termination is preserved.

Metrics retain upstream values, units and source paths: TTFT and completion latency
in ms; per-user decode throughput and effective endpoint-observed prefill throughput
in tokens/sec/user. Missing metrics have null values and an explanation. Prefill is
an endpoint-observed proxy, not isolated engine prefill speed. Reasoning-token
accounting follows endpoint usage; stream chunks are never used as token counts.
No composite score, fingerprint, canonical log, database, or import is involved.

## Failures and partial results

For cancellation from another terminal or HTTP client, choose a run ID before
submission: `bench run submission.json --results results --run-id ID`, or send
`X-Real-Bench-Run-Id: ID` with `POST /api/runs`. IDs must contain exactly 32 lowercase
hexadecimal characters and must be unused. Without an explicit ID, the runner
creates one. Submission remains synchronous.

`bench cancel ID --results results` or `POST /api/runs/ID/cancel` writes a durable
`cancel.json` request in the run directory. The shared tool runner checks that
request while executing and before starting another tool. The cancellation
response acknowledges the request; the retained result describes the work that
actually completed or was interrupted. Ctrl-C in the run CLI also preserves partial
results.

Every workload records its effective `request_timeout_seconds` in `workload.json`;
the default is 900 seconds (15 minutes). Request failures and timeouts retain their
diagnostics and permit later work. A declared context-window exclusion remains
`unsupported` with its reason. Transport unreachability stops further endpoint
work, while cancellation stops further owned work. Neither path retries a scored
request. Fresh-context repetitions use separate one-request tool invocations,
so an outage stops dispatch before the next repetition. This adds tool startup time
to wall-clock execution; it does not change the 60 declared measured observations
or include startup in native request latency.

Completed native observations remain available after interruption. Interrupted and
unattempted observations remain distinct from successful measurements; missing
latency, output attainment, or quality coverage is never invented. The run remains
`partial`, and the CLI, API and HTML result view read the same retained directory.
Quality summaries expose their actual graded/scored coverage instead of presenting
an incomplete run as a complete quality score.

The shared external-tool boundary retains invocation and execution evidence and
terminates owned processes on interruption. This applies to preparation, excluded
warm-ups, performance measurement, quality generation, and grading. It does not
stop or reconfigure the externally operated endpoint. This lifecycle uses ordinary
run files and introduces no database recovery or request replay protocol.


## HumanEval+ coding workload

Use the same setup record, `bench run`, POST API, result reads, and HTML view with:

```json
{
  "profile": "humaneval-plus",
  "problems_path": "src/llm_bench/datasets/HumanEvalPlus-v0.1.10.jsonl.gz",
  "solutions_per_task": 1,
  "request_timeout_seconds": 900,
  "grading_timeout_seconds": 3600
}
```

This is the `workload` object in the submission. The coding output budget is
fixed by the workload definition at 65,536 tokens, independently of speed output
targets; it is not an operator choice. Supplying `max_tokens` with any other
value is rejected before generation, and omission uses the fixed budget. The
budget bounds reasoning plus final-answer generation where the endpoint counts
reasoning within the total completion allowance. Generation/reasoning settings
come from `setup.requests`; the recorded benchmark seed is used when generation
has no explicit seed. Every instance receives the same fixed instruction and
original problem prompts. All 164 source records must match the pinned
HumanEval+ v0.1.10 SHA-256; subsets and replacement problems are rejected. No
generated solution is silently retried.

When a complete setup state exists, `bench setup` prepares the pinned dataset
and the grading container and writes `submissions/humaneval-plus.json` (see
the quality preparation section); a launch of that prepared submission
grades against the prepared dataset copy and is refused before any work if
the prepared resources no longer verify.

The runner needs Docker with Linux/amd64 support and the image pinned in
`src/llm_bench/humaneval.py` (`IMAGE`). Generation uses
AIPerf 0.12.0 and transformers 5.17.0. EvalPlus 0.4.0.dev2 at revision
`f11cfb92c1d52896a87f988cbebbd74727d56c7e` sanitizes and grades solutions in the
pinned container, with network disabled, one grading worker, and fixed
resource limits. The run records Docker version output and exact commands.
`grading_timeout_seconds` bounds each sanitizer/evaluator process with the shared
runner's additional 120-second startup allowance.

Each run retains original problems, per-task inputs and native streams (including
reasoning and endpoint usage where exposed), generated and sanitized solutions,
the unchanged grading problem records for generated tasks, native grading output,
execution logs, setup/workload, outcomes and summary. Empty visible content after
reasoning exhaustion remains a generated solution for upstream grading.

`coverage.coding` reports expected, attempted, generated, and graded task counts.
`coding.accuracy` and its descriptive Wilson 95% interval use only graded tasks;
`full_suite_accuracy` remains null unless all 164 are graded. Inspect coverage:
missing tasks may bias the observed accuracy, and the interval does not prove a
model ranking. Correctness requires both base and extended tests to pass; native
fail/timeout means incorrect code. Generation failure, not attempted, and grading
infrastructure failure have separate outcomes and never become incorrect answers.
Missing or mismatched native dataset/solution identities cannot produce a grade.
HTTP failures permit later tasks; unavailable transport stops generation and grades
already retained solutions. Cancellation preserves work already written. Completed
coding coverage is not completion of the other benchmark families or agent success.

`tests/test_humaneval_runs.py` exercises the public boundary with controlled tool
outcomes and recorded native tool exports. This validates the integration and
full-164 orchestration deterministically; it is not a live all-164 run.


## Sampled RULER long-context workload

Use the same setup record, `bench run`, POST API, retained result reads, and HTML
view with this `workload` object:

```json
{
  "profile": "sampled-ruler",
  "source_path": "/absolute/path/to/pinned/RULER",
  "request_timeout_seconds": 900
}
```

The profile contains all 13 upstream task types, three samples per task at each
nominal input ceiling of 32,768, 131,072, and 196,608 tokens: 117 requests.
The RULER output budget is fixed by the workload definition at 65,536 tokens,
independently of speed targets; it is not an operator choice. Supplying
`max_tokens` with any other value is rejected before generation, and omission
uses the fixed budget. The largest checkpoint plus the budget (262,144 tokens)
fits a 262,144-token window; checkpoints that leave no room for the budget are
retained as unsupported.

Prepare the official source and data using [the RULER runtime instructions](../third_party/ruler/README.md)
(`bench setup` automates this). The source
must be pinned to `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
When a complete setup state exists, `bench setup` prepares the pinned source and
hydrated data and writes `submissions/sampled-ruler.json` (see the
long-context preparation section); a launch of that prepared submission
generates from the prepared checkout and is refused before any work if the
prepared resources no longer verify.
The submitted model tokenizer is a local directory containing its tokenizer files
and explicit `chat_template.jinja`. Preparation uses the submitted template kwargs
and records the seed, source/data hashes, tokenizer/template hashes, generation
commands and settings. The upstream task generators and metric functions own task
construction and scoring.

Generated text is preserved without padding, truncation, or resampling to force
exact input lengths. Each complete chat prompt is counted with the submitted
tokenizer and template; reports expose actual ranges alongside nominal ceilings.
Output room is reserved separately. Each outcome retains `nominal_input_ceiling`,
`local_input_tokens` and its `local_input_tokens_source`, plus
`endpoint_input_tokens` and its `endpoint_input_tokens_source` when the native
response exposes `usage.prompt_tokens`. `input_count_delta_tokens` records the
observed difference without rewriting either count. A positive endpoint count at
least as large as local preparation and at or below the nominal ceiling is
`endpoint-authoritative` when it differs from local preparation; an exact match
is `exact`. Endpoint undercounts, malformed counts and endpoint counts above the
nominal ceiling are incompatible and remain failed. Missing
endpoint usage is `unavailable`, never inferred from local preparation or stream
chunks. The summary exposes separate local/endpoint ranges, compatibility counts,
the accounting policy, and the output reservation. Unsupported checkpoints and
unavailable or failed outcomes remain visible with no invented quality score.
Scored requests are not silently retried.

Results prominently identify **sampled diagnostic coverage with limited precision**,
not a full official RULER score. Inspect per-context quality together with per-task
coverage: means over available native scores can be biased by missing work, and
three samples per task/context cannot resolve small model differences. Uncertainty is explicitly marked unavailable; the small sample has no calibrated
interval and must not be used to infer a ranking or statistical significance. Scores may
be fractional because upstream metrics count retrieved references. Completion of
this profile does not complete the speed or coding workloads.

Verification exercises full-profile orchestration through the public submission and
independent results interface using controlled tool output. It is implementation
evidence, not a live 117-request model evaluation.

## Retained-run comparison

`bench serve --results results` serves the comparison at `/` and `/reports`.
Select arbitrary runs for side-by-side inspection; setup/workload equality is not
an admission requirement. Exact-value filters and numeric-aware sorting expose
all recorded setup/workload parameters, including hardware, artifacts, engine,
request generation/reasoning settings, tool versions, seeds, and workload profiles.
Selections persist while filtering. The parameter table identifies differing or
missing values across selected runs.

`bench rebuild-index --results results` rebuilds `report-index.json` from the
32-character run directories' `summary.json`, `setup.json`, and `workload.json`.
The index is disposable. Reads refresh it from directories so newly submitted or
partial runs stay visible; no event log, database, or import is required. Corrupt
or incomplete directories are listed as unreadable, without suppressing other
runs. The index duplicates summary data and may grow with retained outcomes.
Directory scans are intended for local result collections, not large hosted stores.

- `GET /api/reports` returns runs, parameter values and unreadable-directory errors.
- Repeat `filter=parameter=JSON` for AND filters, e.g.
  `filter=setup.declared.model.source="org/model"` (URL-encode query values).
- `sort=setup.declared.hardware.count&descending=true` sorts recorded values;
  missing values sort last. Repeat `run_id` to select specific runs.
- `POST /api/reports/rebuild` explicitly rebuilds the same report.
- `GET /api/reports?compact=true` omits prompt and response transcripts from outcome
  rows; `GET /api/dashboard` returns live progress and the run catalog; `GET /runs/{run_id}`
  renders one run.

Metrics and native units remain separate. Fresh/growing curves use shared scales
per scenario, metric, unit and output target; unsupported or missing points retain
gaps and their outcome counts. Quality views retain coding coverage/uncertainty
and sampled-RULER nominal ceilings, actual input ranges and diagnostic limitations.
Raw outcomes expose actual output lengths, timing sources and failures. No missing
family becomes a zero score, and no combined winner score is calculated.

The browser check uses a retained fixture run plus a submitted
unsupported growing-session profile, without sending endpoint requests:
`make web-e2e`.

## Complete default suite

One default-suite submission evaluates one setup through both performance scenarios,
all 164 HumanEval+ problems, and the 117-request sampled RULER profile. The speed
workloads retain their fixed 120 measured observations, with warm-up excluded.
Quality output limits remain separate from the performance output targets. No
thinking variants, concurrent load, or agent trials are added automatically.

When a complete setup state exists, `bench setup` composes the suite for you: once
every workload resource (the pinned performance tools and tokenizer, the HumanEval+
dataset and grading container, and the sampled-RULER source and data) is prepared
and verified, it saves `submissions/default-suite.json` under the setup directory
and reports it with its exact launch command in the setup summary. The submission
carries `workload.profile` set to `default-suite` with the four nested workload
objects — `fresh_context`, `growing_session`, `coding`, and `long_context` — the
redacted setup record, and a `run_config` left to the launch boundary. If any
workload resource is incomplete or stale, the suite submission is not written (or
is removed) and the summary names the missing preparation.

A launch of the prepared suite submission takes the explicit per-launch choices and
then automatically finalizes or revalidates the configuration-dependent preparation
before execution: the speed launch gate re-verifies the tools and tokenizer, the
quality gate re-verifies the dataset and grading container and binds the prepared
problems path, the long-context gate re-verifies the source and data and binds the
prepared source path, and the compatibility gate validates the submitted settings
for gated engines before any scored request. A launch is refused before any work
if the prepared resources no longer verify. A hand-assembled submission remains
valid: set `workload.profile` to `default-suite`, supply the four nested profile
objects documented above (their fixed-coverage validators and timeout settings
apply), and one top-level `setup`.

The same failure policies apply across the combined run: retain individual failures
and unsupported work, continue after ordinary failures, and stop later endpoint
requests after cancellation or unreachability. Retained partial results remain
readable through the CLI, API, result page, and comparison interface. Coding and
long-context quality remain separate from both speed curves, with their original
coverage and uncertainty limits.

The parent run directory retains `setup.json`, `workload.json`, `outcomes.json`,
and `summary.json`. Each phase retains its ordinary run directory under
`phases/PHASE/PHASE_RUN_ID/`, including native outputs and effective tool settings.
The parent summary exposes `phase_progress`, `wall_seconds`, and the two curves
and coding/RULER summaries. Per-phase `observed_timing` includes wall time,
warm-up process time, and growing session construction time. Wall time includes
those costs; do not add them again.

Use the existing stable run ID and cancellation controls for the entire suite.
Skipped phases retain their expected coverage and an explicit reason. A result
from controlled native outputs proves orchestration only; completed live-default
workload evidence still requires executing the declared workload on the instance.
