# Benchmark Specification

This document defines the benchmark methodology. For commands, workload objects, and
the API, see [run-interface.md](run-interface.md).

## Purpose and scope

Compare model instances and configurations, including quantizations, using separate
performance, coding-quality, and long-context-quality statistics. Coding-agent
workloads motivate the scenarios, but the default suite does not establish success
through any agent harness.

Use existing benchmark tools for requests, measurement, task generation, and
grading. real-bench owns workload configuration, invocation, result retention,
normalization, and comparison; it is not a measurement engine of its own.

An operator hosts the instance remotely. real-bench does not launch, stop, flush,
or reconfigure it. Each run evaluates one submitted configuration, including its
reasoning settings. Alternative configurations require separate runs.

At every launch, choose `reasoning_effort` explicitly. `"off"` disables model
thinking; it is not inherited from saved setup or a prior run. The requested
mode is retained with the run, even though the outgoing no-thinking switch is
engine-specific.

Durations are measured and retained per request, phase, and run. No runtime
forecast or timing target precedes or gates execution, and coverage is fixed
rather than reduced to meet a time budget.

## Reproduction record

Store one readable setup file with each run, containing:

- Hardware: accelerator model, count and memory, CPU, RAM, OS, and driver.
- Model: source/revision, exact quantized artifact identifier, and tokenizer or
  chat-template overrides.
- Serving: engine/version or container digest, launch command, and relevant
  environment variables.
- Requests: endpoint model ID, generation settings, and reasoning settings.
- Benchmark: tool versions, workload settings, and seeds.

Require these categories without exhaustive per-weight hashes or engine-specific
configuration schemas. Distinguish operator declarations from observed facts.
Redact credentials from retained records. Retain setup as executed so reports
can explain differences between runs.

## Default performance workload

Concurrency is one. Concurrent load and actual agent evaluations are out of scope.

Measure both:

- Fresh context: independent requests with distinct prefixes to discourage reuse.
- Growing sessions: accumulated conversation and tool outputs with shared history
  preserved between turns. Independent repetitions use independent histories.

Use input checkpoints of 4K, 16K, 32K, 64K, 128K, and 192K, where K = 1,024 tokens.
The ceiling is 196,608 input tokens, reflecting compaction before a 256K window
fills. Reserve output space separately. Use the submitted model's tokenizer for
sizing and account for chat-template overhead during tool validation.

At each checkpoint use output targets of 256 and 1,024 tokens. Measure five
repetitions per condition; exclude warm-up. This defines 24 conditions and 120
measured checkpoint observations. Growing-session construction may require
additional unscored requests; record their settings.

Record actual output length and early termination rather than treating a target as
guaranteed attainment. Retain generated inputs and endpoint token counts when
available. Record cache configuration and available observations, without asserting
unobserved cache hits or cache-cold state.

Report TTFT, decode rate, completion latency, variability, and context curves
separately for the scenarios. Preserve upstream metric definitions and units.
If prefill throughput is estimated as input tokens divided by TTFT, label it
effective endpoint-observed prefill throughput, not isolated engine prefill speed.
Retain reasoning settings and token-accounting semantics when interpreting metrics.
Thinking-off runs are comparable only with other thinking-off runs; do not
interpret a thinking-on/off difference as an instance-only difference. When
thinking is off, no reasoning tokens consume the output target or fixed
quality output budget; when thinking is on, reasoning tokens count wherever
the endpoint includes them in the generated-token allowance.

## Default quality workloads

### Coding

Run all 164 HumanEval+ problems through EvalPlus, generating one solution per
problem. Problems remain identical across instances. Use EvalPlus's existing Docker
grading workflow on the benchmark runner and record its version. Generation uses
the remote endpoint.

Report correctness and per-task outcomes separately from speed, with uncertainty.
Quality tasks use a fixed recorded output budget of 65,536 tokens rather than
the speed workload's output targets. The budget is fixed by the workload
definition, not chosen by the operator; exhaustion is retained as a `length`
finish reason and never auto-grows the limit or retries the scored task. This is
standalone code quality, not agent success.

### Long context

Use a sampled RULER profile: all 13 task types, three samples each, at 32K, 128K,
and 192K nominal input ceilings, for 117 requests. K = 1,024; the largest
ceiling is 196,608 input tokens. Use the submitted tokenizer and account for
chat-template overhead. Reserve the fixed 65,536-token quality output budget
separately and retain generated inputs. The budget is fixed by the workload
definition, not chosen by the operator; the largest checkpoint plus the budget
fits a 256K window, and checkpoints whose input leaves no room for the budget are
retained as unsupported rather than reduced.

For sampled RULER, accept the unmodified upstream generators' inputs below each
ceiling. Retain both the nominal ceiling and actual input-token count per request,
and expose actual ranges in reports and comparisons. Reject ceiling overshoots;
do not pad, truncate, or resample solely to force exact lengths. Freeze the task
source, seeds, tokenizer/template, and preparation settings. This sizing rule applies only to
sampled RULER; performance input targets remain exact.

Label this a diagnostic subset with limited precision, not the full official
benchmark score. Retain task-level outcomes and report quality by context.

## Failure and completion behavior

Requests time out after a configurable 15 minutes by default.

- Retain request failures and continue to later conditions.
- Skip sizes known to exceed the declared context window and show them as unsupported.
- Do not silently retry scored requests.
- Stop with partial results if the endpoint becomes unreachable.
- Preserve partial results when cancelled.
- Do not fabricate latency or represent incomplete quality coverage as complete.

Selected workloads should match for controlled comparisons; reports nevertheless
permit arbitrary runs together and expose their differences.

## Results and comparison

Each ordinary run directory contains setup, workload definition, tool-native
outputs, request-level measurements and outcomes, and a normalized summary.
Preserve prompts and responses where tools expose them.

A UI index is rebuildable from these directories. Support filtering, sorting, and
side-by-side inspection of arbitrary instances, with setup and workload parameters
available after execution. Do not require an exact hardware or environment
fingerprint for admission.

Show coding accuracy, long-context accuracy, TTFT, decode speed, effective prefill
throughput when available, and completion latency separately. No combined
quality/speed score.

## Non-goals

real-bench has no mandatory broad quality suite, model-compression comparison,
agent-harness trials, host telemetry, composite ranking, or admission gate.
Results are ordinary run directories rather than an event log or database, so
there is no recovery protocol or import/export machinery.

Reproduction metadata, explicit failures, native outputs, and honest measurement
boundaries are always preserved.
