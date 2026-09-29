# real-bench Domain Language

real-bench compares externally operated model instances using performance,
coding-quality, and long-context-quality measurements.

## Language

**Instance**:
An externally operated model deployment with a particular hardware, model, serving,
and request configuration submitted for evaluation.
_Avoid_: Selected entrant, automatically tuned candidate

**Endpoint**:
The remote interface through which benchmark tools send requests to an instance.
_Avoid_: Managed deployment

**Setup record**:
The readable description of hardware, model artifacts, serving configuration,
request settings, and benchmark settings needed to reproduce a run.
_Avoid_: Exhaustive deployment manifest

**Run**:
One execution of a declared workload against one submitted instance configuration.
_Avoid_: Canonical run, candidate revision

**Phase**:
One workload within a combined run, with its own progress, outcomes, and elapsed time.
_Avoid_: Separate instance configuration

**Workload profile**:
The recorded tasks, context sizes, output targets, repetitions, seeds, and tool
settings defining what a run measures.
_Avoid_: Comparison fingerprint

**Fresh-context scenario**:
A performance scenario using independent requests across growing input sizes with
distinct prefixes to discourage reuse.
_Avoid_: Guaranteed cache-cold request

**Growing-session scenario**:
A performance scenario measuring successive turns as conversation and tool outputs
accumulate, allowing available prefix reuse to affect performance.
_Avoid_: Guaranteed cache hit

**Context checkpoint**:
A declared input-token size at which performance or long-context quality is measured,
with output space reserved separately.
_Avoid_: Total context-window capacity

**Output target**:
The requested output-token length of a speed condition, recorded separately from
the number of tokens actually generated.
_Avoid_: Guaranteed output length

**Output budget**:
The fixed maximum generated-token allowance for a test. It includes reasoning
tokens where the endpoint counts them, but a thinking-off run generates none.
_Avoid_: Guaranteed answer length, runtime estimate

**Speed condition**:
One scenario, context checkpoint, and output target measured through fixed repetitions.
_Avoid_: Quality task

**Sampled RULER profile**:
The declared diagnostic subset of RULER tasks and context sizes used to compare
long-context quality with limited precision.
_Avoid_: Full official RULER score

**Run directory**:
The retained setup, workload, native tool outputs, measurements, outcomes, and
summary for one run.
_Avoid_: Canonical artifact graph, hydrated export

**Comparison**:
A user-selected view of run metrics and setup parameters, permitting different
instances and workloads while retaining their differences.
_Avoid_: Composite quality-speed ranking, fingerprint admission

**Partial result**:
Retained measurements from a run that stopped before finishing its declared work.
_Avoid_: Complete quality score, fabricated failure latency
