# Verification

## Required local gate

From a clean checkout at the repository root, run:

```sh
make verify
```

The command must print each exact subcommand and one machine-readable
`VERIFY_REPORT` record containing at least:

- source commit and clean-checkout result;
- requested command and component exit codes;
- unchanged-worktree result; and
- test counts and warnings.

The gate runs, in order:

1. Python lint, formatting, and type checking;
2. the full backend test suite; and
3. the report UI browser checks
   (`web/e2e-reports/comparison.spec.ts` and `dashboard.spec.ts` against
   `scripts/report_browser_server.py`).

Verification must not regenerate fixtures, rewrite expected artifacts, create a
benchmark result, or modify the checkout. Use the focused make targets
(`make check`, `make test`, `make web-e2e`) while iterating, but report
`make verify` for closure.

## Deterministic gate contents

The backend suite covers the focused public seam: ordinary-run invocation,
tokenizer sizing, retention, normalization, failure and cancellation outcomes,
independent CLI/API/result reads, report index rebuild, comparison, and the
report UI data. Controlled external-tool fixtures drive the owned-process
boundary; no production endpoint is contacted.

The report UI browser checks load retained results and exercise run listing,
filters, details, comparison charts, and responsive layouts. Controlled dashboard
responses verify progress changes, metric updates, and connection recovery. They
perform no new live benchmark work.

## Evidence boundary

A green `make verify` proves deterministic software behavior for frozen
inputs. It does not prove live model quality, target-hardware latency,
endpoint exclusivity, operator declarations, or engine validation. Live
evidence comes from operator-local pilot runs, which are not part of this
repository.
