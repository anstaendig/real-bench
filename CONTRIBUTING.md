# Contributing

Thanks for your interest in real-bench.

## Development setup

```sh
uv sync
uv tool install --editable .   # optional: puts `bench` on your PATH
```

Python 3.12+ and `uv` are required; Docker is only needed for HumanEval+ grading
and is not needed to run the tests, which use controlled tool fixtures and never
contact a real endpoint.

## Before opening a pull request

```sh
make check      # ruff, ruff format --check, mypy --strict
make test       # backend tests
make web-e2e    # dashboard browser tests (needs `npm --prefix web ci` once)
```

`make verify` runs the full gate. See [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Guidelines

- Read [docs/README.md](docs/README.md) and [CONTEXT.md](CONTEXT.md) first; changes
  to benchmark methodology must follow [docs/SPEC.md](docs/SPEC.md).
- Benchmark results, run directories, and setup state (`results/`, `.real-bench/`)
  are local and are never committed.
- Never commit credentials, endpoint addresses, or personal paths. Use
  placeholders such as `http://localhost:8080` in docs and tests.
- Add or update tests with behavior changes.

Contributions are accepted under the [Apache License 2.0](LICENSE).
