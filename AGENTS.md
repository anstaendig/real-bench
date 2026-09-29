## Agent skills

### Benchmark companion skill

The [real-bench companion skill](.agents/skills/real-bench/SKILL.md) guides the
setup-to-run journey with the same `bench` commands terminal users run.

## Development

- Domain language: [CONTEXT.md](CONTEXT.md). Benchmark methodology: [docs/SPEC.md](docs/SPEC.md).
- Before finishing a change, run `make check` and `make test` (see
  [CONTRIBUTING.md](CONTRIBUTING.md)).
- Never commit results, setup state (`results/`, `.real-bench/`), credentials, or
  personal paths and addresses.
