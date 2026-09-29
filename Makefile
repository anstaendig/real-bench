.PHONY: check test format web-e2e verify

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest

format:
	uv run ruff format src tests

web-e2e:
	npm --prefix web run e2e-reports

verify:
	VERIFY_COMMAND='make verify' uv run python scripts/verify_gate.py
