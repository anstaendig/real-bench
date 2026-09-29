# AIPerf normalization fixture

Unmodified native request metrics and gzip-compressed raw output from the excluded
2026-09-12 warm-up.
AIPerf 0.12.0 sent a synthetic prompt to the served model; endpoint usage reported
180 input tokens and 256 completion tokens, with the entire budget spent in
reasoning. The fixture tests real-bench's public report normalization, not AIPerf's
timing or tokenization internals. It contains no authorization credentials.
