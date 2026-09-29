# Security policy

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/anstaendig/real-bench/security/advisories/new)
rather than a public issue.

Notes for operators:

- `bench serve` binds to `127.0.0.1` by default and has no authentication. Do not
  expose it to untrusted networks.
- The endpoint key is read from `REAL_BENCH_API_KEY` and is scrubbed from retained
  setup records, submissions, and logs. Never place credentials in the endpoint
  URL or setup answers.
