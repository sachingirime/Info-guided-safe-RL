#!/usr/bin/env bash
# Run the simulator test suite.  Fast tests by default; everything with --slow.
# Writes tests/report.json (test, result, key numbers).
#   ./run_all_tests.sh            # fast
#   ./run_all_tests.sh --slow     # all, incl. long runs
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ARGS=()
[[ "${1:-}" == "--slow" ]] && ARGS+=(--slow)
exec "$PY" -m pytest tests/ -q -p no:cacheprovider "${ARGS[@]}"
