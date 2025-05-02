#!/bin/bash
set -e

# Run pytest and capture its exit code
pytest --quiet tests/test.py
PYTEST_EXIT=$?

# Run benchmark with regression checking
python tests/benchmark.py --check-regression
BENCHMARK_EXIT=$?

# Exit with non-zero status if either command failed
if [ $PYTEST_EXIT -ne 0 ] || [ $BENCHMARK_EXIT -ne 0 ]; then
    exit 1
fi
