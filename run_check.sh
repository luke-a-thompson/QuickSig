#!/bin/bash
set -e

pytest --quiet tests/test.py

# Run benchmark with minimal parameters for quick pre-commit check
python tests/benchmark.py --n-runs 3 --combinations "[(100, 1, 2), (100, 2, 3)]"
