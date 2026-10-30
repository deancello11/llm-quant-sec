#!/usr/bin/env bash
# Full grid. Safe to re-run after an interruption: finished prompts are skipped.
set -euo pipefail
python -m src.generate --model all --precision all --dataset all
python -m src.judge
python -m src.bench_perf --model all --precision all    # separate pass, nothing else on the GPU
python -m src.analyze
