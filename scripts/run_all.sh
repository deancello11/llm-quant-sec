#!/usr/bin/env bash
# The real thing: generate, judge, benchmark speed and energy, then analyze. Run it after the pilot
# looks good and the GPTQ/AWQ checkpoints are built. If it gets interrupted, just run it again and
# it picks up where it stopped. The benchmark step is last and alone on purpose, so nothing else
# is using the GPU while energy is measured.
# Full grid. Safe to re-run after an interruption: finished prompts are skipped.
set -euo pipefail
python -m src.generate --model all --precision all --dataset all
python -m src.judge
python -m src.bench_perf --model all --precision all    # separate pass, nothing else on the GPU
python -m src.analyze
