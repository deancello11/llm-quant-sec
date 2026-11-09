#!/usr/bin/env bash
# A small practice run before the big one. It sends only 20 prompts per cell to one model (qwen25-7b
# unless you pass another name), at every precision, then judges them. Run this first to make sure
# everything works, and read a few outputs by hand to check they look sensible. Cheap and quick.
# Pilot: 20 prompts per cell, one model, all precisions. Then judge them. Read the outputs by hand afterwards.
set -euo pipefail
MODEL=${1:-qwen25-7b}
python -m src.generate --model "$MODEL" --precision all --pilot
python -m src.judge --pilot
echo "Pilot outputs: outputs/pilot/*.jsonl   judgments: judgments/pilot/*.jsonl"
