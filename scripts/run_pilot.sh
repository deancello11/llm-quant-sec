#!/usr/bin/env bash
# Pilot: 20 prompts per cell, one model, all precisions. Then judge them. Read the outputs by hand afterwards.
set -euo pipefail
MODEL=${1:-qwen25-7b}
python -m src.generate --model "$MODEL" --precision all --pilot
python -m src.judge --pilot
echo "Pilot outputs: outputs/pilot/*.jsonl   judgments: judgments/pilot/*.jsonl"
