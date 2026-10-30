#!/usr/bin/env bash
# One venv for generation/judging/perf. Put HF_HOME and QS_QUANT_DIR on the network volume first.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf}
export QS_QUANT_DIR=${QS_QUANT_DIR:-/workspace/quantized}
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
echo "Now: hf auth login   (needed for Llama, Mistral, Llama Guard)"
