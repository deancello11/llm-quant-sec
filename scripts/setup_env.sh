#!/usr/bin/env bash
# Run this ONCE on a fresh machine (for example a newly rented GPU pod). It builds a Python
# "virtual environment", which is a private toolbox so this project's libraries do not clash
# with anything else on the computer, then installs everything in requirements.txt.
# Point HF_HOME and QS_QUANT_DIR at the network volume first, so the big model files survive a restart.
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
