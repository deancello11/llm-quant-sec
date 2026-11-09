"""Builds the GPTQ and AWQ 4-bit checkpoints ourselves, all from one fixed calibration set.

This is step one of the pipeline: quantize -> generate -> judge -> analyze. Think of it as the
"shrinking" station. We take each full-size model and squeeze it down to 4 bits per weight, two
different ways (GPTQ and AWQ), then save the result to disk so later steps can just load it.
(FP16, INT8 and NF4 are shrunk on the fly when loaded, so they do not need this step.)

Why build them ourselves instead of downloading ready-made ones? Because other people calibrate
with different text, and then you would be comparing apples to oranges. Here every model and both
methods see the exact same sample text, so any difference in safety is down to the method and not
to the study material.

    python -m src.quantize --model qwen25-7b --method gptq
    python -m src.quantize --model all --method all

A 24 GB card is enough: I built every GPTQ and AWQ checkpoint in this project on one RTX 4090, and each one took
about 6 to 10 minutes. Quantizing is the one step that needs a lot of GPU memory, so a smaller card may struggle.
GPTQ and AWQ libraries sometimes pin conflicting torch/transformers versions. If so, use a separate venv
per method (see section 10 of REPORT.md). The calibration set is identical either way.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random

from .common import QUANT_DIR, load_config, now_iso, expand
from .load_model import checkpoint_dir_name

# Maps the method name you type on the command line to the precision name used everywhere else.
METHOD_TO_PRECISION = {"gptq": "gptq4", "awq": "awq4"}

# The quantization libraries download the whole repo when given a model name, including duplicate weight files
# (Llama's original/consolidated.00.pth, Mistral's consolidated.safetensors, 14-16 GB each). Fetch only what loading needs.
_NEEDED = ["*.json", "*.model", "*.model.v3", "*.txt", "*.jinja", "model-*.safetensors", "model.safetensors"]


def local_snapshot(hf_id: str) -> str:
    """Download just the files we need for one model and return the local folder they landed in."""
    from huggingface_hub import snapshot_download
    return snapshot_download(hf_id, allow_patterns=_NEEDED)


def load_calibration_texts(cfg: dict) -> list[str]:
    """The same n_samples generic web-text documents for every model and method (seeded).

    "Calibration" means showing the model some ordinary text while we shrink it, so the shrinking
    method can see which weights matter most. It is like a tailor taking measurements before
    cutting the cloth. We use plain web text, not safety data, so safety is not tuned for or against.
    """
    from huggingface_hub import hf_hub_download
    c = cfg["calibration"]
    path = hf_hub_download(c["source"], c["data_file"], repo_type="dataset")
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            if len(t) >= c["min_chars"]:
                texts.append(t)
    # A seeded shuffle gives a "random" pick that is the same on every run and every machine.
    random.Random(c["seed"]).shuffle(texts)
    return texts[: c["n_samples"]]


def truncate_to_tokens(texts: list[str], tok, max_tokens: int) -> list[str]:
    """Cut each document to at most max_tokens tokens, so every sample has a similar, bounded length."""
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][:max_tokens]
        out.append(tok.decode(ids))
    return out


def quantize_gptq(base: str, out: str, texts: list[str], bits: int, group_size: int, batch_size: int = 2):
    """Shrink one model with GPTQ (round weights one at a time, nudging the rest to cancel the error)."""
    from gptqmodel import GPTQModel, QuantizeConfig
    model = GPTQModel.load(base, QuantizeConfig(bits=bits, group_size=group_size))
    model.quantize(texts, batch_size=batch_size)
    model.save(out)


def quantize_awq(base: str, out: str, texts: list[str], bits: int, group_size: int):
    """Shrink one model with AWQ (use the sample text to find the weights that matter most, then protect them)."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoAWQForCausalLM.from_pretrained(base, low_cpu_mem_usage=True, use_cache=False)
    # The saved AWQ folder needs a tokenizer next to the weights, which is why we save one at the end.
    model.quantize(tok, quant_config={"zero_point": True, "q_group_size": group_size, "w_bit": bits, "version": "GEMM"},
                   calib_data=texts)
    model.save_quantized(out)
    tok.save_pretrained(out)


def main():
    """Command-line entry point: loop over the chosen models and methods and build whatever is missing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--method", default="all", help="gptq, awq or all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    models = expand(args.model, list(cfg["models"]))
    methods = expand(args.method, list(METHOD_TO_PRECISION))
    from transformers import AutoTokenizer

    for m in models:
        base = cfg["models"][m]["hf_id"]
        tok = AutoTokenizer.from_pretrained(base)
        # Each model has its own tokenizer, so trimming happens per model. The documents themselves are the same.
        texts = truncate_to_tokens(load_calibration_texts(cfg), tok, cfg["calibration"]["max_tokens"])
        # The hash is a fingerprint of the text. Two checkpoints with the same fingerprint studied identical material.
        manifest = {"calibration_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
                    "n_samples": len(texts), **cfg["calibration"]}
        for method in methods:
            prec = METHOD_TO_PRECISION[method]
            out = QUANT_DIR / checkpoint_dir_name(m, prec)
            # Quantizing takes a long time, so a finished checkpoint is never redone unless you ask.
            if out.exists() and not args.force:
                print(f"[{m} / {prec}] exists, skipping (use --force to rebuild)")
                continue
            p = cfg["precisions"][prec]
            print(f"[{m} / {prec}] quantizing with {len(texts)} calibration samples")
            out.parent.mkdir(parents=True, exist_ok=True)
            src = local_snapshot(base)
            if method == "gptq":
                quantize_gptq(src, str(out), texts, p["bits"], p["group_size"])
            else:
                quantize_awq(src, str(out), texts, p["bits"], p["group_size"])
            # Leave a receipt next to the checkpoint recording exactly how it was made.
            (out / "calibration_manifest.json").write_text(json.dumps(
                {**manifest, "method": method, "base_model": base, "built": now_iso()}, indent=2))


if __name__ == "__main__":
    main()
