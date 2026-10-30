"""One function: (model, precision) -> model, tokenizer.

The tokenizer and chat template always come from the original FP16 repo, so every precision of a model
sees byte-identical prompts. A wrong template can fake a "safety change".
"""
from __future__ import annotations

import gc
import importlib.metadata as md

from .common import QUANT_DIR, load_config


def checkpoint_dir_name(model_key: str, precision: str) -> str:
    return f"{model_key}-{precision}"


def resolve_checkpoint(model_key: str, precision: str, cfg: dict | None = None) -> tuple[str, bool]:
    """Where the weights come from. Returns (path or repo id, is_community_checkpoint)."""
    cfg = cfg or load_config()
    base = cfg["models"][model_key]["hf_id"]
    method = cfg["precisions"][precision]["method"]
    if method in ("none", "bnb-int8", "bnb-nf4"):
        return base, False
    override = cfg.get("community_checkpoints", {}).get(f"{model_key}/{precision}")
    if override:
        return override, True
    path = QUANT_DIR / checkpoint_dir_name(model_key, precision)
    if not path.exists():
        raise FileNotFoundError(
            f"No quantized checkpoint at {path}. Build it with:\n"
            f"  python -m src.quantize --model {model_key} --method {method}")
    return str(path), False


def _dtype_kwarg(dtype) -> dict:
    """transformers 4.56+ renamed torch_dtype to dtype."""
    from packaging.version import Version
    return {"dtype": dtype} if Version(md.version("transformers")) >= Version("4.56") else {"torch_dtype": dtype}


def load_tokenizer(model_key: str, cfg: dict | None = None):
    from transformers import AutoTokenizer
    cfg = cfg or load_config()
    tok = AutoTokenizer.from_pretrained(cfg["models"][model_key]["hf_id"])
    tok.padding_side = "left"          # needed for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_key: str, precision: str, cfg: dict | None = None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = cfg or load_config()
    method = cfg["precisions"][precision]["method"]
    source, _community = resolve_checkpoint(model_key, precision, cfg)
    kw = dict(device_map={"": 0}, attn_implementation="sdpa", **_dtype_kwarg(torch.float16))
    if method == "bnb-int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif method == "bnb-nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    # gptq / awq: transformers reads the quantization config saved in the checkpoint
    model = AutoModelForCausalLM.from_pretrained(source, **kw)
    model.eval()
    return model, load_tokenizer(model_key, cfg)


def free(model) -> None:
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
