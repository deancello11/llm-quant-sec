"""One function: (model, precision) -> model, tokenizer.

The tokenizer and chat template always come from the original FP16 repo, so every precision of a model
sees byte-identical prompts. A wrong template can fake a "safety change".

Where this fits: the pipeline is quantize -> generate -> judge -> analyze. This file sits just before "generate".
quantize.py builds the shrunken GPTQ and AWQ checkpoints. This file finds the right weights for any of the
five precisions (FP16, INT8, NF4, GPTQ, AWQ) and loads them onto the GPU, so generate.py does not need to
know how each one is stored.
"""
from __future__ import annotations

import gc
import importlib.metadata as md

from .common import QUANT_DIR, load_config


def checkpoint_dir_name(model_key: str, precision: str) -> str:
    """Folder name for a locally built checkpoint, like 'qwen25-7b-awq'."""
    return f"{model_key}-{precision}"


def resolve_checkpoint(model_key: str, precision: str, cfg: dict | None = None) -> tuple[str, bool]:
    """Where the weights come from. Returns (path or repo id, is_community_checkpoint)."""
    cfg = cfg or load_config()
    base = cfg["models"][model_key]["hf_id"]
    method = cfg["precisions"][precision]["method"]
    # FP16, INT8 and NF4 all start from the original weights. bitsandbytes shrinks them on the fly while loading,
    # so there is no separate file to build. Only GPTQ and AWQ need a pre-built checkpoint.
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
    """transformers 4.56+ renamed torch_dtype to dtype. This picks whichever spelling the installed version wants."""
    from packaging.version import Version
    return {"dtype": dtype} if Version(md.version("transformers")) >= Version("4.56") else {"torch_dtype": dtype}


def load_tokenizer(model_key: str, cfg: dict | None = None):
    """Load the tokenizer (and chat template) from the ORIGINAL model repo, whatever the precision."""
    from transformers import AutoTokenizer
    cfg = cfg or load_config()
    # Always the original repo's tokenizer, never the one saved next to a quantized checkpoint. A quantized
    # copy could ship a slightly different template, and then the "same" prompt would not be the same.
    tok = AutoTokenizer.from_pretrained(cfg["models"][model_key]["hf_id"])
    # Left padding puts the blank filler before the prompt, so every answer starts right where the prompt ends.
    # With right padding, a decoder-only model would try to continue after the filler and write garbage.
    tok.padding_side = "left"          # needed for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_key: str, precision: str, cfg: dict | None = None):
    """Load one model at one precision onto GPU 0. Returns (model, tokenizer), with the model in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cfg = cfg or load_config()
    method = cfg["precisions"][precision]["method"]
    source, _community = resolve_checkpoint(model_key, precision, cfg)
    # Same loading settings for every precision (fp16 compute, same attention code, whole model on one GPU),
    # so the only thing that changes between runs is how the weights are stored.
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
    """Give GPU memory back so the next model fits.

    Heads up: `del model` here only removes this function's own reference. The caller must drop its references
    too (generate.py does), or the memory stays taken. The collect and empty_cache steps then hand it back.
    """
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
