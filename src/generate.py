"""Resumable generation: one JSONL line per prompt in outputs/<model>__<precision>__<dataset>.jsonl.

    python -m src.generate --model qwen25-7b --precision all --dataset all
    python -m src.generate --model qwen25-7b --precision all --pilot      # 20 prompts per cell -> outputs/pilot/

Re-running skips every prompt_id already in the file, so an interrupted pod just continues.

Where this fits: the pipeline is quantize -> generate -> judge -> analyze. This file is the "generate" step.
It loads each model at each precision, feeds it every prompt, and saves what the model said. It does not decide
whether an answer was harmful (that is judge.py's job). It only collects the answers, carefully and the same way
every time, so that any difference we see later is about the precision and not about how we asked.

A "cell" is one (model, precision, dataset) combination. Each cell gets its own output file.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Callable

from .common import (OUTPUTS_DIR, append_jsonl, cell_name, config_hash, env_info, expand, load_config,
                      new_run_id, now_iso, read_jsonl)
from .prompts import PromptRow, build_prompts

# generate_fn takes a list of chat-formatted prompt strings and returns, per prompt, (response, n_tokens, truncated)
GenerateFn = Callable[[list[str]], list[tuple[str, int, bool]]]


def run_cell(rows: list[PromptRow], out_path: pathlib.Path, meta: dict, format_fn: Callable[[str], str],
             generate_fn: GenerateFn, batch_size: int, token_len_fn: Callable[[str], int]) -> int:
    """Generate every prompt not yet in out_path. Returns the number of new records written."""
    # Resumable on purpose: rented GPUs can vanish mid-run. Each finished answer is saved right away, so on restart
    # we just skip the prompt_ids we already have, like a bookmark in a long book.
    done = {r["prompt_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r.prompt_id not in done]
    if not todo:
        return 0
    # Sort by prompt length (ties by id). The order depends only on the tokenizer, so every precision gets identical batches.
    # Why it matters: in a batch, short prompts get padded to match the longest one. Similar lengths means little
    # padding (faster). And identical batches means padding is not a hidden difference between precisions.
    texts = {r.prompt_id: format_fn(r.prompt) for r in todo}
    todo.sort(key=lambda r: (token_len_fn(texts[r.prompt_id]), r.prompt_id))

    written, i, bs = 0, 0, batch_size
    while i < len(todo):
        batch = todo[i:i + bs]
        try:
            results = generate_fn([texts[r.prompt_id] for r in batch])
        except Exception as e:  # CUDA OOM: halve the batch and retry
            # Long prompts plus a big batch can overflow the 24 GB card. Smaller batches fix it without
            # skipping anything. (Matching on the message text is a bit fragile, but it works for CUDA OOM.)
            if "out of memory" in str(e).lower() and bs > 1:
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                bs = max(1, bs // 2)
                print(f"  OOM, batch size -> {bs}")
                continue
            raise
        for r, (resp, n_tok, trunc) in zip(batch, results):
            # One line per answer, written immediately (not at the end of the batch or the run).
            append_jsonl(out_path, {
                "model": meta["model"], "precision": meta["precision"], "dataset": r.dataset,
                "prompt_id": r.prompt_id, "behavior_id": r.behavior_id, "variant": r.variant,
                "category": r.category, "prompt": r.prompt, "response": resp,
                "n_tokens": n_tok, "truncated": trunc, "seed": meta["seed"],
                "timestamp": now_iso(), "run_id": meta["run_id"],
            })
            written += 1
        i += len(batch)
        print(f"  {out_path.name}: {len(done) + written}/{len(done) + len(todo)}", flush=True)
    return written


def make_hf_functions(model, tok, cfg):
    """Real generation functions for a loaded HF model.

    Returns three helpers: format_fn (wrap a prompt in the model's chat template), token_len_fn (count its tokens)
    and generate_fn (answer a batch of formatted prompts). run_cell() takes them as plain callables so the
    batching and resume logic can be tested without a GPU.
    """
    import torch

    gen_cfg = cfg["generation"]
    max_new = gen_cfg["max_new_tokens"]
    # Some models have several "I am done" tokens (a list), others have one. Collect them all so we can
    # tell a finished answer from one that was cut off at the token limit.
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, list) else [stop]) if stop is not None else {tok.eos_token_id}

    def format_fn(prompt: str) -> str:
        """Wrap one prompt in the model's own chat template (the way it was trained to be talked to)."""
        msgs = ([{"role": "system", "content": gen_cfg["system_prompt"]}] if gen_cfg.get("system_prompt") else [])
        msgs.append({"role": "user", "content": prompt})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def token_len_fn(text: str) -> int:
        """How many tokens a formatted prompt takes. Used only to sort prompts into batches."""
        return len(tok(text, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def generate_fn(texts: list[str]):
        """Answer a batch of formatted prompts. Returns (response, n_tokens, truncated) for each one."""
        # add_special_tokens=False: the chat template already contains BOS, so adding it again would change the prompt
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # Greedy decoding (do_sample=False): always pick the most likely next token. No dice rolls, so the same
        # prompt gives the same answer, and differences between precisions are not just luck. The None settings
        # silence warnings about sampling knobs that greedy decoding ignores anyway.
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()  # keep only the new tokens, not the prompt we sent in
        results = []
        for ids in gen:
            # No stop token anywhere means the model ran into the token limit mid-thought: mark it truncated.
            # Otherwise cut at the first stop token, because shorter answers in a batch are padded afterwards.
            stop_at = next((k for k, t in enumerate(ids) if t in stop_ids), None)
            if stop_at is None:
                n, trunc, body = len(ids), True, ids
            else:
                n, trunc, body = stop_at + 1, False, ids[:stop_at]
            results.append((tok.decode(body, skip_special_tokens=True).strip(), n, trunc))
        return results

    return format_fn, token_len_fn, generate_fn


def main():
    """Command-line entry point: loop over models and precisions, generating every dataset for each."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--precision", default="all")
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--pilot", action="store_true", help="20 prompts per cell, written to outputs/pilot/")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    from .load_model import free, load_model, resolve_checkpoint

    cfg = load_config()
    models = expand(args.model, list(cfg["models"]))
    precisions = expand(args.precision, list(cfg["precisions"]))
    datasets = expand(args.dataset, list(cfg["datasets"]))
    out_dir = OUTPUTS_DIR / "pilot" if args.pilot else OUTPUTS_DIR
    bs = args.batch_size or cfg["generation"]["batch_size"]

    prompts = {d: build_prompts(cfg, d, pilot=args.pilot) for d in datasets}
    print({d: len(p) for d, p in prompts.items()}, "prompts per dataset")

    for m in models:
        for p in precisions:
            # Check before loading: loading a model takes minutes, so skip it entirely if nothing is left to do.
            pending = any(
                len({r["prompt_id"] for r in read_jsonl(out_dir / cell_name(m, p, d))}) < len(prompts[d])
                for d in datasets)
            if not pending:
                print(f"[{m} / {p}] already complete")
                continue
            print(f"[{m} / {p}] loading")
            source, community = resolve_checkpoint(m, p, cfg)
            model, tok = load_model(m, p, cfg)
            run_id = new_run_id()
            # A receipt for this run (versions, GPU, git commit, config hash), so results can be traced later.
            append_jsonl(out_dir / "_runs.jsonl", {
                "run_id": run_id, "kind": "generate", "model": m, "precision": p, "checkpoint": source,
                "community": community, "seed": cfg["seed"], "max_new_tokens": cfg["generation"]["max_new_tokens"],
                "batch_size": bs, "config_hash": config_hash(cfg), "pilot": args.pilot,
                "timestamp": now_iso(), "env": env_info()})
            fmt, tlen, gen = make_hf_functions(model, tok, cfg)
            meta = {"model": m, "precision": p, "seed": cfg["seed"], "run_id": run_id}
            for d in datasets:
                run_cell(prompts[d], out_dir / cell_name(m, p, d), meta, fmt, gen, bs, tlen)
            # drop every reference (the closures hold the model too), or the next load runs out of GPU memory
            # (Think of a parking space: the old car must fully leave before the next one fits. Python only
            # frees the model once nothing at all still points to it.)
            del model, tok, fmt, tlen, gen
            free(None)


if __name__ == "__main__":
    main()
