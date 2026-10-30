"""Resumable generation: one JSONL line per prompt in outputs/<model>__<precision>__<dataset>.jsonl.

    python -m src.generate --model qwen25-7b --precision all --dataset all
    python -m src.generate --model qwen25-7b --precision all --pilot      # 20 prompts per cell -> outputs/pilot/

Re-running skips every prompt_id already in the file, so an interrupted pod just continues.
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
    done = {r["prompt_id"] for r in read_jsonl(out_path)}
    todo = [r for r in rows if r.prompt_id not in done]
    if not todo:
        return 0
    # Sort by prompt length (ties by id). The order depends only on the tokenizer, so every precision gets identical batches.
    texts = {r.prompt_id: format_fn(r.prompt) for r in todo}
    todo.sort(key=lambda r: (token_len_fn(texts[r.prompt_id]), r.prompt_id))

    written, i, bs = 0, 0, batch_size
    while i < len(todo):
        batch = todo[i:i + bs]
        try:
            results = generate_fn([texts[r.prompt_id] for r in batch])
        except Exception as e:  # CUDA OOM: halve the batch and retry
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
    """Real generation functions for a loaded HF model."""
    import torch

    gen_cfg = cfg["generation"]
    max_new = gen_cfg["max_new_tokens"]
    stop = model.generation_config.eos_token_id
    stop_ids = set(stop if isinstance(stop, list) else [stop]) if stop is not None else {tok.eos_token_id}

    def format_fn(prompt: str) -> str:
        msgs = ([{"role": "system", "content": gen_cfg["system_prompt"]}] if gen_cfg.get("system_prompt") else [])
        msgs.append({"role": "user", "content": prompt})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def token_len_fn(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    @torch.inference_mode()
    def generate_fn(texts: list[str]):
        # add_special_tokens=False: the chat template already contains BOS, so adding it again would change the prompt
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        results = []
        for ids in gen:
            stop_at = next((k for k, t in enumerate(ids) if t in stop_ids), None)
            if stop_at is None:
                n, trunc, body = len(ids), True, ids
            else:
                n, trunc, body = stop_at + 1, False, ids[:stop_at]
            results.append((tok.decode(body, skip_special_tokens=True).strip(), n, trunc))
        return results

    return format_fn, token_len_fn, generate_fn


def main():
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
            append_jsonl(out_dir / "_runs.jsonl", {
                "run_id": run_id, "kind": "generate", "model": m, "precision": p, "checkpoint": source,
                "community": community, "seed": cfg["seed"], "max_new_tokens": cfg["generation"]["max_new_tokens"],
                "batch_size": bs, "config_hash": config_hash(cfg), "pilot": args.pilot,
                "timestamp": now_iso(), "env": env_info()})
            fmt, tlen, gen = make_hf_functions(model, tok, cfg)
            meta = {"model": m, "precision": p, "seed": cfg["seed"], "run_id": run_id}
            for d in datasets:
                run_cell(prompts[d], out_dir / cell_name(m, p, d), meta, fmt, gen, bs, tlen)
            free(model)


if __name__ == "__main__":
    main()
