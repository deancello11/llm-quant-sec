"""Speed and energy per precision, as a SEPARATE pass on a fixed synthetic prompt set.

    python -m src.bench_perf --model qwen25-7b --precision all

Run it on its own, with nothing else on the GPU. Judging, logging and safety generation would contaminate the energy numbers.
Same method as the earlier llm-inference-bench project: random token ids of fixed length, EOS blocked so every reply has
exactly output_len tokens, NVML energy counter read before and after, each timed window lasts at least min_seconds.
Results append to results/perf.jsonl and are resumable.
"""
from __future__ import annotations

import argparse
import random
import time

from .common import RESULTS_DIR, append_jsonl, env_info, expand, load_config, new_run_id, now_iso, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--precision", default="all")
    args = ap.parse_args()

    import torch
    from .gpu import GpuMonitor
    from .load_model import free, load_model

    cfg = load_config()
    pc = cfg["perf"]
    models = expand(args.model, list(cfg["models"]))
    precisions = expand(args.precision, list(cfg["precisions"]))
    path = RESULTS_DIR / "perf.jsonl"
    done = {(r["model"], r["precision"], r["concurrency"], r["repeat"]) for r in read_jsonl(path)}
    mon = GpuMonitor()
    mon.start()
    gpu_info = mon.info()

    def make_batch(tok, n, seed):
        rng = random.Random(seed)
        hi = min(tok.vocab_size, 30000)
        ids = torch.tensor([[rng.randrange(1000, hi) for _ in range(pc["input_len"])] for _ in range(n)], device="cuda")
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    @torch.inference_mode()
    def gen(model, tok, batch):
        n = pc["output_len"]
        model.generate(**batch, max_new_tokens=n, min_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                       top_k=None, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()

    for m in models:
        for p in precisions:
            if all((m, p, c, r) in done for c in pc["concurrency"] for r in range(pc["repeats"])):
                continue
            print(f"[{m} / {p}] loading")
            model, tok = load_model(m, p, cfg)
            idle_w = mon.measure_idle_power_w(pc["idle_seconds"])
            weights_gb = torch.cuda.memory_allocated() / 2**30
            run_id = new_run_id()
            for c in pc["concurrency"]:
                for _ in range(pc["warmup_runs"]):
                    gen(model, tok, make_batch(tok, c, seed=0))
                for r in range(pc["repeats"]):
                    if (m, p, c, r) in done:
                        continue
                    w = mon.begin_window()
                    t0, calls = time.perf_counter(), 0
                    while time.perf_counter() - t0 < pc["min_seconds"] or calls == 0:
                        gen(model, tok, make_batch(tok, c, seed=1000 * r + calls + 1))
                        calls += 1
                    res = mon.end_window(w)
                    tokens = calls * c * pc["output_len"]
                    append_jsonl(path, {
                        "run_id": run_id, "model": m, "precision": p, "concurrency": c, "repeat": r, "calls": calls,
                        "tokens": tokens, "tokens_per_s": tokens / res["duration_s"], "j_per_token": res["energy_j"] / tokens,
                        "mj_per_token": 1000 * res["energy_j"] / tokens, "idle_power_w": idle_w, "weights_gb_allocated": weights_gb,
                        **res, "gpu": gpu_info, "timestamp": now_iso(), "env": env_info() if r == 0 else None})
                    print(f"  c={c} r={r}: {tokens / res['duration_s']:.0f} tok/s, {1000 * res['energy_j'] / tokens:.0f} mJ/tok"
                          f"{'  THROTTLED' if res['thermal_throttled'] else ''}", flush=True)
            del model, tok
            free(None)
    mon.stop()


if __name__ == "__main__":
    main()
