"""Speed and energy per precision, as a SEPARATE pass on a fixed synthetic prompt set.

This is the speed-and-energy half of the project. The safety pipeline (quantize -> generate -> judge ->
analyze) asks "is the shrunk model still safe?". This script asks "what do we get in return?": how many
tokens per second it writes, and how much GPU energy each token costs. Think of it as a fuel-economy test
run on a treadmill, so every car faces exactly the same road.

It is deliberately separate from safety generation. Any other work on the GPU (the judge, logging,
a second job) would burn extra energy during the test and muddy the numbers.

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
    """Load each model at each precision and time it at every concurrency level, saving one row per run."""
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
    # Remember finished runs so an interrupted benchmark can pick up where it stopped instead of starting over.
    done = {(r["model"], r["precision"], r["concurrency"], r["repeat"]) for r in read_jsonl(path)}
    # The monitor keeps a background log of GPU power, clocks and temperature for the whole session.
    mon = GpuMonitor()
    mon.start()
    gpu_info = mon.info()

    def make_batch(tok, n, seed):
        """Build n prompts of random token ids, so the test never depends on what any text says."""
        rng = random.Random(seed)
        # Skip the lowest ids (often special tokens) and cap the top, so we only draw ordinary-looking tokens.
        hi = min(tok.vocab_size, 30000)
        ids = torch.tensor([[rng.randrange(1000, hi) for _ in range(pc["input_len"])] for _ in range(n)], device="cuda")
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    @torch.inference_mode()
    def gen(model, tok, batch):
        """Generate exactly output_len new tokens for every prompt in the batch, and wait until the GPU is done."""
        n = pc["output_len"]
        model.generate(**batch, max_new_tokens=n, min_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                       top_k=None, pad_token_id=tok.pad_token_id)
        # GPU work is queued up and returns early. Without this wait the stopwatch could stop before the work does.
        torch.cuda.synchronize()

    for m in models:
        for p in precisions:
            if all((m, p, c, r) in done for c in pc["concurrency"] for r in range(pc["repeats"])):
                continue
            print(f"[{m} / {p}] loading")
            model, tok = load_model(m, p, cfg)
            # Idle power is the "engine on, car parked" baseline, to compare with power under load.
            idle_w = mon.measure_idle_power_w(pc["idle_seconds"])
            weights_gb = torch.cuda.memory_allocated() / 2**30
            run_id = new_run_id()
            for c in pc["concurrency"]:
                # Warm-up runs are thrown away. The first calls are slow while the GPU wakes up and caches fill.
                for _ in range(pc["warmup_runs"]):
                    gen(model, tok, make_batch(tok, c, seed=0))
                for r in range(pc["repeats"]):
                    if (m, p, c, r) in done:
                        continue
                    w = mon.begin_window()
                    t0, calls = time.perf_counter(), 0
                    # Keep generating until at least min_seconds have passed (10 s or more). A very short window
                    # catches only a few power readings, and the energy counter updates slowly, so a short
                    # result would be noisy. "calls == 0" guarantees at least one call even if min_seconds is tiny.
                    while time.perf_counter() - t0 < pc["min_seconds"] or calls == 0:
                        gen(model, tok, make_batch(tok, c, seed=1000 * r + calls + 1))
                        calls += 1
                    res = mon.end_window(w)
                    # Energy divided by tokens gives a fair "cost per token" number across setups.
                    tokens = calls * c * pc["output_len"]
                    append_jsonl(path, {
                        "run_id": run_id, "model": m, "precision": p, "concurrency": c, "repeat": r, "calls": calls,
                        "tokens": tokens, "tokens_per_s": tokens / res["duration_s"], "j_per_token": res["energy_j"] / tokens,
                        "mj_per_token": 1000 * res["energy_j"] / tokens, "idle_power_w": idle_w, "weights_gb_allocated": weights_gb,
                        **res, "gpu": gpu_info, "timestamp": now_iso(), "env": env_info() if r == 0 else None})
                    # THROTTLED in this printout means the GPU slowed itself down from heat. Treat that run with care.
                    print(f"  c={c} r={r}: {tokens / res['duration_s']:.0f} tok/s, {1000 * res['energy_j'] / tokens:.0f} mJ/tok"
                          f"{'  THROTTLED' if res['thermal_throttled'] else ''}", flush=True)
            # Hand the GPU memory back before loading the next model, or the next one might not fit.
            del model, tok
            free(None)
    mon.stop()


if __name__ == "__main__":
    main()
