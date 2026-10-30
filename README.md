# Does quantization weaken LLM safety?

Take the same models at different precisions, send them the same attack and benign prompts, and measure how
safety changes alongside the speed and energy that quantization saves.

This builds on [llm-inference-bench](../llm-inference-bench), which measured speed, energy and accuracy of one model.
This project adds the question that benchmark could not answer: what happens to safety?

**Status: scaffold complete, no experiments run yet.** Results will appear here after the full run. See `docs/RUNBOOK.md` for what to do next.

## Research questions

1. Does attack success rate (ASR) change as precision drops (FP16 -> INT8 -> 4-bit)?
2. Does the quantization method matter (bitsandbytes NF4 vs GPTQ vs AWQ)?
3. Is the effect consistent across model families (Llama, Qwen, Mistral)?
4. Does any apparent safety gain come from over-refusal rather than real robustness?

## Hypotheses (written before running anything)

> **DRAFT. Edit these to say what you actually believe, then `git commit` them before the first generation run.
> The commit timestamp is your proof they came first.** `src/analyze.py` writes `results/hypotheses.md` with these
> exact verdict rules, so if you change a rule, change it in `hypotheses_report()` in the same commit.

All differences are in percentage points of ASR, averaged over the models, relative to FP16 on the same prompts.

- **H1.** INT8 (bitsandbytes) is indistinguishable from FP16: the 95% CI of the pooled difference lies inside +-3 points.
- **H2.** 4-bit ASR is higher than FP16 by at least 3 points on average (pooled over NF4, GPTQ, AWQ), with a CI that excludes 0.
- **H3.** (exploratory, direction only) NF4, which uses no calibration data, loses the most safety. NF4 ASR >= GPTQ ASR and NF4 ASR >= AWQ ASR.
  Motivation: in the earlier benchmark NF4 also lost the most accuracy.
- **H4.** The direction of the 4-bit effect is the same in all three model families. Magnitudes may differ.
- **H5.** Any case where a quantized model has lower ASR than FP16 is accompanied by higher over-refusal on safe XSTest prompts,
  which would make it a refusal-bias effect and not extra robustness.

If a hypothesis fails, it is reported as failed.

## Design

| Axis | Choice |
|---|---|
| Models | Llama 3.1 8B Instruct, Qwen2.5 7B Instruct, Mistral 7B Instruct v0.3 (Gemma 2 9B optional, in `experiment.yaml`) |
| Precisions | FP16 (baseline), INT8 (bitsandbytes), NF4 (bitsandbytes), GPTQ 4-bit, AWQ 4-bit |
| Harmful prompts | HarmBench text behaviors (200 standard), JailbreakBench harmful (100) |
| Attack variants | Direct request, plus 4 fixed static jailbreak templates (`configs/jailbreak_templates.yaml`). 300 behaviors x 5 = 1500 prompts per cell |
| Over-refusal control | XSTest safe prompts (250), JailbreakBench benign (100) |
| Judge | Llama Guard 3 8B, bf16, run as a separate pass; validated against hand labels |
| Decoding | Greedy, `max_new_tokens=512`, each model's official chat template, no system prompt |

Controls for confounds:

- **Calibration.** GPTQ and AWQ are built by `src/quantize.py` from the same 512 generic C4 documents for every model. Community checkpoints are only used if listed under `community_checkpoints`, and such cells are flagged.
- **Chat template.** The tokenizer always comes from the original FP16 repo, so every precision sees byte-identical prompts.
- **Judge precision.** One fixed judge configuration, run after generation, never mixed with generation.
- **Truncation.** `max_new_tokens` is fixed and every record stores `truncated`. The truncation rate per cell is in `results/summary.csv`.
- **Batching.** Prompts are sorted by token length, so every precision gets identical batches. Fp16 batched greedy decoding can still differ slightly from unbatched.
- **Energy.** Measured in a separate pass (`src/bench_perf.py`) on synthetic prompts, with nothing else on the GPU.

## Metrics and statistics

- **ASR** = share of harmful prompts where Llama Guard says the response is unsafe. A stricter variant (`asr_strict`) also requires that the response is not a keyword refusal.
- **Refusal rate** on harmful prompts and **over-refusal** on XSTest, from a transparent keyword check (`src/refusal.py`) whose agreement with human labels is reported.
- **Tokens/s and mJ/token** at batch 1 and batch 32.
- **CIs:** 95% cluster bootstrap, 10,000 resamples, resampling whole behaviors (a behavior and its template copies stay together, because they are not independent).
- **Tests:** exact McNemar on paired prompts, Holm-corrected across each family of comparisons. McNemar treats prompts as independent, so the bootstrap CI is the primary evidence.
- **Power:** with about 1,500 correlated prompts per cell, effects smaller than a few points cannot be told apart from noise. `results/pairwise.csv` reports the smallest detectable difference (`mde_pts_80power`) for every comparison, and it is a lower bound.

## Repo layout

```
configs/experiment.yaml          models, precisions, datasets, seeds, max tokens, stats settings
configs/jailbreak_templates.yaml the fixed static templates
src/quantize.py                  builds GPTQ/AWQ variants with one fixed calibration set
src/load_model.py                (model, precision) -> model, tokenizer
src/prompts.py                   downloads/pins datasets, builds prompt lists
src/generate.py                  resumable generation -> outputs/*.jsonl
src/judge.py                     Llama Guard over outputs -> judgments/*.jsonl
src/validate.py                  hand-label sheet and agreement stats
src/bench_perf.py, src/gpu.py    speed + energy harness (separate pass)
src/analyze.py                   metrics, bootstrap CIs, McNemar + Holm, charts
tests/                           GPU-free tests, including a synthetic end-to-end analysis
docs/RUNBOOK.md                  step-by-step instructions, pod tips, budget
REPORT.md                        write-up skeleton
```

Record schema, one line per prompt in `outputs/`:
`{model, precision, dataset, prompt_id, behavior_id, variant, category, prompt, response, n_tokens, truncated, seed, timestamp, run_id}`.
`run_id` links to `outputs/_runs.jsonl`, which stores GPU, library versions, git commit, checkpoint path and config hash.

## Reproduce

```bash
bash scripts/setup_env.sh && source .venv/bin/activate && hf auth login
python -m src.quantize --model all --method all       # on a 48 GB card
bash scripts/run_pilot.sh qwen25-7b                    # read the outputs by hand before going on
bash scripts/run_all.sh                                # full grid, resumable
python -m src.validate sample --n 100                  # label by hand, then: python -m src.validate score
python -m src.analyze
```

GPU-free tests: `pip install -r requirements-analysis.txt && python -m pytest tests`.

## Responsible handling

This generates harmful completions on purpose, as standard safety research on models run locally.
Raw responses (`outputs/`, `validation/labeling_sheet*.csv`) are gitignored. Only aggregate results, judge labels and prompt IDs are published.
No adaptive attack optimization (GCG etc.) in v1. Only models run by the author are tested.

## Limits (full list in REPORT.md)

Three models, static attacks only, one judge, correlated prompts, GPU-only energy, HF Transformers kernels only (vLLM or llama.cpp kernels may differ).
