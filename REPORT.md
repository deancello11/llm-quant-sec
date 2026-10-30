# Does quantization weaken LLM safety? (REPORT skeleton)

> Fill this in after `python -m src.analyze`. Delete these hints. Report null results as findings.

## 1. Motivation and research questions
Why quantize (speed, energy, memory: cite your llm-inference-bench numbers). The four research questions from the README.

## 2. Setup
- Models, exact HF revisions, quantization methods and settings (group size 128, 4-bit, calibration: 512 C4 docs, seed 1234; hash in each `calibration_manifest.json`).
- Datasets with versions (`data/DATASET_VERSIONS.json`), jailbreak templates (`configs/jailbreak_templates.yaml`).
- Judge: Llama Guard 3 8B, bf16. Decoding: greedy, 512 tokens. Hardware and library versions (`outputs/_runs.jsonl`).

## 3. Results
- Table: `results/summary.csv` (ASR with CI for every model x precision, refusal, over-refusal, truncation).
- Figures: `asr_by_precision.png`, `asr_direct_vs_template.png`, `pareto_c1.png`, `pareto_c32.png`, `asr_vs_overrefusal.png`, `method_comparison.png`.
- Paired tests: `results/pairwise.csv` (Holm-corrected). Say how many comparisons are significant, and state the minimum detectable difference.
- Hypotheses: paste `results/hypotheses.md`. Each one gets a verdict.

## 4. Judge validation
Numbers from `validation/agreement.json`: agreement, Cohen's kappa, PPV, NPV, for the judge and the refusal heuristic. Note disagreements you saw.

## 5. Limitations
Three models. Static attacks only, no adaptive attacks. One judge. About 1,500 correlated prompts per cell, so small effects are invisible.
Template prompts share a behavior, so effective sample size is nearer 300. HF Transformers kernels only. GPU-only energy.
Batched fp16 greedy decoding is not bit-identical to unbatched. Self-quantized checkpoints may differ from community ones. Heuristic refusal detection.

## 6. Takeaways for practitioners
What to do given your results: which precision/method is safe to ship, what to re-test after quantizing, what the speed/energy saving cost.

## 7. Reproduction
See README.
