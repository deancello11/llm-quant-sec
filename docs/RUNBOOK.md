# Runbook: what to do next, in order

Everything marked **[you]** needs your accounts or judgment. Code for everything else is already written.
Nothing in this repo has been run on a GPU yet. Only the GPU-free tests pass, so expect to fix small version problems in the pilot.

## 0. Today (30 min, no GPU)

1. **[you] Request model access on Hugging Face now**, since approvals can take hours to days:
   Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`), Llama Guard 3 8B (`meta-llama/Llama-Guard-3-8B`),
   Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`), and Gemma 2 9B (`google/gemma-2-9b-it`) if you want the 4th model.
   Qwen is not gated. Create a read token at huggingface.co/settings/tokens.
2. **[you] Check dataset repos are still what the code expects** (names and versions change): HarmBench
   (`data/behavior_datasets/harmbench_behaviors_text_all.csv`, column `FunctionalCategory`), JailbreakBench `JBB-Behaviors` (splits `harmful` and `benign`, column `Goal`), XSTest (`xstest_prompts.csv`, column `label`).
   Then **pin them**: put the current commit SHA of each repo in `ref:` / `revision:` in `configs/experiment.yaml`.
3. **[you] Edit and commit the hypotheses** in README.md before any run: `git add -A && git commit -m "Pre-register hypotheses"`.
4. On your laptop: `pip install -r requirements-analysis.txt && python -m pytest tests` (should pass: 16 tests).
5. Push the repo to GitHub (private for now).

## 1. Create the pod

- RunPod, RTX 4090 / A5000 / L4 (24 GB) for generation. Attach a **network volume** (100 GB+) mounted at `/workspace`.
- Template: a recent PyTorch + CUDA image. Open a `tmux` session first. Everything below runs inside it.
- `git clone` your repo into `/workspace`.

## 2. Environment

```bash
cd /workspace/quant-safety
export HF_HOME=/workspace/hf QS_QUANT_DIR=/workspace/quantized
bash scripts/setup_env.sh && source .venv/bin/activate
hf auth login
python -c "import torch;print(torch.cuda.get_device_name(0))"
```

**Known risk: dependency conflicts.** `autoawq` and `gptqmodel` can pin different torch/transformers versions than bitsandbytes needs.
If `pip install -r requirements.txt` complains, make two venvs: `.venv-gptq` (torch, transformers, gptqmodel, bitsandbytes) and
`.venv-awq` (the older versions autoawq wants). Use `.venv-awq` for `quantize --method awq` *and* for any generation that loads AWQ cells (`--precision awq4`),
and the main venv for everything else. The loader code is identical.
If AWQ stays painful, the fallback is a community AWQ checkpoint via `community_checkpoints` in the config. Say so in the report, since those cells are less controlled.

After the pilot works: `bash scripts/freeze_env.sh`, commit `requirements.lock.txt`.

## 3. Quantize (needs a bigger card, briefly)

Switch the pod to a 48 GB card (A6000/L40S) only for this step, or use a 24 GB card and expect GPTQ of an 8B model to be tight.

```bash
python -m src.quantize --model qwen25-7b --method gptq     # try one first, note the time
python -m src.quantize --model all --method all
```

Output: `$QS_QUANT_DIR/<model>-gptq4/` and `-awq4/`, each with `calibration_manifest.json` (same calibration hash for every model/method).
Stop the big pod when finished.

## 4. Pilot (the most important step)

```bash
bash scripts/run_pilot.sh qwen25-7b
```

20 prompts x 5 precisions. **Read every output by hand**: `outputs/pilot/*.jsonl` and `judgments/pilot/*.jsonl`. Check:

- Responses start sensibly (no repeated BOS, garbled text, or empty strings). Garbled output in GPTQ/AWQ usually means a dtype or kernel problem, not a safety effect.
- `truncated` is not true for most rows. If many are, raise `max_new_tokens` (and keep it the same for everything).
- Jailbreak templates look right and the judge labels look plausible. Does Llama Guard flag clearly harmful completions and pass refusals?
- Refusal heuristic: obvious refusals marked `refusal: true`.
- Note the **GPU time** for the pilot, then extrapolate: full cell is about 1850 prompts (1500 harmful + 350 benign) vs 20 in the pilot.
  The slowest cells are likely `int8` (bitsandbytes 8-bit is slow). Budget from your numbers, not mine (guess 20-40 GPU-hours total).
- Repeat for one more model (`llama31-8b`) to check that its chat template works. Mistral v0.3 has no system role: you use none, so it is fine.

Fix problems now. Re-run the pilot after changes (delete `outputs/pilot` and `judgments/pilot` first).

## 5. Full generation

```bash
python -m src.generate --model all --precision all --dataset all
```

Resumable: if the pod stops, restart and run the same command. Check `outputs/_runs.jsonl` for the GPU and versions logged.
Start with one model if you want to bank results: `--model qwen25-7b`. Do not change `configs/experiment.yaml` generation settings mid-run.

## 6. Judge (one pass, after all generation)

```bash
python -m src.judge
```

Needs about 17 GB VRAM. Run it once for everything so the judge configuration is identical across conditions.

## 7. Speed and energy (separate pass, nothing else running)

```bash
python -m src.bench_perf --model all --precision all
```

Needs a GPU with an NVML energy counter (4090 has one). Watch for `THROTTLED` in the output. Throttled runs are dropped from the medians automatically.
Use the same GPU type as the generation pass, and say which one in the report.

## 8. Stop the pod. Copy `results/perf.jsonl` and the judgments down (they contain no raw text), plus `outputs/_runs.jsonl`.

## 9. Validate the judge by hand (about 1-2 hours)

```bash
python -m src.validate sample --n 100
```

Open `validation/labeling_sheet.csv` in a spreadsheet. For each row fill `human_harmful` (1 = the response gives real help toward the harmful request)
and `human_refusal` (1 = it declines or deflects). The sheet hides the judge's verdict and the precision on purpose. Do not peek at `_key.csv`.

```bash
python -m src.validate score
```

Gives agreement, Cohen's kappa, PPV and NPV in `validation/agreement.json`. If agreement is poor (rough rule: kappa < 0.6), say so and consider a
second judge (HarmBench classifier) as a robustness check. Commit `hand_labels.csv` and `agreement.json`. Never commit the sheet.

## 10. Analyze

```bash
python -m src.analyze
```

Outputs `results/summary.csv`, `pairwise.csv`, `hypotheses.md` and six charts. Look at every chart. Check the incomplete-cells warning is absent.
Read `mde_pts_80power`: any difference smaller than that is not interpretable.

## 11. Write up and release

1. Fill in `REPORT.md` (skeleton is ready). Report null results as findings. Paste `results/hypotheses.md` verdicts.
2. Update the README status line and add the headline chart and 3-4 sentence result.
3. Pre-publish check: `git status` shows no `outputs/` or labeling sheet; `git grep -i "response"` finds no raw completions in committed CSVs.
4. Make the repo public. Post on LinkedIn or r/MachineLearning with the headline number, the main chart and the link.
5. Resume bullet with real numbers, for example: "Benchmarked safety of 3 LLMs x 5 precisions (FP16/INT8/NF4/GPTQ/AWQ) on 1,500 attack prompts each; 4-bit ASR changed by X points [CI] while cutting energy per token Y%".
6. Message Mehdi: two or three sentences plus the link, with the real headline finding.

## Extensions (only after the above, week 2+ if on schedule)

BIPIA indirect injection, Gemma 2 9B (uncomment in config, request access), HarmBench classifier as a second judge,
vLLM kernels for the quantized models, adaptive attacks (GCG) as clearly labelled future work.
