# Your step-by-step plan ($23 RunPod budget)

Tick boxes as you go. "Laptop" = your Windows VS Code terminal. "Pod" = the RunPod terminal. RunPod's web pages change, so if a button is named
differently, look for the closest one. Prices below are rough: always trust the live price shown on RunPod.

**Golden rules**
- Stop (not just close the tab on) a pod the moment a job ends. A running pod bills every minute.
- Never paste your HF token into a chat, file or the repo. Only type it into `hf auth login`.
- Do not change `configs/experiment.yaml` generation settings after the first real run.
- Write down GPU-hours and dollars after every phase in the table at the bottom.

---------------------------------------------------------------------------------------------------------

## PHASE 0: Finish on the laptop (about 1 hour, free)

Done already: HF access, datasets pinned, tests pass.

- [ ] 0.1 Open `README.md`. Under "Hypotheses", read H1-H5. Rewrite each in your own words (keep them testable and keep the +-3 point margins unless you
      have a reason). Delete the `> **DRAFT...` block.
- [ ] 0.2 Decide the budget cuts NOW, before committing (changing the design after seeing results is not allowed):
      - Default (recommended): keep the full design, but run models one at a time and stop when money runs out.
      - If you want a safer budget: edit `configs/jailbreak_templates.yaml` to keep only 2 templates (delete the other two blocks), and change "4 fixed
        static jailbreak templates" and "1500 prompts per cell" in README.md (2 templates: 300 behaviors x 3 = 900 prompts per cell).
      - Also fine: drop `int8` from `precisions:` in the config if the pilot shows it is far too slow (decide after the pilot, but then re-commit and say so
        in the report).
- [ ] 0.3 In the VS Code terminal: `python -m pytest tests -q` (expect 16 passed, or the new count if you changed templates).
- [ ] 0.4 Commit:
      ```
      git add -A
      git commit -m "Pre-register hypotheses and pin datasets"
      ```
      (If git asks who you are: `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`, then repeat.)
- [ ] 0.5 Create a **private** GitHub repo called `quant-safety` at github.com/new (no README, no .gitignore, no license: the repo already has them).
- [ ] 0.6 Push (GitHub shows these commands after creating the repo; replace YOURUSER):
      ```
      git branch -M main
      git remote add origin https://github.com/YOURUSER/quant-safety.git
      git push -u origin main
      ```
      GitHub may open a browser login. If it asks for a password, it wants a Personal Access Token (github.com/settings/tokens, scope `repo`).
- [ ] 0.7 Check the push: refresh the repo page, confirm `outputs/` does not exist and nothing secret is committed (`git grep -i hf_` in the terminal should print nothing).

## PHASE 1: Create the pod (about 30 min)

- [ ] 1.1 Go to https://www.runpod.io and log in. Confirm your balance (Billing page). If you see a "spend limit" setting, leave it on.
- [ ] 1.2 **Create a network volume first.** Menu: Storage -> New Network Volume.
      - Pick a datacenter that has the 24 GB GPUs you want (4090 / A5000 / L4 / RTX 3090 all fine). The volume can only attach to pods in the same datacenter.
      - Size **70 GB** (about $5/month). Name it `quant-safety`. Create.
      - Why 70 GB: Qwen+Llama+Mistral FP16 about 45 GB, quantized copies about 30 GB, judge about 16 GB. If you run out of space later, delete one
        FP16 model folder in `/workspace/hf` that you are finished with.
- [ ] 1.3 Menu: Pods -> Deploy.
      - Select the same datacenter as the volume, attach the `quant-safety` volume.
      - GPU: **RTX 4090 or A5000 (24 GB)**, cheapest "Community" option is fine. Do NOT pick a 48 GB card.
      - Template: the latest "RunPod PyTorch" template (PyTorch 2.x, CUDA 12.x, Ubuntu).
      - On-demand (not Spot), because spot pods can be killed mid-run. (Spot is cheaper; the code is resumable, so it is acceptable if you check often.)
      - Container disk 30 GB. Make sure the volume mounts at `/workspace`.
      - Deploy. Wait until the status shows Running.
- [ ] 1.4 Click Connect -> choose **Web Terminal** (simplest) or Jupyter Lab -> terminal. A Linux shell opens.
- [ ] 1.5 Start tmux so jobs survive a closed browser tab: `tmux new -s qs`. To reattach later: `tmux attach -t qs`. Detach: Ctrl+B then D.
- [ ] 1.6 `nvidia-smi` should show your 24 GB GPU. Note the GPU name for the report.

## PHASE 2: Set up the environment (about 30-60 min)

- [ ] 2.1 In tmux on the pod:
      ```
      cd /workspace
      git clone https://github.com/YOURUSER/quant-safety.git
      cd quant-safety
      ```
      (Private repo: use `https://YOURUSER:GITHUB_TOKEN@github.com/YOURUSER/quant-safety.git`, with a GitHub token you will delete afterwards.)
- [ ] 2.2 `export HF_HOME=/workspace/hf QS_QUANT_DIR=/workspace/quantized`
      Add those two lines to `~/.bashrc` too so every new shell has them (also `/workspace` persists, `~` may not: so re-export after any pod restart).
- [ ] 2.3 `bash scripts/setup_env.sh && source .venv/bin/activate`
      Expect a long install. Watch for errors, in particular from `autoawq` and `gptqmodel`.
- [ ] 2.4 If it fails on autoawq/gptqmodel conflicts: make the two-environment setup from `docs/RUNBOOK.md` section 2 (`.venv-gptq` and `.venv-awq`).
      Time-box this to about 1 hour of pod time. If AWQ still fails, skip AWQ: remove `awq4` from `precisions:` in the config, recommit, and say so
      in the report (4-bit still has NF4 and GPTQ).
- [ ] 2.5 `hf auth login` (paste your token, answer `n` to git credentials).
- [ ] 2.6 Verify: `python -c "import torch;print(torch.cuda.get_device_name(0), torch.__version__)"`
- [ ] 2.7 Run `python -m pytest tests -q` on the pod too (should still pass).

## PHASE 3: Pilot on Qwen (about $1-2, the most important phase)

Cheapest model first: no gating, so nothing blocks you.

- [ ] 3.1 Quantize Qwen (24 GB card). Time it with `time`:
      ```
      time python -m src.quantize --model qwen25-7b --method gptq
      time python -m src.quantize --model qwen25-7b --method awq
      ```
      If GPTQ runs out of memory on 24 GB: lower `calibration.n_samples` is NOT allowed (it must be the same for every model). Instead tell me the error:
      the fix is switching the pod briefly to a 48 GB card.
- [ ] 3.2 Check `ls /workspace/quantized/` shows `qwen25-7b-gptq4` and `qwen25-7b-awq4` and each has `calibration_manifest.json`.
- [ ] 3.3 `bash scripts/run_pilot.sh qwen25-7b` (generates 20 prompts x 5 precisions, then judges them). Time it.
- [ ] 3.4 Read the outputs by hand (on the pod, e.g. `python -c` or Jupyter, or copy `outputs/pilot` down to your laptop; never commit it):
      - [ ] responses are fluent in all 5 precisions. Garbled GPTQ/AWQ text = a kernel/dtype bug to fix, not a result.
      - [ ] very few `truncated: true`. If most are truncated, raise `generation.max_new_tokens` ONCE (for everything) and re-run the pilot.
      - [ ] the 4 jailbreak templates render correctly.
      - [ ] Llama Guard marks clearly harmful completions "unsafe" and clear refusals "safe".
      - [ ] the `refusal` field matches obvious refusals.
- [ ] 3.5 Write down pilot GPU time. **Budget check:** the full grid is about 1,850 prompts per cell (1,500 harmful + 350 benign) vs 20 in the pilot.
      Full-cell time is roughly pilot_generation_time x 92.5 (1850 / 20), already summed over the 5 precisions. That is the cost for ONE model.
      Multiply by 3 models, add judging (assume +20%) and perf (about 2 hours), then multiply by the pod's $/hour. Compare with your remaining balance.
      Pilots are dominated by model loading, so this overestimates. Treat it as an upper bound.
- [ ] 3.6 If the estimate exceeds what you can afford, apply cuts in this order and re-commit with the reason: (a) 2 templates, (b) drop `int8`,
      (c) drop Mistral. Never cut mid-run.
- [ ] 3.7 Fix any bugs. Commit and push from the pod, or edit on the laptop, push, and `git pull` on the pod.
      After code changes delete the pilot data: `rm -rf outputs/pilot judgments/pilot`.
- [ ] 3.8 **Stop the pod.** (Pods page -> Stop.) The volume keeps the models and quantized checkpoints.

## PHASE 4: Full run, Qwen first (bank a complete result)

- [ ] 4.1 Start the pod again, attach the same volume, reconnect, `tmux attach -t qs` (or `tmux new -s qs`), then:
      ```
      cd /workspace/quant-safety && source .venv/bin/activate
      export HF_HOME=/workspace/hf QS_QUANT_DIR=/workspace/quantized
      git pull
      ```
- [ ] 4.2 Generate: `python -m src.generate --model qwen25-7b --precision all --dataset all`
      If the pod stops, restart and run the identical command: finished prompts are skipped.
- [ ] 4.3 Judge only Qwen: `python -m src.judge --only qwen25-7b`
      (Judge settings are identical regardless of when you run it, so judging per model is fine.)
- [ ] 4.4 Back up small results to the laptop: `judgments/` (not `judgments/pilot`), `outputs/_runs.jsonl`. Use the RunPod web file browser, or
      `runpodctl send`, or `scp`. Also copy `outputs/` somewhere private as a backup (never commit it).
- [ ] 4.5 Check balance. Write cost in the table. Decide: continue with Llama?
- [ ] 4.6 **Stop the pod** between models if you will not continue immediately.

## PHASE 5: Llama and Mistral

For each of `llama31-8b` then `mistral-7b-v03`, repeat:

- [ ] 5.1 Quantize: `python -m src.quantize --model MODEL --method all`
- [ ] 5.2 Short pilot: `bash scripts/run_pilot.sh MODEL`, read 20 outputs by hand (chat template works? Llama especially). Delete pilot dirs.
- [ ] 5.3 Full generation: `python -m src.generate --model MODEL --precision all --dataset all`
- [ ] 5.4 Judge: `python -m src.judge --only MODEL`
- [ ] 5.5 Back up judgments; note cost; stop the pod if pausing.
- [ ] 5.6 Check the calibration hash is identical across all models: `cat /workspace/quantized/*/calibration_manifest.json | grep -i hash`

If money runs low, stop after Llama. Two models is still a valid study (but H4 then compares two families and must be reported that way: edit the README before analysis and say so).

## PHASE 6: Speed and energy (about 1-2 hours of GPU, last)

- [ ] 6.1 Nothing else running on the GPU: `nvidia-smi` shows no other processes.
- [ ] 6.2 `python -m src.bench_perf --model all --precision all` (or only the models you finished). Same GPU type as generation.
- [ ] 6.3 If outputs say `THROTTLED`, those runs are dropped automatically; if too many are dropped, re-run later.
- [ ] 6.4 Copy `results/perf.jsonl` to the laptop. **Stop the pod. Then delete the network volume** once everything is backed up (it bills monthly otherwise).
      Before deleting: confirm on the laptop you have `judgments/`, `results/perf.jsonl`, `outputs/_runs.jsonl`, `data/DATASET_VERSIONS.json`, and a private copy of `outputs/`.
- [ ] 6.5 Revoke the GitHub token you used for cloning, if you made one.

## PHASE 7: Validate the judge (laptop, 1-2 hours, free)

- [ ] 7.1 `python -m src.validate sample --n 100`
- [ ] 7.2 Open `validation/labeling_sheet.csv` in Excel. For each row fill `human_harmful` (1 = the response really helps with the harmful request, 0 = not)
      and `human_refusal` (1 = it declines or deflects). Do not open `_key.csv`. Label honestly even if it feels slow.
- [ ] 7.3 `python -m src.validate score` -> `validation/agreement.json`.
- [ ] 7.4 If Cohen's kappa is below 0.6, say so in the report and treat all ASR numbers as judge-dependent. (A second judge is an extension.)
- [ ] 7.5 Commit `validation/hand_labels.csv` and `validation/agreement.json`. NEVER commit `labeling_sheet.csv`.

## PHASE 8: Analyze (laptop)

- [ ] 8.1 `python -m src.analyze`
- [ ] 8.2 Confirm there is no "incomplete cells" warning. If a model was skipped, the README/report must say so.
- [ ] 8.3 Look at all six charts in `results/`. Read `results/summary.csv`, `results/pairwise.csv`, `results/hypotheses.md`.
- [ ] 8.4 Read the `mde_pts_80power` column. Any difference smaller than that is not interpretable.
- [ ] 8.5 Come back to me with these files and I will help check the numbers and draft the write-up.

## PHASE 9: Write up and release

- [ ] 9.1 Fill in `REPORT.md` with real numbers. Paste the hypothesis verdicts. Null and failed results are findings: say them plainly.
- [ ] 9.2 Update the README status line and add the headline chart plus 3-4 sentences of results.
- [ ] 9.3 Pre-publish checks: `git status` (no `outputs/`, no labeling sheet), `git grep -i "token"`, `git grep hf_`.
- [ ] 9.4 Make the repo public (GitHub -> Settings -> Danger Zone -> Change visibility).
- [ ] 9.5 Post on LinkedIn (and r/MachineLearning if you want) with the headline number, the main chart and the link.
- [ ] 9.6 Message Mehdi: 2-3 sentences, the real finding, the link.
- [ ] 9.7 Resume bullet with real numbers and CIs. Delete your HF token and RunPod volume if you have not already.

---------------------------------------------------------------------------------------------------------

## Cost log (fill in)

| Phase | GPU | Hours | Dollars | Balance after |
|---|---|---|---|---|
| Start | | | | $23.00 |
| Volume (per month) | | | | |
| Phase 3 pilot | | | | |
| Phase 4 Qwen | | | | |
| Phase 5 Llama | | | | |
| Phase 5 Mistral | | | | |
| Phase 6 perf | | | | |

## If something breaks

- `403`/gated on model download: your approval or token is missing (`hf auth whoami`).
- CUDA out of memory: the code halves batch size automatically; if a quantize step fails, tell me the exact error.
- Pod disappeared / spot interruption: restart, re-export the two env vars, rerun the same command.
- Anything else: copy the last 30 lines of the error and ask.
