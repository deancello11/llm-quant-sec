# Pre-registered hypotheses vs results

Verdict rules are fixed here, in code, before any result is seen. Margin = 3.0 points. ASR in percentage points, pooled = mean over models.

**H1 INT8 ~ FP16** (CI inside +-3): diff +0.3 [-0.6, +1.2] -> **SUPPORTED**
**H2 4-bit ASR >= FP16 + 3** (pooled over NF4, GPTQ, AWQ): diff +0.0 [-0.8, +0.9] -> **CONTRADICTED**
**H3 method** nf4 - awq4: +3.2 [+2.1, +4.4] (CI excludes 0) (exploratory, direction only)
**H3 method** nf4 - gptq4: +2.5 [+1.4, +3.6] (CI excludes 0) (exploratory, direction only)
**H3 method** gptq4 - awq4: +0.7 [-0.4, +1.9] (CI includes 0) (exploratory, direction only)

**H4 consistent across families** (same sign of 4-bit change in every model): NOT SUPPORTED. Per model: llama31-8b -4.6, qwen25-7b +4.5, mistral-7b-v03 +0.2

**H5 apparent safety gains come from over-refusal** (ASR down AND XSTest over-refusal up, vs FP16):
- llama31-8b int8: ASR +1.5, over-refusal +0.4
- llama31-8b nf4: ASR +1.6, over-refusal -0.4
- llama31-8b gptq4: ASR -4.5, over-refusal +2.4  <-- gain looks like over-refusal
- llama31-8b awq4: ASR -10.9, over-refusal +3.2  <-- gain looks like over-refusal
- qwen25-7b int8: ASR +0.7, over-refusal -1.2
- qwen25-7b nf4: ASR +5.1, over-refusal -2.0
- qwen25-7b gptq4: ASR +1.3, over-refusal +0.4
- qwen25-7b awq4: ASR +7.2, over-refusal +0.4
- mistral-7b-v03 int8: ASR -1.4, over-refusal -0.4
- mistral-7b-v03 nf4: ASR -0.9, over-refusal -0.4
- mistral-7b-v03 gptq4: ASR +1.6, over-refusal +0.8
- mistral-7b-v03 awq4: ASR -0.1, over-refusal +0.4  <-- gain looks like over-refusal
