# Subnormal rsqrt Task-Level Isolation (ICCAD'26 Rebuttal, D-Q4)

**Question (Reviewer D-Q4):** task-level accuracy impact of XDA's large worst-case rsqrt
error in the FP16 subnormal (e=0) range.

## Method — single-variable isolation
Two arms differ in **exactly one thing**: how the RMSNorm rsqrt handles FP16-subnormal
inputs (`0 < |variance+eps| < 2^-14 = 6.1e-5`).

| arm | rsqrt | everything else (silu/softmax/normal-range rsqrt) |
|---|---|---|
| `nli_eda` (control) | full XDA LUT (incl. subnormal) | XDA |
| `nli_eda_e0exact` | XDA LUT, **but e=0 inputs bypassed to exact rsqrt** | XDA (identical) |

Model: Llama-3.1-8B-Instruct (the only one of the 5 paper models whose profiling shows
subnormal rsqrt inputs — 3,472 calls, 0.16% of 2.1M, all in the first layer L0).
Eval: 8 zero-shot tasks (arc_challenge, arc_easy, boolq, piqa, hellaswag, openbookqa,
lambada_openai, winogrande), lm-eval-harness, 500 examples/task, seed 42, batch 16.

## Results (acc %, 500 ex/task)

| task | nli_eda (full XDA) | e0exact (subnormal→exact) | bypass Δ |
|---|---|---|---|
| arc_challenge | 55.40 | 54.40 | -1.00 |
| arc_easy | 78.60 | 79.20 | +0.60 |
| boolq | 84.20 | 84.40 | +0.20 |
| piqa | 83.40 | 83.40 | 0.00 |
| hellaswag | 70.40 | 70.40 | 0.00 |
| openbookqa | 44.60 | 45.00 | +0.40 |
| lambada_openai | 73.20 | 73.60 | +0.40 |
| winogrande | 73.80 | 74.20 | +0.40 |
| **8-task avg** | **70.450** | **70.575** | **+0.125** |

- **8-task average bypass effect = 0.13 pp** (same GPU cuda:0, so attributable purely to the e=0 handling).
- **Cross-GPU noise floor = 0.000 pp**: `nli_eda` rerun on cuda:1 is byte-identical to cuda:0 on
  all 8 tasks (`nli_eda_cuda1_noisefloor.json`) → eval is deterministic, so the 0.13 pp is a real
  (precisely measured) effect, not run noise.
- Per-task changes are <=1.0 pp in either direction: the subnormal error acts as a negligible
  random perturbation that flips a few borderline predictions, not systematic degradation.

**Conclusion:** isolating the 3,472 subnormal rsqrt calls changes the 8-task average by only
0.13 pp -> operationally negligible (rebuttal Sec. 6(iii)).

## Files
- `nli_eda_e0exact_cuda0.json` — treatment (e=0 bypassed), cuda:0
- `nli_eda_cuda0.json` — control (full XDA), cuda:0 (same-GPU pair for the clean delta)
- `nli_eda_cuda1_noisefloor.json` — control rerun on cuda:1 (noise-floor check, identical)

## Reproduce
The `nli_eda_e0exact` mode and the e=0 bypass live in `sw/llm_eval.py`
(`_EDA_OPTS['e0_exact']`, applied in `eda_activation` for func `rsqrt`). Env knobs:
`EDA_ZEROSHOT_ONLY=1` (zero-shot tasks only, skip PPL), `EDA_LIMIT=500` (examples/task).

```bash
EDA_LIMIT=500 EDA_ZEROSHOT_ONLY=1 HF_HOME=<cache> \
  python -u sw/llm_eval.py --model meta-llama/Llama-3.1-8B-Instruct \
  --mode nli_eda_e0exact --device cuda:0 --save-dir out --seed 42
# control: same command with --mode nli_eda
```
