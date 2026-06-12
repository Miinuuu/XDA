#!/usr/bin/env python3
"""Re-derive the rebuttal Sec. 6(iv) task-level numbers from results/table1_*.json.

The JSONs under results/ are the canonical lm-eval outputs behind the paper's
Table 2 (five models x {baseline, XDA (internal name nli_eda), NLI, NN-LUT16,
NN-LUT256}; produced by sw/llm_eval.py). From the shipped files alone this
script reproduces the numbers cited in the rebuttal (Sec. 6(iv)) and printed
in the paper (abstract, Sec. 4.4):

  XDA : 0.30 pp mean absolute deviation vs. the FP16 baseline over the
        40 zero-shot task-model pairs; largest single deviation 1.66 pp
        (WinoGrande, Qwen2.5-7B)
  NLI : 0.36 pp mean absolute deviation over the same 40 pairs

Usage: python3 derive_task_mad.py   (stdlib only)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

MODELS = [
    'Qwen_Qwen2.5-0.5B-Instruct',
    'meta-llama_Llama-3.2-3B-Instruct',
    'Qwen_Qwen2.5-7B-Instruct',
    'meta-llama_Llama-3.1-8B-Instruct',
    'Qwen_Qwen3-30B-A3B',
]
VARIANTS = [('nli_eda', 'XDA'), ('nli', 'NLI')]


def zeroshot_detail(model, variant):
    path = os.path.join(RESULTS, f'table1_{model}_{variant}.json')
    with open(path) as f:
        d = json.load(f)
    (cfg,) = d['table1'].keys()
    return d['table1'][cfg]['zeroshot_detail']


def main():
    for variant, label in VARIANTS:
        devs = []  # (abs deviation in pp, task, model)
        for model in MODELS:
            base = zeroshot_detail(model, 'baseline')
            mod = zeroshot_detail(model, variant)
            assert base.keys() == mod.keys()
            for task in sorted(base):
                devs.append((abs(mod[task] - base[task]), task, model))
        mad = sum(d for d, _, _ in devs) / len(devs)
        worst = max(devs)
        print(f'{label:4s} vs. baseline: {len(devs)} zero-shot task-model '
              f'pairs, mean |dev| = {mad:.4f} pp, '
              f'max |dev| = {worst[0]:.4f} pp ({worst[1]}, {worst[2]})')


if __name__ == '__main__':
    main()
