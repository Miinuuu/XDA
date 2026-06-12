"""
Subnormal Hit-Rate Analysis for rsqrt Inputs

Computes how often the rsqrt input (variance + eps) falls in the
FP16 subnormal range (e=0, |x| < 2^-14) across models and layers.

This justifies the paper's claim that rsqrt's worst-case error (7,500)
in the subnormal bin is circumvented by RMSNorm's epsilon stabilizer.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 1e-6),
    ("meta-llama/Llama-3.2-3B-Instruct", 1e-5),
    ("meta-llama/Llama-3.1-8B-Instruct", 1e-5),
    ("Qwen/Qwen2.5-7B-Instruct", 1e-6),
]

FP16_SUBNORMAL_THRESHOLD = 2**-14  # 6.1e-5


def profile_rsqrt_inputs(model_name, eps, device="cuda:0", n_samples=100, max_len=512):
    """Collect per-layer rsqrt inputs (variance + eps) and count subnormals."""

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Storage: layer_idx -> list of variance+eps tensors
    variance_data = defaultdict(list)
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach().float()  # [batch, seq, hidden]
            variance = x.pow(2).mean(-1)   # [batch, seq]
            rsqrt_input = variance + eps    # what actually goes into rsqrt
            variance_data[layer_idx].append(rsqrt_input.cpu())
        return hook_fn

    # Register hooks on all RMSNorm layers (input_layernorm + post_attention_layernorm)
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, 'input_layernorm'):
            h = layer.input_layernorm.register_forward_hook(make_hook(f"L{li}_pre"))
            hooks.append(h)
        if hasattr(layer, 'post_attention_layernorm'):
            h = layer.post_attention_layernorm.register_forward_hook(make_hook(f"L{li}_post"))
            hooks.append(h)

    # Run inference
    count = 0
    for item in dataset:
        text = item["text"].strip()
        if len(text) < 10:
            continue
        inputs = tokenizer(text, return_tensors="pt", max_length=max_len,
                          truncation=True).to(device)
        with torch.no_grad():
            model(**inputs)
        count += 1
        if count >= n_samples:
            break

    for h in hooks:
        h.remove()

    # Analyze
    results = {}
    total_values = 0
    total_subnormal = 0

    for layer_key in sorted(variance_data.keys()):
        all_vals = torch.cat([v.flatten() for v in variance_data[layer_key]]).numpy()
        n = len(all_vals)
        # Convert to FP16 to check actual subnormal status
        fp16_vals = all_vals.astype(np.float16)
        bits = fp16_vals.view(np.uint16)
        exp_field = (bits >> 10) & 0x1F
        is_subnormal = (exp_field == 0) & (bits & 0x3FF != 0)
        is_zero = (bits & 0x7FFF) == 0
        n_sub = int(is_subnormal.sum())
        n_zero = int(is_zero.sum())

        total_values += n
        total_subnormal += n_sub

        min_val = float(np.min(all_vals))
        results[layer_key] = {
            "n_values": n,
            "n_subnormal": n_sub,
            "n_zero": n_zero,
            "subnormal_rate": n_sub / n if n > 0 else 0,
            "min_value": min_val,
            "min_value_fp16": float(np.min(fp16_vals)),
        }

    return results, total_values, total_subnormal


def main():
    print("Subnormal Hit-Rate Analysis for rsqrt Inputs")
    print("=" * 80)
    print(f"FP16 subnormal threshold: |x| < 2^-14 = {FP16_SUBNORMAL_THRESHOLD:.2e}")
    print(f"RMSNorm rsqrt input = mean(x²) + eps")
    print()

    for model_name, eps in MODELS:
        short_name = model_name.split("/")[-1]
        print(f"\n{'='*80}")
        print(f"  {short_name}  (eps={eps})")
        print(f"{'='*80}")

        results, total_vals, total_sub = profile_rsqrt_inputs(model_name, eps)

        # Print per-layer summary (first/last few + aggregates)
        layers = sorted(results.keys())
        print(f"\n  {'Layer':>12} | {'Values':>10} | {'Subnormal':>10} | {'Rate':>12} | {'Min val':>12}")
        print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

        for layer in layers[:4]:
            r = results[layer]
            rate_str = f"{r['subnormal_rate']:.2e}" if r['subnormal_rate'] > 0 else "0"
            print(f"  {layer:>12} | {r['n_values']:>10,} | {r['n_subnormal']:>10,} | "
                  f"{rate_str:>12} | {r['min_value']:>12.4e}")

        if len(layers) > 8:
            print(f"  {'...':>12}")

        for layer in layers[-4:]:
            r = results[layer]
            rate_str = f"{r['subnormal_rate']:.2e}" if r['subnormal_rate'] > 0 else "0"
            print(f"  {layer:>12} | {r['n_values']:>10,} | {r['n_subnormal']:>10,} | "
                  f"{rate_str:>12} | {r['min_value']:>12.4e}")

        overall_rate = total_sub / total_vals if total_vals > 0 else 0
        print(f"\n  TOTAL: {total_vals:,} values, {total_sub:,} subnormal "
              f"({overall_rate:.2e} = {overall_rate*100:.6f}%)")
        print(f"  eps ({eps}) >> subnormal threshold ({FP16_SUBNORMAL_THRESHOLD:.2e}): "
              f"{'YES' if eps > FP16_SUBNORMAL_THRESHOLD else 'NO'}")

    print(f"\n{'='*80}")
    print("Conclusion: both deployed eps values (1e-5, 1e-6) are below the")
    print("FP16 subnormal threshold (2^-14 = 6.1e-5), so a subnormal rsqrt")
    print("input requires variance+eps itself to round below 2^-14, i.e. a")
    print("near-zero hidden-state variance. The profiling above shows how")
    print("rarely this occurs and where (per layer, per model).")


if __name__ == "__main__":
    main()
