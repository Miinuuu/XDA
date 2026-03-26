#!/usr/bin/env python3
"""Exhaustive FP16 accuracy comparison: FMA vs original for all 9 functions."""
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from nli_eda import optimize_eda, get_function, get_domain
from gen_eda_mem import fp32_to_fp16_bits, _fp16_bits, _bits_to_fp16, _fp_adder, hw_eda_forward_scalar
from gen_eda_mem_4s import hw_eda_forward_fma

FUNCTIONS = ['silu', 'gelu', 'exp', 'sigmoid', 'tanh', 'hardswish', 'mish', 'rsqrt', 'reciprocal']
TAU = 2**(-14)

def build_config(func_name, max_lut=254, max_k=5):
    config = optimize_eda(func_name, max_lut=max_lut, max_k=max_k, device='cpu', verbose=False)
    domain = get_domain(func_name)
    original_bins = {}
    for e in range(1, 31):
        original_bins[(0, e)] = (2.0**(e-15), 2.0**(e-14))
        original_bins[(1, e)] = (-(2.0**(e-14)), -(2.0**(e-15)))
    original_bins[(0, 0)] = (2.0**(-24), 2.0**(-14))
    original_bins[(1, 0)] = (-(2.0**(-14)), -(2.0**(-24)))
    bin_map, clipped = {}, set()
    for i, (bs, be, s, ev) in enumerate(config.bins):
        bin_map[(s, ev)] = i
        orig = original_bins.get((s, ev))
        if orig and (abs(bs-orig[0])>1e-10 or abs(be-orig[1])>1e-10):
            clipped.add((s, ev))
    first_idx, last_idx = 0, len(config.lut_values)-1
    rom = []
    for addr in range(64):
        s, ev = (addr>>5)&1, addr&0x1F
        if ev==31:
            base = last_idx if s==0 else first_idx
            entry = (1<<12)|(base&0x1FF)
        elif (s,ev) in bin_map:
            idx = bin_map[(s,ev)]
            k = config.k_alloc[idx]
            base = int(config.base_offsets[idx].item())
            cl = 1 if (s,ev) in clipped else 0
            entry = (cl<<12)|((k&7)<<9)|(base&0x1FF)
        else:
            base = last_idx if s==0 else first_idx
            entry = (1<<12)|(base&0x1FF)
        rom.append(entry)
    lut = config.lut_values.cpu().float().tolist()
    return rom, lut, domain

print(f"{'Function':12s} {'Total':>7s} {'Differ':>7s} {'%':>6s} {'FMA+':>6s} {'Orig+':>6s} {'Ratio':>6s} {'MeanOld':>10s} {'MeanFMA':>10s} {'MaxOld':>10s} {'MaxFMA':>10s}")
print("-"*105)

for fn in FUNCTIONS:
    func = get_function(fn)
    rom, lut, domain = build_config(fn)
    lo, hi = domain

    diff_count = fma_better = orig_better = total = 0
    errs_old, errs_fma = [], []

    for bits in range(65536):
        xf = np.frombuffer(np.array([bits], dtype=np.uint16).tobytes(), dtype=np.float16)[0]
        x = float(xf)
        if np.isnan(x) or np.isinf(x) or x < lo or x > hi:
            continue
        y_old = hw_eda_forward_scalar(bits, rom, lut)
        y_new = hw_eda_forward_fma(bits, rom, lut)
        if np.isnan(y_old) or np.isnan(y_new):
            continue
        total += 1
        y_true = float(func(torch.tensor([x])).item())
        denom = max(abs(y_true), TAU)
        eo = abs(y_old - y_true) / denom
        en = abs(y_new - y_true) / denom
        errs_old.append(eo)
        errs_fma.append(en)
        ob = fp32_to_fp16_bits(y_old)
        nb = fp32_to_fp16_bits(y_new)
        if ob != nb:
            diff_count += 1
            if en < eo: fma_better += 1
            elif eo < en: orig_better += 1

    eo_arr = np.array(errs_old)
    en_arr = np.array(errs_fma)
    ratio = f"{fma_better/max(orig_better,1):.1f}:1" if orig_better > 0 else f"{fma_better}:0"
    print(f"{fn:12s} {total:7d} {diff_count:7d} {100*diff_count/total:5.1f}% {fma_better:6d} {orig_better:6d} {ratio:>6s} "
          f"{eo_arr.mean()*1e4:9.2f} {en_arr.mean()*1e4:9.2f} {eo_arr.max()*1e4:9.1f} {en_arr.max()*1e4:9.1f}")
