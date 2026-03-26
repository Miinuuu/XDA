"""
EDA-NLI Engine: Forward Pass with Exponent-Direct Addressing
=============================================================
Simulates the 3-stage hardware pipeline:
  Stage 1: Bit Extract → LUT address + interpolation fraction (0 gates)
  Stage 2: SRAM read LUT[g], LUT[g+1]
  Stage 3: y = y0 + t * (y1 - y0)

No comparators, no multipliers for address generation.
Address is derived purely from FP16 bit fields.
"""

import torch
import torch.nn as nn
from typing import Optional
from nli_eda import optimize_eda, EDAConfig, get_function, get_domain

# ─────────────────────────────────────────────────────────────
#  Cache for precomputed configs
# ─────────────────────────────────────────────────────────────

_EDA_CACHE = {}


def get_eda_config(func_name: str, max_lut: int = 256, max_k: int = 5,
                   device: str = 'cuda') -> EDAConfig:
    """Get or compute EDA config (cached)."""
    key = f"{func_name}_{max_lut}_{max_k}_{device}"
    if key not in _EDA_CACHE:
        config = optimize_eda(func_name, max_lut=max_lut, max_k=max_k,
                              device=device, verbose=True)
        # Move tensors to device
        config.bin_starts = config.bin_starts.to(device)
        config.bin_ends = config.bin_ends.to(device)
        config.base_offsets = config.base_offsets.to(device)
        config.k_bits_tensor = config.k_bits_tensor.to(device)
        config.lut_values = config.lut_values.to(device)
        _EDA_CACHE[key] = config
    return _EDA_CACHE[key]


# ─────────────────────────────────────────────────────────────
#  EDA Forward Pass (Vectorized PyTorch)
# ─────────────────────────────────────────────────────────────

def _eda_forward_chunk(x_flat: torch.Tensor, config: EDAConfig,
                       t_bits: int = None, fused: bool = True) -> torch.Tensor:
    """Core EDA computation on a flat 1-D chunk (bounded memory)."""
    bin_starts = config.bin_starts.float()
    n_bins = len(bin_starts)

    bin_idx = torch.bucketize(x_flat, bin_starts, right=False) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    bin_ends = config.bin_ends.float()
    too_low = x_flat < bin_starts[0]
    too_high = x_flat > bin_ends[-1]

    b_start = bin_starts[bin_idx]
    b_end = bin_ends[bin_idx]
    base_offset = config.base_offsets[bin_idx]
    k_bits = config.k_bits_tensor[bin_idx]

    bin_width = b_end - b_start
    rel_pos = (x_flat - b_start) / bin_width.clamp(min=1e-30)
    rel_pos = rel_pos.clamp(0.0, 1.0 - 1e-7)

    num_microbins = (1 << k_bits).float()
    scaled = rel_pos * num_microbins
    micro_idx = scaled.long()
    max_micro = (num_microbins - 1).long()
    micro_idx = torch.minimum(micro_idx, max_micro)

    t = scaled - micro_idx.float()
    t = t.clamp(0.0, 1.0)

    if t_bits is not None:
        if t_bits == 'adaptive':
            # HW-accurate: t_int uses all (10-K) remaining mantissa bits
            per_elem_t = 10 - k_bits
            scale = (1 << per_elem_t).float()
            t = torch.floor(t * scale) / scale
        else:
            t = torch.floor(t * (1 << t_bits)) / (1 << t_bits)

    lut_idx = base_offset + micro_idx
    lut_idx_next = lut_idx + 1
    max_lut_idx = len(config.lut_values) - 1
    lut_idx = lut_idx.clamp(0, max_lut_idx)
    lut_idx_next = lut_idx_next.clamp(0, max_lut_idx)

    # RTL: func_lut stores FP16 values; diff in FP16
    lut_fp16 = config.lut_values.half()
    y0 = lut_fp16[lut_idx]
    y1 = lut_fp16[lut_idx_next]
    diff = (y1 - y0)                       # FP16 subtractor (round 1)
    if fused:
        # Fused interpolation (2 roundings): multiply in FP32, add in FP32, round once
        product_fp32 = t.float() * diff.float()
        y = (y0.float() + product_fp32).half().float()  # round 2 (fused)
    else:
        # Non-fused (3 roundings): multiply rounds to FP16, then add rounds to FP16
        product = (t.half() * diff).half()      # round 2 (multiply)
        y = (y0 + product).half().float()       # round 3 (add)

    y = torch.where(too_low, lut_fp16[0].float(), y)
    y = torch.where(too_high, lut_fp16[-1].float(), y)
    return y


_EDA_CHUNK = 4 * 1024 * 1024   # 4M elements


def eda_forward(x_input: torch.Tensor, func_name: str,
                max_lut: int = 256, max_k: int = 5,
                t_bits = None, fused: bool = True) -> torch.Tensor:
    """
    EDA-NLI forward pass simulating the 3-stage bit-extraction pipeline.

    Tries full-tensor first; on OOM, automatically falls back to chunked processing.
    """
    device = x_input.device
    original_shape = x_input.shape
    x_dtype = x_input.dtype
    # EDA uses FP16 bit-field extraction — must quantize to fp16 first
    x_flat = x_input.reshape(-1).half().float()

    config = get_eda_config(func_name, max_lut, max_k, device=str(device))

    n = x_flat.numel()
    if n > _EDA_CHUNK:
        parts = []
        for i in range(0, n, _EDA_CHUNK):
            end = min(i + _EDA_CHUNK, n)
            parts.append(_eda_forward_chunk(x_flat[i:end], config, t_bits, fused))
        y = torch.cat(parts, dim=0)
    else:
        y = _eda_forward_chunk(x_flat, config, t_bits, fused)

    return y.reshape(original_shape).to(x_dtype)


# ─────────────────────────────────────────────────────────────
#  Module Wrappers
# ─────────────────────────────────────────────────────────────

class EDANLIFunction(nn.Module):
    """Drop-in replacement for nonlinear functions using EDA-NLI."""

    def __init__(self, func_name: str, max_lut: int = 256, max_k: int = 5):
        super().__init__()
        self.func_name = func_name
        self.max_lut = max_lut
        self.max_k = max_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return eda_forward(x, self.func_name, self.max_lut, self.max_k)

    def __repr__(self):
        return f"EDANLIFunction(func={self.func_name}, max_lut={self.max_lut}, max_k={self.max_k})"


class EDA_SiLU(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('silu', **kwargs)

class EDA_GELU(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('gelu', **kwargs)

class EDA_Exp(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('exp', **kwargs)

class EDA_Rsqrt(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('rsqrt', **kwargs)

class EDA_Sigmoid(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('sigmoid', **kwargs)

class EDA_Tanh(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('tanh', **kwargs)

class EDA_Reciprocal(EDANLIFunction):
    def __init__(self, **kwargs): super().__init__('reciprocal', **kwargs)


# ─────────────────────────────────────────────────────────────
#  Hardware Cost Estimation
# ─────────────────────────────────────────────────────────────

def estimate_hardware_cost(config: EDAConfig) -> dict:
    """
    Estimate hardware area/power based on EDA-NLI configuration.

    NLI baseline (from Table 5 of paper):
      LUT: 6445 μm²  |  Comparator: 410 μm²  |  Multiplier: 205 μm²
      Adder: 134 μm²  |  Others: 191 μm²  |  Total: 7787 μm²
      Power: 34 mW

    EDA-NLI eliminates: Comparator (410) + Multiplier (205) = 615 μm²
    EDA-NLI LUT scales with total entries (proportional to NLI's 259).
    """
    nli_lut_area = 6445  # μm² for 259 entries
    nli_total_entries = 259

    eda_entries = config.total_lut + len(config.bins)  # entries + shared endpoints
    lut_area = nli_lut_area * (eda_entries / nli_total_entries)

    # EDA eliminates: comparator tree + multiplier + scale registers
    eliminated = 410 + 205 + 50  # ~665 μm²

    # EDA keeps: adder (134), FMA (part of others), SRAM
    kept_area = 134 + 100  # adder + minimal control logic

    total_area = lut_area + kept_area
    nli_total = 7787

    # Power estimation (proportional to area, roughly)
    nli_power = 34  # mW
    power = nli_power * (total_area / nli_total)

    return {
        'total_area_um2': total_area,
        'lut_area_um2': lut_area,
        'eliminated_area_um2': eliminated,
        'nli_baseline_area_um2': nli_total,
        'area_reduction_pct': (1 - total_area / nli_total) * 100,
        'estimated_power_mw': power,
        'pipeline_stages': 3,
        'num_comparators': 0,
        'num_multipliers_addr': 0,
        'lut_entries': eda_entries,
    }


# ─────────────────────────────────────────────────────────────
#  Main: Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== EDA-NLI Engine Quick Test ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for func_name in ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh']:
        func = get_function(func_name)
        domain = get_domain(func_name)

        lo, hi = domain
        if func_name in ('rsqrt', 'reciprocal'):
            test_x = torch.logspace(
                torch.log10(torch.tensor(max(lo, 1e-6))).item(),
                torch.log10(torch.tensor(hi)).item(),
                10000, device=device
            )
        else:
            test_x = torch.linspace(lo, hi, 10000, device=device)

        y_eda = eda_forward(test_x, func_name, max_lut=256, max_k=5)
        y_ref = func(test_x)

        abs_err = torch.abs(y_eda - y_ref)
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()

        denom = torch.clamp(torch.abs(y_ref), min=1e-3)
        rel_err = abs_err / denom
        max_rel = rel_err.max().item()

        print(f"  {func_name:12s}: max_abs={max_err:.4e}  mean_abs={mean_err:.4e}  max_rel={max_rel:.4e}")

    # Hardware cost
    config = get_eda_config('silu', 256, 5, device)
    hw = estimate_hardware_cost(config)
    print(f"\n  === Hardware Estimate (silu) ===")
    print(f"  Area: {hw['total_area_um2']:.0f} μm² (NLI: {hw['nli_baseline_area_um2']} μm²)")
    print(f"  Reduction: {hw['area_reduction_pct']:.1f}%")
    print(f"  Power: {hw['estimated_power_mw']:.1f} mW")
    print(f"  Pipeline: {hw['pipeline_stages']} stages")
    print(f"  Comparators: {hw['num_comparators']}")
