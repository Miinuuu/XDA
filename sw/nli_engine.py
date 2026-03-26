"""
NLI Engine: LUT Generation and Forward Pass (Algorithm 2)
==========================================================
Implements the NLI computation flow for nonlinear function approximation.

Two-level address translation:
  - 10 macro-intervals with 11 cutpoints
  - Middle 8 intervals uniformly split into 32 micro-bins
  - Total: 2 + 8×32 + 1 = 259 LUT entries
"""

import torch
import torch.nn.functional as F
from typing import Callable, Tuple, Optional, Dict
from nli_dp import PAPER_CUTPOINTS, get_function, get_domain
import numpy as np

# ─────────────────────────────────────────────────────────────
#  LUT Builder
# ─────────────────────────────────────────────────────────────

def build_lut(
    func: Callable[[torch.Tensor], torch.Tensor],
    macro_cutpoints: torch.Tensor,
    D_n: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the 259-entry LUT from 11 macro cutpoints.
    
    Layout: 2 + 8×32 + 1 = 259 entries
    - Interval 0 (boundary): 1 entry (clamped, no subdivision)
    - Intervals 1-8: 32 bins each → 256 entries  
    - Interval 9 (boundary): 1 entry (clamped, no subdivision)
    - Plus shared endpoints → 259 total
    
    Args:
        func: Target nonlinear function
        macro_cutpoints: 11 macro cutpoint x-values (sorted)
        D_n: Number of uniform bins per middle interval (default 32)
    
    Returns:
        point_reg: Tensor of 11 macro cutpoint values (for interval selection)
        mul_reg: Tensor of 10 multiply scale factors (for address computation)
        lut_reg: Tensor of 259 LUT entries (function values at all cutpoints)
    """
    M = len(macro_cutpoints)
    num_intervals = M - 1
    
    point_reg = macro_cutpoints.clone().float()
    
    # Build mul_reg: scale factor for each interval
    mul_reg = torch.zeros(num_intervals, dtype=torch.float32)
    for i in range(num_intervals):
        width = point_reg[i + 1] - point_reg[i]
        if width == 0:
            mul_reg[i] = 0.0
        elif i == 0 or i == num_intervals - 1:
            # Boundary intervals: not subdivided, single bin
            mul_reg[i] = 1.0 / width.item()
        else:
            # Middle intervals: D_n bins
            mul_reg[i] = D_n / width.item()
    
    # Build LUT: generate all fine cutpoints and evaluate f
    all_x = []
    for i in range(num_intervals):
        b_start = point_reg[i].item()
        b_end = point_reg[i + 1].item()
        
        if i == 0 or i == num_intervals - 1:
            num_bins = 1
        else:
            num_bins = D_n
        
        sub_points = torch.linspace(b_start, b_end, num_bins + 1, dtype=torch.float32)
        
        if i == 0:
            all_x.append(sub_points)  # Include all points for first interval
        else:
            all_x.append(sub_points[1:])  # Skip start (shared with previous end)
    
    all_x = torch.cat(all_x)
    lut_reg = func(all_x)
    # Clamp to FP16 representable range: inf in LUT causes NaN during
    # interpolation (inf - inf = NaN), so saturate to ±65504.
    lut_reg = lut_reg.clamp(-65504.0, 65504.0)

    expected_size = 2 + (num_intervals - 2) * D_n + 1
    assert len(lut_reg) == expected_size, \
        f"LUT size mismatch: got {len(lut_reg)}, expected {expected_size}"
    
    return point_reg, mul_reg, lut_reg


def build_lut_from_paper(func_name: str, D_n: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build LUT using paper-provided cutpoints."""
    func = get_function(func_name)
    cutpoints = torch.tensor(PAPER_CUTPOINTS[func_name], dtype=torch.float32)
    return build_lut(func, cutpoints, D_n)


# ─────────────────────────────────────────────────────────────
#  NLI Forward Pass (Algorithm 2)
# ─────────────────────────────────────────────────────────────

def _nli_forward_chunk(
    x_flat: torch.Tensor,
    point_reg: torch.Tensor,
    mul_reg: torch.Tensor,
    lut_reg: torch.Tensor,
    D_n: int,
    M: int,
    num_intervals: int,
    fp16_hw: bool = False,
) -> torch.Tensor:
    """Core NLI computation on a flat 1-D chunk (bounded memory).

    When fp16_hw=True, simulates the RTL pipeline constraints:
      Stage 1: FP16 comparator + FP16 subtractor → offset
      Stage 2: FP16 multiplier → scaled_pos, 6.10 fixed-point → 10-bit frac
      Stage 3: FP16 LUT lookup + FP16 subtractor → diff
      Stage 4: FP16 multiplier + FP16 adder → output
    """
    if fp16_hw:
        # All registers are FP16 in hardware
        pr = point_reg.half()
        mr = mul_reg.half()
        lr = lut_reg.half()
        x_fp16 = x_flat.half()

        # Stage 1: FP16 comparator chain + FP16 subtractor
        x_clamped = torch.clamp(x_fp16, min=pr[0], max=pr[M - 1])
        index = torch.bucketize(x_clamped.float(), pr[1:num_intervals].float())

        base_point = pr[index]
        offset = (x_clamped - base_point)  # FP16 subtraction (catastrophic cancellation)

        # Stage 2: FP16 multiply → 6.10 fixed-point extraction
        scale = mr[index]
        scaled_pos = (offset * scale)  # FP16 multiply

        # Convert FP16 scaled_pos to 6.10 fixed-point (matching RTL)
        # RTL clamps overflow to 16'hFFFF (63.999) in 6.10 format
        sp_f32 = scaled_pos.float()
        sp_f32 = torch.clamp(sp_f32, min=0.0, max=63.999023)  # 6.10 max
        # Truncate to 6.10 fixed-point: multiply by 1024, floor, divide by 1024
        fixed_6_10 = torch.floor(sp_f32 * 1024.0) / 1024.0
        floor_val = torch.floor(fixed_6_10)
        address = floor_val.long()
        address = torch.where((index == 0) | (index == num_intervals - 1),
                              torch.zeros_like(address), address)
        address = torch.clamp(address, min=0, max=D_n - 1)

        # 10-bit fractional part → convert back to FP16
        frac = fixed_6_10 - floor_val
        frac = torch.clamp(frac, min=0.0, max=1.0)
        # RTL: when scaled_pos < 1.0, decimal = scaled_pos (full FP16 precision)
        decimal = torch.where(sp_f32 < 1.0, sp_f32, frac)
        decimal = torch.clamp(decimal, min=0.0, max=1.0).half()

        # Stage 3: FP16 LUT lookup + FP16 subtractor
        indices = torch.where(
            index == 0,
            address,
            1 + (index - 1) * D_n + address
        )
        max_idx = len(lr) - 2
        indices = torch.clamp(indices, min=0, max=max_idx)

        left_val = lr[indices]
        right_val = lr[indices + 1]
        diff = (right_val - left_val)  # FP16 subtract

        # Stage 4: FP16 multiply + FP16 add
        y = (left_val + decimal * diff).half()
        # Replace NaN/Inf from FP16 overflow with nearest LUT value
        y = torch.where(torch.isfinite(y), y, left_val)

        y = torch.where(x_fp16 <= pr[0], lr[0], y)
        y = torch.where(x_fp16 >= pr[M - 1], lr[-1], y)
        return y.float()

    # Original float32 path
    x_clamped = torch.clamp(x_flat, min=point_reg[0].item(), max=point_reg[M - 1].item())
    index = torch.bucketize(x_clamped, point_reg[1:num_intervals])

    base_point = point_reg[index]
    offset = x_clamped - base_point
    scale = mul_reg[index]
    scaled_pos = offset * scale

    address = torch.floor(scaled_pos).long()
    address = torch.where((index == 0) | (index == num_intervals - 1),
                          torch.zeros_like(address), address)
    address = torch.clamp(address, min=0, max=D_n - 1)

    decimal = scaled_pos - address.float()
    decimal = torch.clamp(decimal, min=0.0, max=1.0)

    indices = torch.where(
        index == 0,
        address,
        1 + (index - 1) * D_n + address
    )
    max_idx = len(lut_reg) - 2
    indices = torch.clamp(indices, min=0, max=max_idx)

    left_val = lut_reg[indices]
    right_val = lut_reg[indices + 1]
    y = left_val + decimal * (right_val - left_val)

    y = torch.where(x_flat <= point_reg[0], lut_reg[0], y)
    y = torch.where(x_flat >= point_reg[M - 1], lut_reg[-1], y)
    return y


_NLI_CHUNK = 4 * 1024 * 1024   # 4M elements


def nli_forward(
    x: torch.Tensor,
    point_reg: torch.Tensor,
    mul_reg: torch.Tensor,
    lut_reg: torch.Tensor,
    D_n: int = 32,
    fp16_hw: bool = False,
) -> torch.Tensor:
    """
    NLI Computation Flow (Algorithm 2).

    Two-level address translation + linear interpolation.
    Tries full-tensor first; on OOM, automatically falls back to chunked processing.

    When fp16_hw=True, simulates the 4-stage RTL pipeline with FP16
    arithmetic and 10-bit fractional precision (matching nli_engine.v).
    """
    M = len(point_reg)
    num_intervals = M - 1

    original_shape = x.shape
    original_dtype = x.dtype
    # NLI simulates FP16 hardware — quantize to fp16 first
    x_flat = x.reshape(-1).half().float()

    device = x_flat.device
    point_reg = point_reg.to(device)
    mul_reg = mul_reg.to(device)
    lut_reg = lut_reg.to(device)

    n = x_flat.numel()
    if n > _NLI_CHUNK:
        # Always chunk large tensors to avoid OOM
        parts = []
        for i in range(0, n, _NLI_CHUNK):
            end = min(i + _NLI_CHUNK, n)
            parts.append(_nli_forward_chunk(
                x_flat[i:end], point_reg, mul_reg, lut_reg,
                D_n, M, num_intervals, fp16_hw=fp16_hw))
        y = torch.cat(parts, dim=0)
    else:
        y = _nli_forward_chunk(x_flat, point_reg, mul_reg, lut_reg,
                               D_n, M, num_intervals, fp16_hw=fp16_hw)

    return y.reshape(original_shape).to(original_dtype)


# ─────────────────────────────────────────────────────────────
#  NLI Function Wrapper (drop-in replacement)
# ─────────────────────────────────────────────────────────────

class NLIFunction(torch.nn.Module):
    """
    Drop-in replacement for a nonlinear function using NLI approximation.
    
    Example:
        nli_silu = NLIFunction('silu')
        y = nli_silu(x)   # same as F.silu(x), but via NLI lookup
    """
    
    def __init__(self, func_name: str, D_n: int = 32, 
                 custom_cutpoints: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.func_name = func_name
        self.D_n = D_n
        
        func = get_function(func_name)
        
        if custom_cutpoints is not None:
            cutpoints = custom_cutpoints
        else:
            cutpoints = torch.tensor(PAPER_CUTPOINTS[func_name], dtype=torch.float32)
        
        point_reg, mul_reg, lut_reg = build_lut(func, cutpoints, D_n)
        
        # Register as buffers (move with model, saved in state_dict)
        self.register_buffer('point_reg', point_reg)
        self.register_buffer('mul_reg', mul_reg)
        self.register_buffer('lut_reg', lut_reg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nli_forward(x, self.point_reg, self.mul_reg, self.lut_reg, self.D_n)
    
    def __repr__(self):
        return f"NLIFunction(func={self.func_name}, cutpoints=11, lut_size={len(self.lut_reg)})"


# ─────────────────────────────────────────────────────────────
#  Prebuilt NLI modules for common LLM operations
# ─────────────────────────────────────────────────────────────

class NLI_SiLU(NLIFunction):
    """NLI approximation of SiLU activation."""
    def __init__(self, **kwargs):
        super().__init__('silu', **kwargs)


class NLI_Exp(NLIFunction):
    """NLI approximation of exp (for Softmax)."""
    def __init__(self, **kwargs):
        super().__init__('exp', **kwargs)


class NLI_Rsqrt(NLIFunction):
    """NLI approximation of rsqrt (for RMSNorm)."""
    def __init__(self, **kwargs):
        super().__init__('rsqrt', **kwargs)


class NLI_GELU(NLIFunction):
    """NLI approximation of GELU activation."""
    def __init__(self, **kwargs):
        super().__init__('gelu', **kwargs)


class NLI_Sigmoid(NLIFunction):
    """NLI approximation of sigmoid."""
    def __init__(self, **kwargs):
        super().__init__('sigmoid', **kwargs)


class NLI_Tanh(NLIFunction):
    """NLI approximation of tanh."""
    def __init__(self, **kwargs):
        super().__init__('tanh', **kwargs)

class NLI_Reciprocal(NLIFunction):
    """NLI approximation of reciprocal."""
    def __init__(self, **kwargs):
        super().__init__('reciprocal', **kwargs)


# ─────────────────────────────────────────────────────────────
#  NLI-based RMSNorm and Softmax
# ─────────────────────────────────────────────────────────────

class NLI_RMSNorm(torch.nn.Module):
    """
    RMSNorm using NLI rsqrt approximation.
    
    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.nli_rsqrt = NLI_Rsqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True) + self.eps
        # Use NLI rsqrt instead of torch.rsqrt
        inv_rms = self.nli_rsqrt(variance)
        return x * inv_rms * self.weight


class NLI_Softmax(torch.nn.Module):
    """
    Softmax using NLI exp approximation.
    
    Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.nli_exp = NLI_Exp()
        self.nli_reciprocal=NLI_Reciprocal()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = x.max(dim=self.dim, keepdim=True).values
        x_shifted = x - x_max
        exp_x = self.nli_exp(x_shifted)
        return exp_x * self.nli_reciprocal(exp_x.sum(dim=self.dim, keepdim=True))


# ─────────────────────────────────────────────────────────────
#  Main: Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=== NLI Engine Quick Test ===\n")
    
    for func_name in ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh']:
        func = get_function(func_name)
        domain = get_domain(func_name)
        
        # Build LUT from paper cutpoints
        point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)
        
        # Generate test points in domain
        lo, hi = domain
        # Avoid extreme values for rsqrt/reciprocal
        if func_name in ('rsqrt', 'reciprocal'):
            test_x = torch.logspace(
                np.log10(max(lo, 1e-6)), np.log10(hi), 10000
            )
            test_x = torch.linspace(lo, hi, 10000)
        else:
            test_x = torch.linspace(lo, hi, 10000)
        
        # Compute NLI and reference
        y_nli = nli_forward(test_x, point_reg, mul_reg, lut_reg)
        y_ref = func(test_x)
        
        # Error stats
        abs_err = torch.abs(y_nli - y_ref)
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        
        rel_err = abs_err / torch.clamp(torch.abs(y_ref), min=1e-10)
        max_rel_err = rel_err.max().item()
        
        print(f"{func_name:12s}: max_abs_err={max_err:.4e}, mean_abs_err={mean_err:.4e}, max_rel_err={max_rel_err:.4e}")
    
    print("\n=== NLI Module Test ===\n")
    
    # Test NLIFunction as module
    nli_silu = NLI_SiLU()
    x_test = torch.randn(100)
    y_nli = nli_silu(x_test)
    y_ref = F.silu(x_test)
    err = (y_nli - y_ref).abs().max().item()
    print(f"NLI_SiLU module max error: {err:.4e}")
    
    # Test NLI_RMSNorm
    rms = NLI_RMSNorm(64)
    x_test = torch.randn(2, 10, 64)
    y_rms = rms(x_test)
    print(f"NLI_RMSNorm output shape: {y_rms.shape}")
    
    # Test NLI_Softmax
    sm = NLI_Softmax()
    x_test = torch.randn(2, 10, 64)
    y_sm = sm(x_test)
    print(f"NLI_Softmax output shape: {y_sm.shape}, sum={y_sm.sum(-1)[0,0].item():.4f}")
    
    print("\nDone!")
