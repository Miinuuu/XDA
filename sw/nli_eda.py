"""
EDA-NLI: Exponent-Direct Addressing NLI
========================================
Zero-FP-arithmetic address generation using FP16 bit-field extraction.

Instead of comparators + multiplier for address translation,
EDA-NLI uses the FP16 exponent bits directly as macro-interval selector
and top-K mantissa bits as micro-bin index. Both are pure wiring (0 FP arithmetic gates).

Hardware Pipeline (3-stage, down from NLI's 4-stage):
  Stage 1: Bit Extract {sign, exp, mantissa_top_K} → LUT address  (0 FP arithmetic; 9-bit int add only)
  Stage 2: Dual-port SRAM read LUT[g], LUT[g+1]                   (1 subtractor)
  Stage 3: y = y0 + t * (y1 - y0)                                 (1 FMA)

Key advantage: eliminates 10 comparators, 1 multiplier, scale registers.
"""

import struct
import json
import os
import torch
import numpy as np
import math
import time
from typing import Tuple, List, Callable, Optional, Dict

TAU = 2 ** (-14)
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eda_results')
_OPTIMAL_DOMAINS_PATH = os.path.join(_RESULTS_DIR, 'optimal_domains.json')

# ─────────────────────────────────────────────────────────────
#  FP16 Exponent Bin Generation
# ─────────────────────────────────────────────────────────────

def get_fp16_exponent_bins(domain: Tuple[float, float]) -> List[Tuple[float, float, int, int]]:
    """
    Generate FP16 exponent-aligned bins within domain.

    Each bin corresponds to one FP16 exponent value.
    Returns list of (bin_start, bin_end, sign_bit, exponent_value).

    FP16 structure: sign(1) + exponent(5) + mantissa(10)
    - Exponent 0 (with mantissa!=0): subnormals [0, 2^-14)
    - Exponent 1-30: normals [2^(e-15), 2^(e-14))
    - Exponent 31: inf/nan
    """
    bins = []

    # Positive normals: exponent e=1..30 → [2^(e-15), 2^(e-14))
    for e in range(1, 31):
        lo = 2.0 ** (e - 15)
        hi = 2.0 ** (e - 14)
        bins.append((lo, hi, 0, e))

    # Positive subnormals: exponent e=0 → [0, 2^-14)
    # For practical purposes, use (smallest_subnormal, 2^-14)
    bins.append((2.0**(-24), 2.0**(-14), 0, 0))  # subnormal range

    # Negative normals: exponent e=1..30
    for e in range(1, 31):
        lo = -(2.0 ** (e - 14))
        hi = -(2.0 ** (e - 15))
        bins.append((lo, hi, 1, e))

    # Negative subnormals
    bins.append((-(2.0**(-14)), -(2.0**(-24)), 1, 0))

    # Filter to domain
    dom_lo, dom_hi = domain
    filtered = []
    for (b_lo, b_hi, sign, exp_val) in bins:
        if b_hi <= dom_lo or b_lo >= dom_hi:
            continue
        clip_lo = max(b_lo, dom_lo)
        clip_hi = min(b_hi, dom_hi)
        if clip_lo < clip_hi:
            filtered.append((clip_lo, clip_hi, sign, exp_val))

    filtered.sort(key=lambda x: x[0])
    return filtered


def get_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    funcs = {
        'silu': lambda x: x * torch.sigmoid(x),
        'gelu': lambda x: torch.nn.functional.gelu(x),
        'exp': lambda x: torch.exp(x),
        'rsqrt': lambda x: torch.rsqrt(x),
        'reciprocal': lambda x: 1.0 / x,
        'sigmoid': lambda x: torch.sigmoid(x),
        'tanh': lambda x: torch.tanh(x),
        'hardswish': lambda x: torch.nn.functional.hardswish(x),
        'mish': lambda x: x * torch.tanh(torch.nn.functional.softplus(x)),
    }
    return funcs[name]


# def get_domain(name: str) -> Tuple[float, float]:
#     domains = {
#         'silu':       (-20.0, 20.0),
#         'gelu':       (-20.0, 20.0),
#         'sigmoid':    (-20.0, 20.0),
#         'tanh':       (-20.0, 20.0),
#         'hardswish':  (-6.0, 10.0),
#         'mish':       (-20.0, 20.0),
#         'exp':        (-20.0, 11.1),
#         'rsqrt':      (2**(-14), 65504.0),
#         'reciprocal': (2**(-14), 65504.0),
#     }

#     return domains.get(name, (-20.0, 20.0))


_DEFAULT_DOMAINS = {
    'silu':       (-32.0,               65504.0),
    'gelu':       (-8.0,                65504.0),
    'exp':        (-32.0,               16.0),
    'sigmoid':    (-32.0,               16.0),
    'tanh':       (-8.0,                8.0),
    'hardswish':  (-8.0,                65504.0),
    'mish':       (-32.0,               65504.0),
    'rsqrt':      (2**-24,              65504.0),
    'reciprocal': (2**-16,              65504.0),
}

# Cache: loaded once per process
_loaded_domains = None  # Optional[Dict[str, Tuple[float, float]]]


def _invalidate_domain_cache():
    """Clear cached domains so next get_domain() reloads from JSON."""
    global _loaded_domains
    _loaded_domains = None


def _load_optimal_domains() -> Dict[str, Tuple[float, float]]:
    """Load optimal domains from JSON if available."""
    global _loaded_domains
    if _loaded_domains is not None:
        return _loaded_domains
    if os.path.exists(_OPTIMAL_DOMAINS_PATH):
        with open(_OPTIMAL_DOMAINS_PATH) as f:
            data = json.load(f)
        _loaded_domains = {
            name: tuple(entry['domain'])
            for name, entry in data.items()
        }
        return _loaded_domains
    _loaded_domains = {}
    return _loaded_domains


def get_domain(name: str) -> Tuple[float, float]:
    """
    Knapsack optimization domain per function.

    Loads from eda_results/optimal_domains.json if available
    (produced by --optimize-domain), otherwise falls back to hardcoded defaults.
    """
    loaded = _load_optimal_domains()
    if name in loaded:
        return loaded[name]
    if name not in _DEFAULT_DOMAINS:
        raise ValueError(f"Unknown function: {name}. Available: {list(_DEFAULT_DOMAINS.keys())}")
    return _DEFAULT_DOMAINS[name]



# ─────────────────────────────────────────────────────────────
#  Error Computation for Exponent Bins
# ─────────────────────────────────────────────────────────────

def _fp16_grid_for_bin(b_start: float, b_end: float, device: str = 'cuda') -> torch.Tensor:
    """Generate all representable FP16 values in [b_start, b_end]."""
    vals = []
    for bits in range(0, 0x7C00):  # positive FP16 (excl inf/nan)
        val = struct.unpack('e', struct.pack('H', bits))[0]
        if b_start <= val <= b_end:
            vals.append(val)
        neg = -val
        if val != 0 and b_start <= neg <= b_end:
            vals.append(neg)
    return torch.tensor(sorted(set(vals)), dtype=torch.float32, device=device)


def compute_bin_error(func: Callable, b_start: float, b_end: float,
                      k_bits: int, device: str = 'cuda',
                      **_kwargs) -> Tuple[float, int]:
    """
    Compute interpolation error for an exponent bin with k_bits mantissa bits.

    k_bits mantissa bits → 2^k_bits micro-bins within this exponent interval.
    Cutpoints are uniformly spaced (matching mantissa bit semantics).

    Uses exhaustive FP16 grid points (no sampling bias).

    Returns: (total_relative_error, num_lut_entries)
    """
    num_microbins = 2 ** k_bits

    x = _fp16_grid_for_bin(b_start, b_end, device)
    if len(x) == 0:
        return 0.0, num_microbins
    y_true = func(x)

    # Micro-bin cutpoints (uniform within exponent interval)
    cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)
    y_cut = func(cutpoints)

    # Find which micro-bin each sample falls in
    idx = torch.searchsorted(cutpoints, x) - 1
    idx = torch.clamp(idx, 0, num_microbins - 1)

    # Linear interpolation
    x0 = cutpoints[idx]
    x1 = cutpoints[idx + 1]
    y0 = y_cut[idx]
    y1 = y_cut[idx + 1]

    dx = x1 - x0
    t = torch.where(dx > 0, (x - x0) / dx, torch.zeros_like(dx))
    T_bits = 10 - k_bits
    scale = float(1 << T_bits)
    t_q = torch.floor(t * scale) / scale
    y_pred = y0 + t_q * (y1 - y0)

    # Relative error
    abs_err = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=TAU)
    rel_err = abs_err / denom

    return rel_err.sum().item(), num_microbins


# ─────────────────────────────────────────────────────────────
#  Knapsack DP: Allocate mantissa bits per exponent bin
# ─────────────────────────────────────────────────────────────

def precompute_bin_costs(func: Callable, bins: List[Tuple[float, float, int, int]],
                         max_k: int = 5, device: str = 'cuda',
                         samples_per_bin: int = 10000
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute error and LUT weight for each (bin, k_bits) pair.

    Returns:
        err_matrix: (N_bins, max_k+1) - error for each bin with k mantissa bits
        weight_matrix: (N_bins, max_k+1) - LUT entries needed (2^k per bin)
    """
    N = len(bins)
    K = max_k + 1
    err_matrix = np.full((N, K), np.inf, dtype=np.float64)
    weight_matrix = np.zeros((N, K), dtype=np.int32)

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        for k in range(K):
            err, n_entries = compute_bin_error(func, b_start, b_end, k,
                                               device=device, n_samples=samples_per_bin)
            err_matrix[i, k] = err
            weight_matrix[i, k] = n_entries

    return err_matrix, weight_matrix


def solve_knapsack_dp(err_matrix: np.ndarray, weight_matrix: np.ndarray,
                       max_lut: int) -> Tuple[float, List[int]]:
    """
    Knapsack DP to allocate mantissa bits per exponent bin.

    Minimizes total error subject to sum of 2^k_i <= max_lut.

    Returns: (best_error, k_alloc) where k_alloc[i] = k_bits for bin i.
    """
    N, K = err_matrix.shape

    dp = np.full((N + 1, max_lut + 1), np.inf, dtype=np.float64)
    choice = np.full((N + 1, max_lut + 1), -1, dtype=np.int32)
    dp[0][0] = 0.0

    for i in range(N):
        reachable = np.where(dp[i] < np.inf)[0]
        if len(reachable) == 0:
            continue
        for j in range(K):
            w_add = int(weight_matrix[i, j])
            cost_j = err_matrix[i, j]
            new_ws = reachable + w_add
            mask = new_ws <= max_lut
            rw = reachable[mask]
            nw = new_ws[mask]
            new_v = dp[i][rw] + cost_j
            impr = new_v < dp[i + 1][nw]
            dp[i + 1][nw[impr]] = new_v[impr]
            choice[i + 1][nw[impr]] = j

    best_w = int(np.argmin(dp[N]))
    best_val = dp[N][best_w]

    if best_val == np.inf:
        raise ValueError(f"Cannot fit within max_lut={max_lut}. Increase budget.")

    # Backtrack
    curr_w = best_w
    k_alloc = []
    for i in range(N, 0, -1):
        j = int(choice[i][curr_w])
        k_alloc.append(j)
        curr_w -= int(weight_matrix[i - 1, j])
    k_alloc.reverse()

    return best_val, k_alloc


# ─────────────────────────────────────────────────────────────
#  Alternative Allocation Strategies (for ablation studies)
# ─────────────────────────────────────────────────────────────

def allocate_uniform(bins, max_lut, max_k=5):
    """Uniform allocation: same K for all bins.

    K = max K such that N_bins * 2^K <= max_lut.
    Returns (0.0, k_alloc) matching solve_knapsack_dp signature.
    """
    n_bins = len(bins)
    best_k = 0
    for k in range(max_k + 1):
        if n_bins * (2 ** k) <= max_lut:
            best_k = k
    return 0.0, [best_k] * n_bins


def allocate_curvature_proportional(func, bins, max_lut, max_k=5, device='cuda'):
    """Curvature-proportional allocation: K proportional to mean |f''(x)|.

    Computes numerical 2nd derivative per bin, then greedily assigns
    highest feasible K to highest-curvature bins first.
    Returns (0.0, k_alloc) matching solve_knapsack_dp signature.
    """
    n_bins = len(bins)
    curvatures = []
    for b_start, b_end, sign, exp_val in bins:
        width = b_end - b_start
        if width < 1e-30:
            curvatures.append(0.0)
            continue
        x = torch.linspace(b_start, b_end, 1000, device=device, dtype=torch.float64)
        y = func(x.float()).double()
        if len(y) < 3:
            curvatures.append(0.0)
            continue
        dy2 = torch.diff(y, n=2)
        dx = width / 999.0
        f_pp = torch.abs(dy2) / (dx ** 2)
        curvatures.append(f_pp.mean().item())

    # Greedy: highest-curvature bins get first allocation
    k_alloc = [0] * n_bins
    budget_used = n_bins  # k=0 → 1 entry per bin
    order = sorted(range(n_bins), key=lambda i: curvatures[i], reverse=True)

    for idx in order:
        for k in range(max_k, 0, -1):
            additional = (2 ** k) - (2 ** k_alloc[idx])
            if budget_used + additional <= max_lut:
                budget_used += additional
                k_alloc[idx] = k
                break

    return 0.0, k_alloc


def optimize_eda_with_allocation(func_name, alloc_strategy='knapsack',
                                 max_lut=254, max_k=5, device='cuda'):
    """Build EDAConfig using specified allocation strategy.

    Args:
        alloc_strategy: 'knapsack' | 'uniform' | 'curvature'
    """
    func = get_function(func_name)
    domain = get_domain(func_name)
    bins = get_fp16_exponent_bins(domain)

    if alloc_strategy == 'knapsack':
        err_matrix, weight_matrix = precompute_bin_costs(func, bins, max_k, device)
        total_error, k_alloc = solve_knapsack_dp(err_matrix, weight_matrix, max_lut)
    elif alloc_strategy == 'uniform':
        total_error, k_alloc = allocate_uniform(bins, max_lut, max_k)
    elif alloc_strategy == 'curvature':
        total_error, k_alloc = allocate_curvature_proportional(
            func, bins, max_lut, max_k, device)
    else:
        raise ValueError(f"Unknown strategy: {alloc_strategy}")

    # Build config (same logic as optimize_eda)
    config = EDAConfig()
    config.bins = [(b[0], b[1], b[2], b[3]) for b in bins]
    config.k_alloc = k_alloc
    config.total_error = total_error

    all_lut_values = []
    base_offsets = []
    current_offset = 0

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        k = k_alloc[i]
        num_microbins = 2 ** k
        cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)
        y_cut = func(cutpoints)
        y_cut = y_cut.clamp(-65504.0, 65504.0)
        # Share boundary with previous bin only if they are contiguous in x
        if i > 0 and b_start == bins[i-1][1]:
            base_offsets.append(current_offset - 1)  # point to shared boundary
            all_lut_values.append(y_cut[1:])
            current_offset += num_microbins
        else:
            base_offsets.append(current_offset)
            all_lut_values.append(y_cut)
            current_offset += num_microbins + 1

    config.total_lut = sum(2 ** k for k in k_alloc)
    config.bin_starts = torch.tensor([b[0] for b in bins], dtype=torch.float32, device=device)
    config.bin_ends = torch.tensor([b[1] for b in bins], dtype=torch.float32, device=device)
    config.base_offsets = torch.tensor(base_offsets, dtype=torch.long, device=device)
    config.k_bits_tensor = torch.tensor(k_alloc, dtype=torch.long, device=device)
    config.lut_values = torch.cat(all_lut_values)

    return config


# ─────────────────────────────────────────────────────────────
#  EDA-NLI Full Pipeline: Optimize + Build LUT
# ─────────────────────────────────────────────────────────────

class EDAConfig:
    """Configuration for EDA-NLI engine."""
    def __init__(self):
        self.bins = []           # List of (b_start, b_end, sign, exp_val)
        self.k_alloc = []        # k_bits per bin
        self.total_lut = 0       # Total LUT entries used
        self.total_error = 0.0

        # Hardware registers
        self.bin_starts = None   # Tensor: start of each exponent bin
        self.bin_ends = None     # Tensor: end of each exponent bin
        self.base_offsets = None # Tensor: LUT base offset per bin
        self.k_bits = None       # Tensor: mantissa bits per bin
        self.lut_values = None   # Tensor: all LUT function values


def optimize_eda(func_name: str, max_lut: int = 254, max_k: int = 5,
                  device: str = 'cuda', verbose: bool = True,
                  domain: Optional[Tuple[float, float]] = None) -> EDAConfig:
    """
    Full EDA-NLI optimization pipeline.

    1. Generate exponent bins for the function's domain
    2. Precompute error for each (bin, k_bits) pair
    3. Run knapsack DP to optimally allocate mantissa bits
    4. Build the LUT
    """
    t0 = time.time()
    func = get_function(func_name)
    domain = domain or get_domain(func_name)

    if verbose:
        print(f"\n=== EDA-NLI Optimization: {func_name.upper()} ===")
        print(f"  Domain: {domain}, Max LUT: {max_lut}, Max K: {max_k}")

    # Step 1: Generate exponent bins
    bins = get_fp16_exponent_bins(domain)
    if len(bins) == 0:
        raise ValueError(f"No exponent bins in domain {domain}")
    if verbose:
        print(f"  [Step 1] {len(bins)} exponent bins in domain")

    # Step 2: Precompute costs
    if verbose:
        print(f"  [Step 2] Precomputing error costs...")
    err_matrix, weight_matrix = precompute_bin_costs(func, bins, max_k, device)

    # Step 3: Knapsack DP
    if verbose:
        print(f"  [Step 3] Running knapsack DP...")
    total_error, k_alloc = solve_knapsack_dp(err_matrix, weight_matrix, max_lut)

    # Step 4: Build configuration
    config = EDAConfig()
    config.bins = [(b[0], b[1], b[2], b[3]) for b in bins]
    config.k_alloc = k_alloc
    config.total_error = total_error

    # Build LUT (contiguous sharing: adjacent bins share boundary endpoints)
    all_lut_values = []
    base_offsets = []
    current_offset = 0

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        k = k_alloc[i]
        num_microbins = 2 ** k

        # For negative bins, mantissa bit-extraction maps micro_idx=0
        # to the RIGHT edge (least negative). Reverse cutpoint order so
        # LUT[base+0] = f(right_edge), matching HW addressing.
        if sign == 1:
            cutpoints = torch.linspace(b_end, b_start, num_microbins + 1, device=device)
        else:
            cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)
        y_cut = func(cutpoints)
        # Clamp to FP16 representable range: inf in LUT causes NaN during
        # interpolation (inf - inf = NaN), so saturate to ±65504.
        y_cut = y_cut.clamp(-65504.0, 65504.0)
        # Boundary sharing: adjacent bins share one LUT entry.
        # Positive bins (ascending LUT): shared value at y_cut[0] (left edge)
        # Negative bins (reversed LUT): shared value at y_cut[-1] (left edge, stored last)
        contiguous = (i > 0 and abs(b_start - bins[i-1][1]) < 1e-30)
        if contiguous and sign == 0 and bins[i-1][2] == 0:
            # Positive → positive: share first element
            base_offsets.append(current_offset - 1)
            all_lut_values.append(y_cut[1:])
            current_offset += num_microbins
        elif contiguous and sign == 1 and bins[i-1][2] == 1:
            # Negative → negative: share last element (left edge in reversed LUT)
            base_offsets.append(current_offset)
            all_lut_values.append(y_cut[:-1])
            current_offset += num_microbins
        else:
            base_offsets.append(current_offset)
            all_lut_values.append(y_cut)
            current_offset += num_microbins + 1

    config.total_lut = sum(2 ** k for k in k_alloc)
    config.bin_starts = torch.tensor([b[0] for b in bins], dtype=torch.float32)
    config.bin_ends = torch.tensor([b[1] for b in bins], dtype=torch.float32)
    config.base_offsets = torch.tensor(base_offsets, dtype=torch.long)
    config.k_bits_tensor = torch.tensor(k_alloc, dtype=torch.long)
    config.lut_values = torch.cat(all_lut_values)

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  === EDA-NLI Complete ({elapsed:.2f}s) ===")
        print(f"  Total LUT entries: {config.total_lut}")
        print(f"  Total LUT values stored: {len(config.lut_values)}")
        print(f"  Exponent bins: {len(bins)}")
        print(f"  K allocation: {k_alloc}")
        print(f"  Total error: {total_error:.6e}")

        # Print per-bin breakdown
        print(f"\n  {'Exponent Bin':<30s} | {'K':>3s} | {'Micro-bins':>10s} | {'Sign':>4s} | {'Exp':>3s}")
        print(f"  {'-'*30} | {'-'*3} | {'-'*10} | {'-'*4} | {'-'*3}")
        for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
            k = k_alloc[i]
            n_micro = 2 ** k
            label = f"[{b_start:<12g}, {b_end:<12g})"
            print(f"  {label:<30s} | {k:>3d} | {n_micro:>10d} | {sign:>4d} | {exp_val:>3d}")

    return config


# ─────────────────────────────────────────────────────────────
#  Optimal Domain Search
# ─────────────────────────────────────────────────────────────

# Full FP16 evaluation domains per function (from ablation_sweep.py)
EVAL_DOMAINS = {
    'silu':       (-65504.0, 65504.0),
    'gelu':       (-65504.0, 65504.0),
    'exp':        (-65504.0, 11.0859375),   # exp(x>11.09) overflows to inf
    'sigmoid':    (-65504.0, 65504.0),
    'tanh':       (-65504.0, 65504.0),
    'hardswish':  (-65504.0, 65504.0),
    'mish':       (-65504.0, 65504.0),
    'rsqrt':      (5.9604644775390625e-08, 65504.0),  # positive only
    'reciprocal': (1.5318393707275391e-05, 65504.0),   # positive only
}

ALL_FUNCS = ['silu', 'gelu', 'exp', 'sigmoid', 'tanh',
             'hardswish', 'mish', 'rsqrt', 'reciprocal']


def _generate_fp16_grid(domain: Tuple[float, float], device: str = 'cuda') -> torch.Tensor:
    """Generate sorted array of all finite FP16 values within domain."""
    all_vals = []
    for bits in range(0, 0x7C00):
        val = struct.unpack('e', struct.pack('H', bits))[0]
        if domain[0] <= val <= domain[1]:
            all_vals.append(val)
        neg = -val
        if val != 0 and domain[0] <= neg <= domain[1]:
            all_vals.append(neg)
    vals = sorted(set(all_vals))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _eval_config_on_grid(config: EDAConfig, func_name: str,
                          device: str = 'cuda') -> float:
    """Evaluate EDA config on full FP16 grid, return mean relative error."""
    from ablation_sweep import eda_forward_with_config
    domain = EVAL_DOMAINS.get(func_name, (-65504.0, 65504.0))
    grid = _generate_fp16_grid(domain, device)
    func = get_function(func_name)
    y_ref = func(grid.float())

    y_approx = eda_forward_with_config(grid, config, t_bits='adaptive')

    valid = torch.isfinite(y_ref)
    y_ref = y_ref[valid]
    y_approx = y_approx[valid]

    abs_err = torch.abs(y_approx.float() - y_ref)
    denom = torch.clamp(torch.abs(y_ref), min=TAU)
    rel_err = abs_err / denom
    return rel_err.mean().item()


def _generate_domain_candidates(func_name: str) -> List[Tuple[float, float]]:
    """Generate power-of-2 domain boundary candidates for a function.

    FP16 exponent bins are power-of-2 aligned, so only power-of-2 boundaries
    matter. Domain can extend BEYOND eval grid — the eval grid measures true
    error, while wider domain gives LUT coverage (less clamping error).
    """
    # All power-of-2 boundaries in full FP16 range
    pos_bounds = [2.0 ** e for e in range(-24, 16)]
    pos_bounds.append(65504.0)
    neg_bounds = [-b for b in pos_bounds if b != 65504.0]
    neg_bounds.append(-65504.0)
    all_bounds = sorted(set(pos_bounds + neg_bounds + [0.0]))

    if func_name in ('rsqrt', 'reciprocal'):
        # Positive-only: lo varies (all positive p2), hi=65504 fixed
        return [(lo, 65504.0) for lo in all_bounds
                if 0 < lo < 65504.0]

    if func_name in ('silu', 'gelu', 'mish'):
        # Activation: negative lo varies, hi=65504 fixed
        return [(lo, 65504.0) for lo in all_bounds if lo < 0]

    if func_name == 'hardswish':
        return [(lo, 65504.0) for lo in all_bounds
                if -32.0 <= lo < 0]

    if func_name == 'exp':
        # lo: negative, hi: 4..16 (need to cover up to exp overflow ~11.09)
        lo_cands = [b for b in all_bounds if b < 0]
        hi_cands = [b for b in all_bounds if 4 <= b <= 16]
        return [(lo, hi) for lo in lo_cands for hi in hi_cands]

    if func_name == 'sigmoid':
        lo_cands = [b for b in all_bounds if b < 0]
        hi_cands = [b for b in all_bounds if 4 <= b <= 32]
        return [(lo, hi) for lo in lo_cands for hi in hi_cands]

    if func_name == 'tanh':
        lo_cands = [b for b in all_bounds if -32 <= b < 0]
        hi_cands = [b for b in all_bounds if 0 < b <= 32]
        return [(lo, hi) for lo in lo_cands for hi in hi_cands]

    # Fallback
    return [(lo, hi) for lo in all_bounds for hi in all_bounds if hi > lo]


def find_optimal_domain(func_name: str, max_lut: int = 256, max_k: int = 5,
                         device: str = 'cuda') -> Dict:
    """Exhaustive search over power-of-2 domain boundaries.

    For each candidate (lo, hi), runs knapsack DP + evaluates on full FP16 grid
    (including out-of-domain clamping error).

    Returns dict with 'domain', 'mean_rel', 'all_results', etc.
    """
    candidates = _generate_domain_candidates(func_name)
    print(f"\n{'='*70}")
    print(f"  Optimal Domain Search: {func_name.upper()}")
    print(f"  {len(candidates)} candidates, max_lut={max_lut}, max_k={max_k}")
    print(f"{'='*70}")

    current_domain = get_domain(func_name)

    results = []
    best_err = float('inf')
    best_domain = None
    t0 = time.time()

    for i, (lo, hi) in enumerate(candidates):
        try:
            config = optimize_eda(func_name, max_lut=max_lut, max_k=max_k,
                                   device=device, verbose=False, domain=(lo, hi))
            n_bins = len(config.bins)
            mean_rel = _eval_config_on_grid(config, func_name, device)
            results.append((lo, hi, n_bins, mean_rel, config.total_error))

            if mean_rel < best_err:
                best_err = mean_rel
                best_domain = (lo, hi)

            if (i + 1) % 10 == 0 or i == len(candidates) - 1:
                print(f"  [{i+1}/{len(candidates)}] best so far: "
                      f"{best_err*1e4:.4f}×10⁻⁴ @ ({best_domain[0]:g}, {best_domain[1]:g})")

        except ValueError:
            results.append((lo, hi, -1, float('inf'), float('inf')))

    elapsed = time.time() - t0
    results.sort(key=lambda r: r[3])

    # Print top 10
    print(f"\n  Top 10 domains (of {len(candidates)} searched, {elapsed:.1f}s):")
    print(f"  {'lo':>12s} {'hi':>12s} {'bins':>5s} {'mean_rel×1e4':>13s} {'knapsack_err':>13s}")
    print(f"  {'-'*12} {'-'*12} {'-'*5} {'-'*13} {'-'*13}")
    for lo, hi, n_bins, mean_rel, ks_err in results[:10]:
        mark = ' <-- current' if (lo, hi) == current_domain else ''
        print(f"  {lo:>12g} {hi:>12g} {n_bins:>5d} {mean_rel*1e4:>13.4f} {ks_err:>13.4e}{mark}")

    current_rank = next((i for i, r in enumerate(results)
                         if r[0] == current_domain[0] and r[1] == current_domain[1]), None)
    if current_rank is not None:
        r = results[current_rank]
        print(f"\n  Current domain ({current_domain}): rank {current_rank+1}/{len(results)}, "
              f"mean_rel={r[3]*1e4:.4f}×10⁻⁴")

    print(f"\n  OPTIMAL: ({best_domain[0]:g}, {best_domain[1]:g}) → "
          f"mean_rel={best_err*1e4:.4f}×10⁻⁴")

    return {
        'domain': list(best_domain),
        'mean_rel': best_err,
        'n_candidates': len(candidates),
        'elapsed_s': elapsed,
        'max_lut': max_lut,
        'max_k': max_k,
        'top10': [
            {'lo': lo, 'hi': hi, 'n_bins': n_bins,
             'mean_rel': mean_rel, 'knapsack_err': ks_err}
            for lo, hi, n_bins, mean_rel, ks_err in results[:10]
        ],
    }


def save_optimal_domains(results: Dict[str, Dict], path: str = None):
    """Save domain search results to JSON."""
    path = path or _OPTIMAL_DOMAINS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved optimal domains → {path}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EDA-NLI: Exponent-Direct Addressing')
    parser.add_argument('--func', type=str, default='silu')
    parser.add_argument('--max_lut', type=int, default=256)
    parser.add_argument('--max_k', type=int, default=5)
    parser.add_argument('--optimize-domain', action='store_true',
                        help='Run exhaustive domain search')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.optimize_domain:
        funcs = ALL_FUNCS if args.func == 'all' else [args.func]
        # Load existing results to allow incremental runs
        if os.path.exists(_OPTIMAL_DOMAINS_PATH):
            with open(_OPTIMAL_DOMAINS_PATH) as f:
                all_results = json.load(f)
        else:
            all_results = {}
        for fn in funcs:
            all_results[fn] = find_optimal_domain(
                fn, max_lut=args.max_lut, max_k=args.max_k, device=device)
        save_optimal_domains(all_results)
        _invalidate_domain_cache()
    else:
        config = optimize_eda(args.func, max_lut=args.max_lut, max_k=args.max_k, device=device)
