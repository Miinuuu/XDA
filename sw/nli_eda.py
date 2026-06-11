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

    Loads from the canonical profiling results cache if available.
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
#  Learned LUT Values: Interpolation-Optimal via Least Squares
# ─────────────────────────────────────────────────────────────

def _optimize_lut_values_for_bin(func: Callable, b_start: float, b_end: float,
                                  k_bits: int, device: str = 'cuda'
                                  ) -> torch.Tensor:
    """
    Find interpolation-optimal LUT values for a single exponent bin.

    Solves weighted least squares to minimize sum of squared relative errors:
        min_y  Σ_x  ((f(x) - lerp(x; y)) / max(|f(x)|, τ))²

    where lerp(x; y) = (1 - t_q) * y_i + t_q * y_{i+1}
    and t_q = floor(t * 2^T) / 2^T  (hardware-quantized fraction).

    Returns: Tensor of optimal y values (num_microbins + 1 entries) on `device`.
    """
    num_microbins = 2 ** k_bits
    N = num_microbins + 1
    cutpoints_np = np.linspace(b_start, b_end, N, dtype=np.float64)

    x = _fp16_grid_for_bin(b_start, b_end, device)
    if len(x) < N:
        # Too few points for regression — fall back to exact values
        cp = torch.tensor(cutpoints_np, dtype=torch.float32, device=device)
        return func(cp).clamp(-65504.0, 65504.0)

    y_true = func(x).cpu().double().numpy()
    x_np = x.cpu().double().numpy()

    # Micro-bin assignment and quantized interpolation fraction
    idx = np.searchsorted(cutpoints_np, x_np) - 1
    idx = np.clip(idx, 0, num_microbins - 1)

    x0 = cutpoints_np[idx]
    x1 = cutpoints_np[idx + 1]
    dx = x1 - x0

    T_bits = 10 - k_bits
    scale = float(1 << T_bits)
    t = np.where(dx > 0, (x_np - x0) / dx, 0.0)
    t_q = np.floor(t * scale) / scale

    # Build interpolation matrix A (M × N): y_pred = A @ y_vec
    M = len(x_np)
    A = np.zeros((M, N), dtype=np.float64)
    rows = np.arange(M)
    A[rows, idx] = 1.0 - t_q
    A[rows, idx + 1] = t_q

    # Relative-error weights: w_j = 1 / max(|f(x_j)|, τ)
    w = 1.0 / np.maximum(np.abs(y_true), TAU)

    # Weighted least squares: min ‖diag(w)(A y − f)‖²
    WA = A * w[:, None]
    Wf = y_true * w
    y_opt, _, _, _ = np.linalg.lstsq(WA, Wf, rcond=None)

    y_opt = np.clip(y_opt, -65504.0, 65504.0)
    return torch.tensor(y_opt, dtype=torch.float32, device=device)


def compute_bin_error_learned(func: Callable, b_start: float, b_end: float,
                               k_bits: int, device: str = 'cuda',
                               **_kwargs) -> Tuple[float, int]:
    """
    Compute interpolation error using learned-optimal LUT values.

    Same interface as compute_bin_error() but replaces exact f(cutpoints)
    with weighted-least-squares optimized values.

    Returns: (total_relative_error, num_lut_entries)
    """
    num_microbins = 2 ** k_bits

    x = _fp16_grid_for_bin(b_start, b_end, device)
    if len(x) == 0:
        return 0.0, num_microbins
    y_true = func(x)

    y_opt = _optimize_lut_values_for_bin(func, b_start, b_end, k_bits, device)
    cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)

    idx = torch.searchsorted(cutpoints, x) - 1
    idx = torch.clamp(idx, 0, num_microbins - 1)

    x0 = cutpoints[idx]
    x1 = cutpoints[idx + 1]
    y0 = y_opt[idx]
    y1 = y_opt[idx + 1]

    dx = x1 - x0
    t = torch.where(dx > 0, (x - x0) / dx, torch.zeros_like(dx))
    T_bits = 10 - k_bits
    scale = float(1 << T_bits)
    t_q = torch.floor(t * scale) / scale
    y_pred = y0 + t_q * (y1 - y0)

    abs_err = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=TAU)
    rel_err = abs_err / denom

    return rel_err.sum().item(), num_microbins


# ─────────────────────────────────────────────────────────────
#  Knapsack DP: Allocate mantissa bits per exponent bin
# ─────────────────────────────────────────────────────────────

def precompute_bin_costs(func: Callable, bins: List[Tuple[float, float, int, int]],
                         max_k: int = 5, device: str = 'cuda',
                         samples_per_bin: int = 10000,
                         lut_mode: str = 'exact'
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute error and LUT weight for each (bin, k_bits) pair.

    Args:
        lut_mode: 'exact' uses f(cutpoints); 'learned' uses least-squares
                  optimized LUT values that minimize interpolation error.

    Returns:
        err_matrix: (N_bins, max_k+1) - error for each bin with k mantissa bits
        weight_matrix: (N_bins, max_k+1) - LUT entries needed (2^k per bin)
    """
    N = len(bins)
    K = max_k + 1
    err_matrix = np.full((N, K), np.inf, dtype=np.float64)
    weight_matrix = np.zeros((N, K), dtype=np.int32)

    error_fn = compute_bin_error_learned if lut_mode == 'learned' else compute_bin_error

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        for k in range(K):
            err, n_entries = error_fn(func, b_start, b_end, k,
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
#  Minimax Solver: minimize worst-case per-bin error
# ─────────────────────────────────────────────────────────────

def solve_minimax(err_matrix: np.ndarray, weight_matrix: np.ndarray,
                  max_lut: int) -> Tuple[float, List[int]]:
    """
    Minimax optimization: minimize the maximum per-bin error.

    Binary search on error threshold T.
    For each T, greedily assign minimum-cost K to each bin s.t. err <= T.
    After finding T*, use remaining LUT budget to further reduce total error.

    Returns: (best_error, k_alloc) where best_error = max per-bin error.
    """
    N, K = err_matrix.shape

    # Collect all unique error values as candidate thresholds
    candidates = np.unique(err_matrix[np.isfinite(err_matrix)])
    candidates.sort()

    def feasible(threshold):
        """Check if threshold is achievable within LUT budget.
        Returns (is_feasible, k_alloc, total_weight)."""
        total_weight = 0
        alloc = []
        for i in range(N):
            best_k = -1
            best_w = max_lut + 1
            for k in range(K):
                if err_matrix[i, k] <= threshold and weight_matrix[i, k] < best_w:
                    best_k = k
                    best_w = int(weight_matrix[i, k])
            if best_k == -1:
                return False, [], 0
            total_weight += best_w
            alloc.append(best_k)
        return total_weight <= max_lut, alloc, total_weight

    # Binary search on candidate thresholds
    lo, hi = 0, len(candidates) - 1
    best_alloc = None
    best_threshold = float('inf')

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, alloc, tw = feasible(candidates[mid])
        if ok:
            best_alloc = alloc
            best_threshold = candidates[mid]
            hi = mid - 1
        else:
            lo = mid + 1

    if best_alloc is None:
        raise ValueError("Cannot find feasible minimax solution")

    # Phase 2: use remaining LUT budget to reduce total error (greedy)
    remaining = max_lut - sum(int(weight_matrix[i, best_alloc[i]]) for i in range(N))
    if remaining > 0:
        # For each bin, compute marginal error reduction per LUT cost for each K upgrade
        improvements = []
        for i in range(N):
            curr_k = best_alloc[i]
            curr_err = err_matrix[i, curr_k]
            for k in range(curr_k + 1, K):
                new_err = err_matrix[i, k]
                if new_err < curr_err:
                    cost = int(weight_matrix[i, k]) - int(weight_matrix[i, curr_k])
                    if cost > 0:
                        efficiency = (curr_err - new_err) / cost
                        improvements.append((efficiency, i, k, cost))
        # Greedy: pick best efficiency first
        improvements.sort(reverse=True)
        for eff, i, k, cost in improvements:
            if cost <= remaining:
                old_k = best_alloc[i]
                if err_matrix[i, k] < err_matrix[i, old_k]:
                    remaining -= cost
                    best_alloc[i] = k

    total_error = sum(err_matrix[i, best_alloc[i]] for i in range(N))
    return best_threshold, best_alloc


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
                                 max_lut=254, max_k=5, device='cuda',
                                 lut_mode='exact'):
    """Build EDAConfig using specified allocation strategy.

    Args:
        alloc_strategy: 'knapsack' | 'uniform' | 'curvature'
        lut_mode: 'exact' | 'learned'
    """
    func = get_function(func_name)
    domain = get_domain(func_name)
    bins = get_fp16_exponent_bins(domain)

    if alloc_strategy == 'knapsack':
        err_matrix, weight_matrix = precompute_bin_costs(
            func, bins, max_k, device, lut_mode=lut_mode)
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
    config.lut_mode = lut_mode

    base_offsets, lut_values = _build_lut_layout(bins, k_alloc, func, lut_mode, device)

    config.total_lut = sum(2 ** k for k in k_alloc)
    config.bin_starts = torch.tensor([b[0] for b in bins], dtype=torch.float32, device=device)
    config.bin_ends = torch.tensor([b[1] for b in bins], dtype=torch.float32, device=device)
    config.base_offsets = torch.tensor(base_offsets, dtype=torch.long, device=device)
    config.k_bits_tensor = torch.tensor(k_alloc, dtype=torch.long, device=device)
    config.lut_values = lut_values

    return config


def _build_lut_layout(bins, k_alloc, func, lut_mode='exact', device='cuda',
                      hw_order=False):
    """Build the contiguous LUT array and per-bin base offsets.

    Both layouts realize the identical piecewise-linear approximant (same
    cutpoints, same segments); they differ only in storage direction:

    hw_order=False (value-axis, default): bins stored in value-ascending
    order, entries run b_start -> b_end. Used by the Python evaluation
    pipeline and all paper accuracy tables.

    hw_order=True (mantissa-axis): entry j corresponds to mantissa
    micro-index j, so negative-bin entries run b_end -> b_start (|x|
    ascending) and negative bins are stored outward from zero. This is the
    layout the RTL consumes (micro index = top-K mantissa bits); used by
    gen_eda_mem.py when emitting func_lut.mem / config_rom.mem.

    Adjacent same-storage-direction contiguous bins share one boundary entry
    via base_offset-1 + y_cut[1:]. base_offsets are returned in the original
    bin order.
    """
    n = len(bins)
    if hw_order:
        order = [i for i in range(n) if bins[i][2] == 1][::-1] + \
                [i for i in range(n) if bins[i][2] == 0]
    else:
        order = list(range(n))
    base_offsets = [0] * n
    all_vals = []
    current_offset = 0
    prev = None

    for i in order:
        b_start, b_end, sign, _exp = bins[i]
        k = k_alloc[i]
        m = 2 ** k
        rev = hw_order and sign == 1
        if rev:
            cutpoints = torch.linspace(b_end, b_start, m + 1, device=device)
        else:
            cutpoints = torch.linspace(b_start, b_end, m + 1, device=device)
        if lut_mode == 'learned':
            y_cut = _optimize_lut_values_for_bin(func, b_start, b_end, k, device)
            if rev:
                y_cut = torch.flip(y_cut, [0])
        else:
            y_cut = func(cutpoints)
        # Clamp to FP16 representable range: inf in LUT causes NaN during
        # interpolation (inf - inf = NaN), so saturate to ±65504.
        y_cut = y_cut.clamp(-65504.0, 65504.0)

        # Share one boundary entry when the previously *stored* bin ends at
        # this bin's entry coordinate with the same storage direction.
        share = False
        if prev is not None and (bins[prev][2] == sign or not hw_order):
            prev_rev = hw_order and bins[prev][2] == 1
            prev_exit = bins[prev][0] if prev_rev else bins[prev][1]
            cur_entry = b_end if rev else b_start
            share = (prev_exit == cur_entry) and (prev_rev == rev)
        if share:
            base_offsets[i] = current_offset - 1
            all_vals.append(y_cut[1:])
            current_offset += m
        else:
            base_offsets[i] = current_offset
            all_vals.append(y_cut)
            current_offset += m + 1
        prev = i

    return base_offsets, torch.cat(all_vals)


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
        self.lut_mode = 'exact'  # 'exact' | 'learned'

        # Hardware registers
        self.bin_starts = None   # Tensor: start of each exponent bin
        self.bin_ends = None     # Tensor: end of each exponent bin
        self.base_offsets = None # Tensor: LUT base offset per bin
        self.k_bits = None       # Tensor: mantissa bits per bin
        self.lut_values = None   # Tensor: all LUT function values


def optimize_eda(func_name: str, max_lut: int = 254, max_k: int = 5,
                  device: str = 'cuda', verbose: bool = True,
                  domain: Optional[Tuple[float, float]] = None,
                  lut_mode: str = 'exact', hw_order: bool = False) -> EDAConfig:
    """
    Full EDA-NLI optimization pipeline.

    1. Generate exponent bins for the function's domain
    2. Precompute error for each (bin, k_bits) pair
    3. Run knapsack DP to optimally allocate mantissa bits
    4. Build the LUT

    Args:
        lut_mode: 'exact' stores f(cutpoints); 'learned' stores
                  least-squares optimized values that minimize
                  interpolation error (zero hardware change).
    """
    t0 = time.time()
    func = get_function(func_name)
    domain = domain or get_domain(func_name)

    if verbose:
        mode_label = f", lut_mode={lut_mode}" if lut_mode != 'exact' else ""
        print(f"\n=== EDA-NLI Optimization: {func_name.upper()}{mode_label} ===")
        print(f"  Domain: {domain}, Max LUT: {max_lut}, Max K: {max_k}")

    # Step 1: Generate exponent bins
    bins = get_fp16_exponent_bins(domain)
    if len(bins) == 0:
        raise ValueError(f"No exponent bins in domain {domain}")
    if verbose:
        print(f"  [Step 1] {len(bins)} exponent bins in domain")

    # Step 2: Precompute costs
    if verbose:
        print(f"  [Step 2] Precomputing error costs ({lut_mode})...")
    err_matrix, weight_matrix = precompute_bin_costs(
        func, bins, max_k, device, lut_mode=lut_mode)

    # Step 3: Knapsack DP
    if verbose:
        print(f"  [Step 3] Running knapsack DP...")
    total_error, k_alloc = solve_knapsack_dp(err_matrix, weight_matrix, max_lut)

    # Step 4: Build configuration
    config = EDAConfig()
    config.bins = [(b[0], b[1], b[2], b[3]) for b in bins]
    config.k_alloc = k_alloc
    config.total_error = total_error
    config.lut_mode = lut_mode

    # Build LUT via the shared HW-faithful layout helper (negative bins
    # stored |x|-ascending; same-sign contiguous bins share one entry).
    base_offsets, lut_values = _build_lut_layout(bins, k_alloc, func, lut_mode, device, hw_order=hw_order)

    config.total_lut = sum(2 ** k for k in k_alloc)
    config.bin_starts = torch.tensor([b[0] for b in bins], dtype=torch.float32)
    config.bin_ends = torch.tensor([b[1] for b in bins], dtype=torch.float32)
    config.base_offsets = torch.tensor(base_offsets, dtype=torch.long)
    config.k_bits_tensor = torch.tensor(k_alloc, dtype=torch.long)
    config.lut_values = lut_values

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


# =============================================================================
#  Multi-Format XDA: Format-Agnostic Nonlinear Approximation
# =============================================================================

class NumberFormat:
    """IEEE 754-style floating-point format descriptor.

    Supports FP16, BF16, FP8 E5M2, FP8 E4M3fn, and custom formats.
    Provides exponent-bin generation and exhaustive value grid for XDA.
    """

    def __init__(self, name: str, exp_bits: int, mant_bits: int, bias: int,
                 has_inf: bool = True):
        self.name = name
        self.exp_bits = exp_bits
        self.mant_bits = mant_bits
        self.bias = bias
        self.total_bits = 1 + exp_bits + mant_bits
        self.has_inf = has_inf  # False for E4M3fn

        self.max_exp = (1 << exp_bits) - 1  # all-ones exponent
        self.max_k = min(mant_bits, 5)

    @property
    def max_normal(self) -> float:
        if self.has_inf:
            e = self.max_exp - 1
            m = (1 << self.mant_bits) - 1
        else:
            # E4M3fn: exp=max_exp is valid, only top mantissa pattern is NaN
            e = self.max_exp
            m = (1 << self.mant_bits) - 2
        return 2.0 ** (e - self.bias) * (1.0 + m / (1 << self.mant_bits))

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias) / (1 << self.mant_bits)

    def _decode_positive(self, bits: int) -> Optional[float]:
        """Decode unsigned bit pattern to positive float. None for inf/nan."""
        exp = (bits >> self.mant_bits) & ((1 << self.exp_bits) - 1)
        mant = bits & ((1 << self.mant_bits) - 1)

        if self.has_inf:
            if exp == self.max_exp:
                return None
        else:
            if exp == self.max_exp and mant == (1 << self.mant_bits) - 1:
                return None

        if exp == 0:
            return 2.0 ** (1 - self.bias) * (mant / (1 << self.mant_bits))
        return 2.0 ** (exp - self.bias) * (1.0 + mant / (1 << self.mant_bits))

    def _max_positive_bits(self) -> int:
        if self.has_inf:
            return ((self.max_exp - 1) << self.mant_bits) | ((1 << self.mant_bits) - 1)
        return (self.max_exp << self.mant_bits) | ((1 << self.mant_bits) - 2)

    def generate_grid(self, domain: Tuple[float, float],
                      device: str = 'cpu') -> torch.Tensor:
        """All representable values within domain."""
        vals = []
        for bits in range(0, self._max_positive_bits() + 1):
            val = self._decode_positive(bits)
            if val is None:
                continue
            if domain[0] <= val <= domain[1]:
                vals.append(val)
            if val != 0 and domain[0] <= -val <= domain[1]:
                vals.append(-val)
        return torch.tensor(sorted(set(vals)), dtype=torch.float32, device=device)

    def generate_grid_for_bin(self, b_start: float, b_end: float,
                              device: str = 'cpu') -> torch.Tensor:
        """All representable values within a single exponent bin.

        O(2^mant_bits) per bin instead of O(total_values) by directly
        computing mantissa values for the matching exponent.
        """
        vals = []
        n_mant = 1 << self.mant_bits
        top_exp = self.max_exp if not self.has_inf else self.max_exp - 1

        for sign in (0, 1):
            for e in range(0, top_exp + 1):
                if e == 0:
                    lo_e = 0.0
                    hi_e = 2.0 ** (1 - self.bias)
                else:
                    lo_e = 2.0 ** (e - self.bias)
                    hi_e = 2.0 ** (e - self.bias + 1)
                if sign:
                    lo_e, hi_e = -hi_e, -lo_e

                # Quick reject: bin doesn't overlap
                if hi_e <= b_start or lo_e >= b_end:
                    continue

                max_m = n_mant - 1
                if not self.has_inf and e == self.max_exp:
                    max_m = n_mant - 2  # E4M3fn: top mant is NaN

                for m in range(0, max_m + 1):
                    if e == 0:
                        v = 2.0 ** (1 - self.bias) * (m / n_mant)
                    else:
                        v = 2.0 ** (e - self.bias) * (1.0 + m / n_mant)
                    if sign:
                        v = -v
                    if b_start <= v <= b_end:
                        vals.append(v)

        if not vals:
            return torch.tensor([], dtype=torch.float32, device=device)
        return torch.tensor(sorted(set(vals)), dtype=torch.float32, device=device)

    def get_exponent_bins(self, domain: Tuple[float, float]
                          ) -> List[Tuple[float, float, int, int]]:
        """Exponent-aligned bins within domain, same format as FP16 version."""
        bins = []
        top_exp = self.max_exp if not self.has_inf else self.max_exp - 1

        # Positive normals
        for e in range(1, top_exp + 1):
            lo = 2.0 ** (e - self.bias)
            hi = 2.0 ** (e - self.bias + 1)
            if not self.has_inf and e == self.max_exp:
                hi = min(hi, self.max_normal * 1.001)
            bins.append((lo, hi, 0, e))

        # Positive subnormals
        bins.append((self.min_subnormal, 2.0 ** (1 - self.bias), 0, 0))

        # Negative normals
        for e in range(1, top_exp + 1):
            lo = -(2.0 ** (e - self.bias + 1))
            hi = -(2.0 ** (e - self.bias))
            if not self.has_inf and e == self.max_exp:
                lo = max(lo, -self.max_normal * 1.001)
            bins.append((lo, hi, 1, e))

        # Negative subnormals
        bins.append((-(2.0 ** (1 - self.bias)), -self.min_subnormal, 1, 0))

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

    def __repr__(self):
        return (f"NumberFormat({self.name}: E{self.exp_bits}M{self.mant_bits}, "
                f"bias={self.bias}, max={self.max_normal:.4g})")


# ── Predefined formats ──────────────────────────────────────

FP16     = NumberFormat('fp16',     exp_bits=5, mant_bits=10, bias=15)
BF16     = NumberFormat('bf16',     exp_bits=8, mant_bits=7,  bias=127)
FP8_E5M2 = NumberFormat('fp8_e5m2', exp_bits=5, mant_bits=2,  bias=15)
FP8_E4M3 = NumberFormat('fp8_e4m3', exp_bits=4, mant_bits=3,  bias=7, has_inf=False)


def get_format(name: str) -> NumberFormat:
    _FORMATS = {'fp16': FP16, 'bf16': BF16, 'fp8_e5m2': FP8_E5M2, 'fp8_e4m3': FP8_E4M3}
    if name not in _FORMATS:
        raise ValueError(f"Unknown format: {name}. Available: {list(_FORMATS.keys())}")
    return _FORMATS[name]


# ── Format-aware domains ────────────────────────────────────

def get_domain_mf(func_name: str, fmt: NumberFormat) -> Tuple[float, float]:
    """Function domain clipped to format's representable range."""
    base = get_domain(func_name)
    fmax = fmt.max_normal
    return (max(base[0], -fmax), min(base[1], fmax))


# ── Multi-format error computation ──────────────────────────

def compute_bin_error_mf(func: Callable, b_start: float, b_end: float,
                         k_bits: int, fmt: NumberFormat,
                         device: str = 'cpu') -> Tuple[float, int]:
    """Format-aware interpolation error for a single exponent bin."""
    num_microbins = 2 ** k_bits
    T_bits = fmt.mant_bits - k_bits
    if T_bits < 0:
        return float('inf'), num_microbins

    x = fmt.generate_grid_for_bin(b_start, b_end, device)
    if len(x) == 0:
        return 0.0, num_microbins

    y_true = func(x)
    cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)
    y_cut = func(cutpoints)

    idx = torch.searchsorted(cutpoints, x) - 1
    idx = torch.clamp(idx, 0, num_microbins - 1)

    x0 = cutpoints[idx]
    x1 = cutpoints[idx + 1]
    y0 = y_cut[idx]
    y1 = y_cut[idx + 1]

    dx = x1 - x0
    t = torch.where(dx > 0, (x - x0) / dx, torch.zeros_like(dx))
    scale = float(1 << T_bits) if T_bits > 0 else 1.0
    t_q = torch.floor(t * scale) / scale
    y_pred = y0 + t_q * (y1 - y0)

    abs_err = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=TAU)
    rel_err = abs_err / denom

    return rel_err.sum().item(), num_microbins


def precompute_bin_costs_mf(func: Callable,
                            bins: List[Tuple[float, float, int, int]],
                            fmt: NumberFormat,
                            device: str = 'cpu'
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute error matrix for multi-format Knapsack."""
    N = len(bins)
    K = fmt.max_k + 1
    err_matrix = np.full((N, K), np.inf, dtype=np.float64)
    weight_matrix = np.zeros((N, K), dtype=np.int32)

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        for k in range(K):
            err, n_entries = compute_bin_error_mf(
                func, b_start, b_end, k, fmt, device)
            err_matrix[i, k] = err
            weight_matrix[i, k] = n_entries

    return err_matrix, weight_matrix


def optimize_eda_mf(func_name: str, fmt: NumberFormat,
                    max_lut: int = 254, device: str = 'cpu',
                    verbose: bool = True) -> EDAConfig:
    """Multi-format XDA optimization pipeline.

    Same Knapsack DP, but bin structure and T_bits adapt to the format.
    """
    t0 = time.time()
    func = get_function(func_name)
    domain = get_domain_mf(func_name, fmt)

    if verbose:
        print(f"\n=== XDA-MF: {func_name.upper()} / {fmt.name} ===")
        print(f"  Format: {fmt}, Domain: {domain}, Budget: {max_lut}")

    bins = fmt.get_exponent_bins(domain)
    if len(bins) == 0:
        raise ValueError(f"No bins in domain {domain} for {fmt.name}")

    # Budget must cover at least 1 entry per bin (K=0)
    effective_budget = max(max_lut, len(bins))

    if verbose:
        budget_note = f" (auto→{effective_budget})" if effective_budget != max_lut else ""
        print(f"  {len(bins)} bins, max_k={fmt.max_k}, budget={effective_budget}{budget_note}")

    err_matrix, weight_matrix = precompute_bin_costs_mf(
        func, bins, fmt, device)
    total_error, k_alloc = solve_knapsack_dp(err_matrix, weight_matrix, effective_budget)

    config = EDAConfig()
    config.bins = [(b[0], b[1], b[2], b[3]) for b in bins]
    config.k_alloc = k_alloc
    config.total_error = total_error
    config.lut_mode = 'exact'

    all_lut_values = []
    base_offsets = []
    current_offset = 0

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        k = k_alloc[i]
        num_microbins = 2 ** k
        cutpoints = torch.linspace(b_start, b_end, num_microbins + 1, device=device)
        y_cut = func(cutpoints).clamp(-fmt.max_normal, fmt.max_normal)

        if i > 0 and abs(b_start - bins[i-1][1]) < 1e-30:
            base_offsets.append(current_offset - 1)
            all_lut_values.append(y_cut[1:])
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
        print(f"  Done ({elapsed:.2f}s): {len(bins)} bins, "
              f"{config.total_lut} LUT entries, error={total_error:.4e}")

    return config


def eval_mf_on_grid(func_name: str, config: EDAConfig,
                    fmt: NumberFormat, device: str = 'cpu') -> Dict:
    """Evaluate XDA config on exhaustive grid of the given format.

    Uses EVAL_DOMAINS (full range) clipped to format, with out-of-domain
    clamping to boundary LUT values — matching the paper's methodology.
    """
    # Eval domain: full function eval range clipped to format
    func_eval_dom = EVAL_DOMAINS.get(func_name, (-65504.0, 65504.0))
    fmax = fmt.max_normal
    eval_dom = (max(func_eval_dom[0], -fmax), min(func_eval_dom[1], fmax))

    grid = fmt.generate_grid(eval_dom, device)
    func = get_function(func_name)
    y_ref = func(grid.float())

    x_flat = grid.reshape(-1).float()
    lut = config.lut_values.to(device)
    bin_starts = config.bin_starts.to(device)
    bin_ends = config.bin_ends.to(device)
    n_bins = len(config.bins)

    # Out-of-optimization-domain clamping
    opt_lo = bin_starts[0].item()
    opt_hi = bin_ends[-1].item()
    too_low = x_flat < opt_lo
    too_high = x_flat > opt_hi
    in_domain = ~too_low & ~too_high

    # Bin assignment via searchsorted on bin_starts (fast, vectorized)
    bin_starts_np = bin_starts.cpu().numpy()
    x_clamped = x_flat.clamp(opt_lo, opt_hi)
    bin_idx = torch.searchsorted(bin_starts, x_clamped) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    b_start_v = bin_starts[bin_idx]
    b_end_v = bin_ends[bin_idx]
    base_offset = config.base_offsets.to(device)[bin_idx]
    k_bits = config.k_bits_tensor.to(device)[bin_idx]

    bin_width = (b_end_v - b_start_v).clamp(min=1e-30)
    rel_pos = ((x_clamped - b_start_v) / bin_width).clamp(0.0, 1.0 - 1e-7)

    num_microbins = (1 << k_bits).float()
    scaled = rel_pos * num_microbins
    micro_idx = scaled.long()
    micro_idx = torch.minimum(micro_idx, (num_microbins - 1).long())

    t = (scaled - micro_idx.float()).clamp(0.0, 1.0)
    per_t = (fmt.mant_bits - k_bits).clamp(min=0)
    t_scale = (1 << per_t).float()
    t = torch.floor(t * t_scale) / t_scale

    idx0 = (base_offset + micro_idx).clamp(0, len(lut) - 1)
    idx1 = (idx0 + 1).clamp(0, len(lut) - 1)
    y0 = lut[idx0]
    y1 = lut[idx1]
    y_approx = y0 + t * (y1 - y0)

    # Out-of-domain: clamp to boundary LUT values
    y_approx = torch.where(too_low, lut[0], y_approx)
    y_approx = torch.where(too_high, lut[-1], y_approx)

    valid = torch.isfinite(y_ref)
    y_ref = y_ref[valid]
    y_approx = y_approx[valid]

    abs_err = torch.abs(y_approx - y_ref)
    denom = torch.clamp(torch.abs(y_ref), min=TAU)
    rel_err = abs_err / denom

    return {
        'mean_rel': rel_err.mean().item(),
        'max_rel': rel_err.max().item(),
        'n_points': int(valid.sum()),
    }


# ── Multi-Format Domain Optimization ────────────────────────

def _generate_domain_candidates_mf(func_name: str, fmt: NumberFormat
                                    ) -> List[Tuple[float, float]]:
    """Power-of-2 domain candidates clipped to format range.

    Restricted to exponents -24..+16 (functional range) to keep
    the search tractable for wide formats like BF16.
    """
    fmax = fmt.max_normal
    # Practical exponent range: functions are trivial beyond 2^16
    exp_lo = max(1 - fmt.bias - fmt.mant_bits, -24)
    exp_hi = min(fmt.max_exp - fmt.bias, 16)

    pos_bounds = [2.0 ** e for e in range(exp_lo, exp_hi + 2) if 2.0 ** e <= fmax]
    pos_bounds.append(fmax)
    pos_bounds = sorted(set(b for b in pos_bounds if b <= fmax))

    neg_bounds = [-b for b in pos_bounds]
    all_bounds = sorted(set(pos_bounds + neg_bounds + [0.0]))

    if func_name in ('rsqrt', 'reciprocal'):
        return [(lo, fmax) for lo in all_bounds if 0 < lo < fmax]
    if func_name in ('silu', 'gelu', 'mish'):
        return [(lo, fmax) for lo in all_bounds if lo < 0]
    if func_name == 'hardswish':
        return [(lo, fmax) for lo in all_bounds if -min(32, fmax) <= lo < 0]
    if func_name == 'exp':
        lo_c = [b for b in all_bounds if b < 0]
        hi_c = [b for b in all_bounds if 4 <= b <= min(16, fmax)] or [fmax]
        return [(lo, hi) for lo in lo_c for hi in hi_c]
    if func_name == 'sigmoid':
        lo_c = [b for b in all_bounds if b < 0]
        hi_c = [b for b in all_bounds if 4 <= b <= min(32, fmax)] or [fmax]
        return [(lo, hi) for lo in lo_c for hi in hi_c]
    if func_name == 'tanh':
        lo_c = [b for b in all_bounds if -min(32, fmax) <= b < 0]
        hi_c = [b for b in all_bounds if 0 < b <= min(32, fmax)]
        return [(lo, hi) for lo in lo_c for hi in hi_c]
    return [(lo, hi) for lo in all_bounds for hi in all_bounds if hi > lo]


def find_optimal_domain_mf(func_name: str, fmt: NumberFormat,
                            max_lut: int = 254,
                            device: str = 'cpu') -> Dict:
    """Exhaustive domain search for a given number format."""
    candidates = _generate_domain_candidates_mf(func_name, fmt)
    if not candidates:
        raise ValueError(f"No domain candidates for {func_name}/{fmt.name}")

    func = get_function(func_name)
    best_err = float('inf')
    best_domain = None
    best_cfg = None

    for lo, hi in candidates:
        try:
            bins = fmt.get_exponent_bins((lo, hi))
            if len(bins) == 0:
                continue
            eff_budget = max(max_lut, len(bins))
            err_mat, w_mat = precompute_bin_costs_mf(func, bins, fmt, device)
            _, k_alloc = solve_knapsack_dp(err_mat, w_mat, eff_budget)

            # Build config
            cfg = EDAConfig()
            cfg.bins = [(b[0], b[1], b[2], b[3]) for b in bins]
            cfg.k_alloc = k_alloc
            all_lut, base_off, cur = [], [], 0
            for i, (bs, be, s, e) in enumerate(bins):
                k = k_alloc[i]
                cp = torch.linspace(bs, be, (2**k)+1, device=device)
                yc = func(cp).clamp(-fmt.max_normal, fmt.max_normal)
                if i > 0 and abs(bs - bins[i-1][1]) < 1e-30:
                    base_off.append(cur - 1); all_lut.append(yc[1:]); cur += 2**k
                else:
                    base_off.append(cur); all_lut.append(yc); cur += 2**k + 1
            cfg.total_lut = sum(2**k for k in k_alloc)
            cfg.bin_starts = torch.tensor([b[0] for b in bins], dtype=torch.float32)
            cfg.bin_ends = torch.tensor([b[1] for b in bins], dtype=torch.float32)
            cfg.base_offsets = torch.tensor(base_off, dtype=torch.long)
            cfg.k_bits_tensor = torch.tensor(k_alloc, dtype=torch.long)
            cfg.lut_values = torch.cat(all_lut)

            res = eval_mf_on_grid(func_name, cfg, fmt, device)
            if res['mean_rel'] < best_err:
                best_err = res['mean_rel']
                best_domain = (lo, hi)
                best_cfg = cfg
        except (ValueError, RuntimeError):
            continue

    return {'domain': list(best_domain) if best_domain else None,
            'mean_rel': best_err, 'config': best_cfg}
