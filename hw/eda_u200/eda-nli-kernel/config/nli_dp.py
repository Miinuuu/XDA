"""
NLI: DP-Optimal Macro Cutpoint Search (Algorithm 1)
=====================================================
Reproduces the dynamic programming cutpoint optimization from:
"Non-uniform Linear Interpolation Approximation of Nonlinear Operations
 for Efficient LLMs Inference" (ICLR 2026)

Given a nonlinear function f and the FP16 grid, finds M optimal macro
cutpoints that minimize the mean relative interpolation error.
"""

import torch
import numpy as np
from typing import Callable, Tuple, List, Optional
import time

try:
    from nli_triton import (
        compute_left_clamp_triton,
        compute_tail_errors_triton,
        compute_err_block_triton,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────
#  FP16 grid generation
# ─────────────────────────────────────────────────────────────

def generate_fp16_grid(domain: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Generate sorted array of all finite FP16 values within domain.
    
    If domain is None, uses full FP16 range.
    Returns 1D tensor on CPU (float32 for precision during DP).
    """
    # FP16: sign(1) + exponent(5) + mantissa(10)
    # Total finite values: 2 * (2^15 - 2^10) + 1 (for zero)
    # Range: [-65504, 65504]
    
    # Generate all positive FP16 values by iterating through bit patterns
    all_vals = []
    for bits in range(0, 0x7C00 + 1):  # 0 to max finite (0x7BFF + 1 = 0x7C00 is inf, so up to 0x7BFF) //31744,        63488이 31744 + - 31744
        if bits >= 0x7C00:  # inf and NaN //  fp16 range: [-65504, 65504]
            break
        # Convert uint16 bit pattern to float16
        val = np.array([bits], dtype=np.uint16).view(np.float16).item()
        if np.isfinite(val):
            all_vals.append(val)
    
    # Add negative values (mirror)
    neg_vals = [-v for v in all_vals if v > 0]
    all_vals = neg_vals + all_vals
    
    # Sort
    all_vals = sorted(set(all_vals))
    grid = torch.tensor(all_vals, dtype=torch.float32)
    
    # Filter to domain if specified
    if domain is not None:
        lo, hi = domain
        mask = (grid >= lo) & (grid <= hi)
        grid = grid[mask]
    
    return grid


# ─────────────────────────────────────────────────────────────
#  Error functionals
# ─────────────────────────────────────────────────────────────

TAU = 2 ** (-14)  # Smallest positive normal in FP16


def segment_error(y: torch.Tensor, x: torch.Tensor, i: int, k: int) -> float:
    """
    Mean relative error of linear interpolation on segment [i, k].

    Err(i→k) = 1/(k-i+1) * Σ_{j=i}^{k} |f(x_j) - P_{i,k}(x_j)| / max(|f(x_j)|, τ)
    """
    if i == k:
        return 0.0

    x_seg = x[i:k+1]
    y_seg = y[i:k+1]

    x_i, x_k = x[i].item(), x[k].item()
    y_i, y_k = y[i].item(), y[k].item()

    if x_k == x_i:
        interp = torch.full_like(y_seg, y_i)
    else:
        slope = (y_k - y_i) / (x_k - x_i)
        interp = y_i + slope * (x_seg - x_i)

    abs_error = torch.abs(y_seg - interp)
    denom = torch.clamp(torch.abs(y_seg), min=TAU)
    rel_error = abs_error / denom

    return rel_error.sum().item()


def left_clamp_error(y: torch.Tensor, k: int) -> float:
    """
    D[0,k] = Σ_{j=0}^{k} |f(x_j) - f(x_k)| / max(|f(x_j)|, τ)
    """
    y_seg = y[:k+1]
    y_k = y[k].item()

    abs_error = torch.abs(y_seg - y_k)
    denom = torch.clamp(torch.abs(y_seg), min=TAU)
    rel_error = abs_error / denom

    return rel_error.sum().item()


def right_clamp_error(y: torch.Tensor, k: int, N: int) -> float:
    """
    last_error(L,k) = Σ_{j=k+1}^{N-1} |f(x_j)-f(x_k)| / max(|f(x_j)|, τ)
    """
    if k >= N - 1:
        return 0.0

    y_tail = y[k+1:N]
    y_k = y[k].item()

    abs_error = torch.abs(y_tail - y_k)
    denom = torch.clamp(torch.abs(y_tail), min=TAU)
    rel_error = abs_error / denom

    return rel_error.sum().item()


# ─────────────────────────────────────────────────────────────
#  Vectorized error computation for GPU acceleration
# ─────────────────────────────────────────────────────────────

def precompute_segment_errors_gpu(x: torch.Tensor, y: torch.Tensor, 
                                   device: str = 'cuda') -> torch.Tensor:
    """
    Precompute Err(i→k) for all valid (i, k) pairs.
    
    Returns a 2D tensor err_table[i, k] of shape (N, N).
    
    WARNING: For full FP16 grid (N≈63488), this requires ~16 GB.
    Use subsampled grid or chunk computation for large N.
    """
    N = len(x)
    x = x.to(device)
    y = y.to(device)
    
    # We'll compute row by row to manage memory
    err_table = torch.zeros(N, N, device=device)
    
    for i in range(N):
        if i % 1000 == 0:
            print(f"  Precomputing errors: {i}/{N}", end='\r')
        
        # For all k > i, compute interpolation on [i, k]
        for k in range(i + 1, N):
            x_i, x_k = x[i], x[k]
            y_i, y_k = y[i], y[k]
            
            if x_k == x_i:
                continue
            
            # Interpolant values at all points in [i, k]
            x_seg = x[i:k+1]
            y_seg = y[i:k+1]
            
            slope = (y_k - y_i) / (x_k - x_i)
            interp = y_i + slope * (x_seg - x_i)
            
            abs_err = torch.abs(y_seg - interp)
            denom = torch.clamp(torch.abs(y_seg), min=TAU)
            err_table[i, k] = (abs_err / denom).sum()
    
    return err_table


# ─────────────────────────────────────────────────────────────
#  DP search (CPU version — practical for M=11 with subsampled grid)
# ─────────────────────────────────────────────────────────────

def dp_cutpoint_search(
    f: Callable[[torch.Tensor], torch.Tensor],
    M: int = 11,
    domain: Optional[Tuple[float, float]] = None,
    max_grid_points: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    DP-Optimal Macro Cutpoint Search (Algorithm 1 from NLI paper).
    
    Args:
        f: Target nonlinear function (operates on float32 tensors)
        M: Number of cutpoints (default 11 for 10 macro-intervals)
        domain: (lo, hi) domain restriction, or None for full FP16 range
        max_grid_points: If set, subsample grid to this many points
        verbose: Print progress
    
    Returns:
        cutpoints: Tensor of M optimal cutpoint x-values
        cutpoint_values: Tensor of M corresponding f(cutpoint) values 
        optimal_cost: The minimum DP cost
    """
    t0 = time.time()
    
    # Step 1: Generate FP16 grid
    if verbose:
        print(f"[Step 1] Generating FP16 grid (domain={domain})...")
    
    x = generate_fp16_grid(domain)
    y = f(x)
    N = len(x)
    
    if verbose:
        print(f"  Grid size: N = {N}")
    
    # Optionally subsample for faster search
    if max_grid_points is not None and N > max_grid_points:
        if verbose:
            print(f"  Subsampling from {N} to {max_grid_points} points...")
        indices = torch.linspace(0, N - 1, max_grid_points).long()
        # Always include first and last
        indices[0] = 0
        indices[-1] = N - 1
        x = x[indices]
        y = y[indices]
        N = len(x)
        if verbose:
            print(f"  New grid size: N = {N}")
    
    # Step 2: Initialize DP tables
    if verbose:
        print(f"[Step 2] Initializing DP tables (M={M}, N={N})...")
    
    D = torch.full((M, N), float('inf'), dtype=torch.float64)
    P = torch.full((M, N), -1, dtype=torch.long)
    
    # Boundary: left clamping for first cutpoint
    if verbose:
        print("[Step 3] Computing left-clamp boundary (D[0, :])...")
    
    for k in range(N):
        D[0, k] = left_clamp_error(y, k)
        P[0, k] = k
    
    # Step 4: Fill DP
    if verbose:
        print("[Step 4] Filling DP table...")
    
    for L in range(1, M):
        t_layer = time.time()
        for k in range(L, N):
            best_val = float('inf')
            best_arg = -1
            
            for i in range(L - 1, k):
                # Compute segment error
                err_ik = segment_error(y, x, i, k)
                
                # Add tail clamping if this is the last cutpoint
                tail = 0.0
                if L == M - 1:
                    tail = right_clamp_error(y, k, N)
                
                val = D[L - 1, i].item() + err_ik + tail
                
                if val < best_val:
                    best_val = val
                    best_arg = i
            
            D[L, k] = best_val
            P[L, k] = best_arg
        
        elapsed = time.time() - t_layer
        if verbose:
            print(f"  Layer L={L}/{M-1} done ({elapsed:.1f}s)")
    
    # Step 5: Backtrack — k* = argmin_k D[M-1, k]  (Algorithm 1)
    if verbose:
        print("[Step 5] Backtracking optimal cutpoints...")

    k_star = D[M - 1, :].argmin().item()
    optimal_cost = D[M - 1, k_star].item() / N

    idx = [0] * M
    idx[M - 1] = k_star
    for L in range(M - 1, 0, -1):
        idx[L - 1] = P[L, idx[L]].item()

    cutpoints = x[idx]
    cutpoint_values = y[idx]

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\n=== DP Search Complete ===")
        print(f"  Total time: {elapsed_total:.1f}s")
        print(f"  Optimal cost (global mean relative error): {optimal_cost:.6e}")
        print(f"  Cutpoints (x): {cutpoints.tolist()}")
        print(f"  Cutpoints (f(x)): {cutpoint_values.tolist()}")

    return cutpoints, cutpoint_values, optimal_cost


# ─────────────────────────────────────────────────────────────
#  Vectorized DP search (GPU-accelerated, for large grids)
# ─────────────────────────────────────────────────────────────

def dp_cutpoint_search_gpu(
    f: Callable[[torch.Tensor], torch.Tensor],
    M: int = 11,
    domain: Optional[Tuple[float, float]] = None,
    max_grid_points: Optional[int] = None,
    device: str = 'cuda',
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    GPU-accelerated DP cutpoint search.
    
    Vectorizes the inner loop over k for each DP layer L.
    Uses cumulative sums for efficient Err computation.
    """
    t0 = time.time()
    
    # Step 1: Generate FP16 grid
    if verbose:
        print(f"[Step 1] Generating FP16 grid (domain={domain})...")
    
    x = generate_fp16_grid(domain)
    y = f(x)
    N = len(x)
    
    if verbose:
        print(f"  Grid size: N = {N}")
    
    if max_grid_points is not None and N > max_grid_points:
        if verbose:
            print(f"  Subsampling from {N} to {max_grid_points} points...")
        indices = torch.linspace(0, N - 1, max_grid_points).long()
        indices[0] = 0
        indices[-1] = N - 1
        x = x[indices]
        y = y[indices]
        N = len(x)
    
    x = x.to(device, dtype=torch.float64)
    y = y.to(device, dtype=torch.float64)
    
    if verbose:
        print(f"  Working grid size: N = {N}")
        print(f"[Step 2] Computing segment error table (vectorized)...")
    
    # Precompute relative error contribution for each point j
    # For segment [i, k]: interpolant P_{i,k}(x_j) = y_i + (y_k - y_i)/(x_k - x_i) * (x_j - x_i)
    # |y_j - P_{i,k}(x_j)| / max(|y_j|, tau)
    # This is O(N^2) to store all Err(i,k), which may be too large.
    # Instead, compute Err on-the-fly per DP layer using vectorization.
    
    # Step 3: Initialize DP
    if verbose:
        print(f"[Step 3] Initializing DP tables (M={M}, N={N})...")

    D = torch.full((M, N), float('inf'), dtype=torch.float64, device=device)
    P = torch.full((M, N), -1, dtype=torch.long, device=device)

    # Left clamp boundary: D[0,k]
    denom_all = torch.clamp(torch.abs(y), min=TAU)

    use_triton = HAS_TRITON and device == 'cuda'

    if use_triton:
        if verbose:
            print("  [Triton] Computing left-clamp + tail errors...")
        D[0, :] = compute_left_clamp_triton(y, denom_all, device)
        P[0, :] = torch.arange(N, device=device, dtype=torch.long)
        tail_errors = compute_tail_errors_triton(y, denom_all, device)
    else:
        for k in range(N):
            y_seg = y[:k+1]
            y_k = y[k]
            abs_err = torch.abs(y_seg - y_k)
            denom = denom_all[:k+1]
            D[0, k] = (abs_err / denom).sum()
            P[0, k] = k
        tail_errors = torch.zeros(N, dtype=torch.float64, device=device)
        for k in range(N - 1):
            y_tail = y[k+1:]
            y_k = y[k]
            abs_err = torch.abs(y_tail - y_k)
            denom = denom_all[k+1:]
            tail_errors[k] = (abs_err / denom).sum()

    if verbose:
        print("[Step 4] Filling DP table...")

    t_start = time.time()

    if use_triton:
        # Triton path: no 3D tensor materialization
        # K_CHUNK sized to keep err_block under ~2 GB
        MAX_ERR_BYTES = 2 * 1024**3
        K_CHUNK = max(1, min(N - 1, MAX_ERR_BYTES // (max(N, 1) * 8 * 2)))
        if verbose:
            print(f"  [Triton] K_CHUNK={K_CHUNK}")

        for k_start in range(1, N, K_CHUNK):
            k_end = min(N, k_start + K_CHUNK)
            k_len = k_end - k_start

            err_block = compute_err_block_triton(x, y, denom_all, k_start, k_end)

            for L in range(1, M):
                vals = D[L-1, :k_end].unsqueeze(1) + err_block
                if L == M - 1:
                    vals = vals + tail_errors[k_start:k_end].unsqueeze(0)
                best_val, best_idx = vals.min(dim=0)
                D[L, k_start:k_end] = best_val
                P[L, k_start:k_end] = best_idx

            if verbose:
                elapsed = time.time() - t_start
                print(f"  Processed k up to {k_end}/{N} ({elapsed:.1f}s)   ", end='\r')
    else:
        # Original PyTorch path (3D broadcasting)
        LIMIT = 64_000_0000
        K_CHUNK = int(max(1, min(N, int(np.sqrt(LIMIT)))))
        I_CHUNK = int(max(1, LIMIT // (K_CHUNK * N)))
        if verbose:
            print(f"  [PyTorch] K_CHUNK={K_CHUNK}, I_CHUNK={I_CHUNK}")

        for k_start in range(1, N, K_CHUNK):
            k_end = min(N, k_start + K_CHUNK)
            k_len = k_end - k_start

            sum_block = torch.zeros((k_end, k_len), dtype=torch.float64, device=device)
            x_j = x[:k_end]
            y_j = y[:k_end]
            denom_j = denom_all[:k_end]
            x_k = x[k_start:k_end]
            y_k = y[k_start:k_end]

            for i_start in range(0, k_end, I_CHUNK):
                i_end = min(k_end, i_start + I_CHUNK)
                i_len = i_end - i_start

                xi = x[i_start:i_end].view(i_len, 1, 1)
                yi = y[i_start:i_end].view(i_len, 1, 1)
                xj = x_j.view(1, -1, 1)
                yj = y_j.view(1, -1, 1)
                dj = denom_j.view(1, -1, 1)
                xk = x_k.view(1, 1, k_len)
                yk = y_k.view(1, 1, k_len)

                dx = xk - xi
                slope = torch.where(dx == 0, torch.zeros_like(dx), (yk - yi) / dx)
                interp = yi + slope * (xj - xi)
                rel_err = torch.abs(yj - interp) / dj

                i_idx = torch.arange(i_start, i_end, device=device).view(i_len, 1, 1)
                k_idx = torch.arange(k_start, k_end, device=device).view(1, 1, k_len)
                j_idx = torch.arange(k_end, device=device).view(1, -1, 1)

                valid = (j_idx >= i_idx) & (j_idx <= k_idx)
                rel_err = rel_err * valid.to(rel_err.dtype)
                sum_block[i_start:i_end, :] = rel_err.sum(dim=1)

            i_idx2 = torch.arange(k_end, device=device).view(-1, 1).to(torch.float64)
            k_idx2 = torch.arange(k_start, k_end, device=device).view(1, -1).to(torch.float64)
            err_block = sum_block
            invalid_mask = i_idx2 >= k_idx2  # i < k strictly (Algorithm 1)
            err_block.masked_fill_(invalid_mask, float('inf'))

            for L in range(1, M):
                vals = D[L-1, :k_end].unsqueeze(1) + err_block
                if L == M - 1:
                    vals = vals + tail_errors[k_start:k_end].unsqueeze(0)
                best_val, best_idx = vals.min(dim=0)
                D[L, k_start:k_end] = best_val
                P[L, k_start:k_end] = best_idx

            if verbose:
                elapsed = time.time() - t_start
                print(f"  Processed k up to {k_end}/{N} ({elapsed:.1f}s)   ", end='\r')
    print(" " * 60, end='\r')  # clear line
    
    # Step 5: Backtrack — k* = argmin_k D[M-1, k]  (Algorithm 1)
    if verbose:
        print("[Step 5] Backtracking optimal cutpoints...")

    k_star = D[M - 1, :].argmin().item()
    optimal_cost = D[M - 1, k_star].item() / N

    idx = [0] * M
    idx[M - 1] = k_star
    for L in range(M - 1, 0, -1):
        idx[L - 1] = P[L, idx[L]].item()

    cutpoints = x[idx].cpu().float()
    cutpoint_values = y[idx].cpu().float()

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\n=== DP Search Complete ===")
        print(f"  Total time: {elapsed_total:.1f}s")
        print(f"  Optimal cost (global mean relative error): {optimal_cost:.6e}")
        print(f"  Cutpoints (x): {cutpoints.tolist()}")

    return cutpoints, cutpoint_values, optimal_cost


# ─────────────────────────────────────────────────────────────
#  Predefined cutpoints from Paper Table 9
# ─────────────────────────────────────────────────────────────

PAPER_CUTPOINTS = {
    'silu': [-20.359375, -17.109375, -8.3671875, -1.9755859375, -0.255615234375,-0.007244110107421875, 0.0072174072265625, 0.228515625, 1.58203125,10.46875, 65504.0],
    'exp': [-17.34375, -15.171875, -8.890625, -5.2734375, -2.35546875,-0.3583984375, 0.91650390625, 3.451171875, 6.84765625,10.9453125, 11.0859375],


    'rsqrt': [5.9604645e-08, 7.7486038e-07, 1.1140108e-04, 1.8644333e-03,
              3.0029297e-02, 0.48193359375, 7.7734375, 129.75, 2406.0,
              47456.0, 65504.0],

    'gelu': [-5.5390625, -5.15625, -3.18359375, -0.98046875, -0.1229248046875,
             -0.00374603271484375, 0.0035247802734375, 0.11322021484375,
             0.78076171875, 4.10546875, 65504.0],
    'sigmoid': [-17.34375, -15.765625, -10.65625, -8.15625, -6.3046875,
                -4.421875, -2.6640625, -0.7998046875, 1.9462890625,
                6.90234375, 8.3203125],
    'tanh': [-4.5078125, -3.79296875, -1.55078125, -0.5302734375,
             -0.028564453125, 0.0364990234375, 0.423828125, 1.076171875,
             2.0390625, 4.0625, 4.5078125],
    'hardswish': [-3.0, -2.984375, -1.87890625, -0.5390625, -0.059326171875,
                  -0.000743865966796875, 0.0034942626953125, 0.11968994140625,
                  0.78369140625, 3.001953125, 65504.0],
    'mish': [-20.34375, -19.90625, -10.921875, -6.2265625, -1.615234375,
             -0.237060546875, -0.00699615478515625, 0.01538848876953125,
             0.491455078125, 4.70703125, 65504.0],
    'reciprocal': [1.5318394e-05, 2.2590160e-05, 4.6992302e-04, 7.0533752e-03,
                   8.8378906e-02, 1.07421875, 15.546875, 244.5, 3694.0,
                   46560.0, 65504.0],
}


# ─────────────────────────────────────────────────────────────
#  Nonlinear function definitions
# ─────────────────────────────────────────────────────────────

def get_function(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get a nonlinear function by name."""
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
    if name not in funcs:
        raise ValueError(f"Unknown function: {name}. Available: {list(funcs.keys())}")
    return funcs[name]


def get_domain(name: str) -> Tuple[float, float]:
    """
    DP search domain per function — matches Table 9 of the NLI paper.

    The range is set to where f(x) produces non-trivial (non-saturated,
    non-overflow) FP16 outputs.  The DP constrains the last cutpoint to
    the last grid point (k*=argmin over last DP row), so the upper bound
    directly determines the largest possible cutpoint.
    """
    domains = {
        # Paper Table 9 Range column
        'silu':       (-20.359375,           65504.0),
        'gelu':       (-5.5390625,           65504.0),
        'exp':        (-17.34375,            11.0859375),
        'sigmoid':    (-17.34375,            8.3203125),
        'tanh':       (-4.5078125,           4.5078125),
        'hardswish':  (-3.0,                 65504.0),
        'mish':       (-20.34375,            65504.0),
        'rsqrt':      (5.9604644775390625e-08, 65504.0),
        'reciprocal': (1.5318393707275391e-05,  65504.0),
    }
    if name not in domains:
        raise ValueError(f"Unknown function: {name}. Available: {list(domains.keys())}")
    return domains[name]



# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NLI DP Cutpoint Search')
    parser.add_argument('--func', type=str, default='silu',
                        choices=list(PAPER_CUTPOINTS.keys()),
                        help='Target nonlinear function')
    parser.add_argument('--M', type=int, default=11,
                        help='Number of cutpoints')
    parser.add_argument('--max-points', type=int, default=None,
                        help='Max grid points (subsample for speed)')
                        #63,488
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU-accelerated version')
    args = parser.parse_args()
    
    func = get_function(args.func)
    domain = get_domain(args.func)
    
    print(f"=== NLI DP Search for {args.func} ===")
    print(f"Domain: {domain}, M={args.M}, max_points={args.max_points}")
    print()
    
    if args.gpu and torch.cuda.is_available():
        cutpoints, values, cost = dp_cutpoint_search_gpu(
            func, M=args.M, domain=domain,
            max_grid_points=args.max_points, verbose=True
        )
    else:
        cutpoints, values, cost = dp_cutpoint_search(
            func, M=args.M, domain=domain,
            max_grid_points=args.max_points, verbose=True
        )
    
    print(f"\n--- Paper cutpoints for comparison ---")
    print(f"  {PAPER_CUTPOINTS[args.func]}")

    # Save the results to a log file
    import datetime
    import os
    
    log_filename = "nli_dp_results.log"
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)
    
    with open(log_path, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Func: {args.func}, M: {args.M}, Max Points: {args.max_points}, GPU: {args.gpu}\n")
        f.write(f"  Optimal cost: {cost:.6e}\n")
        f.write(f"  Cutpoints (x): {cutpoints.tolist()}\n")
        f.write(f"  Cutpoints (f(x)): {values.tolist()}\n")
        f.write(f"  Paper cutpoints: {PAPER_CUTPOINTS[args.func]}\n")
        f.write("-" * 60 + "\n")
        
    print(f"[*] Results appended to {log_path}\n")
