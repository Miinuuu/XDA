"""
BF16 preliminary experiment.
BF16: 1 sign + 8 exponent + 7 mantissa → K+T=7, up to 511 coarse bins.

We evaluate XDA's Knapsack allocation under BF16 bin structure on the
same 9 functions, comparing against the FP16 XDA baseline.
Since no BF16 hardware baseline exists, we report absolute error levels
to show that the mechanism is directly applicable.
"""
import torch
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from XDA.eda.nli_eda import get_function, get_domain, EDAConfig, solve_knapsack_dp

TAU = 2.0 ** (-14)


def get_bf16_exponent_bins(domain):
    """
    Generate BF16 exponent-aligned bins.
    BF16: 8-bit exponent (bias=127), 7-bit mantissa.
    Normal: e=1..254 → [2^(e-127), 2^(e-126))
    Subnormal: e=0 → [2^-133, 2^-126)  (smallest bf16 subnormal to largest)
    """
    bins = []

    # Positive normals: e=1..254
    for e in range(1, 255):
        lo = 2.0 ** (e - 127)
        hi = 2.0 ** (e - 126)
        bins.append((lo, hi, 0, e))

    # Positive subnormals: e=0
    # Smallest BF16 subnormal: 2^-133 (mantissa=1, no implicit 1)
    # Largest BF16 subnormal: (127/128) * 2^-126 ≈ 2^-126
    bins.append((2.0**(-133), 2.0**(-126), 0, 0))

    # Negative normals
    for e in range(1, 255):
        lo = -(2.0 ** (e - 126))
        hi = -(2.0 ** (e - 127))
        bins.append((lo, hi, 1, e))

    # Negative subnormals
    bins.append((-(2.0**(-126)), -(2.0**(-133)), 1, 0))

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


def compute_bf16_bin_error(func, b_start, b_end, k_bits, mantissa_total=7, device='cuda'):
    """
    Compute interpolation error for a BF16 bin.
    BF16 mantissa = 7 bits, so T = 7 - K (not 10 - K).

    Uses FP16 grid points within the bin range as test vectors
    (BF16 representable values are a subset of FP32).
    """
    num_microbins = 2 ** k_bits

    # Generate test points: use linspace (BF16 has fewer representable values)
    n_test = min(max(num_microbins * 32, 256), 4096)
    x = torch.linspace(b_start, b_end, n_test, device=device, dtype=torch.float32)
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
    T_bits = mantissa_total - k_bits  # 7 - K for BF16
    T_bits = max(T_bits, 0)
    scale = float(1 << T_bits)
    t_q = torch.floor(t * scale) / scale
    y_pred = y0 + t_q * (y1 - y0)

    abs_err = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=TAU)
    rel_err = abs_err / denom

    return rel_err.mean().item(), num_microbins


def precompute_bf16_costs(func, bins, max_k=5, device='cuda'):
    """Precompute error matrix for BF16 bins."""
    N = len(bins)
    K = max_k + 1
    err_matrix = np.full((N, K), np.inf, dtype=np.float64)
    weight_matrix = np.zeros((N, K), dtype=np.int32)

    for i, (b_start, b_end, sign, exp_val) in enumerate(bins):
        for k in range(K):
            if k > 7:  # BF16 only has 7 mantissa bits
                continue
            err, n_entries = compute_bf16_bin_error(
                func, b_start, b_end, k, mantissa_total=7, device=device)
            err_matrix[i, k] = err
            weight_matrix[i, k] = n_entries

    return err_matrix, weight_matrix


def optimize_bf16(func_name, max_lut=254, max_k=5, device='cuda'):
    """Run Knapsack for BF16 bins."""
    func = get_function(func_name)
    domain = get_domain(func_name)
    bins = get_bf16_exponent_bins(domain)

    max_k = min(max_k, 7)  # BF16: K+T=7, so K<=7

    err_matrix, weight_matrix = precompute_bf16_costs(func, bins, max_k, device)
    total_error, k_alloc = solve_knapsack_dp(err_matrix, weight_matrix, max_lut)

    return bins, k_alloc, total_error


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    FUNCS = ['silu', 'gelu', 'exp', 'sigmoid', 'tanh',
             'hardswish', 'mish', 'rsqrt', 'reciprocal']
    W_VALUES = [512, 1024]

    # Summary: FP16 vs BF16 structure
    print("=" * 70)
    print("BF16 Preliminary: Knapsack Allocation under BF16 Bin Structure")
    print("mantissa=7 bits, K_max=5 (T_min=2)")
    print("=" * 70)

    from XDA.eda.nli_eda import get_fp16_exponent_bins

    print(f"\n{'Function':>12s} | {'FP16 bins':>9s} | {'BF16 bins':>9s} | {'min W':>5s}")
    print('-' * 45)
    for fname in FUNCS:
        d = get_domain(fname)
        fp16_bins = get_fp16_exponent_bins(d)
        bf16_bins = get_bf16_exponent_bins(d)
        print(f"{fname:>12s} | {len(fp16_bins):>9d} | {len(bf16_bins):>9d} | {len(bf16_bins):>5d}")

    print(f"\n{'Function':>12s} | {'W':>5s} | {'#Bins':>5s} | {'K range':>8s} | "
          f"{'K>0':>4s} | {'Used':>5s} | {'DP Err':>10s} | {'Time':>5s}")
    print('-' * 70)

    for fname in FUNCS:
        for W in W_VALUES:
            d = get_domain(fname)
            bf16_bins = get_bf16_exponent_bins(d)
            if len(bf16_bins) > W:
                print(f"{fname:>12s} | {W:>5d} | {len(bf16_bins):>5d} | "
                      f"{'--':>8s} | {'--':>4s} | {'--':>5s} | {'INFEASIBLE':>10s} | {'--':>5s}")
                continue
            t0 = time.time()
            bins, k_alloc, total_err = optimize_bf16(fname, max_lut=W, device=device)
            elapsed = time.time() - t0

            k_arr = np.array(k_alloc)
            k_nonzero = int(np.sum(k_arr > 0))
            k_min, k_max = int(k_arr.min()), int(k_arr.max())
            budget_used = sum(2**k for k in k_alloc)

            print(f"{fname:>12s} | {W:>5d} | {len(bins):>5d} | {k_min:>2d}--{k_max:<2d}    | "
                  f"{k_nonzero:>4d} | {budget_used:>5d} | {total_err:>10.4f} | {elapsed:>4.1f}s")
        print()
