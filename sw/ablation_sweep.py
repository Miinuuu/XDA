"""
Ablation Sweep for EDA-NLI Paper Section 5.5
=============================================
Three ablation studies:
  1. LUT budget scaling (W ∈ {64, 128, 256, 512})
  2. Interpolation bit-width (T ∈ {3..8})
  3. Allocation strategy comparison (Uniform / Curvature / Knapsack / NLI)

Outputs:
  - nli_results/ablation_results.json
  - LaTeX tables to stdout
"""

import torch
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nli_dp import generate_fp16_grid, PAPER_CUTPOINTS
from nli_eda import (
    get_function, get_domain, optimize_eda_with_allocation, EDAConfig,
)
from nli_engine import build_lut, nli_forward

REL_CLAMP = 2 ** (-14)  # TAU: smallest positive normal in FP16, matching NLI paper

ABLATION_FUNCS = ['silu', 'exp', 'rsqrt', 'sigmoid']
ALL_FUNCS = ['silu', 'gelu', 'exp', 'sigmoid', 'tanh',
             'hardswish', 'mish', 'rsqrt', 'reciprocal']

# ─── Helpers ───────────────────────────────────────────────────


EVAL_DOMAINS = {
    # Full FP16 range for most functions; restricted only where function overflows
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


def eval_on_fp16_grid(func_name, forward_fn, device='cuda'):
    """Evaluate on exhaustive FP16 grid over full representable range."""
    domain = EVAL_DOMAINS.get(func_name, (-65504.0, 65504.0))
    grid = generate_fp16_grid(domain).to(device)
    func = get_function(func_name)
    y_ref = func(grid.float())
    y_approx = forward_fn(grid)

    # Filter out inf/nan in reference (safety)
    valid = torch.isfinite(y_ref)
    y_ref = y_ref[valid]
    y_approx = y_approx[valid]

    abs_err = torch.abs(y_approx.float() - y_ref)
    denom = torch.clamp(torch.abs(y_ref), min=REL_CLAMP)
    rel_err = abs_err / denom

    return {
        'max_rel': rel_err.max().item(),
        'mean_rel': rel_err.mean().item(),
        'max_abs': abs_err.max().item(),
        'mean_abs': abs_err.mean().item(),
        'n_points': len(grid),
    }


def eda_forward_with_config(x_input, config, t_bits=None):
    """EDA forward pass with explicit config (bypasses cache)."""
    device = x_input.device
    original_shape = x_input.shape
    x_dtype = x_input.dtype
    # EDA uses FP16 bit-field extraction — must quantize to fp16 first
    x_flat = x_input.reshape(-1).half().float()

    bin_starts = config.bin_starts.float().to(device)
    n_bins = len(bin_starts)

    bin_idx = torch.bucketize(x_flat, bin_starts, right=False) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    bin_ends = config.bin_ends.float().to(device)
    too_low = x_flat < bin_starts[0]
    too_high = x_flat > bin_ends[-1]

    b_start = bin_starts[bin_idx]
    b_end = bin_ends[bin_idx]
    base_offset = config.base_offsets.to(device)[bin_idx]
    k_bits = config.k_bits_tensor.to(device)[bin_idx]

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
            per_elem_t = 10 - k_bits
            scale = (1 << per_elem_t).float()
            t = torch.floor(t * scale) / scale
        else:
            t = torch.floor(t * (1 << t_bits)) / (1 << t_bits)

    lut_values = config.lut_values.to(device)
    lut_idx = base_offset + micro_idx
    lut_idx_next = lut_idx + 1
    max_lut_idx = len(lut_values) - 1
    lut_idx = lut_idx.clamp(0, max_lut_idx)
    lut_idx_next = lut_idx_next.clamp(0, max_lut_idx)

    # RTL: func_lut stores FP16; diff/product/sum all FP16
    lut_fp16 = lut_values.half()
    y0 = lut_fp16[lut_idx]
    y1 = lut_fp16[lut_idx_next]
    diff = (y1 - y0)
    product = (t.half() * diff).half()
    y = (y0 + product).half().float()

    y = torch.where(too_low, lut_fp16[0].float(), y)
    y = torch.where(too_high, lut_fp16[-1].float(), y)

    return y.reshape(original_shape).to(x_dtype)


# ─── Ablation 1: LUT Budget Sweep ─────────────────────────────


def run_ablation_budget(device='cuda', t_bits='adaptive'):
    """LUT budget scaling: W ∈ {62, 126, 254, 510}."""
    print("\n" + "=" * 70)
    print("  ABLATION 1: LUT Budget Scaling")
    print("=" * 70)

    budgets = [62, 126, 254, 510]
    nli_Dn = {62: 8, 126: 16, 254: 32, 510: 64}
    results = {}

    for fname in ABLATION_FUNCS:
        results[fname] = {}
        for W in budgets:
            print(f"  {fname} W={W}...", end='', flush=True)

            # EDA-Knapsack
            cfg_ks = optimize_eda_with_allocation(
                fname, 'knapsack', max_lut=W, device=device)
            res_ks = eval_on_fp16_grid(
                fname, lambda x, c=cfg_ks, tb=t_bits: eda_forward_with_config(x, c, t_bits=tb), device)

            # EDA-Uniform
            cfg_uni = optimize_eda_with_allocation(
                fname, 'uniform', max_lut=W, device=device)
            res_uni = eval_on_fp16_grid(
                fname, lambda x, c=cfg_uni, tb=t_bits: eda_forward_with_config(x, c, t_bits=tb), device)

            # NLI
            D_n = nli_Dn[W]
            func = get_function(fname)
            cuts = torch.tensor(PAPER_CUTPOINTS[fname], dtype=torch.float32)
            p, m, l = build_lut(func, cuts, D_n)
            p, m, l = p.to(device), m.to(device), l.to(device)
            res_nli = eval_on_fp16_grid(
                fname,
                lambda x, pp=p, mm=m, ll=l, dn=D_n: nli_forward(x, pp, mm, ll, dn, fp16_hw=True),
                device)
            res_nli['entries'] = len(l)

            results[fname][W] = {
                'knapsack': res_ks,
                'uniform': res_uni,
                'nli': res_nli,
            }
            print(" done")

    return results


# ─── Ablation 2: Interpolation Bit-Width ──────────────────────


def run_ablation_tbits(device='cuda'):
    """Interpolation bit-width T: 3..8 + continuous."""
    print("\n" + "=" * 70)
    print("  ABLATION 2: Interpolation Bit-Width")
    print("=" * 70)

    t_values = [3, 4, 5, 6, 7, 8]
    results = {}

    for fname in ABLATION_FUNCS:
        results[fname] = {}
        cfg = optimize_eda_with_allocation(
            fname, 'knapsack', max_lut=256, device=device)

        # Continuous baseline
        res_cont = eval_on_fp16_grid(
            fname, lambda x, c=cfg: eda_forward_with_config(x, c), device)
        results[fname]['continuous'] = res_cont

        for T in t_values:
            print(f"  {fname} T={T}...", end='', flush=True)
            res = eval_on_fp16_grid(
                fname,
                lambda x, c=cfg, tb=T: eda_forward_with_config(x, c, t_bits=tb),
                device)
            results[fname][T] = res
            print(" done")

    return results


# ─── Ablation 3: Allocation Strategy ──────────────────────────


def run_ablation_alloc(device='cuda', t_bits='adaptive'):
    """Allocation strategy comparison on all 9 functions."""
    print("\n" + "=" * 70)
    print("  ABLATION 3: Allocation Strategy Comparison")
    print("=" * 70)

    strategies = ['uniform', 'curvature', 'knapsack']
    results = {}

    for fname in ALL_FUNCS:
        results[fname] = {}

        for strat in strategies:
            print(f"  {fname} {strat}...", end='', flush=True)
            cfg = optimize_eda_with_allocation(
                fname, strat, max_lut=256, device=device)
            res = eval_on_fp16_grid(
                fname, lambda x, c=cfg, tb=t_bits: eda_forward_with_config(x, c, t_bits=tb), device)
            results[fname][strat] = res
            print(" done")

        # NLI baseline
        print(f"  {fname} nli...", end='', flush=True)
        func = get_function(fname)
        cuts = torch.tensor(PAPER_CUTPOINTS[fname], dtype=torch.float32)
        p, m, l = build_lut(func, cuts, 32)
        p, m, l = p.to(device), m.to(device), l.to(device)
        res_nli = eval_on_fp16_grid(
            fname,
            lambda x, pp=p, mm=m, ll=l: nli_forward(x, pp, mm, ll, 32, fp16_hw=True),
            device)
        results[fname]['nli'] = res_nli
        print(" done")

    return results


# ─── Ablation 4: max_k × t_bits Joint Sweep ─────────────────


def run_ablation_k_t(device='cuda'):
    """Joint max_k × t_bits sweep (W=256, Knapsack)."""
    print("\n" + "=" * 70)
    print("  ABLATION 4: max_k × t_bits Joint Sweep")
    print("=" * 70)

    k_values = [1, 2, 3, 4, 5]
    t_values = [3, 4, 5, 6, 7, 8]
    results = {}

    for fname in ABLATION_FUNCS:
        results[fname] = {}
        for mk in k_values:
            results[fname][mk] = {}
            for tb in t_values:
                print(f"  {fname} max_k={mk} t={tb}...", end='', flush=True)
                cfg = optimize_eda_with_allocation(
                    fname, 'knapsack', max_lut=256, max_k=mk, device=device)
                res = eval_on_fp16_grid(
                    fname,
                    lambda x, c=cfg, t=tb: eda_forward_with_config(x, c, t_bits=t),
                    device)
                results[fname][mk][tb] = res
                print(f" {res['mean_rel']*1e4:.2f}", flush=True)

    return results


def latex_table_k_t(results):
    """LaTeX for Ablation 4: max_k × t_bits joint sweep."""
    k_values = [1, 2, 3, 4, 5]
    t_values = [3, 4, 5, 6, 7, 8]

    for fname in results:
        print(f"\n% --- {fname} ---")
        # Find best cell
        best_val = float('inf')
        for mk in k_values:
            for tb in t_values:
                v = results[fname][mk][tb]['mean_rel']
                if v < best_val:
                    best_val = v

        header = " & ".join([f"$T{{=}}{t}$" for t in t_values])
        print(f"% $K_{{\\max}}$ & {header}")
        for mk in k_values:
            cells = []
            for tb in t_values:
                v = results[fname][mk][tb]['mean_rel'] * 1e4
                cells.append(f"{v:.2f}")
            row = " & ".join(cells)
            print(f"% {mk} & {row}")


# ─── LaTeX Table Generators ───────────────────────────────────


def latex_table_budget(results):
    """LaTeX for Ablation 1: LUT budget scaling."""
    print("\n% === Ablation 1: LUT Budget ===")
    budgets = [62, 126, 254, 510]
    methods = [('Knapsack', 'knapsack'), ('Uniform', 'uniform'), ('NLI', 'nli')]

    print(r"\begin{table}[t]")
    print(r"\caption{Mean relative error ($\times 10^{-4}$) vs.\ LUT micro-bin "
          r"budget $W$. Lower is better. \textbf{Bold}: best per $(f, W)$ cell. "
          r"NLI uses $D_n \in \{8, 16, 32, 64\}$ to match budget.}")
    print(r"\label{tab:abl-budget}")
    print(r"\centering\small")
    print(r"\begin{tabular}{ll" + "r" * len(budgets) + "}")
    print(r"\toprule")
    cols = " & ".join([f"$W{{=}}{w}$" for w in budgets])
    print(r"\textbf{Function} & \textbf{Method} & " + cols + r" \\")
    print(r"\midrule")

    for fi, fname in enumerate(ABLATION_FUNCS):
        if fi > 0:
            print(r"\midrule")
        for mi, (mlabel, mkey) in enumerate(methods):
            prefix = (f"\\multirow{{3}}{{*}}{{{fname.capitalize()}}}"
                      if mi == 0 else "")
            vals = []
            for W in budgets:
                v = results[fname][W][mkey]['mean_rel'] * 1e4
                all_v = [results[fname][W][mk]['mean_rel'] * 1e4
                         for _, mk in methods]
                s = f"{v:.2f}"
                if v <= min(all_v) + 1e-10:
                    s = f"\\textbf{{{s}}}"
                vals.append(s)
            print(f"{prefix} & {mlabel} & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def latex_table_tbits(results):
    """LaTeX for Ablation 2: interpolation bit-width."""
    print("\n% === Ablation 2: Interpolation Bit-Width ===")
    t_values = [3, 4, 5, 6, 7, 8]

    print(r"\begin{table}[t]")
    print(r"\caption{Mean relative error ($\times 10^{-4}$) vs.\ interpolation "
          r"bit-width $T$ ($W{=}256$, Knapsack). $T{=}\infty$: continuous "
          r"(no quantization).}")
    print(r"\label{tab:abl-tbits}")
    print(r"\centering\small")
    print(r"\begin{tabular}{l" + "r" * (len(t_values) + 1) + "}")
    print(r"\toprule")
    cols = " & ".join([f"$T{{=}}{t}$" for t in t_values])
    print(r"\textbf{Function} & " + cols + r" & $T{=}\infty$ \\")
    print(r"\midrule")

    for fname in ABLATION_FUNCS:
        vals = []
        for T in t_values:
            v = results[fname][T]['mean_rel'] * 1e4
            vals.append(f"{v:.2f}")
        v_cont = results[fname]['continuous']['mean_rel'] * 1e4
        vals.append(f"{v_cont:.2f}")
        print(f"{fname.capitalize()} & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def latex_table_alloc(results):
    """LaTeX for Ablation 3: allocation strategy."""
    print("\n% === Ablation 3: Allocation Strategy ===")
    strategies = [('Uniform', 'uniform'), ('Curvature', 'curvature'),
                  ('Knapsack', 'knapsack'), ('NLI', 'nli')]

    print(r"\begin{table}[t]")
    print(r"\caption{Mean relative error ($\times 10^{-4}$) by allocation "
          r"strategy ($W{=}256$). \textbf{Bold}: best per function.}")
    print(r"\label{tab:abl-alloc}")
    print(r"\centering\small")
    print(r"\begin{tabular}{l" + "r" * len(strategies) + "}")
    print(r"\toprule")
    cols = " & ".join([f"\\textbf{{{s[0]}}}" for s in strategies])
    print(r"\textbf{Function} & " + cols + r" \\")
    print(r"\midrule")

    for fname in ALL_FUNCS:
        all_v = [results[fname][sk]['mean_rel'] for _, sk in strategies]
        best = min(all_v)
        vals = []
        for _, skey in strategies:
            v = results[fname][skey]['mean_rel'] * 1e4
            s = f"{v:.2f}"
            if results[fname][skey]['mean_rel'] <= best + 1e-12:
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        print(f"{fname.capitalize()} & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─── Main ─────────────────────────────────────────────────────


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    t0 = time.time()

    r1 = run_ablation_budget(device)
    r2 = run_ablation_tbits(device)
    r3 = run_ablation_alloc(device)
    r4 = run_ablation_k_t(device)

    # Sanity check: W=254 Knapsack should match across ablation 1 and 3
    print("\n--- Sanity check: W=254 Knapsack consistency ---")
    for fname in ABLATION_FUNCS:
        v1 = r1[fname][254]['knapsack']['mean_rel']
        v3 = r3[fname]['knapsack']['mean_rel']
        diff = abs(v1 - v3)
        status = "PASS" if diff < 1e-10 else f"FAIL (diff={diff:.2e})"
        print(f"  {fname}: abl1={v1:.6e}  abl3={v3:.6e}  [{status}]")

    # Save JSON
    all_results = {
        'ablation_budget': r1,
        'ablation_tbits': r2,
        'ablation_alloc': r3,
        'ablation_k_t': r4,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nli_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ablation_results.json')

    def convert(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    # Print LaTeX tables
    latex_table_budget(r1)
    latex_table_tbits(r2)
    latex_table_alloc(r3)
    latex_table_k_t(r4)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
