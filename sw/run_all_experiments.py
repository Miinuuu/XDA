#!/usr/bin/env python3
"""
Re-run all paper experiments with updated domains.
Outputs JSON results for updating LaTeX tables.
"""
import torch
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nli_eda import (
    get_function, get_domain, optimize_eda, EDAConfig,
    EVAL_DOMAINS, _generate_fp16_grid, TAU, ALL_FUNCS,
    _invalidate_domain_cache,
)
from nli_eda_engine import _EDA_CACHE
from ablation_sweep import eda_forward_with_config, eval_on_fp16_grid
from nli_dp import PAPER_CUTPOINTS, generate_fp16_grid
from nli_engine import build_lut, nli_forward

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nli_results')

# Model configurations for Part B
MODEL_CONFIGS = {
    'Qwen2.5-0.5B': {
        'stats_file': 'Qwen_Qwen2.5-0.5B-Instruct_activation_stats.json',
        'hidden_dim': 896,
        'head_dim': 64,
        'n_heads': 14,
        'eps': 1e-6,
    },
    'Llama-3.2-1B': {
        'stats_file': 'meta-llama_Llama-3.2-1B-Instruct_activation_stats.json',
        'hidden_dim': 2048,
        'head_dim': 64,
        'n_heads': 32,
        'eps': 1e-5,
    },
}


def clear_caches():
    """Clear all EDA caches to force recomputation with new domains."""
    _invalidate_domain_cache()
    _EDA_CACHE.clear()
    print("Caches cleared.")


# ─── Part A: FP16 Full-Range Elementary Functions ─────────────

def run_part_a():
    """9 elementary functions on exhaustive FP16 grid."""
    print("\n" + "=" * 70)
    print("  PART A: FP16 Full-Range Elementary Functions")
    print("=" * 70)

    results = {}
    for fn in ALL_FUNCS:
        func = get_function(fn)
        domain = EVAL_DOMAINS[fn]
        grid = _generate_fp16_grid(domain, DEVICE)
        y_ref = func(grid.float())
        valid = torch.isfinite(y_ref)
        grid_v, y_ref_v = grid[valid], y_ref[valid]

        # EDA-NLI
        cfg = optimize_eda(fn, verbose=False, device=DEVICE)
        y_eda = eda_forward_with_config(grid_v, cfg, t_bits='adaptive')
        eda_rel = torch.abs(y_eda.float() - y_ref_v) / torch.clamp(torch.abs(y_ref_v), min=TAU)

        # NLI
        cuts = torch.tensor(PAPER_CUTPOINTS[fn], dtype=torch.float32)
        p, m, l = build_lut(func, cuts, 32)
        p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
        y_nli = nli_forward(grid_v, p, m, l, 32, fp16_hw=True)
        nli_rel = torch.abs(y_nli.float() - y_ref_v) / torch.clamp(torch.abs(y_ref_v), min=TAU)

        results[fn] = {
            'eda_mean': eda_rel.mean().item() * 1e4,
            'eda_max': eda_rel.max().item() * 1e4,
            'nli_mean': nli_rel.mean().item() * 1e4,
            'nli_max': nli_rel.max().item() * 1e4,
            'n_points': len(grid_v),
        }
        print(f"  {fn:12s}: EDA mean={results[fn]['eda_mean']:.2f} max={results[fn]['eda_max']:.1f} | "
              f"NLI mean={results[fn]['nli_mean']:.2f} max={results[fn]['nli_max']:.1f}")

    return results


# ─── Part B: Profiled Activation Distributions ────────────────

def _sample_from_hist(hist_bins, hist_counts, n_samples):
    """Sample from a histogram distribution."""
    bins = np.array(hist_bins)
    counts = np.array(hist_counts, dtype=float)
    probs = counts / counts.sum()
    bin_indices = np.random.choice(len(probs), size=n_samples, p=probs)
    lo = bins[bin_indices]
    hi = bins[bin_indices + 1]
    return torch.tensor(lo + (hi - lo) * np.random.rand(n_samples), dtype=torch.float32)


def _load_activation_stats(stats_file):
    """Load profiled activation stats."""
    path = os.path.join(RESULTS_DIR, stats_file)
    with open(path) as f:
        return json.load(f)


def _aggregate_histograms(stats, func_name):
    """Aggregate histograms across all layers for a given function.

    Each layer has its own bin edges (from np.histogram auto-binning),
    so we rebuild a common set of bin edges spanning the global range
    and re-bin every layer's histogram onto them via linear interpolation
    of the per-layer CDFs.
    """
    entries = [e for e in stats if e['function_name'] == func_name]
    if not entries:
        raise ValueError(f"No entries for {func_name}")

    # Determine global range across all layers
    global_lo = min(e['hist_bins'][0] for e in entries)
    global_hi = max(e['hist_bins'][-1] for e in entries)
    n_bins = len(entries[0]['hist_counts'])  # 200
    common_edges = np.linspace(global_lo, global_hi, n_bins + 1)

    # Re-bin each layer onto common edges via CDF interpolation
    all_counts = np.zeros(n_bins)
    for e in entries:
        edges = np.array(e['hist_bins'])
        counts = np.array(e['hist_counts'], dtype=float)
        # Build CDF at original bin edges
        cdf = np.zeros(len(edges))
        cdf[1:] = np.cumsum(counts)
        # Interpolate CDF at common edges
        common_cdf = np.interp(common_edges, edges, cdf)
        layer_counts = np.diff(common_cdf)
        layer_counts = np.maximum(layer_counts, 0.0)
        all_counts += layer_counts

    return common_edges.tolist(), all_counts.tolist()


def run_part_b_silu(stats, n_samples=500000, n_trials=50):
    """SiLU: profiled silu_input distribution."""
    print("\n  Part B: SiLU (profiled distribution)")
    bins, counts = _aggregate_histograms(stats, 'silu_input')

    eda_means, eda_maxs = [], []
    nli_means, nli_maxs = [], []

    func = get_function('silu')
    cfg = optimize_eda('silu', verbose=False, device=DEVICE)
    cuts = torch.tensor(PAPER_CUTPOINTS['silu'], dtype=torch.float32)
    p, m, l = build_lut(func, cuts, 32)
    p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)

    for trial in range(n_trials):
        x = _sample_from_hist(bins, counts, n_samples).to(DEVICE)
        y_ref = func(x.float())
        valid = torch.isfinite(y_ref) & (y_ref.abs() > 0)
        x_v, y_ref_v = x[valid], y_ref[valid]

        y_eda = eda_forward_with_config(x_v, cfg, t_bits='adaptive')
        y_nli = nli_forward(x_v, p, m, l, 32, fp16_hw=True)

        denom = torch.clamp(torch.abs(y_ref_v), min=TAU)
        eda_rel = torch.abs(y_eda.float() - y_ref_v) / denom
        nli_rel = torch.abs(y_nli.float() - y_ref_v) / denom

        eda_means.append(eda_rel.mean().item())
        eda_maxs.append(eda_rel.max().item())
        nli_means.append(nli_rel.mean().item())
        nli_maxs.append(nli_rel.max().item())

    result = {
        'eda_mean': np.mean(eda_means) * 1e4,
        'eda_max': np.mean(eda_maxs) * 1e4,
        'nli_mean': np.mean(nli_means) * 1e4,
        'nli_max': np.mean(nli_maxs) * 1e4,
    }
    print(f"    EDA: mean={result['eda_mean']:.2f} max={result['eda_max']:.1f}")
    print(f"    NLI: mean={result['nli_mean']:.2f} max={result['nli_max']:.1f}")
    return result


def run_part_b_softmax(stats, head_dim=64, n_heads=14, n_trials=50):
    """Softmax: construct attention logits from q_proj/k_proj profiles."""
    print("\n  Part B: Softmax (profiled q/k distributions)")
    q_bins, q_counts = _aggregate_histograms(stats, 'q_proj')
    k_bins, k_counts = _aggregate_histograms(stats, 'k_proj')

    func_exp = get_function('exp')
    func_recip = get_function('reciprocal')
    cfg_exp = optimize_eda('exp', verbose=False, device=DEVICE)
    cfg_recip = optimize_eda('reciprocal', verbose=False, device=DEVICE)

    cuts_exp = torch.tensor(PAPER_CUTPOINTS['exp'], dtype=torch.float32)
    p_e, m_e, l_e = build_lut(func_exp, cuts_exp, 32)
    p_e, m_e, l_e = p_e.to(DEVICE), m_e.to(DEVICE), l_e.to(DEVICE)
    cuts_recip = torch.tensor(PAPER_CUTPOINTS['reciprocal'], dtype=torch.float32)
    p_r, m_r, l_r = build_lut(func_recip, cuts_recip, 32)
    p_r, m_r, l_r = p_r.to(DEVICE), m_r.to(DEVICE), l_r.to(DEVICE)

    eda_means, eda_maxs = [], []
    nli_means, nli_maxs = [], []

    seq_len = 128
    batch_heads = 2 * n_heads

    for trial in range(n_trials):
        # Sample q, k from profiled distributions
        q = _sample_from_hist(q_bins, q_counts, batch_heads * seq_len * head_dim).to(DEVICE)
        k = _sample_from_hist(k_bins, k_counts, batch_heads * seq_len * head_dim).to(DEVICE)
        q = q.reshape(batch_heads, seq_len, head_dim)
        k = k.reshape(batch_heads, seq_len, head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Reference softmax
        y_ref = torch.softmax(logits.float(), dim=-1)

        # EDA softmax
        x_shifted = logits.float() - logits.float().max(dim=-1, keepdim=True).values
        exp_eda = eda_forward_with_config(x_shifted, cfg_exp, t_bits='adaptive')
        exp_sum = exp_eda.sum(dim=-1, keepdim=True)
        recip_eda = eda_forward_with_config(exp_sum, cfg_recip, t_bits='adaptive')
        y_eda = exp_eda * recip_eda

        # NLI softmax
        exp_nli = nli_forward(x_shifted, p_e, m_e, l_e, 32, fp16_hw=True)
        exp_sum_nli = exp_nli.sum(dim=-1, keepdim=True)
        recip_nli = nli_forward(exp_sum_nli, p_r, m_r, l_r, 32, fp16_hw=True)
        y_nli = exp_nli * recip_nli

        denom = torch.clamp(torch.abs(y_ref), min=TAU)
        eda_rel = torch.abs(y_eda.float() - y_ref) / denom
        nli_rel = torch.abs(y_nli.float() - y_ref) / denom

        eda_means.append(eda_rel.mean().item())
        eda_maxs.append(eda_rel.max().item())
        nli_means.append(nli_rel.mean().item())
        nli_maxs.append(nli_rel.max().item())

    result = {
        'eda_mean': np.mean(eda_means) * 1e4,
        'eda_max': np.mean(eda_maxs) * 1e4,
        'nli_mean': np.mean(nli_means) * 1e4,
        'nli_max': np.mean(nli_maxs) * 1e4,
    }
    print(f"    EDA: mean={result['eda_mean']:.2f} max={result['eda_max']:.1f}")
    print(f"    NLI: mean={result['nli_mean']:.2f} max={result['nli_max']:.1f}")
    return result


def run_part_b_rmsnorm(stats, hidden_dim=896, eps=1e-6, n_trials=50):
    """RMSNorm: profiled rmsnorm_input distribution."""
    print("\n  Part B: RMSNorm (profiled distribution)")
    bins, counts = _aggregate_histograms(stats, 'rmsnorm_input')

    func_rsqrt = get_function('rsqrt')
    cfg_rsqrt = optimize_eda('rsqrt', verbose=False, device=DEVICE)
    cuts = torch.tensor(PAPER_CUTPOINTS['rsqrt'], dtype=torch.float32)
    p, m, l = build_lut(func_rsqrt, cuts, 32)
    p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)

    eda_means, eda_maxs = [], []
    nli_means, nli_maxs = [], []

    for trial in range(n_trials):
        # Sample hidden states from profiled distribution
        n_tokens = 512
        x = _sample_from_hist(bins, counts, 4 * n_tokens * hidden_dim).to(DEVICE)
        x = x.reshape(4, n_tokens, hidden_dim)
        weight = torch.ones(hidden_dim, device=DEVICE)

        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True) + eps

        # Reference
        y_ref = x_f32 * torch.rsqrt(variance) * weight

        # EDA
        inv_rms_eda = eda_forward_with_config(variance, cfg_rsqrt, t_bits='adaptive')
        y_eda = (x_f32 * inv_rms_eda * weight).half().float()

        # NLI
        inv_rms_nli = nli_forward(variance, p, m, l, 32, fp16_hw=True)
        y_nli = (x_f32 * inv_rms_nli * weight).half().float()

        denom = torch.clamp(torch.abs(y_ref), min=TAU)
        eda_rel = torch.abs(y_eda - y_ref) / denom
        nli_rel = torch.abs(y_nli - y_ref) / denom

        eda_means.append(eda_rel.mean().item())
        eda_maxs.append(eda_rel.max().item())
        nli_means.append(nli_rel.mean().item())
        nli_maxs.append(nli_rel.max().item())

    result = {
        'eda_mean': np.mean(eda_means) * 1e4,
        'eda_max': np.mean(eda_maxs) * 1e4,
        'nli_mean': np.mean(nli_means) * 1e4,
        'nli_max': np.mean(nli_maxs) * 1e4,
    }
    print(f"    EDA: mean={result['eda_mean']:.2f} max={result['eda_max']:.1f}")
    print(f"    NLI: mean={result['nli_mean']:.2f} max={result['nli_max']:.1f}")
    return result


def run_part_b():
    """Part B: Profiled activation distributions for all models."""
    results = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        print("\n" + "=" * 70)
        print(f"  PART B: Profiled Activation Distributions ({model_name})")
        print("=" * 70)

        stats = _load_activation_stats(cfg['stats_file'])
        silu = run_part_b_silu(stats)
        softmax = run_part_b_softmax(stats,
                                     head_dim=cfg['head_dim'],
                                     n_heads=cfg['n_heads'])
        rmsnorm = run_part_b_rmsnorm(stats,
                                     hidden_dim=cfg['hidden_dim'],
                                     eps=cfg['eps'])
        results[model_name] = {'silu': silu, 'softmax': softmax, 'rmsnorm': rmsnorm}

    return results


# ─── Error Distribution Table ────────────────────────────────

def run_errdist():
    """Error distribution: % of FP16 points above thresholds."""
    print("\n" + "=" * 70)
    print("  ERROR DISTRIBUTION TABLE")
    print("=" * 70)

    thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    results = {}

    for fn in ALL_FUNCS:
        func = get_function(fn)
        domain = EVAL_DOMAINS[fn]
        grid = _generate_fp16_grid(domain, DEVICE)
        y_ref = func(grid.float())
        valid = torch.isfinite(y_ref)
        grid_v, y_ref_v = grid[valid], y_ref[valid]
        denom = torch.clamp(torch.abs(y_ref_v), min=TAU)

        # EDA
        cfg = optimize_eda(fn, verbose=False, device=DEVICE)
        y_eda = eda_forward_with_config(grid_v, cfg, t_bits='adaptive')
        eda_rel = torch.abs(y_eda.float() - y_ref_v) / denom

        # NLI
        cuts = torch.tensor(PAPER_CUTPOINTS[fn], dtype=torch.float32)
        p, m, l = build_lut(func, cuts, 32)
        p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
        y_nli = nli_forward(grid_v, p, m, l, 32, fp16_hw=True)
        nli_rel = torch.abs(y_nli.float() - y_ref_v) / denom

        eda_pcts = [(eda_rel > t).float().mean().item() * 100 for t in thresholds]
        nli_pcts = [(nli_rel > t).float().mean().item() * 100 for t in thresholds]

        results[fn] = {'eda': eda_pcts, 'nli': nli_pcts}
        print(f"  {fn:12s}: EDA {[f'{p:.1f}' for p in eda_pcts]} | NLI {[f'{p:.1f}' for p in nli_pcts]}")

    return results


# ─── Ablation Budget ─────────────────────────────────────────

def run_ablation_budget():
    """LUT budget sweep: W ∈ {62, 126, 254, 510}."""
    print("\n" + "=" * 70)
    print("  ABLATION: LUT Budget Scaling")
    print("=" * 70)

    from nli_eda import optimize_eda_with_allocation

    budgets = [62, 126, 254, 510]
    nli_Dn = {62: 8, 126: 16, 254: 32, 510: 64}
    ablation_funcs = ALL_FUNCS
    results = {}

    for fname in ablation_funcs:
        results[fname] = {}
        for W in budgets:
            print(f"  {fname} W={W}...", end='', flush=True)

            # EDA-Knapsack
            cfg_ks = optimize_eda_with_allocation(fname, 'knapsack', max_lut=W, device=DEVICE)
            res_ks = eval_on_fp16_grid(
                fname, lambda x, c=cfg_ks: eda_forward_with_config(x, c, t_bits='adaptive'), DEVICE)

            # NLI
            D_n = nli_Dn[W]
            func = get_function(fname)
            cuts = torch.tensor(PAPER_CUTPOINTS[fname], dtype=torch.float32)
            p, m, l = build_lut(func, cuts, D_n)
            p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
            res_nli = eval_on_fp16_grid(
                fname,
                lambda x, pp=p, mm=m, ll=l, dn=D_n: nli_forward(x, pp, mm, ll, dn, fp16_hw=True),
                DEVICE)

            results[fname][W] = {
                'knapsack_mean': res_ks['mean_rel'] * 1e4,
                'nli_mean': res_nli['mean_rel'] * 1e4,
            }
            print(f" KS={results[fname][W]['knapsack_mean']:.2f} NLI={results[fname][W]['nli_mean']:.2f}")

    return results


# ─── Tab:Functions ────────────────────────────────────────────

def run_tab_functions():
    """Collect config data for tab:functions."""
    print("\n" + "=" * 70)
    print("  TAB:FUNCTIONS Configuration Data")
    print("=" * 70)

    results = {}
    for fn in ALL_FUNCS:
        cfg = optimize_eda(fn, verbose=False, device=DEVICE)
        domain = get_domain(fn)
        results[fn] = {
            'bins': len(cfg.bins),
            'total_lut': len(cfg.lut_values),
            'max_k': max(cfg.k_alloc),
            'domain': list(domain),
        }
        print(f"  {fn:12s}: bins={results[fn]['bins']} lut={results[fn]['total_lut']} "
              f"max_k={results[fn]['max_k']} domain={domain}")

    return results


# ─── Main ─────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Device: {DEVICE}")
    clear_caches()

    part_a = run_part_a()
    part_b = run_part_b()
    errdist = run_errdist()
    ablation = run_ablation_budget()
    tab_funcs = run_tab_functions()

    all_results = {
        'part_a': part_a,
        'part_b': part_b,
        'errdist': errdist,
        'ablation_budget': ablation,
        'tab_functions': tab_funcs,
    }

    out_path = os.path.join(RESULTS_DIR, 'all_experiments.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    # Print summary for paper update
    print("\n" + "=" * 70)
    print("  SUMMARY FOR PAPER UPDATE")
    print("=" * 70)

    print("\n--- tab:accuracy Part A ---")
    for fn in ALL_FUNCS:
        r = part_a[fn]
        print(f"  {fn:12s} & {r['eda_mean']:.2f} & {r['eda_max']:.1f} & {r['nli_mean']:.2f} & {r['nli_max']:.1f}")

    print("\n--- tab:accuracy Part B ---")
    for model_name, model_results in part_b.items():
        print(f"  [{model_name}]")
        for name in ['silu', 'softmax', 'rmsnorm']:
            r = model_results[name]
            label = name.capitalize() if name != 'rmsnorm' else 'RMSNorm'
            print(f"    {label:12s} & {r['eda_mean']:.2f} & {r['eda_max']:.1f} & {r['nli_mean']:.2f} & {r['nli_max']:.1f}")

    print("\n--- tab:errdist-all ---")
    for fn in ALL_FUNCS:
        r = errdist[fn]
        eda_s = " & ".join(f"{v:.1f}" for v in r['eda'])
        nli_s = " & ".join(f"{v:.1f}" for v in r['nli'])
        print(f"  {fn:12s} EDA: {eda_s}")
        print(f"  {fn:12s} NLI: {nli_s}")

    print("\n--- tab:abl-budget ---")
    for fname in ALL_FUNCS:
        ks_vals = " & ".join(f"{ablation[fname][W]['knapsack_mean']:.2f}" for W in [62,126,254,510])
        nli_vals = " & ".join(f"{ablation[fname][W]['nli_mean']:.2f}" for W in [62,126,254,510])
        print(f"  {fname:12s} KS:  {ks_vals}")
        print(f"  {fname:12s} NLI: {nli_vals}")

    print("\n--- tab:functions ---")
    for fn in ALL_FUNCS:
        r = tab_funcs[fn]
        lo, hi = r['domain']
        if lo > 0:
            import math
            exp = int(math.log2(lo))
            lo_s = f"$2^{{{exp}}}$" if 2**exp == lo else f"{lo:g}"
        else:
            lo_s = f"{lo:g}"
        print(f"  {fn:12s} & {r['bins']} & {r['total_lut']} & {r['max_k']} & [{lo_s}, {hi:g}]")


if __name__ == '__main__':
    main()
