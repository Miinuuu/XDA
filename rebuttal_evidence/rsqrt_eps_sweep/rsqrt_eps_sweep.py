"""
Exhaustive, distribution-free rsqrt epsilon-sensitivity bound (ICCAD'26 rebuttal, B-Q4).

RMSNorm computes rsqrt(variance + eps), so the hardware input is lower-bounded
by eps regardless of model family, dataset, or activation distribution.
We enumerate EVERY FP16 value x >= eps on the SAME exhaustive grid as paper
Table Part A (EVAL_DOMAINS) and report the worst-case relative error of both
XDA and the NLI baseline — a deterministic bound no input distribution can exceed.
"""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))

from XDA.eda.nli_eda import (optimize_eda, _generate_fp16_grid, get_function,
                             EVAL_DOMAINS, TAU)
from XDA.eda.ablation_sweep import eda_forward_with_config
from XDA.eda.nli_dp import PAPER_CUTPOINTS
from XDA.eda.nli_engine import build_lut, nli_forward

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
fn = 'rsqrt'
func = get_function(fn)
domain = EVAL_DOMAINS[fn]
grid = _generate_fp16_grid(domain, DEVICE)
y_ref = func(grid.float())
valid = torch.isfinite(y_ref)
grid_v, y_ref_v = grid[valid], y_ref[valid]
print(f"domain={domain}, n_points={len(grid_v)}  (paper Part A: 31,743)")

# XDA — identical pipeline to run_all_experiments.run_part_a
cfg = optimize_eda(fn, verbose=False, device=DEVICE)
y_eda = eda_forward_with_config(grid_v, cfg, t_bits='adaptive')
eda_rel = torch.abs(y_eda.float() - y_ref_v) / torch.clamp(torch.abs(y_ref_v), min=TAU)

# NLI (FP16 hw path) — identical pipeline
cuts = torch.tensor(PAPER_CUTPOINTS[fn], dtype=torch.float32)
p, m, l = build_lut(func, cuts, 32)
p, m, l = p.to(DEVICE), m.to(DEVICE), l.to(DEVICE)
y_nli = nli_forward(grid_v, p, m, l, 32, variant='fp16_hw')
nli_rel = torch.abs(y_nli.float() - y_ref_v) / torch.clamp(torch.abs(y_ref_v), min=TAU)

print(f"sanity full-range max (x1e-4): XDA={eda_rel.max().item()*1e4:.1f} "
      f"(paper 7,500), NLI={nli_rel.max().item()*1e4:.1f} (paper 9,134.5)")

print(f"\n{'eps':>8s} | {'#pts>=eps':>9s} | {'XDA max':>9s} | {'NLI max':>9s} | {'XDA mean':>9s} | {'NLI mean':>9s}   (x1e-4)")
print('-' * 75)
rows = []
for eps in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    msk = grid_v >= eps
    xm = eda_rel[msk].max().item() * 1e4
    nm = nli_rel[msk].max().item() * 1e4
    xa = eda_rel[msk].mean().item() * 1e4
    na = nli_rel[msk].mean().item() * 1e4
    rows.append((eps, int(msk.sum()), xm, nm, xa, na))
    print(f"{eps:>8.0e} | {int(msk.sum()):>9d} | {xm:>9.1f} | {nm:>9.1f} | {xa:>9.2f} | {na:>9.2f}")

import json
# The canonical (committed) results JSON is GPU-derived; near-tied Knapsack
# allocations can differ slightly across devices, so a non-CUDA run writes a
# device-suffixed file instead of overwriting the canonical artifact.
suffix = '' if DEVICE == 'cuda' else f'_{DEVICE}'
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f'rsqrt_eps_sweep_results{suffix}.json')
json.dump([{'eps': e, 'n': n, 'xda_max': xm, 'nli_max': nm,
            'xda_mean': xa, 'nli_mean': na} for e, n, xm, nm, xa, na in rows],
          open(out, 'w'), indent=2)
print(f"\nsaved -> {out}")
if suffix:
    print("note: canonical rsqrt_eps_sweep_results.json is GPU-derived and was "
          "left untouched; this run wrote a device-suffixed variant.")
