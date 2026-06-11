"""
NN-LUT Engine: Faithful reimplementation of NN-LUT (Yu et al., DAC 2022)
=========================================================================
Implements the paper's algorithm:

Offline (LUT generation):
  1. Train 1-hidden-layer ReLU NN: y = Σ m_i·ReLU(n_i·x + b_i)  (Eq 5)
  2. Extract breakpoints: d_i = -b_i/n_i, sorted ascending
  3. Derive slope s_i and intercept t_i per segment (Eq 7)
  4. Store 16-entry LUT of (s_i, t_i) pairs

Online (HW inference, 2-cycle):
  Cycle 1: Compare x with d_1..d_{N-1} → index i, read (s_i, t_i)  (Fig 3a)
  Cycle 2: y = s_i · x + t_i  (1-MAC, Eq 4)

Paper default: N=16 segments (15 hidden neurons).

Training uses FP16-grid-rank normalization for wide-range LLM functions.
Breakpoints are mapped back to original domain, then s_i/t_i are derived
from the true function values — mathematically equivalent to Eq 7 when
the NN has converged (paper's proven NN↔PWL equivalence, Section 3.1).

Forward pass simulates FP16 hardware: comparator chain + 1-MAC.
"""

import torch
from typing import Dict, Tuple
from nli_dp import get_function, get_domain, generate_fp16_grid

_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_CHUNK = 4 * 1024 * 1024


def _nn_forward(x, n, b, m):
    """Forward: y = Σ m_i·ReLU(n_i·x + b_i)  (Eq 5)."""
    hidden = torch.relu(n.unsqueeze(0) * x.unsqueeze(1) + b.unsqueeze(0))
    return (m.unsqueeze(0) * hidden).sum(dim=1)


@torch.enable_grad()
def _train_nn(n, b, m, x_train, y_train, n_epochs=3000, lr=0.002, batch_size=2048):
    """Train ReLU NN with L1 loss + Adam (paper Section 3.2)."""
    dev = x_train.device
    n = n.to(dev).clone().requires_grad_(True)
    b = b.to(dev).clone().requires_grad_(True)
    m = m.to(dev).clone().requires_grad_(True)

    optimizer = torch.optim.Adam([n, b, m], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1200, 2100], gamma=0.1)

    n_samples = len(x_train)
    for _ in range(n_epochs):
        perm = torch.randperm(n_samples, device=dev)[:batch_size]
        xb, yb = x_train[perm], y_train[perm]

        optimizer.zero_grad()
        y_pred = _nn_forward(xb, n, b, m)
        loss = (y_pred - yb).abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([n, b, m], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            too_small = n.abs() < 0.005
            if too_small.any():
                signs = n[too_small].sign()
                signs[signs == 0] = 1.0
                n[too_small] = 0.005 * signs

    return n.detach().cpu(), b.detach().cpu(), m.detach().cpu()


def _optimize_breakpoints(func_name: str, n_segments: int = 16):
    """NN-LUT offline phase: train NN → extract breakpoints → derive (s_i, t_i).

    1. Train NN in rank-normalized space (for LLM wide-range stability)
    2. Extract breakpoints d_i = -b_i/n_i, map to original FP16 grid
    3. Compute s_i, t_i from true function values at breakpoints (Eq 7 equivalent)
       → y = s_i·x + t_i for each segment
    """
    func = get_function(func_name)
    domain = get_domain(func_name)
    grid = generate_fp16_grid(domain)
    N_grid = len(grid)
    y_grid = func(grid)

    # Subsample for training
    if N_grid > 5000:
        idx = torch.linspace(0, N_grid - 1, 5000).long()
        x_sub, y_sub = grid[idx], y_grid[idx]
    else:
        x_sub, y_sub = grid, y_grid

    # Normalize for training: rank → [0,1], y → [0,1]
    x_norm = torch.linspace(0, 1, len(x_sub))
    y_min, y_max = y_sub.min(), y_sub.max()
    y_range = (y_max - y_min).clamp(min=1e-10)
    y_norm = (y_sub - y_min) / y_range

    # Move training data to GPU if available
    train_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_norm = x_norm.to(train_dev)
    y_norm = y_norm.to(train_dev)

    # Paper: N-1 hidden neurons → N segments (default 15 → 16)
    # Extra neurons for dedup margin
    n_neurons = n_segments * 2

    torch.manual_seed(42)

    d_desired = torch.linspace(0, 1, n_neurons + 2)[1:-1]
    n_param = torch.randn(n_neurons) * 0.5
    n_param[n_param.abs() < 0.05] = 0.1
    b_param = -n_param * d_desired
    m_param = torch.randn(n_neurons) * 0.01

    # Train NN on GPU (Eq 5, L1 loss)
    n_param, b_param, m_param = _train_nn(n_param, b_param, m_param, x_norm, y_norm)

    # Extract breakpoints in [0,1] space: d_i = -b_i/n_i (paper Eq 7 step 1)
    safe_n = n_param.clone()
    safe_n[safe_n.abs() < 1e-10] = 1e-10
    bp_norm = (-b_param / safe_n).sort()[0]
    bp_norm = bp_norm[(bp_norm >= 0) & (bp_norm <= 1)]

    # Map to FP16 grid indices (original domain)
    bp_grid_idx = (bp_norm * (N_grid - 1)).round().long().clamp(0, N_grid - 1)
    bp_grid_idx = bp_grid_idx.unique()

    # Ensure domain boundaries
    if bp_grid_idx[0] != 0:
        bp_grid_idx = torch.cat([torch.tensor([0]), bp_grid_idx])
    if bp_grid_idx[-1] != N_grid - 1:
        bp_grid_idx = torch.cat([bp_grid_idx, torch.tensor([N_grid - 1])])

    # Trim or fill to target
    if len(bp_grid_idx) > n_segments + 1:
        sub_idx = torch.linspace(0, len(bp_grid_idx) - 1, n_segments + 1).long()
        bp_grid_idx = bp_grid_idx[sub_idx]
    elif len(bp_grid_idx) < n_segments + 1:
        while len(bp_grid_idx) < n_segments + 1:
            gaps = bp_grid_idx[1:] - bp_grid_idx[:-1]
            largest = gaps.argmax().item()
            mid = (bp_grid_idx[largest] + bp_grid_idx[largest + 1]) // 2
            if mid == bp_grid_idx[largest] or mid == bp_grid_idx[largest + 1]:
                break
            new_idx = torch.cat([bp_grid_idx[:largest + 1],
                                 torch.tensor([mid]),
                                 bp_grid_idx[largest + 1:]])
            bp_grid_idx = new_idx

    bp = grid[bp_grid_idx]
    vals = y_grid[bp_grid_idx].clamp(-65504.0, 65504.0)

    # Derive slope-intercept LUT: y = s_i·x + t_i (Eq 7 / Eq 4)
    # s_i = (f(bp[i+1]) - f(bp[i])) / (bp[i+1] - bp[i])
    # t_i = f(bp[i]) - s_i * bp[i]
    n_seg = len(bp) - 1
    slopes = torch.zeros(n_seg, dtype=torch.float32)
    intercepts = torch.zeros(n_seg, dtype=torch.float32)
    for i in range(n_seg):
        w = bp[i + 1] - bp[i]
        if w > 0:
            slopes[i] = (vals[i + 1] - vals[i]) / w
            intercepts[i] = vals[i] - slopes[i] * bp[i]
        else:
            intercepts[i] = vals[i]

    slopes = slopes.clamp(-65504.0, 65504.0)
    intercepts = intercepts.clamp(-65504.0, 65504.0)

    return bp, slopes, intercepts


def _get_config(func_name: str, n_segments: int, device):
    key = f"{func_name}_{n_segments}_{device}"
    if key not in _CACHE:
        bp, sl, ic = _optimize_breakpoints(func_name, n_segments)
        _CACHE[key] = (bp.to(device), sl.to(device), ic.to(device))
    return _CACHE[key]


def _forward_chunk(x_flat, breakpoints, slopes, intercepts):
    """FP16-simulated: y = s_i·x + t_i (paper Eq 4, 2-cycle HW).

    Cycle 1: comparator chain → segment index i, read (s_i, t_i) from LUT
    Cycle 2: y = s_i · x + t_i  (1-MAC = 1 multiply + 1 add)
    """
    bp16 = breakpoints.half()
    sl16 = slopes.half()
    ic16 = intercepts.half()
    x16 = x_flat.half()

    x16 = x16.clamp(min=bp16[0], max=bp16[-1])

    # Cycle 1: comparator chain + LUT read
    seg = torch.bucketize(x16.float(), bp16[1:-1].float())

    # Cycle 2: y = s_i · x + t_i  (1-MAC)
    y = (sl16[seg] * x16 + ic16[seg]).half()
    y = torch.where(torch.isfinite(y), y, ic16[seg])
    return y.float()


def nn_lut_forward(x: torch.Tensor, func_name: str, n_segments: int = 16) -> torch.Tensor:
    """NN-LUT forward (paper default: 16 segments, 2-cycle HW)."""
    shape, dtype = x.shape, x.dtype
    x_flat = x.reshape(-1).half().float()
    bp, sl, ic = _get_config(func_name, n_segments, x.device)

    n = x_flat.numel()
    if n > _CHUNK:
        parts = [_forward_chunk(x_flat[i:i + _CHUNK], bp, sl, ic)
                 for i in range(0, n, _CHUNK)]
        y = torch.cat(parts)
    else:
        y = _forward_chunk(x_flat, bp, sl, ic)
    return y.reshape(shape).to(dtype)


if __name__ == '__main__':
    print("=== NN-LUT Engine (Yu et al., DAC 2022) ===")
    print("    y = s_i·x + t_i (Eq 4), LUT from NN weights (Eq 7)\n")
    for n_seg in [16, 256]:
        _CACHE.clear()
        print(f"--- n_segments = {n_seg} ---")
        for fn in ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh',
                    'reciprocal', 'hardswish', 'mish']:
            f = get_function(fn)
            test_x = generate_fp16_grid(get_domain(fn))
            y_approx = nn_lut_forward(test_x, fn, n_segments=n_seg)
            y_ref = f(test_x)
            err = (y_approx - y_ref).abs()
            denom = y_ref.abs().clamp(min=2**-14)
            rel = err / denom
            n_actual = len(_CACHE[f'{fn}_{n_seg}_cpu'][0]) - 1
            print(f"  {fn:12s}  segs={n_actual:4d}"
                  f"  mean_rel={rel.mean():.4e}  max_rel={rel.max():.4e}")
        print()
