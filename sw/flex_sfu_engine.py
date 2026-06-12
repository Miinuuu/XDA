"""
Flex-SFU Engine: Software reimplementation of Flex-SFU (Reggiani et al., DAC 2023)
===================================================================================
Non-uniform piecewise linear approximation with SGD-optimized breakpoints.

Faithful to the paper's algorithm:
  Phase 1: Adam optimizer with ReduceLROnPlateau to optimize breakpoint positions
  Phase 2: Iterative insertion/removal heuristic — remove breakpoint with minimal
           removal loss, insert at center of segment with maximal insertion loss,
           then retrain with lower learning rate.
Forward pass simulates FP16 hardware: binary-tree lookup + PWL interpolation.
"""

import torch
from typing import Dict, Tuple
from nli_dp import get_function, get_domain, generate_fp16_grid

_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_CHUNK = 4 * 1024 * 1024


def _pwl_forward_differentiable(x, internal_bp, lo, hi, func):
    """Differentiable PWL forward for training breakpoints."""
    sorted_bp, _ = internal_bp.sort()
    all_bp = torch.cat([torch.tensor([lo], device=x.device),
                        sorted_bp,
                        torch.tensor([hi], device=x.device)])
    bp_vals = func(all_bp)

    seg_idx = torch.bucketize(x, all_bp[1:-1])
    bp_left = all_bp[seg_idx]
    bp_right = all_bp[seg_idx + 1]
    y_left = bp_vals[seg_idx]
    y_right = bp_vals[seg_idx + 1]

    t = (x - bp_left) / (bp_right - bp_left).clamp(min=1e-20)
    t = t.clamp(0.0, 1.0)
    return y_left + t * (y_right - y_left)


@torch.enable_grad()
def _train_breakpoints(x_train, y_train, internal_bp, lo, hi, func,
                       n_steps=300, lr=0.1):
    """Adam optimization of breakpoint positions."""
    internal_bp.requires_grad_(True)
    optimizer = torch.optim.Adam([internal_bp], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5)
    for _ in range(n_steps):
        optimizer.zero_grad()
        y_pred = _pwl_forward_differentiable(x_train, internal_bp, lo, hi, func)
        # Relative MSE: necessary for wide-range LLM functions (rsqrt, reciprocal)
        # Paper uses absolute MSE for narrow-range BERT functions
        denom = y_train.abs().clamp(min=2**-14)
        loss = (((y_pred - y_train) / denom) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        with torch.no_grad():
            internal_bp.clamp_(lo, hi)
    internal_bp.requires_grad_(False)
    return internal_bp


def _segment_mse(x_train, y_train, all_bp, func, seg_start, seg_end):
    """MSE for a single segment [seg_start, seg_end]."""
    mask = (x_train >= all_bp[seg_start]) & (x_train < all_bp[seg_end])
    if seg_end == len(all_bp) - 1:
        mask = mask | (x_train == all_bp[seg_end])
    xs = x_train[mask]
    if len(xs) == 0:
        return 0.0
    ys = y_train[mask]
    bl, br = all_bp[seg_start], all_bp[seg_end]
    yl, yr = func(bl.unsqueeze(0)).squeeze(), func(br.unsqueeze(0)).squeeze()
    t = (xs - bl) / (br - bl).clamp(min=1e-20)
    y_pred = yl + t * (yr - yl)
    return ((y_pred - ys) ** 2).mean().item()


def _optimize_breakpoints(func_name: str, n_segments: int = 256,
                          n_steps_phase1: int = 300, n_rounds: int = 5):
    """Flex-SFU optimization: Adam SGD + insertion/removal heuristic."""
    func = get_function(func_name)
    domain = get_domain(func_name)
    x_grid = generate_fp16_grid(domain)
    y_grid = func(x_grid)
    N = len(x_grid)
    lo, hi = x_grid[0].item(), x_grid[-1].item()

    # Subsample for training speed
    if N > 5000:
        idx = torch.linspace(0, N - 1, 5000).long()
        x_train = x_grid[idx]
        y_train = y_grid[idx]
    else:
        x_train, y_train = x_grid, y_grid

    # Phase 1: Adam optimization from FP16-grid-spaced initialization
    # (log-spaced for positive domains, handles multi-decade functions)
    init_idx = torch.linspace(0, N - 1, n_segments + 1).long()
    internal_bp = x_grid[init_idx[1:-1]].clone().float()
    internal_bp = _train_breakpoints(
        x_train, y_train, internal_bp, lo, hi, func,
        n_steps=n_steps_phase1, lr=0.1)

    # Phase 2: Insertion/removal heuristic
    for _round in range(n_rounds):
        with torch.no_grad():
            sorted_bp, _ = internal_bp.sort()
            all_bp = torch.cat([torch.tensor([lo]), sorted_bp, torch.tensor([hi])])
            n_bp = len(all_bp)

            # Removal: find breakpoint with minimal removal loss
            # (removing it causes the smallest MSE increase)
            min_removal_loss = float('inf')
            min_removal_idx = -1
            for j in range(1, n_bp - 1):  # skip domain boundaries
                # MSE of merged segment [j-1, j+1] minus sum of two original segments
                mse_merged = _segment_mse(x_train, y_train, all_bp, func, j - 1, j + 1)
                mse_left = _segment_mse(x_train, y_train, all_bp, func, j - 1, j)
                mse_right = _segment_mse(x_train, y_train, all_bp, func, j, j + 1)
                removal_loss = mse_merged - (mse_left + mse_right)
                if removal_loss < min_removal_loss:
                    min_removal_loss = removal_loss
                    min_removal_idx = j

            # Insertion: find segment with maximal insertion loss
            # (segment_length * segment_MSE)
            max_insertion_loss = -1.0
            max_insertion_seg = -1
            for j in range(n_bp - 1):
                seg_len = (all_bp[j + 1] - all_bp[j]).item()
                seg_mse = _segment_mse(x_train, y_train, all_bp, func, j, j + 1)
                insertion_loss = seg_len * seg_mse
                if insertion_loss > max_insertion_loss:
                    max_insertion_loss = insertion_loss
                    max_insertion_seg = j

            # Apply removal and insertion
            if min_removal_idx >= 0 and max_insertion_seg >= 0:
                # Remove
                new_bp_list = [all_bp[k].item() for k in range(n_bp)
                               if k != min_removal_idx]
                # Insert at center of worst segment
                ins_idx = max_insertion_seg
                if ins_idx >= min_removal_idx:
                    ins_idx -= 1  # adjust for removal
                ins_idx = max(0, min(ins_idx, len(new_bp_list) - 2))
                center = (new_bp_list[ins_idx] + new_bp_list[ins_idx + 1]) / 2.0
                new_bp_list.insert(ins_idx + 1, center)

                # Reconstruct internal breakpoints (exclude domain boundaries)
                internal_bp = torch.tensor(new_bp_list[1:-1], dtype=torch.float32)

        # Retrain with lower learning rate
        internal_bp = _train_breakpoints(
            x_train, y_train, internal_bp, lo, hi, func,
            n_steps=100, lr=0.01)

    # Extract final breakpoints and compute slopes/intercepts
    with torch.no_grad():
        sorted_bp, _ = internal_bp.sort()
        all_bp = torch.cat([torch.tensor([lo]), sorted_bp, torch.tensor([hi])])
        vals = func(all_bp).clamp(-65504.0, 65504.0)

        slopes = torch.zeros(len(all_bp) - 1, dtype=torch.float32)
        for i in range(len(slopes)):
            w = all_bp[i + 1] - all_bp[i]
            if w > 0:
                slopes[i] = (vals[i + 1] - vals[i]) / w

    return all_bp.detach(), slopes, vals[:-1].detach()


def _get_config(func_name: str, n_segments: int, device):
    key = f"{func_name}_{n_segments}_{device}"
    if key not in _CACHE:
        bp, sl, ic = _optimize_breakpoints(func_name, n_segments)
        _CACHE[key] = (bp.to(device), sl.to(device), ic.to(device))
    return _CACHE[key]


def _forward_chunk(x_flat, breakpoints, slopes, intercepts):
    """FP16-simulated PWL on a flat chunk."""
    bp16 = breakpoints.half()
    sl16 = slopes.half()
    ic16 = intercepts.half()
    x16 = x_flat.half()

    x16 = x16.clamp(min=bp16[0], max=bp16[-1])
    seg = torch.bucketize(x16.float(), bp16[1:-1].float())

    offset = (x16 - bp16[seg])                       # FP16 sub
    y = (ic16[seg] + sl16[seg] * offset).half()       # FP16 mul+add
    y = torch.where(torch.isfinite(y), y, ic16[seg])
    return y.float()


def flex_sfu_forward(x: torch.Tensor, func_name: str, n_segments: int = 256) -> torch.Tensor:
    """Flex-SFU forward pass with FP16 quantization and chunking."""
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
    print("=== Flex-SFU Engine Test (Adam + insertion/removal) ===\n")
    for fn in ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh', 'reciprocal', 'hardswish', 'mish']:
        f = get_function(fn)
        lo, hi = get_domain(fn)
        test_x = torch.linspace(lo, hi, 5000)
        y_approx = flex_sfu_forward(test_x, fn, n_segments=256)
        y_ref = f(test_x)
        err = (y_approx - y_ref).abs()
        denom = y_ref.abs().clamp(min=2**-14)
        rel = err / denom
        print(f"  {fn:12s}  mean_rel={rel.mean():.4e}  max_rel={rel.max():.4e}")
