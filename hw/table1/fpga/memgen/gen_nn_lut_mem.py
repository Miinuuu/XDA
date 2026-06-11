#!/usr/bin/env python3
"""Generate .mem files for NN-LUT Verilog simulation (HW bit-exact expected)."""

import sys, os, argparse
import numpy as np

from nn_lut_engine import _optimize_breakpoints
from nli_dp import get_function, get_domain, generate_fp16_grid
from HW.fp16_hw_emu import fp16_bits, bits_to_fp16, fp16_ge, fp_adder, fp_mult_norm


def fp16_to_hex(val):
    return format(np.float16(val).view(np.uint16), '04X')


def hw_nn_lut_forward(x_bits, bp_bits, slope_bits, inter_bits):
    """Bit-exact emulation of nn_lut_engine.v: y = s_i * x + t_i."""
    N_BP = len(bp_bits)
    # Stage 1: comparator chain
    cmp_index = 0
    for ci in range(N_BP):
        if fp16_ge(x_bits, bp_bits[ci]):
            cmp_index = ci + 1
    # Stage 2: MAC  y = slope * x + intercept
    product_bits = fp_mult_norm(slope_bits[cmp_index], x_bits)
    if bits_to_fp16(product_bits) == 0:
        product_bits = 0x0000
    result_bits = fp_adder(product_bits, inter_bits[cmp_index])
    if bits_to_fp16(result_bits) == 0:
        result_bits = 0x0000
    return result_bits


def generate_mem_files(func_name, n_segments=16, out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    bp, slopes, intercepts = _optimize_breakpoints(func_name, n_segments)
    bp_np = bp.numpy(); slopes_np = slopes.numpy(); intercepts_np = intercepts.numpy()
    n_seg = len(slopes_np); n_internal_bp = n_seg - 1

    # breakpoint_reg.mem: N-1 internal breakpoints
    with open(os.path.join(out_dir, 'breakpoint_reg.mem'), 'w') as f:
        f.write(f"// breakpoint_reg for {func_name} ({n_internal_bp} internal breakpoints, FP16 hex)\n")
        for i in range(n_internal_bp):
            f.write(f"{fp16_to_hex(bp_np[i + 1])}  // [{i}] = {bp_np[i+1]:.6g}\n")

    # slope_reg.mem
    with open(os.path.join(out_dir, 'slope_reg.mem'), 'w') as f:
        f.write(f"// slope_reg for {func_name} ({n_seg} slopes, FP16 hex)\n")
        for i, s in enumerate(slopes_np):
            f.write(f"{fp16_to_hex(s)}  // [{i}] = {s:.6g}\n")

    # intercept_reg.mem
    with open(os.path.join(out_dir, 'intercept_reg.mem'), 'w') as f:
        f.write(f"// intercept_reg for {func_name} ({n_seg} intercepts, FP16 hex)\n")
        for i, t in enumerate(intercepts_np):
            f.write(f"{fp16_to_hex(t)}  // [{i}] = {t:.6g}\n")

    # Prepare HW-format registers (fp16 bits)
    bp_bits = [fp16_bits(bp_np[i + 1]) for i in range(n_internal_bp)]
    slope_bits = [fp16_bits(s) for s in slopes_np]
    inter_bits = [fp16_bits(t) for t in intercepts_np]

    # Test vectors with HW bit-exact expected
    grid = generate_fp16_grid(get_domain(func_name))
    n_test = min(200, len(grid))
    idx = np.linspace(0, len(grid) - 1, n_test, dtype=int)
    test_x = grid[idx]

    with open(os.path.join(out_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// test_vectors for {func_name} ({n_test} inputs, FP16 hex, HW bit-exact)\n")
        for x in test_x.numpy():
            f.write(f"{fp16_to_hex(x)}  // x={x:.6g}\n")

    with open(os.path.join(out_dir, 'test_expected.mem'), 'w') as f:
        f.write(f"// test_expected for {func_name} ({n_test} outputs, FP16 hex, HW bit-exact)\n")
        for x in test_x.numpy():
            x_b = fp16_bits(x)
            y_b = hw_nn_lut_forward(x_b, bp_bits, slope_bits, inter_bits)
            f.write(f"{y_b:04X}  // y_hw={float(bits_to_fp16(y_b)):.6g}\n")

    print(f"Generated {func_name}: {n_internal_bp} bp, {n_seg} seg, {n_test} tests (HW bit-exact)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='silu')
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        for fn in ['silu','exp','rsqrt','gelu','sigmoid','tanh','reciprocal','hardswish','mish']:
            generate_mem_files(fn, args.segments, os.path.join(args.outdir, fn))
    else:
        generate_mem_files(args.func, args.segments, args.outdir)
