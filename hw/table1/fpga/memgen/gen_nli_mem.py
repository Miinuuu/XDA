#!/usr/bin/env python3
"""Generate .mem files for NLI engine Verilog testbench (HW bit-exact expected)."""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'sw'))

from nli_engine import build_lut_from_paper
from nli_dp import get_function, get_domain
from HW.fp16_hw_emu import fp16_bits, bits_to_fp16, fp16_ge, fp_adder, fp_mult_norm


def fp16_hex(val):
    return format(np.float16(val).view(np.uint16), '04X')


def _pack_isZero(bits):
    """RTL pattern: isZero ? 16'h0000 : packed"""
    return 0x0000 if bits_to_fp16(bits) == 0 else bits


# =========================================================================
# Bit-exact Stage 2: fixed-point floor/frac (matches nli_engine.v lines 206-287)
# =========================================================================
def _stage2(scaled_bits, mul2_isZero, p1_index, NUM_INTERVALS, D_N=32):
    BIAS = 15; D_N_BITS = 5
    exp_s2 = (scaled_bits >> 10) & 0x1F
    mant_s2 = scaled_bits & 0x3FF

    # full_mant_s2 = (exp==0) ? {0, mant} : {1, mant}
    full_mant = mant_s2 if exp_s2 == 0 else (1 << 10) | mant_s2

    # Convert to 6.10 fixed point: {5'b0, full_mant} shifted
    val_16 = full_mant  # starts as 11-bit, will be placed in 16-bit
    if mul2_isZero or exp_s2 == 0:
        fixed_6_10 = 0
    elif exp_s2 < BIAS:
        # right shift by (BIAS - exp_s2), treating {5'b0, full_mant} as 16-bit
        fixed_6_10 = (val_16 >> (BIAS - exp_s2)) & 0xFFFF
    elif (exp_s2 - BIAS) <= 5:
        fixed_6_10 = (val_16 << (exp_s2 - BIAS)) & 0xFFFF
    else:
        fixed_6_10 = 0xFFFF

    floor_raw = (fixed_6_10 >> 10) & 0x3F
    frac_bits = fixed_6_10 & 0x3FF

    # LZC on 10-bit fraction (casez logic)
    lzc_frac = 10
    for i in range(9, -1, -1):
        if (frac_bits >> i) & 1:
            lzc_frac = 9 - i
            break

    # frac → FP16
    if lzc_frac >= 10 or mul2_isZero:
        decimal_fp16 = 0x0000
    else:
        dec_exp = 14 - lzc_frac
        dec_mant = (frac_bits << (lzc_frac + 1)) & 0x3FF
        decimal_fp16 = (dec_exp << 10) | dec_mant

    # Bypass: if exp < BIAS and not zero, use scaled_pos directly
    if exp_s2 < BIAS and not mul2_isZero:
        decimal = scaled_bits & 0xFFFF
    else:
        decimal = decimal_fp16

    # Clamp decimal >= 1.0 to 0x3C00
    dec_exp_val = (decimal >> 10) & 0x1F
    if dec_exp_val >= BIAS and ((decimal >> 15) & 1) == 0 and decimal != 0:
        decimal = 0x3C00

    # Address clamping
    if p1_index == 0 or p1_index == NUM_INTERVALS - 1:
        address = 0
    elif floor_raw >= D_N:
        address = D_N - 1
    else:
        address = floor_raw & ((1 << D_N_BITS) - 1)

    # Global LUT index
    if p1_index == 0:
        global_idx = address
    else:
        global_idx = 1 + (p1_index - 1) * D_N + address

    return global_idx, decimal


def hw_nli_forward(x_bits, point_bits, mul_bits, lut_bits):
    """Bit-exact emulation of nli_engine.v 4-stage pipeline."""
    M = len(point_bits); NUM_INTERVALS = M - 1; D_N = 32
    LUT_DEPTH = len(lut_bits)

    # --- Stage 1: Clamp + Comparator + Sub ---
    below_min = not fp16_ge(x_bits, point_bits[0])
    above_max = fp16_ge(x_bits, point_bits[M - 1])
    clamped = point_bits[0] if below_min else (point_bits[M - 1] if above_max else x_bits)

    cmp_index = 0
    for ci in range(1, M):
        if fp16_ge(clamped, point_bits[ci]):
            cmp_index = ci
    if cmp_index >= NUM_INTERVALS:
        cmp_index = NUM_INTERVALS - 1

    offset_bits = _pack_isZero(fp_adder(clamped, point_bits[cmp_index] ^ 0x8000))

    # --- Stage 2: Scale * offset + Floor/Frac ---
    scaled_bits = _pack_isZero(fp_mult_norm(mul_bits[cmp_index], offset_bits))
    mul2_isZero = (scaled_bits == 0)

    global_idx, decimal_bits = _stage2(scaled_bits, mul2_isZero, cmp_index, NUM_INTERVALS, D_N)
    if global_idx > LUT_DEPTH - 2:
        global_idx = LUT_DEPTH - 2

    # --- Stage 3: LUT read + Sub ---
    y0 = lut_bits[global_idx]; y1 = lut_bits[global_idx + 1]
    diff_bits = _pack_isZero(fp_adder(y1, y0 ^ 0x8000))

    # --- Stage 4: Mul + Add ---
    prod_bits = _pack_isZero(fp_mult_norm(decimal_bits, diff_bits))
    result_bits = _pack_isZero(fp_adder(y0, prod_bits))
    return result_bits


def generate_mem_files(func_name, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating NLI memory files for: {func_name}")

    point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)

    with open(os.path.join(output_dir, 'point_reg.mem'), 'w') as f:
        f.write(f"// point_reg for {func_name} (11 macro cutpoints, FP16 hex)\n")
        for i, val in enumerate(point_reg.tolist()):
            f.write(f"{fp16_hex(val)}  // [{i}] = {val}\n")

    with open(os.path.join(output_dir, 'mul_reg.mem'), 'w') as f:
        f.write(f"// mul_reg for {func_name} (10 scale factors, FP16 hex)\n")
        for i, val in enumerate(mul_reg.tolist()):
            f.write(f"{fp16_hex(val)}  // [{i}] = {val}\n")

    with open(os.path.join(output_dir, 'lut_reg.mem'), 'w') as f:
        f.write(f"// lut_reg for {func_name} (259 function values, FP16 hex)\n")
        for i, val in enumerate(lut_reg.tolist()):
            f.write(f"{fp16_hex(val)}  // [{i}] = {val:.6f}\n")

    # HW-format registers
    point_bits = [fp16_bits(v) for v in point_reg.tolist()]
    mul_bits = [fp16_bits(v) for v in mul_reg.tolist()]
    lut_bits_arr = [fp16_bits(v) for v in lut_reg.tolist()]

    # Test vectors
    domain = get_domain(func_name); lo, hi = domain
    test_inputs = torch.linspace(lo, hi, 200)
    extras = torch.tensor([lo - 1.0, lo, lo + 0.001, hi - 0.001, hi, hi + 1.0])
    test_inputs = torch.cat([test_inputs, extras])

    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// Test vectors for {func_name} (HW bit-exact)\n")
        f.write(f"// {len(test_inputs)} test cases\n")
        count = 0
        for x_val in test_inputs.tolist():
            x_b = fp16_bits(x_val)
            y_b = hw_nli_forward(x_b, point_bits, mul_bits, lut_bits_arr)
            y_hw = float(bits_to_fp16(y_b))
            if np.isnan(y_hw): continue
            f.write(f"{x_b:04X} {y_b:04X}  // x={x_val:.4f} y_hw={y_hw:.6f}\n")
            count += 1
    print(f"  test_vectors.mem : {count} test pairs (HW bit-exact)")


if __name__ == '__main__':
    func_name = sys.argv[1] if len(sys.argv) > 1 else 'silu'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    generate_mem_files(func_name, output_dir)
