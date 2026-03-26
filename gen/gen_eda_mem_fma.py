#!/usr/bin/env python3
"""
Generate .mem files for EDA-NLI 7-stage FMA engine testbench.

Uses bit-exact fma_interp emulation (single normalize/round) instead of
separate frac_mult + fp_adder (2x normalize/round).
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sw'))
from nli_eda import optimize_eda, get_function, get_domain

# Import shared utilities from the original gen_eda_mem
from gen_eda_mem import (
    fp32_to_fp16_hex, fp32_to_fp16_bits, fp16_bits_to_float,
    _fp16_bits, _bits_to_fp16, _fp_adder,
    generate_mem_files as _gen_config_and_lut,
)

T_BITS = 10


def _fma_interp(y0_bits: int, diff_bits: int, t_int: int) -> int:
    """Bit-exact emulation of fma_interp.v: result = y0 + diff * (t_int / 2^T_BITS).

    Single normalize/round at output (no intermediate rounding).
    """
    EXP_WIDTH = 5
    MANT_WIDTH = 10
    EXP_MAX = 31
    FULL_MANT = MANT_WIDTH + 1        # 11
    PROD_WIDTH = T_BITS + FULL_MANT    # 21
    GUARD_BITS = 3
    WORK_WIDTH = PROD_WIDTH + GUARD_BITS + 1  # 25
    FRAC_START = GUARD_BITS + (PROD_WIDTH - FULL_MANT)  # 13
    MANT_MSB = WORK_WIDTH - 3          # 22
    MANT_LSB = FRAC_START              # 13

    MASK_WORK = (1 << WORK_WIDTH) - 1
    MASK_WORK1 = (1 << (WORK_WIDTH + 1)) - 1

    # --- Unpack y0 ---
    y0_sign = (y0_bits >> 15) & 1
    y0_exp  = (y0_bits >> 10) & 0x1F
    y0_mant = y0_bits & 0x3FF
    y0_isZero   = (y0_exp == 0) and (y0_mant == 0)
    y0_isDenorm = (y0_exp == 0) and (y0_mant != 0)
    y0_isInf    = (y0_exp == EXP_MAX) and (y0_mant == 0)
    y0_isNaN    = (y0_exp == EXP_MAX) and (y0_mant != 0)

    if y0_isZero:        y0_full = 0
    elif y0_isDenorm:    y0_full = y0_mant
    else:                y0_full = (1 << MANT_WIDTH) | y0_mant
    y0_eff_exp = 1 if y0_exp == 0 else y0_exp

    # --- Unpack diff ---
    diff_sign = (diff_bits >> 15) & 1
    diff_exp  = (diff_bits >> 10) & 0x1F
    diff_mant = diff_bits & 0x3FF
    diff_isZero   = (diff_exp == 0) and (diff_mant == 0)
    diff_isDenorm = (diff_exp == 0) and (diff_mant != 0)
    diff_isInf    = (diff_exp == EXP_MAX) and (diff_mant == 0)
    diff_isNaN    = (diff_exp == EXP_MAX) and (diff_mant != 0)

    if diff_isZero:      diff_full = 0
    elif diff_isDenorm:  diff_full = diff_mant
    else:                diff_full = (1 << MANT_WIDTH) | diff_mant
    diff_eff_exp = 1 if diff_exp == 0 else diff_exp

    t_isZero = (t_int == 0)

    # --- Stage 1: Integer multiply + extend y0 + compare ---
    product_raw = diff_full * t_int
    prod_isZero = diff_isZero or t_isZero or (product_raw == 0)

    y0_extended = y0_full << T_BITS  # {y0_full, 10'b0}

    y0_exp_larger  = y0_eff_exp > diff_eff_exp
    exp_equal      = y0_eff_exp == diff_eff_exp
    y0_mant_larger = y0_extended >= product_raw
    swap = not y0_exp_larger and (not exp_equal or not y0_mant_larger)

    x_exp = diff_eff_exp if swap else y0_eff_exp
    y_exp = y0_eff_exp   if swap else diff_eff_exp

    shift_amt = x_exp - y_exp

    # --- Stage 2: Swap + Align + Add/Subtract ---
    x_sign = diff_sign if swap else y0_sign
    y_sign = y0_sign   if swap else diff_sign
    x_mant_raw = product_raw if swap else y0_extended
    y_mant_raw = y0_extended  if swap else product_raw
    y_isZero_flag = y0_isZero if swap else prod_isZero

    x_mant_ext = (x_mant_raw << GUARD_BITS) & MASK_WORK
    y_mant_ext = (y_mant_raw << GUARD_BITS) & MASK_WORK

    # Alignment shift with sticky
    if shift_amt >= WORK_WIDTH:
        y_shifted = 0
        sticky_mask_val = y_mant_ext
    else:
        y_shifted = (y_mant_ext >> shift_amt) & MASK_WORK
        sticky_mask_val = y_mant_ext & ((1 << shift_amt) - 1) if shift_amt > 0 else 0

    sticky_s = 1 if sticky_mask_val != 0 else 0

    if y_isZero_flag:
        y_aligned = 0
    else:
        y_aligned = ((y_shifted & ~1) | (y_shifted & 1) | sticky_s) & MASK_WORK

    # Add or subtract
    eff_sub = x_sign ^ y_sign
    if eff_sub:
        s = (x_mant_ext - y_aligned) & MASK_WORK1
    else:
        s = (x_mant_ext + y_aligned) & MASK_WORK1

    sum_negative = (s >> WORK_WIDTH) & 1
    if sum_negative:
        sum_abs = ((~s + 1) & MASK_WORK)
        result_sign = 1 - x_sign
    else:
        sum_abs = s & MASK_WORK
        result_sign = x_sign

    # --- Stage 3: LZC + Normalize + Round ---
    lzc = 0
    for i in range(WORK_WIDTH - 1, -1, -1):
        if (sum_abs >> i) & 1:
            break
        lzc += 1
    if sum_abs == 0:
        lzc = WORK_WIDTH

    add_overflow = (sum_abs >> (WORK_WIDTH - 1)) & 1

    if add_overflow:
        norm_mant = (sum_abs >> 1) & MASK_WORK
        norm_exp = x_exp + 1
    else:
        shift = max(lzc - 1, 0)
        norm_mant = (sum_abs << shift) & MASK_WORK
        norm_exp = x_exp - lzc + 1

    # Extract mantissa and rounding bits
    trunc_mant = (norm_mant >> MANT_LSB) & ((1 << MANT_WIDTH) - 1)
    guard     = (norm_mant >> (FRAC_START - 1)) & 1
    round_bit = (norm_mant >> (FRAC_START - 2)) & 1
    sticky_r  = 1 if (norm_mant & ((1 << (FRAC_START - 2)) - 1)) != 0 else 0
    round_up  = guard & (round_bit | sticky_r | (trunc_mant & 1))

    rounded = trunc_mant + (1 if round_up else 0)
    round_overflow = (rounded >> MANT_WIDTH) & 1

    if round_overflow:
        final_exp  = norm_exp + 1
        final_mant = (rounded >> 1) & ((1 << MANT_WIDTH) - 1)
    else:
        final_exp  = norm_exp
        final_mant = rounded & ((1 << MANT_WIDTH) - 1)

    # --- Special cases ---
    prod_isInf    = diff_isInf and not prod_isZero
    result_isNaN  = y0_isNaN or diff_isNaN or \
                    (y0_isInf and prod_isInf and (y0_sign != diff_sign))
    result_isInf_w = ((y0_isInf or prod_isInf) and not result_isNaN) or \
                     (not result_isNaN and not y0_isInf and not prod_isInf and final_exp >= EXP_MAX)
    sum_is_zero   = (sum_abs == 0)
    result_isZero = (y0_isZero and prod_isZero) or \
                    (sum_is_zero and not y0_isInf and not prod_isInf and
                     not y0_isNaN and not diff_isNaN)
    underflow     = (final_exp <= 0) and not result_isZero and not result_isInf_w and not result_isNaN

    # --- Output ---
    if result_isNaN:
        return (0 << 15) | (EXP_MAX << 10) | 1
    elif result_isZero:
        out_sign = y0_sign & diff_sign
        return (out_sign << 15)
    elif result_isInf_w:
        out_sign = y0_sign if y0_isInf else diff_sign
        return (out_sign << 15) | (EXP_MAX << 10)
    elif underflow:
        return (result_sign << 15)
    else:
        return (result_sign << 15) | ((final_exp & 0x1F) << 10) | (final_mant & 0x3FF)


def hw_eda_forward_fma(x_bits: int, config_rom: list, func_lut_f32: list) -> float:
    """Simulate the 7-stage FMA pipeline for a single FP16 input (as bits)."""
    sign = (x_bits >> 15) & 1
    exp  = (x_bits >> 10) & 0x1F
    mant = x_bits & 0x3FF

    # NaN passthrough
    if exp == 31 and mant != 0:
        return float('nan')

    # Stage 1: Config ROM lookup + bit extraction
    bin_addr = (sign << 5) | exp
    cfg_entry = config_rom[bin_addr]
    clamp     = (cfg_entry >> 12) & 1
    k_bits    = (cfg_entry >> 9) & 7
    base_off  = cfg_entry & 0x1FF

    if k_bits > 0 and k_bits <= 5:
        micro_idx = (mant >> (10 - k_bits)) & ((1 << k_bits) - 1)
        remaining = mant & ((1 << (10 - k_bits)) - 1)
        t_int = remaining << k_bits
    else:
        micro_idx = 0
        t_int = mant & 0x3FF

    lut_addr = base_off + micro_idx
    max_addr = len(func_lut_f32) - 2
    if lut_addr > max_addr:
        lut_addr = max_addr

    y0 = func_lut_f32[lut_addr]
    y1 = func_lut_f32[lut_addr + 1]

    if clamp:
        return float(np.float16(y0))

    # Stage 2a: LUT read (done above)
    y0_bits = _fp16_bits(y0)
    y1_bits = _fp16_bits(y1)

    # Stages 2b-2c: fp_adder_2s subtraction (same function as fp_adder)
    neg_y0_bits = y0_bits ^ 0x8000
    diff_bits = _fp_adder(y1_bits, neg_y0_bits)

    # isZero handling
    diff_f16 = _bits_to_fp16(diff_bits)
    if diff_f16 == 0:
        diff_bits = 0x0000

    # Stages 3-5: FMA interpolation (single normalize/round)
    result_bits = _fma_interp(y0_bits, diff_bits, t_int)

    return float(_bits_to_fp16(result_bits))


def generate_test_vectors(func_name: str = 'silu', max_lut: int = 254,
                          max_k: int = 5, output_dir: str = '.',
                          exhaustive: bool = False):
    """Generate config_rom, func_lut, and FMA-accurate test vectors."""
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Generating EDA-NLI 4s (FMA) memory files for: {func_name}")
    config = optimize_eda(func_name, max_lut=max_lut, max_k=max_k,
                          device=device, verbose=True)
    domain = get_domain(func_name)
    func = get_function(func_name)

    # Detect clipped bins
    original_bins = {}
    for e in range(1, 31):
        original_bins[(0, e)] = (2.0**(e-15), 2.0**(e-14))
        original_bins[(1, e)] = (-(2.0**(e-14)), -(2.0**(e-15)))
    original_bins[(0, 0)] = (2.0**(-24), 2.0**(-14))
    original_bins[(1, 0)] = (-(2.0**(-14)), -(2.0**(-24)))

    bin_map = {}
    clipped_bins = set()
    for i, (b_start, b_end, sign_val, exp_val) in enumerate(config.bins):
        bin_map[(sign_val, exp_val)] = i
        orig = original_bins.get((sign_val, exp_val))
        if orig and (abs(b_start - orig[0]) > 1e-10 or abs(b_end - orig[1]) > 1e-10):
            clipped_bins.add((sign_val, exp_val))

    # Build config ROM (64 entries) — same as original
    first_lut_idx = 0
    last_lut_idx = len(config.lut_values) - 1
    config_rom_entries = []

    with open(os.path.join(output_dir, 'config_rom.mem'), 'w') as f:
        f.write(f"// config_rom for {func_name} (4s FMA)\n")
        for addr in range(64):
            sign_val = (addr >> 5) & 1
            exp_val = addr & 0x1F
            if exp_val == 31:
                base = last_lut_idx if sign_val == 0 else first_lut_idx
                entry = (1 << 12) | (0 << 9) | (base & 0x1FF)
            elif (sign_val, exp_val) in bin_map:
                idx = bin_map[(sign_val, exp_val)]
                k = config.k_alloc[idx]
                base = int(config.base_offsets[idx].item())
                is_clipped = (sign_val, exp_val) in clipped_bins
                clamp_flag = 1 if is_clipped else 0
                entry = (clamp_flag << 12) | ((k & 0x7) << 9) | (base & 0x1FF)
            else:
                base = last_lut_idx if sign_val == 0 else first_lut_idx
                entry = (1 << 12) | (0 << 9) | (base & 0x1FF)
            config_rom_entries.append(entry)
            f.write(f"{entry:04X}\n")
    print(f"  config_rom.mem : 64 entries")

    # Build function LUT — same as original
    lut_vals = config.lut_values.cpu().float().tolist()
    with open(os.path.join(output_dir, 'func_lut.mem'), 'w') as f:
        f.write(f"// func_lut for {func_name}: {len(lut_vals)} entries\n")
        for i, val in enumerate(lut_vals):
            f.write(f"{fp32_to_fp16_hex(val)}\n")
    print(f"  func_lut.mem   : {len(lut_vals)} entries")

    # Generate test vectors with FMA-accurate expected values
    lo, hi = domain
    if exhaustive:
        # Enumerate ALL representable FP16 values within domain
        test_inputs_f32 = []
        for bits in range(0, 0x7C00):  # all positive FP16 (excl inf/nan)
            val = fp16_bits_to_float(bits)
            if val is not None and lo <= val <= hi:
                test_inputs_f32.append(val)
            if bits != 0:  # skip -0
                neg_val = fp16_bits_to_float(bits | 0x8000)
                if neg_val is not None and lo <= neg_val <= hi:
                    test_inputs_f32.append(neg_val)
        print(f"  Exhaustive: {len(test_inputs_f32)} FP16 values in [{lo}, {hi}]")
    else:
        test_inputs_f32 = torch.linspace(lo, hi, 200, device='cpu').tolist()
        test_inputs_f32 += [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]

    with open(os.path.join(output_dir, 'test_vectors_4s.mem'), 'w') as f:
        f.write(f"// test vectors for {func_name} (FMA pipeline, HW-accurate)\n")
        count = 0
        for x_f32 in test_inputs_f32:
            x_bits = fp32_to_fp16_bits(x_f32)
            y_hw = hw_eda_forward_fma(x_bits, config_rom_entries, lut_vals)
            if np.isnan(y_hw):
                continue
            y_bits = fp32_to_fp16_bits(y_hw)
            f.write(f"{x_bits:04X} {y_bits:04X}\n")
            count += 1
    print(f"  test_vectors_4s.mem : {count} test pairs (FMA-accurate)")

    # Error summary: FMA vs reference
    y_ref = func(torch.tensor(test_inputs_f32))
    y_hw_all = []
    for x in test_inputs_f32:
        xb = fp32_to_fp16_bits(x)
        yh = hw_eda_forward_fma(xb, config_rom_entries, lut_vals)
        y_hw_all.append(yh if not np.isnan(yh) else 0.0)
    y_hw_t = torch.tensor(y_hw_all)
    abs_err = torch.abs(y_hw_t - y_ref)
    print(f"\n  FMA HW vs reference: max_abs={abs_err.max().item():.4e} "
          f"mean_abs={abs_err.mean().item():.4e}")

    # Compare FMA vs original (2-round) pipeline
    from gen_eda_mem import hw_eda_forward_scalar
    diff_count = 0
    for x in test_inputs_f32:
        xb = fp32_to_fp16_bits(x)
        y_old = hw_eda_forward_scalar(xb, config_rom_entries, lut_vals)
        y_new = hw_eda_forward_fma(xb, config_rom_entries, lut_vals)
        if not np.isnan(y_old) and not np.isnan(y_new):
            if fp32_to_fp16_bits(y_old) != fp32_to_fp16_bits(y_new):
                diff_count += 1
    print(f"  FMA vs original pipeline: {diff_count}/{len(test_inputs_f32)} differ")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate mem files for EDA-NLI 4s FMA engine')
    parser.add_argument('--func', type=str, default='silu', help='Target function')
    parser.add_argument('--max-lut', type=int, default=254, help='Max LUT entries')
    parser.add_argument('--max-k', type=int, default=5, help='Max K bits')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--exhaustive', action='store_true',
                        help='Generate exhaustive FP16 test vectors (all representable values)')
    args = parser.parse_args()

    generate_test_vectors(args.func, args.max_lut, args.max_k, args.output_dir,
                          exhaustive=args.exhaustive)
