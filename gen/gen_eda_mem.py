#!/usr/bin/env python3
"""
Generate .mem files for EDA-NLI engine Verilog testbench.

Generates hardware-accurate expected values by simulating the exact
bit-extraction pipeline (T_BITS truncation, config ROM lookup, etc.).
"""

import sys, os, struct
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sw'))
from nli_eda import optimize_eda, get_function, get_domain, get_fp16_exponent_bins

T_BITS = 10  # adaptive: left-justified to 10 bits (matches RTL T_BITS=10)


def fp32_to_fp16_hex(val: float) -> str:
    fp16 = np.float16(val)
    bits = int.from_bytes(np.array([fp16], dtype=np.float16).tobytes(), 'little')
    return f"{bits:04X}"


def fp32_to_fp16_bits(val: float) -> int:
    fp16 = np.float16(val)
    return int.from_bytes(np.array([fp16], dtype=np.float16).tobytes(), 'little')


def fp16_bits_to_float(bits: int) -> float:
    return np.frombuffer(np.array([bits], dtype=np.uint16).tobytes(),
                         dtype=np.float16)[0].item()


def _fp16_bits(val) -> int:
    """Convert numpy float16 to 16-bit integer."""
    return int.from_bytes(np.array([np.float16(val)], dtype=np.float16).tobytes(), 'little')


def _bits_to_fp16(bits: int) -> np.float16:
    """Convert 16-bit integer to numpy float16."""
    return np.frombuffer((bits & 0xFFFF).to_bytes(2, 'little'), dtype=np.float16)[0]


def _fp_adder(a_bits: int, b_bits: int) -> int:
    """Bit-exact emulation of fp_adder.v for FP16. Returns packed 16-bit result."""
    EXP_WIDTH = 5
    MANT_WIDTH = 10
    EXP_MAX = 31
    FULL_MANT = MANT_WIDTH + 1  # 11
    GUARD_BITS = 3
    WORK_WIDTH = FULL_MANT + GUARD_BITS + 1  # 15

    a_sign = (a_bits >> 15) & 1
    a_exp  = (a_bits >> 10) & 0x1F
    a_mant = a_bits & 0x3FF
    b_sign = (b_bits >> 15) & 1
    b_exp  = (b_bits >> 10) & 0x1F
    b_mant = b_bits & 0x3FF

    a_isZero = (a_exp == 0) and (a_mant == 0)
    a_isInf  = (a_exp == EXP_MAX) and (a_mant == 0)
    a_isNaN  = (a_exp == EXP_MAX) and (a_mant != 0)
    a_isDenorm = (a_exp == 0) and (a_mant != 0)
    b_isZero = (b_exp == 0) and (b_mant == 0)
    b_isInf  = (b_exp == EXP_MAX) and (b_mant == 0)
    b_isNaN  = (b_exp == EXP_MAX) and (b_mant != 0)
    b_isDenorm = (b_exp == 0) and (b_mant != 0)

    # Full mantissa
    if a_isZero:    a_full = 0
    elif a_isDenorm: a_full = a_mant
    else:            a_full = (1 << MANT_WIDTH) | a_mant

    if b_isZero:    b_full = 0
    elif b_isDenorm: b_full = b_mant
    else:            b_full = (1 << MANT_WIDTH) | b_mant

    a_eff = 1 if a_exp == 0 else a_exp
    b_eff = 1 if b_exp == 0 else b_exp

    # Swap so X is larger
    a_exp_larger = a_eff > b_eff
    exp_equal = a_eff == b_eff
    a_mant_larger = a_full >= b_full
    swap = not a_exp_larger and (not exp_equal or not a_mant_larger)

    if swap:
        x_sign, y_sign = b_sign, a_sign
        x_exp, y_exp = b_eff, a_eff
        x_mant, y_mant = b_full, a_full
        x_isZero, y_isZero = b_isZero, a_isZero
    else:
        x_sign, y_sign = a_sign, b_sign
        x_exp, y_exp = a_eff, b_eff
        x_mant, y_mant = a_full, b_full
        x_isZero, y_isZero = a_isZero, b_isZero

    # Alignment
    shift_amt = x_exp - y_exp
    MASK = (1 << WORK_WIDTH) - 1

    x_mant_ext = (x_mant << GUARD_BITS) & MASK
    y_mant_ext = (y_mant << GUARD_BITS) & MASK

    if shift_amt >= WORK_WIDTH:
        y_shifted = 0
        sticky_bits = y_mant_ext
    else:
        y_shifted = (y_mant_ext >> shift_amt) & MASK
        shifted_out_mask = (1 << shift_amt) - 1 if shift_amt > 0 else 0
        sticky_bits = y_mant_ext & shifted_out_mask

    sticky = 1 if sticky_bits != 0 else 0

    if y_isZero:
        y_aligned = 0
    else:
        y_aligned = (y_shifted & ~1) | (y_shifted & 1) | sticky
        y_aligned &= MASK

    # Add or subtract
    eff_sub = x_sign ^ y_sign
    MASK_W1 = (1 << (WORK_WIDTH + 1)) - 1

    if eff_sub:
        s = (x_mant_ext - y_aligned) & MASK_W1
    else:
        s = (x_mant_ext + y_aligned) & MASK_W1

    sum_negative = (s >> WORK_WIDTH) & 1
    if sum_negative:
        sum_abs = ((~s + 1) & MASK)
        result_sign = 1 - x_sign
    else:
        sum_abs = s & MASK
        result_sign = x_sign

    # LZC
    lzc = 0
    for i in range(WORK_WIDTH - 1, -1, -1):
        if (sum_abs >> i) & 1:
            break
        lzc += 1

    add_overflow = (sum_abs >> (WORK_WIDTH - 1)) & 1

    if add_overflow:
        norm_mant = (sum_abs >> 1) & MASK
    else:
        shift = max(lzc - 1, 0)
        norm_mant = (sum_abs << shift) & MASK

    if add_overflow:
        norm_exp = x_exp + 1
    else:
        norm_exp = x_exp - lzc + 1

    # Rounding
    guard_val = norm_mant & ((1 << GUARD_BITS) - 1)
    lsb = (norm_mant >> GUARD_BITS) & 1
    round_up = ((guard_val >> (GUARD_BITS - 1)) & 1) and ((guard_val & ((1 << (GUARD_BITS - 1)) - 1)) != 0 or lsb)

    rounded = ((norm_mant >> GUARD_BITS) & ((1 << FULL_MANT) - 1)) + (1 if round_up else 0)
    round_overflow = (rounded >> FULL_MANT) & 1

    if round_overflow:
        final_exp = norm_exp + 1
        final_mant = (rounded >> 1) & ((1 << MANT_WIDTH) - 1)
    else:
        final_exp = norm_exp
        final_mant = rounded & ((1 << MANT_WIDTH) - 1)

    # Special cases
    result_isNaN = a_isNaN or b_isNaN or (a_isInf and b_isInf and (a_sign != b_sign))
    result_isInf = ((a_isInf or b_isInf) and not result_isNaN) or (final_exp >= EXP_MAX)
    sum_is_zero = (sum_abs == 0)
    result_isZero = (a_isZero and b_isZero) or (sum_is_zero and not a_isInf and not b_isInf and not a_isNaN and not b_isNaN)
    underflow = (final_exp <= 0) and not result_isZero and not result_isInf and not result_isNaN

    # Output
    if result_isNaN:
        return (0 << 15) | (EXP_MAX << 10) | 1
    elif result_isZero:
        out_sign = a_sign & b_sign
        return (out_sign << 15) | 0
    elif result_isInf:
        out_sign = a_sign if a_isInf else b_sign
        return (out_sign << 15) | (EXP_MAX << 10) | 0
    elif underflow:
        return (result_sign << 15) | 0
    else:
        return (result_sign << 15) | ((final_exp & 0x1F) << 10) | (final_mant & 0x3FF)


def _frac_mult_fp16(a_fp16: np.float16, t_int: int, t_bits: int = 10) -> np.float16:
    """Bit-exact emulation of frac_mult.v: result = a × (t_int / 2^t_bits)."""
    EXP_WIDTH = 5
    MANT_WIDTH = 10
    BIAS = 15
    EXP_MAX = 31
    FULL_MANT = MANT_WIDTH + 1  # 11
    PROD_WIDTH = t_bits + FULL_MANT  # 21

    a_bits = int.from_bytes(np.array([a_fp16], dtype=np.float16).tobytes(), 'little')
    a_sign = (a_bits >> 15) & 1
    a_exp  = (a_bits >> 10) & 0x1F
    a_mant = a_bits & 0x3FF

    a_isZero   = (a_exp == 0) and (a_mant == 0)
    a_isDenorm = (a_exp == 0) and (a_mant != 0)
    a_isInf    = (a_exp == EXP_MAX) and (a_mant == 0)
    a_isNaN    = (a_exp == EXP_MAX) and (a_mant != 0)

    if a_isZero:
        a_full = 0
    elif a_isDenorm:
        a_full = a_mant  # no hidden bit
    else:
        a_full = (1 << MANT_WIDTH) | a_mant  # hidden bit

    a_eff_exp = 1 if a_exp == 0 else a_exp
    t_isZero = (t_int == 0)

    # Integer multiply
    product_raw = t_int * a_full
    prod_zero = (product_raw == 0)

    # Special cases
    result_isZero = a_isZero or t_isZero or prod_zero
    result_isNaN  = a_isNaN
    result_isInf  = a_isInf

    if result_isZero:
        return np.float16(0.0)
    if result_isNaN:
        return np.float16(float('nan'))
    if result_isInf:
        sign_char = '-' if a_sign else ''
        return np.float16(float(f'{sign_char}inf'))

    # LZC
    lzc = 0
    for i in range(PROD_WIDTH - 1, -1, -1):
        if (product_raw >> i) & 1:
            break
        lzc += 1

    # Normalize
    norm_product = (product_raw << lzc) & ((1 << PROD_WIDTH) - 1)

    # result_exp = eff_exp - lzc
    result_exp_raw = a_eff_exp - lzc

    # Rounding (Round to Nearest Even)
    trunc_mant = (norm_product >> (PROD_WIDTH - 1 - MANT_WIDTH)) & ((1 << MANT_WIDTH) - 1)
    GUARD_POS = PROD_WIDTH - 2 - MANT_WIDTH
    guard = (norm_product >> GUARD_POS) & 1
    round_bit = (norm_product >> (GUARD_POS - 1)) & 1 if GUARD_POS > 0 else 0
    sticky = 1 if (GUARD_POS > 1 and (norm_product & ((1 << (GUARD_POS - 1)) - 1)) != 0) else 0
    lsb = trunc_mant & 1
    round_up = guard & (round_bit | sticky | lsb)

    rounded = trunc_mant + round_up
    round_ovf = (rounded >> MANT_WIDTH) & 1

    if round_ovf:
        final_exp = result_exp_raw + 1
        final_mant = (rounded >> 1) & ((1 << MANT_WIDTH) - 1)
    else:
        final_exp = result_exp_raw
        final_mant = rounded & ((1 << MANT_WIDTH) - 1)

    # Underflow / overflow
    underflow = (final_exp <= 0) and not result_isZero
    overflow  = (final_exp >= EXP_MAX)

    if underflow:
        out_sign = a_sign
        out_exp = 0
        out_mant = 0
    elif overflow:
        out_sign = a_sign
        out_exp = EXP_MAX
        out_mant = 0
    else:
        out_sign = a_sign
        out_exp = final_exp & 0x1F
        out_mant = final_mant & 0x3FF

    out_bits = (out_sign << 15) | (out_exp << 10) | out_mant
    return np.frombuffer(out_bits.to_bytes(2, 'little'), dtype=np.float16)[0]


def hw_eda_forward_scalar(x_bits: int, config_rom: list, func_lut_f32: list) -> float:
    """Simulate the exact hardware pipeline for a single FP16 input (as bits)."""
    sign = (x_bits >> 15) & 1
    exp  = (x_bits >> 10) & 0x1F
    mant = x_bits & 0x3FF

    # NaN passthrough
    if exp == 31 and mant != 0:
        return float('nan')

    # Config ROM lookup
    bin_addr = (sign << 5) | exp
    cfg_entry = config_rom[bin_addr]
    clamp     = (cfg_entry >> 12) & 1
    k_bits    = (cfg_entry >> 9) & 7
    base_off  = cfg_entry & 0x1FF

    # Bit extraction (matching RTL mux with left-justified t_int)
    # RTL: micro_idx = top K bits of mantissa
    #       t_int    = remaining (10-K) bits, left-justified to 10 bits
    if k_bits > 0 and k_bits <= 5:
        micro_idx = (mant >> (10 - k_bits)) & ((1 << k_bits) - 1)
        remaining = mant & ((1 << (10 - k_bits)) - 1)
        t_int = remaining << k_bits  # left-justify to 10 bits
    else:
        micro_idx = 0
        t_int = mant & 0x3FF  # K=0: all 10 mantissa bits

    lut_addr = base_off + micro_idx
    max_addr = len(func_lut_f32) - 2
    if lut_addr > max_addr:
        lut_addr = max_addr

    y0 = func_lut_f32[lut_addr]
    y1 = func_lut_f32[lut_addr + 1]

    if clamp:
        return float(np.float16(y0))

    # ---- RTL bit-exact interpolation ----
    # Stage 2: FP16 subtraction via fp_adder(y1, -y0)
    y0_bits = _fp16_bits(y0)
    y1_bits = _fp16_bits(y1)
    neg_y0_bits = y0_bits ^ 0x8000  # flip sign for subtraction
    diff_bits = _fp_adder(y1_bits, neg_y0_bits)

    # isZero handling (RTL: s2_diff = sub_isZero ? 0 : packed)
    diff_f16 = _bits_to_fp16(diff_bits)
    if diff_f16 == 0:
        diff_bits = 0x0000

    # Stage 3: frac_mult — bit-exact emulation
    product_f16 = _frac_mult_fp16(_bits_to_fp16(diff_bits), t_int, T_BITS)

    # Stage 4: FP16 addition via fp_adder(y0, product)
    product_bits = _fp16_bits(product_f16)
    result_bits = _fp_adder(y0_bits, product_bits)

    # isZero handling (RTL: interp_result = add_isZero ? 0 : packed)
    result_f16 = _bits_to_fp16(result_bits)
    if result_f16 == 0:
        result_bits = 0x0000

    return float(_bits_to_fp16(result_bits))


def generate_mem_files(func_name: str = 'silu', max_lut: int = 254,
                       max_k: int = 5, output_dir: str = '.'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Generating EDA-NLI memory files for: {func_name}")
    config = optimize_eda(func_name, max_lut=max_lut, max_k=max_k,
                          device=device, verbose=True)
    domain = get_domain(func_name)

    # Detect clipped (edge) bins by comparing with original exponent ranges
    original_bins = {}
    for e in range(1, 31):
        original_bins[(0, e)] = (2.0**(e-15), 2.0**(e-14))
        original_bins[(1, e)] = (-(2.0**(e-14)), -(2.0**(e-15)))
    original_bins[(0, 0)] = (2.0**(-24), 2.0**(-14))
    original_bins[(1, 0)] = (-(2.0**(-14)), -(2.0**(-24)))

    bin_map = {}
    clipped_bins = set()
    for i, (b_start, b_end, sign, exp_val) in enumerate(config.bins):
        bin_map[(sign, exp_val)] = i
        orig = original_bins.get((sign, exp_val))
        if orig and (abs(b_start - orig[0]) > 1e-10 or abs(b_end - orig[1]) > 1e-10):
            clipped_bins.add((sign, exp_val))

    # Build config ROM (64 entries)
    first_lut_idx = 0
    last_lut_idx = len(config.lut_values) - 1
    config_rom_entries = []

    with open(os.path.join(output_dir, 'config_rom.mem'), 'w') as f:
        f.write(f"// config_rom for {func_name}\n")
        for addr in range(64):
            sign = (addr >> 5) & 1
            exp_val = addr & 0x1F

            if exp_val == 31:
                base = last_lut_idx if sign == 0 else first_lut_idx
                entry = (1 << 12) | (0 << 9) | (base & 0x1FF)
            elif (sign, exp_val) in bin_map:
                idx = bin_map[(sign, exp_val)]
                k = config.k_alloc[idx]
                base = int(config.base_offsets[idx].item())
                is_clipped = (sign, exp_val) in clipped_bins
                clamp_flag = 1 if is_clipped else 0
                entry = (clamp_flag << 12) | ((k & 0x7) << 9) | (base & 0x1FF)
            else:
                base = last_lut_idx if sign == 0 else first_lut_idx
                entry = (1 << 12) | (0 << 9) | (base & 0x1FF)

            config_rom_entries.append(entry)
            f.write(f"{entry:04X}  // [{addr:2d}] s={sign} e={exp_val:2d}")
            if (sign, exp_val) in bin_map:
                idx = bin_map[(sign, exp_val)]
                tag = " CLAMP(edge)" if (sign, exp_val) in clipped_bins else ""
                f.write(f" k={config.k_alloc[idx]} base={int(config.base_offsets[idx].item())}{tag}")
            else:
                f.write(f" CLAMP(domain)")
            f.write("\n")
    print(f"  config_rom.mem : 64 entries ({len(clipped_bins)} edge-clamped)")

    # Build function LUT
    lut_vals = config.lut_values.cpu().float().tolist()
    with open(os.path.join(output_dir, 'func_lut.mem'), 'w') as f:
        f.write(f"// func_lut for {func_name}: {len(lut_vals)} entries\n")
        for i, val in enumerate(lut_vals):
            f.write(f"{fp32_to_fp16_hex(val)}  // [{i}] = {val:.6f}\n")
    print(f"  func_lut.mem   : {len(lut_vals)} entries")

    # Generate test vectors with hardware-accurate expected values
    func = get_function(func_name)
    lo, hi = domain
    test_inputs_f32 = torch.linspace(lo, hi, 200, device='cpu').tolist()
    test_inputs_f32 += [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]

    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// test vectors for {func_name} (HW-accurate expected)\n")
        count = 0
        for x_f32 in test_inputs_f32:
            x_bits = fp32_to_fp16_bits(x_f32)
            y_hw = hw_eda_forward_scalar(x_bits, config_rom_entries, lut_vals)
            if np.isnan(y_hw):
                continue
            y_bits = fp32_to_fp16_bits(y_hw)
            f.write(f"{x_bits:04X} {y_bits:04X}  // x={x_f32:.4f} y_hw={y_hw:.6f}\n")
            count += 1
    print(f"  test_vectors.mem : {count} test pairs (HW-accurate)")

    # Error summary vs reference
    test_t = torch.tensor(test_inputs_f32)
    y_ref = func(test_t)
    y_hw_all = []
    for x in test_inputs_f32:
        xb = fp32_to_fp16_bits(x)
        yh = hw_eda_forward_scalar(xb, config_rom_entries, lut_vals)
        y_hw_all.append(yh if not np.isnan(yh) else 0.0)
    y_hw_t = torch.tensor(y_hw_all)
    abs_err = torch.abs(y_hw_t - y_ref)
    print(f"\n  HW EDA vs reference: max_abs={abs_err.max().item():.4e} "
          f"mean_abs={abs_err.mean().item():.4e}")


if __name__ == '__main__':
    func_name = sys.argv[1] if len(sys.argv) > 1 else 'silu'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    generate_mem_files(func_name, output_dir=output_dir)
