"""Bit-exact FP16 hardware emulation: fp_adder.v, fp_mult_norm.v, fma_interp.v.

Shared by all testbench generators (NN-LUT, NLI, EDA).
"""
import numpy as np


def fp16_bits(val) -> int:
    return int.from_bytes(np.array([np.float16(val)], dtype=np.float16).tobytes(), 'little')


def bits_to_fp16(bits: int) -> np.float16:
    return np.frombuffer((bits & 0xFFFF).to_bytes(2, 'little'), dtype=np.float16)[0]


def fp16_ge(a: int, b: int) -> bool:
    a_s = (a >> 15) & 1; b_s = (b >> 15) & 1
    a_mag = a & 0x7FFF; b_mag = b & 0x7FFF
    if a_s == 0 and b_s == 0: return a_mag >= b_mag
    elif a_s == 1 and b_s == 1: return a_mag <= b_mag
    elif a_s == 0 and b_s == 1: return True
    else: return (a_mag == 0 and b_mag == 0)


def fp_adder(a_bits: int, b_bits: int) -> int:
    """Bit-exact emulation of fp_adder.v for FP16."""
    EXP_WIDTH = 5; MANT_WIDTH = 10; EXP_MAX = 31
    FULL_MANT = 11; GUARD_BITS = 3; WORK_WIDTH = 15

    a_sign = (a_bits >> 15) & 1; a_exp = (a_bits >> 10) & 0x1F; a_mant = a_bits & 0x3FF
    b_sign = (b_bits >> 15) & 1; b_exp = (b_bits >> 10) & 0x1F; b_mant = b_bits & 0x3FF

    a_isZero = (a_exp == 0) and (a_mant == 0); a_isInf = (a_exp == EXP_MAX) and (a_mant == 0)
    a_isNaN = (a_exp == EXP_MAX) and (a_mant != 0); a_isDenorm = (a_exp == 0) and (a_mant != 0)
    b_isZero = (b_exp == 0) and (b_mant == 0); b_isInf = (b_exp == EXP_MAX) and (b_mant == 0)
    b_isNaN = (b_exp == EXP_MAX) and (b_mant != 0); b_isDenorm = (b_exp == 0) and (b_mant != 0)

    a_full = 0 if a_isZero else (a_mant if a_isDenorm else (1 << MANT_WIDTH) | a_mant)
    b_full = 0 if b_isZero else (b_mant if b_isDenorm else (1 << MANT_WIDTH) | b_mant)
    a_eff = 1 if a_exp == 0 else a_exp; b_eff = 1 if b_exp == 0 else b_exp

    swap = not (a_eff > b_eff) and (not (a_eff == b_eff) or not (a_full >= b_full))
    if swap:
        x_sign, y_sign, x_exp, x_mant, y_mant = b_sign, a_sign, b_eff, b_full, a_full
        x_isZero, y_isZero = b_isZero, a_isZero
    else:
        x_sign, y_sign, x_exp, x_mant, y_mant = a_sign, b_sign, a_eff, a_full, b_full
        x_isZero, y_isZero = a_isZero, b_isZero

    shift_amt = x_exp - (a_eff if swap else b_eff)
    MASK = (1 << WORK_WIDTH) - 1
    x_ext = (x_mant << GUARD_BITS) & MASK; y_ext = (y_mant << GUARD_BITS) & MASK

    if shift_amt >= WORK_WIDTH: y_shifted = 0; sticky_bits = y_ext
    else:
        y_shifted = (y_ext >> shift_amt) & MASK
        sticky_bits = y_ext & ((1 << shift_amt) - 1) if shift_amt > 0 else 0
    sticky = 1 if sticky_bits != 0 else 0
    y_aligned = 0 if y_isZero else ((y_shifted & ~1) | (y_shifted & 1) | sticky) & MASK

    eff_sub = x_sign ^ y_sign; MASK_W1 = (1 << (WORK_WIDTH + 1)) - 1
    s = ((x_ext - y_aligned) if eff_sub else (x_ext + y_aligned)) & MASK_W1
    sum_negative = (s >> WORK_WIDTH) & 1
    if sum_negative: sum_abs = (~s + 1) & MASK; result_sign = 1 - x_sign
    else: sum_abs = s & MASK; result_sign = x_sign

    lzc = 0
    for i in range(WORK_WIDTH - 1, -1, -1):
        if (sum_abs >> i) & 1: break
        lzc += 1

    add_overflow = (sum_abs >> (WORK_WIDTH - 1)) & 1
    if add_overflow: norm_mant = (sum_abs >> 1) & MASK; norm_exp = x_exp + 1
    else: norm_mant = (sum_abs << max(lzc - 1, 0)) & MASK; norm_exp = x_exp - lzc + 1

    guard_val = norm_mant & ((1 << GUARD_BITS) - 1); lsb = (norm_mant >> GUARD_BITS) & 1
    round_up = ((guard_val >> (GUARD_BITS - 1)) & 1) and ((guard_val & ((1 << (GUARD_BITS - 1)) - 1)) != 0 or lsb)
    rounded = ((norm_mant >> GUARD_BITS) & ((1 << FULL_MANT) - 1)) + (1 if round_up else 0)
    ro = (rounded >> FULL_MANT) & 1
    final_exp = norm_exp + 1 if ro else norm_exp
    final_mant = (rounded >> 1 if ro else rounded) & ((1 << MANT_WIDTH) - 1)

    result_isNaN = a_isNaN or b_isNaN or (a_isInf and b_isInf and (a_sign != b_sign))
    result_isInf = ((a_isInf or b_isInf) and not result_isNaN) or (final_exp >= EXP_MAX)
    result_isZero = (a_isZero and b_isZero) or ((sum_abs == 0) and not a_isInf and not b_isInf and not a_isNaN and not b_isNaN)
    underflow = (final_exp <= 0) and not result_isZero and not result_isInf and not result_isNaN

    if result_isNaN: return (0 << 15) | (EXP_MAX << 10) | 1
    elif result_isZero: return ((a_sign & b_sign) << 15)
    elif result_isInf: return ((a_sign if a_isInf else b_sign) << 15) | (EXP_MAX << 10)
    elif underflow: return (result_sign << 15)
    else: return (result_sign << 15) | ((final_exp & 0x1F) << 10) | (final_mant & 0x3FF)


def fp_mult_norm(a_bits: int, b_bits: int) -> int:
    """Bit-exact emulation of fp_mult_norm.v for FP16."""
    MANT_WIDTH = 10; BIAS = 15; EXP_MAX = 31
    FULL_MANT = 11; PROD_WIDTH = 22

    a_sign = (a_bits >> 15) & 1; a_exp = (a_bits >> 10) & 0x1F; a_mant = a_bits & 0x3FF
    b_sign = (b_bits >> 15) & 1; b_exp = (b_bits >> 10) & 0x1F; b_mant = b_bits & 0x3FF

    a_isZero = (a_exp == 0) and (a_mant == 0); a_isInf = (a_exp == EXP_MAX) and (a_mant == 0)
    a_isNaN = (a_exp == EXP_MAX) and (a_mant != 0); a_isDenorm = (a_exp == 0) and (a_mant != 0)
    b_isZero = (b_exp == 0) and (b_mant == 0); b_isInf = (b_exp == EXP_MAX) and (b_mant == 0)
    b_isNaN = (b_exp == EXP_MAX) and (b_mant != 0); b_isDenorm = (b_exp == 0) and (b_mant != 0)

    result_sign = a_sign ^ b_sign
    a_full = 0 if a_isZero else (a_mant if a_isDenorm else (1 << MANT_WIDTH) | a_mant)
    b_full = 0 if b_isZero else (b_mant if b_isDenorm else (1 << MANT_WIDTH) | b_mant)

    product = a_full * b_full
    a_eff = 1 if a_exp == 0 else a_exp; b_eff = 1 if b_exp == 0 else b_exp
    exp_sum = a_eff + b_eff - BIAS

    prod_zero = (product == 0)
    lzc = 0
    if not prod_zero:
        for i in range(PROD_WIDTH - 1, -1, -1):
            if (product >> i) & 1: break
            lzc += 1

    if prod_zero: norm_product = 0; norm_exp = 0
    else:
        norm_product = (product << lzc) & ((1 << PROD_WIDTH) - 1)
        norm_exp = exp_sum + 1 - lzc

    MANT_END = PROD_WIDTH - 2 - MANT_WIDTH + 1  # 11
    GUARD_POS = MANT_END - 1  # 10
    ROUND_POS = MANT_END - 2  # 9

    trunc_mant = (norm_product >> MANT_END) & ((1 << MANT_WIDTH) - 1)
    guard = (norm_product >> GUARD_POS) & 1 if GUARD_POS >= 0 else 0
    round_bit = (norm_product >> ROUND_POS) & 1 if ROUND_POS >= 0 else 0
    sticky = 1 if (ROUND_POS > 0 and (norm_product & ((1 << ROUND_POS) - 1)) != 0) else 0
    round_up = guard and (round_bit or sticky or (trunc_mant & 1))

    rounded = trunc_mant + (1 if round_up else 0)
    ro = (rounded >> MANT_WIDTH) & 1
    final_exp = norm_exp + 1 if ro else norm_exp
    final_mant = (rounded >> 1 if ro else rounded) & ((1 << MANT_WIDTH) - 1)

    result_isNaN = a_isNaN or b_isNaN or ((a_isZero and b_isInf) or (a_isInf and b_isZero))
    result_isInf = ((a_isInf or b_isInf) and not result_isNaN) or (final_exp >= EXP_MAX and not result_isNaN)
    result_isZero = (a_isZero or b_isZero) and not result_isNaN
    underflow = (final_exp <= 0) and not result_isZero and not result_isInf and not result_isNaN

    if result_isNaN: return (0 << 15) | (EXP_MAX << 10) | 1
    elif result_isZero or underflow: return 0
    elif result_isInf: return (result_sign << 15) | (EXP_MAX << 10)
    else: return (result_sign << 15) | ((final_exp & 0x1F) << 10) | (final_mant & 0x3FF)
