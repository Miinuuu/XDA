#!/usr/bin/env python3
"""
Generate .mem files for nli_engine_fp32 Verilog testbench (FP32 version).

Produces hex-encoded FP32 values (8 hex digits each) for:
  - point_reg.mem  : 11 macro cutpoints (FP32)
  - mul_reg.mem    : 10 scale factors (FP32)
  - lut_reg.mem    : 259 function value LUT entries (FP32)
  - test_vectors.mem : input/expected_output pairs (FP32)
"""

import sys, os, struct
import numpy as np

from nli_engine import build_lut_from_paper, nli_forward
from nli_dp import PAPER_CUTPOINTS, get_function, get_domain
import torch


def float_to_fp32_hex(val: float) -> str:
    """Convert float to IEEE 754 FP32 hex string (8 hex digits)."""
    bits = struct.unpack('I', struct.pack('f', float(val)))[0]
    return f"{bits:08X}"


def generate_mem_files(func_name: str = 'silu', output_dir: str = '.'):
    """Generate all .mem files for nli_engine_fp32 for a given function."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating NLI FP32 memory files for: {func_name}")

    # Build LUT (all values in float32 — no FP16 quantization)
    point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)

    # --- point_reg.mem (11 entries, FP32) ---
    with open(os.path.join(output_dir, 'point_reg.mem'), 'w') as f:
        f.write(f"// point_reg for {func_name} (11 macro cutpoints, FP32 hex)\n")
        for i, val in enumerate(point_reg.tolist()):
            f.write(f"{float_to_fp32_hex(val)}  // [{i}] = {val}\n")
    print(f"  point_reg.mem : {len(point_reg)} entries")

    # --- mul_reg.mem (10 entries, FP32) ---
    with open(os.path.join(output_dir, 'mul_reg.mem'), 'w') as f:
        f.write(f"// mul_reg for {func_name} (10 scale factors, FP32 hex)\n")
        for i, val in enumerate(mul_reg.tolist()):
            f.write(f"{float_to_fp32_hex(val)}  // [{i}] = {val}\n")
    print(f"  mul_reg.mem   : {len(mul_reg)} entries")

    # --- lut_reg.mem (259 entries, FP32) ---
    with open(os.path.join(output_dir, 'lut_reg.mem'), 'w') as f:
        f.write(f"// lut_reg for {func_name} (259 function values, FP32 hex)\n")
        for i, val in enumerate(lut_reg.tolist()):
            f.write(f"{float_to_fp32_hex(val)}  // [{i}] = {val:.8f}\n")
    print(f"  lut_reg.mem   : {len(lut_reg)} entries")

    # --- test_vectors.mem (FP32 input/output pairs) ---
    domain = get_domain(func_name)
    func = get_function(func_name)

    lo, hi = domain
    lo_fp16 = torch.tensor(lo, dtype=torch.float16)
    hi_fp16 = torch.tensor(hi, dtype=torch.float16)
    lo_next = torch.nextafter(lo_fp16, torch.tensor(float('inf'), dtype=torch.float16)).float()
    hi_prev = torch.nextafter(hi_fp16, torch.tensor(float('-inf'), dtype=torch.float16)).float()
    test_inputs = torch.linspace(lo, hi_prev.item(), 200)
    extras = torch.tensor([lo - 1.0, lo, lo_next.item(), hi_prev.item()])
    test_inputs = torch.cat([test_inputs, extras])

    # NLI forward in FP32 (no FP16 quantization)
    y_nli = nli_forward(test_inputs, point_reg, mul_reg, lut_reg, variant='full_fp32')

    # Store as pairs: each line has two 8-digit hex words (input, expected)
    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// Test vectors for {func_name} FP32: input_fp32  expected_output_fp32\n")
        f.write(f"// {len(test_inputs)} test cases\n")
        for i in range(len(test_inputs)):
            x_hex = float_to_fp32_hex(test_inputs[i].item())
            y_hex = float_to_fp32_hex(y_nli[i].item())
            f.write(f"{x_hex} {y_hex}  // x={test_inputs[i].item():.6f} y={y_nli[i].item():.8f}\n")
    print(f"  test_vectors.mem : {len(test_inputs)} test pairs")

    y_ref = func(test_inputs)
    abs_err = torch.abs(y_nli - y_ref)
    print(f"\n  NLI FP32 vs reference: max_abs_err={abs_err.max().item():.4e}, "
          f"mean_abs_err={abs_err.mean().item():.4e}")


if __name__ == '__main__':
    func_name = sys.argv[1] if len(sys.argv) > 1 else 'silu'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    generate_mem_files(func_name, output_dir)
