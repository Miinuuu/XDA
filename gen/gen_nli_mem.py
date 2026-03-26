#!/usr/bin/env python3
"""
Generate .mem files for NLI engine Verilog testbench.

Produces hex-encoded FP16 values for:
  - point_reg.mem  : 11 macro cutpoints
  - mul_reg.mem    : 10 scale factors
  - lut_reg.mem    : 259 function value LUT entries
  - test_vectors.mem : input/expected_output pairs for verification
"""

import sys, os, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sw'))
from nli_engine import build_lut_from_paper, nli_forward
from nli_dp import PAPER_CUTPOINTS, get_function, get_domain
import torch


def fp32_to_fp16_hex(val: float) -> str:
    """Convert float to FP16 hex string (4 hex digits)."""
    fp16 = np.float16(val)
    bits = int.from_bytes(np.array([fp16], dtype=np.float16).tobytes(), 'little')
    return f"{bits:04X}"


def generate_mem_files(func_name: str = 'silu', output_dir: str = '.'):
    """Generate all .mem files for a given function."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating NLI memory files for: {func_name}")

    # Build LUT
    point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)

    # --- point_reg.mem (11 entries) ---
    with open(os.path.join(output_dir, 'point_reg.mem'), 'w') as f:
        f.write(f"// point_reg for {func_name} (11 macro cutpoints, FP16 hex)\n")
        for i, val in enumerate(point_reg.tolist()):
            f.write(f"{fp32_to_fp16_hex(val)}  // [{i}] = {val}\n")
    print(f"  point_reg.mem : {len(point_reg)} entries")

    # --- mul_reg.mem (10 entries) ---
    with open(os.path.join(output_dir, 'mul_reg.mem'), 'w') as f:
        f.write(f"// mul_reg for {func_name} (10 scale factors, FP16 hex)\n")
        for i, val in enumerate(mul_reg.tolist()):
            f.write(f"{fp32_to_fp16_hex(val)}  // [{i}] = {val}\n")
    print(f"  mul_reg.mem   : {len(mul_reg)} entries")

    # --- lut_reg.mem (259 entries) ---
    with open(os.path.join(output_dir, 'lut_reg.mem'), 'w') as f:
        f.write(f"// lut_reg for {func_name} (259 function values, FP16 hex)\n")
        for i, val in enumerate(lut_reg.tolist()):
            f.write(f"{fp32_to_fp16_hex(val)}  // [{i}] = {val:.6f}\n")
    print(f"  lut_reg.mem   : {len(lut_reg)} entries")

    # --- test_vectors.mem (input, expected_output pairs) ---
    domain = get_domain(func_name)
    func = get_function(func_name)

    # Generate test inputs spanning the domain
    lo, hi = domain
    test_inputs = torch.linspace(lo, hi, 200)
    # Add boundary test cases
    extras = torch.tensor([lo - 1.0, lo, lo + 0.001, hi - 0.001, hi, hi + 1.0])
    test_inputs = torch.cat([test_inputs, extras])

    # Compute NLI expected outputs
    y_nli = nli_forward(test_inputs, point_reg, mul_reg, lut_reg)

    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// Test vectors for {func_name}: input_fp16  expected_output_fp16\n")
        f.write(f"// {len(test_inputs)} test cases\n")
        for i in range(len(test_inputs)):
            x_hex = fp32_to_fp16_hex(test_inputs[i].item())
            y_hex = fp32_to_fp16_hex(y_nli[i].item())
            f.write(f"{x_hex} {y_hex}  // x={test_inputs[i].item():.4f} y={y_nli[i].item():.6f}\n")
    print(f"  test_vectors.mem : {len(test_inputs)} test pairs")

    # Print summary
    y_ref = func(test_inputs)
    abs_err = torch.abs(y_nli - y_ref)
    print(f"\n  NLI vs reference: max_abs_err={abs_err.max().item():.4e}, "
          f"mean_abs_err={abs_err.mean().item():.4e}")


if __name__ == '__main__':
    func_name = sys.argv[1] if len(sys.argv) > 1 else 'silu'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    generate_mem_files(func_name, output_dir)
