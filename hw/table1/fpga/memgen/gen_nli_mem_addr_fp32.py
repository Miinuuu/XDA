#!/usr/bin/env python3
"""
Generate .mem files for nli_engine_addr_fp32 Verilog testbench.

Mixed-precision layout:
  - point_reg.mem: FP32 cutpoints
  - mul_reg.mem: FP32 scale factors
  - lut_reg.mem: FP16 LUT entries
  - test_vectors.mem: FP16 input / FP16 expected output pairs
"""

import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'sw'))

from nli_dp import get_domain, get_function
from nli_engine import build_lut_from_paper, nli_forward


def float_to_fp32_hex(val: float) -> str:
    bits = struct.unpack('I', struct.pack('f', float(val)))[0]
    return f"{bits:08X}"


def fp32_to_fp16_hex(val: float) -> str:
    fp16 = np.float16(val)
    bits = int.from_bytes(np.array([fp16], dtype=np.float16).tobytes(), 'little')
    return f"{bits:04X}"


def generate_mem_files(func_name: str = 'silu', output_dir: str = '.'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating NLI addr-FP32 memory files for: {func_name}")
    point_reg, mul_reg, lut_reg = build_lut_from_paper(func_name)

    with open(os.path.join(output_dir, 'point_reg.mem'), 'w') as f:
        f.write(f"// point_reg for {func_name} (11 macro cutpoints, FP32 hex)\n")
        for i, val in enumerate(point_reg.tolist()):
            f.write(f"{float_to_fp32_hex(val)}  // [{i}] = {val}\n")
    print(f"  point_reg.mem : {len(point_reg)} entries")

    with open(os.path.join(output_dir, 'mul_reg.mem'), 'w') as f:
        f.write(f"// mul_reg for {func_name} (10 scale factors, FP32 hex)\n")
        for i, val in enumerate(mul_reg.tolist()):
            f.write(f"{float_to_fp32_hex(val)}  // [{i}] = {val}\n")
    print(f"  mul_reg.mem   : {len(mul_reg)} entries")

    with open(os.path.join(output_dir, 'lut_reg.mem'), 'w') as f:
        f.write(f"// lut_reg for {func_name} (259 function values, FP16 hex)\n")
        for i, val in enumerate(lut_reg.tolist()):
            f.write(f"{fp32_to_fp16_hex(val)}  // [{i}] = {val:.6f}\n")
    print(f"  lut_reg.mem   : {len(lut_reg)} entries")

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

    y_nli = nli_forward(test_inputs, point_reg, mul_reg, lut_reg, variant='addr_fp32')

    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// Test vectors for {func_name}: input_fp16 expected_output_fp16\n")
        f.write(f"// {len(test_inputs)} test cases\n")
        for i in range(len(test_inputs)):
            x_hex = fp32_to_fp16_hex(test_inputs[i].item())
            y_hex = fp32_to_fp16_hex(y_nli[i].item())
            f.write(f"{x_hex} {y_hex}  // x={test_inputs[i].item():.4f} y={y_nli[i].item():.6f}\n")
    print(f"  test_vectors.mem : {len(test_inputs)} test pairs")

    y_ref = func(test_inputs)
    abs_err = torch.abs(y_nli - y_ref)
    print(
        f"\n  NLI addr-FP32 vs reference: max_abs_err={abs_err.max().item():.4e}, "
        f"mean_abs_err={abs_err.mean().item():.4e}"
    )


if __name__ == '__main__':
    func_name = sys.argv[1] if len(sys.argv) > 1 else 'silu'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    generate_mem_files(func_name, output_dir)
