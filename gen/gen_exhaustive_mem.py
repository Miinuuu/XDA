#!/usr/bin/env python3
"""Generate exhaustive .mem files for FPGA on-board verification.

For each of 9 functions:
  <func>/config_rom.mem, func_lut.mem, test_vectors.mem

test_vectors.mem uses the FULL FP16 grid (31K-51K vectors per function)
with HW-accurate expected values from bit-exact EDA pipeline simulation.

Usage:
  cd gen
  python gen_exhaustive_mem.py
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sw'))
from gen_eda_mem import (generate_mem_files as _gen_mem_base,
                         fp32_to_fp16_hex, fp32_to_fp16_bits)
from gen_eda_mem_fma import hw_eda_forward_fma as hw_eda_forward_scalar
from nli_eda import optimize_eda, get_function, get_domain, _generate_fp16_grid


def generate_exhaustive(func_name, output_dir):
    """Generate config + exhaustive test vectors for one function."""
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n=== {func_name} ===")

    # 1. Generate config_rom.mem + func_lut.mem (reuse existing generator)
    _gen_mem_base(func_name, max_lut=254, max_k=5, output_dir=output_dir)

    # 2. Read back config_rom entries and lut values for HW simulation
    config_rom_entries = []
    with open(os.path.join(output_dir, 'config_rom.mem')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            hex_val = line.split()[0]
            config_rom_entries.append(int(hex_val, 16))

    lut_vals = []
    with open(os.path.join(output_dir, 'func_lut.mem')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            hex_val = line.split()[0]
            bits = int(hex_val, 16)
            val = np.frombuffer(bits.to_bytes(2, 'little'), dtype=np.float16)[0].item()
            lut_vals.append(val)

    # 3. Exhaustive test vectors: ALL FP16 grid points
    grid = _generate_fp16_grid(get_domain(func_name), 'cpu')
    n_total = len(grid)

    with open(os.path.join(output_dir, 'test_vectors.mem'), 'w') as f:
        f.write(f"// exhaustive test vectors for {func_name} ({n_total} vectors)\n")
        count = 0
        for x_val in grid.numpy():
            x_bits = fp32_to_fp16_bits(x_val)
            y_hw = hw_eda_forward_scalar(x_bits, config_rom_entries, lut_vals)
            if np.isnan(y_hw):
                continue
            y_bits = fp32_to_fp16_bits(y_hw)
            f.write(f"{x_bits:04X} {y_bits:04X}\n")
            count += 1
    print(f"  test_vectors.mem : {count} exhaustive vectors (was ~210)")


if __name__ == '__main__':
    config_dir = os.path.join(os.path.dirname(__file__),
                              '..', 'hw', 'eda_u200', 'eda-nli-kernel', 'config')
    config_dir = os.path.abspath(config_dir)
    funcs = ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh',
             'reciprocal', 'hardswish', 'mish']

    print("=== Generating exhaustive .mem files ===")
    for fn in funcs:
        out = os.path.join(config_dir, fn)
        generate_exhaustive(fn, out)

    print(f"\nDone. Files in {config_dir}/<func>/")
