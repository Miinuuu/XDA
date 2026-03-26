#!/usr/bin/env python3
"""Generate exhaustive .mem + .h files for FPGA on-board verification.

For each of 9 functions:
  <func>/config_rom.mem, func_lut.mem, test_vectors.mem  (for RTL sim)
  <func>_config_rom.h, <func>_func_lut.h                 (for host C++)

test_vectors.mem uses the FULL FP16 grid (31K-51K vectors per function)
with HW-accurate expected values from bit-exact EDA pipeline simulation.

Usage:
  cd /home/jmw/ing/eda_submission/hw/eda_u200/eda-nli-kernel/config
  conda run -n vllm python gen_exhaustive_mem.py
"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, '/home/jmw/ing/research/eda')
sys.path.insert(0, '/home/jmw/ing/research/eda/HW/eda_nli')
from gen_eda_mem import (generate_mem_files as _gen_mem_base,
                         hw_eda_forward_scalar,
                         fp32_to_fp16_hex, fp32_to_fp16_bits)
from nli_eda import optimize_eda
from nli_dp import get_function, get_domain, generate_fp16_grid


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
    grid = generate_fp16_grid(get_domain(func_name))
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


def generate_h_files(func_name, config_dir):
    """Generate C header files from .mem files."""
    upper = func_name.upper()

    # config_rom.h
    entries = []
    with open(os.path.join(config_dir, func_name, 'config_rom.mem')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            entries.append(int(line.split()[0], 16))

    with open(os.path.join(config_dir, f'{func_name}_config_rom.h'), 'w') as f:
        f.write(f"#ifndef {upper}_CONFIG_ROM_H\n#define {upper}_CONFIG_ROM_H\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"static const uint16_t {func_name}_config_rom[] = {{\n")
        for e in entries:
            f.write(f"    0x{e:04X},\n")
        f.write(f"}};\n")
        f.write(f"static const int {upper}_CONFIG_ROM_SIZE = "
                f"sizeof({func_name}_config_rom)/sizeof({func_name}_config_rom[0]);\n")
        f.write(f"#endif\n")

    # func_lut.h
    entries = []
    with open(os.path.join(config_dir, func_name, 'func_lut.mem')) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            entries.append(int(line.split()[0], 16))

    with open(os.path.join(config_dir, f'{func_name}_func_lut.h'), 'w') as f:
        f.write(f"#ifndef {upper}_FUNC_LUT_H\n#define {upper}_FUNC_LUT_H\n")
        f.write("#include <cstdint>\n\n")
        f.write(f"static const uint16_t {func_name}_func_lut[] = {{\n")
        for e in entries:
            f.write(f"    0x{e:04X},\n")
        f.write(f"}};\n")
        f.write(f"static const int {upper}_FUNC_LUT_SIZE = "
                f"sizeof({func_name}_func_lut)/sizeof({func_name}_func_lut[0]);\n")
        f.write(f"#endif\n")

    print(f"  {func_name}_config_rom.h, {func_name}_func_lut.h updated")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    funcs = ['silu', 'exp', 'rsqrt', 'gelu', 'sigmoid', 'tanh',
             'reciprocal', 'hardswish', 'mish']

    print("=== Generating exhaustive .mem + .h files ===")
    for fn in funcs:
        out = os.path.join(script_dir, fn)
        generate_exhaustive(fn, out)
        generate_h_files(fn, script_dir)

    print(f"\nDone. Files in {script_dir}/<func>/")
