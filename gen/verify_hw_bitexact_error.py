#!/usr/bin/env python3
"""Compare checked-in EDA HW bit-exact configs against PyTorch references.

The reported error scale matches Table 3: relative error times 1e4, with
denominator clamped by the FP16 normal threshold (2^-14).
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'gen'))
sys.path.insert(0, os.path.join(ROOT, 'sw'))

from gen_eda_mem import (  # noqa: E402
    HW_MODE_FTZ,
    HW_MODE_IEEE_SUBNORMAL,
    fp32_to_fp16_bits,
)
from gen_eda_mem_fma import hw_eda_forward_fma  # noqa: E402
from nli_eda import (  # noqa: E402
    ALL_FUNCS,
    EVAL_DOMAINS,
    TAU,
    _generate_fp16_grid,
    get_function,
)


DEFAULT_CONFIG_BASE = os.path.join(
    ROOT, 'hw', 'eda_u200', 'eda-nli-kernel', 'config')
DEFAULT_TABLE3_JSON = os.path.join(
    ROOT, 'sw', 'eda_results', 'all_experiments.json')
INPUT_THRESHOLD = 2.0 ** -14


def _fp16_hex_to_float(hex_val: str) -> float:
    bits = int(hex_val, 16)
    return np.frombuffer(bits.to_bytes(2, 'little'), dtype=np.float16)[0].item()


def load_mem_config(config_base: str, func_name: str) -> tuple[list[int], list[float]]:
    func_dir = os.path.join(config_base, func_name)
    config_rom = []
    with open(os.path.join(func_dir, 'config_rom.mem')) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):
                config_rom.append(int(line.split()[0], 16))

    func_lut = []
    with open(os.path.join(func_dir, 'func_lut.mem')) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):
                func_lut.append(_fp16_hex_to_float(line.split()[0]))

    if len(config_rom) != 64:
        raise ValueError(f'{func_name}: expected 64 config entries, got {len(config_rom)}')
    if len(func_lut) < 2:
        raise ValueError(f'{func_name}: expected at least 2 LUT entries, got {len(func_lut)}')
    return config_rom, func_lut


def make_eval_grid(func_name: str, input_threshold: float | None) -> torch.Tensor:
    grid = _generate_fp16_grid(EVAL_DOMAINS[func_name], 'cpu')
    if input_threshold is None:
        return grid
    if func_name in ('rsqrt', 'reciprocal'):
        return grid[grid >= input_threshold]
    return grid[grid.abs() >= input_threshold]


def eval_fma_error(func_name: str, config_base: str, hw_mode: str,
                   input_threshold: float | None) -> dict:
    config_rom, func_lut = load_mem_config(config_base, func_name)
    grid = make_eval_grid(func_name, input_threshold)
    func = get_function(func_name)
    y_ref = func(grid).float()

    y_hw = []
    for x_val in grid.numpy():
        y = hw_eda_forward_fma(
            fp32_to_fp16_bits(x_val), config_rom, func_lut, hw_mode=hw_mode)
        y_hw.append(y if not math.isnan(y) else float('nan'))
    y_hw = torch.tensor(y_hw, dtype=torch.float32)

    valid = torch.isfinite(y_ref) & torch.isfinite(y_hw)
    grid = grid[valid]
    y_ref = y_ref[valid]
    y_hw = y_hw[valid]
    rel = torch.abs(y_hw - y_ref) / torch.clamp(torch.abs(y_ref), min=TAU)
    worst_idx = int(rel.argmax().item())
    return {
        'n': int(rel.numel()),
        'mean': rel.mean().item() * 1e4,
        'max': rel.max().item() * 1e4,
        'worst_x': float(grid[worst_idx].item()),
        'worst_ref': float(y_ref[worst_idx].item()),
        'worst_hw': float(y_hw[worst_idx].item()),
    }


def load_table3(table3_json: str) -> dict:
    with open(table3_json) as f:
        return json.load(f)['part_a']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify HW bit-exact EDA FMA error against PyTorch references.')
    parser.add_argument('--config-base', default=DEFAULT_CONFIG_BASE,
                        help='Directory containing config/<func>/*.mem files')
    parser.add_argument('--table3-json', default=DEFAULT_TABLE3_JSON,
                        help='all_experiments.json used for Table 3 EDA columns')
    parser.add_argument('--func', action='append', choices=ALL_FUNCS,
                        help='Function to evaluate; repeatable. Defaults to all functions.')
    parser.add_argument('--no-input-threshold', action='store_true',
                        help='Do not restrict inputs to |x| >= 2^-14 (or x >= 2^-14).')
    parser.add_argument('--include-ftz', action='store_true',
                        help='Also report the FTZ sensitivity path.')
    args = parser.parse_args()

    funcs = args.func or list(ALL_FUNCS)
    input_threshold = None if args.no_input_threshold else INPUT_THRESHOLD
    table3 = load_table3(args.table3_json)

    headers = [
        'func', 'n', 'table3_mean', 'table3_max',
        'fma_sub_mean', 'fma_sub_max', 'worst_x', 'worst_ref', 'worst_hw',
    ]
    if args.include_ftz:
        headers.extend(['fma_ftz_mean', 'fma_ftz_max'])
    print(','.join(headers))

    for func_name in funcs:
        sub = eval_fma_error(
            func_name, args.config_base, HW_MODE_IEEE_SUBNORMAL, input_threshold)
        row = [
            func_name,
            str(sub['n']),
            f"{table3[func_name]['eda_mean']:.3f}",
            f"{table3[func_name]['eda_max']:.3f}",
            f"{sub['mean']:.3f}",
            f"{sub['max']:.3f}",
            f"{sub['worst_x']:.8g}",
            f"{sub['worst_ref']:.8g}",
            f"{sub['worst_hw']:.8g}",
        ]
        if args.include_ftz:
            ftz = eval_fma_error(
                func_name, args.config_base, HW_MODE_FTZ, input_threshold)
            row.extend([f"{ftz['mean']:.3f}", f"{ftz['max']:.3f}"])
        print(','.join(row))


if __name__ == '__main__':
    main()
