# EDA-NLI: Exponent-Direct Addressing Nonlinear Interpolation

A nonlinear function approximation engine based on zero-FP-arithmetic address generation using FP16 bit-field extraction.

Supported functions: `sigmoid`, `tanh`, `silu`, `mish`, `gelu`, `hardswish`, `exp`, `reciprocal`, `rsqrt`

## Directory Structure

```
├── sw/                          # Python software
│   ├── nli_eda.py               # EDA-NLI core (optimize, evaluate)
│   ├── nli_eda_engine.py        # EDA engine implementation
│   ├── nli_engine.py            # Baseline NLI engine
│   ├── nli_dp.py                # DP-based cutpoint optimization
│   ├── ablation_sweep.py        # Ablation study
│   ├── nli_table1_eval.py       # Table 1 reproduction
│   └── run_all_experiments.py   # Run all experiments
│
├── gen/                         # Memory file generators
│   ├── gen_nli_mem.py           # NLI engine .mem generation
│   ├── gen_eda_mem.py           # EDA engine .mem generation
│   ├── gen_eda_mem_fma.py       # EDA 4-stage FMA .mem generation
│   └── gen_exhaustive_mem.py    # Exhaustive test vector generation
│
└── hw/                          # Hardware RTL
    ├── nli/                     # NLI engine RTL + testbench
    ├── fpu/                     # FP adder IP
    └── eda_u200/                # Xilinx U200 Vitis kernel
        └── eda-nli-kernel/
            ├── src/IP/          # Kernel RTL (encrypted .vp)
            ├── src/host/        # Host application (C++)
            ├── src/nli_engine/  # Standalone engine RTL + testbench
            ├── src/c-model/     # C reference model
            ├── config/          # LUT config & test vectors (.mem)
            └── Makefile
```

## Requirements

- **SW**: Python 3, PyTorch, NumPy
- **NLI RTL Simulation**: Icarus Verilog (`iverilog`), GTKWave (optional)
- **EDA RTL Simulation**: Vivado (xvlog/xelab/xsim)
- **FPGA Build & Run**: Vitis 2021.1+, Xilinx U200 platform (`xilinx_u200_gen3x16_xdma_2_202110_1`)

## Workflow

The overall flow follows three stages: **optimize** (sw) → **generate .mem** (gen) → **simulate or run on FPGA** (hw).

### Step 1: SW — Run Optimization & Experiments

The `sw/` scripts find optimal LUT configurations for each nonlinear function. This step is independent of hardware.

```bash
cd sw

# Run all experiments (reproduce Tables 1–5)
python3 run_all_experiments.py

# Or run individually
python3 nli_eda.py           # EDA-NLI optimization + evaluation
python3 ablation_sweep.py    # Ablation study
python3 nli_table1_eval.py   # Table 1 reproduction
```

### Step 2: Generate .mem Files

The `gen/` scripts read optimization results from `sw/` and produce `.mem` files (config ROM, function LUT, test vectors) that hardware consumes.

#### For NLI Engine (baseline)

```bash
# Generates: point_reg.mem, mul_reg.mem, lut_reg.mem, test_vectors.mem
python3 gen/gen_nli_mem.py silu hw/nli/
```

#### For EDA Engine (4-stage FMA)

```bash
# Single function — generates: config_rom.mem, func_lut.mem, test_vectors_4s.mem
python3 gen/gen_eda_mem_fma.py --func silu --output-dir hw/eda_u200/eda-nli-kernel/src/nli_engine/

# All 9 functions — exhaustive FP16 test vectors into config/<func>/
cd gen
python3 gen_exhaustive_mem.py
```

### Step 3a: RTL Simulation — NLI Engine (Icarus Verilog)

Simulates the baseline NLI engine. Makefile handles mem generation + simulation automatically.

```bash
cd hw/nli

make all              # Generate .mem + simulate (default: silu)
make all FUNC=gelu    # Specify function
make wave             # View waveform (GTKWave)
```

### Step 3b: RTL Simulation — EDA Engine (Vivado xsim)

Simulates the 4-stage FMA EDA engine with Xilinx simulator.

```bash
cd hw/eda_u200/eda-nli-kernel/src/nli_engine

make all              # Generate .mem + simulate (default: silu)
make all FUNC=gelu    # Specify function
make sim-all          # Test all 9 functions
make sim-exhaustive   # Exhaustive FP16 verification (all 9 functions)
```

### Step 3c: FPGA — Vitis U200 Kernel Build & Run

Builds the kernel xclbin and runs on FPGA (or HW emulation). The host application loads `config/<func>/*.mem` files at runtime via AXI-Lite.

```bash
cd hw/eda_u200/eda-nli-kernel

# --- HW Emulation ---
make all TARGET=hw_emu
make run TARGET=hw_emu FUNC=sigmoid       # Single function
make run_all TARGET=hw_emu                # All 9 functions

# --- HW (actual FPGA) ---
make all TARGET=hw
make run TARGET=hw FUNC=sigmoid
make run_all TARGET=hw
```

## Pipeline Summary

```
sw/nli_eda.py          gen/gen_eda_mem_fma.py       hw/eda_u200/.../src/nli_engine/
(optimize LUT)  ──→   (produce .mem files)   ──→   (RTL simulation, xsim)
                            │
                            │  gen/gen_exhaustive_mem.py
                            ▼
                       hw/eda_u200/.../config/<func>/*.mem
                            │
                            ▼
                       hw/eda_u200/.../Makefile
                       (Vitis build → xclbin → FPGA run)
```

## Note

The core EDA RTL (`eda_nli_engine_4s.vp`) is currently provided in encrypted form. Simulation, synthesis, and bitstream generation work as expected. The unencrypted source code will be released upon paper acceptance.
