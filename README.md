# XDA: Representation-Aware Exponent-Direct Addressing for Efficient Shared Nonlinear Units

Hardware-software co-design for efficient nonlinear function approximation via **zero-FP-arithmetic address generation** using FP16 bit-field extraction.

## Key Idea

Traditional nonlinear interpolation (NLI) requires comparators and multipliers to locate intervals and compute addresses. XDA eliminates this overhead by directly extracting the FP16 exponent and mantissa bits to generate LUT addresses -- **no comparators, no multipliers, just wiring**.

```
FP16 input:  [sign(1) | exponent(5) | mantissa(10)]
                 │           │              │
                 │           │         ┌────┴────┐
                 │           │     top-K bits  remaining bits
                 │           │         │           │
                 ▼           ▼         ▼           ▼
            ┌─────────────────────┐  ┌───────────────┐
            │  Config ROM lookup  │  │ Interpolation  │
            │  (bin select)       │  │ fraction t     │
            └────────┬────────────┘  └───────┬───────┘
                     │                       │
                     ▼                       ▼
              LUT[base + micro_idx]    y = y0 + t*(y1-y0)
```

**Supported functions (9):** `sigmoid`, `tanh`, `silu`, `mish`, `gelu`, `hardswish`, `exp`, `reciprocal`, `rsqrt`

## Method Overview

| Component | Description |
|-----------|-------------|
| **Exponent-Direct Addressing** | FP16 exponent bits select the bin; top-K mantissa bits select the micro-bin within it. Remaining bits become the interpolation fraction. |
| **Knapsack DP Allocation** | Given a LUT budget W, allocates mantissa bits per bin to minimize total approximation error (solves a 0/1 knapsack). |
| **Boundary Sharing** | Adjacent exponent bins share LUT endpoints, reducing LUT size by ~10-20% without accuracy loss. |
| **4-Stage FMA Pipeline** | Stage 1: bit-extract & address gen. Stage 2: dual SRAM read & FP16 subtract. Stage 3: FMA multiply & align. Stage 4: normalize & round (RNE). |

## Directory Structure

```
├── run_pipeline.sh              # Full pipeline automation script
├── sw/                          # Python software
│   ├── nli_eda.py               # XDA core: knapsack DP optimization & evaluation
│   ├── nli_eda_engine.py        # PyTorch forward-pass simulator (3/4-stage)
│   ├── ablation_sweep.py        # Ablation study (budget, bit-width, strategy)
│   └── run_all_experiments.py   # Run all paper experiments
│
├── gen/                         # Memory file generators
│   ├── gen_eda_mem.py           # Base .mem generator with bit-exact simulation
│   ├── gen_eda_mem_fma.py       # FMA-variant .mem generator (4-stage)
│   └── gen_exhaustive_mem.py    # Exhaustive FP16 test vector generation
│
└── hw/                          # Hardware RTL
    ├── fpu/
    │   └── fp_adder.v           # Parameterized FP16/FP32 adder
    └── eda_u200/eda-nli-kernel/
        ├── src/IP/              # Kernel RTL (wrappers)
        ├── src/nli_engine/      # Standalone engine RTL + testbench
        ├── src/host/            # Host application (C++)
        ├── src/c-model/         # C reference model
        ├── config/              # LUT config & test vectors (.mem)
        └── Makefile
```

## Requirements

- **SW**: Python 3, PyTorch, NumPy
- **RTL Simulation**: Vivado (xvlog/xelab/xsim)
- **FPGA Build & Run**: Vitis 2021.1+, Xilinx U200 platform (`xilinx_u200_gen3x16_xdma_2_202110_1`)

## Quick Start

```bash
# Full pipeline (SW -> generate .mem -> RTL sim -> Vitis build -> FPGA run)
./run_pipeline.sh                          # default: hw_emu, all 9 functions
./run_pipeline.sh --target hw              # real FPGA
./run_pipeline.sh --func sigmoid           # single function only
./run_pipeline.sh --skip-sw                # skip SW experiments

# Run individual steps
./run_pipeline.sh --step sw                # SW experiments only
./run_pipeline.sh --step gen               # generate .mem only
./run_pipeline.sh --step sim               # .mem generation + RTL simulation
./run_pipeline.sh --step build             # .mem generation + Vitis build
./run_pipeline.sh --step run               # FPGA run only (build must exist)
```

## Workflow Details

The overall flow follows three stages: **optimize** (sw) -> **generate .mem** (gen) -> **simulate or run on FPGA** (hw).

### Step 1: SW -- Optimization & Experiments

Find optimal LUT configurations for each nonlinear function via knapsack DP.

```bash
cd sw

python3 run_all_experiments.py       # Run all experiments
python3 nli_eda.py                   # XDA optimization + evaluation
python3 ablation_sweep.py            # Ablation study
```

### Step 2: Generate .mem Files

Produce config ROM, function LUT, and test vectors for hardware.

```bash
# Single function
python3 gen/gen_eda_mem_fma.py --func silu --output-dir hw/eda_u200/eda-nli-kernel/src/nli_engine/

# All 9 functions -- exhaustive FP16 test vectors
cd gen && python3 gen_exhaustive_mem.py
```

**Generated files per function:**

| File | Description |
|------|-------------|
| `config_rom.mem` | 64-entry ROM: `[clamp(1) \| reserved(3) \| k_bits(3) \| base_offset(9)]` per exponent/sign |
| `func_lut.mem` | LUT values in FP16 hex (~250-300 entries) |
| `test_vectors.mem` | `<x_hex> <y_hex>` pairs for verification |

### Step 3a: RTL Simulation (Vivado xsim)

```bash
cd hw/eda_u200/eda-nli-kernel/src/nli_engine

make all              # Generate .mem + simulate (default: silu)
make all FUNC=gelu    # Specify function
make sim-all          # Test all 9 functions
make sim-exhaustive   # Exhaustive FP16 verification
```

### Step 3b: FPGA Build & Run (Vitis U200)

```bash
cd hw/eda_u200/eda-nli-kernel

# HW Emulation
make all TARGET=hw_emu
make run TARGET=hw_emu FUNC=sigmoid
make run_all TARGET=hw_emu

# Real FPGA
make all TARGET=hw
make run TARGET=hw FUNC=sigmoid
make run_all TARGET=hw
```

## Hardware Architecture

32 parallel XDA engines on a 512-bit AXI datapath (32 FP16 elements/cycle):

```
DDR ──[AXI Read Master]──> PISO (512b→32×16b)
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              XDA Engine  XDA Engine ... XDA Engine  (×32)
                    │          │          │
                    └──────────┼──────────┘
                               │
                           SIPO (32×16b→512b) ──[AXI Write Master]──> DDR
```

**4-Stage Pipeline (per engine):**

| Stage | Operation | FP Gates |
|-------|-----------|----------|
| S1 | Bit-extract, config ROM lookup, address gen | **0** |
| S2 | Dual SRAM read, FP16 subtract (y1-y0) | 1 subtractor |
| S3 | Integer multiply, exponent align | 0 (integer) |
| S4 | Normalize, round-to-nearest-even | 1 rounder |

## Pipeline Summary

```
sw/nli_eda.py          gen/gen_eda_mem_fma.py       hw/.../src/nli_engine/
(knapsack DP     -->  (produce .mem files)   -->   (RTL simulation, xsim)
 optimize LUT)              |
                            |  gen/gen_exhaustive_mem.py
                            v
                       hw/.../config/<func>/*.mem
                            |
                            v
                       hw/.../Makefile
                       (Vitis build -> xclbin -> FPGA run)
```

