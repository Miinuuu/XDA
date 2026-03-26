# EDA-NLI: Exponent-Direct Addressing Nonlinear Interpolation

A nonlinear function approximation engine based on zero-FP-arithmetic address generation using FP16 bit-field extraction.

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
└── hw/                          # Hardware RTL
    ├── nli/                     # NLI engine RTL + testbench
    │   ├── nli_engine.v
    │   ├── gen_nli_mem.py       # .mem file generation
    │   ├── tb_nli_engine.sv
    │   └── Makefile
    ├── fpu/                     # FP adder IP
    └── eda_u200/                # Xilinx U200 Vitis kernel
        └── eda-nli-kernel/
            ├── src/IP/          # Kernel RTL (encrypted .vp)
            ├── src/host/        # Host application (C++)
            ├── src/c-model/     # C reference model
            ├── config/          # LUT config & test vectors
            └── Makefile
```

## How to Run

### SW: Python Experiments

```bash
cd sw

# Run all experiments (reproduce Tables 1–5)
python3 run_all_experiments.py

# Run individually
python3 nli_eda.py           # EDA-NLI standalone
python3 ablation_sweep.py    # Ablation study
python3 nli_table1_eval.py   # Table 1
```

Requirements: Python 3, PyTorch, NumPy

### HW: RTL Simulation (Icarus Verilog)

```bash
cd hw/nli

# Generate .mem files + run simulation (default: silu)
make all

# Specify function
make all FUNC=gelu

# View waveform
make wave
```

Requirements: Icarus Verilog (`iverilog`), Python 3, GTKWave (for waveform viewing)

### HW: Vitis U200 Kernel

```bash
cd hw/eda_u200/eda-nli-kernel

# --- HW Emulation ---
make all TARGET=hw_emu
make run TARGET=hw_emu FUNC=sigmoid       # Single function
make run_all TARGET=hw_emu                # All 9 functions

# --- HW (actual FPGA build & run) ---
make all TARGET=hw
make run TARGET=hw FUNC=sigmoid
make run_all TARGET=hw
```

Supported functions: `sigmoid`, `tanh`, `silu`, `mish`, `gelu`, `hardswish`, `exp`, `reciprocal`, `rsqrt`

Requirements: Vitis 2021.1+, Xilinx U200 platform (`xilinx_u200_gen3x16_xdma_2_202110_1`)

## Note

The core EDA RTL (`eda_nli_engine_4s.vp`) is currently provided in encrypted form. Simulation, synthesis, and bitstream generation work as expected. The unencrypted source code will be released upon paper acceptance.
