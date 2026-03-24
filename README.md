# EDA-NLI: Exponent-Direct Addressing with Fused Shift-Add Interpolation

Hardware implementation of comparator-free nonlinear function approximation for FP16 LLM inference acceleration on Xilinx Alveo U200.

## Repository Structure

```
eda_u200/eda-nli-kernel/
├── src/
│   ├── IP/                            # RTL source
│   │   ├── eda_nli_engine_4s.vp       # 4-stage EDA-NLI pipeline (IEEE P1735 encrypted)
│   │   ├── eda_nli_compute.sv         # PISO → NLI engine → SIPO datapath + FSM
│   │   ├── EDA_NLI_wrapper.sv         # AXI4 master/slave top-level wrapper
│   │   ├── EDA.v                      # Vitis kernel top
│   │   ├── EDA_control_s_axi.v        # AXI-Lite control registers
│   │   ├── EDA_axi_read_master.sv     # AXI4 burst read master (512-bit)
│   │   ├── EDA_axi_write_master.sv    # AXI4 burst write master (512-bit)
│   │   ├── EDA_counter.sv             # Transaction counter
│   │   └── fp_adder.v                 # Parameterized floating-point adder
│   ├── host/                          # C++ host application
│   │   ├── eda_nli_host.cpp           # XRT host — LUT config + sigmoid verification
│   │   └── user-host.cpp              # Host template
│   ├── testbench/                     # Simulation & verification
│   │   ├── tb_eda_nli_kernel.sv       # AXI VIP kernel testbench
│   │   ├── tb_axi_vip_kernel.sv       # AXI protocol compliance test
│   │   ├── tb_top_simple.sv           # Simplified top-level test
│   │   ├── tb_reset_verify.sv         # Reset sequence test
│   │   ├── create_project.tcl         # Vivado project setup
│   │   ├── create_axi_vip_tb.tcl      # AXI VIP testbench generator
│   │   └── run_sim.tcl                # Simulation runner (VCD export)
│   ├── c-model/
│   │   └── EDA.cpp                    # C reference model
│   └── xml/
│       └── user.xml                   # Vitis kernel interface definition
├── scripts/
│   ├── package_kernel.tcl             # IP packaging for Vitis
│   └── gen_xo.tcl                     # XO generation from RTL
├── config/
│   ├── sigmoid_config_rom.h           # Config ROM (32 entries, exponent → LUT address)
│   ├── sigmoid_func_lut.h             # Function LUT (256 entries, FP16)
│   └── test_vectors.mem               # FP16 test vectors (207 vectors)
└── Makefile
```

## Architecture

### EDA-NLI Engine (`eda_nli_engine_4s.vp`)

4-stage pipelined FP16 nonlinear function approximation unit. The core engine RTL is provided as an IEEE P1735 encrypted file. Simulation, synthesis, and bitstream generation are all permitted.

| Stage | Operation |
|-------|-----------|
| 1 | Bit extraction + Config ROM lookup → LUT address + interpolation fraction |
| 2 | Dual LUT read + FP16 subtract (diff = y₁ − y₀) |
| 3 | FMA Part A — integer multiply + extend + align + add/sub |
| 4 | FMA Part B — LZC + normalize + round + output MUX |

### System Data Flow

```
DDR ──→ AXI4 Read Master ──→ 512-bit ──→ PISO (32×FP16)
                                              │
                                     EDA-NLI Engine (×32)
                                              │
                              SIPO (32×FP16) ──→ 512-bit ──→ AXI4 Write Master ──→ DDR
```

- **Data width**: 512-bit AXI4, processing 32 FP16 elements per beat
- **Clock**: 150 MHz kernel frequency
- **Pipeline latency**: 4 cycles per element
- **Configuration**: 32-entry Config ROM + 256-entry Function LUT, loadable at runtime via AXI-Lite

### AXI-Lite Register Map (`EDA_control_s_axi`)

| Offset | Register | Description |
|--------|----------|-------------|
| 0x00 | Control | ap_start / ap_done / ap_idle / ap_ready |
| 0x10 | User Status | user_start / user_done / user_idle / user_ready |
| 0x14 | scalar00 | Transfer size in bytes (default: 16384) |
| 0x1C–0x20 | A[63:0] | DDR read source address |
| 0x28–0x2C | B[63:0] | DDR write destination address |
| 0x40 | cfg_ctrl | cfg_sel[0] (0=Config ROM, 1=Func LUT), cfg_addr[9:1] |
| 0x44 | cfg_wdata | LUT write data (FP16) |

### Kernel Interface (Vitis)

| Port | Direction | Width | Function |
|------|-----------|-------|----------|
| s_axi_control | Slave | 32-bit | AXI-Lite control/config registers |
| m00_axi | Master | 512-bit | DDR burst read |
| m01_axi | Master | 512-bit | DDR burst write |

## Requirements

| Tool | Version |
|------|---------|
| Xilinx Vivado | 2024.1 |
| Xilinx Vitis | 2024.1 |
| XRT (Xilinx Runtime) | 2.16+ |
| Platform | xilinx_u200_gen3x16_xdma_2_202110_1 |
| Hardware | Xilinx Alveo U200 (for on-board execution) |

## Build & Run

All commands are executed from `eda_u200/eda-nli-kernel/`.

```bash
cd eda_u200/eda-nli-kernel
```

### Vivado Simulation (AXI VIP testbench)

```bash
make create_proj          # Create Vivado project + generate AXI VIP IPs
make sim                  # Run simulation (exports VCD waveform)
make sim_all              # Create project + simulate in one step
```

### Hardware Emulation (hw_emu)

```bash
make all TARGET=hw_emu    # Build: host + XO + XCLBIN + emconfig
make run TARGET=hw_emu    # Build + run emulation (207 test vectors)
```

### Hardware Build (hw)

```bash
make all TARGET=hw        # Build for Alveo U200
make run TARGET=hw        # Run on FPGA hardware
```

### Individual Build Steps

```bash
make exe                  # Compile host application only
make xclbin TARGET=hw_emu # Build XCLBIN only
make emconfig             # Generate emulation config
```

### Clean

```bash
make clean                # Remove build artifacts (host, xclbin, xo)
make clean_proj           # Remove Vivado simulation project
make cleanall             # Remove everything
```

## Verification

The host application (`eda_nli_host.cpp`) runs 207 FP16 test vectors through the kernel and compares output against expected values using ULP (Unit in the Last Place) distance. Pass criterion: all results within 4 ULP.

### Expected Output (hw_emu)

```
=== EDA-NLI Sigmoid Kernel ===
Test vectors: 207, padded: 224 (448 bytes)
...
--- Summary ---
Test vectors: 207
Bit-exact matches: 207/207
Max ULP distance: 0
*** TEST PASSED (all within 4 ULP) ***
```

## Encrypted RTL Notice

The core EDA-NLI engine (`eda_nli_engine_4s.vp`) is provided as an IEEE P1735 encrypted Verilog file during the submission review period. The encrypted source fully supports simulation, synthesis, and bitstream generation in Xilinx Vivado. The complete unencrypted RTL source will be disclosed after the review process is concluded.

## License

This code is provided for academic review purposes only.
