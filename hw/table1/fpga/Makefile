# ==============================================================================
# SAIF-based FPGA Power Analysis — Makefile
#
# Usage:
#   make all          — run all 6 designs (sequential)
#   make nli          — run NLI only
#   make nli_addr_fp32 — run mixed-precision NLI only
#   make nli_full_fp32 — run full-FP32 NLI only
#   make eda          — run EDA only
#   make nnlut16      — run NN-LUT-16 only
#   make nnlut256     — run NN-LUT-256 only
#   make fp32         — alias of nli_full_fp32
#   make mem          — generate .mem files only
#   make clean        — remove all output
#
# Prerequisites: source Vivado settings64.sh, conda activate vllm
# ==============================================================================

SHELL     := /bin/bash
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_DIR    := $(SCRIPT_DIR)src
MEMGEN_DIR := $(SCRIPT_DIR)memgen
OUTDIR     := $(SCRIPT_DIR)out
PART       := xcu200-fsgd2104-2-e
CLOCK_XDC  := $(SRC_DIR)/clock.xdc
GLBL       := $(XILINX_VIVADO)/data/verilog/src/glbl.v

DESIGNS := nli nli_addr_fp32 nli_full_fp32 eda nnlut16 nnlut256

.PHONY: all $(DESIGNS) mem clean check-vivado

all: $(DESIGNS)

# --- Check Vivado ---
check-vivado:
	@if [ ! -f "$(GLBL)" ]; then \
		echo "ERROR: XILINX_VIVADO not set. Source Vivado settings64.sh first."; \
		exit 1; \
	fi

# --- .mem generation ---
mem:
	@echo "=== Generating .mem files ==="
	mkdir -p $(OUTDIR)/nli_mem && cd $(OUTDIR)/nli_mem && \
		python3 $(MEMGEN_DIR)/gen_nli_mem.py silu .
	mkdir -p $(OUTDIR)/nli_addr_fp32_mem && cd $(OUTDIR)/nli_addr_fp32_mem && \
		python3 $(MEMGEN_DIR)/gen_nli_mem_addr_fp32.py silu .
	mkdir -p $(OUTDIR)/nli_full_fp32_mem && cd $(OUTDIR)/nli_full_fp32_mem && \
		python3 $(MEMGEN_DIR)/gen_nli_mem_fp32.py silu .
	mkdir -p $(OUTDIR)/eda_mem && cd $(OUTDIR)/eda_mem && \
		python3 $(MEMGEN_DIR)/gen_eda_mem.py silu . --hw-mode ftz
	mkdir -p $(OUTDIR)/nnlut16_mem && cd $(OUTDIR)/nnlut16_mem && \
		python3 $(MEMGEN_DIR)/gen_nn_lut_mem.py --func silu --outdir .
	mkdir -p $(OUTDIR)/nnlut256_mem
	python3 $(MEMGEN_DIR)/gen_nn_lut_mem.py --func silu --segments 256 \
		--outdir $(OUTDIR)/nnlut256_mem

# ==============================================================================
# Per-design targets
# Each: synth → funcsim → GL sim → SAIF cleanup → power
# ==============================================================================

# --- NLI ---
nli: check-vivado mem
	@echo "=== NLI: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh nli nli_engine \
		"$(SRC_DIR)/nli_engine.v $(SRC_DIR)/fp_mult_norm.v $(SRC_DIR)/fp_adder.v" \
		"" \
		$(SCRIPT_DIR)tb_nli_power.sv tb_nli_engine \
		tb_nli_engine/u_dut \
		$(OUTDIR)/nli_mem \
		nli_engine.saif

# --- NLI-addr-FP32 ---
nli_addr_fp32: check-vivado mem
	@echo "=== NLI-addr-FP32: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh nli_addr_fp32 nli_engine_addr_fp32 \
		"$(SRC_DIR)/nli_engine_addr_fp32.v $(SRC_DIR)/fp32_to_fp16.v $(SRC_DIR)/fp_mult_norm.v $(SRC_DIR)/fp_adder.v" \
		"" \
		$(SCRIPT_DIR)tb_nli_addr_fp32_power.sv tb_nli_engine_addr_fp32 \
		tb_nli_engine_addr_fp32/u_dut \
		$(OUTDIR)/nli_addr_fp32_mem \
		nli_addr_fp32_engine.saif

# --- NN-LUT-16 ---
nnlut16: check-vivado mem
	@echo "=== NN-LUT-16: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh nnlut16 nn_lut_engine \
		"$(SRC_DIR)/nn_lut_engine.v $(SRC_DIR)/fp_mult_norm.v $(SRC_DIR)/fp_adder.v" \
		"" \
		$(SCRIPT_DIR)tb_nnlut_power.sv tb_nn_lut_engine \
		tb_nn_lut_engine/u_dut \
		$(OUTDIR)/nnlut16_mem \
		nn_lut_engine.saif

# --- NN-LUT-256 ---
nnlut256: check-vivado mem
	@echo "=== NN-LUT-256: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh nnlut256 nn_lut_engine \
		"$(SRC_DIR)/nn_lut_engine.v $(SRC_DIR)/fp_mult_norm.v $(SRC_DIR)/fp_adder.v" \
		"N_SEGMENTS=256 N_BP=255 LUT_DEPTH=256" \
		$(SCRIPT_DIR)tb_nnlut256_power.sv tb_nn_lut_engine \
		tb_nn_lut_engine/u_dut \
		$(OUTDIR)/nnlut256_mem \
		nnlut256.saif

# --- NLI-full-FP32 ---
nli_full_fp32: check-vivado mem
	@echo "=== NLI-full-FP32: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh nli_full_fp32 nli_engine_fp32 \
		"$(SRC_DIR)/nli_engine_fp32.v $(SRC_DIR)/fp_mult_norm.v $(SRC_DIR)/fp_adder.v" \
		"" \
		$(SCRIPT_DIR)tb_fp32_power.sv tb_nli_engine_fp32 \
		tb_nli_engine_fp32/u_dut \
		$(OUTDIR)/nli_full_fp32_mem \
		nli_fp32_engine.saif

# --- Alias ---
fp32: nli_full_fp32

# --- EDA ---
eda: check-vivado mem
	@echo "=== EDA: synth + impl ==="
	$(SCRIPT_DIR)run_one.sh eda eda_nli_engine_4s \
		"$(SRC_DIR)/eda_nli_engine_4s.v $(SRC_DIR)/fp_adder.v" \
		"GRADUAL_UNDERFLOW=0" \
		$(SCRIPT_DIR)tb_eda_power.sv tb_eda_nli_engine \
		tb_eda_nli_engine/u_dut \
		$(OUTDIR)/eda_mem \
		eda_nli_engine_4s.saif

# --- Clean ---
clean:
	rm -rf $(OUTDIR)
