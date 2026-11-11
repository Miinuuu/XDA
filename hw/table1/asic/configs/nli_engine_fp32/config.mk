export DESIGN_NAME = nli_engine_fp32
export PLATFORM    = nangate45

DESIGN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Pre-synthesized generic netlist (RTL is encrypted; see hw/RTL_ENCRYPTION.md)
export VERILOG_FILES = $(DESIGN_DIR)../../netlist/nli_engine_fp32.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

export CORE_UTILIZATION = 40
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export SYNTH_MOCK_LARGE_MEMORIES = 1
export SYNTH_MEMORY_MAX_BITS = 512
