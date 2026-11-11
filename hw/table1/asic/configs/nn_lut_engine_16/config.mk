export DESIGN_NAME = nn_lut_engine_16
export PLATFORM    = nangate45

DESIGN_DIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Pre-synthesized generic netlist (RTL is encrypted; see hw/RTL_ENCRYPTION.md)
export VERILOG_FILES = $(DESIGN_DIR)../../netlist/nn_lut_engine_16.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

# NN-LUT-16: all memories are FF (bp_flat=240b, si_lut=512b)
# No SRAM macros needed

export CORE_UTILIZATION = 60
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
