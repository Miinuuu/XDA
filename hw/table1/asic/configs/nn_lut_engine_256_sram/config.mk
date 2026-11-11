export DESIGN_NAME = nn_lut_engine_256
export PLATFORM    = nangate45

DESIGN_DIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Pre-synthesized generic netlist (RTL is encrypted; see hw/RTL_ENCRYPTION.md)
export VERILOG_FILES = $(DESIGN_DIR)../../netlist/nn_lut_engine_256_sram.v \
                       $(DESIGN_DIR)../../src/fakeram45_256x16.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_256x16.lib
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x16.lef

export CORE_UTILIZATION = 30
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export MACRO_PLACE_HALO = 10 10
