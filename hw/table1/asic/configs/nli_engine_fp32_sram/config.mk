export DESIGN_NAME = nli_engine_fp32
export PLATFORM    = nangate45

DESIGN_DIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ASIC_SRC_DIR := $(DESIGN_DIR)../../src
FPGA_SRC_DIR := $(DESIGN_DIR)../../../fpga/src

export VERILOG_FILES = $(ASIC_SRC_DIR)/nli_engine_fp32_sram.v \
                       $(ASIC_SRC_DIR)/fakeram45_256x16.v \
                       $(FPGA_SRC_DIR)/fp_mult_norm.v \
                       $(FPGA_SRC_DIR)/fp_adder.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_256x16.lib
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x16.lef

export CORE_UTILIZATION = 55
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export MACRO_PLACE_HALO = 10 10
