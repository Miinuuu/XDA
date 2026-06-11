export DESIGN_NAME = eda_nli_engine_4s
export PLATFORM    = nangate45

DESIGN_DIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ASIC_SRC_DIR := $(DESIGN_DIR)../../src
FPGA_SRC_DIR := $(DESIGN_DIR)../../../fpga/src

export VERILOG_FILES = $(ASIC_SRC_DIR)/eda_nli_engine_4s_sram.v \
                       $(ASIC_SRC_DIR)/fakeram45_256x16.v \
                       $(ASIC_SRC_DIR)/fakeram45_64x15.v \
                       $(FPGA_SRC_DIR)/fp_adder.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

export ADDITIONAL_LIBS = $(PLATFORM_DIR)/lib/fakeram45_256x16.lib \
                         $(PLATFORM_DIR)/lib/fakeram45_64x15.lib
export ADDITIONAL_LEFS = $(PLATFORM_DIR)/lef/fakeram45_256x16.lef \
                         $(PLATFORM_DIR)/lef/fakeram45_64x15.lef

export CORE_UTILIZATION = 30
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
export MACRO_PLACE_HALO = 10 10
