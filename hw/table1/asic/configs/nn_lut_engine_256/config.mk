export DESIGN_NAME = nn_lut_engine_256
export PLATFORM    = nangate45

DESIGN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SRC_DIR := $(DESIGN_DIR)../../../fpga/src

export VERILOG_FILES = $(SRC_DIR)/nn_lut_engine.v \
                       $(SRC_DIR)/fp_mult_norm.v \
                       $(SRC_DIR)/fp_adder.v \
                       $(SRC_DIR)/nn_lut_engine_256.v
export SDC_FILE      = $(DESIGN_DIR)../../constraint.sdc
export ABC_AREA      = 1

export CORE_UTILIZATION = 40
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT = 100
