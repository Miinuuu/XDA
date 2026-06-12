#!/usr/bin/env bash
# Reproduces the shipped pwr_<design>_uniform.log files (rebuttal Sec. 1).
#
# Prerequisite: build the three ASIC designs post-route with the flow under
# hw/table1/asic/ in an OpenROAD-flow-scripts checkout ($ORFS), so that
#   $ORFS/flow/results/nangate45/<design>/base/6_final.{odb,spef,sdc}
# exist for each design.
#
# Usage:
#   export ORFS=/path/to/OpenROAD-flow-scripts
#   ./run_power.sh
#
# Outputs are written as repro_pwr_<design>_uniform.log so they can be diffed
# against the shipped pwr_<design>_uniform.log evidence files.
set -eu
: "${ORFS:?export ORFS=<OpenROAD-flow-scripts checkout>}"
export ORFS
cd "$(dirname "$0")"
for d in nli_engine eda_nli_engine_4s nn_lut_engine_16; do
  PWR_DESIGN="$d" PWR_MODE=uniform \
  "$ORFS/tools/install/OpenROAD/bin/openroad" -no_init -exit report_power_v2.tcl \
    | tee "repro_pwr_${d}_uniform.log"
done
