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
set -euo pipefail
: "${ORFS:?export ORFS=<OpenROAD-flow-scripts checkout>}"
export ORFS
cd "$(dirname "$0")"
for d in nli_engine eda_nli_engine_4s nn_lut_engine_16; do
  log="repro_pwr_${d}_uniform.log"
  PWR_DESIGN="$d" PWR_MODE=uniform \
  "$ORFS/tools/install/OpenROAD/bin/openroad" -no_init -exit report_power_v2.tcl \
    | tee "$log"
  # report_power prints its banner before running; require the actual table.
  if ! grep -q "POWER REPORT: ${d}" "$log" || ! grep -q '^Total ' "$log"; then
    echo "ERROR: ${d}: no complete power table in ${log} (failed OpenROAD run?)" >&2
    exit 1
  fi
done
echo "OK: all three designs produced complete power tables."
