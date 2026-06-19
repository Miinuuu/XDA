#!/usr/bin/env bash
# Measured switching-based post-route ASIC power (rebuttal §1).
# Flow: gate-level simulation (Icarus Verilog) of the post-route 6_final.v netlist with
# the same FP16 domain-sweep stimulus as the FPGA SAIF run -> VCD -> OpenROAD report_power
# with per-net activity from read_vcd, iso-frequency 10 ns. Produces pwr_<design>_vcd.log.
#
# Prereq: ORFS = OpenROAD-flow-scripts checkout with the designs built post-route
#   ($ORFS/flow/results/nangate45/<design>/base/6_final.{odb,spef,sdc}).
# Usage: export ORFS=/path/to/OpenROAD-flow-scripts ; ./run_vcd_power.sh
set -euo pipefail
: "${ORFS:?export ORFS=<OpenROAD-flow-scripts checkout>}"
OR="$ORFS/tools/install/OpenROAD/bin/openroad"
ROOT="$(cd "$(dirname "$0")" && pwd)"
TCL="$ROOT/report_power_vcd.tcl"

run_one() {  # subdir tb netlist vcd design scope
  local dir="$ROOT/$1"
  ( cd "$dir"
    echo ">>> $5: compile + gate-level sim"
    iverilog -g2012 -o sim.vvp "$2" "$3" nangate_cells.v fakeram45_256x16.v fakeram45_64x15.v
    vvp -n sim.vvp | grep -E 'XCHECK|VCD:' || true
    PWR_DESIGN="$5" PWR_VCD="$dir/$4" PWR_SCOPE="$6" "$OR" -no_init -exit "$TCL" \
      | tee "$ROOT/pwr_$5_vcd.log" \
      | grep -E 'VCD POWER|^Total|^Macro|^Sequential|^Combinational|^Clock|Annotated'
  )
}

run_one "."   tb_eda_vcd.v eda_6_final.v eda.vcd eda_nli_engine_4s tb_eda_nli_engine/u_dut
run_one "nli" tb_nli_vcd.v nli_6_final.v nli.vcd nli_engine        tb_nli_engine/u_dut
echo "OK: pwr_eda_nli_engine_4s_vcd.log + pwr_nli_engine_vcd.log in $ROOT"
