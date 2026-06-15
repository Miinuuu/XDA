#!/bin/bash
# DSP-enabled FPGA synthesis for ICCAD'26 rebuttal (Reviewer B, Q1).
# Identical flow to run_one.sh step [1/4] except `-max_dsp 0` is removed,
# so Vivado may infer DSP48 slices. Same part, same clock.xdc, same
# -flatten_hierarchy rebuilt. Reports util + timing only (no SAIF power).
set -e
# Prerequisite: source <Vivado>/settings64.sh first
command -v vivado >/dev/null || { echo "ERROR: vivado not in PATH"; exit 1; }

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC=$RUN_DIR/src
OUT=$RUN_DIR/out
mkdir -p "$OUT"

run_design() {
    local NAME=$1 TOP=$2 RTL=$3 GENERICS=$4
    local D=$OUT/$NAME
    mkdir -p "$D"
    cat > "$D/synth.tcl" << TCLEOF
create_project -force -part xcu200-fsgd2104-2-e proj $D/proj
add_files {$RTL}
add_files -fileset constrs_1 $SRC/clock.xdc
set_property top $TOP [current_fileset]
$([ -n "$GENERICS" ] && echo "set_property generic {$GENERICS} [current_fileset]")
synth_design -top $TOP -flatten_hierarchy rebuilt
opt_design
place_design
route_design
report_utilization -file $D/util.rpt
report_timing_summary -max_paths 1 -file $D/timing.rpt
close_project
TCLEOF
    echo "[$(date +%H:%M:%S)] start $NAME"
    vivado -mode batch -source "$D/synth.tcl" -log "$D/vivado.log" \
        -journal "$D/vivado.jou" > /dev/null 2>&1
    echo "[$(date +%H:%M:%S)] done  $NAME"
}

run_design eda_dsp     eda_nli_engine_4s "$SRC/eda_nli_engine_4s.v $SRC/fp_adder.v" "GRADUAL_UNDERFLOW=0"
run_design nli_dsp     nli_engine        "$SRC/nli_engine.v $SRC/fp_mult_norm.v $SRC/fp_adder.v" ""
run_design nnlut16_dsp nn_lut_engine     "$SRC/nn_lut_engine.v $SRC/fp_mult_norm.v $SRC/fp_adder.v" ""

echo ALL_DONE
