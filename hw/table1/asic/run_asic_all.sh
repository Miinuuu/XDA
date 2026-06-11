#!/bin/bash
# ASIC P&R — Nangate45 with fakeram SRAM macros
# NLI, XDA: 2× fakeram45_256x16 per design
# NN-LUT-16: all FF (no SRAM)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFS_DIR="${ORFS_DIR:?Set ORFS_DIR to your OpenROAD-flow-scripts checkout}"
FLOW_DIR="$ORFS_DIR/flow"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"

DESIGNS="nli_engine nli_engine_addr_fp32 eda_nli_engine_4s nn_lut_engine_16"

source "$ORFS_DIR/env.sh"

mkdir -p "$RESULTS_DIR"

run_design() {
    local design=$1
    local config="$CONFIGS_DIR/$design/config.mk"
    local log_dir="$FLOW_DIR/logs/nangate45/$design/base"
    local res_dir="$FLOW_DIR/results/nangate45/$design/base"

    echo "=========================================="
    echo "  Running: $design"
    echo "  Config:  $config"
    echo "  Time:    $(date)"
    echo "=========================================="

    cd "$FLOW_DIR"
    make DESIGN_CONFIG="$config" clean_all 2>&1 | tail -3 || true

    if make DESIGN_CONFIG="$config" 2>&1 | tee "$RESULTS_DIR/${design}_build.log"; then
        echo "  BUILD: SUCCESS"
    else
        echo "  BUILD: FAILED"
    fi

    local out="$RESULTS_DIR/$design"
    mkdir -p "$out"
    for f in 1_1_yosys_canonicalize.log 1_2_yosys.log 1_synth.log 5_1_grt.log 6_report.log 6_report.json; do
        cp -f "$log_dir/$f" "$out/" 2>/dev/null || true
    done
    cp -f "$res_dir/6_final.v" "$out/" 2>/dev/null || echo "  WARN: no 6_final.v"

    if [ -f "$out/6_report.json" ]; then
        echo ""
        echo "--- Results: $design ---"
        python3 -c "
import json
d=json.load(open('$out/6_report.json'))
area=d.get('finish__design__instance__area',0)
area_macro=d.get('finish__design__instance__area__macros',0)
area_std=d.get('finish__design__instance__area__stdcell',0)
fmax=d.get('finish__timing__fmax',0)/1e6
power=d.get('finish__power__total',0)*1000
print(f'  Total Area:  {area:,.0f} um2')
print(f'  Macro Area:  {area_macro:,.0f} um2')
print(f'  StdCell Area:{area_std:,.0f} um2')
print(f'  Fmax:        {fmax:.1f} MHz')
print(f'  Power:       {power:.2f} mW')
" 2>/dev/null
        echo ""
    fi

    echo "  Finished: $design at $(date)"
    echo ""
}

echo "ASIC P&R — Nangate45 + fakeram SRAM"
echo "Date: $(date)"
echo "Designs: $DESIGNS"
echo ""

for design in $DESIGNS; do
    run_design "$design"
done

echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
for design in $DESIGNS; do
    out="$RESULTS_DIR/$design"
    echo "--- $design ---"
    if [ -f "$out/6_report.json" ]; then
        python3 -c "
import json
d=json.load(open('$out/6_report.json'))
area=d.get('finish__design__instance__area',0)
area_macro=d.get('finish__design__instance__area__macros',0)
area_std=d.get('finish__design__instance__area__stdcell',0)
fmax=d.get('finish__timing__fmax',0)/1e6
power=d.get('finish__power__total',0)*1000
print(f'  Total: {area:,.0f}  Macro: {area_macro:,.0f}  Logic: {area_std:,.0f}  Fmax: {fmax:.1f}MHz  Power: {power:.2f}mW')
" 2>/dev/null
    else
        echo "  FAILED"
    fi
done | tee "$RESULTS_DIR/nangate45_sram_summary.txt"

echo ""
echo "Done at $(date)"
