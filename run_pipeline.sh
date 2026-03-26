#!/bin/bash
# ==============================================================
# EDA-NLI Full Pipeline: optimize → generate .mem → HW run
#
# Usage:
#   ./run_pipeline.sh                     # All steps, hw_emu
#   ./run_pipeline.sh --target hw         # All steps, real FPGA
#   ./run_pipeline.sh --step gen          # Only generate .mem
#   ./run_pipeline.sh --step sim          # Only RTL simulation
#   ./run_pipeline.sh --func sigmoid      # Single function
#   ./run_pipeline.sh --skip-sw           # Skip SW experiments
# ==============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SW_DIR="$ROOT_DIR/sw"
GEN_DIR="$ROOT_DIR/gen"
NLI_DIR="$ROOT_DIR/hw/nli"
EDA_DIR="$ROOT_DIR/hw/eda_u200/eda-nli-kernel"
CONFIG_DIR="$EDA_DIR/config"
ENGINE_DIR="$EDA_DIR/src/nli_engine"

# Defaults
TARGET="hw_emu"
STEP="all"
FUNC=""
SKIP_SW=false

ALL_FUNCS="sigmoid tanh silu mish gelu hardswish exp reciprocal rsqrt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)  TARGET="$2"; shift 2 ;;
        --step)    STEP="$2"; shift 2 ;;
        --func)    FUNC="$2"; shift 2 ;;
        --skip-sw) SKIP_SW=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --target <hw_emu|hw>   Vitis target (default: hw_emu)"
            echo "  --step <STEP>          Run specific step:"
            echo "                           sw    - SW experiments only"
            echo "                           gen   - Generate .mem files only"
            echo "                           sim   - RTL simulation only (xsim)"
            echo "                           build - Vitis build only"
            echo "                           run   - FPGA run only"
            echo "                           all   - Full pipeline (default)"
            echo "  --func <name>          Single function (default: all 9)"
            echo "  --skip-sw              Skip SW experiments"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Functions: $ALL_FUNCS"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FUNCS="${FUNC:-$ALL_FUNCS}"

header() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
}

# ==============================================================
# Step 1: SW Experiments
# ==============================================================
step_sw() {
    header "Step 1: SW Optimization & Experiments"
    cd "$SW_DIR"
    python3 run_all_experiments.py
    echo "[DONE] SW experiments complete"
}

# ==============================================================
# Step 2: Generate .mem files
# ==============================================================
step_gen() {
    header "Step 2: Generate .mem files"
    cd "$GEN_DIR"

    echo "--- Generating exhaustive .mem for FPGA (all 9 functions) ---"
    python3 gen_exhaustive_mem.py

    echo ""
    echo "--- Generating FMA test vectors for RTL simulation ---"
    for func in $FUNCS; do
        python3 gen_eda_mem_fma.py --func "$func" --output-dir "$ENGINE_DIR"
    done

    echo "[DONE] .mem generation complete"
}

# ==============================================================
# Step 3a: RTL Simulation (xsim)
# ==============================================================
step_sim() {
    header "Step 3a: RTL Simulation (xsim)"
    cd "$ENGINE_DIR"

    if [ -n "$FUNC" ]; then
        make all FUNC="$FUNC"
    else
        make sim-all
    fi

    echo "[DONE] RTL simulation complete"
}

# ==============================================================
# Step 3b: Vitis Build
# ==============================================================
step_build() {
    header "Step 3b: Vitis Build (TARGET=$TARGET)"
    cd "$EDA_DIR"
    make all TARGET="$TARGET"
    echo "[DONE] Vitis build complete"
}

# ==============================================================
# Step 3c: FPGA Run
# ==============================================================
step_run() {
    header "Step 3c: FPGA Run (TARGET=$TARGET)"
    cd "$EDA_DIR"

    if [ -n "$FUNC" ]; then
        make run TARGET="$TARGET" FUNC="$FUNC"
    else
        make run_all TARGET="$TARGET"
    fi

    echo "[DONE] FPGA run complete"
}

# ==============================================================
# Execute
# ==============================================================
header "EDA-NLI Pipeline (step=$STEP, target=$TARGET, func=${FUNC:-all})"

case $STEP in
    sw)
        step_sw
        ;;
    gen)
        step_gen
        ;;
    sim)
        step_gen
        step_sim
        ;;
    build)
        step_gen
        step_build
        ;;
    run)
        step_run
        ;;
    all)
        if [ "$SKIP_SW" = false ]; then
            step_sw
        fi
        step_gen
        step_sim
        step_build
        step_run
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid steps: sw, gen, sim, build, run, all"
        exit 1
        ;;
esac

header "Pipeline Complete"
