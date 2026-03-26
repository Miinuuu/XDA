#!/bin/bash
# ==============================================================
# Run hw_emu test for all 9 NLI functions
# ==============================================================
set -e

FUNCTIONS="sigmoid tanh silu mish gelu hardswish exp reciprocal rsqrt"
XCLBIN="./eda.xclbin"
HOST="./host"
PASS_COUNT=0
FAIL_COUNT=0
RESULTS=""

# Check prerequisites
if [ ! -f "$XCLBIN" ]; then
    echo "ERROR: $XCLBIN not found. Run 'make all TARGET=hw_emu' first."
    exit 1
fi

if [ ! -f "$HOST" ]; then
    echo "ERROR: $HOST not found. Building..."
    make exe
fi

# Copy emconfig if needed
if [ ! -f "./emconfig.json" ] && [ -f "./user-xclbin/emconfig.json" ]; then
    cp ./user-xclbin/emconfig.json .
fi

echo "============================================================"
echo " EDA-NLI hw_emu Test: All 9 Functions"
echo "============================================================"
echo ""

for FUNC in $FUNCTIONS; do
    echo "============================================================"
    echo " Testing: $FUNC"
    echo "============================================================"

    # Check config directory exists
    if [ ! -d "./config/$FUNC" ]; then
        echo "  SKIP: config/$FUNC directory not found"
        RESULTS="$RESULTS  $FUNC: SKIP\n"
        continue
    fi

    # Run hw_emu
    LOG_FILE="./hw_emu_${FUNC}.log"
    if XCL_EMULATION_MODE=hw_emu ./$HOST "$XCLBIN" "$FUNC" 2>&1 | tee "$LOG_FILE"; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS="$RESULTS  $FUNC: PASS\n"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS="$RESULTS  $FUNC: FAIL\n"
    fi
    echo ""
done

echo "============================================================"
echo " FINAL RESULTS"
echo "============================================================"
echo -e "$RESULTS"
echo "PASSED: $PASS_COUNT / $((PASS_COUNT + FAIL_COUNT))"
echo "FAILED: $FAIL_COUNT / $((PASS_COUNT + FAIL_COUNT))"
echo "============================================================"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
