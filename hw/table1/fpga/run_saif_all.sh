#!/bin/bash
# ==============================================================================
# SAIF-based FPGA Power Analysis — All 5 designs
#
# Designs: NLI, EDA-NLI-4s, NN-LUT-16, NN-LUT-256, NLI-FP32
# Target:  xcu200-fsgd2104-2-e, DSP-free (-max_dsp 0)
# Flow:    .mem gen → Vivado synth+impl → GL sim (xsim SAIF) → SAIF cleanup → Vivado power
#
# Output:  out/{design}/power.rpt
#
# Key: EDA loads func_lut BEFORE config_rom to avoid X propagation in fp_adder.
#      SAIF byte-level cleanup removes Verilog escaped identifiers (\\name\).
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
MEMGEN_DIR="$SCRIPT_DIR/memgen"
OUTDIR="$SCRIPT_DIR/out"
GLBL="$XILINX_VIVADO/data/verilog/src/glbl.v"

if [ ! -f "$GLBL" ]; then
    echo "ERROR: XILINX_VIVADO not set. Source Vivado settings64.sh first."
    exit 1
fi

CLOCK_XDC="$SRC_DIR/clock.xdc"

# ==============================================================================
# SAIF cleanup function (removes Verilog escaped identifiers from SAIF)
# ==============================================================================
cleanup_saif() {
    local input="$1" output="$2"
    python3 -c "
import re, sys
data = open('$input', 'rb').read()
out = bytearray()
i = 0
while i < len(data):
    if data[i:i+3] == b'(\x5c\x5c':
        j = i + 3
        while j < len(data) - 1:
            if data[j] == 0x5c and data[j+1] == 0x20: break
            j += 1
        if j < len(data) - 1:
            name = data[i+3:j].replace(b'\x5c[', b'[').replace(b'\x5c]', b']').replace(b'\x5c/', b'/')
            out.extend(b'(' + name + b' ')
            i = j + 2
            continue
    out.append(data[i])
    i += 1
result = bytes(out).replace(b'\x5c[', b'[').replace(b'\x5c]', b']')
open('$output', 'wb').write(result)
s = result.decode()
dur = int(re.search(r'DURATION\s+(\d+)', s).group(1))
entries = re.findall(r'\((\S+)\s+\(T0\s+(\d+)\)\s+\(T1\s+(\d+)\)\s+\(TX\s+(\d+)\)', s)
toggle = sum(1 for _,t0,t1,_ in entries if int(t0)>0 and int(t1)>0)
high_tx = sum(1 for _,_,_,tx in entries if int(tx) > dur*0.5)
print(f'  Signals:{len(entries)} Toggle:{toggle} TX>50%:{high_tx}')
"
}

# ==============================================================================
# Per-design flow function
# ==============================================================================
run_design() {
    local NAME="$1"       # design name
    local TOP="$2"        # top module
    local RTL="$3"        # space-separated RTL files
    local GENERICS="$4"   # Vivado generics (empty if none)
    local TB="$5"         # testbench .sv file
    local TB_MODULE="$6"  # testbench module name
    local STRIP="$7"      # SAIF strip_path
    local MEM_DIR="$8"    # directory with .mem files
    local SAIF_NAME="$9"  # output SAIF filename

    local D="$OUTDIR/$NAME"
    mkdir -p "$D"
    echo ""
    echo "========================================="
    echo " $NAME"
    echo "========================================="

    # --- Vivado synth + impl ---
    echo "  [1/4] Vivado synth + impl..."
    cat > "$D/synth.tcl" << TCLEOF
create_project -force -part xcu200-fsgd2104-2-e proj $D/proj
add_files {$RTL}
add_files -fileset constrs_1 $CLOCK_XDC
set_property top $TOP [current_fileset]
$([ -n "$GENERICS" ] && echo "set_property generic {$GENERICS} [current_fileset]")
synth_design -top $TOP -flatten_hierarchy rebuilt -max_dsp 0
write_verilog -mode funcsim -force $D/funcsim.v
opt_design
place_design
route_design
write_checkpoint -force $D/routed.dcp
report_utilization -file $D/util.rpt
report_timing_summary -max_paths 1 -file $D/timing.rpt
close_project
TCLEOF
    vivado -mode batch -source "$D/synth.tcl" -log "$D/vivado_synth.log" \
        -journal "$D/vivado_synth.jou" > /dev/null 2>&1

    # --- GL sim → SAIF ---
    echo "  [2/4] GL simulation → SAIF..."
    cd "$MEM_DIR"
    rm -rf xsim.dir .Xil
    xvlog "$D/funcsim.v" --log "$D/xvlog.log" > /dev/null 2>&1
    xvlog --sv "$TB" --log "$D/xvlog_tb.log" > /dev/null 2>&1
    xvlog "$GLBL" --log "$D/xvlog_g.log" > /dev/null 2>&1
    xelab "$TB_MODULE" glbl -debug typical -L unisims_ver -s "${NAME}_sim" \
        --log "$D/xelab.log" > /dev/null 2>&1

    cat > "$D/saif.tcl" << TCLEOF
open_saif $SAIF_NAME
log_saif [get_objects -r /${TB_MODULE}/u_dut/*]
run -all
close_saif
quit
TCLEOF
    xsim "${NAME}_sim" -tclbatch "$D/saif.tcl" --log "$D/xsim.log" > /dev/null 2>&1
    mv "$SAIF_NAME" "$D/"
    rm -rf xsim.dir .Xil

    # --- SAIF cleanup ---
    echo "  [3/4] SAIF cleanup..."
    cleanup_saif "$D/$SAIF_NAME" "$D/${SAIF_NAME%.saif}_clean.saif"

    # --- Vivado power ---
    echo "  [4/4] Vivado power..."
    cat > "$D/power.tcl" << TCLEOF
open_checkpoint $D/routed.dcp
read_saif -strip_path $STRIP $D/${SAIF_NAME%.saif}_clean.saif
set_operating_conditions -process typical
report_power -file $D/power.rpt -advisory
close_design
TCLEOF
    vivado -mode batch -source "$D/power.tcl" -log "$D/vivado_power.log" \
        -journal "$D/vivado_power.jou" > /dev/null 2>&1

    # --- Summary ---
    matched=$(grep "Design nets matched" "$D/vivado_power.log" | grep -oP '\d+ of \d+')
    dynamic=$(grep "Dynamic (W)" "$D/power.rpt" | head -1 | awk '{print $4}')
    echo "  → Match: $matched, Dynamic: ${dynamic}W"
}

# ==============================================================================
# Step 0: Generate .mem files
# ==============================================================================
echo "========================================="
echo " Generating .mem files"
echo "========================================="

# NLI & NLI-FP32 .mem → $OUTDIR/nli_mem
mkdir -p "$OUTDIR/nli_mem"
cd "$OUTDIR/nli_mem"
python3 "$MEMGEN_DIR/gen_nli_mem.py" silu . && echo "  NLI .mem OK"
python3 "$MEMGEN_DIR/gen_nli_mem_fp32.py" silu . && echo "  NLI-FP32 .mem OK"

# EDA .mem → $OUTDIR/eda_mem
mkdir -p "$OUTDIR/eda_mem"
cd "$OUTDIR/eda_mem"
python3 "$MEMGEN_DIR/gen_eda_mem.py" silu . && echo "  EDA .mem OK"

# NN-LUT-16 .mem → $OUTDIR/nnlut16_mem
mkdir -p "$OUTDIR/nnlut16_mem"
cd "$OUTDIR/nnlut16_mem"
python3 "$MEMGEN_DIR/gen_nn_lut_mem.py" --func silu --outdir . && echo "  NN-LUT-16 .mem OK"

# NN-LUT-256 .mem → $OUTDIR/nnlut256_mem
python3 "$MEMGEN_DIR/gen_nn_lut_mem.py" --func silu --segments 256 --outdir "$OUTDIR/nnlut256_mem" && echo "  NN-LUT-256 .mem OK"

mkdir -p "$OUTDIR"

# ==============================================================================
# NLI
# ==============================================================================
run_design "nli" "nli_engine" \
    "$SRC_DIR/nli_engine.v $SRC_DIR/fp_mult_norm.v $SRC_DIR/fp_adder.v" \
    "" \
    "$SCRIPT_DIR/tb_nli_power.sv" "tb_nli_engine" \
    "tb_nli_engine/u_dut" \
    "$OUTDIR/nli_mem" \
    "nli_engine.saif"

# ==============================================================================
# EDA (func_lut loaded first to avoid X propagation)
# ==============================================================================
run_design "eda" "eda_nli_engine_4s" \
    "$SRC_DIR/eda_nli_engine_4s.v $SRC_DIR/fp_adder.v" \
    "" \
    "$SCRIPT_DIR/tb_eda_power.sv" "tb_eda_nli_engine" \
    "tb_eda_nli_engine/u_dut" \
    "$OUTDIR/eda_mem" \
    "eda_nli_engine_4s.saif"

# ==============================================================================
# NN-LUT-16
# ==============================================================================
run_design "nnlut16" "nn_lut_engine" \
    "$SRC_DIR/nn_lut_engine.v $SRC_DIR/fp_mult_norm.v $SRC_DIR/fp_adder.v" \
    "" \
    "$SCRIPT_DIR/tb_nnlut_power.sv" "tb_nn_lut_engine" \
    "tb_nn_lut_engine/u_dut" \
    "$OUTDIR/nnlut16_mem" \
    "nn_lut_engine.saif"

# ==============================================================================
# NN-LUT-256
# ==============================================================================
run_design "nnlut256" "nn_lut_engine" \
    "$SRC_DIR/nn_lut_engine.v $SRC_DIR/fp_mult_norm.v $SRC_DIR/fp_adder.v" \
    "N_SEGMENTS=256 N_BP=255 LUT_DEPTH=256" \
    "$SCRIPT_DIR/tb_nnlut256_power.sv" "tb_nn_lut_engine" \
    "tb_nn_lut_engine/u_dut" \
    "$OUTDIR/nnlut256_mem" \
    "nnlut256.saif"

# ==============================================================================
# NLI-FP32
# ==============================================================================
run_design "fp32" "nli_engine_fp32" \
    "$SRC_DIR/nli_engine_fp32.v $SRC_DIR/fp_mult_norm.v $SRC_DIR/fp_adder.v" \
    "" \
    "$SCRIPT_DIR/tb_fp32_power.sv" "tb_nli_engine_fp32" \
    "tb_nli_engine_fp32/u_dut" \
    "$OUTDIR/nli_mem" \
    "nli_fp32_engine.saif"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "========================================="
echo " All done! Results in: $OUTDIR/"
echo "========================================="
for d in nli eda nnlut16 nnlut256 fp32; do
    matched=$(grep "Design nets matched" "$OUTDIR/$d/vivado_power.log" 2>/dev/null | grep -oP '\d+ of \d+')
    dynamic=$(grep "Dynamic (W)" "$OUTDIR/$d/power.rpt" 2>/dev/null | head -1 | awk '{print $4}')
    echo "  $d: Match=$matched Dynamic=${dynamic}W"
done
