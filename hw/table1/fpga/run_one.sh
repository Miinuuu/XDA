#!/bin/bash
# ==============================================================================
# Run SAIF-based FPGA power flow for a single design
#
# Usage: ./run_one.sh NAME TOP RTL GENERICS TB TB_MODULE STRIP MEM_DIR SAIF_NAME
#
# Called by Makefile or directly.
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUTDIR="$SCRIPT_DIR/out"
REPORT_DIR="$SCRIPT_DIR/reports"
GLBL="$XILINX_VIVADO/data/verilog/src/glbl.v"
CLOCK_XDC="$SRC_DIR/clock.xdc"

NAME="$1"       # design name (nli, eda, nnlut16, nnlut256, fp32)
TOP="$2"        # top module
RTL="$3"        # space-separated RTL files
GENERICS="$4"   # Vivado generics (empty if none)
TB="$5"         # testbench .sv file
TB_MODULE="$6"  # testbench module name
STRIP="$7"      # SAIF strip_path
MEM_DIR="$8"    # directory with .mem files
SAIF_NAME="$9"  # output SAIF filename

D="$OUTDIR/$NAME"
mkdir -p "$D"
mkdir -p "$REPORT_DIR"

# ==============================================================================
# SAIF cleanup (removes Verilog escaped identifiers)
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

echo ""
echo "========================================="
echo " $NAME"
echo "========================================="

# --- [1/4] Vivado synth + impl ---
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

# --- [2/4] GL sim → SAIF ---
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

# --- [3/4] SAIF cleanup ---
echo "  [3/4] SAIF cleanup..."
cleanup_saif "$D/$SAIF_NAME" "$D/${SAIF_NAME%.saif}_clean.saif"

# --- [4/4] Vivado power ---
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

# --- Canonical report snapshot ---
cp -f "$D/util.rpt" "$REPORT_DIR/${NAME}_util.rpt"
cp -f "$D/timing.rpt" "$REPORT_DIR/${NAME}_timing.rpt"
cp -f "$D/power.rpt" "$REPORT_DIR/${NAME}_power.rpt"

if [ "$NAME" = "nli_full_fp32" ]; then
    cp -f "$D/util.rpt" "$REPORT_DIR/fp32_util.rpt"
    cp -f "$D/timing.rpt" "$REPORT_DIR/fp32_timing.rpt"
    cp -f "$D/power.rpt" "$REPORT_DIR/fp32_power.rpt"
fi

# --- Summary ---
matched=$(grep "Design nets matched" "$D/vivado_power.log" | grep -oP '\d+ of \d+')
dynamic=$(awk -F'|' '/Dynamic \(W\)/ {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3; exit}' "$D/power.rpt")
echo "  → Match: $matched, Dynamic: ${dynamic}W"
