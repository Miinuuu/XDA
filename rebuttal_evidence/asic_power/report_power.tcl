# Post-route switching-based power estimate (OpenSTA via OpenROAD)
# Identical activity assumptions for all designs:
#   - clk from signoff SDC (10 ns)
#   - i_data: activity 0.5, duty 0.5  (new random FP16 operand each cycle)
#   - i_valid: constant 1 (streaming), rst_n: constant 1, cfg_*: static
# Activities are statistically propagated through the post-route netlist
# with 6_final.spef parasitics (Nangate45 typical liberty).

set design $::env(PWR_DESIGN)
set base $::env(ORFS_ROOT)/flow/results/nangate45/$design/base
set libdir $::env(ORFS_ROOT)/flow/platforms/nangate45/lib

read_liberty $libdir/NangateOpenCellLibrary_typical.lib
read_liberty $libdir/fakeram45_256x16.lib
read_liberty $libdir/fakeram45_64x15.lib
read_db $base/6_final.odb
read_spef $base/6_final.spef
read_sdc $base/6_final.sdc

set_power_activity -input -activity 0.0 -duty 0.5
set_power_activity -input_port i_data -activity 0.5 -duty 0.5
set_power_activity -input_port i_valid -activity 0.0 -duty 1.0
set_power_activity -input_port rst_n -activity 0.0 -duty 1.0

puts "===== POWER REPORT: $design ====="
report_power
exit
