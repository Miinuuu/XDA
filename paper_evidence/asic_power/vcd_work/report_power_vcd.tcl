# VCD-activity (measured switching) post-route power, iso-frequency 10 ns.
# Activity comes from a gate-level simulation VCD (real stimulus) rather than a
# uniform/propagated assumption. Env: PWR_DESIGN, PWR_VCD, PWR_SCOPE.
set design $::env(PWR_DESIGN)
set vcd    $::env(PWR_VCD)
set scope  $::env(PWR_SCOPE)
set base   $::env(ORFS)/flow/results/nangate45/$design/base
set libdir $::env(ORFS)/flow/platforms/nangate45/lib

read_liberty $libdir/NangateOpenCellLibrary_typical.lib
read_liberty $libdir/fakeram45_256x16.lib
read_liberty $libdir/fakeram45_64x15.lib
read_db   $base/6_final.odb
read_spef $base/6_final.spef
read_sdc  $base/6_final.sdc
create_clock -name core_clock -period 10.0 [get_ports clk]

read_vcd -scope $scope $vcd

puts "===== VCD POWER: $design  (vcd=[file tail $vcd]) ====="
report_power
exit
