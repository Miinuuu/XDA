# Post-route switching-based power estimate, two activity models:
#   MODE=uniform    : alpha=0.2 transitions/cycle on ALL nets (no propagation bias;
#                     capacitance/internal-energy weighted, identical assumption per design)
#   MODE=annotated  : statistical propagation from inputs (i_data alpha=0.5) PLUS
#                     SRAM rd_out pins annotated alpha=0.5 so activity crosses macros
# Both read post-route 6_final.odb + SPEF parasitics, Nangate45 typical liberty.

set design $::env(PWR_DESIGN)
set mode $::env(PWR_MODE)
set base $::env(ORFS)/flow/results/nangate45/$design/base
set libdir $::env(ORFS)/flow/platforms/nangate45/lib

read_liberty $libdir/NangateOpenCellLibrary_typical.lib
read_liberty $libdir/fakeram45_256x16.lib
read_liberty $libdir/fakeram45_64x15.lib
read_db $base/6_final.odb
read_spef $base/6_final.spef
read_sdc $base/6_final.sdc

if {$mode == "uniform"} {
    set_power_activity -global -activity 0.2 -duty 0.5
} else {
    set_power_activity -input -activity 0.0 -duty 0.5
    set_power_activity -input_ports i_data -activity 0.5 -duty 0.5
    set_power_activity -input_ports i_valid -activity 0.0 -duty 1.0
    set_power_activity -input_ports rst_n -activity 0.0 -duty 1.0
    set srams [get_pins -quiet {u_sram_a/rd_out* u_sram_b/rd_out* u_config_rom/rd_out*}]
    if {[llength $srams] > 0} {
        set_power_activity -pins $srams -activity 0.5 -duty 0.5
    }
}

puts "===== POWER REPORT: $design  (mode=$mode) ====="
report_power
exit
