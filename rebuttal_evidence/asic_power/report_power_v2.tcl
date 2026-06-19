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
# Enforce the common 10 ns signoff clock for an ISO-FREQUENCY cross-design power
# comparison. Dynamic power scales with f, so if a build dir's 6_final.sdc was written
# by a different (e.g. Fmax-tight) synthesis run, reading it verbatim would distort the
# XDA-vs-NLI delta. This override is a no-op when the design was built with the shipped
# 10 ns shared constraint.sdc, and corrects a stale/overridden build SDC otherwise.
create_clock -name core_clock -period 10.0 [get_ports clk]

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
