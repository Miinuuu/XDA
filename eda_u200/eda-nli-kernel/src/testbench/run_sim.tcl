# ==============================================================
# Run simulation on existing Vivado project
# Usage: vivado -mode batch -source src/testbench/run_sim.tcl
# ==============================================================

set proj_dir "./vivado_proj"
set proj_name "eda_nli_sim"
set cfg_dir [file normalize "./config"]
set vcd_dir [file normalize "./src/testbench"]

# Open project
open_project $proj_dir/$proj_name.xpr

# Ensure .mem files are in xsim directory
set xsim_dir "$proj_dir/${proj_name}.sim/sim_1/behav/xsim"
file mkdir $xsim_dir
file copy -force $cfg_dir/config_rom.mem $xsim_dir/
file copy -force $cfg_dir/func_lut.mem $xsim_dir/

# Launch simulation
launch_simulation

# VCD dump - key signals
open_vcd $vcd_dir/eda_nli_kernel.vcd

# Clock/Reset
log_vcd /tb_axi_vip_kernel/ap_clk
log_vcd /tb_axi_vip_kernel/aresetn
log_vcd /tb_axi_vip_kernel/dut/rst_n_sync

# Control
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/wstate
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_ap_start
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_ap_done
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_cfg_we

# Wrapper
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/ap_start_pulse
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/areset
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/ap_idle_r
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/read_done
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/write_done

# AXI-Stream
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tlast
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tlast

# Compute
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/state
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/in_cnt
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/sipo_cnt
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/fifo_count
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_i_valid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_o_valid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_o_data

# M00 AXI read
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_arvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_arready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_rvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_rready

# M01 AXI write
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_awvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_awready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_wvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_wready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_wlast
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_bvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m01_axi_bready

run all
close_vcd
puts "VCD saved to $vcd_dir/eda_nli_kernel.vcd"

# Copy log
set log_src "$proj_dir/${proj_name}.sim/sim_1/behav/xsim/simulate.log"
file copy -force $log_src "./xsim_simulate.log"
puts "Log saved to ./xsim_simulate.log"

close_project
