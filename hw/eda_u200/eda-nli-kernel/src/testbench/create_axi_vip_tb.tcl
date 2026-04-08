# ==============================================================
# Create Vivado project with AXI VIP for EDA-NLI kernel testbench
# Usage: cd eda-nli-kernel && vivado -mode batch -source src/testbench/create_axi_vip_tb.tcl
# ==============================================================

set proj_dir "/tmp/eda_nli_vip_sim"
set proj_name "eda_nli_vip_tb"
set src_dir [file normalize "./src/IP"]
set tb_dir  [file normalize "./src/testbench"]
set cfg_dir [file normalize "./config"]

# Create project
create_project -force $proj_name $proj_dir -part xcu200-fsgd2104-2-e

# Add RTL sources
add_files -norecurse [glob $src_dir/*.v $src_dir/*.sv $src_dir/*.vp]
update_compile_order -fileset sources_1

# ==============================================================
# Create AXI VIP IPs
# ==============================================================
puts "Creating AXI VIP IPs..."

# Control AXI VIP (Master) - AXI4-Lite
create_ip -name axi_vip -vendor xilinx.com -library ip -version 1.1 -module_name control_eda_nli_vip
set_property -dict [list \
  CONFIG.ADDR_WIDTH {12} \
  CONFIG.DATA_WIDTH {32} \
  CONFIG.Component_Name {control_eda_nli_vip} \
  CONFIG.HAS_ARESETN {1} \
  CONFIG.HAS_WSTRB {1} \
  CONFIG.HAS_BRESP {1} \
  CONFIG.HAS_RRESP {1} \
  CONFIG.HAS_PROT {0} \
  CONFIG.INTERFACE_MODE {MASTER} \
  CONFIG.PROTOCOL {AXI4LITE} \
] [get_ips control_eda_nli_vip]

# Slave AXI VIP for m00_axi (Read input from DDR)
create_ip -name axi_vip -vendor xilinx.com -library ip -version 1.1 -module_name slv_m00_axi_vip
set_property -dict [list \
  CONFIG.ADDR_WIDTH {64} \
  CONFIG.DATA_WIDTH {512} \
  CONFIG.Component_Name {slv_m00_axi_vip} \
  CONFIG.HAS_ARESETN {1} \
  CONFIG.HAS_WSTRB {1} \
  CONFIG.HAS_BRESP {1} \
  CONFIG.HAS_BURST {1} \
  CONFIG.HAS_CACHE {0} \
  CONFIG.HAS_PROT {0} \
  CONFIG.HAS_QOS {0} \
  CONFIG.HAS_REGION {0} \
  CONFIG.HAS_RRESP {1} \
  CONFIG.HAS_LOCK {0} \
  CONFIG.INTERFACE_MODE {SLAVE} \
  CONFIG.SUPPORTS_NARROW {0} \
] [get_ips slv_m00_axi_vip]

# Slave AXI VIP for m01_axi (Write output to DDR)
create_ip -name axi_vip -vendor xilinx.com -library ip -version 1.1 -module_name slv_m01_axi_vip
set_property -dict [list \
  CONFIG.ADDR_WIDTH {64} \
  CONFIG.DATA_WIDTH {512} \
  CONFIG.Component_Name {slv_m01_axi_vip} \
  CONFIG.HAS_ARESETN {1} \
  CONFIG.HAS_WSTRB {1} \
  CONFIG.HAS_BRESP {1} \
  CONFIG.HAS_BURST {1} \
  CONFIG.HAS_CACHE {0} \
  CONFIG.HAS_PROT {0} \
  CONFIG.HAS_QOS {0} \
  CONFIG.HAS_REGION {0} \
  CONFIG.HAS_RRESP {1} \
  CONFIG.HAS_LOCK {0} \
  CONFIG.INTERFACE_MODE {SLAVE} \
  CONFIG.SUPPORTS_NARROW {0} \
] [get_ips slv_m01_axi_vip]

# Generate all IP targets
generate_target {instantiation_template simulation} [get_ips control_eda_nli_vip]
generate_target {instantiation_template simulation} [get_ips slv_m00_axi_vip]
generate_target {instantiation_template simulation} [get_ips slv_m01_axi_vip]

puts "AXI VIP IPs generated."

# ==============================================================
# Add testbench
# ==============================================================
add_files -fileset sim_1 -norecurse $tb_dir/tb_axi_vip_kernel.sv
set_property top tb_axi_vip_kernel [get_filesets sim_1]

# Copy config files to project dir AND xsim working directory
file copy -force $cfg_dir/config_rom.mem $proj_dir/
file copy -force $cfg_dir/func_lut.mem $proj_dir/
# xsim runs from <proj>/<name>.sim/sim_1/behav/xsim/
set xsim_dir "$proj_dir/${proj_name}.sim/sim_1/behav/xsim"
file mkdir $xsim_dir
file copy -force $cfg_dir/config_rom.mem $xsim_dir/
file copy -force $cfg_dir/func_lut.mem $xsim_dir/

# Simulation settings
set_property -name {xsim.simulate.runtime} -value {10ms} -objects [get_filesets sim_1]

update_compile_order -fileset sim_1

puts "=== Project created at $proj_dir ==="
puts "=== Launching simulation... ==="

# Launch simulation
launch_simulation

# VCD dump - key signals only (smaller file, no gtkwave crash)
set vcd_dir [file normalize "./src/testbench"]
open_vcd $vcd_dir/eda_nli_kernel.vcd

# Clock/Reset
log_vcd /tb_axi_vip_kernel/ap_clk
log_vcd /tb_axi_vip_kernel/aresetn

# Top-level reset sync
log_vcd /tb_axi_vip_kernel/dut/rst_n_sync

# Control S_AXI
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/wstate
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_ap_start
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_ap_done
log_vcd /tb_axi_vip_kernel/dut/inst_control_s_axi/int_cfg_we

# Wrapper control
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/ap_start_pulse
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/areset
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/ap_idle_r
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/read_done
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/write_done

# AXI-Stream: read master → compute
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/rd_tlast

# AXI-Stream: compute → write master
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/wr_tlast

# Compute internals
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/state
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/in_cnt
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/sipo_cnt
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/fifo_count
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_i_valid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_o_valid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/u_compute/nli_o_data

# M00 AXI read channel
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_arvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_arready
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_rvalid
log_vcd /tb_axi_vip_kernel/dut/EDA_NLI_inst/m00_axi_rready

# M01 AXI write channel
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
close_project
