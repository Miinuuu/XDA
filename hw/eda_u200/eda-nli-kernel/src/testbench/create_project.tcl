# ==============================================================
# Create Vivado project with AXI VIP (project only, no simulation)
# Usage: vivado -mode batch -source src/testbench/create_project.tcl
# ==============================================================

set proj_dir "./vivado_proj"
set proj_name "eda_nli_sim"
set src_dir [file normalize "./src/IP"]
set tb_dir  [file normalize "./src/testbench"]
set cfg_dir [file normalize "./config"]

# Create project
create_project -force $proj_name $proj_dir -part xcu200-fsgd2104-2-e

# Add RTL sources
add_files -norecurse [glob $src_dir/*.v $src_dir/*.sv]
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

# Slave AXI VIP for m00_axi
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

# Slave AXI VIP for m01_axi
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

# Simulation settings
set_property -name {xsim.simulate.runtime} -value {10ms} -objects [get_filesets sim_1]

update_compile_order -fileset sim_1

# Copy .mem files to xsim working directory
set xsim_dir "$proj_dir/${proj_name}.sim/sim_1/behav/xsim"
file mkdir $xsim_dir
file copy -force $cfg_dir/config_rom.mem $xsim_dir/
file copy -force $cfg_dir/func_lut.mem $xsim_dir/

puts "=== Project created at $proj_dir/$proj_name ==="
puts "=== Open with: vivado $proj_dir/$proj_name.xpr ==="
close_project
