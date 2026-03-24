.PHONY: help

help::
	@echo "EDA-NLI Kernel Makefile"
	@echo ""
	@echo "=== Simulation (Vivado) ==="
	@echo "  make create_proj     - Create Vivado project with AXI VIP IPs"
	@echo "  make sim             - Run simulation (requires project)"
	@echo "  make sim_all         - Create project + run simulation"
	@echo "  make open_proj       - Open Vivado GUI with project"
	@echo ""
	@echo "=== Vitis Build ==="
	@echo "  make all TARGET=<hw_emu/hw> DEVICE=<platform>"
	@echo "  make run TARGET=<hw_emu/hw> DEVICE=<platform>"
	@echo ""
	@echo "=== Clean ==="
	@echo "  make clean           - Remove build artifacts"
	@echo "  make clean_proj      - Remove Vivado project"
	@echo "  make cleanall        - Remove everything"

# ==============================================================
# Tool paths
# ==============================================================
VIVADO := vivado
VPP := v++
CXX := g++
RM = rm -f
RMDIR = rm -rf

# ==============================================================
# Vivado Simulation
# ==============================================================
PROJ_DIR := ./vivado_proj
PROJ_NAME := eda_nli_sim
VCD_FILE := ./src/testbench/eda_nli_kernel.vcd
SIM_LOG := ./xsim_simulate.log

.PHONY: create_proj sim sim_all open_proj clean_proj

# Create Vivado project + generate AXI VIP IPs
create_proj:
	$(VIVADO) -mode batch -source src/testbench/create_project.tcl -nojournal -nolog
	@echo "=== Project created: $(PROJ_DIR)/$(PROJ_NAME).xpr ==="

# Run simulation (project must exist)
sim: $(PROJ_DIR)/$(PROJ_NAME).xpr
	$(VIVADO) -mode batch -source src/testbench/run_sim.tcl -nojournal -nolog
	@echo ""
	@echo "=== Results ==="
	@grep -E "Phase|PASS|FAIL|Summary|Max|Errors" $(SIM_LOG) 2>/dev/null || true
	@echo ""
	@echo "VCD: $(VCD_FILE)"
	@echo "Log: $(SIM_LOG)"

# Create + simulate in one step
sim_all: create_proj sim

# Open Vivado GUI
open_proj: $(PROJ_DIR)/$(PROJ_NAME).xpr
	$(VIVADO) $(PROJ_DIR)/$(PROJ_NAME).xpr &

$(PROJ_DIR)/$(PROJ_NAME).xpr:
	@echo "Project not found. Run 'make create_proj' first."
	@exit 1

# Clean simulation
clean_proj:
	-$(RMDIR) $(PROJ_DIR)
	-$(RM) $(VCD_FILE) $(SIM_LOG)
	-$(RMDIR) .Xil vivado*.jou vivado*.log

# ==============================================================
# Vitis Build (hw_emu / hw)
# ==============================================================
TARGET := hw_emu
DEVICE := xilinx_u200_gen3x16_xdma_2_202110_1
XCLBIN := ./user-xclbin
XO := ./user-xo
FREQ := 150

CXXFLAGS += -g -I./ -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++17
LDFLAGS += -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread -lrt -lstdc++
HOST_SRCS := ./src/host/eda_nli_host.cpp
EXECUTABLE = host
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps --kernel_frequency $(FREQ)
EMCONFIG_DIR = $(XCLBIN)/

.PHONY: all xclbin exe run emconfig check-devices

all: check-devices $(EXECUTABLE) $(XCLBIN)/eda.$(TARGET).xclbin emconfig

exe: $(EXECUTABLE)

xclbin: $(XCLBIN)/eda.$(TARGET).xclbin

# XO from RTL
$(XO)/eda.xo: ./src/xml/user.xml ./scripts/package_kernel.tcl ./scripts/gen_xo.tcl ./src/IP/*.sv ./src/IP/*.v $(wildcard ./src/IP/*.vp)
	mkdir -p $(XO)
	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(XO)/eda.xo eda $(TARGET) $(DEVICE)

# XCLBIN from XO
$(XCLBIN)/eda.$(TARGET).xclbin: $(XO)/eda.xo
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) -l -o $(XCLBIN)/eda.$(TARGET).xclbin $(XO)/eda.xo
	cp -rf $(XCLBIN)/eda.$(TARGET).xclbin ./eda.xclbin

# Host executable
$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o '$@' $(LDFLAGS)

# Emconfig
emconfig: $(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

# Run
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) ./eda.xclbin
else
	./$(EXECUTABLE) ./eda.xclbin
endif

# ==============================================================
# Clean
# ==============================================================
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN) $(XO)
	-$(RMDIR) *.log emconfig.json eda.xclbin vivado.jou

cleanall: clean clean_proj
	-$(RMDIR) _x .Xil
	-$(RMDIR) ./tmp_kernel_pack* ./packaged_kernel*

check-devices:
ifndef DEVICE
	$(error DEVICE not set.)
endif
ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set)
endif
