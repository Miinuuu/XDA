# Makefile for NLI Engine simulation
# Requires: Icarus Verilog (iverilog) and Python 3 with PyTorch

FUNC ?= silu

# Source files
RTL_SRCS = nli_engine.v \
           fp_mult_norm.v \
           ../fpu/fp_adder.v

TB_SRC   = tb_nli_engine.sv

# Simulation
SIM_OUT  = sim_nli_engine
VCD_FILE = nli_engine.vcd

.PHONY: all gen sim wave clean

all: gen sim

# Generate .mem files from Python
gen:
	@echo "=== Generating .mem files for $(FUNC) ==="
	python3 gen_nli_mem.py $(FUNC) .

# Compile and run simulation
sim: gen
	@echo "=== Compiling RTL ==="
	iverilog -g2012 -o $(SIM_OUT) $(TB_SRC) $(RTL_SRCS)
	@echo "=== Running simulation ==="
	./$(SIM_OUT)

# Open waveform viewer
wave: $(VCD_FILE)
	gtkwave $(VCD_FILE) &

clean:
	rm -f $(SIM_OUT) $(VCD_FILE)
	rm -f point_reg.mem mul_reg.mem lut_reg.mem test_vectors.mem
	rm -f *.log
