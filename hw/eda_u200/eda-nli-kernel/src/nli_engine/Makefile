FUNC ?= silu
ALL_FUNCS = silu gelu exp sigmoid tanh hardswish mish rsqrt reciprocal
GEN_DIR = ../../../../../gen

RTL_4S   = eda_nli_engine_4s.v ../IP/fp_adder.v
TB_4S    = tb_eda_nli_engine_4s.sv
SIM_4S   = sim_eda_4s

.PHONY: all gen sim sim-all sim-exhaustive clean

all: gen sim

gen:
	@echo "=== Generating EDA-NLI FMA .mem files for $(FUNC) ==="
	python3 $(GEN_DIR)/gen_eda_mem_fma.py --func $(FUNC) --output-dir $(CURDIR)

sim: gen
	@echo "=== Compiling 4-stage FMA ==="
	xvlog --sv $(TB_4S) $(RTL_4S)
	xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@echo "=== Simulating 4-stage FMA ==="
	xsim $(SIM_4S) -runall

sim-all:
	@echo "=== Compiling 4-stage FMA (once) ==="
	@xvlog --sv $(TB_4S) $(RTL_4S)
	@xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@pass=0; fail=0; \
	for func in $(ALL_FUNCS); do \
		echo ""; \
		echo "========== Testing: $$func =========="; \
		python3 $(GEN_DIR)/gen_eda_mem_fma.py --func $$func --output-dir $(CURDIR) 2>&1 | grep -E "Complete|error"; \
		result=$$(xsim $(SIM_4S) -runall 2>&1); \
		echo "$$result" | grep -E "Results|Total|Max ULP|PASSED|FAILED"; \
		if echo "$$result" | grep -q "ALL TESTS PASSED"; then \
			pass=$$((pass + 1)); \
		else \
			fail=$$((fail + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "=== Summary: $$pass/9 passed, $$fail/9 failed ==="

sim-exhaustive:
	@echo "=== Compiling 4-stage FMA (once) ==="
	@xvlog --sv $(TB_4S) $(RTL_4S)
	@xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@pass=0; fail=0; total_vectors=0; \
	for func in $(ALL_FUNCS); do \
		echo ""; \
		echo "========== Exhaustive: $$func =========="; \
		python3 $(GEN_DIR)/gen_eda_mem_fma.py --func $$func --exhaustive --output-dir $(CURDIR) 2>&1 | grep -E "Complete|error|Exhaustive"; \
		result=$$(xsim $(SIM_4S) -runall 2>&1); \
		echo "$$result" | grep -E "Results|Total|Max ULP|PASSED|FAILED"; \
		nvec=$$(echo "$$result" | grep -oP 'Total: \K[0-9]+'); \
		total_vectors=$$((total_vectors + nvec)); \
		if echo "$$result" | grep -q "ALL TESTS PASSED"; then \
			pass=$$((pass + 1)); \
		else \
			fail=$$((fail + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "=== Exhaustive Summary: $$pass/9 passed, $$fail/9 failed, $$total_vectors total vectors ==="

clean:
	rm -f $(SIM_4S) *.vcd *.mem *.log *.pb *.jou
	rm -rf xsim.dir .Xil
