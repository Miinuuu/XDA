FUNC ?= silu
ALL_FUNCS = silu gelu exp sigmoid tanh hardswish mish rsqrt reciprocal

RTL_SRCS = eda_nli_engine.v frac_mult.v ../fpu/fp_adder.v
TB_SRC   = tb_eda_nli_engine.sv
SIM_OUT  = sim_eda_nli

RTL_4S   = eda_nli_engine_4s.v ../fpu/fp_adder.v
TB_4S    = tb_eda_nli_engine_4s.sv
SIM_4S   = sim_eda_4s

RTL_7S   = eda_nli_engine_7s.v fma_interp.v fp_adder_2s.v
TB_7S    = tb_eda_nli_engine_7s.sv
SIM_7S   = sim_eda_7s

.PHONY: all gen sim gen4s sim4s sim4s-all sim4s-exhaustive gen7s sim7s clean

all: gen sim

gen:
	@echo "=== Generating EDA-NLI .mem files for $(FUNC) ==="
	python3 gen_eda_mem.py $(FUNC) .

sim: gen
	@echo "=== Compiling ==="
	xvlog --sv $(TB_SRC) $(RTL_SRCS)
	xelab tb_eda_nli_engine -debug typical -s $(SIM_OUT)
	@echo "=== Simulating ==="
	xsim $(SIM_OUT) -runall

gen4s:
	@echo "=== Generating EDA-NLI FMA .mem files for $(FUNC) ==="
	python3 gen_eda_mem_7s.py --func $(FUNC)

sim4s: gen4s
	@echo "=== Compiling 4-stage FMA ==="
	xvlog --sv $(TB_4S) $(RTL_4S)
	xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@echo "=== Simulating 4-stage FMA ==="
	xsim $(SIM_4S) -runall

sim4s-all:
	@echo "=== Compiling 4-stage FMA (once) ==="
	@xvlog --sv $(TB_4S) $(RTL_4S)
	@xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@pass=0; fail=0; \
	for func in $(ALL_FUNCS); do \
		echo ""; \
		echo "========== Testing: $$func =========="; \
		python3 gen_eda_mem_7s.py --func $$func 2>&1 | grep -E "Complete|error"; \
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

sim4s-exhaustive:
	@echo "=== Compiling 4-stage FMA (once) ==="
	@xvlog --sv $(TB_4S) $(RTL_4S)
	@xelab tb_eda_nli_engine_4s -debug typical -s $(SIM_4S)
	@pass=0; fail=0; total_vectors=0; \
	for func in $(ALL_FUNCS); do \
		echo ""; \
		echo "========== Exhaustive: $$func =========="; \
		python3 gen_eda_mem_7s.py --func $$func --exhaustive 2>&1 | grep -E "Complete|error|Exhaustive"; \
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

gen7s:
	@echo "=== Generating EDA-NLI 7s FMA .mem files for $(FUNC) ==="
	python3 gen_eda_mem_7s.py --func $(FUNC)

sim7s: gen7s
	@echo "=== Compiling 7-stage FMA ==="
	xvlog --sv $(TB_7S) $(RTL_7S)
	xelab tb_eda_nli_engine_7s -debug typical -s $(SIM_7S)
	@echo "=== Simulating 7-stage FMA ==="
	xsim $(SIM_7S) -runall

clean:
	rm -f $(SIM_OUT) $(SIM_4S) *.vcd *.mem *.log *.pb *.jou
	rm -rf xsim.dir .Xil
