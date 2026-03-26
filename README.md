# EDA-NLI: Exponent-Direct Addressing Nonlinear Interpolation

FP16 비트필드 추출을 이용한 zero-FP-arithmetic 주소 생성 기반 비선형 함수 근사 엔진.

## Directory Structure

```
├── sw/                          # Python software
│   ├── nli_eda.py               # EDA-NLI core (optimize, evaluate)
│   ├── nli_eda_engine.py        # EDA engine implementation
│   ├── nli_engine.py            # baseline NLI engine
│   ├── nli_dp.py                # DP-based cutpoint optimization
│   ├── ablation_sweep.py        # ablation study
│   ├── nli_table1_eval.py       # Table 1 reproduction
│   └── run_all_experiments.py   # 전체 실험 실행
│
└── hw/                          # Hardware RTL
    ├── nli/                     # NLI engine RTL + testbench
    │   ├── nli_engine.v
    │   ├── gen_nli_mem.py       # .mem 파일 생성
    │   ├── tb_nli_engine.sv
    │   └── Makefile
    ├── fpu/                     # FP adder IP
    └── eda_u200/                # Xilinx U200 Vitis kernel
        └── eda-nli-kernel/
            ├── src/IP/          # kernel RTL (encrypted .vp)
            ├── src/host/        # host application (C++)
            ├── src/c-model/     # C reference model
            ├── config/          # LUT config & test vectors
            └── Makefile
```

## How to Run

### SW: Python 실험

```bash
cd sw

# 전체 실험 (Table 1~5 재현)
python3 run_all_experiments.py

# 개별 실행
python3 nli_eda.py           # EDA-NLI 단독
python3 ablation_sweep.py    # ablation study
python3 nli_table1_eval.py   # Table 1
```

Requirements: Python 3, PyTorch, NumPy

### HW: RTL 시뮬레이션 (Icarus Verilog)

```bash
cd hw/nli

# .mem 생성 + 시뮬레이션 (기본: silu)
make all

# 함수 지정
make all FUNC=gelu

# 파형 확인
make wave
```

Requirements: Icarus Verilog (`iverilog`), Python 3, GTKWave (파형 확인 시)

### HW: Vitis U200 커널

```bash
cd hw/eda_u200/eda-nli-kernel

# --- HW Emulation ---
make all TARGET=hw_emu
make run TARGET=hw_emu FUNC=sigmoid       # 단일 함수
make run_all TARGET=hw_emu                # 전체 9개 함수

# --- HW (실제 FPGA 빌드 & 실행) ---
make all TARGET=hw
make run TARGET=hw FUNC=sigmoid
make run_all TARGET=hw
```

지원 함수: `sigmoid`, `tanh`, `silu`, `mish`, `gelu`, `hardswish`, `exp`, `reciprocal`, `rsqrt`

Requirements: Vitis 2021.1+, Xilinx U200 platform (`xilinx_u200_gen3x16_xdma_2_202110_1`)

## Note

EDA 핵심 RTL(`eda_nli_engine_4s.vp`)은 현재 암호화된 형태로 제공됩니다. 시뮬레이션, 합성, 비트스트림 생성은 정상적으로 가능합니다. 논문 게재 확정 후 암호화되지 않은 소스 코드를 공개할 예정입니다.
