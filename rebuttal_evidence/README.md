# Rebuttal Evidence (ICCAD 2026 Paper #1387)

Pre-generated outputs backing the new quantitative results reported in the author
rebuttal, so that each number can be inspected directly without re-running the flows.
The corresponding regeneration flows live under `hw/` and `sw/` (see the top-level
README). Vivado/OpenSTA report headers are anonymized (`<RUN_DIR>`, `Host`); report
bodies are unmodified tool output. `eda_*` file names refer to the XDA design (its
internal name in this artifact).

| Rebuttal section | Evidence files | Key numbers |
|---|---|---|
| §1 ASIC power/energy (A-Q3 = B-Q2; D-W2) | `asic_power/POWER_SUMMARY.md`, `asic_power/pwr_*_uniform.log`, `asic_power/report_power.tcl` | NLI 2.683 → XDA 1.800 mW total (−33%); logic −57%; α ∈ [0.1, 0.4] sweep stays −30–35%; 26.8 → 18.0 pJ/op at the common 10 ns clock |
| §2 SRAM-area breakdown (A-Q2) | `sram_breakdown/asic_{xda,nli}_6_report.json` | sequential cells: NLI 2,296 vs. XDA 668 µm²; XDA's +659 µm² = one 64×15 config ROM |
| §3 BF16/FP8 generalization (A-Q1) | `bf16/exp_bf16_preliminary.py`, `bf16/bf16_preliminary_out.log` | W=512 covers all 9 functions; nonzero K on 19–34 octaves for the activation functions; rsqrt 36/36 and reciprocal 32/32 octaves |
| §4 DSP-enabled FPGA synthesis (B-Q1 = D-Q3; D-W6) | `fpga_dsp_enabled/{design}_{dsp,nodsp}_{util,timing}.rpt` | CLB LUTs 881 (XDA) / 1,829 (NLI) / 801 (NN-LUT16); FFs 136/287/306; DSPs 1/2/1; Fmax = 1/(2.0 ns − WNS) → 156.6/102.7/109.3 MHz; nodsp reruns reproduce Table 1 within ±2 LUTs |
| §6 rsqrt distribution-free ε-bound (B-Q4 = D-Q4; D-W1) | `rsqrt_eps_sweep/rsqrt_eps_sweep.py`, `rsqrt_eps_sweep/rsqrt_eps_sweep_results.json` | exhaustive max relative error over every FP16 value ≥ ε: XDA 50.0 vs. NLI 7,002 (×10⁻⁴) at ε = 10⁻⁵ (140×); 1,180 vs. 9,046 at 10⁻⁶ |
| §7 NN-LUT256 row (D-W4) | `nnlut256/fpga_nnlut256_{util,timing,power}.rpt`, `nnlut256/asic_nnlut256_sram_6_report.json` | FPGA 6,814 CLB LUTs / 4,130 FFs / 76.0 MHz / 99 mW dynamic; ASIC total area 62,116 µm² |

Notes:
- `asic_power/report_power.tcl` reads the post-route `6_final.odb`/`6_final.spef`
  produced by the ASIC flow under `hw/table1/asic/` (set `ORFS_ROOT` to your
  OpenROAD-flow-scripts checkout).
- The rsqrt sweep script regenerates `rsqrt_eps_sweep_results.json` byte-identically
  on GPU; a CPU fallback is included (near-tie cells may differ in the last digit, as
  documented in the script header).
- The BF16 log's W=512 block is the source of the rebuttal's 19–34 / 36/36 / 32/32
  octave counts (one row per function).
