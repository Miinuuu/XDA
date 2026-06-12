# Rebuttal Evidence (ICCAD 2026 Paper #1387)

Pre-generated outputs backing the new quantitative results reported in the author
rebuttal, so that each number can be inspected directly without re-running the flows.
The corresponding regeneration flows live under `hw/` and `sw/` (see the top-level
README). Vivado/OpenSTA report headers are anonymized (`<RUN_DIR>`, `Host`); report
bodies are unmodified tool output. `eda_*` file names refer to the XDA design (its
internal name in this artifact).

| Rebuttal section | Evidence files | Key numbers |
|---|---|---|
| §1 ASIC power/energy (A-Q3 = B-Q2; D-W2) | `asic_power/POWER_SUMMARY.md`, `asic_power/pwr_*_uniform.log`, `asic_power/report_power_v2.tcl` | NLI 2.683 → XDA 1.800 mW total (−33%); logic −57%; α ∈ [0.1, 0.4] sweep stays −30–35%; 26.8 → 18.0 pJ/op at the common 10 ns clock |
| §2 SRAM-area breakdown (A-Q2) | `sram_breakdown/asic_{xda,nli}_6_report.json` | sequential cells: NLI 2,296 vs. XDA 668 µm²; XDA's +659 µm² = one 64×15 config ROM |
| §3 BF16/FP8 generalization (A-Q1) | `bf16/exp_bf16_preliminary.py`, `bf16/bf16_preliminary_out.log` | W=512 covers all 9 functions; nonzero K on 19–34 octaves for the activation functions; rsqrt 36/36 and reciprocal 32/32 octaves |
| §4 DSP-enabled FPGA synthesis (B-Q1 = D-Q3; D-W6) | `fpga_dsp_enabled/{design}_{dsp,nodsp}_{util,timing}.rpt` | CLB LUTs 881 (XDA) / 1,829 (NLI) / 801 (NN-LUT16); FFs 136/287/306; DSPs 1/2/1; Fmax = 1/(2.0 ns − WNS) → 156.6/102.7/109.3 MHz; nodsp reruns reproduce Table 1 within ±2 LUTs |
| §6 rsqrt distribution-free ε-bound (B-Q4 = D-Q4; D-W1) | `rsqrt_eps_sweep/rsqrt_eps_sweep.py`, `rsqrt_eps_sweep/rsqrt_eps_sweep_results.json` | exhaustive max relative error over every FP16 value ≥ ε: XDA 50.0 vs. NLI 7,002 (×10⁻⁴) at ε = 10⁻⁵ (140×); 1,180 vs. 9,046 at 10⁻⁶ |
| §7 NN-LUT256 row (D-W4) | `nnlut256/fpga_nnlut256_{util,timing,power}.rpt`, `nnlut256/asic_nnlut256_sram_6_report.json` | FPGA 6,814 CLB LUTs / 4,130 FFs / 76.0 MHz / 99 mW dynamic; ASIC total area 62,116 µm² |

Notes:
- `asic_power/run_power.sh` regenerates the uniform power logs: it runs
  `report_power_v2.tcl` with `PWR_MODE=uniform` for the three designs
  (`nli_engine`, `eda_nli_engine_4s`, `nn_lut_engine_16` — the same directory names
  as `hw/table1/asic/configs/`), reading each post-route `6_final.odb`/`.spef` from
  `$ORFS/flow/results/nangate45/<design>/base`. First build the designs with the
  flow under `hw/table1/asic/`, then `export ORFS=<your OpenROAD-flow-scripts
  checkout>` and run the script; outputs land as `repro_pwr_*_uniform.log` for
  diffing against the shipped logs. The `Reproduce` block inside `POWER_SUMMARY.md`
  records the same loop as originally run in the source tree. `PWR_MODE=annotated`
  is the input-propagation variant that `POWER_SUMMARY.md` documents as biased
  across SRAM macro boundaries (kept for transparency, not used for the rebuttal
  numbers).
- The rsqrt sweep script regenerates `rsqrt_eps_sweep_results.json` byte-identically
  on GPU; a CPU fallback is included (near-tie cells may differ in the last digit, as
  documented in the script header).
- The BF16 log's W=512 block is the source of the rebuttal's 19–34 / 36/36 / 32/32
  octave counts (one row per function).
