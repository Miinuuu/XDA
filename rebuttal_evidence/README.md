# Rebuttal Evidence (ICCAD 2026 Paper #1387)

Pre-generated outputs backing the new quantitative results reported in the author
rebuttal, so that each number can be inspected directly without re-running the flows.
The corresponding regeneration flows live under `hw/` and `sw/` (see the top-level
README). Vivado/OpenSTA report headers are anonymized (`<RUN_DIR>`, `Host`); report
bodies are unmodified tool output. `eda_*` file names refer to the XDA design (its
internal name in this artifact).

| Rebuttal section | Evidence files | Key numbers |
|---|---|---|
| §1 ASIC power/energy (A-Q3 = B-Q2; D-W2) | `asic_power/POWER_SUMMARY.md`, `asic_power/vcd_work/pwr_{eda_nli_engine_4s,nli_engine}_vcd.log`, `asic_power/vcd_work/run_vcd_power.sh`; vectorless cross-check: `asic_power/pwr_*_uniform.log`, `asic_power/report_power_v2.tcl` | **gate-level-sim VCD-annotated estimate** (the reported number): NLI 2.23 → XDA 1.52 mW total (**−32%**); logic −63%; 22.3 → 15.2 pJ/op at the common 10 ns clock. Assumption-free vectorless cross-check agrees: NLI 2.683 → XDA 1.800 mW (−33%; logic −57%) |
| §2 SRAM-area breakdown (A-Q2) | `sram_breakdown/asic_{xda,nli}_6_report.json` | sequential cells: NLI 2,296 vs. XDA 668 µm²; XDA's +659 µm² = one 64×15 config ROM |
| §3 BF16/FP8 generalization (A-Q1) | `bf16/exp_bf16_preliminary.py`, `bf16/bf16_preliminary_out.log`, `bf16/exp_bf16_floor.py`, `bf16/bf16_floor_out.log` | W=512 covers all 9 functions; nonzero K on 19–34 octaves for the activation/exp-family functions (the 34 maximum is exp); rsqrt 36/36 and reciprocal 32/32 octaves; **XDA-BF16 mean relative error within 1.5–3.2× of the BF16 nearest-even floor** across all 9 functions (a small constant factor above the format's quantization floor) |
| §4 DSP-enabled FPGA synthesis (B-Q1 = D-Q3; D-W6) | `fpga_dsp_enabled/{design}_{dsp,nodsp}_{util,timing}.rpt` | CLB LUTs 881 (XDA) / 1,829 (NLI) / 801 (NN-LUT16); FFs 136/287/306; DSPs 1/2/1; Fmax = 1/(2.0 ns − WNS) → 156.6/102.7/109.3 MHz; nodsp reruns reproduce Table 1 within ±2 LUTs |
| §6 rsqrt distribution-free ε-bound (B-Q4 = D-Q4; D-W1) | `rsqrt_eps_sweep/rsqrt_eps_sweep.py`, `rsqrt_eps_sweep/rsqrt_eps_sweep_results.json` | exhaustive max relative error over every FP16 value ≥ ε: XDA 50.0 vs. NLI 7,002 (×10⁻⁴) at ε = 10⁻⁵ (140×); 1,180 vs. 9,046 at 10⁻⁶ |
| §6(iii)(iv) task-level impact (D-Q4) | `llm_task_eval/results/table1_*.json` (25 = 5 models × {baseline, XDA, NLI, NN-LUT16, NN-LUT256}), `llm_task_eval/derive_task_mad.py`, `llm_task_eval/subnormal_profile.log` | XDA 0.30 pp / NLI 0.36 pp mean absolute deviation vs. the FP16 baseline over the 40 zero-shot task–model pairs; largest XDA deviation 1.66 pp (WinoGrande, Qwen2.5-7B); rsqrt-input subnormal profiling: 3,472 of 2.1M values (0.16%), first layer of Llama-3.1-8B only, zero in the other three models |
| §7 NN-LUT256 row (D-W4) | `nnlut256/fpga_nnlut256_{util,timing,power}.rpt`, `nnlut256/asic_nnlut256_sram_6_report.json` | FPGA 6,814 CLB LUTs / 4,130 FFs / 96.2 MHz (= 1/(2.0 ns − WNS), WNS = −8.397 ns in `fpga_nnlut256_timing.rpt`) / 99 mW dynamic; ASIC total area 62,116 µm² |

Notes:
- **`asic_power/vcd_work/run_vcd_power.sh` regenerates the §1 ASIC power numbers** (the
  reported switching-based estimate): Icarus Verilog gate-level simulation of each post-route
  `6_final.v` (behavioral fakeram + Nangate functional `nangate_cells.v`) under the same FP16
  domain-sweep stimulus as the FPGA SAIF run → `eda.vcd`/`nli.vcd` → OpenROAD `read_vcd` +
  `report_power_vcd.tcl` (iso-10 ns) → saved `pwr_{eda_nli_engine_4s,nli_engine}_vcd.log`
  (NLI 2.23 / XDA 1.52 mW total). Needs `ORFS` with the designs built post-route (as below).
- `asic_power/run_power.sh` regenerates the vectorless cross-check logs (assumption-free; the
  §1 headline number is the measured VCD above): it runs
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
- The §6(iii)(iv) task-level numbers are the paper's printed values (abstract,
  Sec. 4.4) that the rebuttal cites; they are included here for direct inspection.
  The JSONs under `llm_task_eval/results/` are the canonical lm-eval outputs the
  printed Table 2 was generated from (regeneration flow: `sw/llm_eval.py`, see the
  top-level README); `derive_task_mad.py` recomputes 0.30/0.36/1.66 pp from the
  shipped JSONs alone, and the NN-LUT256 rows also back §7's "comparable LLM
  quality" reference. `subnormal_profile.log` is a rerun of `sw/subnormal_hitrate.py`
  as shipped; the subnormal count reproduces the paper's Sec. 4.4 value exactly
  (3,472, all in Llama-3.1-8B's first pre-attention RMSNorm; per-model totals sum
  to 2,144,792 ≈ 2.1M values).
- The rebuttal's LUT-as-distributed-RAM comparison (XDA 176 vs. NLI 210) uses Vivado's
  "LUT as Distributed RAM" subcategory; the parent "LUT as Memory" row additionally
  includes shift-register LUTs (XDA 178 = 176 + 2 SRL; NLI 210 = 210 + 0 — see
  `fpga_dsp_enabled/*_nodsp_util.rpt`). Either level yields the same conclusion.
