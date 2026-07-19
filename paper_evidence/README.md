# Camera-Ready Paper Evidence

This directory contains pre-generated outputs backing quantitative results in
the camera-ready XDA paper. The corresponding regeneration flows live under
`hw/` and `sw/` (see the top-level README), so the reported values can be
inspected without rerunning the longest hardware and LLM experiments.

Vivado/OpenSTA report headers are anonymized (`<RUN_DIR>`, `Host`); report
bodies are otherwise unmodified tool output. Files prefixed with `eda_` refer
to XDA, the design's original internal name.

| Paper location | Evidence files | Key results |
|---|---|---|
| Sec. 4.2, ASIC power | `asic_power/POWER_SUMMARY.md`, `asic_power/vcd_work/pwr_{eda_nli_engine_4s,nli_engine}_vcd.log`, `asic_power/vcd_work/run_vcd_power.sh`; vectorless cross-check: `asic_power/pwr_*_uniform.log`, `asic_power/report_power_v2.tcl` | Gate-level VCD estimate: NLI 2.23 -> XDA 1.52 mW total (-32%); logic -63%. Uniform-activity cross-check: 2.683 -> 1.800 mW (-33%). |
| Sec. 4.2, area breakdown | `sram_breakdown/asic_{xda,nli}_6_report.json` | Sequential cells: NLI 2,296 vs. XDA 668 um2; XDA's additional 659 um2 is one 64x15 configuration ROM. |
| Sec. 3.1, BF16 extension | `bf16/exp_bf16_preliminary.py`, `bf16/bf16_preliminary_out.log`, `bf16/exp_bf16_floor.py`, `bf16/bf16_floor_out.log` | A 512-entry budget covers all nine functions; XDA-BF16 mean relative error is within 1.5--3.2x of the BF16 nearest-even floor. |
| Sec. 4.2, DSP-enabled FPGA | `fpga_dsp_enabled/{design}_{dsp,nodsp}_{util,timing}.rpt` | XDA/NLI/NN-LUT16 use 881/1,829/801 CLB LUTs and 1/2/1 DSPs; timing-derived Fmax is 156.6/102.7/109.3 MHz. |
| Table 3(C), rsqrt epsilon bound | `rsqrt_eps_sweep/rsqrt_eps_sweep.py`, `rsqrt_eps_sweep/rsqrt_eps_sweep_results.json` | Exhaustive max relative error at epsilon=1e-5: XDA 50.0 vs. NLI 7,002 (x1e-4). |
| Secs. 4.3--4.4, LLM/task impact | `llm_task_eval/results/table1_*.json`, `llm_task_eval/derive_task_mad.py`, `llm_task_eval/subnormal_profile.log` | XDA/NLI deviations are 0.30/0.36 pp over 40 zero-shot task-model pairs; 3,472 of 2.1M profiled rsqrt inputs are subnormal. |
| Sec. 4.4, subnormal isolation | `subnormal_task_ablation/SUMMARY.md`, `subnormal_task_ablation/*.json` | Replacing XDA only in the FP16-subnormal rsqrt range changes the eight-task mean by 0.13 pp. |
| Tables 1--2, NN-LUT256 | `nnlut256/fpga_nnlut256_{util,timing,power}.rpt`, `nnlut256/asic_nnlut256_sram_6_report.json`, `llm_task_eval/results/*_nn_lut_256.json` | FPGA: 6,814 CLB LUTs, 4,130 FFs, 96.2 MHz, 99 mW dynamic; ASIC total area: 62,116 um2. |

## Reproduction Notes

- `asic_power/vcd_work/run_vcd_power.sh` regenerates the switching-based ASIC
  power results using Icarus Verilog gate-level simulation and OpenROAD
  `read_vcd`. `asic_power/run_power.sh` regenerates the uniform-activity
  cross-check. Both require post-route OpenROAD-flow-scripts results under
  `$ORFS/flow/results/nangate45/<design>/base`.
- `rsqrt_eps_sweep/rsqrt_eps_sweep.py` regenerates the canonical JSON on GPU.
  A CPU run writes a device-suffixed file because near-tied Knapsack allocations
  can differ slightly across devices.
- The BF16 W=512 logs contain the per-function octave counts and the
  1.5--3.2x error-floor comparison quoted in the paper.
- `llm_task_eval/derive_task_mad.py` is standard-library-only and recomputes
  the 0.30/0.36/1.66 pp statistics directly from the 25 shipped lm-eval JSONs.
- `subnormal_profile.log` was produced by `sw/subnormal_hitrate.py`; its
  per-model totals sum to 2,144,792 profiled values.
- Vivado's `LUT as Distributed RAM` row is the source of the XDA/NLI 176/210
  comparison. The parent `LUT as Memory` row additionally includes shift-register
  LUTs; both levels give the same conclusion.
