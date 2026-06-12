# Post-Route ASIC Power Estimate (for ICCAD'26 Rebuttal)

Date: 2026-06-11
Tool: OpenROAD 26Q1 (OpenSTA 3.0) `report_power`
Inputs: post-route `6_final.odb` + `6_final.spef` (parasitic-annotated), signoff SDC (10 ns clock, both designs)
Liberty: NangateOpenCellLibrary_typical.lib + fakeram45 macro libs

## Method

Vectorless, uniform-activity model: `set_power_activity -global -activity 0.2 -duty 0.5`
(α = 0.2 transitions/cycle on **all** nets, identical assumption for every design).

Rationale: statistical input-propagation is unusable here — XDA's datapath sits behind
SRAM macros (activity propagation stops at macro boundaries → underestimate), while
NLI's comparator tree is fed directly by inputs (XOR-tree propagation explodes without
correlation → NLI 71 mW, clearly unphysical). The uniform model weights every design's
post-route switched capacitance + liberty internal energy identically, with no
propagation bias. It is conservative for XDA: in reality comparator trees glitch more
than wires, which would favor XDA further.

## Results @ 100 MHz (10 ns signoff clock, both designs)

| Group (mW)             | NLI   | XDA   | Δ |
|------------------------|-------|-------|---|
| Sequential             | 0.365 | 0.107 | -71% |
| Combinational          | 1.190 | 0.562 | -53% |
| Clock network          | 0.241 | 0.101 | -58% |
| **Logic subtotal**     | **1.796** | **0.770** | **-57%** |
| SRAM macros (fakeram)  | 0.887 | 1.030 | +16% (2 -> 3 macros) |
| **Total**              | **2.683** | **1.800** | **-33%** |

- Leakage included in groups: NLI 0.623 mW vs XDA 0.524 mW total leakage.
- Dynamic-only (total minus leakage): 2.06 -> 1.28 mW = **-38%**.
- NN-LUT16 (no SRAM): 1.77 mW total (FF-bank based).
- Energy/op at iso-frequency (II=1): same ratios (-33% total, -57% logic).
- Sensitivity: ratio stable across α ∈ [0.1, 0.4] → total -30% to -35% (linear dynamic).

## Cross-checks

1. FPGA SAIF (measured via gate-level simulation; domain-sweep test vectors,
   linspace(200)+specials per gen_eda_mem.py): dynamic 37 -> 31 mW, energy/op
   0.37 -> 0.21 nJ (**-43%**) — consistent with ASIC vectorless -33%/-57%.
2. Structural: XDA removes 2× FP16 multipliers + 10-comparator interval-search tree;
   sequential area -71%. Toggled-capacitance reduction is architectural, not
   technology-dependent → FPGA→ASIC transfer expected.
3. fakeram macro power is a placeholder per-access model; under the identical model
   for both designs, XDA's extra 64×15 config ROM adds +0.14 mW, far smaller than
   the -1.03 mW logic saving.

## Reproduce

```bash
cd results/hw/table1/asic/power
for d in nli_engine eda_nli_engine_4s nn_lut_engine_16; do
  PWR_DESIGN=$d PWR_MODE=uniform \
  $ORFS/tools/install/OpenROAD/bin/openroad -no_init -exit report_power_v2.tcl
done
```

Raw logs: `pwr_<design>_uniform.log` (adopted), `pwr_<design>_annotated.log`
(propagated-activity variant, rejected — see Method).
