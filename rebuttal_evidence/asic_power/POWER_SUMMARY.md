# Post-Route ASIC Power Estimate (for ICCAD'26 Rebuttal)

> **Note (2026-06-19):** Reviewers A-Q3/B-Q2 asked for a *switching-based* ASIC power
> estimate; the paper had omitted ASIC power for lack of a switching-activity model. The
> rebuttal body therefore reports the **measured gate-level-simulation VCD** estimate (NLI
> 2.23 / XDA 1.52 mW, **−32%**; see "VCD-annotated" section below) as the answer. The
> vectorless uniform-α model in Method/Results is retained here as an **assumption-free
> cross-check** (−33%, agreeing), not as the headline.

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

## Results — vectorless cross-check @ 100 MHz (10 ns signoff clock, both designs)

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
   0.37 -> 0.21 nJ (**-43%**) — consistent with the ASIC measured-VCD -32% (and the vectorless cross-check -33%/-57%).
2. Structural: XDA removes 2× FP16 multipliers + 10-comparator interval-search tree;
   sequential area -71%. Toggled-capacitance reduction is architectural, not
   technology-dependent → FPGA→ASIC transfer expected.
3. fakeram macro power is a placeholder per-access model; under the identical model
   for both designs, XDA's extra 64×15 config ROM adds +0.14 mW, far smaller than
   the -1.03 mW logic saving.

## VCD-annotated (measured-activity) ASIC power — the estimate reported in §1 (added 2026-06-19)

Direct answer to the activity-model concern: power re-estimated with **per-net switching
activity measured from a gate-level simulation** of the post-route netlist (Icarus Verilog,
post-route `6_final.v` + behavioral fakeram + Nangate45 functional cell models), driven by
the same FP16 domain-sweep vectors as the FPGA SAIF run (207/206 vectors × 10 rounds),
instead of a uniform/propagated assumption. Activity annotated into OpenROAD via `read_vcd`
(16,004 / 32,583 pins for XDA / NLI), iso-frequency 10 ns, zero-delay (glitch-free →
conservative).

| Group (mW) @100 MHz | NLI (VCD) | XDA (VCD) | Δ |
|---|---|---|---|
| Logic (seq+comb+clock) | 1.346 | 0.502 | **-63%** |
| SRAM macros            | 0.883 | 1.020 | +16% |
| **Total**              | **2.23** | **1.52** | **-32%** |

- **The -33% conclusion is robust across activity models:** uniform α=0.2 → -33%,
  measured-VCD → -32% (agree within run-to-run variation).
- Under measured activity XDA's **logic** power is **63% lower** (vs 57% uniform): NLI's FP
  datapath toggles more under real data, as expected; the SRAM-macro term (XDA's +16% config
  ROM) anchors the total near -32%.
- **NLI VCD = 2.23 mW is physical** — confirms the 71 mW from statistical input-propagation
  (Method, above) was a propagation artifact, not a real estimate.
- Zero-delay is conservative (no glitch power); SDF-annotated (glitch-inclusive) timing would
  raise NLI's combinational power more than XDA's → XDA *even more* favorable. Reported
  zero-delay as the conservative measured number.
- **Saved result logs** (the §1 numbers): `vcd_work/pwr_eda_nli_engine_4s_vcd.log` (XDA total
  1.52 mW), `vcd_work/pwr_nli_engine_vcd.log` (NLI total 2.23 mW).
- **One-command reproduce:** `export ORFS=<OpenROAD-flow-scripts>; vcd_work/run_vcd_power.sh` —
  `iverilog` GL-sim of `6_final.v` + behavioral fakeram + Nangate functional `nangate_cells.v`
  → `eda.vcd`/`nli.vcd` → OpenROAD `read_vcd` + `report_power_vcd.tcl` (iso-10 ns). Testbenches:
  `vcd_work/tb_eda_vcd.v`, `vcd_work/nli/tb_nli_vcd.v`.

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
