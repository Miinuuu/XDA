# Reproducing Paper Table 1 (Post-PnR Hardware Comparison)

This directory contains the complete, self-contained flows that produce every
number in Table 1: XDA, NLI, and NN-LUT on (A) DSP-free FPGA and (B)
SRAM-backed ASIC. All RTL is plain Verilog, including the XDA engine
(`fpga/src/eda_nli_engine_4s.v`, `asic/src/eda_nli_engine_4s_sram.v`).

```
fpga/
├── Makefile            # per-design: synth+impl → GL-sim SAIF → power
├── run_one.sh          # single-design flow (called by Makefile)
├── run_saif_all.sh     # all designs
├── dsp_run.sh          # DSP-ENABLED synth/impl variant (util+timing only)
├── nodsp_run.sh        # DSP-free synth/impl variant (cross-check)
├── src/                # RTL + clock.xdc (create_clock 2.0 ns)
├── memgen/             # .mem generators (NLI cutpoints, LUT budgets, vectors)
└── tb_*.sv             # SAIF testbenches
asic/
├── run_asic_all.sh     # OpenROAD-flow-scripts driver (set ORFS_DIR)
├── constraint.sdc      # shared 10 ns signoff SDC
├── configs/<design>/   # per-design ORFS config.mk (relative paths)
├── src/                # SRAM-macro RTL variants + fakeram behavioral models
├── fakeram/            # fakeram45_256x16 / 64x15 .lib/.lef
└── power/              # vectorless post-route power estimate (OpenSTA)
```

## FPGA (Table 1A)

Prerequisites: Vivado on PATH (`source <Vivado>/settings64.sh`), Python 3 +
PyTorch (for `make mem`).

```bash
cd fpga
make eda nli nnlut16          # full flow per design
# reports/: {eda,nli,nnlut16}_{util,timing,power}.rpt
```

Metric mapping:
- CLB LUTs / FFs / LUT-as-memory — `*_util.rpt`
- Fmax = 1 / (2.0 ns − WNS), WNS from `*_timing.rpt` (clock.xdc targets 2.0 ns)
- Dynamic power / Energy-per-op — `*_power.rpt` (SAIF-annotated, ≥94% net match);
  Energy/op = dynamic power / Fmax

DSP-enabled comparison (rebuttal): `./dsp_run.sh` runs the identical flow with
only `-max_dsp 0` removed; `./nodsp_run.sh` is the DSP-free counterpart for a
same-tool-version baseline.

## ASIC (Table 1B)

Prerequisites: [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
built locally; copy the fakeram macros once:

```bash
cp asic/fakeram/*.lib  $ORFS_DIR/flow/platforms/nangate45/lib/
cp asic/fakeram/*.lef  $ORFS_DIR/flow/platforms/nangate45/lef/

ORFS_DIR=/path/to/OpenROAD-flow-scripts ./asic/run_asic_all.sh
```

Logic / SRAM-macro / total area and Fmax come from each design's
`6_report.json` (post-route, fakeram SRAM). The shared `constraint.sdc` pins
the 10 ns signoff clock for all designs.

Vectorless post-route power estimate (uniform-activity, identical assumptions
per design):

```bash
cd asic/power
PWR_DESIGN=eda_nli_engine_4s PWR_MODE=uniform \
  $ORFS_DIR/tools/install/OpenROAD/bin/openroad -no_init -exit report_power_v2.tcl
```

## Notes

- **Tool versions**: the paper's reports were produced with Vivado 2024.1
  (xcu200, `clock.xdc` = 2.0 ns) and OpenROAD-flow-scripts on Nangate45.
  A Vivado 2024.2 rerun reproduces the FPGA utilization within ±2 CLB LUTs
  and identical FF counts, so nearby versions are expected to match closely.
- Pipeline staging and address-arithmetic precision are fixed in the RTL
  (not synthesis-dependent); NLI uses the paper-provided cutpoints emitted by
  `memgen/gen_nli_mem.py`, and per-bin LUT budgets are the generated `.mem`
  files — each item of the reproducibility checklist is an explicit file here.
- NN-LUT₂₅₆ configs (`nn_lut_engine_256*`) are included for the scaling
  comparison quoted in §4.2.
