# Encrypted RTL

All synthesizable RTL under `hw/` (`*.v` / `*.sv`, excluding testbenches and
fakeram behavioral models) is protected with the Vivado IEEE-1735 (v2)
encryption flow using the Xilinx public key (`xilinxt_2021_07`). File names and
module interfaces are unchanged, so the existing Makefiles and Tcl scripts work
as-is.

Usage rights granted by the key block:

| Activity                                  | Allowed |
|-------------------------------------------|---------|
| Vivado synthesis / implementation         | yes     |
| Bitstream / xclbin generation             | yes     |
| Vivado simulation (xsim) / funcsim export | yes     |
| Source-level view or modification         | no      |

Consequences:

- The FPGA flows (`hw/table1/fpga`, `hw/eda_u200`) require **Vivado 2024.x or
  newer**. Third-party simulators and synthesis tools cannot decrypt the files.
- The ASIC flow (`hw/table1/asic`, OpenROAD/Yosys) cannot read Xilinx-encrypted
  RTL. Gate-level netlists for the ASIC designs will be provided in
  `hw/table1/asic` (pending).

Source access for collaborators is available on request.
