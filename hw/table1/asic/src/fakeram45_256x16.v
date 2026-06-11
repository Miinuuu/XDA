// Black-box stub for fakeram45_256x16 (Nangate45 FakeRAM)
// Single-port SRAM: 256 words × 16 bits, 1RW
// Actual timing/power/area from .lib/.lef

module fakeram45_256x16(
    input         clk,
    input         ce_in,
    input         we_in,
    input  [7:0]  addr_in,
    input  [15:0] wd_in,
    input  [15:0] w_mask_in,
    output [15:0] rd_out
);
endmodule
