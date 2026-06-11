// Black-box stub for fakeram45_64x15 (Nangate45 FakeRAM)
// Single-port SRAM: 64 words × 15 bits, 1RW

module fakeram45_64x15(
    input         clk,
    input         ce_in,
    input         we_in,
    input  [5:0]  addr_in,
    input  [14:0] wd_in,
    input  [14:0] w_mask_in,
    output [14:0] rd_out
);
endmodule
