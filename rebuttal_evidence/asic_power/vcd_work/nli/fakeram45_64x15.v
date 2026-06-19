`timescale 1ns/1ps
// Behavioral simulation model for fakeram45_64x15 (1RW synchronous SRAM).
// Replaces the synthesis black-box stub for gate-level VCD activity capture.
// Convention (matches XDA RTL usage): ce_in=1 always; we_in=1 → masked write,
// we_in=0 → synchronous read; w_mask_in[j]=1 writes bit j (RTL drives all-ones).
module fakeram45_64x15(
    input             clk,
    input             ce_in,
    input             we_in,
    input      [5:0]  addr_in,
    input      [14:0] wd_in,
    input      [14:0] w_mask_in,
    output reg [14:0] rd_out
);
    reg [14:0] mem [0:63];
    integer j;
    always @(posedge clk) begin
        if (ce_in) begin
            if (we_in) begin
                for (j = 0; j < 15; j = j + 1)
                    if (w_mask_in[j]) mem[addr_in][j] <= wd_in[j];
            end else begin
                rd_out <= mem[addr_in];
            end
        end
    end
endmodule
