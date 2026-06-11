`timescale 1ns/1ps

module nn_lut_engine_16 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        i_valid,
    input  wire [15:0] i_data,
    output wire        o_valid,
    output wire [15:0] o_data,
    input  wire        cfg_we,
    input  wire [1:0]  cfg_sel,
    input  wire [8:0]  cfg_addr,
    input  wire [15:0] cfg_wdata
);

    nn_lut_engine #(.N_SEGMENTS(16), .N_BP(15)) u_eng (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(i_valid),
        .i_data(i_data),
        .o_valid(o_valid),
        .o_data(o_data),
        .cfg_we(cfg_we),
        .cfg_sel(cfg_sel),
        .cfg_addr(cfg_addr),
        .cfg_wdata(cfg_wdata)
    );

endmodule
