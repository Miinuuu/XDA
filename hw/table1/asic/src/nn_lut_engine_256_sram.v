//==============================================================================
// nn_lut_engine_256_sram - NN-LUT Engine (256 segments) with SRAM-backed S+T
//
// 3-stage pipeline (vs original 2-stage):
//   Stage 1: Comparator chain (255 FP16 >= comparisons) → cmp_index
//            Present cmp_index to slope/intercept SRAMs
//   Stage 2: SRAM output ready → feed to MAC (s_i · x)
//   Stage 3: MAC complete → y = s_i · x + t_i
//
// Memory:
//   D (breakpoints): FF (flat wide register, 255 × 16 = 4080 bits)
//   S (slopes):      1× fakeram45_256x16
//   T (intercepts):  1× fakeram45_256x16
//
// Throughput: 1 result/cycle (pipelined). Latency: 3 cycles.
//==============================================================================

`timescale 1ns/1ps

module nn_lut_engine_256 #(
    parameter N_SEGMENTS = 256,
    parameter N_BP       = 255,
    parameter IDX_BITS   = 8
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        i_valid,
    input  wire [15:0] i_data,
    output reg         o_valid,
    output reg  [15:0] o_data,

    input  wire        cfg_we,
    input  wire [1:0]  cfg_sel,    // 0: breakpoints, 1: slopes, 2: intercepts
    input  wire [8:0]  cfg_addr,
    input  wire [15:0] cfg_wdata
);

    localparam EXP_W  = 5;
    localparam MANT_W = 10;
    localparam BIAS   = 15;

    //==========================================================================
    // D: Comparator breakpoints — flat wide register (guaranteed FF)
    //==========================================================================
    reg [N_BP*16-1:0] bp_flat;

    wire [15:0] bp [0:N_BP-1];
    genvar gi;
    generate for (gi = 0; gi < N_BP; gi = gi + 1) begin : bp_slice
        assign bp[gi] = bp_flat[gi*16 +: 16];
    end endgenerate

    //==========================================================================
    // S+T: SRAM-backed slope and intercept tables
    //==========================================================================
    wire cfg_we_slope = cfg_we && (cfg_sel == 2'd1);
    wire cfg_we_inter = cfg_we && (cfg_sel == 2'd2);

    // Config write to breakpoints
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 2'd0)
            bp_flat[cfg_addr[IDX_BITS-1:0]*16 +: 16] <= cfg_wdata;
    end

    //==========================================================================
    // FP16 comparison: returns 1 if a >= b
    //==========================================================================
    function automatic fp16_ge;
        input [15:0] a, b;
        reg a_s, b_s;
        reg [14:0] a_mag, b_mag;
        begin
            a_s   = a[15];
            b_s   = b[15];
            a_mag = a[14:0];
            b_mag = b[14:0];
            if (a_s == 0 && b_s == 0)
                fp16_ge = (a_mag >= b_mag);
            else if (a_s == 1 && b_s == 1)
                fp16_ge = (a_mag <= b_mag);
            else if (a_s == 0 && b_s == 1)
                fp16_ge = 1'b1;
            else
                fp16_ge = (a_mag == 0 && b_mag == 0);
        end
    endfunction

    //==========================================================================
    // Stage 1: Comparator Chain → present address to SRAMs
    //==========================================================================
    reg [IDX_BITS-1:0] cmp_index;
    integer ci;
    always @(*) begin
        cmp_index = {IDX_BITS{1'b0}};
        for (ci = 0; ci < N_BP; ci = ci + 1) begin
            if (fp16_ge(i_data, bp[ci]))
                cmp_index = ci[IDX_BITS-1:0] + {{(IDX_BITS-1){1'b0}}, 1'b1};
        end
    end

    // SRAM address mux: config write vs inference read
    wire [7:0] slope_addr = cfg_we_slope ? cfg_addr[7:0] : cmp_index;
    wire [7:0] inter_addr = cfg_we_inter ? cfg_addr[7:0] : cmp_index;

    wire [15:0] slope_rd, inter_rd;

    fakeram45_256x16 u_slope_sram (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (cfg_we_slope),
        .addr_in  (slope_addr),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (slope_rd)
    );

    fakeram45_256x16 u_inter_sram (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (cfg_we_inter),
        .addr_in  (inter_addr),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (inter_rd)
    );

    // Pipeline Register: Stage 1 → Stage 2
    reg        p1_valid;
    reg [15:0] p1_x;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid <= i_valid;
            p1_x     <= i_data;
        end
    end

    //==========================================================================
    // Stage 2: SRAM output ready → MAC (s_i · x)
    //==========================================================================
    wire [15:0] p2_slope     = slope_rd;
    wire [15:0] p2_intercept = inter_rd;

    wire        mul_sign, mul_isZero, mul_isInf, mul_isNaN;
    wire [4:0]  mul_exp;
    wire [9:0]  mul_mant;

    fp_mult_norm #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_mul (
        .io_a_sign (p2_slope[15]),
        .io_a_exp  (p2_slope[14:10]),
        .io_a_mant (p2_slope[9:0]),
        .io_b_sign (p1_x[15]),
        .io_b_exp  (p1_x[14:10]),
        .io_b_mant (p1_x[9:0]),
        .io_out_sign  (mul_sign),
        .io_out_exp   (mul_exp),
        .io_out_mant  (mul_mant),
        .io_out_isZero(mul_isZero),
        .io_out_isInf (mul_isInf),
        .io_out_isNaN (mul_isNaN)
    );

    wire [15:0] product = mul_isZero ? 16'h0000 :
                           {mul_sign, mul_exp, mul_mant};

    wire        add_sign, add_isZero, add_isInf, add_isNaN;
    wire [4:0]  add_exp;
    wire [9:0]  add_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_add (
        .io_a_sign (product[15]),
        .io_a_exp  (product[14:10]),
        .io_a_mant (product[9:0]),
        .io_b_sign (p2_intercept[15]),
        .io_b_exp  (p2_intercept[14:10]),
        .io_b_mant (p2_intercept[9:0]),
        .io_out_sign  (add_sign),
        .io_out_exp   (add_exp),
        .io_out_mant  (add_mant),
        .io_out_isZero(add_isZero),
        .io_out_isInf (add_isInf),
        .io_out_isNaN (add_isNaN)
    );

    wire [15:0] result = add_isZero ? 16'h0000 :
                          {add_sign, add_exp, add_mant};

    // Pipeline Register: Stage 2 → Stage 3 (output)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            o_data  <= 16'h0000;
        end else begin
            o_valid <= p1_valid;
            o_data  <= p1_valid ? result : 16'h0000;
        end
    end

    // Pipeline latency: 3 clock cycles

endmodule
