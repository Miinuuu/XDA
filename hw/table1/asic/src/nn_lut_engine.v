//==============================================================================
// nn_lut_engine - 2-Stage Pipelined NN-LUT Engine (FP16)
//
// ASIC variant:
//   - breakpoint_reg: flat wide register (not memory array) → guaranteed FF
//   - slope+intercept: merged si_lut array → SRAM via fakeram mock
//
// Architecture (NN-LUT paper, Yu et al. DAC 2022):
//   Stage 1: Comparator Chain (FF) + LUT Read (SRAM)
//   Stage 2: 1-MAC  y = s_i · x + t_i
//==============================================================================

`timescale 1ns/1ps

module nn_lut_engine #(
    parameter N_SEGMENTS   = 16,
    parameter N_BP         = 15,
    parameter IDX_BITS     = $clog2(N_SEGMENTS)
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
    // Comparator breakpoints: flat wide register (NOT a memory array)
    // Declared as single vector so Yosys treats it as FFs, not $mem.
    // N_BP breakpoints × 16 bits each.
    //==========================================================================
    reg [N_BP*16-1:0] bp_flat;

    // Combinational view as 16-bit slices
    wire [15:0] bp [0:N_BP-1];
    genvar gi;
    generate for (gi = 0; gi < N_BP; gi = gi + 1) begin : bp_slice
        assign bp[gi] = bp_flat[gi*16 +: 16];
    end endgenerate

    //==========================================================================
    // Slope + Intercept LUT: merged for SRAM mapping
    //   [0 .. N_SEGMENTS-1]            = slopes     s_0..s_{N-1}
    //   [N_SEGMENTS .. 2*N_SEGMENTS-1] = intercepts t_0..t_{N-1}
    //==========================================================================
    reg [15:0] si_lut [0:2*N_SEGMENTS-1];

    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'd0: bp_flat[cfg_addr[IDX_BITS-1:0]*16 +: 16] <= cfg_wdata;
                2'd1: si_lut[cfg_addr[IDX_BITS-1:0]]            <= cfg_wdata;
                2'd2: si_lut[N_SEGMENTS + cfg_addr[IDX_BITS-1:0]] <= cfg_wdata;
                default: ;
            endcase
        end
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
    // Stage 1: Comparator Chain + LUT Read
    //==========================================================================
    reg  [IDX_BITS-1:0] cmp_index;
    integer ci;
    always @(*) begin
        cmp_index = {IDX_BITS{1'b0}};
        for (ci = 0; ci < N_BP; ci = ci + 1) begin
            if (fp16_ge(i_data, bp[ci]))
                cmp_index = ci[IDX_BITS-1:0] + {{(IDX_BITS-1){1'b0}}, 1'b1};
        end
    end

    wire [15:0] s1_slope     = si_lut[cmp_index];
    wire [15:0] s1_intercept = si_lut[N_SEGMENTS + cmp_index];

    //==========================================================================
    // Pipeline Register: Stage 1 → Stage 2
    //==========================================================================
    reg                 p1_valid;
    reg [15:0]          p1_x;
    reg [15:0]          p1_slope;
    reg [15:0]          p1_intercept;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid     <= i_valid;
            p1_x         <= i_data;
            p1_slope     <= s1_slope;
            p1_intercept <= s1_intercept;
        end
    end

    //==========================================================================
    // Stage 2: 1-MAC  y = s_i · x + t_i
    //==========================================================================
    wire        mul_sign, mul_isZero, mul_isInf, mul_isNaN;
    wire [4:0]  mul_exp;
    wire [9:0]  mul_mant;

    fp_mult_norm #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_mul (
        .io_a_sign (p1_slope[15]),
        .io_a_exp  (p1_slope[14:10]),
        .io_a_mant (p1_slope[9:0]),
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
        .io_b_sign (p1_intercept[15]),
        .io_b_exp  (p1_intercept[14:10]),
        .io_b_mant (p1_intercept[9:0]),
        .io_out_sign  (add_sign),
        .io_out_exp   (add_exp),
        .io_out_mant  (add_mant),
        .io_out_isZero(add_isZero),
        .io_out_isInf (add_isInf),
        .io_out_isNaN (add_isNaN)
    );

    wire [15:0] result = add_isZero ? 16'h0000 :
                          {add_sign, add_exp, add_mant};

    //==========================================================================
    // Output Register
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            o_data  <= 16'h0000;
        end else begin
            o_valid <= p1_valid;
            o_data  <= p1_valid ? result : 16'h0000;
        end
    end

endmodule
