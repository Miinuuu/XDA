//==============================================================================
// frac_mult - Fractional Shift-Add Multiplier (NO FP16 Multiplier)
//
// Computes: result = fp16_a × (t_int / 2^T_BITS)
//
// t_int is a T_BITS-bit unsigned integer from mantissa bit extraction.
// Uses a small (T_BITS × 11) integer multiply (shift-add tree) instead of
// a full FP16 multiplier (11 × 11 + exponent logic).
//
// Area: ~10-15% of a full FP16 multiplier for T_BITS=5.
//==============================================================================

`timescale 1ns/1ps

module frac_mult #(
    parameter T_BITS    = 5,
    parameter EXP_WIDTH = 5,
    parameter MANT_WIDTH = 10,
    parameter BIAS      = 15
) (
    input  [15:0]          io_a,   // FP16 value (typically diff = y1 - y0)
    input  [T_BITS-1:0]   io_t,   // fractional coefficient (t_int / 2^T_BITS)
    output [15:0]          io_out  // FP16 result = io_a × io_t / 2^T_BITS
);

    localparam EXP_MAX    = (1 << EXP_WIDTH) - 1; // 31
    localparam FULL_MANT  = MANT_WIDTH + 1;        // 11 (with hidden bit)
    localparam PROD_WIDTH = T_BITS + FULL_MANT;    // 16

    //==========================================================================
    // Unpack FP16 input
    //==========================================================================
    wire           a_sign = io_a[15];
    wire [4:0]     a_exp  = io_a[14:10];
    wire [9:0]     a_mant = io_a[9:0];

    wire a_isZero   = (a_exp == 0) && (a_mant == 0);
    wire a_isDenorm = (a_exp == 0) && (a_mant != 0);
    wire a_isInf    = (a_exp == EXP_MAX) && (a_mant == 0);
    wire a_isNaN    = (a_exp == EXP_MAX) && (a_mant != 0);

    wire [FULL_MANT-1:0] a_full = a_isZero   ? {FULL_MANT{1'b0}} :
                                   a_isDenorm ? {1'b0, a_mant} :
                                                {1'b1, a_mant};

    wire [EXP_WIDTH-1:0] a_eff_exp = (a_exp == 0) ? 5'd1 : a_exp;

    wire t_isZero = (io_t == 0);

    //==========================================================================
    // Integer Multiply: t_int × full_mant  (T_BITS × 11 = 16 bits)
    // Synthesizer implements this as shift-add tree (NOT a full multiplier).
    //==========================================================================
    wire [PROD_WIDTH-1:0] product_raw = io_t * a_full;

    //==========================================================================
    // Leading Zero Count on product
    //==========================================================================
    reg [$clog2(PROD_WIDTH)-1:0] lzc;
    integer i;
    always @(*) begin
        lzc = 0;
        for (i = PROD_WIDTH - 1; i >= 0; i = i - 1) begin
            if (!product_raw[i] && lzc == (PROD_WIDTH - 1 - i))
                lzc = lzc + 1;
        end
    end

    //==========================================================================
    // Normalize: shift leading 1 to bit PROD_WIDTH-1
    //
    // Math derivation:
    //   value = sign × 2^(eff_exp - BIAS) × (full_mant/2^MANT) × (t_int/2^T)
    //         = sign × 2^(eff_exp - BIAS) × product_raw / 2^(MANT + T)
    //   Let p = MSB position of product_raw = PROD_WIDTH - 1 - lzc
    //   product_raw = 2^p × (1 + frac)
    //   value = sign × 2^(eff_exp - BIAS + p - MANT - T) × (1 + frac)
    //   Since p = T + MANT + 1 - 1 - lzc = T + MANT - lzc:
    //   result_exp = eff_exp - lzc   (beautifully simple!)
    //==========================================================================
    wire prod_zero = (product_raw == 0);

    wire [PROD_WIDTH-1:0] norm_product = prod_zero ? {PROD_WIDTH{1'b0}} :
                                                      (product_raw << lzc);

    wire signed [EXP_WIDTH+1:0] result_exp_raw =
        $signed({2'b0, a_eff_exp}) -
        $signed({{(EXP_WIDTH+2-$clog2(PROD_WIDTH)){1'b0}}, lzc});

    //==========================================================================
    // Rounding (Round to Nearest Even)
    //==========================================================================
    wire [MANT_WIDTH-1:0] trunc_mant = norm_product[PROD_WIDTH-2 : PROD_WIDTH-1-MANT_WIDTH];

    localparam GUARD_POS = PROD_WIDTH - 2 - MANT_WIDTH; // bit 4 for T=5
    wire guard  = norm_product[GUARD_POS];
    wire round_bit = (GUARD_POS > 0) ? norm_product[GUARD_POS-1] : 1'b0;
    wire sticky = (GUARD_POS > 1) ? |norm_product[GUARD_POS-2:0] : 1'b0;
    wire lsb    = trunc_mant[0];
    wire round_up = guard & (round_bit | sticky | lsb);

    wire [MANT_WIDTH:0] rounded = {1'b0, trunc_mant} + {{MANT_WIDTH{1'b0}}, round_up};
    wire round_ovf = rounded[MANT_WIDTH];

    wire signed [EXP_WIDTH+1:0] final_exp = round_ovf ? result_exp_raw + 1 : result_exp_raw;
    wire [MANT_WIDTH-1:0]       final_mant = round_ovf ? rounded[MANT_WIDTH:1] :
                                                          rounded[MANT_WIDTH-1:0];

    //==========================================================================
    // Special cases + output
    //==========================================================================
    wire result_isZero = a_isZero || t_isZero || prod_zero;
    wire result_isNaN  = a_isNaN;
    wire result_isInf  = a_isInf;
    wire underflow     = (final_exp <= 0) && !result_isZero && !result_isInf && !result_isNaN;
    wire overflow      = (final_exp >= EXP_MAX) && !result_isInf && !result_isNaN;

    wire       out_sign = result_isNaN ? 1'b0 : a_sign; // t always positive
    wire [4:0] out_exp  = (result_isZero || underflow) ? 5'd0 :
                           (result_isInf || overflow || result_isNaN) ? 5'h1F :
                           final_exp[4:0];
    wire [9:0] out_mant = (result_isZero || underflow) ? 10'd0 :
                           (result_isInf || overflow) ? 10'd0 :
                           result_isNaN ? 10'h200 :
                           final_mant;

    assign io_out = {out_sign, out_exp, out_mant};

endmodule
