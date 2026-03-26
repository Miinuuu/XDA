//==============================================================================
// fp_mult_norm - Floating Point Multiplier with proper subnormal normalization
//
// Based on HW/fpu/fp_mult.v, with fix: lzc is now used for normalization
// so that subnormal × normal products are correctly normalized.
//
// Original bug: normalization only handled overflow (lzc=0) and normal (lzc=1)
// but not subnormal inputs (lzc>1) where additional left-shift is needed.
//==============================================================================

`timescale 1ns/1ps

module fp_mult_norm #(
  parameter EXP_WIDTH  = 8,
  parameter MANT_WIDTH = 23,
  parameter BIAS       = 127
) (
  input                      io_a_sign,
  input  [EXP_WIDTH-1:0]     io_a_exp,
  input  [MANT_WIDTH-1:0]    io_a_mant,

  input                      io_b_sign,
  input  [EXP_WIDTH-1:0]     io_b_exp,
  input  [MANT_WIDTH-1:0]    io_b_mant,

  output                     io_out_sign,
  output [EXP_WIDTH-1:0]     io_out_exp,
  output [MANT_WIDTH-1:0]    io_out_mant,
  output                     io_out_isZero,
  output                     io_out_isInf,
  output                     io_out_isNaN
);

  localparam EXP_MAX   = (1 << EXP_WIDTH) - 1;
  localparam FULL_MANT = MANT_WIDTH + 1;
  localparam PROD_WIDTH = FULL_MANT * 2;

  // Input Classification
  wire a_isZero   = (io_a_exp == 0) && (io_a_mant == 0);
  wire a_isInf    = (io_a_exp == EXP_MAX) && (io_a_mant == 0);
  wire a_isNaN    = (io_a_exp == EXP_MAX) && (io_a_mant != 0);
  wire a_isDenorm = (io_a_exp == 0) && (io_a_mant != 0);

  wire b_isZero   = (io_b_exp == 0) && (io_b_mant == 0);
  wire b_isInf    = (io_b_exp == EXP_MAX) && (io_b_mant == 0);
  wire b_isNaN    = (io_b_exp == EXP_MAX) && (io_b_mant != 0);
  wire b_isDenorm = (io_b_exp == 0) && (io_b_mant != 0);

  // Sign
  wire result_sign = io_a_sign ^ io_b_sign;

  // Expand Mantissa
  wire [FULL_MANT-1:0] a_full_mant = a_isZero   ? {FULL_MANT{1'b0}} :
                                      a_isDenorm ? {1'b0, io_a_mant} :
                                                   {1'b1, io_a_mant};
  wire [FULL_MANT-1:0] b_full_mant = b_isZero   ? {FULL_MANT{1'b0}} :
                                      b_isDenorm ? {1'b0, io_b_mant} :
                                                   {1'b1, io_b_mant};

  // Mantissa Multiplication
  wire [PROD_WIDTH-1:0] product = a_full_mant * b_full_mant;

  // Exponent Calculation
  wire [EXP_WIDTH-1:0] a_eff_exp = (io_a_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_a_exp;
  wire [EXP_WIDTH-1:0] b_eff_exp = (io_b_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_b_exp;

  wire signed [EXP_WIDTH+1:0] exp_sum = $signed({2'b0, a_eff_exp}) +
                                         $signed({2'b0, b_eff_exp}) -
                                         $signed({{2{1'b0}}, BIAS[EXP_WIDTH-1:0]});

  //============================================================================
  // Leading Zero Detection
  //============================================================================
  reg [$clog2(PROD_WIDTH)-1:0] lzc;
  integer i;
  always @(*) begin
    lzc = 0;
    for (i = PROD_WIDTH - 1; i >= 0; i = i - 1) begin
      if (!product[i] && lzc == (PROD_WIDTH - 1 - i))
        lzc = lzc + 1;
    end
  end

  //============================================================================
  // Normalization (FIX: uses lzc for all cases)
  //
  // After product << lzc, the leading 1 is at bit PROD_WIDTH-1.
  // Exponent: norm_exp = exp_sum + 1 - lzc
  //   lzc=0 (overflow):  norm_exp = exp_sum + 1   (product MSB was at PROD_WIDTH-1)
  //   lzc=1 (normal):    norm_exp = exp_sum        (product MSB was at PROD_WIDTH-2)
  //   lzc=k (subnormal): norm_exp = exp_sum + 1 - k
  //============================================================================
  wire prod_zero = (product == 0);

  wire [PROD_WIDTH-1:0] norm_product = prod_zero ? {PROD_WIDTH{1'b0}} :
                                                    (product << lzc);

  wire signed [EXP_WIDTH+1:0] lzc_ext = {{(EXP_WIDTH+2-$clog2(PROD_WIDTH)){1'b0}}, lzc};
  wire signed [EXP_WIDTH+1:0] norm_exp = prod_zero ? {(EXP_WIDTH+2){1'b0}} :
                                                      (exp_sum + 1 - lzc_ext);

  //============================================================================
  // Rounding (Round to Nearest Even)
  //============================================================================
  localparam MANT_START = PROD_WIDTH - 2;
  localparam MANT_END   = MANT_START - MANT_WIDTH + 1;
  localparam GUARD_POS  = MANT_END - 1;
  localparam ROUND_POS  = MANT_END - 2;

  wire [MANT_WIDTH-1:0] trunc_mant = norm_product[MANT_START:MANT_END];
  wire guard  = (GUARD_POS >= 0) ? norm_product[GUARD_POS] : 1'b0;
  wire round_bit  = (ROUND_POS >= 0) ? norm_product[ROUND_POS] : 1'b0;
  wire sticky = (ROUND_POS >  0) ? |norm_product[ROUND_POS-1:0] : 1'b0;
  wire lsb    = trunc_mant[0];

  wire round_up = guard && (round_bit || sticky || lsb);

  wire [MANT_WIDTH:0] rounded_mant = {1'b0, trunc_mant} + {{MANT_WIDTH{1'b0}}, round_up};
  wire round_overflow = rounded_mant[MANT_WIDTH];

  wire signed [EXP_WIDTH+1:0] final_exp  = round_overflow ? norm_exp + 1 : norm_exp;
  wire [MANT_WIDTH-1:0]       final_mant = round_overflow ? rounded_mant[MANT_WIDTH:1] :
                                                             rounded_mant[MANT_WIDTH-1:0];

  //============================================================================
  // Special Cases
  //============================================================================
  wire result_isNaN  = a_isNaN || b_isNaN ||
                       ((a_isZero && b_isInf) || (a_isInf && b_isZero));
  wire result_isInf  = ((a_isInf || b_isInf) && !result_isNaN) ||
                       (final_exp >= EXP_MAX && !result_isNaN);
  wire result_isZero = (a_isZero || b_isZero) && !result_isNaN;
  wire underflow     = (final_exp <= 0) && !result_isZero && !result_isInf && !result_isNaN;

  //============================================================================
  // Output
  //============================================================================
  assign io_out_isZero = result_isZero || underflow;
  assign io_out_isInf  = result_isInf;
  assign io_out_isNaN  = result_isNaN;

  assign io_out_sign = result_isNaN ? 1'b0 : result_sign;

  assign io_out_exp  = (result_isZero || underflow) ? {EXP_WIDTH{1'b0}} :
                       (result_isInf || result_isNaN) ? {EXP_WIDTH{1'b1}} :
                       final_exp[EXP_WIDTH-1:0];

  assign io_out_mant = (result_isZero || underflow) ? {MANT_WIDTH{1'b0}} :
                       result_isInf ? {MANT_WIDTH{1'b0}} :
                       result_isNaN ? {{MANT_WIDTH-1{1'b0}}, 1'b1} :
                       final_mant;

endmodule
