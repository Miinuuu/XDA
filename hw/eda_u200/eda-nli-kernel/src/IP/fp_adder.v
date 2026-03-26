//==============================================================================
// fp_adder - Generic Floating Point Adder
// Description:
//   Performs: Result = A + B
//   Supports parameterized exponent and mantissa widths.
//   Designed for same-format floating-point addition (e.g., FP32+FP32)
//==============================================================================

`timescale 1ns/1ps

module fp_adder #(
  parameter EXP_WIDTH  = 8,   // FP32: 8, FP16: 5
  parameter MANT_WIDTH = 23,  // FP32: 23, FP16: 10
  parameter BIAS       = 127  // FP32: 127, FP16: 15
) (
  // Input A
  input                      io_a_sign,
  input  [EXP_WIDTH-1:0]     io_a_exp,
  input  [MANT_WIDTH-1:0]    io_a_mant,
  
  // Input B
  input                      io_b_sign,
  input  [EXP_WIDTH-1:0]     io_b_exp,
  input  [MANT_WIDTH-1:0]    io_b_mant,
  
  // Output
  output                     io_out_sign,
  output [EXP_WIDTH-1:0]     io_out_exp,
  output [MANT_WIDTH-1:0]    io_out_mant,
  output                     io_out_isZero,
  output                     io_out_isInf,
  output                     io_out_isNaN
);

  //============================================================================
  // Constants
  //============================================================================
  localparam EXP_MAX = (1 << EXP_WIDTH) - 1;  // 255 for FP32
  localparam FULL_MANT = MANT_WIDTH + 1;      // Including hidden bit
  localparam GUARD_BITS = 3;                   // Guard, Round, Sticky
  localparam WORK_WIDTH = FULL_MANT + GUARD_BITS + 1;  // Extra bit for overflow

  //============================================================================
  // Input Classification
  //============================================================================
  wire a_isZero = (io_a_exp == 0) && (io_a_mant == 0);
  wire a_isInf  = (io_a_exp == EXP_MAX) && (io_a_mant == 0);
  wire a_isNaN  = (io_a_exp == EXP_MAX) && (io_a_mant != 0);
  wire a_isDenorm = (io_a_exp == 0) && (io_a_mant != 0);
  
  wire b_isZero = (io_b_exp == 0) && (io_b_mant == 0);
  wire b_isInf  = (io_b_exp == EXP_MAX) && (io_b_mant == 0);
  wire b_isNaN  = (io_b_exp == EXP_MAX) && (io_b_mant != 0);
  wire b_isDenorm = (io_b_exp == 0) && (io_b_mant != 0);

  //============================================================================
  // Expand Mantissa (add hidden bit for normal numbers)
  //============================================================================
  wire [FULL_MANT-1:0] a_full_mant = a_isZero ? {FULL_MANT{1'b0}} : 
                                     a_isDenorm ? {1'b0, io_a_mant} : {1'b1, io_a_mant};
  wire [FULL_MANT-1:0] b_full_mant = b_isZero ? {FULL_MANT{1'b0}} : 
                                     b_isDenorm ? {1'b0, io_b_mant} : {1'b1, io_b_mant};

  // Effective exponent (treat denorm exp as 1)
  wire [EXP_WIDTH-1:0] a_eff_exp = (io_a_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_a_exp;
  wire [EXP_WIDTH-1:0] b_eff_exp = (io_b_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_b_exp;

  //============================================================================
  // Determine larger operand by exponent, then by mantissa
  //============================================================================
  wire a_exp_larger = (a_eff_exp > b_eff_exp);
  wire exp_equal = (a_eff_exp == b_eff_exp);
  wire a_mant_larger = (a_full_mant >= b_full_mant);
  wire swap = !a_exp_larger && (!exp_equal || !a_mant_larger);

  // X = larger, Y = smaller
  wire x_sign = swap ? io_b_sign : io_a_sign;
  wire y_sign = swap ? io_a_sign : io_b_sign;
  wire [EXP_WIDTH-1:0] x_exp = swap ? b_eff_exp : a_eff_exp;
  wire [EXP_WIDTH-1:0] y_exp = swap ? a_eff_exp : b_eff_exp;
  wire [FULL_MANT-1:0] x_mant = swap ? b_full_mant : a_full_mant;
  wire [FULL_MANT-1:0] y_mant = swap ? a_full_mant : b_full_mant;
  wire x_isZero = swap ? b_isZero : a_isZero;
  wire y_isZero = swap ? a_isZero : b_isZero;

  //============================================================================
  // Alignment Shift
  //============================================================================
  wire [EXP_WIDTH-1:0] shift_amt = x_exp - y_exp;
  
  // Extend mantissa with guard bits
  wire [WORK_WIDTH-1:0] x_mant_ext = {1'b0, x_mant, {GUARD_BITS{1'b0}}};
  wire [WORK_WIDTH-1:0] y_mant_ext = {1'b0, y_mant, {GUARD_BITS{1'b0}}};
  
  // Shift Y right by shift_amt
  wire [WORK_WIDTH-1:0] y_shifted = (shift_amt >= WORK_WIDTH) ? {WORK_WIDTH{1'b0}} : (y_mant_ext >> shift_amt);
  
  // Sticky bit from shifted-out bits
  wire [WORK_WIDTH-1:0] sticky_mask = (shift_amt >= WORK_WIDTH) ? y_mant_ext : 
                                       (y_mant_ext & ~({{WORK_WIDTH{1'b1}}} << shift_amt));
  wire sticky = |sticky_mask;
  wire [WORK_WIDTH-1:0] y_aligned = y_isZero ? {WORK_WIDTH{1'b0}} : 
                                     {y_shifted[WORK_WIDTH-1:1], y_shifted[0] | sticky};

  //============================================================================
  // Add or Subtract
  //============================================================================
  wire eff_sub = x_sign ^ y_sign;
  
  wire [WORK_WIDTH:0] sum = eff_sub ? 
                             ({1'b0, x_mant_ext} - {1'b0, y_aligned}) :
                             ({1'b0, x_mant_ext} + {1'b0, y_aligned});
  
  // Result sign: if subtraction caused negative result, flip sign
  wire sum_negative = sum[WORK_WIDTH];
  wire [WORK_WIDTH-1:0] sum_abs = sum_negative ? (~sum[WORK_WIDTH-1:0] + 1) : sum[WORK_WIDTH-1:0];
  wire result_sign = sum_negative ? ~x_sign : x_sign;

  //============================================================================
  // Normalization
  //============================================================================
  // Leading zero count
  reg [$clog2(WORK_WIDTH)-1:0] lzc;
  integer i;
  always @(*) begin
    lzc = 0;
    for (i = WORK_WIDTH - 1; i >= 0; i = i - 1) begin
      if (!sum_abs[i] && lzc == (WORK_WIDTH - 1 - i))
        lzc = lzc + 1;
    end
  end
  
  // Check for addition overflow (MSB set beyond normal position)
  wire add_overflow = sum_abs[WORK_WIDTH-1];
  
  // Normalized mantissa
  wire [WORK_WIDTH-1:0] norm_mant = add_overflow ? (sum_abs >> 1) : 
                                     (sum_abs << (lzc > 0 ? lzc - 1 : 0));
  
  // Adjusted exponent
  wire signed [EXP_WIDTH+1:0] norm_exp = add_overflow ? 
                                          $signed({2'b0, x_exp}) + 1 :
                                          $signed({2'b0, x_exp}) - $signed({{(EXP_WIDTH+2-$clog2(WORK_WIDTH)){1'b0}}, lzc}) + 1;

  //============================================================================
  // Rounding (Round to Nearest Even)
  //============================================================================
  wire [GUARD_BITS-1:0] guard_bits = norm_mant[GUARD_BITS-1:0];
  wire lsb = norm_mant[GUARD_BITS];
  wire round_up = guard_bits[GUARD_BITS-1] && (|guard_bits[GUARD_BITS-2:0] || lsb);
  
  wire [FULL_MANT:0] rounded_mant = {1'b0, norm_mant[WORK_WIDTH-2:GUARD_BITS]} + {{FULL_MANT{1'b0}}, round_up};
  wire round_overflow = rounded_mant[FULL_MANT];
  
  wire signed [EXP_WIDTH+1:0] final_exp = round_overflow ? norm_exp + 1 : norm_exp;
  wire [MANT_WIDTH-1:0] final_mant = round_overflow ? rounded_mant[FULL_MANT-1:1] : rounded_mant[MANT_WIDTH-1:0];

  //============================================================================
  // Special Cases
  //============================================================================
  wire result_isNaN = a_isNaN || b_isNaN || (a_isInf && b_isInf && (io_a_sign != io_b_sign));
  wire result_isInf = ((a_isInf || b_isInf) && !result_isNaN) || (final_exp >= EXP_MAX);
  wire sum_is_zero = (sum_abs == 0);
  wire result_isZero = (a_isZero && b_isZero) || (sum_is_zero && !a_isInf && !b_isInf && !a_isNaN && !b_isNaN);
  wire underflow = (final_exp <= 0) && !result_isZero && !result_isInf && !result_isNaN;

  //============================================================================
  // Output
  //============================================================================
  assign io_out_isZero = result_isZero;
  assign io_out_isInf = result_isInf;
  assign io_out_isNaN = result_isNaN;
  
  assign io_out_sign = result_isNaN ? 1'b0 :
                       result_isZero ? (io_a_sign && io_b_sign) :
                       result_isInf ? (a_isInf ? io_a_sign : io_b_sign) :
                       result_sign;
  
  assign io_out_exp = result_isZero ? {EXP_WIDTH{1'b0}} :
                      (result_isInf || result_isNaN) ? {EXP_WIDTH{1'b1}} :
                      underflow ? {EXP_WIDTH{1'b0}} :
                      final_exp[EXP_WIDTH-1:0];
  
  assign io_out_mant = result_isZero ? {MANT_WIDTH{1'b0}} :
                       result_isInf ? {MANT_WIDTH{1'b0}} :
                       result_isNaN ? {{MANT_WIDTH-1{1'b0}}, 1'b1} :
                       underflow ? {MANT_WIDTH{1'b0}} :
                       final_mant;

endmodule
