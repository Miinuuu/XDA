//==============================================================================
// fp_adder_2s - 2-Stage Pipelined Floating Point Adder
//
// Same function as fp_adder.v but split into 2 pipeline stages for Fmax:
//   Stage 1: Classify + Swap + Align + Add/Subtract
//   Stage 2: Normalize (LZC) + Round + Special Cases
//
// Latency: 2 clock cycles (output valid 2 cycles after input)
//==============================================================================

`timescale 1ns/1ps

module fp_adder_2s #(
  parameter EXP_WIDTH  = 8,
  parameter MANT_WIDTH = 23,
  parameter BIAS       = 127
) (
  input  wire                   clk,
  input  wire                   rst_n,

  // Input A
  input  wire                   io_a_sign,
  input  wire [EXP_WIDTH-1:0]   io_a_exp,
  input  wire [MANT_WIDTH-1:0]  io_a_mant,

  // Input B
  input  wire                   io_b_sign,
  input  wire [EXP_WIDTH-1:0]   io_b_exp,
  input  wire [MANT_WIDTH-1:0]  io_b_mant,

  // Output (valid 2 cycles after input)
  output wire                   io_out_sign,
  output wire [EXP_WIDTH-1:0]   io_out_exp,
  output wire [MANT_WIDTH-1:0]  io_out_mant,
  output wire                   io_out_isZero,
  output wire                   io_out_isInf,
  output wire                   io_out_isNaN
);

  localparam EXP_MAX    = (1 << EXP_WIDTH) - 1;
  localparam FULL_MANT  = MANT_WIDTH + 1;
  localparam GUARD_BITS = 3;
  localparam WORK_WIDTH = FULL_MANT + GUARD_BITS + 1;

  //==========================================================================
  // Stage 1: Classify + Swap + Align + Add/Subtract
  //==========================================================================

  // Input classification
  wire a_isZero   = (io_a_exp == 0) && (io_a_mant == 0);
  wire a_isInf    = (io_a_exp == EXP_MAX) && (io_a_mant == 0);
  wire a_isNaN    = (io_a_exp == EXP_MAX) && (io_a_mant != 0);
  wire a_isDenorm = (io_a_exp == 0) && (io_a_mant != 0);

  wire b_isZero   = (io_b_exp == 0) && (io_b_mant == 0);
  wire b_isInf    = (io_b_exp == EXP_MAX) && (io_b_mant == 0);
  wire b_isNaN    = (io_b_exp == EXP_MAX) && (io_b_mant != 0);
  wire b_isDenorm = (io_b_exp == 0) && (io_b_mant != 0);

  // Expand mantissa
  wire [FULL_MANT-1:0] a_full_mant = a_isZero   ? {FULL_MANT{1'b0}} :
                                      a_isDenorm ? {1'b0, io_a_mant} : {1'b1, io_a_mant};
  wire [FULL_MANT-1:0] b_full_mant = b_isZero   ? {FULL_MANT{1'b0}} :
                                      b_isDenorm ? {1'b0, io_b_mant} : {1'b1, io_b_mant};

  wire [EXP_WIDTH-1:0] a_eff_exp = (io_a_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_a_exp;
  wire [EXP_WIDTH-1:0] b_eff_exp = (io_b_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : io_b_exp;

  // Determine larger operand
  wire a_exp_larger  = (a_eff_exp > b_eff_exp);
  wire exp_equal     = (a_eff_exp == b_eff_exp);
  wire a_mant_larger = (a_full_mant >= b_full_mant);
  wire swap = !a_exp_larger && (!exp_equal || !a_mant_larger);

  wire x_sign = swap ? io_b_sign : io_a_sign;
  wire y_sign = swap ? io_a_sign : io_b_sign;
  wire [EXP_WIDTH-1:0] x_exp  = swap ? b_eff_exp   : a_eff_exp;
  wire [EXP_WIDTH-1:0] y_exp  = swap ? a_eff_exp   : b_eff_exp;
  wire [FULL_MANT-1:0] x_mant = swap ? b_full_mant  : a_full_mant;
  wire [FULL_MANT-1:0] y_mant = swap ? a_full_mant  : b_full_mant;
  wire y_isZero = swap ? a_isZero : b_isZero;

  // Alignment shift
  wire [EXP_WIDTH-1:0] shift_amt = x_exp - y_exp;
  wire [WORK_WIDTH-1:0] x_mant_ext = {1'b0, x_mant, {GUARD_BITS{1'b0}}};
  wire [WORK_WIDTH-1:0] y_mant_ext = {1'b0, y_mant, {GUARD_BITS{1'b0}}};
  wire [WORK_WIDTH-1:0] y_shifted  = (shift_amt >= WORK_WIDTH) ? {WORK_WIDTH{1'b0}} :
                                      (y_mant_ext >> shift_amt);
  wire [WORK_WIDTH-1:0] sticky_mask = (shift_amt >= WORK_WIDTH) ? y_mant_ext :
                                       (y_mant_ext & ~({{WORK_WIDTH{1'b1}}} << shift_amt));
  wire sticky = |sticky_mask;
  wire [WORK_WIDTH-1:0] y_aligned = y_isZero ? {WORK_WIDTH{1'b0}} :
                                     {y_shifted[WORK_WIDTH-1:1], y_shifted[0] | sticky};

  // Add or subtract
  wire eff_sub = x_sign ^ y_sign;
  wire [WORK_WIDTH:0] sum = eff_sub ?
                             ({1'b0, x_mant_ext} - {1'b0, y_aligned}) :
                             ({1'b0, x_mant_ext} + {1'b0, y_aligned});
  wire sum_negative = sum[WORK_WIDTH];
  wire [WORK_WIDTH-1:0] s1_sum_abs = sum_negative ? (~sum[WORK_WIDTH-1:0] + 1) :
                                                      sum[WORK_WIDTH-1:0];
  wire s1_result_sign = sum_negative ? ~x_sign : x_sign;

  //==========================================================================
  // Pipeline Register: Stage 1 -> Stage 2
  //==========================================================================
  reg [WORK_WIDTH-1:0] r_sum_abs;
  reg                  r_result_sign;
  reg [EXP_WIDTH-1:0]  r_x_exp;
  reg                  r_a_isZero, r_b_isZero;
  reg                  r_a_isInf,  r_b_isInf;
  reg                  r_a_isNaN,  r_b_isNaN;
  reg                  r_a_sign,   r_b_sign;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_sum_abs     <= {WORK_WIDTH{1'b0}};
      r_result_sign <= 1'b0;
      r_x_exp       <= {EXP_WIDTH{1'b0}};
      r_a_isZero    <= 1'b0;
      r_b_isZero    <= 1'b0;
      r_a_isInf     <= 1'b0;
      r_b_isInf     <= 1'b0;
      r_a_isNaN     <= 1'b0;
      r_b_isNaN     <= 1'b0;
      r_a_sign      <= 1'b0;
      r_b_sign      <= 1'b0;
    end else begin
      r_sum_abs     <= s1_sum_abs;
      r_result_sign <= s1_result_sign;
      r_x_exp       <= x_exp;
      r_a_isZero    <= a_isZero;
      r_b_isZero    <= b_isZero;
      r_a_isInf     <= a_isInf;
      r_b_isInf     <= b_isInf;
      r_a_isNaN     <= a_isNaN;
      r_b_isNaN     <= b_isNaN;
      r_a_sign      <= io_a_sign;
      r_b_sign      <= io_b_sign;
    end
  end

  //==========================================================================
  // Stage 2: Normalize (LZC) + Round + Special Cases
  //==========================================================================

  // Leading zero count
  reg [$clog2(WORK_WIDTH)-1:0] lzc;
  integer i;
  always @(*) begin
    lzc = 0;
    for (i = WORK_WIDTH - 1; i >= 0; i = i - 1) begin
      if (!r_sum_abs[i] && lzc == (WORK_WIDTH - 1 - i))
        lzc = lzc + 1;
    end
  end

  wire add_overflow = r_sum_abs[WORK_WIDTH-1];

  wire [WORK_WIDTH-1:0] norm_mant = add_overflow ? (r_sum_abs >> 1) :
                                     (r_sum_abs << (lzc > 0 ? lzc - 1 : 0));

  wire signed [EXP_WIDTH+1:0] norm_exp = add_overflow ?
      $signed({2'b0, r_x_exp}) + 1 :
      $signed({2'b0, r_x_exp}) - $signed({{(EXP_WIDTH+2-$clog2(WORK_WIDTH)){1'b0}}, lzc}) + 1;

  // Rounding (Round to Nearest Even)
  wire [GUARD_BITS-1:0] guard_bits = norm_mant[GUARD_BITS-1:0];
  wire lsb = norm_mant[GUARD_BITS];
  wire round_up = guard_bits[GUARD_BITS-1] && (|guard_bits[GUARD_BITS-2:0] || lsb);

  wire [FULL_MANT:0] rounded_mant = {1'b0, norm_mant[WORK_WIDTH-2:GUARD_BITS]} +
                                      {{FULL_MANT{1'b0}}, round_up};
  wire round_overflow = rounded_mant[FULL_MANT];

  wire signed [EXP_WIDTH+1:0] final_exp  = round_overflow ? norm_exp + 1 : norm_exp;
  wire [MANT_WIDTH-1:0]       final_mant = round_overflow ? rounded_mant[FULL_MANT-1:1] :
                                                             rounded_mant[MANT_WIDTH-1:0];

  // Special cases
  wire result_isNaN_w  = r_a_isNaN || r_b_isNaN ||
                          (r_a_isInf && r_b_isInf && (r_a_sign != r_b_sign));
  wire result_isInf_w  = ((r_a_isInf || r_b_isInf) && !result_isNaN_w) ||
                          (final_exp >= EXP_MAX);
  wire sum_is_zero     = (r_sum_abs == 0);
  wire result_isZero_w = (r_a_isZero && r_b_isZero) ||
                          (sum_is_zero && !r_a_isInf && !r_b_isInf && !r_a_isNaN && !r_b_isNaN);
  wire underflow       = (final_exp <= 0) && !result_isZero_w && !result_isInf_w && !result_isNaN_w;

  //==========================================================================
  // Output
  //==========================================================================
  assign io_out_isZero = result_isZero_w;
  assign io_out_isInf  = result_isInf_w;
  assign io_out_isNaN  = result_isNaN_w;

  assign io_out_sign = result_isNaN_w  ? 1'b0 :
                       result_isZero_w ? (r_a_sign && r_b_sign) :
                       result_isInf_w  ? (r_a_isInf ? r_a_sign : r_b_sign) :
                       r_result_sign;

  assign io_out_exp = result_isZero_w ? {EXP_WIDTH{1'b0}} :
                      (result_isInf_w || result_isNaN_w) ? {EXP_WIDTH{1'b1}} :
                      underflow ? {EXP_WIDTH{1'b0}} :
                      final_exp[EXP_WIDTH-1:0];

  assign io_out_mant = result_isZero_w ? {MANT_WIDTH{1'b0}} :
                       result_isInf_w  ? {MANT_WIDTH{1'b0}} :
                       result_isNaN_w  ? {{MANT_WIDTH-1{1'b0}}, 1'b1} :
                       underflow       ? {MANT_WIDTH{1'b0}} :
                       final_mant;

endmodule
