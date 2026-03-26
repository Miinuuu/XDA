//==============================================================================
// fma_interp - Fused Multiply-Add Interpolation Unit
//
// Computes: result = y0 + diff × (t_int / 2^T_BITS)
//
// Eliminates intermediate normalize/round between multiply and add.
// Single normalize/round at the output (vs. 2× in separate frac_mult + fp_adder).
//
// 3-stage pipeline, 3-cycle latency:
//   Stage 1: Unpack + Integer Multiply + Exponent Compare
//   Stage 2: Swap + Alignment Shift + Add/Subtract
//   Stage 3: LZC + Normalize + Round + Output (combinational from r2)
//==============================================================================

`timescale 1ns/1ps

module fma_interp #(
    parameter T_BITS     = 10,
    parameter EXP_WIDTH  = 5,
    parameter MANT_WIDTH = 10,
    parameter BIAS       = 15
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] io_y0,
    input  wire [15:0] io_diff,
    input  wire [T_BITS-1:0] io_t,
    output wire [15:0] io_out
);

    localparam EXP_MAX    = (1 << EXP_WIDTH) - 1; // 31
    localparam FULL_MANT  = MANT_WIDTH + 1;        // 11
    localparam PROD_WIDTH = T_BITS + FULL_MANT;    // 21
    localparam GUARD_BITS = 3;
    localparam WORK_WIDTH = PROD_WIDTH + GUARD_BITS + 1; // 25
    localparam FRAC_START = GUARD_BITS + (PROD_WIDTH - FULL_MANT); // 13
    localparam MANT_MSB   = WORK_WIDTH - 3; // 22
    localparam MANT_LSB   = FRAC_START;     // 13

    //==========================================================================
    // Stage 1: Unpack + Integer Multiply + Exponent Compare
    //==========================================================================

    // Unpack y0
    wire       y0_sign = io_y0[15];
    wire [4:0] y0_exp  = io_y0[14:10];
    wire [9:0] y0_mant = io_y0[9:0];
    wire y0_isZero   = (y0_exp == 0) && (y0_mant == 0);
    wire y0_isDenorm = (y0_exp == 0) && (y0_mant != 0);
    wire y0_isInf    = (y0_exp == EXP_MAX) && (y0_mant == 0);
    wire y0_isNaN    = (y0_exp == EXP_MAX) && (y0_mant != 0);
    wire [FULL_MANT-1:0] y0_full = y0_isZero   ? {FULL_MANT{1'b0}} :
                                    y0_isDenorm ? {1'b0, y0_mant} : {1'b1, y0_mant};
    wire [EXP_WIDTH-1:0] y0_eff_exp = (y0_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : y0_exp;

    // Unpack diff
    wire       diff_sign = io_diff[15];
    wire [4:0] diff_exp  = io_diff[14:10];
    wire [9:0] diff_mant = io_diff[9:0];
    wire diff_isZero   = (diff_exp == 0) && (diff_mant == 0);
    wire diff_isDenorm = (diff_exp == 0) && (diff_mant != 0);
    wire diff_isInf    = (diff_exp == EXP_MAX) && (diff_mant == 0);
    wire diff_isNaN    = (diff_exp == EXP_MAX) && (diff_mant != 0);
    wire [FULL_MANT-1:0] diff_full = diff_isZero   ? {FULL_MANT{1'b0}} :
                                      diff_isDenorm ? {1'b0, diff_mant} : {1'b1, diff_mant};
    wire [EXP_WIDTH-1:0] diff_eff_exp = (diff_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : diff_exp;

    wire t_isZero = (io_t == 0);

    // Integer multiply: diff_mantissa × t_int
    wire [PROD_WIDTH-1:0] product_raw = diff_full * io_t;
    wire prod_isZero = diff_isZero || t_isZero || (product_raw == 0);

    // Extend y0 mantissa to product width (append T_BITS zeros)
    // This equalizes both operands to PROD_WIDTH bits with same exponent base
    wire [PROD_WIDTH-1:0] y0_extended = {y0_full, {T_BITS{1'b0}}};

    // Exponent compare + swap decision (pre-compute for Stage 2)
    wire y0_exp_larger  = (y0_eff_exp > diff_eff_exp);
    wire exp_equal      = (y0_eff_exp == diff_eff_exp);
    wire y0_mant_larger = (y0_extended >= product_raw);
    wire s1_swap = !y0_exp_larger && (!exp_equal || !y0_mant_larger);

    wire [EXP_WIDTH-1:0] s1_x_exp = s1_swap ? diff_eff_exp : y0_eff_exp;
    wire [EXP_WIDTH-1:0] s1_y_exp = s1_swap ? y0_eff_exp   : diff_eff_exp;
    wire [EXP_WIDTH-1:0] s1_shift_amt = s1_x_exp - s1_y_exp;

    //==========================================================================
    // Pipeline Register: Stage 1 -> Stage 2
    //==========================================================================
    reg [PROD_WIDTH-1:0] r1_y0_ext;
    reg [PROD_WIDTH-1:0] r1_product;
    reg                  r1_swap;
    reg [EXP_WIDTH-1:0]  r1_shift_amt;
    reg [EXP_WIDTH-1:0]  r1_x_exp;
    reg                  r1_y0_sign, r1_diff_sign;
    reg                  r1_y0_isZero, r1_y0_isInf, r1_y0_isNaN;
    reg                  r1_diff_isInf, r1_diff_isNaN;
    reg                  r1_prod_isZero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r1_y0_ext      <= {PROD_WIDTH{1'b0}};
            r1_product     <= {PROD_WIDTH{1'b0}};
            r1_swap        <= 1'b0;
            r1_shift_amt   <= {EXP_WIDTH{1'b0}};
            r1_x_exp       <= {EXP_WIDTH{1'b0}};
            r1_y0_sign     <= 1'b0;
            r1_diff_sign   <= 1'b0;
            r1_y0_isZero   <= 1'b0;
            r1_y0_isInf    <= 1'b0;
            r1_y0_isNaN    <= 1'b0;
            r1_diff_isInf  <= 1'b0;
            r1_diff_isNaN  <= 1'b0;
            r1_prod_isZero <= 1'b0;
        end else begin
            r1_y0_ext      <= y0_extended;
            r1_product     <= product_raw;
            r1_swap        <= s1_swap;
            r1_shift_amt   <= s1_shift_amt;
            r1_x_exp       <= s1_x_exp;
            r1_y0_sign     <= y0_sign;
            r1_diff_sign   <= diff_sign;
            r1_y0_isZero   <= y0_isZero;
            r1_y0_isInf    <= y0_isInf;
            r1_y0_isNaN    <= y0_isNaN;
            r1_diff_isInf  <= diff_isInf;
            r1_diff_isNaN  <= diff_isNaN;
            r1_prod_isZero <= prod_isZero;
        end
    end

    //==========================================================================
    // Stage 2: Swap + Alignment Shift + Add/Subtract
    //==========================================================================

    wire x_sign = r1_swap ? r1_diff_sign : r1_y0_sign;
    wire y_sign = r1_swap ? r1_y0_sign   : r1_diff_sign;
    wire [PROD_WIDTH-1:0] x_mant = r1_swap ? r1_product : r1_y0_ext;
    wire [PROD_WIDTH-1:0] y_mant = r1_swap ? r1_y0_ext  : r1_product;
    wire y_isZero = r1_swap ? r1_y0_isZero : r1_prod_isZero;

    // Extend with guard bits
    wire [WORK_WIDTH-1:0] x_mant_ext = {1'b0, x_mant, {GUARD_BITS{1'b0}}};
    wire [WORK_WIDTH-1:0] y_mant_ext = {1'b0, y_mant, {GUARD_BITS{1'b0}}};

    // Alignment shift
    wire [WORK_WIDTH-1:0] y_shifted = (r1_shift_amt >= WORK_WIDTH) ? {WORK_WIDTH{1'b0}} :
                                       (y_mant_ext >> r1_shift_amt);
    wire [WORK_WIDTH-1:0] sticky_mask = (r1_shift_amt >= WORK_WIDTH) ? y_mant_ext :
                                         (y_mant_ext & ~({WORK_WIDTH{1'b1}} << r1_shift_amt));
    wire sticky_s = |sticky_mask;
    wire [WORK_WIDTH-1:0] y_aligned = y_isZero ? {WORK_WIDTH{1'b0}} :
                                       {y_shifted[WORK_WIDTH-1:1], y_shifted[0] | sticky_s};

    // Add or subtract
    wire eff_sub = x_sign ^ y_sign;
    wire [WORK_WIDTH:0] sum = eff_sub ?
                               ({1'b0, x_mant_ext} - {1'b0, y_aligned}) :
                               ({1'b0, x_mant_ext} + {1'b0, y_aligned});
    wire sum_negative = sum[WORK_WIDTH];
    wire [WORK_WIDTH-1:0] s2_sum_abs = sum_negative ? (~sum[WORK_WIDTH-1:0] + 1) :
                                                        sum[WORK_WIDTH-1:0];
    wire s2_result_sign = sum_negative ? ~x_sign : x_sign;

    //==========================================================================
    // Pipeline Register: Stage 2 -> Stage 3
    //==========================================================================
    reg [WORK_WIDTH-1:0] r2_sum_abs;
    reg                  r2_result_sign;
    reg [EXP_WIDTH-1:0]  r2_x_exp;
    reg                  r2_y0_isZero, r2_y0_isInf, r2_y0_isNaN;
    reg                  r2_diff_isInf, r2_diff_isNaN;
    reg                  r2_prod_isZero;
    reg                  r2_y0_sign, r2_diff_sign;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r2_sum_abs     <= {WORK_WIDTH{1'b0}};
            r2_result_sign <= 1'b0;
            r2_x_exp       <= {EXP_WIDTH{1'b0}};
            r2_y0_isZero   <= 1'b0;
            r2_y0_isInf    <= 1'b0;
            r2_y0_isNaN    <= 1'b0;
            r2_diff_isInf  <= 1'b0;
            r2_diff_isNaN  <= 1'b0;
            r2_prod_isZero <= 1'b0;
            r2_y0_sign     <= 1'b0;
            r2_diff_sign   <= 1'b0;
        end else begin
            r2_sum_abs     <= s2_sum_abs;
            r2_result_sign <= s2_result_sign;
            r2_x_exp       <= r1_x_exp;
            r2_y0_isZero   <= r1_y0_isZero;
            r2_y0_isInf    <= r1_y0_isInf;
            r2_y0_isNaN    <= r1_y0_isNaN;
            r2_diff_isInf  <= r1_diff_isInf;
            r2_diff_isNaN  <= r1_diff_isNaN;
            r2_prod_isZero <= r1_prod_isZero;
            r2_y0_sign     <= r1_y0_sign;
            r2_diff_sign   <= r1_diff_sign;
        end
    end

    //==========================================================================
    // Stage 3: LZC + Normalize + Round + Output (combinational from r2)
    //==========================================================================

    // Leading zero count
    reg [$clog2(WORK_WIDTH)-1:0] lzc;
    integer i;
    always @(*) begin
        lzc = 0;
        for (i = WORK_WIDTH - 1; i >= 0; i = i - 1) begin
            if (!r2_sum_abs[i] && lzc == (WORK_WIDTH - 1 - i))
                lzc = lzc + 1;
        end
    end

    wire add_overflow = r2_sum_abs[WORK_WIDTH-1];

    wire [WORK_WIDTH-1:0] norm_mant = add_overflow ? (r2_sum_abs >> 1) :
                                       (r2_sum_abs << (lzc > 0 ? lzc - 1 : 0));

    wire signed [EXP_WIDTH+1:0] norm_exp = add_overflow ?
        $signed({2'b0, r2_x_exp}) + 1 :
        $signed({2'b0, r2_x_exp}) -
        $signed({{(EXP_WIDTH+2-$clog2(WORK_WIDTH)){1'b0}}, lzc}) + 1;

    // Extract mantissa bits and rounding bits from normalized result
    wire [MANT_WIDTH-1:0] trunc_mant = norm_mant[MANT_MSB : MANT_LSB];
    wire guard     = norm_mant[FRAC_START-1];
    wire round_bit = norm_mant[FRAC_START-2];
    wire sticky_r  = |norm_mant[FRAC_START-3:0];
    wire round_up  = guard & (round_bit | sticky_r | trunc_mant[0]);

    wire [MANT_WIDTH:0] rounded_mant = {1'b0, trunc_mant} + {{MANT_WIDTH{1'b0}}, round_up};
    wire round_overflow = rounded_mant[MANT_WIDTH];

    wire signed [EXP_WIDTH+1:0] final_exp  = round_overflow ? norm_exp + 1 : norm_exp;
    wire [MANT_WIDTH-1:0]       final_mant = round_overflow ? rounded_mant[MANT_WIDTH:1] :
                                                               rounded_mant[MANT_WIDTH-1:0];

    // Special cases
    wire prod_isInf    = r2_diff_isInf && !r2_prod_isZero;
    wire result_isNaN  = r2_y0_isNaN || r2_diff_isNaN ||
                          (r2_y0_isInf && prod_isInf && (r2_y0_sign != r2_diff_sign));
    wire result_isInf_w = ((r2_y0_isInf || prod_isInf) && !result_isNaN) ||
                          (!result_isNaN && !r2_y0_isInf && !prod_isInf && (final_exp >= EXP_MAX));
    wire sum_is_zero   = (r2_sum_abs == 0);
    wire result_isZero = (r2_y0_isZero && r2_prod_isZero) ||
                          (sum_is_zero && !r2_y0_isInf && !prod_isInf &&
                           !r2_y0_isNaN && !r2_diff_isNaN);
    wire underflow     = (final_exp <= 0) && !result_isZero && !result_isInf_w && !result_isNaN;

    // Output
    wire out_sign = result_isNaN   ? 1'b0 :
                    result_isZero  ? (r2_y0_sign & r2_diff_sign) :
                    result_isInf_w ? (r2_y0_isInf ? r2_y0_sign : r2_diff_sign) :
                    r2_result_sign;

    wire [EXP_WIDTH-1:0] out_exp = result_isZero ? {EXP_WIDTH{1'b0}} :
                                    (result_isInf_w || result_isNaN) ? {EXP_WIDTH{1'b1}} :
                                    underflow ? {EXP_WIDTH{1'b0}} :
                                    final_exp[EXP_WIDTH-1:0];

    wire [MANT_WIDTH-1:0] out_mant = result_isZero  ? {MANT_WIDTH{1'b0}} :
                                      result_isInf_w ? {MANT_WIDTH{1'b0}} :
                                      result_isNaN   ? {{MANT_WIDTH-1{1'b0}}, 1'b1} :
                                      underflow      ? {MANT_WIDTH{1'b0}} :
                                      final_mant;

    assign io_out = {out_sign, out_exp, out_mant};

endmodule
