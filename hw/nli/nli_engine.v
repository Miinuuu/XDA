//==============================================================================
// nli_engine - 4-Stage Pipelined NLI Inference Engine (FP16)
//
// Implements the NLI computation flow (Algorithm 2) for nonlinear function
// approximation using two-level address translation + linear interpolation.
//
// Architecture (from paper Fig. 7):
//   Stage 1: Comparator + Sub   → interval index, offset
//   Stage 2: Scale LUT + Mul + Floor → micro-bin address, decimal fraction
//   Stage 3: Function LUT + Sub → table1, table2, diff
//   Stage 4: Mul + Add          → interpolated output
//
// Parameters:
//   - 11 macro cutpoints (point_reg)
//   - 10 scale factors (mul_reg), one per interval
//   - 259 function value LUT entries (lut_reg)
//     Layout: 2 (boundary intervals) + 8×32 (middle intervals) + 1 = 259
//
// Uses fp_adder and fp_mult from HW/fpu/ (parameterized for FP16).
//==============================================================================

`timescale 1ns/1ps

module nli_engine #(
    parameter D_N          = 32,   // micro-bins per middle interval
    parameter M            = 11,   // number of macro cutpoints
    parameter NUM_INTERVALS = 10,  // M - 1
    parameter LUT_DEPTH    = 259   // 2 + 8*D_N + 1
) (
    input  wire        clk,
    input  wire        rst_n,

    // Data path
    input  wire        i_valid,
    input  wire [15:0] i_data,     // FP16 input
    output reg         o_valid,
    output reg  [15:0] o_data,     // FP16 output

    // Configuration interface (load registers before inference)
    input  wire        cfg_we,
    input  wire [1:0]  cfg_sel,    // 0: point_reg, 1: mul_reg, 2: lut_reg
    input  wire [8:0]  cfg_addr,
    input  wire [15:0] cfg_wdata
);

    // FP16 parameters
    localparam EXP_W  = 5;
    localparam MANT_W = 10;
    localparam BIAS   = 15;
    localparam D_N_BITS = $clog2(D_N);  // 5 for D_N=32

    //==========================================================================
    // Register files (loaded via cfg interface)
    //==========================================================================
    reg [15:0] point_reg [0:M-1];           // 11 cutpoints
    reg [15:0] mul_reg   [0:NUM_INTERVALS-1]; // 10 scale factors
    reg [15:0] lut_reg   [0:LUT_DEPTH-1];   // 259 function values

    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'd0: point_reg[cfg_addr[3:0]] <= cfg_wdata;
                2'd1: mul_reg[cfg_addr[3:0]]   <= cfg_wdata;
                2'd2: lut_reg[cfg_addr[8:0]]   <= cfg_wdata;
                default: ;
            endcase
        end
    end

    //==========================================================================
    // FP16 comparison function: returns 1 if a >= b
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
                fp16_ge = 1'b1;   // positive >= negative
            else
                fp16_ge = (a_mag == 0 && b_mag == 0); // -0 >= +0
        end
    endfunction

    //==========================================================================
    // Stage 1: Comparator + Subtractor
    //
    // - Clamp input to [point_reg[0], point_reg[M-1]]
    // - Find interval index via parallel comparators
    // - Compute offset = clamped_input - lower_boundary
    //==========================================================================

    // --- Comparator: find interval index ---
    reg  [3:0]  cmp_index;
    reg  [15:0] cmp_lower_bound;
    wire [15:0] clamped_input;

    // Clamp
    wire below_min = ~fp16_ge(i_data, point_reg[0]);
    wire above_max = fp16_ge(i_data, point_reg[M-1]);
    assign clamped_input = below_min ? point_reg[0] :
                           above_max ? point_reg[M-1] :
                           i_data;

    // Parallel comparators: count how many cutpoints the input exceeds
    // index = max i such that input >= point_reg[i], clamped to [0, NUM_INTERVALS-1]
    integer ci;
    always @(*) begin
        cmp_index = 4'd0;
        for (ci = 1; ci < M; ci = ci + 1) begin
            if (fp16_ge(clamped_input, point_reg[ci]))
                cmp_index = ci[3:0];
        end
        // Clamp to valid interval [0, NUM_INTERVALS-1]
        if (cmp_index >= NUM_INTERVALS[3:0])
            cmp_index = NUM_INTERVALS[3:0] - 4'd1;

        cmp_lower_bound = point_reg[cmp_index];
    end

    // --- Subtractor: offset = clamped_input - lower_boundary ---
    wire        sub1_sign, sub1_isZero, sub1_isInf, sub1_isNaN;
    wire [4:0]  sub1_exp;
    wire [9:0]  sub1_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_sub1 (
        .io_a_sign (clamped_input[15]),
        .io_a_exp  (clamped_input[14:10]),
        .io_a_mant (clamped_input[9:0]),
        .io_b_sign (~cmp_lower_bound[15]),   // negate for subtraction
        .io_b_exp  (cmp_lower_bound[14:10]),
        .io_b_mant (cmp_lower_bound[9:0]),
        .io_out_sign  (sub1_sign),
        .io_out_exp   (sub1_exp),
        .io_out_mant  (sub1_mant),
        .io_out_isZero(sub1_isZero),
        .io_out_isInf (sub1_isInf),
        .io_out_isNaN (sub1_isNaN)
    );

    wire [15:0] s1_offset = sub1_isZero ? 16'h0000 :
                             {sub1_sign, sub1_exp, sub1_mant};

    //==========================================================================
    // Pipeline Register: Stage 1 → Stage 2
    //==========================================================================
    reg        p1_valid;
    reg [15:0] p1_offset;
    reg [3:0]  p1_index;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid    <= i_valid;
            p1_offset   <= s1_offset;
            p1_index    <= cmp_index;
        end
    end

    //==========================================================================
    // Stage 2: Scale Factor LUT + Multiplier + Floor/Frac
    //
    // - Lookup scale factor from mul_reg[index]
    // - Multiply: scaled_pos = offset × scale_factor
    // - Extract floor (address) and fractional part (decimal)
    // - Compute global LUT index
    //==========================================================================

    // --- Scale factor lookup (combinational from register file) ---
    wire [15:0] scale_factor = mul_reg[p1_index];

    // --- Multiplier: scaled_pos = offset × scale ---
    wire        mul2_sign, mul2_isZero, mul2_isInf, mul2_isNaN;
    wire [4:0]  mul2_exp;
    wire [9:0]  mul2_mant;

    fp_mult_norm #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_mul2 (
        .io_a_sign (p1_offset[15]),
        .io_a_exp  (p1_offset[14:10]),
        .io_a_mant (p1_offset[9:0]),
        .io_b_sign (scale_factor[15]),
        .io_b_exp  (scale_factor[14:10]),
        .io_b_mant (scale_factor[9:0]),
        .io_out_sign  (mul2_sign),
        .io_out_exp   (mul2_exp),
        .io_out_mant  (mul2_mant),
        .io_out_isZero(mul2_isZero),
        .io_out_isInf (mul2_isInf),
        .io_out_isNaN (mul2_isNaN)
    );

    wire [15:0] scaled_pos_fp16 = mul2_isZero ? 16'h0000 :
                                  {mul2_sign, mul2_exp, mul2_mant};

    // --- Floor / Fractional Part Extraction ---
    // Convert FP16 to 6.10 fixed point, then extract integer and fraction.
    // scaled_pos is always non-negative (offset >= 0, scale >= 0)
    // Expected range: [0, D_N) = [0, 32)

    wire [10:0] full_mant_s2 = (mul2_exp == 0) ? {1'b0, mul2_mant} :
                                                  {1'b1, mul2_mant};
    wire [4:0]  exp_s2 = mul2_exp;

    // Convert to 6.10 fixed point: fixed = full_mant << (exp - BIAS)
    // For exp < BIAS: right shift (value < 1.0)
    // For exp >= BIAS: left shift
    reg [15:0] fixed_6_10;
    always @(*) begin
        if (mul2_isZero || exp_s2 == 0)
            fixed_6_10 = 16'd0;
        else if (exp_s2 < BIAS[4:0])
            fixed_6_10 = {5'b0, full_mant_s2} >> (BIAS[4:0] - exp_s2);
        else if ((exp_s2 - BIAS[4:0]) <= 5'd5)
            fixed_6_10 = {5'b0, full_mant_s2} << (exp_s2 - BIAS[4:0]);
        else
            fixed_6_10 = 16'hFFFF;  // overflow, will be clamped
    end

    // Integer part (address) and fractional part
    wire [5:0]  floor_raw  = fixed_6_10[15:10];
    wire [9:0]  frac_bits  = fixed_6_10[9:0];

    // Clamp address for boundary intervals
    reg [D_N_BITS-1:0] s2_address;
    always @(*) begin
        if (p1_index == 4'd0 || p1_index == (NUM_INTERVALS[3:0] - 4'd1))
            s2_address = {D_N_BITS{1'b0}};  // boundary: force 0
        else if (floor_raw >= D_N[5:0])
            s2_address = D_N[D_N_BITS-1:0] - 1;  // clamp to D_N-1
        else
            s2_address = floor_raw[D_N_BITS-1:0];
    end

    // --- Convert fractional part to FP16 ---
    // frac_bits[9:0] represents 0.frac_bits in [0, 1)
    reg [15:0] s2_decimal_fp16;
    reg [3:0]  lzc_frac;

    always @(*) begin
        // Leading zero count on 10-bit fraction
        casez (frac_bits)
            10'b1?????????: lzc_frac = 4'd0;
            10'b01????????: lzc_frac = 4'd1;
            10'b001???????: lzc_frac = 4'd2;
            10'b0001??????: lzc_frac = 4'd3;
            10'b00001?????: lzc_frac = 4'd4;
            10'b000001????: lzc_frac = 4'd5;
            10'b0000001???: lzc_frac = 4'd6;
            10'b00000001??: lzc_frac = 4'd7;
            10'b000000001?: lzc_frac = 4'd8;
            10'b0000000001: lzc_frac = 4'd9;
            default:        lzc_frac = 4'd10;  // all zeros
        endcase

        if (lzc_frac >= 4'd10 || mul2_isZero) begin
            s2_decimal_fp16 = 16'h0000;
        end else begin
            // exp = BIAS - 1 - lzc = 14 - lzc
            // mant = frac_bits shifted left by (lzc+1), take top 10 bits
            s2_decimal_fp16[15]    = 1'b0;  // always positive
            s2_decimal_fp16[14:10] = (5'd14 - {1'b0, lzc_frac});
            s2_decimal_fp16[9:0]   = (frac_bits << (lzc_frac + 1));
        end
    end

    // When scaled_pos < 1.0 (exp < BIAS), floor=0 so decimal = scaled_pos itself.
    // Using scaled_pos_fp16 directly preserves full FP16 precision (fixed-point
    // extraction loses bits for small values due to right-shift).
    wire [15:0] s2_decimal = (exp_s2 < BIAS[4:0] && !mul2_isZero)
                             ? scaled_pos_fp16   // value < 1.0, floor=0, decimal=value
                             : s2_decimal_fp16;

    // Clamp decimal to [0, 1.0]. If decimal > 1.0, clamp to 1.0 (0x3C00)
    // Check: if exp >= 15 (BIAS) and the value is positive, it's >= 1.0
    wire decimal_ge_one = (s2_decimal[14:10] >= BIAS[4:0]) && (s2_decimal[15] == 1'b0)
                          && (s2_decimal != 16'h0000);
    wire [15:0] s2_decimal_clamped = decimal_ge_one ? 16'h3C00 : s2_decimal;

    // --- Global LUT index computation ---
    // index 0: global_idx = address (= 0)
    // index 1..8: global_idx = 1 + (index-1)*D_N + address
    // index 9: global_idx = 1 + 8*D_N + address (= 257)
    reg [8:0] s2_global_idx;
    always @(*) begin
        if (p1_index == 4'd0)
            s2_global_idx = {4'd0, s2_address};
        else
            s2_global_idx = 9'd1 + ({5'd0, p1_index - 4'd1} * D_N[8:0])
                            + {4'd0, s2_address};
    end

    // Clamp global index to valid range [0, LUT_DEPTH-2]
    wire [8:0] s2_global_idx_clamped = (s2_global_idx > (LUT_DEPTH[8:0] - 9'd2))
                                       ? (LUT_DEPTH[8:0] - 9'd2)
                                       : s2_global_idx;

    //==========================================================================
    // Pipeline Register: Stage 2 → Stage 3
    //==========================================================================
    reg        p2_valid;
    reg [8:0]  p2_global_idx;
    reg [15:0] p2_decimal;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_valid <= 1'b0;
        end else begin
            p2_valid      <= p1_valid;
            p2_global_idx <= s2_global_idx_clamped;
            p2_decimal    <= s2_decimal_clamped;
        end
    end

    //==========================================================================
    // Stage 3: Function Value LUT + Subtractor
    //
    // - Read table1 = lut_reg[global_idx]
    // - Read table2 = lut_reg[global_idx + 1]
    // - Compute diff = table2 - table1
    //==========================================================================

    wire [15:0] table1 = lut_reg[p2_global_idx];
    wire [15:0] table2 = lut_reg[p2_global_idx + 9'd1];

    // Subtractor: diff = table2 - table1
    wire        sub3_sign, sub3_isZero, sub3_isInf, sub3_isNaN;
    wire [4:0]  sub3_exp;
    wire [9:0]  sub3_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_sub3 (
        .io_a_sign (table2[15]),
        .io_a_exp  (table2[14:10]),
        .io_a_mant (table2[9:0]),
        .io_b_sign (~table1[15]),   // negate for subtraction
        .io_b_exp  (table1[14:10]),
        .io_b_mant (table1[9:0]),
        .io_out_sign  (sub3_sign),
        .io_out_exp   (sub3_exp),
        .io_out_mant  (sub3_mant),
        .io_out_isZero(sub3_isZero),
        .io_out_isInf (sub3_isInf),
        .io_out_isNaN (sub3_isNaN)
    );

    wire [15:0] s3_diff = sub3_isZero ? 16'h0000 :
                           {sub3_sign, sub3_exp, sub3_mant};

    //==========================================================================
    // Pipeline Register: Stage 3 → Stage 4
    //==========================================================================
    reg        p3_valid;
    reg [15:0] p3_table1;
    reg [15:0] p3_diff;
    reg [15:0] p3_decimal;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p3_valid <= 1'b0;
        end else begin
            p3_valid    <= p2_valid;
            p3_table1   <= table1;
            p3_diff     <= s3_diff;
            p3_decimal  <= p2_decimal;
        end
    end

    //==========================================================================
    // Stage 4: Multiplier + Adder (Linear Interpolation)
    //
    // product = decimal × diff
    // result  = table1 + product
    //==========================================================================

    // --- Multiplier: product = decimal × diff ---
    wire        mul4_sign, mul4_isZero, mul4_isInf, mul4_isNaN;
    wire [4:0]  mul4_exp;
    wire [9:0]  mul4_mant;

    fp_mult_norm #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_mul4 (
        .io_a_sign (p3_decimal[15]),
        .io_a_exp  (p3_decimal[14:10]),
        .io_a_mant (p3_decimal[9:0]),
        .io_b_sign (p3_diff[15]),
        .io_b_exp  (p3_diff[14:10]),
        .io_b_mant (p3_diff[9:0]),
        .io_out_sign  (mul4_sign),
        .io_out_exp   (mul4_exp),
        .io_out_mant  (mul4_mant),
        .io_out_isZero(mul4_isZero),
        .io_out_isInf (mul4_isInf),
        .io_out_isNaN (mul4_isNaN)
    );

    wire [15:0] product = mul4_isZero ? 16'h0000 :
                           {mul4_sign, mul4_exp, mul4_mant};

    // --- Adder: result = table1 + product ---
    wire        add4_sign, add4_isZero, add4_isInf, add4_isNaN;
    wire [4:0]  add4_exp;
    wire [9:0]  add4_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_add4 (
        .io_a_sign (p3_table1[15]),
        .io_a_exp  (p3_table1[14:10]),
        .io_a_mant (p3_table1[9:0]),
        .io_b_sign (product[15]),
        .io_b_exp  (product[14:10]),
        .io_b_mant (product[9:0]),
        .io_out_sign  (add4_sign),
        .io_out_exp   (add4_exp),
        .io_out_mant  (add4_mant),
        .io_out_isZero(add4_isZero),
        .io_out_isInf (add4_isInf),
        .io_out_isNaN (add4_isNaN)
    );

    wire [15:0] interp_result = add4_isZero ? 16'h0000 :
                                 {add4_sign, add4_exp, add4_mant};

    wire [15:0] final_result = interp_result;

    //==========================================================================
    // Output Register
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            o_data  <= 16'h0000;
        end else begin
            o_valid <= p3_valid;
            o_data  <= p3_valid ? final_result : 16'h0000;
        end
    end

    // Pipeline latency: 4 clock cycles (i_valid → o_valid)

endmodule
