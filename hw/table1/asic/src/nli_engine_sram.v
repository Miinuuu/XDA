//==============================================================================
// nli_engine_sram - NLI Engine with fakeram45_256x16 SRAM
//
// Same as nli_engine but lut_reg replaced with 2× fakeram45_256x16:
//   SRAM_A: read global_idx     (for table1)
//   SRAM_B: read global_idx+1   (for table2)
//   Both written with identical data during config.
//
// 259-entry handling:
//   SRAM: entries 0..255 (256 entries)
//   FF:   entries 256, 257, 258 (3-entry overflow)
//   Read mux selects SRAM or FF based on address
//
// Pipeline depth: 4 stages (unchanged)
// point_reg, mul_reg: remain as FF
//==============================================================================

`timescale 1ns/1ps

module nli_engine #(
    parameter D_N          = 32,
    parameter M            = 11,
    parameter NUM_INTERVALS = 10,
    parameter LUT_DEPTH    = 259
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        i_valid,
    input  wire [15:0] i_data,
    output reg         o_valid,
    output reg  [15:0] o_data,

    input  wire        cfg_we,
    input  wire [1:0]  cfg_sel,
    input  wire [8:0]  cfg_addr,
    input  wire [15:0] cfg_wdata
);

    localparam EXP_W  = 5;
    localparam MANT_W = 10;
    localparam BIAS   = 15;
    localparam D_N_BITS = $clog2(D_N);

    //==========================================================================
    // Register files: point_reg, mul_reg (FF, combinational read)
    //==========================================================================
    reg [15:0] point_reg [0:M-1];
    reg [15:0] mul_reg   [0:NUM_INTERVALS-1];

    // lut overflow FF: entries 256, 257, 258
    reg [15:0] lut_overflow [0:2];

    //==========================================================================
    // SRAM instances: 2× fakeram45_256x16 for lut_reg[0..255]
    //==========================================================================
    wire        lut_cfg_we = cfg_we && (cfg_sel == 2'd2);
    wire        sram_we    = lut_cfg_we && (cfg_addr < 9'd256);
    wire        overflow_we = lut_cfg_we && (cfg_addr >= 9'd256);

    always @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'd0: point_reg[cfg_addr[3:0]] <= cfg_wdata;
                2'd1: mul_reg[cfg_addr[3:0]]   <= cfg_wdata;
                default: ;
            endcase
        end
        if (overflow_we)
            lut_overflow[cfg_addr - 9'd256] <= cfg_wdata;
    end

    //==========================================================================
    // FP16 comparison function
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
    // Stage 1: Comparator + Subtractor (identical to original)
    //==========================================================================
    reg  [3:0]  cmp_index;
    reg  [15:0] cmp_lower_bound;
    wire [15:0] clamped_input;

    wire below_min = ~fp16_ge(i_data, point_reg[0]);
    wire above_max = fp16_ge(i_data, point_reg[M-1]);
    assign clamped_input = below_min ? point_reg[0] :
                           above_max ? point_reg[M-1] :
                           i_data;

    integer ci;
    always @(*) begin
        cmp_index = 4'd0;
        for (ci = 1; ci < M; ci = ci + 1) begin
            if (fp16_ge(clamped_input, point_reg[ci]))
                cmp_index = ci[3:0];
        end
        if (cmp_index >= NUM_INTERVALS[3:0])
            cmp_index = NUM_INTERVALS[3:0] - 4'd1;
        cmp_lower_bound = point_reg[cmp_index];
    end

    wire        sub1_sign, sub1_isZero, sub1_isInf, sub1_isNaN;
    wire [4:0]  sub1_exp;
    wire [9:0]  sub1_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_sub1 (
        .io_a_sign (clamped_input[15]),
        .io_a_exp  (clamped_input[14:10]),
        .io_a_mant (clamped_input[9:0]),
        .io_b_sign (~cmp_lower_bound[15]),
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
    // Stage 2: Scale Factor LUT + Multiplier + Floor/Frac (identical to original)
    //==========================================================================
    wire [15:0] scale_factor = mul_reg[p1_index];

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

    wire [10:0] full_mant_s2 = (mul2_exp == 0) ? {1'b0, mul2_mant} :
                                                  {1'b1, mul2_mant};
    wire [4:0]  exp_s2 = mul2_exp;

    reg [15:0] fixed_6_10;
    always @(*) begin
        if (mul2_isZero || exp_s2 == 0)
            fixed_6_10 = 16'd0;
        else if (exp_s2 < BIAS[4:0])
            fixed_6_10 = {5'b0, full_mant_s2} >> (BIAS[4:0] - exp_s2);
        else if ((exp_s2 - BIAS[4:0]) <= 5'd5)
            fixed_6_10 = {5'b0, full_mant_s2} << (exp_s2 - BIAS[4:0]);
        else
            fixed_6_10 = 16'hFFFF;
    end

    wire [5:0]  floor_raw  = fixed_6_10[15:10];
    wire [9:0]  frac_bits  = fixed_6_10[9:0];

    reg [D_N_BITS-1:0] s2_address;
    always @(*) begin
        if (p1_index == 4'd0 || p1_index == (NUM_INTERVALS[3:0] - 4'd1))
            s2_address = {D_N_BITS{1'b0}};
        else if (floor_raw >= D_N[5:0])
            s2_address = D_N[D_N_BITS-1:0] - 1;
        else
            s2_address = floor_raw[D_N_BITS-1:0];
    end

    reg [15:0] s2_decimal_fp16;
    reg [3:0]  lzc_frac;
    always @(*) begin
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
            default:        lzc_frac = 4'd10;
        endcase
        if (lzc_frac >= 4'd10 || mul2_isZero) begin
            s2_decimal_fp16 = 16'h0000;
        end else begin
            s2_decimal_fp16[15]    = 1'b0;
            s2_decimal_fp16[14:10] = (5'd14 - {1'b0, lzc_frac});
            s2_decimal_fp16[9:0]   = (frac_bits << (lzc_frac + 1));
        end
    end

    wire [15:0] s2_decimal = (exp_s2 < BIAS[4:0] && !mul2_isZero)
                             ? scaled_pos_fp16 : s2_decimal_fp16;

    wire decimal_ge_one = (s2_decimal[14:10] >= BIAS[4:0]) && (s2_decimal[15] == 1'b0)
                          && (s2_decimal != 16'h0000);
    wire [15:0] s2_decimal_clamped = decimal_ge_one ? 16'h3C00 : s2_decimal;

    reg [8:0] s2_global_idx;
    always @(*) begin
        if (p1_index == 4'd0)
            s2_global_idx = {4'd0, s2_address};
        else
            s2_global_idx = 9'd1 + ({5'd0, p1_index - 4'd1} * D_N[8:0])
                            + {4'd0, s2_address};
    end

    wire [8:0] s2_global_idx_clamped = (s2_global_idx > (LUT_DEPTH[8:0] - 9'd2))
                                       ? (LUT_DEPTH[8:0] - 9'd2)
                                       : s2_global_idx;

    //==========================================================================
    // SRAM address: driven from Stage 2 combinational path
    // SRAM captures on posedge → data available in Stage 3
    //==========================================================================
    wire [7:0] sram_addr_a = sram_we ? cfg_addr[7:0] : s2_global_idx_clamped[7:0];
    wire [7:0] sram_addr_b = sram_we ? cfg_addr[7:0] : s2_global_idx_clamped[7:0] + 8'd1;

    wire [15:0] sram_a_rd, sram_b_rd;

    fakeram45_256x16 u_sram_a (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (sram_we),
        .addr_in  (sram_addr_a),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (sram_a_rd)
    );

    fakeram45_256x16 u_sram_b (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (sram_we),
        .addr_in  (sram_addr_b),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (sram_b_rd)
    );

    //==========================================================================
    // Pipeline Register: Stage 2 → Stage 3
    // Also register the address for overflow mux in Stage 3
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
    // Stage 3: SRAM output + overflow mux → Subtraction
    //
    // For addr >= 256: use lut_overflow FF instead of SRAM
    // Max addr is 257 (LUT_DEPTH-2), so addr+1 max is 258
    //==========================================================================
    wire addr_a_overflow = (p2_global_idx >= 9'd256);
    wire addr_b_overflow = ((p2_global_idx + 9'd1) >= 9'd256);

    wire [15:0] table1 = addr_a_overflow ? lut_overflow[p2_global_idx - 9'd256] : sram_a_rd;
    wire [15:0] table2 = addr_b_overflow ? lut_overflow[p2_global_idx + 9'd1 - 9'd256] : sram_b_rd;

    wire        sub3_sign, sub3_isZero, sub3_isInf, sub3_isNaN;
    wire [4:0]  sub3_exp;
    wire [9:0]  sub3_mant;

    fp_adder #(.EXP_WIDTH(EXP_W), .MANT_WIDTH(MANT_W), .BIAS(BIAS)) u_sub3 (
        .io_a_sign (table2[15]),
        .io_a_exp  (table2[14:10]),
        .io_a_mant (table2[9:0]),
        .io_b_sign (~table1[15]),
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
    // Stage 4: Multiplier + Adder (identical to original)
    //==========================================================================
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

    //==========================================================================
    // Output Register
    //==========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_valid <= 1'b0;
            o_data  <= 16'h0000;
        end else begin
            o_valid <= p3_valid;
            o_data  <= p3_valid ? interp_result : 16'h0000;
        end
    end

endmodule
