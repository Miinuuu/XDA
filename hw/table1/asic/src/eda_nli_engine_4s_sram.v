//==============================================================================
// eda_nli_engine_4s_sram - XDA Engine with all-SRAM memory
//
// 5-stage pipeline (vs original 4-stage):
//   Stage 1: bin_addr → config_rom SRAM (fakeram45_64x15)
//   Stage 2: config_rom output → addr gen → func_lut SRAM (2× fakeram45_256x16)
//   Stage 3: func_lut output → FP16 subtraction
//   Stage 4: FMA Part A
//   Stage 5: FMA Part B
//
// Memory:
//   config_rom: 1× fakeram45_64x15 (64 × 13-bit, stored in 15-bit words)
//   func_lut:   2× fakeram45_256x16 (for addr and addr+1 dual read)
//
// Throughput: 1 result/cycle (pipelined). Latency: 5 cycles.
//==============================================================================

`timescale 1ns/1ps

module eda_nli_engine_4s #(
    parameter T_BITS    = 10,
    parameter LUT_DEPTH = 256,
    parameter GRADUAL_UNDERFLOW = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        i_valid,
    input  wire [15:0] i_data,
    output reg         o_valid,
    output reg  [15:0] o_data,

    input  wire        cfg_we,
    input  wire [0:0]  cfg_sel,
    input  wire [8:0]  cfg_addr,
    input  wire [15:0] cfg_wdata
);

    localparam LUT_ABITS  = $clog2(LUT_DEPTH);
    localparam EXP_WIDTH  = 5;
    localparam MANT_WIDTH = 10;
    localparam EXP_MAX    = 31;
    localparam BIAS       = 15;
    localparam FULL_MANT  = MANT_WIDTH + 1;
    localparam PROD_WIDTH = T_BITS + FULL_MANT;
    localparam GUARD_BITS = 3;
    localparam WORK_WIDTH = PROD_WIDTH + GUARD_BITS + 1;
    localparam FRAC_START = GUARD_BITS + (PROD_WIDTH - FULL_MANT);
    //==========================================================================
    // SRAM control signals
    //==========================================================================
    wire cfg_we_rom = cfg_we && (cfg_sel == 1'd0);
    wire cfg_we_lut = cfg_we && (cfg_sel == 1'd1);

    //==========================================================================
    // Stage 1: Present bin_addr to config_rom SRAM
    //==========================================================================
    wire        in_sign  = i_data[15];
    wire [4:0]  in_exp   = i_data[14:10];
    wire [9:0]  in_mant  = i_data[9:0];
    wire        in_isNaN = (in_exp == 5'h1F) && (in_mant != 0);
    wire [5:0]  bin_addr = {in_sign, in_exp};

    // config_rom SRAM: address mux (config write vs inference read)
    wire [5:0]  rom_addr = cfg_we_rom ? cfg_addr[5:0] : bin_addr;

    wire [14:0] rom_rd;
    fakeram45_64x15 u_config_rom (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (cfg_we_rom),
        .addr_in  (rom_addr),
        .wd_in    ({2'b0, cfg_wdata[12:0]}),
        .w_mask_in(15'h7FFF),
        .rd_out   (rom_rd)
    );

    // Pipeline Register: Stage 1 → Stage 2
    reg        p0_valid;
    reg [9:0]  p0_mant;
    reg        p0_isNaN;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p0_valid <= 1'b0;
        end else begin
            p0_valid <= i_valid;
            p0_mant  <= in_mant;
            p0_isNaN <= in_isNaN;
        end
    end

    //==========================================================================
    // Stage 2: config_rom SRAM output → addr gen → func_lut SRAM
    //==========================================================================
    wire [12:0] cfg_entry      = rom_rd[12:0];
    wire        s2_clamp       = cfg_entry[12];
    wire [2:0]  s2_k_bits      = cfg_entry[11:9];
    wire [8:0]  s2_base_offset = cfg_entry[8:0];

    reg [4:0]        s2_micro_idx;
    reg [T_BITS-1:0] s2_t_int;

    always @(*) begin
        case (s2_k_bits)
            3'd0: begin s2_micro_idx = 5'd0;                       s2_t_int = p0_mant[9:0];                end
            3'd1: begin s2_micro_idx = {4'd0, p0_mant[9]};         s2_t_int = {p0_mant[8:0], 1'b0};       end
            3'd2: begin s2_micro_idx = {3'd0, p0_mant[9:8]};       s2_t_int = {p0_mant[7:0], 2'b0};       end
            3'd3: begin s2_micro_idx = {2'd0, p0_mant[9:7]};       s2_t_int = {p0_mant[6:0], 3'b0};       end
            3'd4: begin s2_micro_idx = {1'd0, p0_mant[9:6]};       s2_t_int = {p0_mant[5:0], 4'b0};       end
            3'd5: begin s2_micro_idx = p0_mant[9:5];               s2_t_int = {p0_mant[4:0], 5'b0};       end
            default: begin s2_micro_idx = 5'd0;                    s2_t_int = 10'd0;                       end
        endcase
    end

    wire [LUT_ABITS-1:0] s2_lut_addr = s2_base_offset[LUT_ABITS-1:0] + {4'd0, s2_micro_idx};
    wire [LUT_ABITS-1:0] s2_lut_addr_clamped =
        (s2_lut_addr >= (LUT_DEPTH[LUT_ABITS-1:0] - 1)) ?
        (LUT_DEPTH[LUT_ABITS-1:0] - 2) : s2_lut_addr;

    // func_lut SRAM address mux
    wire [7:0] sram_addr_a = cfg_we_lut ? cfg_addr[7:0] : s2_lut_addr_clamped;
    wire [7:0] sram_addr_b = cfg_we_lut ? cfg_addr[7:0] : (s2_lut_addr_clamped + 8'd1);

    wire [15:0] sram_a_rd, sram_b_rd;

    fakeram45_256x16 u_sram_a (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (cfg_we_lut),
        .addr_in  (sram_addr_a),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (sram_a_rd)
    );

    fakeram45_256x16 u_sram_b (
        .clk      (clk),
        .ce_in    (1'b1),
        .we_in    (cfg_we_lut),
        .addr_in  (sram_addr_b),
        .wd_in    (cfg_wdata),
        .w_mask_in(16'hFFFF),
        .rd_out   (sram_b_rd)
    );

    // Pipeline Register: Stage 2 → Stage 3
    reg                   p1_valid;
    reg [T_BITS-1:0]     p1_t_int;
    reg                   p1_clamp;
    reg                   p1_isNaN;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p1_valid <= 1'b0;
        end else begin
            p1_valid <= p0_valid;
            p1_t_int <= s2_t_int;
            p1_clamp <= s2_clamp;
            p1_isNaN <= p0_isNaN;
        end
    end

    //==========================================================================
    // Stage 3: func_lut SRAM output → FP16 Subtraction
    //==========================================================================
    wire [15:0] y0 = sram_a_rd;
    wire [15:0] y1 = sram_b_rd;

    wire       sub_sign, sub_isZero, sub_isInf, sub_isNaN;
    wire [4:0] sub_exp;
    wire [9:0] sub_mant;

    fp_adder #(
        .EXP_WIDTH(5),
        .MANT_WIDTH(10),
        .BIAS(15),
        .GRADUAL_UNDERFLOW(GRADUAL_UNDERFLOW)
    ) u_sub (
        .io_a_sign (y1[15]),
        .io_a_exp  (y1[14:10]),
        .io_a_mant (y1[9:0]),
        .io_b_sign (~y0[15]),
        .io_b_exp  (y0[14:10]),
        .io_b_mant (y0[9:0]),
        .io_out_sign  (sub_sign),
        .io_out_exp   (sub_exp),
        .io_out_mant  (sub_mant),
        .io_out_isZero(sub_isZero),
        .io_out_isInf (sub_isInf),
        .io_out_isNaN (sub_isNaN)
    );

    wire [15:0] s3_diff = sub_isZero ? 16'h0000 : {sub_sign, sub_exp, sub_mant};

    // Pipeline Register: Stage 3 → Stage 4
    reg                p2_valid;
    reg [15:0]         p2_y0;
    reg [15:0]         p2_diff;
    reg [T_BITS-1:0]   p2_t_int;
    reg                p2_clamp;
    reg                p2_isNaN;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_valid <= 1'b0;
        end else begin
            p2_valid  <= p1_valid;
            p2_y0     <= y0;
            p2_diff   <= s3_diff;
            p2_t_int  <= p1_t_int;
            p2_clamp  <= p1_clamp;
            p2_isNaN  <= p1_isNaN;
        end
    end

    //==========================================================================
    // Stage 4: FMA Part A
    //==========================================================================
    wire       y0_sign = p2_y0[15];
    wire [4:0] y0_exp  = p2_y0[14:10];
    wire [9:0] y0_mant = p2_y0[9:0];
    wire y0_isZero   = (y0_exp == 0) && (y0_mant == 0);
    wire y0_isDenorm = (y0_exp == 0) && (y0_mant != 0);
    wire y0_isInf    = (y0_exp == EXP_MAX) && (y0_mant == 0);
    wire y0_isNaN_w  = (y0_exp == EXP_MAX) && (y0_mant != 0);
    wire [FULL_MANT-1:0] y0_full = y0_isZero   ? {FULL_MANT{1'b0}} :
                                    y0_isDenorm ? {1'b0, y0_mant} : {1'b1, y0_mant};
    wire [EXP_WIDTH-1:0] y0_eff_exp = (y0_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : y0_exp;

    wire       diff_sign = p2_diff[15];
    wire [4:0] diff_exp  = p2_diff[14:10];
    wire [9:0] diff_mant = p2_diff[9:0];
    wire diff_isZero   = (diff_exp == 0) && (diff_mant == 0);
    wire diff_isDenorm = (diff_exp == 0) && (diff_mant != 0);
    wire diff_isInf    = (diff_exp == EXP_MAX) && (diff_mant == 0);
    wire diff_isNaN_w  = (diff_exp == EXP_MAX) && (diff_mant != 0);
    wire [FULL_MANT-1:0] diff_full = diff_isZero   ? {FULL_MANT{1'b0}} :
                                      diff_isDenorm ? {1'b0, diff_mant} : {1'b1, diff_mant};
    wire [EXP_WIDTH-1:0] diff_eff_exp = (diff_exp == 0) ? {{EXP_WIDTH-1{1'b0}}, 1'b1} : diff_exp;

    wire t_isZero = (p2_t_int == 0);
    wire [PROD_WIDTH-1:0] product_raw = diff_full * p2_t_int;
    wire prod_isZero = diff_isZero || t_isZero || (product_raw == 0);

    wire [PROD_WIDTH-1:0] y0_extended = {y0_full, {T_BITS{1'b0}}};

    wire y0_exp_larger  = (y0_eff_exp > diff_eff_exp);
    wire exp_equal      = (y0_eff_exp == diff_eff_exp);
    wire y0_mant_larger = (y0_extended >= product_raw);
    wire s4_swap = !y0_exp_larger && (!exp_equal || !y0_mant_larger);

    wire                  x_sign = s4_swap ? diff_sign    : y0_sign;
    wire                  y_sign = s4_swap ? y0_sign      : diff_sign;
    wire [EXP_WIDTH-1:0]  x_exp  = s4_swap ? diff_eff_exp : y0_eff_exp;
    wire [PROD_WIDTH-1:0] x_mant = s4_swap ? product_raw  : y0_extended;
    wire [PROD_WIDTH-1:0] y_mant = s4_swap ? y0_extended   : product_raw;
    wire                  y_isZ  = s4_swap ? y0_isZero     : prod_isZero;
    wire [EXP_WIDTH-1:0]  shift_amt = x_exp - (s4_swap ? y0_eff_exp : diff_eff_exp);

    wire [WORK_WIDTH-1:0] x_ext = {1'b0, x_mant, {GUARD_BITS{1'b0}}};
    wire [WORK_WIDTH-1:0] y_ext = {1'b0, y_mant, {GUARD_BITS{1'b0}}};

    wire [WORK_WIDTH-1:0] y_shifted = (shift_amt >= WORK_WIDTH) ? {WORK_WIDTH{1'b0}} :
                                       (y_ext >> shift_amt);
    wire [WORK_WIDTH-1:0] sticky_mask = (shift_amt >= WORK_WIDTH) ? y_ext :
                                         (y_ext & ~({WORK_WIDTH{1'b1}} << shift_amt));
    wire sticky_s = |sticky_mask;
    wire [WORK_WIDTH-1:0] y_aligned = y_isZ ? {WORK_WIDTH{1'b0}} :
                                       {y_shifted[WORK_WIDTH-1:1], y_shifted[0] | sticky_s};

    wire eff_sub = x_sign ^ y_sign;
    wire [WORK_WIDTH:0] sum_raw = eff_sub ?
                                   ({1'b0, x_ext} - {1'b0, y_aligned}) :
                                   ({1'b0, x_ext} + {1'b0, y_aligned});
    wire sum_negative = sum_raw[WORK_WIDTH];
    wire [WORK_WIDTH-1:0] s4_sum_abs = sum_negative ? (~sum_raw[WORK_WIDTH-1:0] + 1) :
                                                        sum_raw[WORK_WIDTH-1:0];
    wire s4_result_sign = sum_negative ? ~x_sign : x_sign;

    // Pipeline Register: Stage 4 → Stage 5
    reg [WORK_WIDTH-1:0] p3_sum_abs;
    reg                  p3_result_sign;
    reg [EXP_WIDTH-1:0]  p3_x_exp;
    reg                  p3_y0_isZero, p3_y0_isInf, p3_y0_isNaN;
    reg                  p3_diff_isInf, p3_diff_isNaN;
    reg                  p3_prod_isZero;
    reg                  p3_y0_sign, p3_diff_sign;
    reg                  p3_clamp;
    reg                  p3_engine_isNaN;
    reg [15:0]           p3_y0;
    reg                  p3_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p3_valid <= 1'b0;
        end else begin
            p3_valid        <= p2_valid;
            p3_sum_abs      <= s4_sum_abs;
            p3_result_sign  <= s4_result_sign;
            p3_x_exp        <= x_exp;
            p3_y0_isZero    <= y0_isZero;
            p3_y0_isInf     <= y0_isInf;
            p3_y0_isNaN     <= y0_isNaN_w;
            p3_diff_isInf   <= diff_isInf;
            p3_diff_isNaN   <= diff_isNaN_w;
            p3_prod_isZero  <= prod_isZero;
            p3_y0_sign      <= y0_sign;
            p3_diff_sign    <= diff_sign;
            p3_clamp        <= p2_clamp;
            p3_engine_isNaN <= p2_isNaN;
            p3_y0           <= p2_y0;
        end
    end

    //==========================================================================
    // Stage 5: FMA Part B
    //==========================================================================
    reg [$clog2(WORK_WIDTH)-1:0] lzc;
    integer i;
    always @(*) begin
        lzc = 0;
        for (i = WORK_WIDTH - 1; i >= 0; i = i - 1) begin
            if (!p3_sum_abs[i] && lzc == (WORK_WIDTH - 1 - i))
                lzc = lzc + 1;
        end
    end

    wire add_overflow = p3_sum_abs[WORK_WIDTH-1];

    wire [WORK_WIDTH-1:0] norm_mant = add_overflow ? (p3_sum_abs >> 1) :
                                       (p3_sum_abs << (lzc > 0 ? lzc - 1 : 0));

    wire signed [EXP_WIDTH+1:0] norm_exp = add_overflow ?
        $signed({2'b0, p3_x_exp}) + 1 :
        $signed({2'b0, p3_x_exp}) -
        $signed({{(EXP_WIDTH+2-$clog2(WORK_WIDTH)){1'b0}}, lzc}) + 1;

    wire [FULL_MANT-1:0] sig_mant = norm_mant[FRAC_START+FULL_MANT-1 : FRAC_START];
    wire guard     = norm_mant[FRAC_START-1];
    wire round_bit = norm_mant[FRAC_START-2];
    wire sticky_r  = |norm_mant[FRAC_START-3:0];
    wire round_up  = guard & (round_bit | sticky_r | sig_mant[0]);

    wire [FULL_MANT:0] rounded_mant = {1'b0, sig_mant} + {{FULL_MANT{1'b0}}, round_up};
    wire round_overflow = rounded_mant[FULL_MANT];

    wire signed [EXP_WIDTH+1:0] final_exp  = round_overflow ? norm_exp + 1 : norm_exp;
    wire [MANT_WIDTH-1:0]       final_mant = round_overflow ? rounded_mant[MANT_WIDTH:1] :
                                                               rounded_mant[MANT_WIDTH-1:0];

    // Table 1 resource builds use the submitted FTZ default; set
    // GRADUAL_UNDERFLOW=1 for subnormal accuracy sensitivity checks.
    wire [EXP_WIDTH+2:0] sub_shift = FRAC_START + 1 - final_exp;
    wire [WORK_WIDTH-1:0] sub_quot =
        (sub_shift >= WORK_WIDTH) ? {WORK_WIDTH{1'b0}} : (norm_mant >> sub_shift);
    wire sub_guard =
        (sub_shift == 0 || sub_shift > WORK_WIDTH) ? 1'b0 :
        ((norm_mant >> (sub_shift - 1)) & 1'b1);
    wire [WORK_WIDTH-1:0] sub_sticky_mask =
        (sub_shift > WORK_WIDTH) ? {WORK_WIDTH{1'b1}} :
        (sub_shift <= 1) ? {WORK_WIDTH{1'b0}} :
        ~({WORK_WIDTH{1'b1}} << (sub_shift - 1));
    wire sub_sticky = |(norm_mant & sub_sticky_mask);
    wire sub_round_up = sub_guard && (sub_sticky || sub_quot[0]);
    wire [MANT_WIDTH:0] sub_rounded =
        {1'b0, sub_quot[MANT_WIDTH-1:0]} + {{MANT_WIDTH{1'b0}}, sub_round_up};
    wire sub_rounds_to_normal = sub_rounded[MANT_WIDTH];

    wire prod_isInf    = p3_diff_isInf && !p3_prod_isZero;
    wire result_isNaN  = p3_y0_isNaN || p3_diff_isNaN ||
                          (p3_y0_isInf && prod_isInf && (p3_y0_sign != p3_diff_sign));
    wire result_isInf_w = ((p3_y0_isInf || prod_isInf) && !result_isNaN) ||
                          (!result_isNaN && !p3_y0_isInf && !prod_isInf && (final_exp >= EXP_MAX));
    wire sum_is_zero   = (p3_sum_abs == 0);
    wire result_isZero = (p3_y0_isZero && p3_prod_isZero) ||
                          (sum_is_zero && !p3_y0_isInf && !prod_isInf &&
                           !p3_y0_isNaN && !p3_diff_isNaN);
    wire underflow     = (final_exp <= 0) && !result_isZero && !result_isInf_w && !result_isNaN;

    wire fma_out_sign = result_isNaN   ? 1'b0 :
                        result_isZero  ? (p3_y0_sign & p3_diff_sign) :
                        result_isInf_w ? (p3_y0_isInf ? p3_y0_sign : p3_diff_sign) :
                        p3_result_sign;

    wire [EXP_WIDTH-1:0] fma_out_exp = result_isZero ? {EXP_WIDTH{1'b0}} :
                                        (result_isInf_w || result_isNaN) ? {EXP_WIDTH{1'b1}} :
                                        underflow ? (GRADUAL_UNDERFLOW && sub_rounds_to_normal ?
                                                     {{EXP_WIDTH-1{1'b0}}, 1'b1} :
                                                     {EXP_WIDTH{1'b0}}) :
                                        final_exp[EXP_WIDTH-1:0];

    wire [MANT_WIDTH-1:0] fma_out_mant = result_isZero  ? {MANT_WIDTH{1'b0}} :
                                          result_isInf_w ? {MANT_WIDTH{1'b0}} :
                                          result_isNaN   ? {{MANT_WIDTH-1{1'b0}}, 1'b1} :
                                          underflow      ? (GRADUAL_UNDERFLOW && sub_rounds_to_normal ?
                                                            {MANT_WIDTH{1'b0}} :
                                                            GRADUAL_UNDERFLOW ?
                                                            sub_rounded[MANT_WIDTH-1:0] :
                                                            {MANT_WIDTH{1'b0}}) :
                                          final_mant;

    wire [15:0] interp_result = {fma_out_sign, fma_out_exp, fma_out_mant};

    wire [15:0] final_result = p3_engine_isNaN ? {1'b0, 5'h1F, 10'h200} :
                               p3_clamp        ? p3_y0 :
                               interp_result;

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

    // Pipeline latency: 5 clock cycles

endmodule
