//==============================================================================
// Module: fp32_to_fp16
// Description: Convert FP32 E8M23 to FP16 E5M10 (zero-latency combinational)
//==============================================================================

`timescale 1ns/1ps

module fp32_to_fp16 (
    input  [31:0] fp32_in,
    output [15:0] fp16_out,
    output        is_zero,
    output        is_nan,
    output        is_inf,
    output        overflow,
    output        underflow
);

    wire        fp32_sign = fp32_in[31];
    wire [7:0]  fp32_exp  = fp32_in[30:23];
    wire [22:0] fp32_mant = fp32_in[22:0];

    wire fp32_is_zero    = (fp32_exp == 8'd0) && (fp32_mant == 23'd0);
    wire fp32_is_inf     = (fp32_exp == 8'd255) && (fp32_mant == 23'd0);
    wire fp32_is_nan     = (fp32_exp == 8'd255) && (fp32_mant != 23'd0);
    wire fp32_is_subnorm = (fp32_exp == 8'd0) && (fp32_mant != 23'd0);

    wire signed [8:0] exp_diff = {1'b0, fp32_exp} - 9'd112;

    wire exp_overflow  = !fp32_is_zero && !fp32_is_subnorm &&
                         (exp_diff > 9'd30);
    wire exp_underflow = !fp32_is_zero && !fp32_is_subnorm &&
                         (exp_diff <= 9'd0);

    wire [9:0]  fp16_mant_trunc = fp32_mant[22:13];
    wire        guard = fp32_mant[12];
    wire        round_bit = fp32_mant[11];
    wire        sticky = |fp32_mant[10:0];
    wire        lsb = fp16_mant_trunc[0];

    wire round_up = guard && (round_bit || sticky || lsb);
    wire [10:0] mant_rounded = {1'b0, fp16_mant_trunc} + {10'b0, round_up};
    wire mant_overflow = mant_rounded[10];

    wire [9:0] fp16_mant_final = mant_overflow ? 10'd0 : mant_rounded[9:0];
    wire [5:0] exp_adjusted = exp_diff[5:0] + {5'b0, mant_overflow};

    wire [4:0]  subnorm_shift = (exp_diff <= 0) ? (5'd1 - exp_diff[4:0]) : 5'd0;
    wire [10:0] subnorm_mant_full = {1'b1, fp16_mant_trunc};
    wire [10:0] subnorm_mant_shifted = subnorm_mant_full >> subnorm_shift;
    wire [9:0]  subnorm_mant = subnorm_mant_shifted[9:0];
    wire        subnorm_valid = (subnorm_shift <= 5'd10) && !fp32_is_zero;

    reg [4:0]  fp16_exp_out;
    reg [9:0]  fp16_mant_out;
    reg        out_is_nan, out_is_zero, out_is_inf, out_overflow, out_underflow;

    always @(*) begin
        out_is_nan = 1'b0;
        out_is_zero = 1'b0;
        out_is_inf = 1'b0;
        out_overflow = 1'b0;
        out_underflow = 1'b0;
        fp16_exp_out = 5'd0;
        fp16_mant_out = 10'd0;

        if (fp32_is_zero) begin
            out_is_zero = 1'b1;
        end else if (fp32_is_nan) begin
            fp16_exp_out = 5'd31;
            fp16_mant_out = 10'h200;
            out_is_nan = 1'b1;
        end else if (fp32_is_inf) begin
            fp16_exp_out = 5'd31;
            out_is_inf = 1'b1;
        end else if (exp_overflow || exp_adjusted > 6'd30) begin
            fp16_exp_out = 5'd31;
            out_is_inf = 1'b1;
            out_overflow = 1'b1;
        end else if (exp_underflow) begin
            if (subnorm_valid) begin
                fp16_mant_out = subnorm_mant;
                out_underflow = 1'b1;
            end else begin
                out_is_zero = 1'b1;
                out_underflow = 1'b1;
            end
        end else if (exp_adjusted == 6'd0) begin
            fp16_mant_out = subnorm_mant;
            out_underflow = 1'b1;
        end else begin
            fp16_exp_out = exp_adjusted[4:0];
            fp16_mant_out = fp16_mant_final;
        end
    end

    assign fp16_out = {fp32_sign, fp16_exp_out, fp16_mant_out};
    assign is_zero = out_is_zero;
    assign is_nan = out_is_nan;
    assign is_inf = out_is_inf;
    assign overflow = out_overflow;
    assign underflow = out_underflow;

endmodule
