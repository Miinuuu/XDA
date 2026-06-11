//==============================================================================
// tb_nli_engine_addr_fp32 - Testbench for mixed-precision NLI engine
//==============================================================================

`timescale 1ns/1ps

module tb_nli_engine_addr_fp32;

    localparam CLK_PERIOD = 10;
    localparam M          = 11;
    localparam NUM_INTERVALS = 10;
    localparam LUT_DEPTH  = 259;
    localparam PIPELINE_LATENCY = 4;

    reg         clk, rst_n;
    reg         i_valid;
    reg  [15:0] i_data;
    wire        o_valid;
    wire [15:0] o_data;

    reg         cfg_we;
    reg  [1:0]  cfg_sel;
    reg  [8:0]  cfg_addr;
    reg  [31:0] cfg_wdata;

    nli_engine_addr_fp32 u_dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .i_valid  (i_valid),
        .i_data   (i_data),
        .o_valid  (o_valid),
        .o_data   (o_data),
        .cfg_we   (cfg_we),
        .cfg_sel  (cfg_sel),
        .cfg_addr (cfg_addr),
        .cfg_wdata(cfg_wdata)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [31:0] point_mem [0:M-1];
    reg [31:0] mul_mem   [0:NUM_INTERVALS-1];
    reg [15:0] lut_mem   [0:LUT_DEPTH-1];

    localparam MAX_TESTS = 1024;
    reg [15:0] test_input  [0:MAX_TESTS-1];
    reg [15:0] test_expect [0:MAX_TESTS-1];
    integer    num_tests;

    integer total_tests, pass_count, fail_count;
    integer max_ulp_err;

    function automatic integer ulp_distance;
        input [15:0] a, b;
        reg signed [16:0] a_int, b_int;
        begin
            a_int = a[15] ? -(a[14:0]) : a[14:0];
            b_int = b[15] ? -(b[14:0]) : b[14:0];
            ulp_distance = (a_int > b_int) ? (a_int - b_int) : (b_int - a_int);
        end
    endfunction

    function automatic real fp16_to_real;
        input [15:0] bits;
        reg sign;
        reg [4:0] exp_val;
        reg [9:0] mant_val;
        real result;
        begin
            sign = bits[15];
            exp_val = bits[14:10];
            mant_val = bits[9:0];
            if (exp_val == 0 && mant_val == 0)
                result = 0.0;
            else if (exp_val == 0)
                result = (2.0**(-14)) * (mant_val / 1024.0);
            else if (exp_val == 31)
                result = (mant_val == 0) ? 1.0/0.0 : 0.0/0.0;
            else
                result = (2.0**(exp_val - 15)) * (1.0 + mant_val / 1024.0);
            fp16_to_real = sign ? -result : result;
        end
    endfunction

    integer i, j;

    initial begin
        $dumpfile("nli_engine_addr_fp32.vcd");
        $dumpvars(0, tb_nli_engine_addr_fp32);

        rst_n = 0;
        i_valid = 0;
        i_data = 0;
        cfg_we = 0;
        cfg_sel = 0;
        cfg_addr = 0;
        cfg_wdata = 0;

        $readmemh("point_reg.mem", point_mem);
        $readmemh("mul_reg.mem",   mul_mem);
        $readmemh("lut_reg.mem",   lut_mem);

        begin
            reg [31:0] tv_mem [0:MAX_TESTS*2-1];
            $readmemh("test_vectors.mem", tv_mem);
            num_tests = 0;
            for (i = 0; i < MAX_TESTS; i = i + 1) begin
                if (tv_mem[i*2] !== 32'hxxxxxxxx) begin
                    test_input[i]  = tv_mem[i*2][15:0];
                    test_expect[i] = tv_mem[i*2+1][15:0];
                    num_tests = num_tests + 1;
                end
            end
        end

        $display("=== NLI addr-FP32 Testbench ===");
        $display("  Loaded %0d test vectors", num_tests);

        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        for (i = 0; i < M; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1;
            cfg_sel = 2'd0;
            cfg_addr = i[8:0];
            cfg_wdata = point_mem[i];
        end
        @(negedge clk);
        cfg_we = 0;

        for (i = 0; i < NUM_INTERVALS; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1;
            cfg_sel = 2'd1;
            cfg_addr = i[8:0];
            cfg_wdata = mul_mem[i];
        end
        @(negedge clk);
        cfg_we = 0;

        for (i = 0; i < LUT_DEPTH; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1;
            cfg_sel = 2'd2;
            cfg_addr = i[8:0];
            cfg_wdata = {16'h0000, lut_mem[i]};
        end
        @(negedge clk);
        cfg_we = 0;
        repeat(5) @(posedge clk);

        total_tests = 0;
        pass_count  = 0;
        fail_count  = 0;
        max_ulp_err = 0;

        fork
            begin
                for (i = 0; i < num_tests; i = i + 1) begin
                    @(negedge clk);
                    i_valid = 1;
                    i_data = test_input[i];
                end
                @(negedge clk);
                i_valid = 0;
                i_data = 16'h0000;
            end
            begin
                repeat(PIPELINE_LATENCY - 1) @(posedge clk);
                for (j = 0; j < num_tests; j = j + 1) begin
                    @(posedge clk);
                    #1;
                    if (o_valid) begin
                        integer ulp;
                        total_tests = total_tests + 1;
                        ulp = ulp_distance(o_data, test_expect[j]);
                        if (ulp > max_ulp_err)
                            max_ulp_err = ulp;
                        if (ulp <= 4) begin
                            pass_count = pass_count + 1;
                        end else begin
                            fail_count = fail_count + 1;
                            if (fail_count <= 20) begin
                                $display("  MISMATCH [%0d]: in=%04h got=%04h (%.4f) exp=%04h (%.4f) ulp=%0d",
                                    j, test_input[j],
                                    o_data, fp16_to_real(o_data),
                                    test_expect[j], fp16_to_real(test_expect[j]),
                                    ulp);
                            end
                        end
                    end
                end
            end
        join

        repeat(PIPELINE_LATENCY + 2) @(posedge clk);

        $display("");
        $display("=== Results ===");
        $display("  Total: %0d, Pass: %0d, Fail: %0d", total_tests, pass_count, fail_count);
        $display("  Max ULP error: %0d", max_ulp_err);
        $display("  Pass rate: %0.1f%%",
            (total_tests > 0) ? (100.0 * pass_count / total_tests) : 0.0);

        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> SOME TESTS FAILED <<<");

        repeat(10) @(posedge clk);
        $finish;
    end

    initial begin
        #(CLK_PERIOD * (MAX_TESTS + 500));
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
