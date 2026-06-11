//==============================================================================
// tb_nn_lut_engine - Testbench for NN-LUT Inference Engine
//==============================================================================

`timescale 1ns/1ps

module tb_nn_lut_engine;

    localparam CLK_PERIOD = 10;
    localparam N_SEGMENTS = 16;
    localparam N_BP       = 17;
    localparam LUT_DEPTH  = 17;
    localparam IDX_BITS   = $clog2(N_SEGMENTS) + 1;
    localparam PIPELINE_LATENCY = 4;

    reg         clk, rst_n;
    reg         i_valid;
    reg  [15:0] i_data;
    wire        o_valid;
    wire [15:0] o_data;

    reg         cfg_we;
    reg  [1:0]  cfg_sel;
    reg  [8:0]  cfg_addr;
    reg  [15:0] cfg_wdata;

    nn_lut_engine u_dut (
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

    // Configuration memories
    reg [15:0] bp_mem    [0:N_BP-1];
    reg [15:0] scale_mem [0:N_SEGMENTS-1];
    reg [15:0] lut_mem   [0:LUT_DEPTH-1];

    // Test vectors
    localparam MAX_TESTS = 1024;
    reg [15:0] test_input  [0:MAX_TESTS-1];
    reg [15:0] test_expect [0:MAX_TESTS-1];
    integer    num_tests;

    integer total_tests, pass_count, fail_count, max_ulp_err;

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
                result = (mant_val == 0) ? 1.0e+30 : 0.0;  // Inf / NaN
            else
                result = (2.0**(exp_val - 15)) * (1.0 + mant_val / 1024.0);
            if (sign) result = -result;
            fp16_to_real = result;
        end
    endfunction

    integer i, j, ulp;

    initial begin
        $dumpfile("nn_lut_engine.vcd");
        $dumpvars(0, tb_nn_lut_engine);

        // Load .mem files
        $readmemh("breakpoint_reg.mem", bp_mem);
        $readmemh("scale_reg.mem", scale_mem);
        $readmemh("lut_reg.mem", lut_mem);
        $readmemh("test_vectors.mem", test_input);
        $readmemh("test_expected.mem", test_expect);

        // Count test vectors
        num_tests = 0;
        for (i = 0; i < MAX_TESTS; i = i + 1) begin
            if (test_input[i] !== 16'hxxxx)
                num_tests = num_tests + 1;
            else
                i = MAX_TESTS;  // break
        end
        $display("=== NN-LUT Engine TB: %0d test vectors ===", num_tests);

        // Reset
        rst_n = 0; i_valid = 0; i_data = 0;
        cfg_we = 0; cfg_sel = 0; cfg_addr = 0; cfg_wdata = 0;
        repeat(3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // Load breakpoints
        for (i = 0; i < N_BP; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1; cfg_sel = 2'd0; cfg_addr = i[8:0]; cfg_wdata = bp_mem[i];
        end
        // Load scale factors
        for (i = 0; i < N_SEGMENTS; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1; cfg_sel = 2'd1; cfg_addr = i[8:0]; cfg_wdata = scale_mem[i];
        end
        // Load LUT values
        for (i = 0; i < LUT_DEPTH; i = i + 1) begin
            @(negedge clk);
            cfg_we = 1; cfg_sel = 2'd2; cfg_addr = i[8:0]; cfg_wdata = lut_mem[i];
        end
        @(negedge clk);
        cfg_we = 0;
        @(negedge clk);

        // Initialize counters
        total_tests = 0; pass_count = 0; fail_count = 0; max_ulp_err = 0;

        // Feed test vectors and check outputs
        fork
            // Producer: feed inputs
            begin
                for (i = 0; i < num_tests; i = i + 1) begin
                    @(negedge clk);
                    i_valid = 1;
                    i_data  = test_input[i];
                end
                @(negedge clk);
                i_valid = 0;
            end
            // Consumer: check outputs
            begin
                // Wait for pipeline latency
                repeat(PIPELINE_LATENCY - 1) @(posedge clk);
                for (j = 0; j < num_tests; j = j + 1) begin
                    @(posedge clk);
                    #1;
                    if (o_valid) begin
                        total_tests = total_tests + 1;
                        ulp = ulp_distance(o_data, test_expect[j]);
                        if (ulp > max_ulp_err) max_ulp_err = ulp;
                        if (ulp <= 4) begin
                            pass_count = pass_count + 1;
                        end else begin
                            fail_count = fail_count + 1;
                            if (fail_count <= 10)
                                $display("FAIL [%0d]: in=%04h  hw=%04h (%e)  ref=%04h (%e)  ulp=%0d",
                                    j, test_input[j],
                                    o_data, fp16_to_real(o_data),
                                    test_expect[j], fp16_to_real(test_expect[j]),
                                    ulp);
                        end
                    end
                end
            end
        join

        // Wait for last output
        repeat(PIPELINE_LATENCY + 2) @(posedge clk);

        $display("");
        $display("=== Results ===");
        $display("  Total : %0d", total_tests);
        $display("  Pass  : %0d  (ulp <= 4)", pass_count);
        $display("  Fail  : %0d", fail_count);
        $display("  MaxULP: %0d", max_ulp_err);
        if (fail_count == 0)
            $display("  >>> ALL PASS <<<");
        else
            $display("  >>> %0d FAILURES <<<", fail_count);

        $finish;
    end

endmodule
