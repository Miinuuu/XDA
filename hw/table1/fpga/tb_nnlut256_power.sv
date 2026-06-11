`timescale 1ns/1ps

module tb_nn_lut_engine;

    localparam CLK_PERIOD = 10;
    localparam N_SEGMENTS = 256;
    localparam N_BP       = 257;
    localparam LUT_DEPTH  = 256;
    localparam PIPELINE_LATENCY = 4;
    localparam N_ROUNDS = 10;

    reg         clk, rst_n;
    reg         i_valid;
    reg  [15:0] i_data;
    wire        o_valid;
    wire [15:0] o_data;
    reg         cfg_we;
    reg  [1:0]  cfg_sel;
    reg  [8:0]  cfg_addr;
    reg  [15:0] cfg_wdata;

    // No parameters — funcsim is already elaborated with 256 segments
    nn_lut_engine u_dut (
        .clk(clk), .rst_n(rst_n),
        .i_valid(i_valid), .i_data(i_data),
        .o_valid(o_valid), .o_data(o_data),
        .cfg_we(cfg_we), .cfg_sel(cfg_sel),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [15:0] bp_mem    [0:N_BP-1];
    reg [15:0] scale_mem [0:N_SEGMENTS-1];
    reg [15:0] lut_mem   [0:LUT_DEPTH-1];

    localparam MAX_TESTS = 1024;
    reg [15:0] test_input [0:MAX_TESTS-1];
    integer    num_tests, i, round;

    initial begin
        rst_n = 0; i_valid = 0; i_data = 0;
        cfg_we = 0; cfg_sel = 0; cfg_addr = 0; cfg_wdata = 0;

        $readmemh("breakpoint_reg.mem", bp_mem);
        $readmemh("scale_reg.mem", scale_mem);
        $readmemh("lut_reg.mem", lut_mem);
        $readmemh("test_vectors.mem", test_input);

        num_tests = 0;
        for (i = 0; i < MAX_TESTS; i = i + 1)
            if (test_input[i] !== 16'hxxxx) num_tests = num_tests + 1;
            else i = MAX_TESTS;

        // Reset
        repeat(3) @(negedge clk); rst_n = 1; @(negedge clk);

        // Load LUT first (same principle as EDA — avoid X on read path)
        for (i = 0; i < LUT_DEPTH; i = i + 1) begin
            @(negedge clk); cfg_we = 1; cfg_sel = 2'd2; cfg_addr = i[8:0]; cfg_wdata = lut_mem[i];
        end
        // Load breakpoints
        for (i = 0; i < N_BP; i = i + 1) begin
            @(negedge clk); cfg_we = 1; cfg_sel = 2'd0; cfg_addr = i[8:0]; cfg_wdata = bp_mem[i];
        end
        // Load scale factors
        for (i = 0; i < N_SEGMENTS; i = i + 1) begin
            @(negedge clk); cfg_we = 1; cfg_sel = 2'd1; cfg_addr = i[8:0]; cfg_wdata = scale_mem[i];
        end
        @(negedge clk); cfg_we = 0; cfg_sel = 0; cfg_addr = 0; cfg_wdata = 0;
        repeat(5) @(posedge clk);

        // Data phase: N_ROUNDS x test vectors
        for (round = 0; round < N_ROUNDS; round = round + 1) begin
            for (i = 0; i < num_tests; i = i + 1) begin
                @(negedge clk); i_valid = 1; i_data = test_input[i];
            end
        end
        @(negedge clk); i_valid = 0;
        repeat(PIPELINE_LATENCY + 10) @(posedge clk);
        $display("NN-LUT-256 SAIF: %0d vec x %0d rounds", num_tests, N_ROUNDS);
        $finish;
    end
    initial begin #(CLK_PERIOD * (MAX_TESTS * N_ROUNDS + 3000)); $finish; end
endmodule
