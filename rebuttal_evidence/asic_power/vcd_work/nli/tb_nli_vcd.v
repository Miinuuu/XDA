`timescale 1ns/1ps
// Gate-level VCD-activity testbench for NLI (nli_engine), same stimulus style as
// the XDA tb. Config = 3 banks (point sel=0, mul sel=1, lut sel=2). VCD starts at
// the steady-state data phase (continuous, no $dumpoff/$dumpon).
module tb_nli_engine;
    localparam CLK_PERIOD = 10;
    localparam PIPELINE_LATENCY = 4;
    localparam N_ROUNDS = 10;
    localparam N_WARMUP = 20;
    localparam M = 11, NUM_INTERVALS = 10, LUT_DEPTH = 259;

    reg         clk, rst_n, i_valid;
    reg  [15:0] i_data;
    wire        o_valid;
    wire [15:0] o_data;
    reg         cfg_we;
    reg  [1:0]  cfg_sel;
    reg  [8:0]  cfg_addr;
    reg  [15:0] cfg_wdata;

    nli_engine u_dut (
        .clk(clk), .rst_n(rst_n), .i_valid(i_valid), .i_data(i_data),
        .o_valid(o_valid), .o_data(o_data), .cfg_we(cfg_we), .cfg_sel(cfg_sel),
        .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [15:0] point_mem [0:M-1];
    reg [15:0] mul_mem   [0:NUM_INTERVALS-1];
    reg [15:0] lut_mem   [0:LUT_DEPTH-1];
    localparam MAX_TESTS = 1024;
    reg [15:0] test_input [0:MAX_TESTS-1];
    integer    num_tests, i, round;

    initial begin
        rst_n=0; i_valid=0; i_data=0; cfg_we=0; cfg_sel=0; cfg_addr=0; cfg_wdata=0;
        $readmemh("point_reg.mem", point_mem);
        $readmemh("mul_reg.mem",   mul_mem);
        $readmemh("lut_reg.mem",   lut_mem);
        begin : load_tv
            reg [31:0] tv_mem [0:MAX_TESTS*2-1];
            $readmemh("test_vectors.mem", tv_mem);
            num_tests = 0;
            for (i = 0; i < MAX_TESTS; i = i + 1)
                if (tv_mem[i*2] !== 32'hxxxxxxxx) begin
                    test_input[i] = tv_mem[i*2][15:0]; num_tests = num_tests + 1;
                end
        end

        repeat(5) @(posedge clk); rst_n = 1; repeat(2) @(posedge clk);

        // Load config banks: point (sel0), mul (sel1), lut (sel2)
        for (i=0;i<M;i=i+1)             begin @(negedge clk); cfg_we=1; cfg_sel=2'd0; cfg_addr=i[8:0]; cfg_wdata=point_mem[i]; end
        @(negedge clk); cfg_we=0;
        for (i=0;i<NUM_INTERVALS;i=i+1) begin @(negedge clk); cfg_we=1; cfg_sel=2'd1; cfg_addr=i[8:0]; cfg_wdata=mul_mem[i];   end
        @(negedge clk); cfg_we=0;
        for (i=0;i<LUT_DEPTH;i=i+1)     begin @(negedge clk); cfg_we=1; cfg_sel=2'd2; cfg_addr=i[8:0]; cfg_wdata=lut_mem[i];   end
        @(negedge clk); cfg_we=0; cfg_sel=0; cfg_addr=0; cfg_wdata=0;
        repeat(5) @(posedge clk);

        // Warmup (flush X through pipeline)
        for (i=0;i<N_WARMUP;i=i+1) begin @(negedge clk); i_valid=1; i_data=test_input[i % num_tests]; end
        @(negedge clk); i_valid=0;
        repeat(PIPELINE_LATENCY + 2) @(posedge clk);

        if (^o_data === 1'bx) $display("XCHECK_FAIL o_data has X after warmup: %h", o_data);
        else                  $display("XCHECK_OK o_data X-free after warmup: %h", o_data);
        $display("WARMUP_DONE %0t", $time);

        // Data phase: continuous VCD of u_dut only
        $dumpfile("nli.vcd");
        $dumpvars(0, u_dut);
        for (round=0; round<N_ROUNDS; round=round+1)
            for (i=0;i<num_tests;i=i+1) begin @(negedge clk); i_valid=1; i_data=test_input[i]; end
        @(negedge clk); i_valid=0;
        repeat(PIPELINE_LATENCY + 10) @(posedge clk);
        $display("NLI VCD: %0d vec x %0d rounds, warmup=%0d", num_tests, N_ROUNDS, N_WARMUP);
        $finish;
    end
    initial begin #(CLK_PERIOD * (MAX_TESTS * N_ROUNDS + 3000)); $finish; end
endmodule
