// ==============================================================
// tb_eda_nli_kernel - AXI VIP based testbench for EDA-NLI Kernel
//
// Uses Xilinx AXI VIP to verify:
//   1. AXI-Lite config loading (config_rom + func_lut)
//   2. AXI4 read master → NLI compute → AXI4 write master
//   3. Sigmoid function approximation accuracy
// ==============================================================
`timescale 1ns/1ps

module tb_eda_nli_kernel;

  //==========================================================================
  // Parameters
  //==========================================================================
  localparam CLK_PERIOD = 10;  // 100 MHz
  localparam DATA_WIDTH = 512;
  localparam ADDR_WIDTH = 64;
  localparam CTRL_ADDR_WIDTH = 12;
  localparam CTRL_DATA_WIDTH = 32;

  // Register offsets
  localparam USER_CTRL = 7'h10;
  localparam SCALAR00  = 7'h14;
  localparam A_DATA_0  = 7'h1c;
  localparam A_DATA_1  = 7'h20;
  localparam B_DATA_0  = 7'h28;
  localparam B_DATA_1  = 7'h2c;
  localparam CFG_CTRL  = 7'h40;
  localparam CFG_WDATA = 7'h44;

  // Test data size (must be multiple of 32)
  localparam NUM_ELEMENTS = 64;
  localparam XFER_BYTES = NUM_ELEMENTS * 2;  // FP16 = 2 bytes
  localparam NUM_BEATS = NUM_ELEMENTS / 32;  // 512b / 16b = 32 elements per beat

  //==========================================================================
  // Config ROM and LUT data (sigmoid, from .mem files)
  //==========================================================================
  reg [15:0] config_rom [0:63];
  reg [15:0] func_lut [0:255];
  reg [15:0] test_input [0:NUM_ELEMENTS-1];
  reg [15:0] test_expect [0:NUM_ELEMENTS-1];

  //==========================================================================
  // Clock and Reset
  //==========================================================================
  reg ap_clk = 0;
  always #(CLK_PERIOD/2) ap_clk = ~ap_clk;

  //==========================================================================
  // DUT Signals
  //==========================================================================
  // AXI-Lite control
  reg                          s_axi_control_awvalid;
  wire                         s_axi_control_awready;
  reg  [CTRL_ADDR_WIDTH-1:0]   s_axi_control_awaddr;
  reg                          s_axi_control_wvalid;
  wire                         s_axi_control_wready;
  reg  [CTRL_DATA_WIDTH-1:0]   s_axi_control_wdata;
  reg  [3:0]                   s_axi_control_wstrb;
  reg                          s_axi_control_arvalid;
  wire                         s_axi_control_arready;
  reg  [CTRL_ADDR_WIDTH-1:0]   s_axi_control_araddr;
  wire                         s_axi_control_rvalid;
  reg                          s_axi_control_rready;
  wire [CTRL_DATA_WIDTH-1:0]   s_axi_control_rdata;
  wire [1:0]                   s_axi_control_rresp;
  wire                         s_axi_control_bvalid;
  reg                          s_axi_control_bready;
  wire [1:0]                   s_axi_control_bresp;
  wire                         interrupt;

  // AXI4 m00_axi (read channel - DUT is master, TB is slave/memory)
  wire                         m00_axi_awvalid;
  wire [ADDR_WIDTH-1:0]        m00_axi_awaddr;
  wire [7:0]                   m00_axi_awlen;
  wire                         m00_axi_wvalid;
  wire [DATA_WIDTH-1:0]        m00_axi_wdata;
  wire [DATA_WIDTH/8-1:0]      m00_axi_wstrb;
  wire                         m00_axi_wlast;
  wire                         m00_axi_bvalid;
  wire                         m00_axi_arvalid;
  reg                          m00_axi_arready;
  wire [ADDR_WIDTH-1:0]        m00_axi_araddr;
  wire [7:0]                   m00_axi_arlen;
  reg                          m00_axi_rvalid;
  wire                         m00_axi_rready;
  reg  [DATA_WIDTH-1:0]        m00_axi_rdata;
  reg                          m00_axi_rlast;

  // AXI4 m01_axi (write channel - DUT is master, TB is slave/memory)
  wire                         m01_axi_awvalid;
  reg                          m01_axi_awready;
  wire [ADDR_WIDTH-1:0]        m01_axi_awaddr;
  wire [7:0]                   m01_axi_awlen;
  wire                         m01_axi_wvalid;
  reg                          m01_axi_wready;
  wire [DATA_WIDTH-1:0]        m01_axi_wdata;
  wire [DATA_WIDTH/8-1:0]      m01_axi_wstrb;
  wire                         m01_axi_wlast;
  reg                          m01_axi_bvalid;
  wire                         m01_axi_bready;
  wire                         m01_axi_arvalid;
  wire [ADDR_WIDTH-1:0]        m01_axi_araddr;
  wire [7:0]                   m01_axi_arlen;
  reg                          m01_axi_rvalid;
  wire                         m01_axi_rready;
  reg  [DATA_WIDTH-1:0]        m01_axi_rdata;
  reg                          m01_axi_rlast;

  //==========================================================================
  // Memory model (simple array for AXI read/write)
  //==========================================================================
  reg [DATA_WIDTH-1:0] input_mem  [0:NUM_BEATS-1];
  reg [DATA_WIDTH-1:0] output_mem [0:NUM_BEATS-1];
  integer wr_beat_cnt;

  //==========================================================================
  // DUT instantiation
  //==========================================================================
  EDA #(
    .C_S_AXI_CONTROL_ADDR_WIDTH ( CTRL_ADDR_WIDTH ),
    .C_S_AXI_CONTROL_DATA_WIDTH ( CTRL_DATA_WIDTH ),
    .C_M00_AXI_ADDR_WIDTH       ( ADDR_WIDTH      ),
    .C_M00_AXI_DATA_WIDTH       ( DATA_WIDTH      ),
    .C_M01_AXI_ADDR_WIDTH       ( ADDR_WIDTH      ),
    .C_M01_AXI_DATA_WIDTH       ( DATA_WIDTH      )
  ) dut (
    .ap_clk                  ( ap_clk                  ),
    // m00_axi
    .m00_axi_awvalid         ( m00_axi_awvalid         ),
    .m00_axi_awready         ( 1'b0                    ),
    .m00_axi_awaddr          ( m00_axi_awaddr          ),
    .m00_axi_awlen           ( m00_axi_awlen           ),
    .m00_axi_wvalid          ( m00_axi_wvalid          ),
    .m00_axi_wready          ( 1'b0                    ),
    .m00_axi_wdata           ( m00_axi_wdata           ),
    .m00_axi_wstrb           ( m00_axi_wstrb           ),
    .m00_axi_wlast           ( m00_axi_wlast           ),
    .m00_axi_bvalid          ( 1'b0                    ),
    .m00_axi_bready          (                         ),
    .m00_axi_arvalid         ( m00_axi_arvalid         ),
    .m00_axi_arready         ( m00_axi_arready         ),
    .m00_axi_araddr          ( m00_axi_araddr          ),
    .m00_axi_arlen           ( m00_axi_arlen           ),
    .m00_axi_rvalid          ( m00_axi_rvalid          ),
    .m00_axi_rready          ( m00_axi_rready          ),
    .m00_axi_rdata           ( m00_axi_rdata           ),
    .m00_axi_rlast           ( m00_axi_rlast           ),
    // m01_axi
    .m01_axi_awvalid         ( m01_axi_awvalid         ),
    .m01_axi_awready         ( m01_axi_awready         ),
    .m01_axi_awaddr          ( m01_axi_awaddr          ),
    .m01_axi_awlen           ( m01_axi_awlen           ),
    .m01_axi_wvalid          ( m01_axi_wvalid          ),
    .m01_axi_wready          ( m01_axi_wready          ),
    .m01_axi_wdata           ( m01_axi_wdata           ),
    .m01_axi_wstrb           ( m01_axi_wstrb           ),
    .m01_axi_wlast           ( m01_axi_wlast           ),
    .m01_axi_bvalid          ( m01_axi_bvalid          ),
    .m01_axi_bready          ( m01_axi_bready          ),
    .m01_axi_arvalid         ( m01_axi_arvalid         ),
    .m01_axi_arready         ( 1'b0                    ),
    .m01_axi_araddr          ( m01_axi_araddr          ),
    .m01_axi_arlen           ( m01_axi_arlen           ),
    .m01_axi_rvalid          ( 1'b0                    ),
    .m01_axi_rready          (                         ),
    .m01_axi_rdata           ( {DATA_WIDTH{1'b0}}      ),
    .m01_axi_rlast           ( 1'b0                    ),
    // s_axi_control
    .s_axi_control_awvalid   ( s_axi_control_awvalid   ),
    .s_axi_control_awready   ( s_axi_control_awready   ),
    .s_axi_control_awaddr    ( s_axi_control_awaddr    ),
    .s_axi_control_wvalid    ( s_axi_control_wvalid    ),
    .s_axi_control_wready    ( s_axi_control_wready    ),
    .s_axi_control_wdata     ( s_axi_control_wdata     ),
    .s_axi_control_wstrb     ( s_axi_control_wstrb     ),
    .s_axi_control_arvalid   ( s_axi_control_arvalid   ),
    .s_axi_control_arready   ( s_axi_control_arready   ),
    .s_axi_control_araddr    ( s_axi_control_araddr    ),
    .s_axi_control_rvalid    ( s_axi_control_rvalid    ),
    .s_axi_control_rready    ( s_axi_control_rready    ),
    .s_axi_control_rdata     ( s_axi_control_rdata     ),
    .s_axi_control_rresp     ( s_axi_control_rresp     ),
    .s_axi_control_bvalid    ( s_axi_control_bvalid    ),
    .s_axi_control_bready    ( s_axi_control_bready    ),
    .s_axi_control_bresp     ( s_axi_control_bresp     ),
    .interrupt               ( interrupt               )
  );

  //==========================================================================
  // AXI-Lite write task
  //==========================================================================
  task axi_lite_write(input [CTRL_ADDR_WIDTH-1:0] addr, input [31:0] data);
    begin
      // Address phase
      @(posedge ap_clk);
      s_axi_control_awvalid <= 1'b1;
      s_axi_control_awaddr  <= addr;
      s_axi_control_wvalid  <= 1'b1;
      s_axi_control_wdata   <= data;
      s_axi_control_wstrb   <= 4'hF;

      // Wait for AW handshake
      @(posedge ap_clk);
      while (!s_axi_control_awready) @(posedge ap_clk);
      s_axi_control_awvalid <= 1'b0;

      // Wait for W handshake
      while (!s_axi_control_wready) @(posedge ap_clk);
      s_axi_control_wvalid <= 1'b0;

      // Wait for B response
      s_axi_control_bready <= 1'b1;
      while (!s_axi_control_bvalid) @(posedge ap_clk);
      @(posedge ap_clk);
      s_axi_control_bready <= 1'b0;
    end
  endtask

  //==========================================================================
  // AXI-Lite read task
  //==========================================================================
  task axi_lite_read(input [CTRL_ADDR_WIDTH-1:0] addr, output [31:0] data);
    begin
      @(posedge ap_clk);
      s_axi_control_arvalid <= 1'b1;
      s_axi_control_araddr  <= addr;

      while (!s_axi_control_arready) @(posedge ap_clk);
      s_axi_control_arvalid <= 1'b0;

      s_axi_control_rready <= 1'b1;
      while (!s_axi_control_rvalid) @(posedge ap_clk);
      data = s_axi_control_rdata;
      @(posedge ap_clk);
      s_axi_control_rready <= 1'b0;
    end
  endtask

  //==========================================================================
  // AXI4 Read Slave Model (for m00_axi - serves input data)
  // Handles multiple AR transactions, wraps data around input_mem
  //==========================================================================
  reg [7:0]  rd_burst_len;
  reg [7:0]  rd_beat_cnt;
  reg        rd_active;
  integer    rd_global_beat;  // global beat counter across bursts

  initial begin
    rd_active = 0;
    rd_global_beat = 0;
    m00_axi_arready = 0;
    m00_axi_rvalid = 0;
    m00_axi_rdata = 0;
    m00_axi_rlast = 0;
  end

  // AR channel: always ready when not serving data
  always @(posedge ap_clk) begin
    if (!rd_active) begin
      m00_axi_arready <= 1'b1;
      if (m00_axi_arvalid && m00_axi_arready) begin
        rd_burst_len    <= m00_axi_arlen;
        rd_beat_cnt     <= 8'd0;
        rd_active       <= 1'b1;
        m00_axi_arready <= 1'b0;
      end
    end else begin
      m00_axi_arready <= 1'b0;
    end
  end

  // R channel: serve data beats
  always @(posedge ap_clk) begin
    if (rd_active) begin
      m00_axi_rvalid <= 1'b1;
      m00_axi_rdata  <= input_mem[rd_global_beat % NUM_BEATS];
      m00_axi_rlast  <= (rd_beat_cnt == rd_burst_len);

      if (m00_axi_rready && m00_axi_rvalid) begin
        rd_global_beat <= rd_global_beat + 1;
        if (rd_beat_cnt == rd_burst_len) begin
          rd_active      <= 1'b0;
          m00_axi_rvalid <= 1'b0;
          m00_axi_rlast  <= 1'b0;
        end else begin
          rd_beat_cnt <= rd_beat_cnt + 1;
        end
      end
    end else begin
      m00_axi_rvalid <= 1'b0;
      m00_axi_rlast  <= 1'b0;
    end
  end

  //==========================================================================
  // AXI4 Write Slave Model (for m01_axi - captures output data)
  //==========================================================================
  always @(posedge ap_clk) begin
    // AW channel: always ready
    m01_axi_awready <= 1'b1;
    // W channel: always ready
    m01_axi_wready <= 1'b1;

    if (m01_axi_wvalid && m01_axi_wready) begin
      output_mem[wr_beat_cnt] <= m01_axi_wdata;
      wr_beat_cnt <= wr_beat_cnt + 1;
    end

    // B channel: respond after wlast
    if (m01_axi_wvalid && m01_axi_wready && m01_axi_wlast)
      m01_axi_bvalid <= 1'b1;
    else if (m01_axi_bvalid && m01_axi_bready)
      m01_axi_bvalid <= 1'b0;
  end

  //==========================================================================
  // FP16 helper functions
  //==========================================================================
  function automatic real fp16_to_real(input [15:0] h);
    reg sign;
    reg [4:0] exp;
    reg [9:0] mant;
    real result;
    begin
      sign = h[15];
      exp  = h[14:10];
      mant = h[9:0];

      if (exp == 0 && mant == 0) begin
        result = 0.0;
      end else if (exp == 0) begin
        result = (2.0**(-14)) * (real'(mant) / 1024.0);
      end else if (exp == 31) begin
        result = (mant == 0) ? 1.0/0.0 : 0.0/0.0;  // inf or nan
      end else begin
        result = (2.0**(real'(exp) - 15.0)) * (1.0 + real'(mant) / 1024.0);
      end

      if (sign) result = -result;
      fp16_to_real = result;
    end
  endfunction

  function automatic [15:0] real_to_fp16(input real val);
    reg [15:0] result;
    reg sign;
    integer exp_int;
    integer mant_int;
    real abs_val, mant_real;
    reg [4:0] biased_exp;
    reg [9:0] mant_bits;
    begin
      sign = (val < 0.0) ? 1'b1 : 1'b0;
      abs_val = (val < 0.0) ? -val : val;

      if (abs_val == 0.0) begin
        result = {sign, 15'b0};
      end else if (abs_val >= 65504.0) begin
        result = {sign, 5'b11111, 10'b0};  // infinity
      end else begin
        exp_int = 0;
        mant_real = abs_val;
        while (mant_real >= 2.0) begin mant_real = mant_real / 2.0; exp_int = exp_int + 1; end
        while (mant_real < 1.0 && exp_int > -14) begin mant_real = mant_real * 2.0; exp_int = exp_int - 1; end

        if (exp_int < -14) begin
          // subnormal
          mant_real = abs_val / (2.0**(-14));
          mant_int = $rtoi(mant_real * 1024.0);
          mant_bits = mant_int[9:0];
          result = {sign, 5'b0, mant_bits};
        end else begin
          biased_exp = exp_int[4:0] + 5'd15;
          mant_int = $rtoi((mant_real - 1.0) * 1024.0);
          mant_bits = mant_int[9:0];
          result = {sign, biased_exp, mant_bits};
        end
      end
      real_to_fp16 = result;
    end
  endfunction

  //==========================================================================
  // Main Test Sequence
  //==========================================================================
  integer i, j, errors;
  reg [31:0] read_data;
  real x_val, y_hw, y_ref, abs_err, max_err;

  initial begin
    $display("=== EDA-NLI Kernel AXI VIP Testbench ===");
    $display("NUM_ELEMENTS=%0d, NUM_BEATS=%0d", NUM_ELEMENTS, NUM_BEATS);

    // Initialize signals
    s_axi_control_awvalid <= 0;
    s_axi_control_awaddr  <= 0;
    s_axi_control_wvalid  <= 0;
    s_axi_control_wdata   <= 0;
    s_axi_control_wstrb   <= 0;
    s_axi_control_arvalid <= 0;
    s_axi_control_araddr  <= 0;
    s_axi_control_rready  <= 0;
    s_axi_control_bready  <= 0;

    m00_axi_arready <= 0;
    m00_axi_rvalid  <= 0;
    m00_axi_rdata   <= 0;
    m00_axi_rlast   <= 0;

    m01_axi_awready <= 0;
    m01_axi_wready  <= 0;
    m01_axi_bvalid  <= 0;
    m01_axi_rvalid  <= 0;
    m01_axi_rdata   <= 0;
    m01_axi_rlast   <= 0;

    rd_active <= 0;
    wr_beat_cnt <= 0;

    // Load config data from .mem files (path relative to xsim working dir)
    $readmemh("config/config_rom.mem", config_rom);
    $readmemh("config/func_lut.mem", func_lut);

    // Generate test input: linspace(-8, 8, NUM_ELEMENTS) as FP16
    for (i = 0; i < NUM_ELEMENTS; i++) begin
      x_val = -8.0 + 16.0 * real'(i) / real'(NUM_ELEMENTS - 1);
      test_input[i] = real_to_fp16(x_val);
    end

    // Pack test inputs into 512-bit memory words
    for (i = 0; i < NUM_BEATS; i++) begin
      input_mem[i] = {DATA_WIDTH{1'b0}};
      for (j = 0; j < 32; j++) begin
        if (i*32 + j < NUM_ELEMENTS)
          input_mem[i][j*16 +: 16] = test_input[i*32 + j];
      end
    end

    // Initialize output memory
    for (i = 0; i < NUM_BEATS; i++)
      output_mem[i] = {DATA_WIDTH{1'b0}};

    // Wait for reset
    repeat(20) @(posedge ap_clk);

    //====================================================================
    // Phase 1: Load NLI Configuration via AXI-Lite
    //====================================================================
    $display("\n--- Phase 1: Loading NLI config ---");

    // Load config_rom (64 entries, cfg_sel=0)
    for (i = 0; i < 64; i++) begin
      axi_lite_write(CFG_WDATA, {16'b0, config_rom[i]});
      axi_lite_write(CFG_CTRL, (i << 1) | 0);
    end
    $display("  config_rom loaded (64 entries)");

    // Load func_lut (256 entries, cfg_sel=1)
    for (i = 0; i < 256; i++) begin
      axi_lite_write(CFG_WDATA, {16'b0, func_lut[i]});
      axi_lite_write(CFG_CTRL, (i << 1) | 1);
    end
    $display("  func_lut loaded (256 entries)");

    repeat(10) @(posedge ap_clk);

    //====================================================================
    // Phase 2: Configure and Start Kernel
    //====================================================================
    $display("\n--- Phase 2: Starting kernel ---");

    // Set transfer size (bytes)
    axi_lite_write(SCALAR00, XFER_BYTES);

    // Set A pointer (input) - address 0x0000
    axi_lite_write(A_DATA_0, 32'h0000_0000);
    axi_lite_write(A_DATA_1, 32'h0000_0000);

    // Set B pointer (output) - address 0x1000
    axi_lite_write(B_DATA_0, 32'h0000_1000);
    axi_lite_write(B_DATA_1, 32'h0000_0000);

    // Start kernel
    axi_lite_write(USER_CTRL, 32'h0000_0001);
    $display("  Kernel started, transfer=%0d bytes", XFER_BYTES);

    //====================================================================
    // Phase 3: Wait for completion
    //====================================================================
    $display("\n--- Phase 3: Waiting for completion ---");

    // Poll USER_CTRL for done bit
    read_data = 0;
    while (!(read_data & 32'h2)) begin
      repeat(100) @(posedge ap_clk);
      axi_lite_read(USER_CTRL, read_data);
    end
    $display("  Kernel done!");

    repeat(10) @(posedge ap_clk);

    //====================================================================
    // Phase 4: Verify results
    //====================================================================
    $display("\n--- Phase 4: Verifying results ---");

    errors = 0;
    max_err = 0.0;

    for (i = 0; i < NUM_ELEMENTS; i++) begin
      // Extract output from memory beats
      y_hw  = fp16_to_real(output_mem[i/32][(i%32)*16 +: 16]);
      x_val = fp16_to_real(test_input[i]);

      // Reference sigmoid
      y_ref = 1.0 / (1.0 + $exp(-x_val));

      abs_err = (y_hw > y_ref) ? (y_hw - y_ref) : (y_ref - y_hw);
      if (abs_err > max_err) max_err = abs_err;

      if (abs_err > 0.02) begin  // FP16 tolerance
        errors = errors + 1;
        if (errors <= 10)
          $display("  ERROR [%3d] x=%f hw=%f ref=%f err=%e",
                   i, x_val, y_hw, y_ref, abs_err);
      end

      if (i < 10 || i == NUM_ELEMENTS/2 || i >= NUM_ELEMENTS-5) begin
        $display("  [%3d] x=%8.4f  hw=%8.6f  ref=%8.6f  err=%e",
                 i, x_val, y_hw, y_ref, abs_err);
      end
    end

    $display("\n--- Summary ---");
    $display("  Elements: %0d", NUM_ELEMENTS);
    $display("  Max abs error: %e", max_err);
    $display("  Errors (>0.02): %0d", errors);

    if (errors == 0)
      $display("\n*** TEST PASSED ***\n");
    else
      $display("\n*** TEST FAILED (%0d errors) ***\n", errors);

    repeat(20) @(posedge ap_clk);
    $finish;
  end

  //==========================================================================
  // Timeout watchdog
  //==========================================================================
  initial begin
    #(CLK_PERIOD * 500000);
    $display("ERROR: Simulation timeout!");
    $finish;
  end

endmodule
