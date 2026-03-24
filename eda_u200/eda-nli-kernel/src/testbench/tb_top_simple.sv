// ==============================================================
// tb_top_simple - Top-level (EDA) testbench
// No VIP, simple AXI models. Tests full path:
//   AXI-Lite config → AXI4 read → NLI compute → AXI4 write
// ==============================================================
`timescale 1ns/1ps

module tb_top_simple;

  localparam CLK_PERIOD = 10;
  localparam NUM_ELEMENTS = 32;
  localparam XFER_BYTES = NUM_ELEMENTS * 2;
  localparam NUM_BEATS = NUM_ELEMENTS / 32;

  //==========================================================================
  // Clock / Reset
  //==========================================================================
  reg ap_clk = 0;
  reg ap_rst_n = 0;
  always #(CLK_PERIOD/2) ap_clk = ~ap_clk;

  //==========================================================================
  // AXI-Lite
  //==========================================================================
  reg  [11:0] ctrl_awaddr = 0;
  reg         ctrl_awvalid = 0;
  wire        ctrl_awready;
  reg  [31:0] ctrl_wdata = 0;
  reg  [3:0]  ctrl_wstrb = 4'hF;
  reg         ctrl_wvalid = 0;
  wire        ctrl_wready;
  wire [1:0]  ctrl_bresp;
  wire        ctrl_bvalid;
  reg         ctrl_bready = 0;
  reg  [11:0] ctrl_araddr = 0;
  reg         ctrl_arvalid = 0;
  wire        ctrl_arready;
  wire [31:0] ctrl_rdata;
  wire [1:0]  ctrl_rresp;
  wire        ctrl_rvalid;
  reg         ctrl_rready = 0;

  //==========================================================================
  // m00_axi (read side)
  //==========================================================================
  wire        m00_arvalid;
  reg         m00_arready = 0;
  wire [63:0] m00_araddr;
  wire [7:0]  m00_arlen;
  reg         m00_rvalid = 0;
  wire        m00_rready;
  reg  [511:0] m00_rdata = 0;
  reg         m00_rlast = 0;

  //==========================================================================
  // m01_axi (write side)
  //==========================================================================
  wire        m01_awvalid;
  reg         m01_awready = 1;
  wire [63:0] m01_awaddr;
  wire [7:0]  m01_awlen;
  wire        m01_wvalid;
  reg         m01_wready = 1;
  wire [511:0] m01_wdata;
  wire [63:0] m01_wstrb;
  wire        m01_wlast;
  reg         m01_bvalid = 0;
  wire        m01_bready;

  wire interrupt;

  //==========================================================================
  // Memory models
  //==========================================================================
  reg [511:0] input_mem [0:7];
  reg [511:0] output_mem [0:7];
  integer wr_idx = 0;

  //==========================================================================
  // DUT
  //==========================================================================
  EDA #(
    .C_S_AXI_CONTROL_ADDR_WIDTH(12), .C_S_AXI_CONTROL_DATA_WIDTH(32),
    .C_M00_AXI_ADDR_WIDTH(64), .C_M00_AXI_DATA_WIDTH(512),
    .C_M01_AXI_ADDR_WIDTH(64), .C_M01_AXI_DATA_WIDTH(512)
  ) dut (
    .ap_clk(ap_clk), .ap_rst_n(ap_rst_n),
    // m00: read only, tie off writes
    .m00_axi_awvalid(), .m00_axi_awready(1'b0), .m00_axi_awaddr(), .m00_axi_awlen(),
    .m00_axi_wvalid(), .m00_axi_wready(1'b0), .m00_axi_wdata(), .m00_axi_wstrb(), .m00_axi_wlast(),
    .m00_axi_bvalid(1'b0), .m00_axi_bready(),
    .m00_axi_arvalid(m00_arvalid), .m00_axi_arready(m00_arready),
    .m00_axi_araddr(m00_araddr), .m00_axi_arlen(m00_arlen),
    .m00_axi_rvalid(m00_rvalid), .m00_axi_rready(m00_rready),
    .m00_axi_rdata(m00_rdata), .m00_axi_rlast(m00_rlast),
    // m01: write only, tie off reads
    .m01_axi_awvalid(m01_awvalid), .m01_axi_awready(m01_awready),
    .m01_axi_awaddr(m01_awaddr), .m01_axi_awlen(m01_awlen),
    .m01_axi_wvalid(m01_wvalid), .m01_axi_wready(m01_wready),
    .m01_axi_wdata(m01_wdata), .m01_axi_wstrb(m01_wstrb), .m01_axi_wlast(m01_wlast),
    .m01_axi_bvalid(m01_bvalid), .m01_axi_bready(m01_bready),
    .m01_axi_arvalid(), .m01_axi_arready(1'b0), .m01_axi_araddr(), .m01_axi_arlen(),
    .m01_axi_rvalid(1'b0), .m01_axi_rready(), .m01_axi_rdata(512'b0), .m01_axi_rlast(1'b0),
    // control
    .s_axi_control_awvalid(ctrl_awvalid), .s_axi_control_awready(ctrl_awready),
    .s_axi_control_awaddr(ctrl_awaddr),
    .s_axi_control_wvalid(ctrl_wvalid), .s_axi_control_wready(ctrl_wready),
    .s_axi_control_wdata(ctrl_wdata), .s_axi_control_wstrb(ctrl_wstrb),
    .s_axi_control_arvalid(ctrl_arvalid), .s_axi_control_arready(ctrl_arready),
    .s_axi_control_araddr(ctrl_araddr),
    .s_axi_control_rvalid(ctrl_rvalid), .s_axi_control_rready(ctrl_rready),
    .s_axi_control_rdata(ctrl_rdata), .s_axi_control_rresp(ctrl_rresp),
    .s_axi_control_bvalid(ctrl_bvalid), .s_axi_control_bready(ctrl_bready),
    .s_axi_control_bresp(ctrl_bresp),
    .interrupt(interrupt)
  );

  //==========================================================================
  // AXI-Lite write task
  //==========================================================================
  task axi_write(input [11:0] addr, input [31:0] data);
    @(posedge ap_clk);
    ctrl_awaddr <= addr; ctrl_awvalid <= 1;
    ctrl_wdata <= data; ctrl_wstrb <= 4'hF; ctrl_wvalid <= 1;
    fork
      begin: aw_ch
        while (!(ctrl_awvalid && ctrl_awready)) @(posedge ap_clk);
        ctrl_awvalid <= 0;
      end
      begin: w_ch
        while (!(ctrl_wvalid && ctrl_wready)) @(posedge ap_clk);
        ctrl_wvalid <= 0;
      end
    join
    ctrl_bready <= 1;
    while (!ctrl_bvalid) @(posedge ap_clk);
    @(posedge ap_clk);
    ctrl_bready <= 0;
  endtask

  task axi_read(input [11:0] addr, output [31:0] data);
    @(posedge ap_clk);
    ctrl_araddr <= addr; ctrl_arvalid <= 1;
    while (!(ctrl_arvalid && ctrl_arready)) @(posedge ap_clk);
    ctrl_arvalid <= 0;
    ctrl_rready <= 1;
    while (!ctrl_rvalid) @(posedge ap_clk);
    data = ctrl_rdata;
    @(posedge ap_clk);
    ctrl_rready <= 0;
  endtask

  //==========================================================================
  // AXI4 Read Slave (m00_axi)
  //==========================================================================
  reg [7:0] rd_len;
  reg [7:0] rd_cnt;
  reg       rd_active = 0;
  integer   rd_global = 0;

  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      m00_arready <= 0; m00_rvalid <= 0; m00_rlast <= 0; rd_active <= 0;
    end else if (!rd_active) begin
      m00_arready <= 1;
      if (m00_arvalid && m00_arready) begin
        rd_len <= m00_arlen; rd_cnt <= 0; rd_active <= 1; m00_arready <= 0;
      end
    end else begin
      m00_arready <= 0;
      m00_rvalid <= 1;
      m00_rdata <= input_mem[rd_global % 8];
      m00_rlast <= (rd_cnt == rd_len);
      if (m00_rvalid && m00_rready) begin
        rd_global <= rd_global + 1;
        if (rd_cnt == rd_len) begin
          rd_active <= 0; m00_rvalid <= 0; m00_rlast <= 0;
        end else
          rd_cnt <= rd_cnt + 1;
      end
    end
  end

  //==========================================================================
  // AXI4 Write Slave (m01_axi)
  //==========================================================================
  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      m01_awready <= 1; m01_wready <= 1; m01_bvalid <= 0; wr_idx <= 0;
    end else begin
      if (m01_wvalid && m01_wready) begin
        output_mem[wr_idx] <= m01_wdata;
        $display("  [WR] beat %0d: data[31:0]=0x%08h", wr_idx, m01_wdata[31:0]);
        wr_idx <= wr_idx + 1;
      end
      if (m01_wvalid && m01_wready && m01_wlast)
        m01_bvalid <= 1;
      else if (m01_bvalid && m01_bready)
        m01_bvalid <= 0;
    end
  end

  //==========================================================================
  // Config data + FP16 helpers
  //==========================================================================
  reg [15:0] config_rom [0:63];
  reg [15:0] func_lut [0:255];

  function automatic real fp16_to_real(input [15:0] h);
    reg [4:0] e; reg [9:0] m; real r;
    begin
      e = h[14:10]; m = h[9:0];
      if (e==0 && m==0) r = 0.0;
      else if (e==0) r = (2.0**(-14)) * (real'(m)/1024.0);
      else if (e==31) r = 99999.0;
      else r = (2.0**(real'(e)-15.0)) * (1.0 + real'(m)/1024.0);
      if (h[15]) r = -r;
      fp16_to_real = r;
    end
  endfunction

  //==========================================================================
  // Main Test
  //==========================================================================
  integer i, errors;
  reg [31:0] rd_data;
  real x_val, y_hw, y_ref, abs_err, max_err;

  initial begin
    $display("=== Top-Level (EDA) Testbench ===");
    $readmemh("config/config_rom.mem", config_rom);
    $readmemh("config/func_lut.mem", func_lut);

    // Prepare input: x=0.0 (sigmoid=0.5) for all 32 elements
    input_mem[0] = {32{16'h0000}};  // all zeros = FP16 0.0

    // === Reset ===
    $display("\n--- Reset (20 cycles) ---");
    ap_rst_n = 0;
    repeat(20) @(posedge ap_clk);

    // Check reset propagation
    $display("  control_s_axi ARESET = %0b (expect 1)", dut.inst_control_s_axi.ARESET);
    $display("  wrapper areset = %0b (expect 1)", dut.EDA_NLI_inst.areset);
    $display("  NLI rst_n = %0b (expect 0)", dut.EDA_NLI_inst.u_compute.rst_n);

    ap_rst_n = 1;
    repeat(10) @(posedge ap_clk);
    $display("  After reset deassert:");
    $display("  control_s_axi ARESET = %0b (expect 0)", dut.inst_control_s_axi.ARESET);
    $display("  wrapper areset = %0b (expect 0)", dut.EDA_NLI_inst.areset);
    $display("  compute state = %0d (expect 0=IDLE)", dut.EDA_NLI_inst.u_compute.state);

    // === Load Config ===
    $display("\n--- Load Config ---");
    for (i = 0; i < 64; i++) begin
      axi_write(12'h44, {16'b0, config_rom[i]});
      axi_write(12'h40, (i << 1) | 0);
    end
    $display("  config_rom loaded");

    for (i = 0; i < 256; i++) begin
      axi_write(12'h44, {16'b0, func_lut[i]});
      axi_write(12'h40, (i << 1) | 1);
    end
    $display("  func_lut loaded");
    $display("  LUT[176]=0x%04h (0.5)", dut.EDA_NLI_inst.u_compute.u_nli_engine.func_lut[176]);

    // === Start Kernel ===
    $display("\n--- Start Kernel ---");
    axi_write(12'h14, XFER_BYTES);             // scalar00 = transfer bytes
    axi_write(12'h1c, 32'h0); axi_write(12'h20, 32'h0);  // A = 0
    axi_write(12'h28, 32'h1000); axi_write(12'h2c, 32'h0); // B = 0x1000
    axi_write(12'h10, 32'h1);                  // START!
    $display("  Kernel started at t=%0t", $time);

    // === Wait for Done ===
    rd_data = 0;
    while (!(rd_data & 32'h2)) begin
      repeat(20) @(posedge ap_clk);
      axi_read(12'h10, rd_data);
    end
    $display("  Kernel done at t=%0t", $time);

    // === Verify ===
    $display("\n--- Verify ---");
    errors = 0; max_err = 0.0;
    for (i = 0; i < NUM_ELEMENTS; i++) begin
      reg [15:0] oval;
      oval = output_mem[i/32][(i%32)*16 +: 16];
      y_hw = fp16_to_real(oval);
      y_ref = 0.5;  // sigmoid(0) = 0.5
      abs_err = (y_hw > y_ref) ? y_hw - y_ref : y_ref - y_hw;
      if (abs_err > max_err) max_err = abs_err;
      if (abs_err > 0.01) errors++;
      if (i < 5 || i >= NUM_ELEMENTS-2)
        $display("  [%2d] hw=%f (0x%04h) ref=%f err=%e", i, y_hw, oval, y_ref, abs_err);
    end

    $display("\n  Max error: %e, Errors: %0d", max_err, errors);
    if (errors == 0) $display("\n*** TEST PASSED ***");
    else $display("\n*** TEST FAILED ***");

    #200 $finish;
  end

  initial begin #(CLK_PERIOD*200000); $display("TIMEOUT!"); $finish; end
endmodule
