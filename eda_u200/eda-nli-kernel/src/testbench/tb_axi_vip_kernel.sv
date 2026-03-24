// ==============================================================
// tb_axi_vip_kernel - AXI VIP IP based testbench
//
// Uses Xilinx AXI VIP IP:
//   - control_eda_nli_vip: AXI-Lite Master (drives s_axi_control)
//   - slv_m00_axi_vip:  AXI4 Slave Memory (responds to m00_axi reads)
//   - slv_m01_axi_vip:  AXI4 Slave Memory (accepts m01_axi writes)
// ==============================================================
`timescale 1ns/1ps

import axi_vip_pkg::*;
import control_eda_nli_vip_pkg::*;
import slv_m00_axi_vip_pkg::*;
import slv_m01_axi_vip_pkg::*;

module tb_axi_vip_kernel;

  //==========================================================================
  // Parameters
  //==========================================================================
  localparam CLK_PERIOD = 10;  // 100 MHz
  localparam DATA_WIDTH = 512;
  localparam ADDR_WIDTH = 64;
  localparam CTRL_ADDR_WIDTH = 12;
  localparam CTRL_DATA_WIDTH = 32;

  // Register offsets
  localparam USER_CTRL = 12'h10;
  localparam SCALAR00  = 12'h14;
  localparam A_DATA_0  = 12'h1c;
  localparam A_DATA_1  = 12'h20;
  localparam B_DATA_0  = 12'h28;
  localparam B_DATA_1  = 12'h2c;
  localparam CFG_CTRL  = 12'h40;
  localparam CFG_WDATA = 12'h44;

  // Test parameters
  localparam NUM_ELEMENTS = 1024;
  localparam XFER_BYTES = NUM_ELEMENTS * 2;
  localparam NUM_BEATS = NUM_ELEMENTS / 32;  // 32 beats
  localparam INPUT_BASE_ADDR  = 64'h0000_0000;
  localparam OUTPUT_BASE_ADDR = 64'h0001_0000;

  //==========================================================================
  // Config data
  //==========================================================================
  reg [15:0] config_rom [0:63];
  reg [15:0] func_lut [0:255];

  //==========================================================================
  // Clock and Reset
  //==========================================================================
  reg ap_clk = 0;
  reg aresetn = 0;
  always #(CLK_PERIOD/2) ap_clk = ~ap_clk;

  //==========================================================================
  // AXI-Lite control signals
  //==========================================================================
  wire [CTRL_ADDR_WIDTH-1:0]   ctrl_awaddr;
  wire                         ctrl_awvalid;
  wire                         ctrl_awready;
  wire [CTRL_DATA_WIDTH-1:0]   ctrl_wdata;
  wire [3:0]                   ctrl_wstrb;
  wire                         ctrl_wvalid;
  wire                         ctrl_wready;
  wire [1:0]                   ctrl_bresp;
  wire                         ctrl_bvalid;
  wire                         ctrl_bready;
  wire [CTRL_ADDR_WIDTH-1:0]   ctrl_araddr;
  wire                         ctrl_arvalid;
  wire                         ctrl_arready;
  wire [CTRL_DATA_WIDTH-1:0]   ctrl_rdata;
  wire [1:0]                   ctrl_rresp;
  wire                         ctrl_rvalid;
  wire                         ctrl_rready;

  //==========================================================================
  // AXI4 m00 signals (read port)
  //==========================================================================
  wire                         m00_awvalid, m00_awready;
  wire [ADDR_WIDTH-1:0]        m00_awaddr;
  wire [7:0]                   m00_awlen;
  wire [2:0]                   m00_awsize;
  wire [1:0]                   m00_awburst;
  wire                         m00_wvalid, m00_wready;
  wire [DATA_WIDTH-1:0]        m00_wdata;
  wire [DATA_WIDTH/8-1:0]      m00_wstrb;
  wire                         m00_wlast;
  wire [1:0]                   m00_bresp;
  wire                         m00_bvalid, m00_bready;
  wire                         m00_arvalid, m00_arready;
  wire [ADDR_WIDTH-1:0]        m00_araddr;
  wire [7:0]                   m00_arlen;
  wire [2:0]                   m00_arsize;
  wire [1:0]                   m00_arburst;
  wire                         m00_rvalid, m00_rready;
  wire [DATA_WIDTH-1:0]        m00_rdata;
  wire [1:0]                   m00_rresp;
  wire                         m00_rlast;

  //==========================================================================
  // AXI4 m01 signals (write port)
  //==========================================================================
  wire                         m01_awvalid, m01_awready;
  wire [ADDR_WIDTH-1:0]        m01_awaddr;
  wire [7:0]                   m01_awlen;
  wire [2:0]                   m01_awsize;
  wire [1:0]                   m01_awburst;
  wire                         m01_wvalid, m01_wready;
  wire [DATA_WIDTH-1:0]        m01_wdata;
  wire [DATA_WIDTH/8-1:0]      m01_wstrb;
  wire                         m01_wlast;
  wire [1:0]                   m01_bresp;
  wire                         m01_bvalid, m01_bready;
  wire                         m01_arvalid, m01_arready;
  wire [ADDR_WIDTH-1:0]        m01_araddr;
  wire [7:0]                   m01_arlen;
  wire [2:0]                   m01_arsize;
  wire [1:0]                   m01_arburst;
  wire                         m01_rvalid, m01_rready;
  wire [DATA_WIDTH-1:0]        m01_rdata;
  wire [1:0]                   m01_rresp;
  wire                         m01_rlast;

  wire interrupt;

  //==========================================================================
  // DUT
  //==========================================================================
  EDA #(
    .C_S_AXI_CONTROL_ADDR_WIDTH ( CTRL_ADDR_WIDTH ),
    .C_S_AXI_CONTROL_DATA_WIDTH ( CTRL_DATA_WIDTH ),
    .C_M00_AXI_ADDR_WIDTH       ( ADDR_WIDTH      ),
    .C_M00_AXI_DATA_WIDTH       ( DATA_WIDTH      ),
    .C_M01_AXI_ADDR_WIDTH       ( ADDR_WIDTH      ),
    .C_M01_AXI_DATA_WIDTH       ( DATA_WIDTH      )
  ) dut (
    .ap_clk                  ( ap_clk       ),
    .ap_rst_n                ( aresetn      ),
    // m00_axi
    .m00_axi_awvalid         ( m00_awvalid  ),
    .m00_axi_awready         ( m00_awready  ),
    .m00_axi_awaddr          ( m00_awaddr   ),
    .m00_axi_awlen           ( m00_awlen    ),
    .m00_axi_wvalid          ( m00_wvalid   ),
    .m00_axi_wready          ( m00_wready   ),
    .m00_axi_wdata           ( m00_wdata    ),
    .m00_axi_wstrb           ( m00_wstrb    ),
    .m00_axi_wlast           ( m00_wlast    ),
    .m00_axi_bvalid          ( m00_bvalid   ),
    .m00_axi_bready          ( m00_bready   ),
    .m00_axi_arvalid         ( m00_arvalid  ),
    .m00_axi_arready         ( m00_arready  ),
    .m00_axi_araddr          ( m00_araddr   ),
    .m00_axi_arlen           ( m00_arlen    ),
    .m00_axi_rvalid          ( m00_rvalid   ),
    .m00_axi_rready          ( m00_rready   ),
    .m00_axi_rdata           ( m00_rdata    ),
    .m00_axi_rlast           ( m00_rlast    ),
    // m01_axi
    .m01_axi_awvalid         ( m01_awvalid  ),
    .m01_axi_awready         ( m01_awready  ),
    .m01_axi_awaddr          ( m01_awaddr   ),
    .m01_axi_awlen           ( m01_awlen    ),
    .m01_axi_wvalid          ( m01_wvalid   ),
    .m01_axi_wready          ( m01_wready   ),
    .m01_axi_wdata           ( m01_wdata    ),
    .m01_axi_wstrb           ( m01_wstrb    ),
    .m01_axi_wlast           ( m01_wlast    ),
    .m01_axi_bvalid          ( m01_bvalid   ),
    .m01_axi_bready          ( m01_bready   ),
    .m01_axi_arvalid         ( m01_arvalid  ),
    .m01_axi_arready         ( m01_arready  ),
    .m01_axi_araddr          ( m01_araddr   ),
    .m01_axi_arlen           ( m01_arlen    ),
    .m01_axi_rvalid          ( m01_rvalid   ),
    .m01_axi_rready          ( m01_rready   ),
    .m01_axi_rdata           ( m01_rdata    ),
    .m01_axi_rlast           ( m01_rlast    ),
    // s_axi_control
    .s_axi_control_awvalid   ( ctrl_awvalid ),
    .s_axi_control_awready   ( ctrl_awready ),
    .s_axi_control_awaddr    ( ctrl_awaddr  ),
    .s_axi_control_wvalid    ( ctrl_wvalid  ),
    .s_axi_control_wready    ( ctrl_wready  ),
    .s_axi_control_wdata     ( ctrl_wdata   ),
    .s_axi_control_wstrb     ( ctrl_wstrb   ),
    .s_axi_control_arvalid   ( ctrl_arvalid ),
    .s_axi_control_arready   ( ctrl_arready ),
    .s_axi_control_araddr    ( ctrl_araddr  ),
    .s_axi_control_rvalid    ( ctrl_rvalid  ),
    .s_axi_control_rready    ( ctrl_rready  ),
    .s_axi_control_rdata     ( ctrl_rdata   ),
    .s_axi_control_rresp     ( ctrl_rresp   ),
    .s_axi_control_bvalid    ( ctrl_bvalid  ),
    .s_axi_control_bready    ( ctrl_bready  ),
    .s_axi_control_bresp     ( ctrl_bresp   ),
    .interrupt               ( interrupt    )
  );

  //==========================================================================
  // AXI VIP: AXI-Lite Master (control port)
  //==========================================================================
  control_eda_nli_vip ctrl_vip_inst (
    .aclk          ( ap_clk       ),
    .aresetn       ( aresetn      ),
    .m_axi_awaddr  ( ctrl_awaddr  ),
    .m_axi_awvalid ( ctrl_awvalid ),
    .m_axi_awready ( ctrl_awready ),
    .m_axi_wdata   ( ctrl_wdata   ),
    .m_axi_wstrb   ( ctrl_wstrb   ),
    .m_axi_wvalid  ( ctrl_wvalid  ),
    .m_axi_wready  ( ctrl_wready  ),
    .m_axi_bresp   ( ctrl_bresp   ),
    .m_axi_bvalid  ( ctrl_bvalid  ),
    .m_axi_bready  ( ctrl_bready  ),
    .m_axi_araddr  ( ctrl_araddr  ),
    .m_axi_arvalid ( ctrl_arvalid ),
    .m_axi_arready ( ctrl_arready ),
    .m_axi_rdata   ( ctrl_rdata   ),
    .m_axi_rresp   ( ctrl_rresp   ),
    .m_axi_rvalid  ( ctrl_rvalid  ),
    .m_axi_rready  ( ctrl_rready  )
  );

  //==========================================================================
  // AXI VIP: AXI4 Slave (m00_axi - read input data)
  //==========================================================================
  slv_m00_axi_vip m00_vip_inst (
    .aclk          ( ap_clk      ),
    .aresetn       ( aresetn     ),
    .s_axi_awaddr  ( m00_awaddr  ),
    .s_axi_awlen   ( m00_awlen   ),
    // awsize not present when SUPPORTS_NARROW=0
    .s_axi_awburst ( 2'b01       ),
    .s_axi_awvalid ( m00_awvalid ),
    .s_axi_awready ( m00_awready ),
    .s_axi_wdata   ( m00_wdata   ),
    .s_axi_wstrb   ( m00_wstrb   ),
    .s_axi_wlast   ( m00_wlast   ),
    .s_axi_wvalid  ( m00_wvalid  ),
    .s_axi_wready  ( m00_wready  ),
    .s_axi_bresp   ( m00_bresp   ),
    .s_axi_bvalid  ( m00_bvalid  ),
    .s_axi_bready  ( m00_bready  ),
    .s_axi_araddr  ( m00_araddr  ),
    .s_axi_arlen   ( m00_arlen   ),
    // arsize not present when SUPPORTS_NARROW=0
    .s_axi_arburst ( 2'b01       ),
    .s_axi_arvalid ( m00_arvalid ),
    .s_axi_arready ( m00_arready ),
    .s_axi_rdata   ( m00_rdata   ),
    .s_axi_rresp   ( m00_rresp   ),
    .s_axi_rlast   ( m00_rlast   ),
    .s_axi_rvalid  ( m00_rvalid  ),
    .s_axi_rready  ( m00_rready  )
  );

  //==========================================================================
  // AXI VIP: AXI4 Slave (m01_axi - write output data)
  //==========================================================================
  slv_m01_axi_vip m01_vip_inst (
    .aclk          ( ap_clk      ),
    .aresetn       ( aresetn     ),
    .s_axi_awaddr  ( m01_awaddr  ),
    .s_axi_awlen   ( m01_awlen   ),
    // awsize not present when SUPPORTS_NARROW=0
    .s_axi_awburst ( 2'b01       ),
    .s_axi_awvalid ( m01_awvalid ),
    .s_axi_awready ( m01_awready ),
    .s_axi_wdata   ( m01_wdata   ),
    .s_axi_wstrb   ( m01_wstrb   ),
    .s_axi_wlast   ( m01_wlast   ),
    .s_axi_wvalid  ( m01_wvalid  ),
    .s_axi_wready  ( m01_wready  ),
    .s_axi_bresp   ( m01_bresp   ),
    .s_axi_bvalid  ( m01_bvalid  ),
    .s_axi_bready  ( m01_bready  ),
    .s_axi_araddr  ( m01_araddr  ),
    .s_axi_arlen   ( m01_arlen   ),
    // arsize not present when SUPPORTS_NARROW=0
    .s_axi_arburst ( 2'b01       ),
    .s_axi_arvalid ( m01_arvalid ),
    .s_axi_arready ( m01_arready ),
    .s_axi_rdata   ( m01_rdata   ),
    .s_axi_rresp   ( m01_rresp   ),
    .s_axi_rlast   ( m01_rlast   ),
    .s_axi_rvalid  ( m01_rvalid  ),
    .s_axi_rready  ( m01_rready  )
  );

  //==========================================================================
  // VIP Agent Handles
  //==========================================================================
  control_eda_nli_vip_mst_t  ctrl_agent;
  slv_m00_axi_vip_slv_mem_t  m00_mem_agent;
  slv_m01_axi_vip_slv_mem_t  m01_mem_agent;

  //==========================================================================
  // AXI-Lite helper tasks using VIP API
  //==========================================================================
  task automatic vip_write(input bit [CTRL_ADDR_WIDTH-1:0] addr, input bit [31:0] data);
    axi_transaction wr_trans;
    wr_trans = ctrl_agent.wr_driver.create_transaction("wr_txn");
    wr_trans.set_write_cmd(addr, XIL_AXI_SIZE_4BYTE, XIL_AXI_BURST_TYPE_FIXED, 0, 0);
    wr_trans.set_data_beat(0, data);
    wr_trans.set_driver_return_item_policy(XIL_AXI_PAYLOAD_RETURN);
    ctrl_agent.wr_driver.send(wr_trans);
    ctrl_agent.wr_driver.wait_rsp(wr_trans);
  endtask

  task automatic vip_read(input bit [CTRL_ADDR_WIDTH-1:0] addr, output bit [31:0] data);
    axi_transaction rd_trans;
    rd_trans = ctrl_agent.rd_driver.create_transaction("rd_txn");
    rd_trans.set_read_cmd(addr, XIL_AXI_SIZE_4BYTE, XIL_AXI_BURST_TYPE_FIXED, 0, 0);
    rd_trans.set_driver_return_item_policy(XIL_AXI_PAYLOAD_RETURN);
    ctrl_agent.rd_driver.send(rd_trans);
    ctrl_agent.rd_driver.wait_rsp(rd_trans);
    data = rd_trans.get_data_beat(0);
  endtask

  //==========================================================================
  // FP16 helper
  //==========================================================================
  function automatic real fp16_to_real(input [15:0] h);
    reg [4:0] exp;
    reg [9:0] mant;
    real result;
    begin
      exp  = h[14:10];
      mant = h[9:0];
      if (exp == 0 && mant == 0)
        result = 0.0;
      else if (exp == 0)
        result = (2.0**(-14)) * (real'(mant) / 1024.0);
      else if (exp == 31)
        result = 99999.0;
      else
        result = (2.0**(real'(exp) - 15.0)) * (1.0 + real'(mant) / 1024.0);
      if (h[15]) result = -result;
      fp16_to_real = result;
    end
  endfunction

  function automatic [15:0] real_to_fp16(input real val);
    reg sign;
    integer exp_int, mant_int;
    real abs_val, mant_real;
    reg [4:0] biased_exp;
    reg [9:0] mant_bits;
    begin
      sign = (val < 0.0);
      abs_val = (val < 0.0) ? -val : val;
      if (abs_val == 0.0) begin
        real_to_fp16 = {sign, 15'b0};
      end else if (abs_val >= 65504.0) begin
        real_to_fp16 = {sign, 5'b11111, 10'b0};
      end else begin
        exp_int = 0;
        mant_real = abs_val;
        while (mant_real >= 2.0) begin mant_real = mant_real / 2.0; exp_int = exp_int + 1; end
        while (mant_real < 1.0 && exp_int > -14) begin mant_real = mant_real * 2.0; exp_int = exp_int - 1; end
        biased_exp = exp_int[4:0] + 5'd15;
        mant_int = $rtoi((mant_real - 1.0) * 1024.0);
        mant_bits = mant_int[9:0];
        real_to_fp16 = {sign, biased_exp, mant_bits};
      end
    end
  endfunction

  //==========================================================================
  // Main Test
  //==========================================================================
  integer i, j, errors;
  bit [31:0] read_data;
  reg [15:0] test_input [0:NUM_ELEMENTS-1];
  real x_val, y_hw, y_ref, abs_err, max_err;
  reg [DATA_WIDTH-1:0] beat_data;
  bit [7:0] mem_byte;

  initial begin
    $display("=== EDA-NLI Kernel AXI VIP Testbench ===");

    // Load config
    $readmemh("config_rom.mem", config_rom);
    $readmemh("func_lut.mem", func_lut);

    // Verify config loaded
    $display("  config_rom[0]=0x%04h (expect 0x00B1), func_lut[176]=0x%04h (expect 0x3800)",
             config_rom[0], func_lut[176]);
    if (config_rom[0] !== 16'h00B1)
      $display("  WARNING: $readmemh failed! config_rom[0]=%h", config_rom[0]);
    // Also verify after AXI-Lite load


    // Generate test input
    for (i = 0; i < NUM_ELEMENTS; i++) begin
      x_val = -8.0 + 16.0 * real'(i) / real'(NUM_ELEMENTS - 1);
      test_input[i] = real_to_fp16(x_val);
    end

    // Reset - hold low for 20 cycles, then release through FF chain (10-cycle delay)
    aresetn = 0;
    @(posedge ap_clk);
    $display("  [RST] aresetn=0: wstate=%0d (expect 3=WRRESET)", dut.inst_control_s_axi.wstate);
    $display("  [RST] aresetn=0: ARESET=%0b, rst_n_sync=%0b", dut.inst_control_s_axi.ARESET, dut.rst_n_sync);
    repeat(19) @(posedge ap_clk);
    $display("  [RST] after 20 cyc: wstate=%0d, rstate=%0d", dut.inst_control_s_axi.wstate, dut.inst_control_s_axi.rstate);

    aresetn = 1;
    repeat(5) @(posedge ap_clk);
    $display("  [RST] aresetn=1 +5cyc: rst_n_sync=%0b, ARESET=%0b, wstate=%0d", dut.rst_n_sync, dut.inst_control_s_axi.ARESET, dut.inst_control_s_axi.wstate);
    repeat(10) @(posedge ap_clk);
    $display("  [RST] aresetn=1 +15cyc: rst_n_sync=%0b, ARESET=%0b, wstate=%0d", dut.rst_n_sync, dut.inst_control_s_axi.ARESET, dut.inst_control_s_axi.wstate);
    repeat(5) @(posedge ap_clk);
    $display("  [RST] aresetn=1 +20cyc: rst_n_sync=%0b, ARESET=%0b, wstate=%0d (expect 0=WRIDLE)", dut.rst_n_sync, dut.inst_control_s_axi.ARESET, dut.inst_control_s_axi.wstate);

    // Create VIP agents
    ctrl_agent = new("ctrl_agent", ctrl_vip_inst.inst.IF);
    ctrl_agent.start_master();

    m00_mem_agent = new("m00_mem_agent", m00_vip_inst.inst.IF);
    m00_vip_inst.inst.IF.set_enable_xchecks_to_warn();
    m00_mem_agent.start_slave();
    // Set auto-response mode (memory model)
    m00_mem_agent.mem_model.set_memory_fill_policy(XIL_AXI_MEMORY_FILL_FIXED);
    m00_mem_agent.mem_model.set_default_memory_value(8'h00);

    m01_mem_agent = new("m01_mem_agent", m01_vip_inst.inst.IF);
    m01_vip_inst.inst.IF.set_enable_xchecks_to_warn();
    m01_mem_agent.start_slave();
    m01_mem_agent.mem_model.set_memory_fill_policy(XIL_AXI_MEMORY_FILL_FIXED);
    m01_mem_agent.mem_model.set_default_memory_value(8'h00);

    $display("VIP agents started.");

    // Write input data to m00 VIP memory (512-bit aligned words)
    begin
      bit [DATA_WIDTH-1:0] wr_word;
      bit [DATA_WIDTH/8-1:0] wr_strb;
      wr_strb = {(DATA_WIDTH/8){1'b1}};
      for (i = 0; i < NUM_BEATS; i++) begin
        wr_word = {DATA_WIDTH{1'b0}};
        for (j = 0; j < 32; j++) begin
          if (i*32 + j < NUM_ELEMENTS)
            wr_word[j*16 +: 16] = test_input[i*32 + j];
        end
        m00_mem_agent.mem_model.backdoor_memory_write(INPUT_BASE_ADDR + i*64, wr_word, wr_strb);
      end
    end
    $display("Input data loaded to VIP memory (%0d elements).", NUM_ELEMENTS);

    //====================================================================
    // Phase 1: Load NLI Configuration
    //====================================================================
    $display("\n--- Phase 1: Loading NLI config ---");
    // Test one write first
    $display("  Testing single AXI-Lite write...");
    vip_write(CFG_WDATA, 32'h00B1);
    $display("  First write OK at t=%0t", $time);

    for (i = 0; i < 64; i++) begin
      vip_write(CFG_WDATA, {16'b0, config_rom[i]});
      vip_write(CFG_CTRL, (i << 1) | 0);
    end
    $display("  config_rom loaded (64 entries) at t=%0t", $time);

    for (i = 0; i < 256; i++) begin
      vip_write(CFG_WDATA, {16'b0, func_lut[i]});
      vip_write(CFG_CTRL, (i << 1) | 1);
    end
    $display("  func_lut loaded (256 entries) at t=%0t", $time);
    // Verify NLI engine internal state after config
    $display("  NLI config_rom[0]=0x%04h (expect 0x00B1)", dut.EDA_NLI_inst.u_compute.u_nli_engine.config_rom[0]);
    $display("  NLI func_lut[176]=0x%04h (expect 0x3800)", dut.EDA_NLI_inst.u_compute.u_nli_engine.func_lut[176]);

    //====================================================================
    // Phase 2: Configure and Start
    //====================================================================
    $display("\n--- Phase 2: Starting kernel ---");
    vip_write(SCALAR00, XFER_BYTES);
    vip_write(A_DATA_0, INPUT_BASE_ADDR[31:0]);
    vip_write(A_DATA_1, INPUT_BASE_ADDR[63:32]);
    vip_write(B_DATA_0, OUTPUT_BASE_ADDR[31:0]);
    vip_write(B_DATA_1, OUTPUT_BASE_ADDR[63:32]);
    // Check control_s_axi state before start
    $display("  [CTRL] Before start: int_ap_start=%b user_start=%b user_done=%b user_ready=%b",
             dut.inst_control_s_axi.int_ap_start,
             dut.inst_control_s_axi.user_start,
             dut.inst_control_s_axi.user_done,
             dut.inst_control_s_axi.user_ready);

    vip_write(USER_CTRL, 32'h1);  // Start!

    repeat(3) @(posedge ap_clk);
    $display("  [CTRL] After start: int_ap_start=%b user_start=%b ap_start_pulse=%b",
             dut.inst_control_s_axi.int_ap_start,
             dut.EDA_NLI_inst.user_start,
             dut.EDA_NLI_inst.ap_start_pulse);
    $display("  [CTRL] wrapper: areset=%b ap_idle=%b",
             dut.EDA_NLI_inst.areset,
             dut.EDA_NLI_inst.ap_idle_r);
    $display("  Kernel started.");

    //====================================================================
    // Phase 3: Wait for completion
    //====================================================================
    $display("\n--- Phase 3: Waiting for completion ---");
    read_data = 0;
    while (!(read_data & 32'h2)) begin
      repeat(50) @(posedge ap_clk);
      vip_read(USER_CTRL, read_data);
    end
    $display("  Kernel done!");

    repeat(20) @(posedge ap_clk);

    //====================================================================
    // Phase 4: Read output and verify
    //====================================================================
    $display("\n--- Phase 4: Verifying results ---");
    errors = 0;
    max_err = 0.0;

    begin
      bit [DATA_WIDTH-1:0] rd_word;
      reg [15:0] out_val;

      // Read output beats from m01 VIP memory
      // Debug: try both address 0 and OUTPUT_BASE_ADDR
      $display("  DEBUG: Reading from addr 0x%h and 0x0", OUTPUT_BASE_ADDR);
      rd_word = m01_mem_agent.mem_model.backdoor_memory_read(64'h0);
      $display("  DEBUG: addr 0x0 beat = 0x%h", rd_word[63:0]);
      rd_word = m01_mem_agent.mem_model.backdoor_memory_read(OUTPUT_BASE_ADDR);
      $display("  DEBUG: addr 0x%h beat = 0x%h", OUTPUT_BASE_ADDR, rd_word[63:0]);

      for (i = 0; i < NUM_BEATS; i++) begin
        rd_word = m01_mem_agent.mem_model.backdoor_memory_read(OUTPUT_BASE_ADDR + i*64);
        for (j = 0; j < 32; j++) begin
          if (i*32 + j < NUM_ELEMENTS) begin
            out_val = rd_word[j*16 +: 16];
            x_val = fp16_to_real(test_input[i*32 + j]);
            y_hw  = fp16_to_real(out_val);
            y_ref = 1.0 / (1.0 + $exp(-x_val));
            abs_err = (y_hw > y_ref) ? (y_hw - y_ref) : (y_ref - y_hw);
            if (abs_err > max_err) max_err = abs_err;
            if (abs_err > 0.02) errors++;

            if (i*32+j < 10 || i*32+j == NUM_ELEMENTS/2 || i*32+j >= NUM_ELEMENTS-3)
              $display("  [%3d] x=%8.4f  hw=%8.6f  ref=%8.6f  err=%e",
                       i*32+j, x_val, y_hw, y_ref, abs_err);
          end
        end
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

  // Monitor m01 AXI write channel
  always @(posedge ap_clk) begin
    if (m01_awvalid & m01_awready)
      $display("  [M01-AW] t=%0t addr=0x%h len=%0d", $time, m01_awaddr, m01_awlen);
    if (m01_wvalid & m01_wready)
      $display("  [M01-W]  t=%0t data[63:0]=0x%h wstrb=0x%h last=%b", $time, m01_wdata[63:0], m01_wstrb, m01_wlast);
    if (m01_bvalid & m01_bready)
      $display("  [M01-B]  t=%0t bresp=%0d", $time, m01_bresp);
  end

  // Monitor m00 AXI read channel
  always @(posedge ap_clk) begin
    if (m00_arvalid & m00_arready)
      $display("  [M00-AR] t=%0t addr=0x%h len=%0d", $time, m00_araddr, m00_arlen);
    if (m00_rvalid & m00_rready)
      $display("  [M00-R]  t=%0t data[63:0]=0x%h last=%b", $time, m00_rdata[63:0], m00_rlast);
  end

  // Timeout
  initial begin
    #(CLK_PERIOD * 1000000);
    $display("ERROR: Simulation timeout!");
    $finish;
  end

endmodule
