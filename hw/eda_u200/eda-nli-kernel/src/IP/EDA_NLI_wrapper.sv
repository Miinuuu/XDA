// ==============================================================
// EDA_NLI_wrapper - Replaces Top_wrapper (EDA.sv)
//
// Architecture:
//   m00_axi (read only)  → EDA_axi_read_master → AXI-Stream
//   AXI-Stream → eda_nli_compute (PISO→NLI engine→SIPO) → AXI-Stream
//   AXI-Stream → axi_write_master → m01_axi (write only)
// ==============================================================
`default_nettype none

module EDA_NLI_wrapper #(
  parameter integer C_M00_AXI_ADDR_WIDTH = 64 ,
  parameter integer C_M00_AXI_DATA_WIDTH = 512,
  parameter integer C_M01_AXI_ADDR_WIDTH = 64 ,
  parameter integer C_M01_AXI_DATA_WIDTH = 512
)
(
  // System Signals
  input  wire                              ap_clk         ,
  input  wire                              ap_rst_n       ,
  // AXI4 master interface m00_axi (read input from DDR)
  output wire                              m00_axi_awvalid,
  input  wire                              m00_axi_awready,
  output wire [C_M00_AXI_ADDR_WIDTH-1:0]   m00_axi_awaddr ,
  output wire [8-1:0]                      m00_axi_awlen  ,
  output wire                              m00_axi_wvalid ,
  input  wire                              m00_axi_wready ,
  output wire [C_M00_AXI_DATA_WIDTH-1:0]   m00_axi_wdata  ,
  output wire [C_M00_AXI_DATA_WIDTH/8-1:0] m00_axi_wstrb  ,
  output wire                              m00_axi_wlast  ,
  input  wire                              m00_axi_bvalid ,
  output wire                              m00_axi_bready ,
  output wire                              m00_axi_arvalid,
  input  wire                              m00_axi_arready,
  output wire [C_M00_AXI_ADDR_WIDTH-1:0]   m00_axi_araddr ,
  output wire [8-1:0]                      m00_axi_arlen  ,
  input  wire                              m00_axi_rvalid ,
  output wire                              m00_axi_rready ,
  input  wire [C_M00_AXI_DATA_WIDTH-1:0]   m00_axi_rdata  ,
  input  wire                              m00_axi_rlast  ,
  // AXI4 master interface m01_axi (write output to DDR)
  output wire                              m01_axi_awvalid,
  input  wire                              m01_axi_awready,
  output wire [C_M01_AXI_ADDR_WIDTH-1:0]   m01_axi_awaddr ,
  output wire [8-1:0]                      m01_axi_awlen  ,
  output wire                              m01_axi_wvalid ,
  input  wire                              m01_axi_wready ,
  output wire [C_M01_AXI_DATA_WIDTH-1:0]   m01_axi_wdata  ,
  output wire [C_M01_AXI_DATA_WIDTH/8-1:0] m01_axi_wstrb  ,
  output wire                              m01_axi_wlast  ,
  input  wire                              m01_axi_bvalid ,
  output wire                              m01_axi_bready ,
  output wire                              m01_axi_arvalid,
  input  wire                              m01_axi_arready,
  output wire [C_M01_AXI_ADDR_WIDTH-1:0]   m01_axi_araddr ,
  output wire [8-1:0]                      m01_axi_arlen  ,
  input  wire                              m01_axi_rvalid ,
  output wire                              m01_axi_rready ,
  input  wire [C_M01_AXI_DATA_WIDTH-1:0]   m01_axi_rdata  ,
  input  wire                              m01_axi_rlast  ,
  // Control Signals
  input  wire                              user_start     ,
  output wire                              user_idle      ,
  output wire                              user_done      ,
  output wire                              user_ready     ,
  input  wire [32-1:0]                     scalar00       ,
  input  wire [64-1:0]                     A              ,
  input  wire [64-1:0]                     B              ,
  // NLI configuration
  input  wire                              cfg_we         ,
  input  wire [0:0]                        cfg_sel        ,
  input  wire [8:0]                        cfg_addr       ,
  input  wire [15:0]                       cfg_wdata
);

timeunit 1ps;
timeprecision 1ps;

///////////////////////////////////////////////////////////////////////////////
// Local Parameters
///////////////////////////////////////////////////////////////////////////////
localparam integer  LP_DEFAULT_LENGTH_IN_BYTES = 16384;
localparam integer  LP_DW_BYTES             = C_M00_AXI_DATA_WIDTH/8;
localparam integer  LP_AXI_BURST_LEN        = 4096/LP_DW_BYTES < 256 ? 4096/LP_DW_BYTES : 256;
localparam integer  LP_LOG_BURST_LEN        = $clog2(LP_AXI_BURST_LEN);
localparam integer  LP_BRAM_DEPTH           = 512;
localparam integer  LP_RD_MAX_OUTSTANDING   = LP_BRAM_DEPTH / LP_AXI_BURST_LEN;
localparam integer  LP_WR_MAX_OUTSTANDING   = 32;

///////////////////////////////////////////////////////////////////////////////
// Wires and Variables
///////////////////////////////////////////////////////////////////////////////
(* KEEP = "yes" *)
logic                                areset                         = 1'b0;
logic                                ap_start_r                     = 1'b0;
logic                                ap_idle_r                      = 1'b1;
logic                                ap_start_pulse                ;
wire [32-1:0]                        ctrl_xfer_size_in_bytes;

// Read master → AXI-Stream
logic                          rd_tvalid;
logic                          rd_tready;
logic                          rd_tlast;
logic [C_M00_AXI_DATA_WIDTH-1:0] rd_tdata;

// Compute → write master AXI-Stream
logic                          wr_tvalid;
logic                          wr_tready;
logic                          wr_tlast;
logic [C_M01_AXI_DATA_WIDTH-1:0] wr_tdata;

// Done signals
logic                          read_done;
logic                          write_done;

///////////////////////////////////////////////////////////////////////////////
// Begin RTL
///////////////////////////////////////////////////////////////////////////////

// Register and invert reset signal
always @(posedge ap_clk) begin
  areset <= ~ap_rst_n;
end

// Create pulse when user_start transitions to 1
always @(posedge ap_clk) begin
  ap_start_r <= user_start;
end

assign ap_start_pulse = user_start & ~ap_start_r;

// Transfer size: use scalar00 directly (combinational so masters see correct value at start)
assign ctrl_xfer_size_in_bytes = (scalar00 == 32'd0) ? LP_DEFAULT_LENGTH_IN_BYTES : scalar00;

// Idle logic
always @(posedge ap_clk) begin
  if (areset) begin
    ap_idle_r <= 1'b1;
  end
  else begin
    ap_idle_r <= write_done ? 1'b1 :
      ap_start_pulse ? 1'b0 : ap_idle_r;
  end
end

assign user_idle  = ap_idle_r;
assign user_done  = write_done;
assign user_ready = write_done;

// synthesis translate_off
always @(posedge ap_clk) begin
  if (ap_start_pulse) $display("  [WRAPPER] t=%0t ap_start_pulse! xfer_size=%0d A=0x%h B=0x%h", $time, ctrl_xfer_size_in_bytes, A, B);
  if (read_done) $display("  [WRAPPER] t=%0t read_done", $time);
  if (write_done) $display("  [WRAPPER] t=%0t write_done", $time);
  if (rd_tvalid & rd_tready) $display("  [WRAPPER] t=%0t rd_tdata accepted", $time);
  if (wr_tvalid & wr_tready) $display("  [WRAPPER] t=%0t wr_tdata accepted", $time);
end
// synthesis translate_on

///////////////////////////////////////////////////////////////////////////////
// AXI Read Master (m00_axi) - reads input FP16 array from DDR
///////////////////////////////////////////////////////////////////////////////
EDA_axi_read_master #(
  .C_M_AXI_ADDR_WIDTH  ( C_M00_AXI_ADDR_WIDTH    ),
  .C_M_AXI_DATA_WIDTH  ( C_M00_AXI_DATA_WIDTH    ),
  .C_XFER_SIZE_WIDTH   ( 32                       ),
  .C_MAX_OUTSTANDING   ( LP_RD_MAX_OUTSTANDING    ),
  .C_INCLUDE_DATA_FIFO ( 1                        )
)
u_read_master (
  .aclk                    ( ap_clk                    ),
  .areset                  ( areset                    ),
  .ctrl_start              ( ap_start_pulse            ),
  .ctrl_done               ( read_done                 ),
  .ctrl_addr_offset        ( A                         ),
  .ctrl_xfer_size_in_bytes ( ctrl_xfer_size_in_bytes   ),
  .m_axi_arvalid           ( m00_axi_arvalid           ),
  .m_axi_arready           ( m00_axi_arready           ),
  .m_axi_araddr            ( m00_axi_araddr            ),
  .m_axi_arlen             ( m00_axi_arlen             ),
  .m_axi_rvalid            ( m00_axi_rvalid            ),
  .m_axi_rready            ( m00_axi_rready            ),
  .m_axi_rdata             ( m00_axi_rdata             ),
  .m_axi_rlast             ( m00_axi_rlast             ),
  .m_axis_aclk             ( ap_clk                    ),
  .m_axis_areset           ( areset                    ),
  .m_axis_tvalid           ( rd_tvalid                 ),
  .m_axis_tready           ( rd_tready                 ),
  .m_axis_tlast            ( rd_tlast                  ),
  .m_axis_tdata            ( rd_tdata                  )
);

// Tie off m00_axi write channels (read-only port)
assign m00_axi_awvalid = 1'b0;
assign m00_axi_awaddr  = {C_M00_AXI_ADDR_WIDTH{1'b0}};
assign m00_axi_awlen   = 8'b0;
assign m00_axi_wvalid  = 1'b0;
assign m00_axi_wdata   = {C_M00_AXI_DATA_WIDTH{1'b0}};
assign m00_axi_wstrb   = {(C_M00_AXI_DATA_WIDTH/8){1'b0}};
assign m00_axi_wlast   = 1'b0;
assign m00_axi_bready  = 1'b0;

///////////////////////////////////////////////////////////////////////////////
// EDA-NLI Compute Pipeline (PISO → NLI Engine → SIPO)
///////////////////////////////////////////////////////////////////////////////
eda_nli_compute #(
  .DATA_WIDTH ( C_M00_AXI_DATA_WIDTH )
)
u_compute (
  .clk       ( ap_clk     ),
  .rst_n     ( ~areset     ),
  // Input AXI-Stream (from read master)
  .s_tvalid  ( rd_tvalid  ),
  .s_tready  ( rd_tready  ),
  .s_tdata   ( rd_tdata   ),
  .s_tlast   ( rd_tlast   ),
  // Output AXI-Stream (to write master)
  .m_tvalid  ( wr_tvalid  ),
  .m_tready  ( wr_tready  ),
  .m_tdata   ( wr_tdata   ),
  .m_tlast   ( wr_tlast   ),
  // NLI configuration
  .cfg_we    ( cfg_we     ),
  .cfg_sel   ( cfg_sel    ),
  .cfg_addr  ( cfg_addr   ),
  .cfg_wdata ( cfg_wdata  )
);

///////////////////////////////////////////////////////////////////////////////
// AXI Write Master (m01_axi) - writes output FP16 array to DDR
///////////////////////////////////////////////////////////////////////////////
EDA_axi_write_master #(
  .C_M_AXI_ADDR_WIDTH  ( C_M01_AXI_ADDR_WIDTH    ),
  .C_M_AXI_DATA_WIDTH  ( C_M01_AXI_DATA_WIDTH    ),
  .C_XFER_SIZE_WIDTH   ( 32                       ),
  .C_MAX_OUTSTANDING   ( LP_WR_MAX_OUTSTANDING    ),
  .C_INCLUDE_DATA_FIFO ( 1                        )
)
u_write_master (
  .aclk                    ( ap_clk                    ),
  .areset                  ( areset                    ),
  .ctrl_start              ( ap_start_pulse            ),
  .ctrl_done               ( write_done                ),
  .ctrl_addr_offset        ( B                         ),
  .ctrl_xfer_size_in_bytes ( ctrl_xfer_size_in_bytes   ),
  .m_axi_awvalid           ( m01_axi_awvalid           ),
  .m_axi_awready           ( m01_axi_awready           ),
  .m_axi_awaddr            ( m01_axi_awaddr            ),
  .m_axi_awlen             ( m01_axi_awlen             ),
  .m_axi_wvalid            ( m01_axi_wvalid            ),
  .m_axi_wready            ( m01_axi_wready            ),
  .m_axi_wdata             ( m01_axi_wdata             ),
  .m_axi_wstrb             ( m01_axi_wstrb             ),
  .m_axi_wlast             ( m01_axi_wlast             ),
  .m_axi_bvalid            ( m01_axi_bvalid            ),
  .m_axi_bready            ( m01_axi_bready            ),
  .s_axis_aclk             ( ap_clk                    ),
  .s_axis_areset           ( areset                    ),
  .s_axis_tvalid           ( wr_tvalid                 ),
  .s_axis_tready           ( wr_tready                 ),
  .s_axis_tdata            ( wr_tdata                  )
);

// Tie off m01_axi read channels (write-only port)
assign m01_axi_arvalid = 1'b0;
assign m01_axi_araddr  = {C_M01_AXI_ADDR_WIDTH{1'b0}};
assign m01_axi_arlen   = 8'b0;
assign m01_axi_rready  = 1'b0;

endmodule : EDA_NLI_wrapper
`default_nettype wire
