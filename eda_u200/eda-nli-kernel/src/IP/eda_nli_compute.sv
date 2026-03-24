// ==============================================================
// eda_nli_compute - Compute Pipeline for EDA-NLI Kernel
//
// PISO (512b → 32×16b) → eda_nli_engine_4s → SIPO (32×16b → 512b)
// → Output FIFO (2-deep) → m_tvalid/m_tdata
//
// S_IDLE → S_COMPUTE → S_PUSH → S_IDLE
// ==============================================================
`default_nettype none
`timescale 1ns/1ps

module eda_nli_compute #(
  parameter integer DATA_WIDTH = 512
) (
  input  wire                    clk,
  input  wire                    rst_n,

  input  wire                    s_tvalid,
  output wire                    s_tready,
  input  wire [DATA_WIDTH-1:0]   s_tdata,
  input  wire                    s_tlast,

  output wire                    m_tvalid,
  input  wire                    m_tready,
  output wire [DATA_WIDTH-1:0]   m_tdata,
  output wire                    m_tlast,

  input  wire                    cfg_we,
  input  wire [0:0]              cfg_sel,
  input  wire [8:0]              cfg_addr,
  input  wire [15:0]             cfg_wdata
);

  localparam NUM_ELEMS = DATA_WIDTH / 16;
  localparam CNT_W     = $clog2(NUM_ELEMS);

  //==========================================================================
  // State Machine
  //==========================================================================
  localparam [1:0] S_IDLE    = 2'd0,
                   S_COMPUTE = 2'd1,
                   S_PUSH    = 2'd2;

  reg [1:0] state;

  // PISO
  reg [DATA_WIDTH-1:0] in_buf;
  reg                  in_last;
  reg [5:0]            in_cnt;
  reg                  in_done;

  // SIPO
  reg [DATA_WIDTH-1:0] sipo_buf;
  reg                  sipo_last;
  reg [CNT_W-1:0]      sipo_cnt;

  // NLI engine
  wire        nli_i_valid;
  wire [15:0] nli_i_data;
  wire        nli_o_valid;
  wire [15:0] nli_o_data;

  //==========================================================================
  // Output FIFO (2-deep)
  //==========================================================================
  reg [DATA_WIDTH-1:0] fifo_data [0:1];
  reg                  fifo_last [0:1];
  reg [1:0]            fifo_wr_ptr;
  reg [1:0]            fifo_rd_ptr;
  reg [1:0]            fifo_count;

  wire fifo_full  = (fifo_count == 2'd2);
  wire fifo_empty = (fifo_count == 2'd0);
  wire fifo_push  = (state == S_PUSH);
  wire fifo_pop   = m_tvalid && m_tready;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fifo_wr_ptr <= 2'd0;
    else if (fifo_push && !fifo_full) begin
      fifo_data[fifo_wr_ptr[0]] <= sipo_buf;
      fifo_last[fifo_wr_ptr[0]] <= sipo_last;
      fifo_wr_ptr <= fifo_wr_ptr + 2'd1;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fifo_rd_ptr <= 2'd0;
    else if (fifo_pop)
      fifo_rd_ptr <= fifo_rd_ptr + 2'd1;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fifo_count <= 2'd0;
    else begin
      case ({fifo_push && !fifo_full, fifo_pop})
        2'b10:   fifo_count <= fifo_count + 2'd1;
        2'b01:   fifo_count <= fifo_count - 2'd1;
        default: fifo_count <= fifo_count;
      endcase
    end
  end

  assign m_tvalid = !fifo_empty;
  assign m_tdata  = fifo_data[fifo_rd_ptr[0]];
  assign m_tlast  = fifo_last[fifo_rd_ptr[0]];

  //==========================================================================
  // Handshake
  //==========================================================================
  assign s_tready   = (state == S_IDLE) && !fifo_full;
  assign nli_i_valid = (state == S_COMPUTE) && !in_done;
  assign nli_i_data  = in_buf[in_cnt[CNT_W-1:0] * 16 +: 16];

  //==========================================================================
  // State machine
  //==========================================================================
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      in_cnt    <= 6'd0;
      in_done   <= 1'b0;
      sipo_cnt  <= {CNT_W{1'b0}};
      in_buf    <= {DATA_WIDTH{1'b0}};
      sipo_buf  <= {DATA_WIDTH{1'b0}};
      in_last   <= 1'b0;
      sipo_last <= 1'b0;
    end else begin
      case (state)

        S_IDLE: begin
          if (s_tvalid && !fifo_full) begin
            in_buf   <= s_tdata;
            in_last  <= s_tlast;
            in_cnt   <= 6'd0;
            in_done  <= 1'b0;
            sipo_cnt <= {CNT_W{1'b0}};
            state    <= S_COMPUTE;
          end
        end

        S_COMPUTE: begin
          // Input: feed one FP16 per cycle
          if (!in_done) begin
            if (in_cnt == 6'd31)
              in_done <= 1'b1;
            else
              in_cnt <= in_cnt + 6'd1;
          end

          // Output: collect NLI results
          if (nli_o_valid) begin
            sipo_buf[sipo_cnt * 16 +: 16] <= nli_o_data;
            if (sipo_cnt == 5'd31) begin
              sipo_last <= in_last;
              state     <= S_PUSH;  // go to push state (1 cycle delay for non-blocking)
            end else begin
              sipo_cnt <= sipo_cnt + 1;
            end
          end
        end

        S_PUSH: begin
          // sipo_buf now has all 32 elements (non-blocking from S_COMPUTE settled)
          // fifo_push wire is high in this state → FIFO captures sipo_buf
          state <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  //==========================================================================
  // NLI Engine
  //==========================================================================
  eda_nli_engine_4s #(
    .T_BITS    ( 10  ),
    .LUT_DEPTH ( 256 )
  ) u_nli_engine (
    .clk       ( clk          ),
    .rst_n     ( rst_n        ),
    .i_valid   ( nli_i_valid  ),
    .i_data    ( nli_i_data   ),
    .o_valid   ( nli_o_valid  ),
    .o_data    ( nli_o_data   ),
    .cfg_we    ( cfg_we       ),
    .cfg_sel   ( cfg_sel      ),
    .cfg_addr  ( cfg_addr     ),
    .cfg_wdata ( cfg_wdata    )
  );

endmodule
`default_nettype wire
