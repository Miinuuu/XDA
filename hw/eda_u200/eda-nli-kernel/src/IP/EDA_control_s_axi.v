// ==============================================================
// AXI-Lite Control Slave - Modified for EDA-NLI kernel
// Added cfg_ctrl (0x40) and cfg_wdata (0x44) registers
// ==============================================================
`timescale 1ns/1ps
module EDA_control_s_axi
#(parameter
    C_S_AXI_ADDR_WIDTH = 7,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire                          user_start,
    input  wire                          user_done,
    input  wire                          user_ready,
    input  wire                          user_idle,
    output wire [31:0]                   scalar00,
    output wire [63:0]                   A,
    output wire [63:0]                   B,
    // NLI configuration interface
    output wire                          cfg_we,
    output wire [0:0]                    cfg_sel,
    output wire [8:0]                    cfg_addr,
    output wire [15:0]                   cfg_wdata
);
//------------------------Address Info-------------------
// 0x00 : Control signals
// 0x04 : Global Interrupt Enable Register
// 0x08 : IP Interrupt Enable Register
// 0x0c : IP Interrupt Status Register
// 0x10 : User Control signals (start/done/idle/ready)
// 0x14 : scalar00 (transfer size in bytes)
// 0x1c : A[31:0], 0x20 : A[63:32]
// 0x28 : B[31:0], 0x2c : B[63:32]
// 0x40 : cfg_ctrl - write triggers cfg_we pulse
//        bit 0    : cfg_sel (0=config_rom, 1=func_lut)
//        bit 9:1  : cfg_addr[8:0]
// 0x44 : cfg_wdata[15:0]

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL         = 7'h00,
    ADDR_GIE             = 7'h04,
    ADDR_IER             = 7'h08,
    ADDR_ISR             = 7'h0c,
    ADDR_USER_CTRL       = 7'h10,
    ADDR_SCALAR00_DATA_0 = 7'h14,
    ADDR_SCALAR00_CTRL   = 7'h18,
    ADDR_A_DATA_0        = 7'h1c,
    ADDR_A_DATA_1        = 7'h20,
    ADDR_A_CTRL          = 7'h24,
    ADDR_B_DATA_0        = 7'h28,
    ADDR_B_DATA_1        = 7'h2c,
    ADDR_B_CTRL          = 7'h30,
    ADDR_CFG_CTRL        = 7'h40,
    ADDR_CFG_WDATA       = 7'h44,
    WRIDLE               = 2'd0,
    WRDATA               = 2'd1,
    WRRESP               = 2'd2,
    WRRESET              = 2'd3,
    RDIDLE               = 2'd0,
    RDDATA               = 2'd1,
    RDRESET              = 2'd2,
    ADDR_BITS            = 7;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [31:0]                   wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [31:0]                   rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle;
    reg                           int_ap_ready;
    reg                           int_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [31:0]                   int_scalar00 = 'b0;
    reg  [63:0]                   int_A = 'b0;
    reg  [63:0]                   int_B = 'b0;
    // NLI config registers
    reg  [31:0]                   int_cfg_ctrl = 'b0;
    reg  [31:0]                   int_cfg_wdata = 'b0;
    reg                           int_cfg_we = 1'b0;

//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 1'b0;
            case (raddr)
                ADDR_USER_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_SCALAR00_DATA_0: begin
                    rdata <= int_scalar00[31:0];
                end
                ADDR_A_DATA_0: begin
                    rdata <= int_A[31:0];
                end
                ADDR_A_DATA_1: begin
                    rdata <= int_A[63:32];
                end
                ADDR_B_DATA_0: begin
                    rdata <= int_B[31:0];
                end
                ADDR_B_DATA_1: begin
                    rdata <= int_B[63:32];
                end
                ADDR_CFG_CTRL: begin
                    rdata <= int_cfg_ctrl;
                end
                ADDR_CFG_WDATA: begin
                    rdata <= int_cfg_wdata;
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt   = int_gie & (|int_isr);
assign user_start  = int_ap_start;
assign scalar00    = int_scalar00;
assign A           = int_A;
assign B           = int_B;

// NLI config output
assign cfg_we    = int_cfg_we;
assign cfg_sel   = int_cfg_ctrl[0];
assign cfg_addr  = int_cfg_ctrl[9:1];
assign cfg_wdata = int_cfg_wdata[15:0];

// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_USER_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (user_ready)
            int_ap_start <= int_auto_restart;
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (user_done)
            int_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_USER_CTRL)
            int_ap_done <= 1'b0;
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= user_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_ready <= user_ready;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_USER_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & user_done)
            int_isr[0] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[0] <= int_isr[0] ^ WDATA[0];
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & user_ready)
            int_isr[1] <= 1'b1;
        else if (w_hs && waddr == ADDR_ISR && WSTRB[0])
            int_isr[1] <= int_isr[1] ^ WDATA[1];
    end
end

// int_scalar00[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_scalar00[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_SCALAR00_DATA_0)
            int_scalar00[31:0] <= (WDATA[31:0] & wmask) | (int_scalar00[31:0] & ~wmask);
    end
end

// int_A[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_A[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_A_DATA_0)
            int_A[31:0] <= (WDATA[31:0] & wmask) | (int_A[31:0] & ~wmask);
    end
end

// int_A[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_A[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_A_DATA_1)
            int_A[63:32] <= (WDATA[31:0] & wmask) | (int_A[63:32] & ~wmask);
    end
end

// int_B[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_B[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_B_DATA_0)
            int_B[31:0] <= (WDATA[31:0] & wmask) | (int_B[31:0] & ~wmask);
    end
end

// int_B[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_B[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_B_DATA_1)
            int_B[63:32] <= (WDATA[31:0] & wmask) | (int_B[63:32] & ~wmask);
    end
end

// int_cfg_wdata - data register for NLI config
always @(posedge ACLK) begin
    if (ARESET)
        int_cfg_wdata <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CFG_WDATA)
            int_cfg_wdata <= (WDATA[31:0] & wmask) | (int_cfg_wdata & ~wmask);
    end
end

// int_cfg_ctrl - control register for NLI config
always @(posedge ACLK) begin
    if (ARESET)
        int_cfg_ctrl <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CFG_CTRL)
            int_cfg_ctrl <= (WDATA[31:0] & wmask) | (int_cfg_ctrl & ~wmask);
    end
end

// int_cfg_we - one-cycle pulse on write to ADDR_CFG_CTRL
always @(posedge ACLK) begin
    if (ARESET)
        int_cfg_we <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CFG_CTRL)
            int_cfg_we <= 1'b1;
        else
            int_cfg_we <= 1'b0;
    end
end

endmodule
