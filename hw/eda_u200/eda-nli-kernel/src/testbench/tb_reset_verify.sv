// ==============================================================
// tb_reset_verify - Simple reset + NLI compute verification
// No VIP, direct signal driving. Focuses on:
//   1. Reset propagation to all modules
//   2. Config loading via cfg interface
//   3. NLI engine data path (PISO→engine→SIPO)
// ==============================================================
`timescale 1ns/1ps

module tb_reset_verify;

  localparam CLK_PERIOD = 10;  // 100MHz
  localparam DATA_WIDTH = 512;
  localparam NUM_ELEMS = 32;  // per beat

  //==========================================================================
  // Clock
  //==========================================================================
  reg clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  //==========================================================================
  // DUT: eda_nli_compute (standalone, no AXI)
  //==========================================================================
  reg          rst_n = 0;
  reg          s_tvalid = 0;
  wire         s_tready;
  reg  [511:0] s_tdata = 0;
  reg          s_tlast = 0;
  wire         m_tvalid;
  reg          m_tready = 1;
  wire [511:0] m_tdata;
  wire         m_tlast;
  reg          cfg_we = 0;
  reg [0:0]    cfg_sel = 0;
  reg [8:0]    cfg_addr = 0;
  reg [15:0]   cfg_wdata = 0;

  eda_nli_compute #(.DATA_WIDTH(DATA_WIDTH)) u_compute (
    .clk(clk), .rst_n(rst_n),
    .s_tvalid(s_tvalid), .s_tready(s_tready),
    .s_tdata(s_tdata), .s_tlast(s_tlast),
    .m_tvalid(m_tvalid), .m_tready(m_tready),
    .m_tdata(m_tdata), .m_tlast(m_tlast),
    .cfg_we(cfg_we), .cfg_sel(cfg_sel),
    .cfg_addr(cfg_addr), .cfg_wdata(cfg_wdata)
  );

  //==========================================================================
  // Config data
  //==========================================================================
  reg [15:0] config_rom [0:63];
  reg [15:0] func_lut [0:255];

  //==========================================================================
  // FP16 helper
  //==========================================================================
  function automatic real fp16_to_real(input [15:0] h);
    reg [4:0] exp_v;
    reg [9:0] mant_v;
    real result;
    begin
      exp_v  = h[14:10];
      mant_v = h[9:0];
      if (exp_v == 0 && mant_v == 0) result = 0.0;
      else if (exp_v == 0) result = (2.0**(-14)) * (real'(mant_v) / 1024.0);
      else if (exp_v == 31) result = 99999.0;
      else result = (2.0**(real'(exp_v) - 15.0)) * (1.0 + real'(mant_v) / 1024.0);
      if (h[15]) result = -result;
      fp16_to_real = result;
    end
  endfunction

  function automatic [15:0] real_to_fp16(input real val);
    reg sign_v;
    integer exp_int, mant_int;
    real abs_val, mant_real;
    reg [4:0] biased_exp;
    reg [9:0] mant_bits;
    begin
      sign_v = (val < 0.0);
      abs_val = (val < 0.0) ? -val : val;
      if (abs_val == 0.0) begin
        real_to_fp16 = {sign_v, 15'b0};
      end else if (abs_val >= 65504.0) begin
        real_to_fp16 = {sign_v, 5'b11111, 10'b0};
      end else begin
        exp_int = 0;
        mant_real = abs_val;
        while (mant_real >= 2.0) begin mant_real = mant_real / 2.0; exp_int = exp_int + 1; end
        while (mant_real < 1.0 && exp_int > -14) begin mant_real = mant_real * 2.0; exp_int = exp_int - 1; end
        biased_exp = exp_int[4:0] + 5'd15;
        mant_int = $rtoi((mant_real - 1.0) * 1024.0);
        mant_bits = mant_int[9:0];
        real_to_fp16 = {sign_v, biased_exp, mant_bits};
      end
    end
  endfunction

  //==========================================================================
  // Test
  //==========================================================================
  integer i, errors;
  real x_val, y_hw, y_ref, abs_err, max_err;
  reg [15:0] test_input [0:NUM_ELEMS-1];
  reg [15:0] out_vals [0:NUM_ELEMS-1];

  initial begin
    $display("=== Reset + NLI Compute Verification ===");

    $readmemh("config/config_rom.mem", config_rom);
    $readmemh("config/func_lut.mem", func_lut);

    //====================================================================
    // Phase 1: Reset verification
    //====================================================================
    $display("\n--- Phase 1: Reset ---");
    rst_n = 0;
    repeat(10) @(posedge clk);

    // Check reset state
    $display("  rst_n=0: compute state=%0d (expect 0=IDLE)", u_compute.state);
    $display("  rst_n=0: in_done=%0b out_done=%0b", u_compute.in_done, u_compute.out_done);
    $display("  rst_n=0: NLI engine p1_valid=%0b", u_compute.u_nli_engine.p1_valid);

    // Deassert reset
    rst_n = 1;
    repeat(5) @(posedge clk);
    $display("  rst_n=1: compute state=%0d (expect 0=IDLE)", u_compute.state);
    $display("  rst_n=1: s_tready=%0b (expect 1)", s_tready);

    //====================================================================
    // Phase 2: Config loading
    //====================================================================
    $display("\n--- Phase 2: Load config ---");

    // Load config_rom (64 entries, cfg_sel=0)
    for (i = 0; i < 64; i++) begin
      @(posedge clk);
      cfg_wdata <= config_rom[i];
      cfg_sel <= 0;
      cfg_addr <= i;
      cfg_we <= 1;
      @(posedge clk);
      cfg_we <= 0;
    end

    // Verify config_rom[0]
    $display("  config_rom[0] = 0x%04h (expect 0x00B1)", u_compute.u_nli_engine.config_rom[0]);
    $display("  config_rom[47] = 0x%04h (expect 0x0A68)", u_compute.u_nli_engine.config_rom[47]);

    // Load func_lut (256 entries, cfg_sel=1)
    for (i = 0; i < 256; i++) begin
      @(posedge clk);
      cfg_wdata <= func_lut[i];
      cfg_sel <= 1;
      cfg_addr <= i;
      cfg_we <= 1;
      @(posedge clk);
      cfg_we <= 0;
    end

    // Verify func_lut
    $display("  func_lut[176] = 0x%04h (expect 0x3800 = 0.5)", u_compute.u_nli_engine.func_lut[176]);
    $display("  func_lut[255] = 0x%04h (expect 0x3C00 = 1.0)", u_compute.u_nli_engine.func_lut[255]);

    //====================================================================
    // Phase 3: Compute test - sigmoid on 32 values
    //====================================================================
    $display("\n--- Phase 3: Compute test ---");

    // Generate input: linspace(-4, 4, 32)
    for (i = 0; i < NUM_ELEMS; i++) begin
      x_val = -4.0 + 8.0 * real'(i) / real'(NUM_ELEMS - 1);
      test_input[i] = real_to_fp16(x_val);
    end

    // Pack into 512-bit word
    s_tdata = 0;
    for (i = 0; i < NUM_ELEMS; i++)
      s_tdata[i*16 +: 16] = test_input[i];

    // Send one beat with proper handshake
    @(posedge clk);
    s_tvalid <= 1;
    s_tlast <= 1;

    // Wait for input accepted
    forever begin
      @(posedge clk);
      if (s_tvalid && s_tready) begin
        s_tvalid <= 0;
        s_tlast <= 0;
        $display("  Input accepted at t=%0t", $time);
        break;
      end
    end

    // Wait for output with proper sampling
    $display("  Waiting for output (m_tready=%0b)...", m_tready);
    forever begin
      @(posedge clk);
      if (m_tvalid) begin
        $display("  Output received at t=%0t", $time);
        break;
      end
    end

    // Extract results
    for (i = 0; i < NUM_ELEMS; i++)
      out_vals[i] = m_tdata[i*16 +: 16];

    @(posedge clk);  // tready acknowledged

    //====================================================================
    // Phase 4: Verify
    //====================================================================
    $display("\n--- Phase 4: Verify sigmoid ---");
    errors = 0;
    max_err = 0.0;

    for (i = 0; i < NUM_ELEMS; i++) begin
      x_val = fp16_to_real(test_input[i]);
      y_hw  = fp16_to_real(out_vals[i]);
      y_ref = 1.0 / (1.0 + $exp(-x_val));
      abs_err = (y_hw > y_ref) ? (y_hw - y_ref) : (y_ref - y_hw);
      if (abs_err > max_err) max_err = abs_err;
      if (abs_err > 0.02) errors++;

      $display("  [%2d] x=%7.3f  hw=%8.6f (0x%04h)  ref=%8.6f  err=%e %s",
               i, x_val, y_hw, out_vals[i], y_ref, abs_err,
               (abs_err > 0.02) ? "FAIL" : "OK");
    end

    $display("\n--- Summary ---");
    $display("  Elements: %0d", NUM_ELEMS);
    $display("  Max abs error: %e", max_err);
    $display("  Errors (>0.02): %0d", errors);

    if (errors == 0)
      $display("\n*** TEST PASSED ***\n");
    else
      $display("\n*** TEST FAILED (%0d errors) ***\n", errors);

    #100 $finish;
  end

  // Timeout
  initial begin
    #(CLK_PERIOD * 100000);
    $display("ERROR: Simulation timeout!");
    $display("  state=%0d in_cnt=%0d out_cnt=%0d", u_compute.state, u_compute.in_cnt, u_compute.out_cnt);
    $finish;
  end

endmodule
