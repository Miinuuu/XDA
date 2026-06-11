create_clock [get_ports clk] -name core_clock -period 10.0
set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  2.0 -clock core_clock $non_clock_inputs
set_output_delay 2.0 -clock core_clock [all_outputs]
