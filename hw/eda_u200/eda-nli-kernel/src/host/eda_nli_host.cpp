// ==============================================================
// EDA-NLI Kernel Host Application
// Loads function config via AXI-Lite, processes FP16 data on U200
// Verifies with test_vectors.mem (HW-accurate expected output)
// Usage: ./host <xclbin> <function_name>
//   function_name: sigmoid, tanh, silu, mish, gelu, hardswish, exp, reciprocal, rsqrt
// ==============================================================
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_ip.h"

// Register offsets
#define USER_OFFSET    0x10
#define SCALAR_OFFSET  0x14
#define A_OFFSET       0x1c
#define B_OFFSET       0x28
#define CFG_CTRL       0x40
#define CFG_WDATA      0x44

#define IP_START  0x1
#define IP_DONE   0x2

// FP16 conversion
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(f));
    return result;
}

// Load hex values from .mem file (one hex value per line, // comments)
static std::vector<uint16_t> load_mem_file(const std::string& filename) {
    std::vector<uint16_t> data;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '/' || line[0] == '#') continue;
        // Take first token (hex value)
        std::istringstream iss(line);
        std::string hex_str;
        if (iss >> hex_str) {
            // Skip comment-only lines
            if (hex_str[0] == '/') continue;
            data.push_back((uint16_t)strtoul(hex_str.c_str(), nullptr, 16));
        }
    }
    return data;
}

// Load test vectors from .mem file
struct TestVector {
    uint16_t x_bits;
    uint16_t y_expected;
};

static std::vector<TestVector> load_test_vectors(const std::string& filename) {
    std::vector<TestVector> vecs;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << filename << std::endl;
        return vecs;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '/') continue;
        std::istringstream iss(line);
        std::string x_hex, y_hex;
        if (iss >> x_hex >> y_hex) {
            TestVector tv;
            tv.x_bits = (uint16_t)strtoul(x_hex.c_str(), nullptr, 16);
            tv.y_expected = (uint16_t)strtoul(y_hex.c_str(), nullptr, 16);
            vecs.push_back(tv);
        }
    }
    return vecs;
}

// Load NLI config from mem files
static bool load_nli_config(xrt::ip& ip, const std::string& config_dir) {
    std::string config_rom_file = config_dir + "/config_rom.mem";
    std::string func_lut_file  = config_dir + "/func_lut.mem";

    auto config_rom = load_mem_file(config_rom_file);
    auto func_lut   = load_mem_file(func_lut_file);

    if (config_rom.empty()) {
        std::cerr << "Failed to load config_rom from " << config_rom_file << std::endl;
        return false;
    }
    if (func_lut.empty()) {
        std::cerr << "Failed to load func_lut from " << func_lut_file << std::endl;
        return false;
    }

    std::cout << "Loading config_rom (" << config_rom.size() << " entries) from " << config_rom_file << std::endl;
    for (size_t i = 0; i < config_rom.size(); i++) {
        ip.write_register(CFG_WDATA, config_rom[i]);
        ip.write_register(CFG_CTRL, (i << 1) | 0);
    }

    std::cout << "Loading func_lut (" << func_lut.size() << " entries) from " << func_lut_file << std::endl;
    for (size_t i = 0; i < func_lut.size(); i++) {
        ip.write_register(CFG_WDATA, func_lut[i]);
        ip.write_register(CFG_CTRL, (i << 1) | 1);
    }

    std::cout << "NLI config loaded." << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <xclbin> <function_name>" << std::endl;
        std::cerr << "  function_name: sigmoid, tanh, silu, mish, gelu, hardswish, exp, reciprocal, rsqrt" << std::endl;
        return 1;
    }

    std::string xclbin_file = argv[1];
    std::string func_name   = argv[2];

    // Derive paths from function name
    std::string config_dir    = "./config/" + func_name;
    std::string test_vec_file = config_dir + "/test_vectors.mem";

    // Load test vectors
    auto tvecs = load_test_vectors(test_vec_file);
    if (tvecs.empty()) {
        std::cerr << "No test vectors loaded from " << test_vec_file << std::endl;
        return 1;
    }
    std::cout << "Loaded " << tvecs.size() << " test vectors from " << test_vec_file << std::endl;

    // Pad to multiple of 32 (512-bit AXI alignment)
    size_t data_size = tvecs.size();
    size_t padded_size = ((data_size + 31) / 32) * 32;
    size_t vector_size_bytes = padded_size * sizeof(uint16_t);

    size_t axi_beats = padded_size / 32;

    std::cout << "=== EDA-NLI [" << func_name << "] Kernel ===" << std::endl;
    std::cout << "Test vectors: " << data_size << ", padded: " << padded_size
              << " (" << vector_size_bytes << " bytes)" << std::endl;
    std::cout << "512-bit AXI: " << axi_beats << " beats x 32 values"
              << " (pad " << (padded_size - data_size) << ")" << std::endl;

    // Open device
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_file);
    auto ip = xrt::ip(device, uuid, "EDA:{EDA_1}");
    std::cout << "Device opened, xclbin loaded." << std::endl;

    // Allocate buffers (memory group 1 - must match platform/kernel connectivity)
    std::cout << "Allocate Buffer in Global Memory" << std::endl;
    auto bo_input  = xrt::bo(device, vector_size_bytes, 1);
    auto bo_output = xrt::bo(device, vector_size_bytes, 1);
    auto input_map  = bo_input.map<uint16_t*>();
    auto output_map = bo_output.map<uint16_t*>();
    std::cout << "Buffers allocated and mapped." << std::endl;

    // Fill input from test vectors
    memset(input_map, 0, vector_size_bytes);
    for (size_t i = 0; i < data_size; i++)
        input_map[i] = tvecs[i].x_bits;
    memset(output_map, 0, vector_size_bytes);

    // Sync to device
    std::cout << "Synchronize input buffer data to device global memory" << std::endl;
    bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "Sync done." << std::endl;

    // Load NLI config from mem files
    if (!load_nli_config(ip, config_dir)) {
        std::cerr << "Failed to load NLI config for " << func_name << std::endl;
        return 1;
    }

    // Set kernel arguments
    uint64_t input_addr  = bo_input.address();
    uint64_t output_addr = bo_output.address();
    ip.write_register(SCALAR_OFFSET, (uint32_t)vector_size_bytes);
    ip.write_register(A_OFFSET,     (uint32_t)(input_addr & 0xFFFFFFFF));
    ip.write_register(A_OFFSET + 4, (uint32_t)(input_addr >> 32));
    ip.write_register(B_OFFSET,     (uint32_t)(output_addr & 0xFFFFFFFF));
    ip.write_register(B_OFFSET + 4, (uint32_t)(output_addr >> 32));

    std::cout << "Starting kernel..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    ip.write_register(USER_OFFSET, IP_START);

    // Poll for completion
    uint32_t status = 0;
    int poll_count = 0;
    do {
        status = ip.read_register(USER_OFFSET);
        poll_count++;
        if (poll_count <= 10 || poll_count % 100 == 0) {
            std::cout << "Read Loop iteration: " << poll_count
                      << " status=0x" << std::hex << status << std::dec
                      << " done=" << ((status & IP_DONE) ? 1 : 0) << std::endl;
        }
    } while (!(status & IP_DONE));
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Kernel done (polled " << poll_count << " times, "
              << elapsed_ms << " ms)." << std::endl;

    // Sync results
    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Verify against test vectors (HW-accurate expected)
    int errors = 0;
    int ulp_errors = 0;
    int max_ulp = 0;
    float max_abs_err = 0.0f;
    int printed_errors = 0;
    const int max_print_errors = 20;

    std::cout << "\n--- Verification [" << func_name << "] (vs HW-accurate expected) ---" << std::endl;
    for (size_t i = 0; i < data_size; i++) {
        uint16_t hw_out = output_map[i];
        uint16_t expected = tvecs[i].y_expected;
        float x_f = fp16_to_float(tvecs[i].x_bits);
        float y_hw = fp16_to_float(hw_out);
        float y_exp = fp16_to_float(expected);

        // ULP distance
        int16_t ulp = (int16_t)hw_out - (int16_t)expected;
        int abs_ulp = (ulp < 0) ? -ulp : ulp;
        if (abs_ulp > max_ulp) max_ulp = abs_ulp;
        if (abs_ulp > 4) ulp_errors++;

        float abs_err = fabsf(y_hw - y_exp);
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (hw_out != expected) errors++;

        // Print first 10, middle, last 5, and limited errors
        bool is_err = (abs_ulp > 4);
        if (i < 10 || i == data_size/2 || i >= data_size-5 || (is_err && printed_errors < max_print_errors)) {
            if (is_err) printed_errors++;
            printf("  [%5zu] x=%8.4f  hw=0x%04X(%8.6f)  exp=0x%04X(%8.6f)  ulp=%d %s\n",
                   i, x_f, hw_out, y_hw, expected, y_exp, ulp,
                   is_err ? "FAIL" : (hw_out == expected) ? "EXACT" : "ok");
        }
    }

    if (ulp_errors > max_print_errors)
        printf("  ... and %d more ULP errors omitted\n", ulp_errors - max_print_errors);

    std::cout << "\n--- Summary [" << func_name << "] ---" << std::endl;
    printf("Test vectors: %zu (%zu AXI beats)\n", data_size, axi_beats);
    printf("Bit-exact matches: %zu/%zu (%.2f%%)\n",
           data_size - errors, data_size, 100.0 * (data_size - errors) / data_size);
    std::cout << "Max ULP distance: " << max_ulp << std::endl;
    std::cout << "ULP errors (>4): " << ulp_errors << std::endl;
    std::cout << "Max abs error: " << max_abs_err << std::endl;
    printf("Kernel time: %.3f ms\n", elapsed_ms);

    if (ulp_errors == 0) {
        std::cout << "\n*** TEST PASSED [" << func_name << "] (all within 4 ULP) ***" << std::endl;
        return 0;
    } else {
        std::cout << "\n*** TEST FAILED [" << func_name << "] (" << ulp_errors << " ULP errors) ***" << std::endl;
        return 1;
    }
}
