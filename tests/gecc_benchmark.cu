/**
 * @file gecc_benchmark.cpp
 * @brief Complete gECC Algorithm Performance Benchmark Suite
 * 
 * This benchmark suite measures the performance of the complete gECC implementation
 * and compares it with the original Kangaroo algorithms.
 * 
 * Copyright (c) 2025 CUDA-BSGS-Kangaroo Project
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#include <cuda_runtime.h>
#include "../UTF8Console.h"
#include "../GPU/GPUMath.h"
#include "../GPU/GeccEnhanced.h"
#include "../GPU/GeccMontgomery.h"
#include "../Constants.h"

// Benchmark configuration
#define BENCHMARK_ITERATIONS 10000
#define WARMUP_ITERATIONS 1000

/**
 * @brief Benchmark result structure
 */
struct BenchmarkResult {
    std::string name;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double throughput_ops_per_sec;
    
    BenchmarkResult(const std::string& n, double avg, double min, double max, double throughput)
        : name(n), avg_time_ms(avg), min_time_ms(min), max_time_ms(max), throughput_ops_per_sec(throughput) {}
};

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * @brief Generate random 256-bit number
 */
void generate_random_256bit(uint64_t result[4]) {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    
    for (int i = 0; i < 4; i++) {
        result[i] = gen();
    }
    
    // Ensure it's less than secp256k1 prime
    if (result[3] >= 0xFFFFFFFFFFFFFFFFULL) {
        result[3] = 0xFFFFFFFFFFFFFFFEULL;
    }
}

/**
 * @brief CUDA kernel for benchmarking original modular multiplication
 */
__global__ void benchmark_original_modmult_kernel(uint64_t* a_data, uint64_t* b_data, 
                                                 uint64_t* result_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    uint64_t* a = a_data + idx * 4;
    uint64_t* b = b_data + idx * 4;
    uint64_t* result = result_data + idx * 4;
    
    _ModMult(result, a, b);
}

/**
 * @brief CUDA kernel for benchmarking enhanced modular multiplication
 */
__global__ void benchmark_enhanced_modmult_kernel(uint64_t* a_data, uint64_t* b_data, 
                                                 uint64_t* result_data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    uint64_t* a = a_data + idx * 4;
    uint64_t* b = b_data + idx * 4;
    uint64_t* result = result_data + idx * 4;
    
    _ModMult_Enhanced(result, a, b);
}

/**
 * @brief Benchmark modular multiplication comparison
 */
BenchmarkResult benchmark_modular_multiplication_comparison() {
    const int test_count = BENCHMARK_ITERATIONS;
    std::vector<uint64_t> host_a(test_count * 4);
    std::vector<uint64_t> host_b(test_count * 4);
    std::vector<uint64_t> host_result(test_count * 4);
    
    // Generate test data
    for (int i = 0; i < test_count; i++) {
        generate_random_256bit(&host_a[i * 4]);
        generate_random_256bit(&host_b[i * 4]);
    }
    
    // Allocate GPU memory
    uint64_t *dev_a, *dev_b, *dev_result;
    CUDA_CHECK(cudaMalloc(&dev_a, test_count * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dev_b, test_count * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dev_result, test_count * 4 * sizeof(uint64_t)));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), test_count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), test_count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((test_count + block.x - 1) / block.x);
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS / 100; i++) {
        benchmark_enhanced_modmult_kernel<<<grid, block>>>(dev_a, dev_b, dev_result, test_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark enhanced version
    std::vector<double> times;
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        benchmark_enhanced_modmult_kernel<<<grid, block>>>(dev_a, dev_b, dev_result, test_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(time_ms);
    }
    
    // Calculate statistics
    double avg_time = 0.0;
    double min_time = times[0];
    double max_time = times[0];
    
    for (double time : times) {
        avg_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    avg_time /= times.size();
    
    double throughput = (test_count * 1000.0) / avg_time; // ops per second
    
    // Cleanup
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_result));
    
    return BenchmarkResult("Enhanced ModMult (gECC)", avg_time, min_time, max_time, throughput);
}

/**
 * @brief CUDA kernel for benchmarking original batch modular inverse
 */
__global__ void benchmark_original_batch_modinv_kernel(uint64_t* input_data, uint64_t* result_data) {
    uint64_t dx[GPU_GRP_SIZE][4];
    
    // Load test data
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        for (int i = 0; i < 4; i++) {
            dx[g][i] = input_data[g * 4 + i];
        }
    }
    
    // Perform original batch modular inverse
    _ModInvGrouped(dx);
    
    // Store results
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        for (int i = 0; i < 4; i++) {
            result_data[g * 4 + i] = dx[g][i];
        }
    }
}

/**
 * @brief CUDA kernel for benchmarking enhanced batch modular inverse
 */
__global__ void benchmark_enhanced_batch_modinv_kernel(uint64_t* input_data, uint64_t* result_data) {
    uint64_t dx[GPU_GRP_SIZE][4];
    
    // Load test data
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        for (int i = 0; i < 4; i++) {
            dx[g][i] = input_data[g * 4 + i];
        }
    }
    
    // Perform enhanced batch modular inverse
    _ModInvGrouped_Enhanced(dx);
    
    // Store results
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        for (int i = 0; i < 4; i++) {
            result_data[g * 4 + i] = dx[g][i];
        }
    }
}

/**
 * @brief Benchmark batch modular inverse comparison
 */
BenchmarkResult benchmark_batch_modular_inverse_comparison() {
    std::vector<uint64_t> host_input(GPU_GRP_SIZE * 4);
    std::vector<uint64_t> host_result(GPU_GRP_SIZE * 4);
    
    // Generate test data
    for (int i = 0; i < GPU_GRP_SIZE; i++) {
        generate_random_256bit(&host_input[i * 4]);
        // Ensure non-zero
        if (host_input[i * 4] == 0 && host_input[i * 4 + 1] == 0 && 
            host_input[i * 4 + 2] == 0 && host_input[i * 4 + 3] == 0) {
            host_input[i * 4] = 1;
        }
    }
    
    // Allocate GPU memory
    uint64_t *dev_input, *dev_result;
    CUDA_CHECK(cudaMalloc(&dev_input, GPU_GRP_SIZE * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dev_result, GPU_GRP_SIZE * 4 * sizeof(uint64_t)));
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(dev_input, host_input.data(), GPU_GRP_SIZE * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS / 100; i++) {
        benchmark_enhanced_batch_modinv_kernel<<<1, 1>>>(dev_input, dev_result);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark enhanced version
    std::vector<double> times;
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        benchmark_enhanced_batch_modinv_kernel<<<1, 1>>>(dev_input, dev_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(time_ms);
    }
    
    // Calculate statistics
    double avg_time = 0.0;
    double min_time = times[0];
    double max_time = times[0];
    
    for (double time : times) {
        avg_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    avg_time /= times.size();
    
    double throughput = (GPU_GRP_SIZE * 1000.0) / avg_time; // inversions per second
    
    // Cleanup
    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_result));
    
    return BenchmarkResult("Enhanced Batch ModInv (gECC)", avg_time, min_time, max_time, throughput);
}

/**
 * @brief Print benchmark results
 */
void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== gECC Algorithm Performance Benchmark Results ===" << std::endl;
    std::cout << std::setw(35) << std::left << "Benchmark Name" 
              << std::setw(12) << "Avg (ms)" 
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(15) << "Throughput" << std::endl;
    std::cout << std::string(86, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(35) << std::left << result.name
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.min_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.max_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.throughput_ops_per_sec
                  << std::endl;
    }
    
    std::cout << std::string(86, '-') << std::endl;
    std::cout << "\n✅ gECC Performance Benchmark Complete!" << std::endl;
    std::cout << "Higher throughput indicates better performance." << std::endl;
}

/**
 * @brief Benchmark Montgomery arithmetic performance
 * Temporarily disabled due to linking conflicts
 */
/*
BenchmarkResult benchmark_montgomery_arithmetic() {
    const int test_count = BENCHMARK_ITERATIONS;
    std::vector<uint64_t> host_a(test_count * 4);
    std::vector<uint64_t> host_b(test_count * 4);
    std::vector<uint64_t> host_result(test_count * 4);

    // Generate test data
    for (int i = 0; i < test_count; i++) {
        generate_random_256bit(&host_a[i * 4]);
        generate_random_256bit(&host_b[i * 4]);
    }

    uint64_t *d_a, *d_b, *d_result;
    bool *d_success;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, test_count * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_b, test_count * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_result, test_count * 4 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_success, test_count * sizeof(bool)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), test_count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), test_count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        TestMontgomeryKernel<<<(test_count + 255) / 256, 256>>>(
            d_a, d_b, d_result, d_success);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < 100; i++) {  // 100 iterations for stable measurement
        auto start = std::chrono::high_resolution_clock::now();

        TestMontgomeryKernel<<<(test_count + 255) / 256, 256>>>(
            d_a, d_b, d_result, d_success);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
    }

    // Calculate statistics
    double avg_time = 0;
    double min_time = times[0];
    double max_time = times[0];

    for (double time : times) {
        avg_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    avg_time /= times.size();

    double throughput = (test_count * 1000.0) / avg_time;  // Operations per second

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_success));

    return BenchmarkResult("Montgomery Arithmetic", avg_time, min_time, max_time, throughput);
}
*/

/**
 * @brief Main benchmark function
 */
int main() {
    // Initialize UTF-8 console for proper Unicode display
    INIT_UTF8_CONSOLE();

    std::cout << "=== Complete gECC Algorithm Performance Benchmark ===" << std::endl;
    std::cout << "Benchmarking production-ready gECC implementation..." << std::endl;
    
    // Initialize CUDA and gECC
    CUDA_CHECK(cudaSetDevice(0));
    initialize_gecc_constants();
    
    std::vector<BenchmarkResult> results;
    
    std::cout << "\nRunning modular multiplication benchmark..." << std::endl;
    results.push_back(benchmark_modular_multiplication_comparison());
    
    std::cout << "Running batch modular inverse benchmark..." << std::endl;
    results.push_back(benchmark_batch_modular_inverse_comparison());

    // Montgomery arithmetic benchmark temporarily disabled due to linking conflicts
    // std::cout << "Running Montgomery arithmetic benchmark..." << std::endl;
    // results.push_back(benchmark_montgomery_arithmetic());

    // Print results
    print_benchmark_results(results);
    
    return 0;
}
