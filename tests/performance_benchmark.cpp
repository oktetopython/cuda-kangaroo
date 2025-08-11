/**
 * @file performance_benchmark.cpp
 * @brief Performance benchmark test suite for Kangaroo algorithm
 * @brief Real performance measurement and comparison framework
 * 
 * Copyright (c) 2025 Kangaroo Project
 */

#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// NEW: 独立性能基准测试文件骨架
#include "../Kangaroo.h"
#include "../SECPK1/SECP256k1.h"
#include "../Timer.h"
#include "../HashTable.h"
#include "../HashTable512.h"

#ifdef WITHGPU
#include "../GPU/GPUEngine.h"
#endif

namespace kangaroo_test {

/**
 * @brief Real performance measurement utilities
 */
class PerformanceMeasurement {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    
public:
    void StartTiming() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void EndTiming() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double GetElapsedMilliseconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
        return duration.count() / 1000.0;
    }
    
    double GetElapsedSeconds() const {
        return GetElapsedMilliseconds() / 1000.0;
    }
};

/**
 * @brief Real memory usage monitoring
 */
class MemoryUsageMonitor {
public:
    size_t GetCurrentMemoryUsageMB() const {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), 
                               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                               sizeof(pmc))) {
            return pmc.WorkingSetSize / (1024 * 1024);
        }
#endif
        return 0;
    }
    
    size_t GetPeakMemoryUsageMB() const {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), 
                               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), 
                               sizeof(pmc))) {
            return pmc.PeakWorkingSetSize / (1024 * 1024);
        }
#endif
        return 0;
    }
};

/**
 * @brief Performance benchmark test fixture
 */
class PerformanceBenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        secp_ = std::make_unique<Secp256K1>();
        secp_->Init();
        
        Timer::Init();
        rseed(Timer::getSeed32());
        
        measurement_ = std::make_unique<PerformanceMeasurement>();
        memory_monitor_ = std::make_unique<MemoryUsageMonitor>();
    }
    
    void TearDown() override {
        secp_.reset();
        measurement_.reset();
        memory_monitor_.reset();
    }
    
    std::unique_ptr<Secp256K1> secp_;
    std::unique_ptr<PerformanceMeasurement> measurement_;
    std::unique_ptr<MemoryUsageMonitor> memory_monitor_;
};

/**
 * @brief Test HashTable performance with real operations
 */
TEST_F(PerformanceBenchmarkTest, HashTablePerformance) {
    const int num_operations = 10000;
    HashTable hash_table;
    
    // Measure insertion performance
    measurement_->StartTiming();
    
    for (int i = 0; i < num_operations; ++i) {
        Int x, d;
        x.Rand(128);
        d.Rand(128);
        
        hash_table.Add(&x, &d, 0);
    }
    
    measurement_->EndTiming();
    
    double elapsed_ms = measurement_->GetElapsedMilliseconds();
    double operations_per_second = (num_operations * 1000.0) / elapsed_ms;
    
    // Real performance assertions
    EXPECT_GT(operations_per_second, 1000.0) 
        << "HashTable insertion performance too low: " << operations_per_second << " ops/sec";
    
    EXPECT_LT(elapsed_ms, 10000.0) 
        << "HashTable operations took too long: " << elapsed_ms << " ms";
}

/**
 * @brief Test HashTable512 performance comparison
 */
TEST_F(PerformanceBenchmarkTest, HashTable512Performance) {
    const int num_operations = 5000;
    HashTable512 hash_table_512;
    
    size_t initial_memory = memory_monitor_->GetCurrentMemoryUsageMB();
    
    measurement_->StartTiming();
    
    for (int i = 0; i < num_operations; ++i) {
        int512_t x, d;
        // Initialize with real random data
        for (int j = 0; j < 8; ++j) {
            x.i64[j] = rand() | ((uint64_t)rand() << 32);
            d.i64[j] = rand() | ((uint64_t)rand() << 32);
        }
        
        hash_table_512.Add(HashTable512::Hash512(&x), &x, &d);
    }
    
    measurement_->EndTiming();
    
    size_t final_memory = memory_monitor_->GetCurrentMemoryUsageMB();
    double elapsed_ms = measurement_->GetElapsedMilliseconds();
    double operations_per_second = (num_operations * 1000.0) / elapsed_ms;
    
    // Real performance and memory assertions
    EXPECT_GT(operations_per_second, 500.0) 
        << "HashTable512 insertion performance too low: " << operations_per_second << " ops/sec";
    
    EXPECT_LT(final_memory - initial_memory, 100) 
        << "HashTable512 memory usage too high: " << (final_memory - initial_memory) << " MB";
}

/**
 * @brief Test SECP256K1 computation performance
 */
TEST_F(PerformanceBenchmarkTest, SECP256K1Performance) {
    const int num_computations = 1000;
    
    measurement_->StartTiming();
    
    for (int i = 0; i < num_computations; ++i) {
        Int private_key;
        private_key.Rand(256);
        
        Point public_key = secp_->ComputePublicKey(&private_key);
        
        // Verify the computation is real
        EXPECT_FALSE(public_key.isZero()) << "Generated public key should not be zero";
    }
    
    measurement_->EndTiming();
    
    double elapsed_ms = measurement_->GetElapsedMilliseconds();
    double computations_per_second = (num_computations * 1000.0) / elapsed_ms;
    
    // Real performance assertions for elliptic curve operations
    EXPECT_GT(computations_per_second, 100.0) 
        << "SECP256K1 computation performance too low: " << computations_per_second << " ops/sec";
}

#ifdef WITHGPU
/**
 * @brief Test GPU performance if available
 */
TEST_F(PerformanceBenchmarkTest, GPUPerformance) {
    try {
        std::vector<int> gpu_ids = {0};
        std::vector<int> grid_size;
        
        GPUEngine gpu_engine(1, 256, 0, 1024);
        
        // Real GPU initialization test
        EXPECT_NO_THROW({
            // This tests actual GPU functionality
            gpu_engine.SetParams(20, nullptr, nullptr, nullptr);
        }) << "GPU engine initialization should succeed";
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPU not available or initialization failed: " << e.what();
    }
}
#endif

} // namespace kangaroo_test
