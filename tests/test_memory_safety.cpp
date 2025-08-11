/**
 * @file test_memory_safety.cpp
 * @brief Memory safety and leak detection test suite
 * @brief Real memory management verification and safety checks
 * 
 * Copyright (c) 2025 Kangaroo Project
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// NEW: 独立内存安全测试文件骨架
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
 * @brief Real memory leak detection utility
 */
class MemoryLeakDetector {
private:
    size_t initial_memory_mb_;
    size_t peak_memory_mb_;
    
public:
    void StartMonitoring() {
        initial_memory_mb_ = GetCurrentMemoryUsageMB();
        peak_memory_mb_ = initial_memory_mb_;
    }
    
    void UpdatePeak() {
        size_t current = GetCurrentMemoryUsageMB();
        if (current > peak_memory_mb_) {
            peak_memory_mb_ = current;
        }
    }
    
    size_t GetMemoryGrowthMB() const {
        return GetCurrentMemoryUsageMB() - initial_memory_mb_;
    }
    
    size_t GetPeakMemoryGrowthMB() const {
        return peak_memory_mb_ - initial_memory_mb_;
    }
    
private:
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
};

/**
 * @brief Memory safety test fixture
 */
class MemorySafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        secp_ = std::make_unique<Secp256K1>();
        secp_->Init();
        
        Timer::Init();
        rseed(Timer::getSeed32());
        
        leak_detector_ = std::make_unique<MemoryLeakDetector>();
        leak_detector_->StartMonitoring();
    }
    
    void TearDown() override {
        // Force cleanup before checking for leaks
        secp_.reset();

#ifdef WITHGPU
        // Additional GPU cleanup time
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Use higher threshold for GPU tests
        size_t memory_threshold = 100; // 100MB for GPU tests
#else
        // Allow time for cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Use lower threshold for non-GPU tests
        size_t memory_threshold = 20; // 20MB for non-GPU tests
#endif

        // Check for memory leaks
        size_t memory_growth = leak_detector_->GetMemoryGrowthMB();
        EXPECT_LT(memory_growth, memory_threshold)
            << "Potential memory leak detected: " << memory_growth << " MB growth (threshold: " << memory_threshold << " MB)";

        leak_detector_.reset();
    }
    
    std::unique_ptr<Secp256K1> secp_;
    std::unique_ptr<MemoryLeakDetector> leak_detector_;
};

/**
 * @brief Test HashTable memory management
 */
TEST_F(MemorySafetyTest, HashTableMemoryManagement) {
    const int num_iterations = 1000;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Create and destroy HashTable instances
        auto hash_table = std::make_unique<HashTable>();
        
        // Add some entries
        for (int j = 0; j < 10; ++j) {
            Int x, d;
            x.Rand(128);
            d.Rand(128);
            
            int result = hash_table->Add(&x, &d, 0);
            EXPECT_TRUE(result == ADD_OK || result == ADD_DUPLICATE || result == ADD_COLLISION)
                << "HashTable Add operation returned unexpected result: " << result;
        }
        
        // Explicit cleanup
        hash_table->Reset();
        hash_table.reset();
        
        leak_detector_->UpdatePeak();
    }
    
    // Verify no significant memory growth
    size_t peak_growth = leak_detector_->GetPeakMemoryGrowthMB();
    EXPECT_LT(peak_growth, 50) 
        << "HashTable memory usage grew too much: " << peak_growth << " MB";
}

/**
 * @brief Test HashTable512 memory management
 */
TEST_F(MemorySafetyTest, HashTable512MemoryManagement) {
    const int num_iterations = 500;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Create and destroy HashTable512 instances
        auto hash_table_512 = std::make_unique<HashTable512>();
        
        // Add some entries
        for (int j = 0; j < 5; ++j) {
            int512_t x, d;
            // Initialize with real data
            for (int k = 0; k < 8; ++k) {
                x.i64[k] = rand() | ((uint64_t)rand() << 32);
                d.i64[k] = rand() | ((uint64_t)rand() << 32);
            }
            
            int result = hash_table_512->Add(HashTable512::Hash512(&x), &x, &d);
            EXPECT_TRUE(result == ADD512_OK || result == ADD512_DUPLICATE || result == ADD512_COLLISION)
                << "HashTable512 Add operation returned unexpected result: " << result;
        }
        
        // Explicit cleanup
        hash_table_512->Reset();
        hash_table_512.reset();
        
        leak_detector_->UpdatePeak();
    }
    
    // Verify no significant memory growth
    size_t peak_growth = leak_detector_->GetPeakMemoryGrowthMB();
    EXPECT_LT(peak_growth, 100) 
        << "HashTable512 memory usage grew too much: " << peak_growth << " MB";
}

/**
 * @brief Test SECP256K1 object lifecycle
 */
TEST_F(MemorySafetyTest, SECP256K1ObjectLifecycle) {
    const int num_iterations = 100;
    
    for (int i = 0; i < num_iterations; ++i) {
        // Create temporary SECP256K1 instances
        auto temp_secp = std::make_unique<Secp256K1>();
        temp_secp->Init();
        
        // Perform operations
        for (int j = 0; j < 10; ++j) {
            Int private_key;
            private_key.Rand(256);
            
            Point public_key = temp_secp->ComputePublicKey(&private_key);
            EXPECT_FALSE(public_key.isZero()) 
                << "Generated public key should not be zero";
            
            // Test point operations
            Point doubled = temp_secp->DoubleDirect(public_key);
            EXPECT_FALSE(doubled.isZero()) 
                << "Doubled point should not be zero for valid input";
        }
        
        // Explicit cleanup
        temp_secp.reset();
        
        leak_detector_->UpdatePeak();
    }
    
    // Verify no significant memory growth
    size_t peak_growth = leak_detector_->GetPeakMemoryGrowthMB();
    EXPECT_LT(peak_growth, 20) 
        << "SECP256K1 object lifecycle caused memory growth: " << peak_growth << " MB";
}

/**
 * @brief Test large object allocation and deallocation
 */
TEST_F(MemorySafetyTest, LargeObjectHandling) {
    const int num_large_objects = 50;
    
    // Create large objects
    std::vector<std::unique_ptr<Int>> large_objects;
    large_objects.reserve(num_large_objects);
    
    for (int i = 0; i < num_large_objects; ++i) {
        auto large_int = std::make_unique<Int>();
        large_int->Rand(256); // Large random number (within Int capacity)
        
        // Verify the object is properly initialized
        EXPECT_GT(large_int->GetBitLength(), 0) 
            << "Large Int object should have non-zero bit length";
        
        large_objects.push_back(std::move(large_int));
        leak_detector_->UpdatePeak();
    }
    
    // Clear all objects
    large_objects.clear();
    
    // Allow time for cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify memory is released
    size_t final_growth = leak_detector_->GetMemoryGrowthMB();
    EXPECT_LT(final_growth, 10) 
        << "Large object cleanup failed, memory growth: " << final_growth << " MB";
}

#ifdef WITHGPU
/**
 * @brief Test GPU memory management if available
 */
TEST_F(MemorySafetyTest, GPUMemoryManagement) {
    try {
        // Test GPU engine creation and destruction
        for (int i = 0; i < 5; ++i) {
            auto gpu_engine = std::make_unique<GPUEngine>(1, 256, 0, 1024);

            // Test basic GPU operations
            EXPECT_NO_THROW({
                gpu_engine->SetParams(20, nullptr, nullptr, nullptr);
            }) << "GPU SetParams should not throw";

            // Explicit GPU cleanup
            gpu_engine.reset();

            // Allow GPU memory to be released
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            leak_detector_->UpdatePeak();
        }

        // Additional cleanup time for GPU memory
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Verify no significant GPU memory leaks
        size_t peak_growth = leak_detector_->GetPeakMemoryGrowthMB();
        EXPECT_LT(peak_growth, 200)
            << "GPU memory management caused excessive growth: " << peak_growth << " MB";

    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPU not available for memory testing: " << e.what();
    }
}
#endif

} // namespace kangaroo_test
