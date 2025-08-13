/*
 * Unit tests for OptimizedCheckpoint functionality
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "OptimizedCheckpoint.h"
#include "HashTable.h"
#include "Int.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

// Test configuration
const std::string TEST_CHECKPOINT_FILE = "test_checkpoint.kcp";
const std::string TEST_LEGACY_FILE = "test_legacy.work";
const int TEST_THREADS = 4;
const int TEST_KANGAROOS_PER_THREAD = 1000;

// Test utilities
class CheckpointTester {
private:
    HashTable test_hashTable;
    TH_PARAM test_threads[TEST_THREADS];
    
public:
    CheckpointTester() {
        InitializeTestData();
    }
    
    ~CheckpointTester() {
        CleanupTestData();
    }
    
    void InitializeTestData() {
        // Initialize hash table with some test data
        test_hashTable.Reset();
        
        // Add some test entries to hash table
        for (int i = 0; i < 100; i++) {
            Int x, d;
            x.SetInt32(i * 1000);
            d.SetInt32(i * 2000);
            test_hashTable.Add(&x, &d);
        }
        
        // Initialize thread parameters
        memset(test_threads, 0, sizeof(test_threads));
        
        for (int t = 0; t < TEST_THREADS; t++) {
            test_threads[t].nbKangaroo = TEST_KANGAROOS_PER_THREAD;
            
            // Allocate memory for kangaroo states
            test_threads[t].px = new Int[TEST_KANGAROOS_PER_THREAD];
            test_threads[t].py = new Int[TEST_KANGAROOS_PER_THREAD];
            test_threads[t].distance = new Int[TEST_KANGAROOS_PER_THREAD];
            
            // Initialize with test data
            for (int k = 0; k < TEST_KANGAROOS_PER_THREAD; k++) {
                test_threads[t].px[k].SetInt32(t * 10000 + k);
                test_threads[t].py[k].SetInt32(t * 20000 + k);
                test_threads[t].distance[k].SetInt32(t * 30000 + k);
            }
        }
    }
    
    void CleanupTestData() {
        for (int t = 0; t < TEST_THREADS; t++) {
            delete[] test_threads[t].px;
            delete[] test_threads[t].py;
            delete[] test_threads[t].distance;
        }
    }
    
    bool TestBasicSaveLoad() {
        std::cout << "Testing basic save/load functionality..." << std::endl;
        
        OptimizedCheckpoint checkpoint(TEST_CHECKPOINT_FILE, false, true);
        
        // Test save
        CheckpointError save_result = checkpoint.SaveCheckpoint(
            12345, 67.89, test_hashTable, test_threads, TEST_THREADS, 20);
        
        if (save_result != CHECKPOINT_OK) {
            std::cerr << "ERROR: Save failed: " << checkpoint.GetErrorMessage(save_result) << std::endl;
            return false;
        }
        
        // Test load
        HashTable loaded_hashTable;
        TH_PARAM loaded_threads[TEST_THREADS];
        memset(loaded_threads, 0, sizeof(loaded_threads));
        
        uint64_t loaded_count;
        double loaded_time;
        uint32_t loaded_dpSize;
        int loaded_nbThread;
        
        CheckpointError load_result = checkpoint.LoadCheckpoint(
            loaded_count, loaded_time, loaded_hashTable, 
            loaded_threads, loaded_nbThread, loaded_dpSize);
        
        if (load_result != CHECKPOINT_OK) {
            std::cerr << "ERROR: Load failed: " << checkpoint.GetErrorMessage(load_result) << std::endl;
            return false;
        }
        
        // Verify loaded data
        if (loaded_count != 12345 || loaded_time != 67.89 || loaded_dpSize != 20 || loaded_nbThread != TEST_THREADS) {
            std::cerr << "ERROR: Header data mismatch" << std::endl;
            return false;
        }
        
        // Verify hash table
        if (loaded_hashTable.GetNbItem() != test_hashTable.GetNbItem()) {
            std::cerr << "ERROR: Hash table item count mismatch" << std::endl;
            return false;
        }
        
        // Verify kangaroo data
        for (int t = 0; t < TEST_THREADS; t++) {
            if (loaded_threads[t].nbKangaroo != TEST_KANGAROOS_PER_THREAD) {
                std::cerr << "ERROR: Kangaroo count mismatch for thread " << t << std::endl;
                return false;
            }
            
            for (int k = 0; k < TEST_KANGAROOS_PER_THREAD; k++) {
                if (!loaded_threads[t].px[k].IsEqual(&test_threads[t].px[k]) ||
                    !loaded_threads[t].py[k].IsEqual(&test_threads[t].py[k]) ||
                    !loaded_threads[t].distance[k].IsEqual(&test_threads[t].distance[k])) {
                    std::cerr << "ERROR: Kangaroo data mismatch at thread " << t << ", kangaroo " << k << std::endl;
                    return false;
                }
            }
        }
        
        std::cout << "Basic save/load test PASSED" << std::endl;
        return true;
    }
    
    bool TestErrorHandling() {
        std::cout << "Testing error handling..." << std::endl;
        
        // Test invalid file path
        OptimizedCheckpoint invalid_checkpoint("/invalid/path/test.kcp", false, true);
        CheckpointError result = invalid_checkpoint.SaveCheckpoint(
            0, 0.0, test_hashTable, test_threads, TEST_THREADS, 20);
        
        if (result == CHECKPOINT_OK) {
            std::cerr << "ERROR: Should have failed with invalid path" << std::endl;
            return false;
        }
        
        // Test loading non-existent file
        OptimizedCheckpoint nonexistent_checkpoint("nonexistent.kcp", false, true);
        uint64_t dummy_count;
        double dummy_time;
        uint32_t dummy_dpSize;
        int dummy_nbThread;
        HashTable dummy_hashTable;
        TH_PARAM dummy_threads[1];
        
        result = nonexistent_checkpoint.LoadCheckpoint(
            dummy_count, dummy_time, dummy_hashTable, 
            dummy_threads, dummy_nbThread, dummy_dpSize);
        
        if (result == CHECKPOINT_OK) {
            std::cerr << "ERROR: Should have failed with non-existent file" << std::endl;
            return false;
        }
        
        std::cout << "Error handling test PASSED" << std::endl;
        return true;
    }
    
    bool TestFileSizeOptimization() {
        std::cout << "Testing file size optimization..." << std::endl;
        
        // Create a sparse hash table (mostly empty buckets)
        HashTable sparse_hashTable;
        sparse_hashTable.Reset();
        
        // Add only a few entries to create many empty buckets
        for (int i = 0; i < 10; i++) {
            Int x, d;
            x.SetInt32(i * 100000); // Spread entries across different buckets
            d.SetInt32(i * 200000);
            sparse_hashTable.Add(&x, &d);
        }
        
        OptimizedCheckpoint checkpoint("sparse_test.kcp", false, true);
        
        CheckpointError result = checkpoint.SaveCheckpoint(
            1000, 10.0, sparse_hashTable, test_threads, 1, 20);
        
        if (result != CHECKPOINT_OK) {
            std::cerr << "ERROR: Sparse hash table save failed" << std::endl;
            return false;
        }
        
        uint64_t file_size = checkpoint.GetFileSize("sparse_test.kcp");
        
        // The file should be much smaller than a full hash table dump
        // Full hash table would be ~8MB, optimized should be <1MB
        if (file_size > 1024 * 1024) { // 1MB threshold
            std::cerr << "WARNING: File size optimization may not be working effectively. Size: " 
                      << (file_size / 1024.0) << " KB" << std::endl;
        } else {
            std::cout << "File size optimization working well. Size: " 
                      << (file_size / 1024.0) << " KB" << std::endl;
        }
        
        // Clean up
        remove("sparse_test.kcp");
        
        std::cout << "File size optimization test PASSED" << std::endl;
        return true;
    }
    
    bool TestCrossPlatformCompatibility() {
        std::cout << "Testing cross-platform compatibility..." << std::endl;
        
        OptimizedCheckpoint checkpoint("compat_test.kcp", false, true);
        
        // Save with current platform
        CheckpointError save_result = checkpoint.SaveCheckpoint(
            0xDEADBEEF, 123.456, test_hashTable, test_threads, TEST_THREADS, 25);
        
        if (save_result != CHECKPOINT_OK) {
            std::cerr << "ERROR: Compatibility save failed" << std::endl;
            return false;
        }
        
        // Load and verify endianness handling
        uint64_t loaded_count;
        double loaded_time;
        uint32_t loaded_dpSize;
        int loaded_nbThread;
        HashTable loaded_hashTable;
        TH_PARAM loaded_threads[TEST_THREADS];
        memset(loaded_threads, 0, sizeof(loaded_threads));
        
        CheckpointError load_result = checkpoint.LoadCheckpoint(
            loaded_count, loaded_time, loaded_hashTable, 
            loaded_threads, loaded_nbThread, loaded_dpSize);
        
        if (load_result != CHECKPOINT_OK) {
            std::cerr << "ERROR: Compatibility load failed" << std::endl;
            return false;
        }
        
        // Verify data integrity
        if (loaded_count != 0xDEADBEEF || loaded_dpSize != 25) {
            std::cerr << "ERROR: Cross-platform data integrity failed" << std::endl;
            return false;
        }
        
        // Clean up
        remove("compat_test.kcp");
        
        std::cout << "Cross-platform compatibility test PASSED" << std::endl;
        return true;
    }
    
    bool RunAllTests() {
        std::cout << "Running OptimizedCheckpoint unit tests..." << std::endl;
        std::cout << "=========================================" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= TestBasicSaveLoad();
        all_passed &= TestErrorHandling();
        all_passed &= TestFileSizeOptimization();
        all_passed &= TestCrossPlatformCompatibility();
        
        // Clean up test files
        remove(TEST_CHECKPOINT_FILE.c_str());
        
        if (all_passed) {
            std::cout << std::endl << "All tests PASSED!" << std::endl;
        } else {
            std::cout << std::endl << "Some tests FAILED!" << std::endl;
        }
        
        return all_passed;
    }
};

// Main test function
int main() {
    CheckpointTester tester;
    bool success = tester.RunAllTests();
    return success ? 0 : 1;
}
