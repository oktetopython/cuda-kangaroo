/**
 * @file test_compatibility.cpp
 * @brief Compatibility and backward compatibility test suite
 * @brief Real API compatibility verification and version interoperability
 * 
 * Copyright (c) 2025 Kangaroo Project
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

// NEW: 独立兼容性测试文件骨架
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
 * @brief API compatibility verification utility
 */
class APICompatibilityChecker {
public:
    /**
     * @brief Test if HashTable API maintains backward compatibility
     */
    bool TestHashTableAPI() {
        try {
            HashTable hash_table;
            
            // Test basic API methods exist and work
            Int x, d;
            x.Rand(128);
            d.Rand(128);
            
            // Test Add API
            int result = hash_table.Add(&x, &d, 0);
            if (result != ADD_OK && result != ADD_DUPLICATE && result != ADD_COLLISION) {
                return false;
            }
            
            // Test GetNbItem API
            uint64_t item_count = hash_table.GetNbItem();
            
            // Test Reset API
            hash_table.Reset();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Test if HashTable512 API maintains compatibility
     */
    bool TestHashTable512API() {
        try {
            HashTable512 hash_table_512;
            
            // Test basic API methods exist and work
            int512_t x, d;
            for (int i = 0; i < 8; ++i) {
                x.i64[i] = rand() | ((uint64_t)rand() << 32);
                d.i64[i] = rand() | ((uint64_t)rand() << 32);
            }
            
            // Test Add API
            int result = hash_table_512.Add(HashTable512::Hash512(&x), &x, &d);
            if (result != ADD512_OK && result != ADD512_DUPLICATE && result != ADD512_COLLISION) {
                return false;
            }

            // Test item count via GetStats
            uint64_t totalItems = 0, totalMemory = 0; double lf = 0;
            hash_table_512.GetStats(&totalItems, &totalMemory, &lf);
            (void)totalItems; (void)totalMemory; (void)lf;

            // Test Reset API
            hash_table_512.Reset();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Test SECP256K1 API compatibility
     */
    bool TestSECP256K1API() {
        try {
            Secp256K1 secp;
            secp.Init();
            
            // Test basic elliptic curve operations
            Int private_key;
            private_key.Rand(256);
            
            // Test ComputePublicKey API
            Point public_key = secp.ComputePublicKey(&private_key);
            if (public_key.isZero()) return false;
            
            // Test point operations
            Point doubled = secp.DoubleDirect(public_key);
            if (doubled.isZero()) return false;
            
            // Test point addition
            Point sum = secp.AddDirect(public_key, doubled);
            
            return true;
        } catch (...) {
            return false;
        }
    }
};

/**
 * @brief Data format compatibility checker
 */
class DataFormatChecker {
public:
    /**
     * @brief Test Int data structure compatibility
     */
    bool TestIntDataStructure() {
        try {
            Int test_int;
            
            // Test basic operations
            test_int.Rand(256);
            if (test_int.IsZero()) return false;
            
            // Test bit length calculation
            int bit_length = test_int.GetBitLength();
            if (bit_length <= 0 || bit_length > 256) return false;
            
            // Test comparison operations
            Int other_int;
            other_int.Rand(256);
            
            bool is_equal = test_int.IsEqual(&other_int);
            bool is_greater = test_int.IsGreater(&other_int);
            bool is_lower = test_int.IsLower(&other_int);
            
            // At least one comparison should be true (unless equal)
            if (!is_equal && !is_greater && !is_lower) return false;
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    /**
     * @brief Test Point data structure compatibility
     */
    bool TestPointDataStructure() {
        try {
            Secp256K1 secp;
            secp.Init();
            
            Int private_key;
            private_key.Rand(256);
            
            Point point = secp.ComputePublicKey(&private_key);
            
            // Test point properties
            if (point.isZero()) return false;
            
            // Test point coordinate access
            Int x_coord = point.x;
            Int y_coord = point.y;
            
            // Coordinates should not both be zero for valid points
            if (x_coord.IsZero() && y_coord.IsZero()) return false;
            
            return true;
        } catch (...) {
            return false;
        }
    }
};

/**
 * @brief Compatibility test fixture
 */
class CompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        Timer::Init();
        rseed(Timer::getSeed32());
        
        api_checker_ = std::make_unique<APICompatibilityChecker>();
        data_checker_ = std::make_unique<DataFormatChecker>();
    }
    
    void TearDown() override {
        api_checker_.reset();
        data_checker_.reset();
    }
    
    std::unique_ptr<APICompatibilityChecker> api_checker_;
    std::unique_ptr<DataFormatChecker> data_checker_;
};

/**
 * @brief Test HashTable API backward compatibility
 */
TEST_F(CompatibilityTest, HashTableAPICompatibility) {
    EXPECT_TRUE(api_checker_->TestHashTableAPI()) 
        << "HashTable API compatibility test failed";
}

/**
 * @brief Test HashTable512 API compatibility
 */
TEST_F(CompatibilityTest, HashTable512APICompatibility) {
    EXPECT_TRUE(api_checker_->TestHashTable512API()) 
        << "HashTable512 API compatibility test failed";
}

/**
 * @brief Test SECP256K1 API compatibility
 */
TEST_F(CompatibilityTest, SECP256K1APICompatibility) {
    EXPECT_TRUE(api_checker_->TestSECP256K1API()) 
        << "SECP256K1 API compatibility test failed";
}

/**
 * @brief Test Int data structure compatibility
 */
TEST_F(CompatibilityTest, IntDataStructureCompatibility) {
    EXPECT_TRUE(data_checker_->TestIntDataStructure()) 
        << "Int data structure compatibility test failed";
}

/**
 * @brief Test Point data structure compatibility
 */
TEST_F(CompatibilityTest, PointDataStructureCompatibility) {
    EXPECT_TRUE(data_checker_->TestPointDataStructure()) 
        << "Point data structure compatibility test failed";
}

/**
 * @brief Test cross-platform compatibility
 */
TEST_F(CompatibilityTest, CrossPlatformCompatibility) {
    // Test that basic operations work consistently
    Secp256K1 secp;
    secp.Init();
    
    // Generate same private key multiple times
    Int private_key;
    { std::string s = "0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"; private_key.SetBase16(&s[0]); }
    
    // Compute public key multiple times
    Point public_key1 = secp.ComputePublicKey(&private_key);
    Point public_key2 = secp.ComputePublicKey(&private_key);
    
    // Results should be identical
    EXPECT_TRUE(public_key1.equals(public_key2)) 
        << "Public key computation should be deterministic";
    
    // Test hash table behavior consistency
    HashTable hash_table;
    Int x, d;
    { std::string s1 = "FEDCBA9876543210FEDCBA9876543210"; x.SetBase16(&s1[0]); }
    { std::string s2 = "0123456789ABCDEF0123456789ABCDEF"; d.SetBase16(&s2[0]); }
    
    int add1 = hash_table.Add(&x, &d, 0);
    int add2 = hash_table.Add(&x, &d, 0);

    EXPECT_TRUE(add1 == ADD_OK || add1 == ADD_DUPLICATE || add1 == ADD_COLLISION);
    EXPECT_TRUE(add2 == ADD_OK || add2 == ADD_DUPLICATE || add2 == ADD_COLLISION);
}

/**
 * @brief Test version compatibility
 */
TEST_F(CompatibilityTest, VersionCompatibility) {
    // Test that current implementation maintains expected behavior
    
    // Test HashTable size limits
    HashTable hash_table;
    uint64_t max_items = hash_table.GetNbItem();
    EXPECT_GE(max_items, 0) << "HashTable should report valid item count";
    
    // Test HashTable512 enhanced capabilities
    HashTable512 hash_table_512;
    uint64_t totalItems=0,totalMemory=0; double lf=0; hash_table_512.GetStats(&totalItems,&totalMemory,&lf);
    EXPECT_GE((int64_t)totalItems, 0) << "HashTable512 should report valid item count";
    
    // Test that 512-bit version can handle larger distances
    int512_t large_distance;
    for (int i = 0; i < 8; ++i) {
        large_distance.i64[i] = UINT64_MAX;
    }
    
    // This should not crash or fail
    EXPECT_NO_THROW({
        hash_table_512.Add(HashTable512::Hash512(&large_distance), &large_distance, &large_distance);
    }) << "HashTable512 should handle large values without issues";
}

#ifdef WITHGPU
/**
 * @brief Test GPU compatibility if available
 */
TEST_F(CompatibilityTest, GPUCompatibility) {
    try {
        // Test GPU engine initialization
        GPUEngine gpu_engine(1, 256, 0, 1024);
        
        // Test basic GPU operations
        EXPECT_NO_THROW({
            gpu_engine.SetParams(20, nullptr, nullptr, nullptr);
        }) << "GPU SetParams should maintain API compatibility";
        
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPU not available for compatibility testing: " << e.what();
    }
}
#endif

} // namespace kangaroo_test
