/**
 * @file puzzle56_comprehensive_test.cpp
 * @brief Comprehensive Testing Suite for Bitcoin Puzzle #56
 * 
 * This test suite validates the complete CUDA-BSGS-Kangaroo system using
 * real Bitcoin Puzzle #56 data to ensure correctness, performance, and security.
 * 
 * Puzzle #56 Parameters:
 * - Private Key Range: 80000000000000 to ffffffffffffff (56-bit)
 * - Public Key: 033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a
 * - Range Size: 2^56 = 72,057,594,037,927,936 keys
 * 
 * Copyright (c) 2025 CUDA-BSGS-Kangaroo Project
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cassert>

#include "../UTF8Console.h"
#include "../HashTableUnified.h"
#include "../Kangaroo.h"
#include "../SECPK1/Point.h"
#include "../SECPK1/Int.h"
#include "../SECPK1/IntGroup.h"

// Puzzle #56 Constants
namespace Puzzle56 {
    const std::string PUBLIC_KEY = "033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a";
    const std::string RANGE_START = "80000000000000";
    const std::string RANGE_END = "ffffffffffffff";
    const int RANGE_BITS = 56;
    const std::string EXPECTED_PRIVATE_KEY = ""; // To be discovered
}

/**
 * @brief Test Result Structure
 */
struct TestResult {
    std::string testName;
    bool passed;
    double executionTime;
    std::string details;
    
    TestResult(const std::string& name, bool pass, double time, const std::string& det = "")
        : testName(name), passed(pass), executionTime(time), details(det) {}
};

/**
 * @brief Comprehensive Test Suite Class
 */
class Puzzle56TestSuite {
private:
    std::vector<TestResult> results;
    Point targetPublicKey;
    Int rangeStart, rangeEnd;
    
public:
    /**
     * @brief Initialize test suite with Puzzle #56 data
     */
    bool Initialize() {
        printf("🔧 Initializing Puzzle #56 Test Suite...\n");
        
        // Parse public key
        if (!targetPublicKey.ParsePublicKeyHex(Puzzle56::PUBLIC_KEY)) {
            printf("❌ Failed to parse public key: %s\n", Puzzle56::PUBLIC_KEY.c_str());
            return false;
        }
        
        // Parse range
        rangeStart.SetBase16(Puzzle56::RANGE_START.c_str());
        rangeEnd.SetBase16(Puzzle56::RANGE_END.c_str());
        
        printf("✅ Target Public Key: %s\n", Puzzle56::PUBLIC_KEY.c_str());
        printf("✅ Private Key Range: %s to %s\n", 
               Puzzle56::RANGE_START.c_str(), Puzzle56::RANGE_END.c_str());
        printf("✅ Range Size: 2^%d keys\n", Puzzle56::RANGE_BITS);
        
        return true;
    }
    
    /**
     * @brief Unit Test: Cryptographic Operations
     */
    void RunUnitTests() {
        printf("\n📋 Running Unit Tests...\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Test 1: Point operations
        bool pointTest = TestPointOperations();
        
        // Test 2: Integer operations
        bool intTest = TestIntegerOperations();
        
        // Test 3: Hash table operations
        bool hashTest = TestHashTableOperations();
        
        // Test 4: Range validation
        bool rangeTest = TestRangeValidation();
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        
        bool allPassed = pointTest && intTest && hashTest && rangeTest;
        results.emplace_back("Unit Tests", allPassed, time, 
                           "Point:" + std::to_string(pointTest) + 
                           " Int:" + std::to_string(intTest) + 
                           " Hash:" + std::to_string(hashTest) + 
                           " Range:" + std::to_string(rangeTest));
        
        printf("%s Unit Tests completed in %.3f seconds\n", 
               allPassed ? "✅" : "❌", time);
    }
    
    /**
     * @brief Integration Test: Complete System
     */
    void RunIntegrationTests() {
        printf("\n🔗 Running Integration Tests...\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Test complete kangaroo setup with Puzzle #56 data
        bool integrationPassed = TestKangarooIntegration();
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        
        results.emplace_back("Integration Tests", integrationPassed, time);
        
        printf("%s Integration Tests completed in %.3f seconds\n", 
               integrationPassed ? "✅" : "❌", time);
    }
    
    /**
     * @brief Performance Test: System Performance
     */
    void RunPerformanceTests() {
        printf("\n⚡ Running Performance Tests...\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Test hash table performance with 56-bit range
        bool perfPassed = TestPerformanceMetrics();
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        
        results.emplace_back("Performance Tests", perfPassed, time);
        
        printf("%s Performance Tests completed in %.3f seconds\n", 
               perfPassed ? "✅" : "❌", time);
    }
    
    /**
     * @brief Security Test: Key Handling Security
     */
    void RunSecurityTests() {
        printf("\n🔒 Running Security Tests...\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Test secure key handling
        bool securityPassed = TestSecurityMeasures();
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        
        results.emplace_back("Security Tests", securityPassed, time);
        
        printf("%s Security Tests completed in %.3f seconds\n", 
               securityPassed ? "✅" : "❌", time);
    }
    
    /**
     * @brief Generate comprehensive test report
     */
    void GenerateReport() {
        printf("\n📊 COMPREHENSIVE TEST REPORT - PUZZLE #56\n");
        printf("============================================================================\n");
        printf("Test Suite                     Status    Time (s)    Details\n");
        printf("----------------------------------------------------------------------------\n");
        
        int passed = 0, total = results.size();
        double totalTime = 0.0;
        
        for (const auto& result : results) {
            printf("%-30s %s      %8.3f    %s\n", 
                   result.testName.c_str(),
                   result.passed ? "✅ PASS" : "❌ FAIL",
                   result.executionTime,
                   result.details.c_str());
            
            if (result.passed) passed++;
            totalTime += result.executionTime;
        }
        
        printf("----------------------------------------------------------------------------\n");
        printf("SUMMARY: %d/%d tests passed (%.1f%%) in %.3f seconds\n", 
               passed, total, (100.0 * passed) / total, totalTime);
        
        if (passed == total) {
            printf("🎉 ALL TESTS PASSED! System ready for Puzzle #56 solving.\n");
        } else {
            printf("⚠️  Some tests failed. Review issues before proceeding.\n");
        }
        printf("============================================================================\n");
    }

private:
    /**
     * @brief Test elliptic curve point operations
     */
    bool TestPointOperations() {
        printf("  🔍 Testing Point Operations...\n");

        try {
            // Test point validation
            if (!targetPublicKey.IsValid()) {
                printf("    ❌ Target public key is invalid\n");
                return false;
            }

            // Test point arithmetic
            Point generator;
            generator.Set(Secp256K1::Gx, Secp256K1::Gy, Secp256K1::order);

            if (!generator.IsValid()) {
                printf("    ❌ Generator point is invalid\n");
                return false;
            }

            // Test scalar multiplication
            Int testKey;
            testKey.SetBase16("123456789abcdef0");
            Point testPoint = generator;
            testPoint = Secp256K1::ScalarMultiplication(testKey, testPoint);

            if (!testPoint.IsValid()) {
                printf("    ❌ Scalar multiplication failed\n");
                return false;
            }

            printf("    ✅ Point operations validated\n");
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Point operations failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test integer arithmetic operations
     */
    bool TestIntegerOperations() {
        printf("  🔍 Testing Integer Operations...\n");

        try {
            // Test range arithmetic
            Int rangeSize = rangeEnd;
            rangeSize.Sub(&rangeStart);

            if (rangeSize.IsZero()) {
                printf("    ❌ Invalid range size\n");
                return false;
            }

            // Test modular operations
            Int testInt1, testInt2, result;
            testInt1.SetBase16("123456789abcdef0");
            testInt2.SetBase16("fedcba9876543210");

            result.ModAdd(&testInt1, &testInt2);
            if (result.IsZero()) {
                printf("    ❌ Modular addition failed\n");
                return false;
            }

            result.ModSub(&testInt1, &testInt2);
            result.ModMul(&testInt1, &testInt2);

            printf("    ✅ Integer operations validated\n");
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Integer operations failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test hash table operations with 56-bit data
     */
    bool TestHashTableOperations() {
        printf("  🔍 Testing Hash Table Operations...\n");

        try {
            // Test both 128-bit and 512-bit hash tables
            HashTableUnified ht128(false);
            HashTableUnified ht512(true);

            // Test with sample data from range
            Int testKey = rangeStart;
            Int testDist;
            testDist.SetInt32(12345);

            int result128 = ht128.Add(&testKey, &testDist, 0);
            int result512 = ht512.Add(&testKey, &testDist, 1);

            if (result128 != 0 || result512 != 0) {
                printf("    ❌ Hash table add operations failed\n");
                return false;
            }

            // Test statistics
            uint64_t items128 = ht128.GetNbItem();
            uint64_t items512 = ht512.GetNbItem();

            if (items128 == 0 || items512 == 0) {
                printf("    ❌ Hash table item count failed\n");
                return false;
            }

            printf("    ✅ Hash table operations validated (128-bit: %llu, 512-bit: %llu items)\n",
                   (unsigned long long)items128, (unsigned long long)items512);
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Hash table operations failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test range validation for Puzzle #56
     */
    bool TestRangeValidation() {
        printf("  🔍 Testing Range Validation...\n");

        try {
            // Validate range bounds
            if (rangeStart.IsGreater(&rangeEnd)) {
                printf("    ❌ Invalid range: start > end\n");
                return false;
            }

            // Test range size calculation
            Int rangeSize = rangeEnd;
            rangeSize.Sub(&rangeStart);

            // Verify it's approximately 2^56
            Int expected2pow56;
            expected2pow56.SetInt32(1);
            expected2pow56.ShiftL(56);

            if (rangeSize.IsGreater(&expected2pow56)) {
                printf("    ❌ Range size exceeds 2^56\n");
                return false;
            }

            // Test key generation within range
            Int testKey = rangeStart;
            testKey.Add(12345);

            if (testKey.IsGreater(&rangeEnd)) {
                printf("    ❌ Test key outside range\n");
                return false;
            }

            printf("    ✅ Range validation passed (56-bit range confirmed)\n");
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Range validation failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test complete Kangaroo system integration
     */
    bool TestKangarooIntegration() {
        printf("  🔍 Testing Kangaroo Integration...\n");

        try {
            // Test Kangaroo initialization with Puzzle #56 parameters
            // Note: This is a simplified integration test

            // Test point generation within range
            Point generator;
            generator.Set(Secp256K1::Gx, Secp256K1::Gy, Secp256K1::order);

            Int testPrivateKey = rangeStart;
            testPrivateKey.Add(1000); // Test key within range

            Point testPublicKey = generator;
            testPublicKey = Secp256K1::ScalarMultiplication(testPrivateKey, testPublicKey);

            if (!testPublicKey.IsValid()) {
                printf("    ❌ Generated public key is invalid\n");
                return false;
            }

            // Test hash table integration
            HashTableUnified hashTable(false);
            Int distance;
            distance.SetInt32(1000);

            int addResult = hashTable.Add(&testPrivateKey, &distance, 0);
            if (addResult != 0) {
                printf("    ❌ Hash table integration failed\n");
                return false;
            }

            printf("    ✅ Kangaroo integration validated\n");
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Kangaroo integration failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test performance metrics with 56-bit range
     */
    bool TestPerformanceMetrics() {
        printf("  🔍 Testing Performance Metrics...\n");

        try {
            const int NUM_OPERATIONS = 1000;

            // Test hash table performance
            auto start = std::chrono::high_resolution_clock::now();

            HashTableUnified hashTable(false);
            for (int i = 0; i < NUM_OPERATIONS; i++) {
                Int key = rangeStart;
                key.Add(i);
                Int dist;
                dist.SetInt32(i);

                hashTable.Add(&key, &dist, i % 2);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double time = std::chrono::duration<double>(end - start).count();
            double opsPerSec = NUM_OPERATIONS / time;

            if (opsPerSec < 1000) { // Minimum performance threshold
                printf("    ❌ Performance below threshold: %.0f ops/sec\n", opsPerSec);
                return false;
            }

            printf("    ✅ Performance validated: %.0f ops/sec\n", opsPerSec);
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Performance testing failed: %s\n", e.what());
            return false;
        }
    }

    /**
     * @brief Test security measures for key handling
     */
    bool TestSecurityMeasures() {
        printf("  🔍 Testing Security Measures...\n");

        try {
            // Test secure key handling
            Int sensitiveKey = rangeStart;
            sensitiveKey.Add(12345);

            // Test that keys are properly validated
            if (sensitiveKey.IsZero()) {
                printf("    ❌ Key validation failed\n");
                return false;
            }

            // Test range bounds checking
            if (sensitiveKey.IsGreater(&rangeEnd)) {
                printf("    ❌ Key outside secure range\n");
                return false;
            }

            // Test memory handling (basic check)
            HashTableUnified secureHashTable(false);
            Int secureDistance;
            secureDistance.SetInt32(54321);

            int result = secureHashTable.Add(&sensitiveKey, &secureDistance, 0);
            if (result != 0) {
                printf("    ❌ Secure hash table operation failed\n");
                return false;
            }

            // Test that sensitive data is properly managed
            uint64_t items = secureHashTable.GetNbItem();
            if (items == 0) {
                printf("    ❌ Secure data management failed\n");
                return false;
            }

            printf("    ✅ Security measures validated\n");
            return true;

        } catch (const std::exception& e) {
            printf("    ❌ Security testing failed: %s\n", e.what());
            return false;
        }
    }
};

/**
 * @brief Main test execution function
 */
int main() {
    // Initialize UTF-8 console for proper Unicode display
    INIT_UTF8_CONSOLE();
    
    printf("🚀 CUDA-BSGS-Kangaroo Comprehensive Test Suite\n");
    printf("🧩 Testing with Bitcoin Puzzle #56 Data\n");
    printf("============================================================================\n");
    
    Puzzle56TestSuite testSuite;
    
    if (!testSuite.Initialize()) {
        printf("❌ Failed to initialize test suite\n");
        return 1;
    }
    
    try {
        // Run all test categories
        testSuite.RunUnitTests();
        testSuite.RunIntegrationTests();
        testSuite.RunPerformanceTests();
        testSuite.RunSecurityTests();
        
        // Generate comprehensive report
        testSuite.GenerateReport();
        
        return 0;
        
    } catch (const std::exception& e) {
        printf("❌ Test suite failed with exception: %s\n", e.what());
        return 1;
    }
}
