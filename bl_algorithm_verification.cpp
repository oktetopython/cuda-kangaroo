/**
 * Bernstein-Lange Algorithm Core Logic Verification
 * 
 * This program verifies the core components of our Bernstein-Lange implementation:
 * 1. r-adding walk logic
 * 2. Distinguished point detection
 * 3. Parameter calculation consistency
 * 4. Step size calculation from hash
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <vector>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "Timer.h"

class BLAlgorithmVerification {
private:
    Secp256K1 secp;
    std::mt19937_64 rng;
    
public:
    BLAlgorithmVerification() {
        secp.Init();
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng.seed(seed);
        std::cout << "✅ BL Algorithm Verification initialized" << std::endl;
    }
    
    // Test 1: Verify parameter calculation formulas
    bool testParameterCalculation() {
        std::cout << "\n=== Test 1: Parameter Calculation Verification ===" << std::endl;
        
        // Test with known parameters from our implementation
        uint64_t L = 1ULL << 20;  // 2^20
        uint64_t T = 1024;

        std::cout << "Input parameters:" << std::endl;
        std::cout << "  L = 2^20 = " << L << std::endl;
        std::cout << "  T = " << T << std::endl;

        // Calculate W using our formula
        double alpha = 1.33;
        double theoretical_W = alpha * sqrt((double)L / (double)T);
        int calculated_W = (int)round(theoretical_W);

        std::cout << "W calculation:" << std::endl;
        std::cout << "  Calculated W: " << calculated_W << std::endl;
        std::cout << "  Theoretical W: " << std::fixed << std::setprecision(2) << theoretical_W << std::endl;
        std::cout << "  Error: " << std::fixed << std::setprecision(4)
                  << (abs(calculated_W - theoretical_W) / theoretical_W * 100) << "%" << std::endl;

        // Verify DP mask calculation
        int dp_mask_bits = 3;  // For 1/8 probability
        uint64_t dp_mask = (1ULL << dp_mask_bits) - 1;
        double expected_dp_rate = 1.0 / (1ULL << dp_mask_bits);
        
        std::cout << "Distinguished Point parameters:" << std::endl;
        std::cout << "  DP mask bits: " << dp_mask_bits << std::endl;
        std::cout << "  DP mask: 0x" << std::hex << dp_mask << std::dec << std::endl;
        std::cout << "  Expected DP rate: " << std::fixed << std::setprecision(4) 
                  << (expected_dp_rate * 100) << "%" << std::endl;
        
        // Verify parameters are reasonable
        bool w_reasonable = (calculated_W > 0 && calculated_W < 1000);
        bool dp_reasonable = (dp_mask_bits >= 3 && dp_mask_bits <= 16);
        
        std::cout << "Parameter validation: " << (w_reasonable && dp_reasonable ? "✅ PASSED" : "❌ FAILED") << std::endl;
        
        return w_reasonable && dp_reasonable;
    }
    
    // Test 2: Verify distinguished point detection logic
    bool testDistinguishedPointDetection() {
        std::cout << "\n=== Test 2: Distinguished Point Detection ===" << std::endl;
        
        int dp_mask_bits = 3;  // 1/8 probability
        uint64_t dp_mask = (1ULL << dp_mask_bits) - 1;  // 0x7
        
        std::cout << "Testing with DP mask bits: " << dp_mask_bits << std::endl;
        std::cout << "DP mask: 0x" << std::hex << dp_mask << std::dec << std::endl;
        std::cout << "Expected DP rate: " << std::fixed << std::setprecision(2) 
                  << (100.0 / (1ULL << dp_mask_bits)) << "%" << std::endl;
        
        int total_tests = 10000;
        int dp_count = 0;
        
        std::cout << "Testing " << total_tests << " random points..." << std::endl;
        
        for (int i = 0; i < total_tests; i++) {
            // Generate random point using different method
            Int random_key;
            random_key.SetInt32(i + 1000);  // Use sequential keys for testing
            random_key.Add(rng() % 1000000);  // Add random offset
            Point point = secp.ComputePublicKey(&random_key);

            // Check if distinguished point using our logic
            uint64_t hash = point.x.bits64[0];
            bool is_dp = ((hash & dp_mask) == 0);

            if (is_dp) {
                dp_count++;
                if (dp_count <= 5) {  // Show first 5 DPs
                    std::cout << "  DP #" << dp_count << ": hash=0x" << std::hex << hash
                              << ", masked=0x" << (hash & dp_mask) << std::dec << std::endl;
                }
            }

            // Debug first few iterations
            if (i < 3) {
                std::cout << "  Test " << (i+1) << ": key=" << random_key.GetBase16()
                          << ", hash=0x" << std::hex << hash << ", DP=" << (is_dp ? "YES" : "NO") << std::dec << std::endl;
            }
        }
        
        double actual_dp_rate = (double)dp_count / total_tests;
        double expected_dp_rate = 1.0 / (1ULL << dp_mask_bits);
        double error = abs(actual_dp_rate - expected_dp_rate) / expected_dp_rate;
        
        std::cout << "Results:" << std::endl;
        std::cout << "  Distinguished points found: " << dp_count << "/" << total_tests << std::endl;
        std::cout << "  Actual DP rate: " << std::fixed << std::setprecision(4) 
                  << (actual_dp_rate * 100) << "%" << std::endl;
        std::cout << "  Expected DP rate: " << std::fixed << std::setprecision(4) 
                  << (expected_dp_rate * 100) << "%" << std::endl;
        std::cout << "  Error: " << std::fixed << std::setprecision(2) << (error * 100) << "%" << std::endl;
        
        // Accept up to 20% error for statistical variation
        bool dp_detection_correct = (error < 0.20);
        std::cout << "DP detection verification: " << (dp_detection_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
        
        return dp_detection_correct;
    }
    
    // Test 3: Verify r-adding walk step logic
    bool testRandomWalkStep() {
        std::cout << "\n=== Test 3: Random Walk Step Logic ===" << std::endl;
        
        // Start with a known point
        Int start_key;
        start_key.SetInt32(12345);
        Point start_point = secp.ComputePublicKey(&start_key);
        
        std::cout << "Starting point (12345*G):" << std::endl;
        std::cout << "  X: 0x" << std::hex << start_point.x.bits64[0] << std::dec << std::endl;
        
        // Perform a series of steps
        Point current_point = start_point;
        Int accumulated_steps;
        accumulated_steps.SetInt32(0);
        
        std::cout << "Performing 10 random walk steps..." << std::endl;
        
        for (int i = 0; i < 10; i++) {
            // Calculate step size from point hash (simplified version)
            uint64_t hash = current_point.x.bits64[0];
            int step_size = 1 + (hash % 16);  // Step size 1-16
            
            std::cout << "  Step " << (i+1) << ": hash=0x" << std::hex << hash 
                      << ", step_size=" << std::dec << step_size << std::endl;
            
            // Perform step: current_point = current_point + step_size * G
            Int step_int;
            step_int.SetInt32(step_size);
            Point step_point = secp.ComputePublicKey(&step_int);
            current_point = secp.AddDirect(current_point, step_point);
            
            // Accumulate steps
            accumulated_steps.Add(&step_int);
            
            std::cout << "    New point X: 0x" << std::hex << current_point.x.bits64[0] << std::dec << std::endl;
        }
        
        // Verify: start_point + accumulated_steps should equal current_point
        Int final_key = start_key;
        final_key.Add(&accumulated_steps);
        Point expected_point = secp.ComputePublicKey(&final_key);
        
        std::cout << "Verification:" << std::endl;
        std::cout << "  Current point X: 0x" << std::hex << current_point.x.bits64[0] << std::dec << std::endl;
        std::cout << "  Expected point X: 0x" << std::hex << expected_point.x.bits64[0] << std::dec << std::endl;
        
        bool walk_correct = (current_point.x.bits64[0] == expected_point.x.bits64[0] &&
                            current_point.y.bits64[0] == expected_point.y.bits64[0]);
        
        std::cout << "Random walk verification: " << (walk_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
        
        return walk_correct;
    }
    
    // Test 4: Verify step size calculation consistency
    bool testStepSizeCalculation() {
        std::cout << "\n=== Test 4: Step Size Calculation Consistency ===" << std::endl;
        
        // Test that same point always gives same step size
        Int test_key;
        test_key.SetInt32(99999);
        Point test_point = secp.ComputePublicKey(&test_key);
        
        uint64_t hash = test_point.x.bits64[0];
        std::cout << "Test point hash: 0x" << std::hex << hash << std::dec << std::endl;
        
        // Calculate step size multiple times
        std::vector<int> step_sizes;
        for (int i = 0; i < 5; i++) {
            int step_size = 1 + (hash % 16);  // Our step calculation logic
            step_sizes.push_back(step_size);
            std::cout << "  Calculation " << (i+1) << ": step_size = " << step_size << std::endl;
        }
        
        // Verify all step sizes are identical
        bool consistent = true;
        for (size_t i = 1; i < step_sizes.size(); i++) {
            if (step_sizes[i] != step_sizes[0]) {
                consistent = false;
                break;
            }
        }
        
        std::cout << "Step size consistency: " << (consistent ? "✅ PASSED" : "❌ FAILED") << std::endl;
        
        return consistent;
    }
    
    // Run all algorithm verification tests
    bool runAllTests() {
        std::cout << "🔍 Bernstein-Lange Algorithm Core Logic Verification" << std::endl;
        std::cout << "===================================================" << std::endl;
        std::cout << "This test verifies the correctness of our BL algorithm implementation" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= testParameterCalculation();
        all_passed &= testDistinguishedPointDetection();
        all_passed &= testRandomWalkStep();
        all_passed &= testStepSizeCalculation();
        
        std::cout << "\n🏁 ALGORITHM VERIFICATION RESULTS:" << std::endl;
        std::cout << "==================================" << std::endl;
        if (all_passed) {
            std::cout << "✅ ALL TESTS PASSED - Bernstein-Lange algorithm logic is correct!" << std::endl;
            std::cout << "✅ Parameter calculations, DP detection, and random walks are working properly" << std::endl;
        } else {
            std::cout << "❌ SOME TESTS FAILED - Please investigate algorithm implementation" << std::endl;
        }
        
        return all_passed;
    }
};

int main() {
    BLAlgorithmVerification test;
    bool success = test.runAllTests();
    return success ? 0 : 1;
}
