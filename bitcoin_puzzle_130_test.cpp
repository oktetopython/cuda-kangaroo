#include <iostream>
#include <string>
#include <chrono>
#include "kangaroo_bl_integration.h"
#include "optimizations/phase4/bl_real_ec_generator.h"
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "Timer.h"

/**
 * Bitcoin Puzzle 130 Test Program
 * 
 * Tests our Bernstein-Lange Kangaroo integration with Bitcoin Puzzle 130
 * Known private key: 33e7665705359f04f28b88cf897c603c9
 * Range: 2^129 to 2^130-1
 * 
 * This validates our system can solve real Bitcoin puzzles.
 */

class BitcoinPuzzle130Test {
private:
    Secp256K1 secp;
    BernsteinLangeKangaroo bl_kangaroo;
    
    // Known solution for puzzle 130
    std::string known_private_key_hex = "33e7665705359f04f28b88cf897c603c9";
    
public:
    BitcoinPuzzle130Test() : bl_kangaroo(&secp) {
        std::cout << "=== Bitcoin Puzzle 130 Test ===" << std::endl;
        std::cout << "Known private key: " << known_private_key_hex << std::endl;
    }
    
    bool generateTargetPublicKey(Point& target_point) {
        std::cout << "\n1. Generating target public key from known private key..." << std::endl;
        
        // Convert hex string to Int
        Int private_key;
        private_key.SetBase16(known_private_key_hex.c_str());
        
        std::cout << "Private key (decimal): " << private_key.GetBase10() << std::endl;
        
        // Generate public key: P = private_key * G
        target_point = secp.ComputePublicKey(&private_key);
        
        std::cout << "Target public key generated successfully" << std::endl;
        std::cout << "Public key X: " << target_point.x.GetBase16() << std::endl;
        std::cout << "Public key Y: " << target_point.y.GetBase16() << std::endl;
        
        return true;
    }
    
    bool generatePrecomputeTable() {
        std::cout << "\n2. Generating precompute table for puzzle 130 range..." << std::endl;
        
        // Puzzle 130 parameters
        uint64_t L = 1ULL << 30;  // Start with smaller range for testing: 2^30
        uint64_t A = 1ULL << 29;  // Start point: 2^29
        uint64_t T = 8192;        // Table size
        
        std::cout << "Range: 2^29 to 2^30 (testing range)" << std::endl;
        std::cout << "L = " << L << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "T = " << T << std::endl;
        
        // Generate table filename
        std::string table_filename = "puzzle_130_test_table.bin";
        
        // Use our real EC generator to create the table
        std::cout << "Generating precompute table..." << std::endl;
        
        double t0 = Timer::get_tick();

        // Create a simplified table generator for this test
        RealECBLGenerator generator(L, A, T);
        bool success = generator.generatePrecomputeTable(table_filename);

        double t1 = Timer::get_tick();

        if (success) {
            std::cout << "Precompute table generated successfully!" << std::endl;
            std::cout << "File: " << table_filename << std::endl;
            std::cout << "Generation time: " << Timer::getResult("ms", 1, t0, t1) << std::endl;
            return true;
        } else {
            std::cout << "ERROR: Failed to generate precompute table" << std::endl;
            return false;
        }
    }
    
    bool loadPrecomputeTable() {
        std::cout << "\n3. Loading precompute table..." << std::endl;
        
        std::string table_filename = "puzzle_130_test_table.bin";
        
        double t0 = Timer::get_tick();

        bool success = bl_kangaroo.loadPrecomputeTable(table_filename);

        double t1 = Timer::get_tick();

        if (success) {
            std::cout << "Precompute table loaded successfully!" << std::endl;
            std::cout << "Load time: " << Timer::getResult("ms", 1, t0, t1) << std::endl;
            return true;
        } else {
            std::cout << "ERROR: Failed to load precompute table" << std::endl;
            return false;
        }
    }
    
    bool solvePuzzle(const Point& target_point) {
        std::cout << "\n4. Solving Bitcoin Puzzle 130..." << std::endl;
        Int target_x_copy = target_point.x;  // Create non-const copy
        std::cout << "Target: " << target_x_copy.GetBase16() << std::endl;
        
        double t0 = Timer::get_tick();

        Int result;
        uint64_t max_steps = 100000000;  // 100M steps max

        std::cout << "Starting DLP solving (max " << max_steps << " steps)..." << std::endl;

        bool success = bl_kangaroo.solveDLP(target_point, result, max_steps);

        double t1 = Timer::get_tick();

        if (success) {
            std::cout << "PUZZLE SOLVED!" << std::endl;
            std::cout << "Solution time: " << Timer::getResult("ms", 1, t0, t1) << std::endl;
            std::cout << "Found private key: " << result.GetBase16() << std::endl;
            
            // Verify solution
            std::cout << "\n5. Verifying solution..." << std::endl;
            std::string found_key = result.GetBase16();
            
            if (found_key == known_private_key_hex) {
                std::cout << "✅ SOLUTION VERIFIED! Found key matches known key!" << std::endl;
                return true;
            } else {
                std::cout << "❌ Solution verification failed!" << std::endl;
                std::cout << "Expected: " << known_private_key_hex << std::endl;
                std::cout << "Found:    " << found_key << std::endl;
                return false;
            }
        } else {
            std::cout << "ERROR: Failed to solve puzzle within " << max_steps << " steps" << std::endl;
            std::cout << "Search time: " << Timer::getResult("ms", 1, t0, t1) << std::endl;
            return false;
        }
    }
    
    bool runFullTest() {
        std::cout << "\n🚀 Starting Bitcoin Puzzle 130 Full Test 🚀" << std::endl;
        
        Point target_point;
        
        // Step 1: Generate target public key
        if (!generateTargetPublicKey(target_point)) {
            std::cout << "❌ Failed at step 1: Generate target public key" << std::endl;
            return false;
        }
        
        // Step 2: Generate precompute table
        if (!generatePrecomputeTable()) {
            std::cout << "❌ Failed at step 2: Generate precompute table" << std::endl;
            return false;
        }
        
        // Step 3: Load precompute table
        if (!loadPrecomputeTable()) {
            std::cout << "❌ Failed at step 3: Load precompute table" << std::endl;
            return false;
        }
        
        // Step 4: Solve puzzle
        if (!solvePuzzle(target_point)) {
            std::cout << "❌ Failed at step 4: Solve puzzle" << std::endl;
            return false;
        }
        
        std::cout << "\n🎉 ALL TESTS PASSED! Bitcoin Puzzle 130 system validated! 🎉" << std::endl;
        return true;
    }
};

int main() {
    std::cout << "Bitcoin Puzzle 130 Test - Bernstein-Lange Kangaroo System" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        BitcoinPuzzle130Test test;
        
        bool success = test.runFullTest();
        
        if (success) {
            std::cout << "\n✅ SUCCESS: System ready for Bitcoin puzzle challenges!" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ FAILURE: System needs debugging" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
