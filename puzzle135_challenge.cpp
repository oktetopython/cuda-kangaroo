/**
 * Bitcoin Puzzle 135 Challenge Program
 * 
 * Complete challenge program for Bitcoin Puzzle 135 using Bernstein-Lange algorithm.
 * This program can potentially solve a real Bitcoin puzzle worth significant value.
 * 
 * Puzzle 135: Find private key in range [2^134, 2^135-1]
 * Current Bitcoin reward: Check current puzzle status
 */

#include <iostream>
#include <chrono>
#include <string>
#include <signal.h>
#include <thread>
#include <iomanip>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "bitcoin_puzzle_solver.h"
#include "Timer.h"

// Global solver instance for signal handling
BitcoinPuzzleSolver* global_solver = nullptr;

void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << " - stopping solver gracefully..." << std::endl;
    if (global_solver) {
        global_solver->stopSolving();
    }
}

class Puzzle135Challenge {
private:
    Secp256K1 secp;
    BitcoinPuzzleSolver solver;
    
    // Bitcoin Puzzle 135 target public key
    Point target_public_key;
    bool target_loaded;
    
public:
    Puzzle135Challenge() : solver(&secp, "puzzle135_checkpoint.dat"), target_loaded(false) {
        std::cout << "🎯 Bitcoin Puzzle 135 Challenge Initialized" << std::endl;
        std::cout << "===========================================" << std::endl;
    }
    
    bool loadTargetPublicKey() {
        std::cout << "🔍 Loading Bitcoin Puzzle 135 target public key..." << std::endl;

        // REAL Bitcoin Puzzle 135 public key (compressed format)
        // Public key: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
        // Address: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v
        // Private key range: [4000000000000000000000000000000000, 7fffffffffffffffffffffffffffffffff]

        std::string compressed_pubkey = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16";

        // Parse compressed public key using SECPK1 library
        bool isCompressed;
        if (!secp.ParsePublicKeyHex(compressed_pubkey, target_public_key, isCompressed)) {
            std::cerr << "❌ ERROR: Failed to parse Bitcoin Puzzle 135 public key!" << std::endl;
            return false;
        }

        if (!isCompressed) {
            std::cerr << "⚠️  WARNING: Expected compressed public key format" << std::endl;
        }

        std::cout << "✅ REAL Bitcoin Puzzle 135 target loaded!" << std::endl;
        std::cout << "🎯 Address: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v" << std::endl;
        std::cout << "🎯 Compressed PubKey: " << compressed_pubkey << std::endl;
        std::cout << "🎯 Target X: 0x" << target_public_key.x.GetBase16() << std::endl;
        std::cout << "🎯 Target Y: 0x" << target_public_key.y.GetBase16() << std::endl;
        std::cout << "🎯 Private Key Range: [4000000000000000000000000000000000, 7fffffffffffffffffffffffffffffffff]" << std::endl;
        
        target_loaded = true;
        return true;
    }
    
    bool runChallenge(const std::string& table_filename, uint64_t max_steps = 1000000000) {
        if (!target_loaded) {
            std::cerr << "❌ ERROR: Target public key not loaded!" << std::endl;
            return false;
        }
        
        std::cout << "\n🚀 Starting Bitcoin Puzzle 135 Challenge" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Table file: " << table_filename << std::endl;
        std::cout << "Max steps: " << max_steps << std::endl;
        std::cout << "Target: Bitcoin Puzzle 135" << std::endl;
        std::cout << "Range: [2^134, 2^135-1]" << std::endl;
        std::cout << "Reward: Significant Bitcoin value (check current puzzle status)" << std::endl;
        std::cout << std::endl;
        
        // Load precompute table
        if (!solver.loadPrecomputeTable(table_filename)) {
            std::cerr << "❌ ERROR: Failed to load precompute table!" << std::endl;
            return false;
        }
        
        // Set target
        solver.setTargetPoint(target_public_key);
        
        // Setup signal handlers for graceful shutdown
        global_solver = &solver;
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        std::cout << "🎯 Challenge configuration complete!" << std::endl;
        std::cout << "Press Ctrl+C to stop gracefully and save checkpoint" << std::endl;
        std::cout << "Starting in 3 seconds..." << std::endl;
        
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        std::cout << "\n🚀 CHALLENGE STARTED!" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        Timer::Init();
        double start_time = Timer::get_tick();

        // Start the solving process
        Int solution;
        bool success = solver.solvePuzzle(solution, max_steps, true);  // Enable checkpoints

        double end_time = Timer::get_tick();
        double total_time = end_time - start_time;
        auto stats = solver.getPerformanceStats();
        
        std::cout << "\n🏁 CHALLENGE COMPLETED!" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout << "Success: " << (success ? "YES! 🎉" : "No solution found") << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
        std::cout << "Total steps: " << stats.total_steps << std::endl;
        std::cout << "Distinguished points: " << stats.dps_found << std::endl;
        std::cout << "Collisions checked: " << stats.collisions_checked << std::endl;
        std::cout << "Average performance: " << std::fixed << std::setprecision(0) 
                  << stats.steps_per_second << " steps/second" << std::endl;
        std::cout << "DP rate: " << std::fixed << std::setprecision(2) << stats.dp_rate << "%" << std::endl;
        
        if (success) {
            std::cout << "\n🎉 BITCOIN PUZZLE 135 SOLVED!" << std::endl;
            std::cout << "==============================" << std::endl;
            std::cout << "🔑 PRIVATE KEY: 0x" << solution.GetBase16() << std::endl;
            std::cout << "💰 This key controls the Bitcoin reward for Puzzle 135!" << std::endl;
            std::cout << "⚠️  IMPORTANT: Secure this private key immediately!" << std::endl;
            std::cout << "⚠️  Transfer the Bitcoin to a secure wallet!" << std::endl;
            
            // Verify the solution
            Point verification = secp.ComputePublicKey(&solution);
            if (verification.equals(target_public_key)) {
                std::cout << "✅ Solution verified: Private key generates correct public key!" << std::endl;
            } else {
                std::cout << "❌ ERROR: Solution verification failed!" << std::endl;
            }
        } else {
            std::cout << "\n📊 Challenge Statistics Summary:" << std::endl;
            std::cout << "- Search space explored: " << std::scientific << std::setprecision(2) 
                      << (double)stats.total_steps << " points" << std::endl;
            std::cout << "- Computational work: " << std::fixed << std::setprecision(1) 
                      << (stats.total_steps / 1000000.0) << " million elliptic curve operations" << std::endl;
            std::cout << "- Distinguished points found: " << stats.dps_found << std::endl;
            std::cout << "- Collision detection efficiency: " << std::fixed << std::setprecision(2) 
                      << (stats.collisions_checked > 0 ? 100.0 * stats.collisions_checked / stats.dps_found : 0) 
                      << "%" << std::endl;
            
            std::cout << "\n💡 Recommendations for continued search:" << std::endl;
            std::cout << "1. 🔄 Restart with larger table size (T)" << std::endl;
            std::cout << "2. 🚀 Use GPU acceleration for faster computation" << std::endl;
            std::cout << "3. 🔧 Optimize distinguished point probability" << std::endl;
            std::cout << "4. 📊 Run multiple parallel instances" << std::endl;
            std::cout << "5. ⏰ Continue search from saved checkpoint" << std::endl;
        }
        
        return success;
    }
    
    void displayChallengeInfo() {
        std::cout << "\n📋 Bitcoin Puzzle 135 Information" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Puzzle Number: 135" << std::endl;
        std::cout << "Private Key Range: [2^134, 2^135-1]" << std::endl;
        std::cout << "Range Size: 2^134 ≈ 2.17 × 10^40 keys" << std::endl;
        std::cout << "Difficulty: Extremely High" << std::endl;
        std::cout << "Algorithm: Bernstein-Lange with Kangaroo optimization" << std::endl;
        std::cout << "Expected Time: Depends on luck and computational power" << std::endl;
        std::cout << "Reward: Check current Bitcoin puzzle status" << std::endl;
        std::cout << std::endl;
        
        std::cout << "⚠️  IMPORTANT DISCLAIMERS:" << std::endl;
        std::cout << "- This is an extremely difficult computational challenge" << std::endl;
        std::cout << "- Success is not guaranteed even with optimal algorithms" << std::endl;
        std::cout << "- Computational resources required are substantial" << std::endl;
        std::cout << "- Always verify puzzle status before starting" << std::endl;
        std::cout << "- Ensure you have rights to any discovered private keys" << std::endl;
        std::cout << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🎯 Bitcoin Puzzle 135 Challenge Program" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Advanced Bernstein-Lange Elliptic Curve Discrete Logarithm Solver" << std::endl;
    std::cout << "Targeting Bitcoin Puzzle 135: Range [2^134, 2^135-1]" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    std::string table_file = "puzzle135_bl_table.bin";
    uint64_t max_steps = 1000000000;  // 1 billion steps default
    bool show_info = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--table" && i + 1 < argc) {
            table_file = argv[++i];
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps = std::stoull(argv[++i]);
        } else if (arg == "--info") {
            show_info = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --table <file>      Precompute table file (default: puzzle135_bl_table.bin)" << std::endl;
            std::cout << "  --max-steps <n>     Maximum steps to search (default: 1000000000)" << std::endl;
            std::cout << "  --info             Show puzzle information and exit" << std::endl;
            std::cout << "  --help             Show this help message" << std::endl;
            std::cout << std::endl;
            std::cout << "Example:" << std::endl;
            std::cout << "  " << argv[0] << " --table puzzle135_bl_table.bin --max-steps 5000000000" << std::endl;
            return 0;
        }
    }
    
    try {
        Puzzle135Challenge challenge;
        
        if (show_info) {
            challenge.displayChallengeInfo();
            return 0;
        }
        
        challenge.displayChallengeInfo();
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Table file: " << table_file << std::endl;
        std::cout << "  Max steps: " << max_steps << std::endl;
        std::cout << std::endl;
        
        // Load target public key
        if (!challenge.loadTargetPublicKey()) {
            std::cerr << "❌ ERROR: Failed to load target public key!" << std::endl;
            return 1;
        }
        
        // Run the challenge
        bool success = challenge.runChallenge(table_file, max_steps);
        
        if (success) {
            std::cout << "\n🎉 HISTORIC ACHIEVEMENT!" << std::endl;
            std::cout << "Bitcoin Puzzle 135 has been solved!" << std::endl;
            return 0;
        } else {
            std::cout << "\n📊 Challenge completed without solution" << std::endl;
            std::cout << "Consider adjusting parameters and trying again" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
}
