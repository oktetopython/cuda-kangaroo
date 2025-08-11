/**
 * Real Elliptic Curve Bernstein-Lange Table Generator
 * 
 * This program generates precompute tables using REAL secp256k1 
 * elliptic curve operations, not virtual/simulated data.
 * 
 * Based on Bernstein-Lange paper formulas and real EC arithmetic.
 */

#include "bl_real_ec_generator.h"
#include <iostream>
#include <string>
#include <cmath>

/**
 * Print usage information
 */
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <L_bits> <T> [output_file]" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  L_bits      - Search interval length as power of 2 (e.g., 20 for 2^20)" << std::endl;
    std::cout << "  T           - Target table size (number of entries)" << std::endl;
    std::cout << "  output_file - Output filename (optional, default: auto-generated)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " 20 1024    # Generate L=2^20, T=1024 table" << std::endl;
    std::cout << "  " << program_name << " 30 4096    # Generate L=2^30, T=4096 table" << std::endl;
    std::cout << "  " << program_name << " 40 8192    # Generate L=2^40, T=8192 table" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This uses REAL secp256k1 elliptic curve operations!" << std::endl;
}

/**
 * Validate parameters
 */
bool validateParameters(int L_bits, uint64_t T) {
    if (L_bits < 10 || L_bits > 60) {
        std::cerr << "❌ Error: L_bits must be between 10 and 60" << std::endl;
        return false;
    }
    
    if (T < 64 || T > 1000000) {
        std::cerr << "❌ Error: T must be between 64 and 1,000,000" << std::endl;
        return false;
    }
    
    // Check if T is reasonable for the problem size
    uint64_t L = 1ULL << L_bits;
    double theoretical_W = 1.33 * sqrt((double)L / T);
    
    if (theoretical_W < 10) {
        std::cerr << "⚠️  Warning: Theoretical W = " << theoretical_W 
                  << " is very small. Consider reducing T." << std::endl;
    }
    
    if (theoretical_W > 100000) {
        std::cerr << "⚠️  Warning: Theoretical W = " << theoretical_W 
                  << " is very large. Consider increasing T." << std::endl;
    }
    
    return true;
}

/**
 * Generate auto filename
 */
std::string generateFilename(int L_bits, uint64_t T) {
    return "bl_real_ec_table_L" + std::to_string(L_bits) + "_T" + std::to_string(T) + ".bin";
}

/**
 * Print theoretical analysis
 */
void printTheoreticalAnalysis(int L_bits, uint64_t T) {
    uint64_t L = 1ULL << L_bits;
    double theoretical_W = 1.33 * sqrt((double)L / T);
    double expected_candidates = 8.0 * T;  // Bernstein-Lange paper
    
    std::cout << "📊 Theoretical Analysis (Bernstein-Lange Paper)" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Problem Configuration:" << std::endl;
    std::cout << "  Search interval L = 2^" << L_bits << " = " << L << std::endl;
    std::cout << "  Target table size T = " << T << std::endl;
    std::cout << std::endl;
    std::cout << "Theoretical Parameters:" << std::endl;
    std::cout << "  Walk length W ≈ " << std::fixed << std::setprecision(2) << theoretical_W << std::endl;
    std::cout << "  DP probability = 1/8 = 12.5%" << std::endl;
    std::cout << "  Expected candidates ≈ " << std::fixed << std::setprecision(0) << expected_candidates << std::endl;
    std::cout << std::endl;
    std::cout << "Performance Estimates:" << std::endl;
    std::cout << "  Average steps per DP ≈ " << std::fixed << std::setprecision(0) << theoretical_W << std::endl;
    std::cout << "  Total EC operations ≈ " << std::fixed << std::setprecision(0) << (T * theoretical_W) << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🔧 Real Elliptic Curve Bernstein-Lange Table Generator" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Using REAL secp256k1 elliptic curve operations" << std::endl;
    std::cout << "Based on Bernstein-Lange paper formulas" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    int L_bits = std::atoi(argv[1]);
    uint64_t T = std::stoull(argv[2]);
    std::string output_file;
    
    if (argc >= 4) {
        output_file = argv[3];
    } else {
        output_file = generateFilename(L_bits, T);
    }
    
    // Validate parameters
    if (!validateParameters(L_bits, T)) {
        return 1;
    }
    
    uint64_t L = 1ULL << L_bits;
    uint64_t A = 0;  // Start from 0 for simplicity
    
    std::cout << "Generation Parameters:" << std::endl;
    std::cout << "  L = 2^" << L_bits << " = " << L << std::endl;
    std::cout << "  A = " << A << std::endl;
    std::cout << "  T = " << T << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << std::endl;
    
    // Print theoretical analysis
    printTheoreticalAnalysis(L_bits, T);
    
    // Confirm with user
    std::cout << "⚠️  This will use REAL elliptic curve operations and may take significant time." << std::endl;
    std::cout << "Continue? (y/N): ";
    std::string confirm;
    std::getline(std::cin, confirm);
    
    if (confirm != "y" && confirm != "Y" && confirm != "yes" && confirm != "YES") {
        std::cout << "Operation cancelled." << std::endl;
        return 0;
    }
    
    // Create generator and generate table
    std::cout << std::endl;
    std::cout << "🚀 Starting Real EC Table Generation..." << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        RealECBLGenerator generator(L, A, T);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = generator.generatePrecomputeTable(output_file);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        if (success) {
            std::cout << std::endl;
            std::cout << "🎉 Real EC Table Generation Completed Successfully!" << std::endl;
            std::cout << "==================================================" << std::endl;
            std::cout << "✅ Output file: " << output_file << std::endl;
            std::cout << "✅ Total time: " << total_duration.count() << " seconds" << std::endl;
            std::cout << "✅ Using REAL secp256k1 elliptic curve operations" << std::endl;
            std::cout << "✅ Based on Bernstein-Lange paper formulas" << std::endl;
            std::cout << std::endl;
            std::cout << "🔍 Next Steps:" << std::endl;
            std::cout << "1. Verify table: ./verify_bl_table_v2 " << output_file << std::endl;
            std::cout << "2. Test solver: ./test_bl_online_solver_v2 --table=" << output_file << std::endl;
            std::cout << "3. Run DLP tests with known discrete logarithms" << std::endl;
            std::cout << std::endl;
            std::cout << "🏆 This table uses REAL elliptic curve arithmetic!" << std::endl;
            
            return 0;
        } else {
            std::cerr << "❌ Table generation failed!" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception during generation: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error during generation" << std::endl;
        return 1;
    }
}
