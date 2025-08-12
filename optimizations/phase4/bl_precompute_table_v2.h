/**
 * Bernstein-Lange Precompute Table Structure V2
 * 
 * This header defines the improved precompute table format with proper
 * parameter storage and validation based on user feedback analysis.
 */

#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

// Magic number for file format validation: "BLPT" (Bernstein-Lange Precompute Table)
#define BL_TABLE_MAGIC 0x424C5054

struct PrecomputeTableHeader {
    uint32_t magic;           // Magic number for file format validation
    uint32_t version;         // Version number (current: 1)
    uint64_t ell;            // Group order (curve order for elliptic curves)
    uint64_t L;              // Search interval length (e.g., 2^40)
    uint64_t A;              // Search interval start point
    uint64_t T;              // Target table size
    int32_t W;               // Theoretical walk length
    int32_t dp_mask_bits;    // Distinguished point mask bits
    uint64_t entry_count;    // Actual number of entries in table
    uint64_t reserved[8];    // Reserved fields for future use
};

struct PrecomputeTableEntry {
    // Distinguished point coordinates (secp256k1 field elements)
    struct {
        uint64_t n[4];  // x coordinate (256-bit field element)
    } x;
    struct {
        uint64_t n[4];  // y coordinate (256-bit field element)  
    } y;
    
    // Path information
    uint64_t start_offset;   // Starting point offset relative to A
    uint64_t walk_length;    // Number of steps to reach distinguished point
    uint32_t step_count;     // Actual step count
    uint32_t collision_count; // Number of collisions for this hash
    uint64_t weight;         // Weight value for selection
    uint64_t hash_value;     // Hash value of the distinguished point
};

/**
 * Bernstein-Lange Parameters Calculator
 * 
 * Implements the parameter calculation logic from the paper with
 * practical adjustments for real-world usage.
 */
class BLParameters {
public:
    /**
     * Calculate optimal parameters for given problem size
     * 
     * @param ell Group order (for secp256k1: 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141)
     * @param L Search interval length (e.g., 2^40)
     * @param A Search interval start point (usually 0)
     * @param T Target table size (e.g., 8192)
     * @return Properly configured header structure
     */
    static PrecomputeTableHeader calculateParameters(uint64_t ell, uint64_t L, uint64_t A, uint64_t T) {
        PrecomputeTableHeader header = {0};
        
        header.magic = BL_TABLE_MAGIC;
        header.version = 1;
        header.ell = ell;
        header.L = L;
        header.A = A;
        header.T = T;
        
        // Calculate W parameter using Bernstein-Lange formula
        header.W = calculateW(L, T);
        
        // Calculate DP mask bits for approximately 1/2^k probability
        // We want roughly T distinguished points from N candidates
        // So probability should be around T/N
        header.dp_mask_bits = calculateDPMaskBits(T);
        
        header.entry_count = 0; // Will be set when saving
        
        return header;
    }
    
    /**
     * Calculate theoretical walk length W
     * 
     * Uses the Bernstein-Lange formula: W = α * sqrt(L/T)
     * where α 「 1.33 is the theoretical constant
     */
    static int32_t calculateW(uint64_t L, uint64_t T) {
        double alpha = 1.33;
        
        // Method 1: Theoretical formula from paper
        double W_theory = alpha * sqrt((double)L / (double)T);
        
        // Method 2: Practical formula (more conservative)
        double W_practical = sqrt((double)L / (double)T);
        
        // Choose the more conservative value for better collision probability
        double W_final = std::min(W_theory, W_practical * 1.5);
        
        // Ensure minimum reasonable value
        int32_t W = (int32_t)std::round(W_final);
        return std::max(W, 100); // Minimum 100 steps
    }
    
    /**
     * Calculate DP mask bits for target table size
     * 
     * We want approximately T distinguished points from a larger candidate pool.
     * If we generate N candidates, we want P(DP) 「 T/N
     * For mask bits k, P(DP) = 1/2^k
     * So k 「 log2(N/T)
     */
    static int32_t calculateDPMaskBits(uint64_t T) {
        // For typical usage, we generate about 8-16x more candidates than target
        uint64_t candidate_multiplier = 8;
        uint64_t N = T * candidate_multiplier;
        
        // Calculate required mask bits
        double ratio = (double)N / (double)T;
        int32_t mask_bits = (int32_t)std::round(log2(ratio));
        
        // Clamp to reasonable range
        return std::max(1, std::min(mask_bits, 20));
    }
    
    /**
     * Print parameter analysis for debugging
     */
    static void printParameterAnalysis(const PrecomputeTableHeader& header) {
        std::cout << "\n?? Bernstein-Lange Parameter Analysis" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Problem Configuration:" << std::endl;
        std::cout << "  Group order (ell): " << header.ell << std::endl;
        std::cout << "  Search interval length (L): " << header.L << " (2^" << log2(header.L) << ")" << std::endl;
        std::cout << "  Search start point (A): " << header.A << std::endl;
        std::cout << "  Target table size (T): " << header.T << std::endl;
        
        std::cout << "\nCalculated Parameters:" << std::endl;
        std::cout << "  Walk length (W): " << header.W << std::endl;
        std::cout << "  DP mask bits: " << header.dp_mask_bits << std::endl;
        std::cout << "  DP probability: 1/" << (1ULL << header.dp_mask_bits) << " 「 " 
                  << (100.0 / (1ULL << header.dp_mask_bits)) << "%" << std::endl;
        
        std::cout << "\nTheoretical Analysis:" << std::endl;
        double W_theory = 1.33 * sqrt((double)header.L / (double)header.T);
        std::cout << "  Theoretical W: " << W_theory << std::endl;
        std::cout << "  W ratio (actual/theory): " << ((double)header.W / W_theory) << std::endl;
        
        double expected_candidates = header.T * (1ULL << header.dp_mask_bits);
        std::cout << "  Expected candidates needed: " << expected_candidates << std::endl;
        std::cout << "  Candidate/Target ratio: " << (expected_candidates / header.T) << "x" << std::endl;
    }
    
    /**
     * Validate parameter consistency
     */
    static bool validateParameters(const PrecomputeTableHeader& header) {
        if (header.magic != BL_TABLE_MAGIC) {
            std::cerr << "? Invalid magic number" << std::endl;
            return false;
        }
        
        if (header.version != 1) {
            std::cerr << "? Unsupported version: " << header.version << std::endl;
            return false;
        }
        
        if (header.L == 0 || header.T == 0) {
            std::cerr << "? Invalid L or T parameters" << std::endl;
            return false;
        }
        
        if (header.W <= 0 || header.dp_mask_bits <= 0) {
            std::cerr << "? Invalid W or dp_mask_bits parameters" << std::endl;
            return false;
        }
        
        // Check if parameters are reasonable
        double W_theory = 1.33 * sqrt((double)header.L / (double)header.T);
        if (header.W > W_theory * 3 || header.W < W_theory * 0.3) {
            std::cerr << "??  Warning: W parameter seems unreasonable (theory: " << W_theory << ", actual: " << header.W << ")" << std::endl;
        }
        
        return true;
    }
};

/**
 * Example parameter sets for common problem sizes
 */
namespace BLParameterSets {
    // Small test problems
    inline PrecomputeTableHeader createTestSet20() {
        return BLParameters::calculateParameters(
            0xFFFFFFFFFFFFFFFFULL,  // ell (simplified)
            1ULL << 20,             // L = 2^20 「 1M
            0,                      // A = 0
            1024                    // T = 1024
        );
    }
    
    inline PrecomputeTableHeader createTestSet30() {
        return BLParameters::calculateParameters(
            0xFFFFFFFFFFFFFFFFULL,  // ell (simplified)
            1ULL << 30,             // L = 2^30 「 1B
            0,                      // A = 0
            4096                    // T = 4096
        );
    }
    
    inline PrecomputeTableHeader createTestSet40() {
        return BLParameters::calculateParameters(
            0xFFFFFFFFFFFFFFFFULL,  // ell (simplified)
            1ULL << 40,             // L = 2^40 「 1T
            0,                      // A = 0
            8192                    // T = 8192
        );
    }
}
