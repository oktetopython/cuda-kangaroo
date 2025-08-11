/*
 * Bernstein-Lange Integration for Kangaroo
 * Integrates the V2 format Bernstein-Lange precompute tables with the main Kangaroo solver
 * Based on real secp256k1 elliptic curve operations
 */

#ifndef KANGAROO_BL_INTEGRATION_H
#define KANGAROO_BL_INTEGRATION_H

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "optimizations/phase4/bl_precompute_table_v2.h"
#include "Timer.h"

// Bernstein-Lange integration with main Kangaroo program
class BernsteinLangeKangaroo {
private:
    Secp256K1* secp;
    PrecomputeTableHeader header;
    std::vector<PrecomputeTableEntry> table_entries;
    std::unordered_map<uint64_t, std::vector<int>> hash_index;
    bool table_loaded;
    
    // Performance metrics
    uint64_t total_steps;
    uint64_t total_distinguished_points;
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    BernsteinLangeKangaroo(Secp256K1* secp_instance) : 
        secp(secp_instance), table_loaded(false), total_steps(0), total_distinguished_points(0) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~BernsteinLangeKangaroo() {}
    
    // Load V2 format precompute table
    bool loadPrecomputeTable(const std::string& filename) {
        std::cout << "Loading Bernstein-Lange precompute table: " << filename << std::endl;
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Error: Cannot open table file: " << filename << std::endl;
            return false;
        }
        
        // Read header
        file.read(reinterpret_cast<char*>(&header), sizeof(PrecomputeTableHeader));
        
        // Verify magic number
        if (header.magic != 0x424C5054) { // "BLPT"
            std::cout << "Error: Invalid magic number in table file" << std::endl;
            return false;
        }
        
        std::cout << "Table info:" << std::endl;
        std::cout << "  Version: " << header.version << std::endl;
        std::cout << "  L: 2^" << (int)log2(header.L) << " (" << header.L << ")" << std::endl;
        std::cout << "  T: " << header.T << std::endl;
        std::cout << "  W: " << header.W << std::endl;
        std::cout << "  DP bits: " << header.dp_mask_bits << std::endl;
        std::cout << "  Entries: " << header.entry_count << std::endl;
        
        // Read entries
        table_entries.resize(header.entry_count);
        file.read(reinterpret_cast<char*>(table_entries.data()), 
                  header.entry_count * sizeof(PrecomputeTableEntry));
        
        file.close();
        
        // Build hash index for fast lookup
        buildHashIndex();
        
        table_loaded = true;
        std::cout << "Table loaded successfully with " << table_entries.size() << " entries" << std::endl;
        return true;
    }
    
    // Build hash index for fast collision detection
    void buildHashIndex() {
        std::cout << "Building hash index..." << std::endl;
        hash_index.clear();

        for (size_t i = 0; i < table_entries.size(); i++) {
            uint64_t hash_value = table_entries[i].x.n[0]; // Use first 64-bit word as hash
            hash_index[hash_value].push_back(i);
        }

        std::cout << "Hash index built with " << hash_index.size() << " unique hashes" << std::endl;
    }
    
    // Check if point is distinguished based on dp_mask_bits
    bool isDistinguishedPoint(const Point& point) {
        // Check if lower dp_mask_bits are zero
        uint64_t mask = (1ULL << header.dp_mask_bits) - 1;
        return (point.x.bits64[0] & mask) == 0;
    }
    
    // Calculate step size from point hash (Bernstein-Lange method)
    int calculateStepFromHash(const Point& point) {
        uint64_t hash = point.x.bits64[0];

        // Use hash to determine step size (1 to W)
        int step = (hash % header.W) + 1;
        return step;
    }
    
    // Perform random walk step
    void performRandomWalkStep(Point& current_point, int step_size) {
        // Create step point: step_size * G
        Int step_int((uint64_t)step_size);
        Point step_point = secp->ComputePublicKey(&step_int);

        // Add step to current point
        current_point = secp->AddDirect(current_point, step_point);
    }
    
    // Main Bernstein-Lange DLP solver
    bool solveDLP(const Point& target_point, Int& result, uint64_t max_steps = 10000000) {
        if (!table_loaded) {
            std::cout << "Error: No precompute table loaded" << std::endl;
            return false;
        }
        
        std::cout << "Starting Bernstein-Lange DLP solver..." << std::endl;
        std::cout << "Target point: [Point object]" << std::endl;
        std::cout << "Max steps: " << max_steps << std::endl;
        
        Point current_point = target_point;
        uint64_t walk_length = 0;
        uint64_t steps = 0;
        
        auto solve_start = std::chrono::high_resolution_clock::now();
        
        while (steps < max_steps) {
            // Calculate step size
            int step = calculateStepFromHash(current_point);
            walk_length += step;
            steps++;
            total_steps++;
            
            // Perform random walk step
            performRandomWalkStep(current_point, step);
            
            // Check if distinguished point
            if (isDistinguishedPoint(current_point)) {
                total_distinguished_points++;
                
                // Look for collision in precompute table
                uint64_t hash_value = current_point.x.bits64[0];
                auto it = hash_index.find(hash_value);

                if (it != hash_index.end()) {
                    // Found potential collision, verify
                    for (int idx : it->second) {
                        const auto& entry = table_entries[idx];

                        // Compare x coordinates (simplified comparison)
                        if (current_point.x.bits64[0] == entry.x.n[0] &&
                            current_point.x.bits64[1] == entry.x.n[1]) {
                            // Found collision! Reconstruct solution
                            auto solve_end = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start);
                            
                            std::cout << "COLLISION FOUND!" << std::endl;
                            std::cout << "Steps: " << steps << std::endl;
                            std::cout << "Walk length: " << walk_length << std::endl;
                            std::cout << "Time: " << duration.count() << " ms" << std::endl;
                            
                            // Reconstruct the discrete logarithm
                            // result = entry.start_offset + walk_length (modulo group order)
                            result = Int(entry.start_offset);
                            Int walk_int((uint64_t)walk_length);
                            result.Add(&walk_int);
                            
                            return true;
                        }
                    }
                }
                
                // Progress reporting
                if (total_distinguished_points % 1000 == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - solve_start);
                    std::cout << "Progress: " << steps << " steps, " 
                              << total_distinguished_points << " DPs, "
                              << elapsed.count() << "s" << std::endl;
                }
            }
        }
        
        std::cout << "DLP solver reached maximum steps without finding solution" << std::endl;
        return false;
    }
    
    // Get performance statistics
    void printStatistics() {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        
        std::cout << "Bernstein-Lange Performance Statistics:" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Distinguished points: " << total_distinguished_points << std::endl;
        std::cout << "  Runtime: " << elapsed.count() << " seconds" << std::endl;
        if (elapsed.count() > 0) {
            std::cout << "  Steps/second: " << total_steps / elapsed.count() << std::endl;
        }
        if (total_steps > 0) {
            double dp_rate = (double)total_distinguished_points / total_steps * 100.0;
            std::cout << "  DP rate: " << dp_rate << "%" << std::endl;
        }
    }
    
    // Test with known discrete logarithm (for validation)
    bool testWithKnownDLP(Int& known_private_key) {
        std::cout << "Testing with known private key: " << known_private_key.GetBase16() << std::endl;

        // Generate public key from private key
        Point public_key = secp->ComputePublicKey(&known_private_key);

        std::cout << "Generated public key: [Point object]" << std::endl;
        
        // Solve DLP
        Int result;
        bool success = solveDLP(public_key, result);
        
        if (success) {
            std::cout << "Solution found: " << result.GetBase16() << std::endl;
            std::cout << "Expected: " << known_private_key.GetBase16() << std::endl;

            if (result.IsEqual(&known_private_key)) {
                std::cout << "SUCCESS: Solution matches expected value!" << std::endl;
                return true;
            } else {
                std::cout << "ERROR: Solution does not match expected value" << std::endl;
                return false;
            }
        } else {
            std::cout << "ERROR: No solution found" << std::endl;
            return false;
        }
    }
};

#endif // KANGAROO_BL_INTEGRATION_H
