/**
 * Bitcoin Puzzle 135 Bernstein-Lange Precompute Table Generator
 * 
 * Generates optimized precompute table for Bitcoin Puzzle 135:
 * Range: [2^134, 2^135-1] 
 * Length: 2^134
 * Target: Find private key in this range
 */

#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"
#include "SECPK1/Random.h"
#include "optimizations/phase4/bl_precompute_table_v2.h"
#include "optimizations/phase4/bl_table_io_v2.h"
#include "optimizations/phase4/bl_real_ec_generator.h"
#include "Timer.h"

class Puzzle135BLGenerator {
private:
    Secp256K1 secp;
    double start_time;
    
    // Bitcoin Puzzle 135 parameters
    Int puzzle_start;    // 2^134
    Int puzzle_end;      // 2^135 - 1
    Int puzzle_length;   // 2^134
    
public:
    Puzzle135BLGenerator() {
        // Initialize Puzzle 135 range: [2^134, 2^135-1]
        puzzle_start.SetBase16("4000000000000000000000000000000000");  // 2^134
        puzzle_end.SetBase16("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");    // 2^135-1
        puzzle_length = puzzle_end;
        puzzle_length.Sub(&puzzle_start);
        puzzle_length.Add(1);  // Include end point
        
        std::cout << "🎯 Bitcoin Puzzle 135 Bernstein-Lange Generator" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Range Start: 0x" << puzzle_start.GetBase16() << std::endl;
        std::cout << "Range End:   0x" << puzzle_end.GetBase16() << std::endl;
        std::cout << "Range Length: 2^134 ≈ " << std::scientific << std::setprecision(3) 
                  << pow(2.0, 134) << std::endl;
    }
    
    bool generatePuzzle135Table(uint64_t T = 65536, const std::string& output_filename = "puzzle135_bl_table.bin") {
        std::cout << "\n🔧 Generating Puzzle 135 Precompute Table..." << std::endl;
        std::cout << "Target table size (T): " << T << std::endl;
        
        // Calculate theoretical parameters using the correct API
        // Use a simplified order for parameter calculation
        PrecomputeTableHeader params = BLParameters::calculateParameters(
            0xFFFFFFFFFFFFFFFFULL, // Simplified order for calculation
            134,  // L = 2^134 (exponent)
            134,  // A = 2^134 (start point exponent)
            T     // Target table size
        );

        std::cout << "Theoretical W: " << params.W << std::endl;
        std::cout << "DP mask bits: " << params.dp_mask_bits << std::endl;
        std::cout << "Expected DP rate: " << (100.0 / (1 << params.dp_mask_bits)) << "%" << std::endl;
        
        // Use the calculated header as our table header
        PrecomputeTableHeader header = params;
        header.entry_count = 0;  // Will be updated
        
        // Generate entries using real elliptic curve operations
        std::vector<PrecomputeTableEntry> entries;
        entries.reserve(T);
        
        std::cout << "\n🚀 Starting real elliptic curve precompute generation..." << std::endl;
        Timer::Init();
        start_time = Timer::get_tick();

        uint32_t dp_mask = (1U << header.dp_mask_bits) - 1;
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());

        for (uint64_t i = 0; i < T; i++) {
            // Generate random starting point in Puzzle 135 range
            Int random_offset;
            random_offset.Rand(134);  // Random value in [0, 2^134)
            Int start_point = puzzle_start;
            start_point.Add(&random_offset);

            // Ensure we're within range
            if (start_point.IsGreater(&puzzle_end)) {
                start_point = puzzle_start;
                Int small_offset;
                small_offset.Rand(64);  // Smaller random offset
                start_point.Add(&small_offset);
            }

            // Perform random walk
            Int current_key = start_point;
            Point current_point = secp.ComputePublicKey(&current_key);

            int steps = 0;
            int max_steps = header.W * 2;  // Allow some variation
            
            while (steps < max_steps) {
                // Take a step
                Int step;
                step.SetInt32(1 + (rng() % 16));  // Random step size 1-16
                current_key.Add(&step);
                
                // Ensure we stay in range
                if (current_key.IsGreater(&puzzle_end)) {
                    current_key = start_point;
                    Int new_offset;
                    new_offset.Rand(64);
                    current_key.Add(&new_offset);
                }
                
                current_point = secp.ComputePublicKey(&current_key);
                steps++;
                
                // Check if distinguished point
                uint64_t hash = current_point.x.bits64[0];
                if ((hash & dp_mask) == 0) {
                    // Found distinguished point
                    PrecomputeTableEntry entry;
                    entry.hash_value = hash >> header.dp_mask_bits;
                    entry.start_offset = current_key.bits64[0];
                    entry.step_count = steps;
                    entry.x.n[0] = current_point.x.bits64[0];
                    entry.y.n[0] = current_point.y.bits64[0];
                    entry.walk_length = steps;
                    entry.collision_count = 0;
                    entry.weight = 1;

                    entries.push_back(entry);
                    break;
                }
            }
            
            // Progress reporting
            if ((i + 1) % 1000 == 0) {
                double current_time = Timer::get_tick();
                double elapsed = current_time - start_time;
                double rate = (i + 1) / elapsed;
                double eta = (T - i - 1) / rate;
                
                std::cout << "Progress: " << (i + 1) << "/" << T 
                          << " (" << std::fixed << std::setprecision(1) 
                          << (100.0 * (i + 1) / T) << "%), "
                          << std::fixed << std::setprecision(0) << rate << " entries/sec, "
                          << "ETA: " << std::fixed << std::setprecision(0) << eta << "s (REAL EC)" << std::endl;
            }
        }
        
        header.entry_count = entries.size();
        double end_time = Timer::get_tick();
        double generation_time = end_time - start_time;
        
        std::cout << "\n✅ Precompute generation completed!" << std::endl;
        std::cout << "Generated entries: " << entries.size() << "/" << T << std::endl;
        std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * entries.size() / T) << "%" << std::endl;
        std::cout << "Generation time: " << std::fixed << std::setprecision(3) 
                  << generation_time << "s" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(0) 
                  << (entries.size() / generation_time) << " entries/sec" << std::endl;
        
        // Save table
        std::cout << "\n💾 Saving Puzzle 135 table to: " << output_filename << std::endl;
        
        PrecomputeTableSaver saver;
        if (!saver.saveTable(output_filename, header, entries)) {
            std::cerr << "❌ ERROR: Failed to save table!" << std::endl;
            return false;
        }
        
        // Verify saved table
        PrecomputeTableLoader loader;
        PrecomputeTableHeader verify_header;
        std::vector<PrecomputeTableEntry> verify_entries;
        
        if (!loader.loadTable(output_filename, verify_header, verify_entries)) {
            std::cerr << "❌ ERROR: Failed to verify saved table!" << std::endl;
            return false;
        }
        
        std::cout << "✅ Table verification successful!" << std::endl;
        std::cout << "File size: " << (sizeof(header) + entries.size() * sizeof(PrecomputeTableEntry)) 
                  << " bytes" << std::endl;
        
        // Display final parameters
        std::cout << "\n📊 Final Puzzle 135 Table Parameters:" << std::endl;
        std::cout << "Magic: 0x" << std::hex << verify_header.magic << std::dec << std::endl;
        std::cout << "Version: " << verify_header.version << std::endl;
        std::cout << "Range: 2^" << verify_header.ell << std::endl;
        std::cout << "Length: 2^" << verify_header.L << std::endl;
        std::cout << "Start: 2^" << verify_header.A << std::endl;
        std::cout << "Table size (T): " << verify_header.T << std::endl;
        std::cout << "Theoretical W: " << verify_header.W << std::endl;
        std::cout << "DP mask bits: " << verify_header.dp_mask_bits << std::endl;
        std::cout << "Actual entries: " << verify_header.entry_count << std::endl;
        std::cout << "DP rate: " << std::fixed << std::setprecision(2) 
                  << (100.0 / (1 << verify_header.dp_mask_bits)) << "%" << std::endl;
        
        std::cout << "\n🎯 Puzzle 135 table ready for challenge!" << std::endl;
        return true;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🎯 Bitcoin Puzzle 135 Bernstein-Lange Challenge" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "This generator creates optimized precompute tables for Bitcoin Puzzle 135" << std::endl;
    std::cout << "Range: [2^134, 2^135-1]" << std::endl;
    std::cout << "Prize: Significant Bitcoin reward for solution" << std::endl;
    std::cout << std::endl;
    
    // Parse command line arguments
    uint64_t T = 65536;  // Default table size
    std::string output_file = "puzzle135_bl_table.bin";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--T" && i + 1 < argc) {
            T = std::stoull(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --T <size>        Target table size (default: 65536)" << std::endl;
            std::cout << "  --output <file>   Output filename (default: puzzle135_bl_table.bin)" << std::endl;
            std::cout << "  --help           Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Table size (T): " << T << std::endl;
    std::cout << "  Output file: " << output_file << std::endl;
    std::cout << std::endl;
    
    try {
        Puzzle135BLGenerator generator;
        
        if (generator.generatePuzzle135Table(T, output_file)) {
            std::cout << "\n🎉 SUCCESS! Puzzle 135 table generated successfully!" << std::endl;
            std::cout << "Ready to challenge Bitcoin Puzzle 135!" << std::endl;
            return 0;
        } else {
            std::cerr << "\n❌ FAILED! Table generation failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
}
