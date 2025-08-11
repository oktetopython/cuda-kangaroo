/**
 * Bernstein-Lange Precompute Table I/O Operations V2
 * 
 * Handles saving and loading of precompute tables with proper
 * parameter validation and error handling.
 */

#pragma once
#include "bl_precompute_table_v2.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstddef>

/**
 * Precompute Table Saver
 * 
 * Handles saving precompute tables to binary files with proper
 * header information and validation.
 */
class PrecomputeTableSaver {
public:
    /**
     * Save precompute table to binary file
     * 
     * @param filename Output filename
     * @param header Table header with parameters
     * @param entries Vector of table entries
     * @return true if successful, false otherwise
     */
    static bool saveTable(const std::string& filename, 
                         PrecomputeTableHeader& header,
                         const std::vector<PrecomputeTableEntry>& entries) {
        
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        // Update entry count in header
        header.entry_count = entries.size();
        
        // Validate parameters before saving
        if (!BLParameters::validateParameters(header)) {
            std::cerr << "ERROR: Invalid parameters, cannot save table" << std::endl;
            return false;
        }
        
        // Write header
        if (!file.write(reinterpret_cast<const char*>(&header), sizeof(PrecomputeTableHeader))) {
            std::cerr << "ERROR: Failed to write table header" << std::endl;
            return false;
        }
        
        // Write all entries
        for (size_t i = 0; i < entries.size(); i++) {
            if (!file.write(reinterpret_cast<const char*>(&entries[i]), sizeof(PrecomputeTableEntry))) {
                std::cerr << "ERROR: Failed to write entry " << i << std::endl;
                return false;
            }
        }
        
        file.close();
        
        // Print save summary
        printSaveSummary(filename, header, entries);
        
        return true;
    }
    
private:
    static void printSaveSummary(const std::string& filename, 
                                const PrecomputeTableHeader& header,
                                const std::vector<PrecomputeTableEntry>& entries) {
        
        std::cout << "\nPrecompute Table Saved Successfully" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "File: " << filename << std::endl;
        std::cout << "Entries: " << entries.size() << std::endl;
        
        // Calculate file size
        size_t file_size = sizeof(PrecomputeTableHeader) + entries.size() * sizeof(PrecomputeTableEntry);
        std::cout << "File size: " << file_size << " bytes (" << (file_size / 1024.0 / 1024.0) << " MB)" << std::endl;
        
        // Print parameter summary
        std::cout << "Parameters: L=2^" << log2(header.L) << ", T=" << header.T 
                  << ", W=" << header.W << ", DP_bits=" << header.dp_mask_bits << std::endl;
    }
};

/**
 * Precompute Table Loader
 * 
 * Handles loading precompute tables from binary files with validation
 * and detailed error reporting.
 */
class PrecomputeTableLoader {
public:
    /**
     * Load precompute table from binary file
     * 
     * @param filename Input filename
     * @param header Output header structure
     * @param entries Output vector of entries
     * @return true if successful, false otherwise
     */
    static bool loadTable(const std::string& filename,
                         PrecomputeTableHeader& header,
                         std::vector<PrecomputeTableEntry>& entries) {
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Failed to open precompute table file: " << filename << std::endl;
            return false;
        }
        
        // Read header
        if (!file.read(reinterpret_cast<char*>(&header), sizeof(PrecomputeTableHeader))) {
            std::cerr << "ERROR: Failed to read table header" << std::endl;
            return false;
        }
        
        // Validate header
        if (!BLParameters::validateParameters(header)) {
            std::cerr << "ERROR: Invalid table parameters" << std::endl;
            return false;
        }
        
        // Read entries
        entries.clear();
        entries.resize(header.entry_count);
        
        for (uint64_t i = 0; i < header.entry_count; i++) {
            if (!file.read(reinterpret_cast<char*>(&entries[i]), sizeof(PrecomputeTableEntry))) {
                std::cerr << "ERROR: Failed to read entry " << i << std::endl;
                return false;
            }
        }
        
        file.close();
        
        // Print load summary
        printLoadSummary(filename, header, entries);
        
        return true;
    }
    
    /**
     * Print detailed table information for debugging
     */
    static void printTableInfo(const PrecomputeTableHeader& header) {
        std::cout << "\nPrecompute Table Information" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Magic: 0x" << std::hex << header.magic << std::dec;
        if (header.magic == BL_TABLE_MAGIC) {
            std::cout << " (Valid)";
        } else {
            std::cout << " (Invalid)";
        }
        std::cout << std::endl;
        
        std::cout << "Version: " << header.version << std::endl;
        std::cout << "Group order (ell): " << header.ell << std::endl;
        std::cout << "Search length (L): " << header.L << " (2^" << log2(header.L) << ")" << std::endl;
        std::cout << "Search start (A): " << header.A << std::endl;
        std::cout << "Target size (T): " << header.T << std::endl;
        std::cout << "Walk length (W): " << header.W << std::endl;
        std::cout << "DP mask bits: " << header.dp_mask_bits << std::endl;
        std::cout << "Entry count: " << header.entry_count << std::endl;
        
        // Calculate and display theoretical values
        double W_theory = 1.33 * sqrt((double)header.L / (double)header.T);
        std::cout << "\nTheoretical Analysis:" << std::endl;
        std::cout << "  Theoretical W: " << W_theory << std::endl;
        std::cout << "  W ratio: " << ((double)header.W / W_theory) << std::endl;
        std::cout << "  DP probability: 1/" << (1ULL << header.dp_mask_bits) 
                  << " (~" << (100.0 / (1ULL << header.dp_mask_bits)) << "%)" << std::endl;
    }
    
    /**
     * Print sample entries for debugging
     */
    static void printSampleEntries(const std::vector<PrecomputeTableEntry>& entries, size_t max_samples = 5) {
        std::cout << "\nSample Table Entries" << std::endl;
        std::cout << "=======================" << std::endl;
        
        size_t samples = (entries.size() < max_samples) ? entries.size() : max_samples;
        for (size_t i = 0; i < samples; i++) {
            const auto& entry = entries[i];
            std::cout << "Entry " << i << ":" << std::endl;
            std::cout << "  x[0]: 0x" << std::hex << entry.x.n[0] << std::dec << std::endl;
            std::cout << "  y[0]: 0x" << std::hex << entry.y.n[0] << std::dec << std::endl;
            std::cout << "  start_offset: " << entry.start_offset << std::endl;
            std::cout << "  walk_length: " << entry.walk_length << std::endl;
            std::cout << "  hash_value: 0x" << std::hex << entry.hash_value << std::dec << std::endl;
            std::cout << std::endl;
        }
        
        if (entries.size() > max_samples) {
            std::cout << "... and " << (entries.size() - max_samples) << " more entries" << std::endl;
        }
    }
    
private:
    static void printLoadSummary(const std::string& filename, 
                                const PrecomputeTableHeader& header,
                                const std::vector<PrecomputeTableEntry>& entries) {
        
        std::cout << "\nPrecompute Table Loaded Successfully" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "File: " << filename << std::endl;
        std::cout << "Entries loaded: " << entries.size() << std::endl;
        std::cout << "Parameters: L=2^" << log2(header.L) << ", T=" << header.T 
                  << ", W=" << header.W << ", DP_bits=" << header.dp_mask_bits << std::endl;
        
        // Validate entry count consistency
        if (entries.size() != header.entry_count) {
            std::cout << "WARNING: Entry count mismatch (header: " << header.entry_count
                      << ", loaded: " << entries.size() << ")" << std::endl;
        }
        
        // Calculate statistics
        if (!entries.empty()) {
            uint64_t min_walk = entries[0].walk_length;
            uint64_t max_walk = entries[0].walk_length;
            uint64_t total_walk = 0;
            
            for (const auto& entry : entries) {
                if (entry.walk_length < min_walk) min_walk = entry.walk_length;
                if (entry.walk_length > max_walk) max_walk = entry.walk_length;
                total_walk += entry.walk_length;
            }
            
            std::cout << "Walk length statistics:" << std::endl;
            std::cout << "  Min: " << min_walk << std::endl;
            std::cout << "  Max: " << max_walk << std::endl;
            std::cout << "  Average: " << (total_walk / entries.size()) << std::endl;
            std::cout << "  Theoretical: " << header.W << std::endl;
        }
    }
};

/**
 * Utility functions for table management
 */
namespace BLTableUtils {
    /**
     * Create a test table with specified parameters
     */
    inline bool createTestTable(const std::string& filename, uint64_t L, uint64_t T) {
        PrecomputeTableHeader header = BLParameters::calculateParameters(
            0xFFFFFFFFFFFFFFFFULL, L, 0, T
        );
        
        // Create some dummy entries for testing
        std::vector<PrecomputeTableEntry> entries;
        uint64_t max_entries = (T < 100) ? T : 100;
        for (uint64_t i = 0; i < max_entries; i++) {
            PrecomputeTableEntry entry = {0};
            entry.x.n[0] = i * 0x123456789ABCDEFULL;
            entry.y.n[0] = i * 0xFEDCBA9876543210ULL;
            entry.start_offset = i;
            entry.walk_length = header.W + (i % 100);
            entry.hash_value = entry.x.n[0] ^ entry.y.n[0];
            entries.push_back(entry);
        }
        
        return PrecomputeTableSaver::saveTable(filename, header, entries);
    }
    
    /**
     * Verify table integrity
     */
    inline bool verifyTable(const std::string& filename) {
        PrecomputeTableHeader header;
        std::vector<PrecomputeTableEntry> entries;
        
        if (!PrecomputeTableLoader::loadTable(filename, header, entries)) {
            return false;
        }
        
        PrecomputeTableLoader::printTableInfo(header);
        BLParameters::printParameterAnalysis(header);
        
        return true;
    }
}
