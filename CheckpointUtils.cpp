/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
 * Copyright (c) 2020 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "OptimizedCheckpoint.h"
#include "Timer.h"
#include <iostream>
#include <sstream>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
    #include <errno.h>
#endif

// Kangaroo state save/load functions for OptimizedCheckpoint class
bool OptimizedCheckpoint::SaveKangaroosOptimized(FILE* f, TH_PARAM* threads, int nbThread) {
    // Count total kangaroos
    uint64_t total_kangaroos = 0;
    for (int i = 0; i < nbThread; i++) {
        total_kangaroos += threads[i].nbKangaroo;
    }
    
    // Prepare kangaroo header
    KangarooHeader k_header;
    memset(&k_header, 0, sizeof(k_header));
    
    k_header.magic = KANGAROO_MAGIC;
    k_header.thread_count = static_cast<uint32_t>(nbThread);
    k_header.total_kangaroos = total_kangaroos;
    k_header.storage_type = 0; // Full storage for now
    
    // Write kangaroo header
    if (!WriteWithErrorCheck(f, &k_header, sizeof(k_header))) {
        return false;
    }
    
    // Write thread information and kangaroo states
    uint64_t kangaroos_written = 0;
    for (int i = 0; i < nbThread; i++) {
        // Write thread metadata
        if (!WriteWithErrorCheck(f, &threads[i].nbKangaroo, sizeof(uint64_t))) {
            return false;
        }
        
        // Write kangaroo states for this thread
        for (uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
            // Write position (px, py) and distance
            if (!WriteWithErrorCheck(f, &threads[i].px[n].bits64, 32)) {
                return false;
            }
            if (!WriteWithErrorCheck(f, &threads[i].py[n].bits64, 32)) {
                return false;
            }
            if (!WriteWithErrorCheck(f, &threads[i].distance[n].bits64, 32)) {
                return false;
            }
            kangaroos_written++;
        }
    }
    
    // Verify we wrote the expected number of kangaroos
    if (kangaroos_written != total_kangaroos) {
        std::cerr << "ERROR: Kangaroo count mismatch. Expected: " 
                  << total_kangaroos << ", Written: " << kangaroos_written << std::endl;
        return false;
    }
    
    std::cout << "Kangaroo states saved: " << nbThread << " threads, "
              << total_kangaroos << " total kangaroos" << std::endl;
    
    return true;
}

bool OptimizedCheckpoint::LoadKangaroosOptimized(FILE* f, TH_PARAM* threads, int& nbThread) {
    // Read kangaroo header
    KangarooHeader k_header;
    if (!ReadWithErrorCheck(f, &k_header, sizeof(k_header))) {
        return false;
    }
    
    // Validate kangaroo header
    if (k_header.magic != KANGAROO_MAGIC) {
        std::cerr << "ERROR: Invalid kangaroo magic number: 0x" 
                  << std::hex << k_header.magic << std::dec << std::endl;
        return false;
    }
    
    nbThread = static_cast<int>(k_header.thread_count);
    
    // Read thread information and kangaroo states
    uint64_t kangaroos_read = 0;
    for (int i = 0; i < nbThread; i++) {
        // Read thread metadata
        if (!ReadWithErrorCheck(f, &threads[i].nbKangaroo, sizeof(uint64_t))) {
            return false;
        }
        
        // Validate thread kangaroo count
        if (threads[i].nbKangaroo > 1000000) { // Reasonable upper limit
            std::cerr << "ERROR: Unreasonable kangaroo count for thread " << i 
                      << ": " << threads[i].nbKangaroo << std::endl;
            return false;
        }
        
        // Read kangaroo states for this thread
        for (uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
            // Read position (px, py) and distance
            if (!ReadWithErrorCheck(f, &threads[i].px[n].bits64, 32)) {
                return false;
            }
            if (!ReadWithErrorCheck(f, &threads[i].py[n].bits64, 32)) {
                return false;
            }
            if (!ReadWithErrorCheck(f, &threads[i].distance[n].bits64, 32)) {
                return false;
            }
            
            // Clear the 5th element for safety (as done in original code)
            threads[i].px[n].bits64[4] = 0;
            threads[i].py[n].bits64[4] = 0;
            threads[i].distance[n].bits64[4] = 0;
            
            kangaroos_read++;
        }
    }
    
    // Verify we read the expected number of kangaroos
    if (kangaroos_read != k_header.total_kangaroos) {
        std::cerr << "ERROR: Kangaroo count mismatch. Expected: " 
                  << k_header.total_kangaroos << ", Read: " << kangaroos_read << std::endl;
        return false;
    }
    
    std::cout << "Kangaroo states loaded: " << nbThread << " threads, "
              << k_header.total_kangaroos << " total kangaroos" << std::endl;
    
    return true;
}

// Legacy format detection
bool OptimizedCheckpoint::IsLegacyFormat(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;
    
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    fclose(f);
    
    // Check if it's a legacy format
    return (magic == HEADW || magic == HEADK || magic == HEADKS);
}

// Utility functions implementation
namespace CheckpointUtils {
    
    // Create directory recursively
    bool CreateDirectoryRecursive(const std::string& path) {
        if (path.empty()) return false;
        
        // Check if directory already exists
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            return S_ISDIR(st.st_mode);
        }
        
        // Find parent directory
        size_t pos = path.find_last_of(PATH_SEPARATOR);
        if (pos != std::string::npos) {
            std::string parent = path.substr(0, pos);
            if (!CreateDirectoryRecursive(parent)) {
                return false;
            }
        }
        
        // Create this directory
        return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
    }
    
    // Generate temporary filename
    std::string GetTempFilename(const std::string& base) {
        std::ostringstream oss;
        oss << base << ".tmp." << time(nullptr);
        return oss.str();
    }
    
    // Atomic file replacement
    bool AtomicFileReplace(const std::string& temp_file, const std::string& target_file) {
#ifdef _WIN32
        // On Windows, need to remove target first
        if (GetFileAttributesA(target_file.c_str()) != INVALID_FILE_ATTRIBUTES) {
            if (!DeleteFileA(target_file.c_str())) {
                std::cerr << "ERROR: Failed to remove target file: " << target_file << std::endl;
                return false;
            }
        }
        return MoveFileA(temp_file.c_str(), target_file.c_str()) != 0;
#else
        return rename(temp_file.c_str(), target_file.c_str()) == 0;
#endif
    }
    
    // Validate kangaroo state
    bool ValidateKangarooState(const TH_PARAM& thread) {
        // Basic validation - check for reasonable values
        if (thread.nbKangaroo > 10000000) { // 10M kangaroos per thread seems excessive
            return false;
        }
        
        // Could add more validation here (e.g., check if points are on curve)
        return true;
    }
    
    // Validate hash table integrity
    bool ValidateHashTableIntegrity(const HashTable& hashTable) {
        uint64_t total_items = 0;
        
        for (uint32_t h = 0; h < HASH_SIZE; h++) {
            if (hashTable.E[h].nbItem > hashTable.E[h].maxItem) {
                return false; // Invalid bucket state
            }
            total_items += hashTable.E[h].nbItem;
        }
        
        // Could add more validation here
        return true;
    }
    
    // Log checkpoint operation for performance monitoring
    void LogCheckpointOperation(const std::string& operation, double duration, uint64_t bytes) {
        double mb_per_sec = (bytes / (1024.0 * 1024.0)) / duration;
        std::cout << "Checkpoint " << operation << " performance: " 
                  << mb_per_sec << " MB/s" << std::endl;
    }
}
