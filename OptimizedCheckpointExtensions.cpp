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
#include <iomanip>
#include <vector>
#include <cstring>

// Enhanced compressed kangaroo save function
bool OptimizedCheckpoint::SaveKangaroosCompressed(FILE* f, TH_PARAM* threads, int nbThread) {
    std::cout << "Saving kangaroo states with enhanced compression..." << std::endl;
    
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
    k_header.storage_type = 1; // Compressed storage
    
    // Calculate original size
    size_t original_size = sizeof(uint64_t) * nbThread; // Thread kangaroo counts
    original_size += total_kangaroos * (32 + 32 + 32); // px, py, distance (96 bytes each)
    
    // Create temporary buffer for compression
    std::vector<uint8_t> temp_buffer;
    temp_buffer.reserve(original_size);
    
    // Serialize kangaroo data to buffer
    for (int i = 0; i < nbThread; i++) {
        // Add thread kangaroo count
        temp_buffer.insert(temp_buffer.end(), 
                          reinterpret_cast<const uint8_t*>(&threads[i].nbKangaroo), 
                          reinterpret_cast<const uint8_t*>(&threads[i].nbKangaroo) + sizeof(uint64_t));
        
        // Add kangaroo states for this thread
        for (uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
            // Add position (px, py) and distance
            temp_buffer.insert(temp_buffer.end(), 
                              reinterpret_cast<const uint8_t*>(&threads[i].px[n].bits64), 
                              reinterpret_cast<const uint8_t*>(&threads[i].px[n].bits64) + 32);
            temp_buffer.insert(temp_buffer.end(), 
                              reinterpret_cast<const uint8_t*>(&threads[i].py[n].bits64), 
                              reinterpret_cast<const uint8_t*>(&threads[i].py[n].bits64) + 32);
            temp_buffer.insert(temp_buffer.end(), 
                              reinterpret_cast<const uint8_t*>(&threads[i].distance[n].bits64), 
                              reinterpret_cast<const uint8_t*>(&threads[i].distance[n].bits64) + 32);
        }
    }
    
    // Apply compression using delta encoding and run-length encoding
    std::vector<uint8_t> compressed_buffer;
    compressed_buffer.reserve(temp_buffer.size());
    
    // Simple compression: delta encoding for similar values + RLE for zeros
    uint8_t prev_byte = 0;
    for (size_t i = 0; i < temp_buffer.size(); ) {
        if (temp_buffer[i] == 0) {
            // Count consecutive zeros
            size_t zero_count = 0;
            while (i + zero_count < temp_buffer.size() && 
                   temp_buffer[i + zero_count] == 0 && 
                   zero_count < 255) {
                zero_count++;
            }
            
            if (zero_count >= 4) { // Only compress if we have 4+ zeros
                compressed_buffer.push_back(0xFE); // Escape byte for zeros
                compressed_buffer.push_back(static_cast<uint8_t>(zero_count));
                i += zero_count;
            } else {
                compressed_buffer.push_back(temp_buffer[i]);
                i++;
            }
        } else {
            // Delta encoding for non-zero bytes
            int16_t delta = static_cast<int16_t>(temp_buffer[i]) - static_cast<int16_t>(prev_byte);
            if (delta >= -127 && delta <= 127 && delta != -2) { // Avoid conflict with escape
                compressed_buffer.push_back(0xFF); // Escape byte for delta
                compressed_buffer.push_back(static_cast<uint8_t>(delta + 128)); // Offset by 128
            } else {
                compressed_buffer.push_back(temp_buffer[i]);
            }
            prev_byte = temp_buffer[i];
            i++;
        }
    }
    
    // Update header with compression info
    k_header.data_checksum = compressed_buffer.size();
    
    // Validate compression ratio
    if (!ValidateCompressionRatio(temp_buffer.size(), compressed_buffer.size())) {
        std::cout << "Compression ratio not beneficial, using uncompressed format" << std::endl;
        return SaveKangaroosOptimized(f, threads, nbThread);
    }
    
    // Calculate checksum
    if (validate_checksums) {
        k_header.data_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
    }
    
    // Write kangaroo header
    if (!WriteWithErrorCheck(f, &k_header, sizeof(k_header))) {
        return false;
    }
    
    // Write original size
    uint32_t orig_size = static_cast<uint32_t>(temp_buffer.size());
    if (!WriteWithErrorCheck(f, &orig_size, sizeof(orig_size))) {
        return false;
    }
    
    // Write compressed size
    uint32_t comp_size = static_cast<uint32_t>(compressed_buffer.size());
    if (!WriteWithErrorCheck(f, &comp_size, sizeof(comp_size))) {
        return false;
    }
    
    // Write compressed data
    if (!WriteWithErrorCheck(f, compressed_buffer.data(), compressed_buffer.size())) {
        return false;
    }
    
    double compression_ratio = static_cast<double>(compressed_buffer.size()) / temp_buffer.size();
    std::cout << "Compressed kangaroo states saved: " << nbThread << " threads, "
              << total_kangaroos << " kangaroos, compression ratio: " 
              << std::fixed << std::setprecision(3) << compression_ratio << std::endl;
    
    return true;
}

// Enhanced compressed kangaroo load function
bool OptimizedCheckpoint::LoadKangaroosCompressed(FILE* f, TH_PARAM* threads, int& nbThread) {
    std::cout << "Loading compressed kangaroo states..." << std::endl;
    
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
    
    // Read size information
    uint32_t original_size, compressed_size;
    if (!ReadWithErrorCheck(f, &original_size, sizeof(original_size))) {
        return false;
    }
    if (!ReadWithErrorCheck(f, &compressed_size, sizeof(compressed_size))) {
        return false;
    }
    
    // Read compressed data
    std::vector<uint8_t> compressed_buffer(compressed_size);
    if (!ReadWithErrorCheck(f, compressed_buffer.data(), compressed_size)) {
        return false;
    }
    
    // Validate checksum if enabled
    if (validate_checksums && k_header.data_checksum != 0) {
        uint64_t calculated_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
        if (calculated_checksum != k_header.data_checksum) {
            std::cerr << "ERROR: Kangaroo data checksum mismatch" << std::endl;
            return false;
        }
    }
    
    // Decompress data
    std::vector<uint8_t> decompressed_buffer;
    decompressed_buffer.reserve(original_size);
    
    // Decompress using reverse delta encoding and RLE
    uint8_t prev_byte = 0;
    for (size_t i = 0; i < compressed_buffer.size(); i++) {
        if (compressed_buffer[i] == 0xFE && i + 1 < compressed_buffer.size()) {
            // Zero run-length decoding
            uint8_t zero_count = compressed_buffer[i + 1];
            for (uint8_t j = 0; j < zero_count; j++) {
                decompressed_buffer.push_back(0);
            }
            i++; // Skip the count byte
        } else if (compressed_buffer[i] == 0xFF && i + 1 < compressed_buffer.size()) {
            // Delta decoding
            int16_t delta = static_cast<int16_t>(compressed_buffer[i + 1]) - 128;
            uint8_t decoded_byte = static_cast<uint8_t>(static_cast<int16_t>(prev_byte) + delta);
            decompressed_buffer.push_back(decoded_byte);
            prev_byte = decoded_byte;
            i++; // Skip the delta byte
        } else {
            decompressed_buffer.push_back(compressed_buffer[i]);
            prev_byte = compressed_buffer[i];
        }
    }
    
    // Validate decompressed size
    if (decompressed_buffer.size() != original_size) {
        std::cerr << "ERROR: Decompressed size mismatch. Expected: " << original_size 
                  << ", Got: " << decompressed_buffer.size() << std::endl;
        return false;
    }
    
    // Parse decompressed kangaroo data
    size_t offset = 0;
    uint64_t kangaroos_read = 0;
    
    for (int i = 0; i < nbThread; i++) {
        if (offset + sizeof(uint64_t) > decompressed_buffer.size()) {
            std::cerr << "ERROR: Insufficient data for thread " << i << " kangaroo count" << std::endl;
            return false;
        }
        
        // Read thread kangaroo count
        threads[i].nbKangaroo = *reinterpret_cast<const uint64_t*>(decompressed_buffer.data() + offset);
        offset += sizeof(uint64_t);
        
        // Read kangaroo states for this thread
        for (uint64_t n = 0; n < threads[i].nbKangaroo; n++) {
            if (offset + 96 > decompressed_buffer.size()) {
                std::cerr << "ERROR: Insufficient data for kangaroo " << n << " in thread " << i << std::endl;
                return false;
            }
            
            // Read position (px, py) and distance
            memcpy(&threads[i].px[n].bits64, decompressed_buffer.data() + offset, 32);
            offset += 32;
            memcpy(&threads[i].py[n].bits64, decompressed_buffer.data() + offset, 32);
            offset += 32;
            memcpy(&threads[i].distance[n].bits64, decompressed_buffer.data() + offset, 32);
            offset += 32;
            
            kangaroos_read++;
        }
    }
    
    // Verify we read the expected number of kangaroos
    if (kangaroos_read != k_header.total_kangaroos) {
        std::cerr << "ERROR: Kangaroo count mismatch. Expected: " 
                  << k_header.total_kangaroos << ", Read: " << kangaroos_read << std::endl;
        return false;
    }
    
    double compression_ratio = static_cast<double>(compressed_size) / original_size;
    std::cout << "Compressed kangaroo states loaded: " << nbThread << " threads, "
              << k_header.total_kangaroos << " kangaroos, compression ratio: " 
              << std::fixed << std::setprecision(3) << compression_ratio << std::endl;
    
    return true;
}
