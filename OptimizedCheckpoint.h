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

#ifndef OPTIMIZEDCHECKPOINT_H
#define OPTIMIZEDCHECKPOINT_H

#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "HashTable.h"
#include "Kangaroo.h"

// Cross-platform compatibility
#ifdef _WIN32
#include <io.h>
#define PATH_SEPARATOR '\\'
#define PATH_SEPARATOR_STR "\\"
#else
#include <unistd.h>
#define PATH_SEPARATOR '/'
#define PATH_SEPARATOR_STR "/"
#endif

// Optimized checkpoint format constants
#define HEADW_OPT 0xFA6A8004 // Optimized full work file
#define HEADK_OPT 0xFA6A8005 // Optimized kangaroo only file
#define CHECKPOINT_VERSION 2 // Current checkpoint format version (v2 with enhanced compression)

// Magic numbers for format validation
#define CHECKPOINT_MAGIC 0x4B4E4750 // "KNGP" (Kangaroo Checkpoint)
#define HASHTABLE_MAGIC 0x48544250  // "HTBP" (Hash Table Binary Packed)
#define KANGAROO_MAGIC 0x4B4E4752   // "KNGR" (Kangaroo)

// Checksum algorithm constants
#define CHECKSUM_SEED 0x12345678
#define CHECKSUM_MULTIPLIER 0x5DEECE66D
#define CHECKSUM_ADDEND 0xB

// Compression and optimization constants
#define MAX_COMPRESSION_RATIO 0.95f         // Maximum acceptable compression ratio
#define MIN_BUCKET_SIZE_FOR_COMPRESSION 100 // Minimum bucket size to apply compression
#define KANGAROO_BATCH_SIZE 1000            // Batch size for kangaroo state compression

// Error codes for checkpoint operations
enum CheckpointError
{
  CHECKPOINT_OK = 0,
  CHECKPOINT_FILE_ERROR = 1,
  CHECKPOINT_FORMAT_ERROR = 2,
  CHECKPOINT_CHECKSUM_ERROR = 3,
  CHECKPOINT_VERSION_ERROR = 4,
  CHECKPOINT_MEMORY_ERROR = 5,
  CHECKPOINT_VALIDATION_ERROR = 6
};

// Checkpoint file header structure (cross-platform compatible)
#pragma pack(push, 1)
struct CheckpointHeader
{
  uint32_t magic;           // Magic number for format validation
  uint32_t version;         // Checkpoint format version
  uint32_t file_type;       // File type (HEADW_OPT, HEADK_OPT, etc.)
  uint32_t flags;           // Feature flags (compression, encryption, etc.)
  uint64_t timestamp;       // Creation timestamp (Unix time)
  uint64_t total_count;     // Total operation count
  double total_time;        // Total elapsed time
  uint32_t dp_size;         // Distinguished point size
  uint32_t reserved1;       // Reserved for future use
  uint64_t reserved2;       // Reserved for future use
  uint64_t header_checksum; // Header checksum for validation
};
#pragma pack(pop)

// Hash table section header
#pragma pack(push, 1)
struct HashTableHeader
{
  uint32_t magic;             // Hash table magic number
  uint32_t non_empty_buckets; // Number of non-empty buckets
  uint64_t total_entries;     // Total number of entries
  uint32_t compression_type;  // Compression algorithm used (0=none, 1=sparse, 2=delta)
  uint32_t original_size;     // Original uncompressed size
  uint64_t compressed_size;   // Compressed data size
  uint64_t data_checksum;     // Data section checksum
};
#pragma pack(pop)

// Kangaroo section header
#pragma pack(push, 1)
struct KangarooHeader
{
  uint32_t magic;           // Kangaroo magic number
  uint32_t thread_count;    // Number of threads
  uint64_t total_kangaroos; // Total number of kangaroos
  uint32_t storage_type;    // Storage type (0=full, 1=compressed)
  uint32_t reserved;        // Reserved for alignment
  uint64_t data_checksum;   // Data section checksum
};
#pragma pack(pop)

// Optimized checkpoint manager class
class OptimizedCheckpoint
{
private:
  std::string filename;
  bool use_compression;
  bool validate_checksums;

  // Internal utility functions
  uint64_t CalculateChecksum(const void *data, size_t size);
  bool ValidateHeader(const CheckpointHeader &header);
  bool WriteWithErrorCheck(FILE *f, const void *data, size_t size);
  bool ReadWithErrorCheck(FILE *f, void *data, size_t size);

  // Endianness handling for cross-platform compatibility
  uint32_t SwapEndian32(uint32_t value);
  uint64_t SwapEndian64(uint64_t value);
  bool IsLittleEndian();
  void ConvertHeaderEndianness(CheckpointHeader &header);

  // Hash table optimization functions
  bool SaveHashTableOptimized(FILE *f, HashTable &hashTable);
  bool LoadHashTableOptimized(FILE *f, HashTable &hashTable);

  // Enhanced compression functions
  bool SaveHashTableCompressed(FILE *f, HashTable &hashTable);
  bool LoadHashTableCompressed(FILE *f, HashTable &hashTable);

  // Kangaroo state optimization functions
  bool SaveKangaroosOptimized(FILE *f, TH_PARAM *threads, int nbThread);
  bool LoadKangaroosOptimized(FILE *f, TH_PARAM *threads, int nbThread);

  // Enhanced kangaroo compression functions
  bool SaveKangaroosCompressed(FILE *f, TH_PARAM *threads, int nbThread);
  bool LoadKangaroosCompressed(FILE *f, TH_PARAM *threads, int nbThread);

  // Utility compression functions
  size_t CompressData(const void *input, size_t input_size, void **output);
  size_t DecompressData(const void *input, size_t input_size, void **output, size_t expected_size);
  bool ValidateCompressionRatio(size_t original_size, size_t compressed_size);

public:
  OptimizedCheckpoint(const std::string &filename, bool compression = false, bool checksums = true);
  ~OptimizedCheckpoint();

  // Main save/load functions with comprehensive error handling
  CheckpointError SaveCheckpoint(uint64_t totalCount, double totalTime,
                                 HashTable &hashTable, TH_PARAM *threads,
                                 int nbThread, uint32_t dpSize);

  CheckpointError LoadCheckpoint(uint64_t &totalCount, double &totalTime,
                                 HashTable &hashTable, TH_PARAM *threads,
                                 int &nbThread, uint32_t &dpSize);

  // Legacy compatibility functions
  bool IsLegacyFormat(const std::string &filename);
  CheckpointError ConvertLegacyCheckpoint(const std::string &legacyFile,
                                          const std::string &optimizedFile);

  // Utility functions
  bool FileExists(const std::string &filename);
  uint64_t GetFileSize(const std::string &filename);
  std::string GetErrorMessage(CheckpointError error);

  // Configuration functions
  void SetCompression(bool enable) { use_compression = enable; }
  void SetChecksumValidation(bool enable) { validate_checksums = enable; }
  bool GetCompression() const { return use_compression; }
  bool GetChecksumValidation() const { return validate_checksums; }
};

// Global utility functions for checkpoint operations
namespace CheckpointUtils
{
  // Cross-platform file operations
  bool CreateDirectoryRecursive(const std::string &path);
  std::string GetTempFilename(const std::string &base);
  bool AtomicFileReplace(const std::string &temp_file, const std::string &target_file);

  // Validation functions
  bool ValidateKangarooState(const TH_PARAM &thread);
  bool ValidateHashTableIntegrity(const HashTable &hashTable);

  // Performance monitoring
  void LogCheckpointOperation(const std::string &operation, double duration, uint64_t bytes);
}

#endif // OPTIMIZEDCHECKPOINT_H
