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
#include <cstring>
#include <ctime>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#endif

// Constructor
OptimizedCheckpoint::OptimizedCheckpoint(const std::string &filename, bool compression, bool checksums)
    : filename(filename), use_compression(compression), validate_checksums(checksums)
{
}

// Destructor
OptimizedCheckpoint::~OptimizedCheckpoint()
{
  // Nothing to clean up - using RAII principles
}

// Calculate checksum using a simple but effective algorithm
uint64_t OptimizedCheckpoint::CalculateChecksum(const void *data, size_t size)
{
  if (!data || size == 0)
    return 0;

  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  uint64_t checksum = CHECKSUM_SEED;

  for (size_t i = 0; i < size; i++)
  {
    checksum = (checksum * CHECKSUM_MULTIPLIER + bytes[i] + CHECKSUM_ADDEND) & 0xFFFFFFFFFFFFFFFF;
  }

  return checksum;
}

// Validate checkpoint header
bool OptimizedCheckpoint::ValidateHeader(const CheckpointHeader &header)
{
  // Check magic number
  if (header.magic != CHECKPOINT_MAGIC)
  {
    std::cerr << "ERROR: Invalid checkpoint magic number: 0x"
              << std::hex << header.magic << std::dec << std::endl;
    return false;
  }

  // Check version compatibility - support both v1 and v2
  if (header.version > CHECKPOINT_VERSION)
  {
    std::cerr << "ERROR: Unsupported checkpoint version: " << header.version
              << " (max supported: " << CHECKPOINT_VERSION << ")" << std::endl;
    return false;
  }

  // Version 1 compatibility mode
  if (header.version == 1)
  {
    std::cout << "Loading legacy v1 checkpoint format" << std::endl;
  }

  // Check file type
  if (header.file_type != HEADW_OPT && header.file_type != HEADK_OPT)
  {
    std::cerr << "ERROR: Invalid file type: 0x"
              << std::hex << header.file_type << std::dec << std::endl;
    return false;
  }

  // Validate header checksum if enabled
  if (validate_checksums)
  {
    CheckpointHeader temp_header = header;
    uint64_t stored_checksum = temp_header.header_checksum;
    temp_header.header_checksum = 0;

    uint64_t calculated_checksum = CalculateChecksum(&temp_header, sizeof(temp_header));
    if (stored_checksum != calculated_checksum)
    {
      std::cerr << "ERROR: Header checksum mismatch. Expected: 0x"
                << std::hex << calculated_checksum
                << ", Got: 0x" << stored_checksum << std::dec << std::endl;
      return false;
    }
  }

  return true;
}

// Write data with comprehensive error checking
bool OptimizedCheckpoint::WriteWithErrorCheck(FILE *f, const void *data, size_t size)
{
  if (!f || !data || size == 0)
  {
    std::cerr << "ERROR: Invalid parameters for write operation" << std::endl;
    return false;
  }

  size_t written = fwrite(data, 1, size, f);
  if (written != size)
  {
    std::cerr << "ERROR: Write operation failed. Expected: " << size
              << " bytes, Written: " << written << " bytes" << std::endl;
    if (ferror(f))
    {
      std::cerr << "ERROR: File write error: " << strerror(errno) << std::endl;
    }
    return false;
  }

  return true;
}

// Read data with comprehensive error checking
bool OptimizedCheckpoint::ReadWithErrorCheck(FILE *f, void *data, size_t size)
{
  if (!f || !data || size == 0)
  {
    std::cerr << "ERROR: Invalid parameters for read operation" << std::endl;
    return false;
  }

  size_t read_bytes = fread(data, 1, size, f);
  if (read_bytes != size)
  {
    if (feof(f))
    {
      std::cerr << "ERROR: Unexpected end of file. Expected: " << size
                << " bytes, Read: " << read_bytes << " bytes" << std::endl;
    }
    else if (ferror(f))
    {
      std::cerr << "ERROR: File read error: " << strerror(errno) << std::endl;
    }
    else
    {
      std::cerr << "ERROR: Incomplete read. Expected: " << size
                << " bytes, Read: " << read_bytes << " bytes" << std::endl;
    }
    return false;
  }

  return true;
}

// Endianness handling functions
uint32_t OptimizedCheckpoint::SwapEndian32(uint32_t value)
{
  return ((value & 0xFF000000) >> 24) |
         ((value & 0x00FF0000) >> 8) |
         ((value & 0x0000FF00) << 8) |
         ((value & 0x000000FF) << 24);
}

uint64_t OptimizedCheckpoint::SwapEndian64(uint64_t value)
{
  return ((value & 0xFF00000000000000ULL) >> 56) |
         ((value & 0x00FF000000000000ULL) >> 40) |
         ((value & 0x0000FF0000000000ULL) >> 24) |
         ((value & 0x000000FF00000000ULL) >> 8) |
         ((value & 0x00000000FF000000ULL) << 8) |
         ((value & 0x0000000000FF0000ULL) << 24) |
         ((value & 0x000000000000FF00ULL) << 40) |
         ((value & 0x00000000000000FFULL) << 56);
}

bool OptimizedCheckpoint::IsLittleEndian()
{
  uint32_t test = 0x12345678;
  uint8_t *bytes = reinterpret_cast<uint8_t *>(&test);
  return bytes[0] == 0x78;
}

void OptimizedCheckpoint::ConvertHeaderEndianness(CheckpointHeader &header)
{
  if (!IsLittleEndian())
  {
    // Convert to little endian for cross-platform compatibility
    header.magic = SwapEndian32(header.magic);
    header.version = SwapEndian32(header.version);
    header.file_type = SwapEndian32(header.file_type);
    header.flags = SwapEndian32(header.flags);
    header.timestamp = SwapEndian64(header.timestamp);
    header.total_count = SwapEndian64(header.total_count);
    header.dp_size = SwapEndian32(header.dp_size);
    header.reserved1 = SwapEndian32(header.reserved1);
    header.reserved2 = SwapEndian64(header.reserved2);
    header.header_checksum = SwapEndian64(header.header_checksum);
    // Note: total_time (double) needs special handling for endianness
  }
}

// Check if file exists
bool OptimizedCheckpoint::FileExists(const std::string &filename)
{
  FILE *f = fopen(filename.c_str(), "rb");
  if (f)
  {
    fclose(f);
    return true;
  }
  return false;
}

// Get file size
uint64_t OptimizedCheckpoint::GetFileSize(const std::string &filename)
{
  FILE *f = fopen(filename.c_str(), "rb");
  if (!f)
    return 0;

  if (fseek(f, 0, SEEK_END) != 0)
  {
    fclose(f);
    return 0;
  }

  long size = ftell(f);
  fclose(f);

  return (size < 0) ? 0 : static_cast<uint64_t>(size);
}

// Get error message for error code
std::string OptimizedCheckpoint::GetErrorMessage(CheckpointError error)
{
  switch (error)
  {
  case CHECKPOINT_OK:
    return "Operation completed successfully";
  case CHECKPOINT_FILE_ERROR:
    return "File I/O error occurred";
  case CHECKPOINT_FORMAT_ERROR:
    return "Invalid checkpoint file format";
  case CHECKPOINT_CHECKSUM_ERROR:
    return "Checksum validation failed";
  case CHECKPOINT_VERSION_ERROR:
    return "Unsupported checkpoint version";
  case CHECKPOINT_MEMORY_ERROR:
    return "Memory allocation error";
  case CHECKPOINT_VALIDATION_ERROR:
    return "Data validation error";
  default:
    return "Unknown error";
  }
}

// Main save checkpoint function with comprehensive error handling
CheckpointError OptimizedCheckpoint::SaveCheckpoint(uint64_t totalCount, double totalTime,
                                                    HashTable &hashTable, TH_PARAM *threads,
                                                    int nbThread, uint32_t dpSize)
{
  double start_time = Timer::get_tick();

  std::cout << "Saving optimized checkpoint: " << filename << std::endl;

  // Create temporary file for atomic operation
  std::string temp_filename = CheckpointUtils::GetTempFilename(filename);

  FILE *f = fopen(temp_filename.c_str(), "wb");
  if (!f)
  {
    std::cerr << "ERROR: Cannot create checkpoint file: " << temp_filename
              << " - " << strerror(errno) << std::endl;
    return CHECKPOINT_FILE_ERROR;
  }

  // Prepare checkpoint header
  CheckpointHeader header;
  memset(&header, 0, sizeof(header));

  header.magic = CHECKPOINT_MAGIC;
  header.version = CHECKPOINT_VERSION;
  header.file_type = HEADW_OPT;
  header.flags = use_compression ? 1 : 0;
  header.timestamp = static_cast<uint64_t>(time(nullptr));
  header.total_count = totalCount;
  header.total_time = totalTime;
  header.dp_size = dpSize;

  // Calculate header checksum
  if (validate_checksums)
  {
    header.header_checksum = CalculateChecksum(&header, sizeof(header) - sizeof(header.header_checksum));
  }

  // Convert endianness for cross-platform compatibility
  ConvertHeaderEndianness(header);

  // Write header
  if (!WriteWithErrorCheck(f, &header, sizeof(header)))
  {
    fclose(f);
    remove(temp_filename.c_str());
    return CHECKPOINT_FILE_ERROR;
  }

  // Save hash table with enhanced compression
  bool hash_save_success = false;
  if (use_compression)
  {
    hash_save_success = SaveHashTableCompressed(f, hashTable);
    if (!hash_save_success)
    {
      std::cout << "Compressed hash table save failed, falling back to optimized format" << std::endl;
      hash_save_success = SaveHashTableOptimized(f, hashTable);
    }
  }
  else
  {
    hash_save_success = SaveHashTableOptimized(f, hashTable);
  }

  if (!hash_save_success)
  {
    fclose(f);
    remove(temp_filename.c_str());
    return CHECKPOINT_FILE_ERROR;
  }

  // Save kangaroo states with enhanced compression
  bool kangaroo_save_success = false;
  if (use_compression)
  {
    kangaroo_save_success = SaveKangaroosCompressed(f, threads, nbThread);
    if (!kangaroo_save_success)
    {
      std::cout << "Compressed kangaroo save failed, falling back to optimized format" << std::endl;
      kangaroo_save_success = SaveKangaroosOptimized(f, threads, nbThread);
    }
  }
  else
  {
    kangaroo_save_success = SaveKangaroosOptimized(f, threads, nbThread);
  }

  if (!kangaroo_save_success)
  {
    fclose(f);
    remove(temp_filename.c_str());
    return CHECKPOINT_FILE_ERROR;
  }

  // Flush and close file
  if (fflush(f) != 0)
  {
    std::cerr << "ERROR: Failed to flush checkpoint file" << std::endl;
    fclose(f);
    remove(temp_filename.c_str());
    return CHECKPOINT_FILE_ERROR;
  }

  fclose(f);

  // Atomic file replacement
  if (!CheckpointUtils::AtomicFileReplace(temp_filename, filename))
  {
    remove(temp_filename.c_str());
    return CHECKPOINT_FILE_ERROR;
  }

  double end_time = Timer::get_tick();
  uint64_t file_size = GetFileSize(filename);

  std::cout << "Checkpoint saved successfully: " << filename
            << " [" << (file_size / (1024.0 * 1024.0)) << " MB]"
            << " [" << (end_time - start_time) << "s]" << std::endl;

  CheckpointUtils::LogCheckpointOperation("save", end_time - start_time, file_size);

  return CHECKPOINT_OK;
}

// Main load checkpoint function with comprehensive error handling
CheckpointError OptimizedCheckpoint::LoadCheckpoint(uint64_t &totalCount, double &totalTime,
                                                    HashTable &hashTable, TH_PARAM *threads,
                                                    int &nbThread, uint32_t &dpSize)
{
  double start_time = Timer::get_tick();

  std::cout << "Loading optimized checkpoint: " << filename << std::endl;

  if (!FileExists(filename))
  {
    std::cerr << "ERROR: Checkpoint file does not exist: " << filename << std::endl;
    return CHECKPOINT_FILE_ERROR;
  }

  FILE *f = fopen(filename.c_str(), "rb");
  if (!f)
  {
    std::cerr << "ERROR: Cannot open checkpoint file: " << filename
              << " - " << strerror(errno) << std::endl;
    return CHECKPOINT_FILE_ERROR;
  }

  // Read and validate header
  CheckpointHeader header;
  if (!ReadWithErrorCheck(f, &header, sizeof(header)))
  {
    fclose(f);
    return CHECKPOINT_FILE_ERROR;
  }

  // Convert endianness
  ConvertHeaderEndianness(header);

  // Validate header
  if (!ValidateHeader(header))
  {
    fclose(f);
    return CHECKPOINT_FORMAT_ERROR;
  }

  // Extract header information
  totalCount = header.total_count;
  totalTime = header.total_time;
  dpSize = header.dp_size;

  // Load hash table with format detection
  bool hash_load_success = false;
  if (header.version >= 2 && (header.flags & 1)) // Version 2+ with compression flag
  {
    hash_load_success = LoadHashTableCompressed(f, hashTable);
    if (!hash_load_success)
    {
      std::cout << "Compressed hash table load failed, trying optimized format" << std::endl;
      // Reset file position and try optimized format
      fseek(f, sizeof(CheckpointHeader), SEEK_SET);
      hash_load_success = LoadHashTableOptimized(f, hashTable);
    }
  }
  else
  {
    hash_load_success = LoadHashTableOptimized(f, hashTable);
  }

  if (!hash_load_success)
  {
    fclose(f);
    return CHECKPOINT_FILE_ERROR;
  }

  // Load kangaroo states with format detection
  bool kangaroo_load_success = false;
  if (header.version >= 2 && (header.flags & 1)) // Version 2+ with compression flag
  {
    kangaroo_load_success = LoadKangaroosCompressed(f, threads, nbThread);
    if (!kangaroo_load_success)
    {
      std::cout << "Compressed kangaroo load failed, trying optimized format" << std::endl;
      kangaroo_load_success = LoadKangaroosOptimized(f, threads, nbThread);
    }
  }
  else
  {
    kangaroo_load_success = LoadKangaroosOptimized(f, threads, nbThread);
  }

  if (!kangaroo_load_success)
  {
    fclose(f);
    return CHECKPOINT_FILE_ERROR;
  }

  fclose(f);

  double end_time = Timer::get_tick();
  uint64_t file_size = GetFileSize(filename);

  std::cout << "Checkpoint loaded successfully: " << filename
            << " [" << (file_size / (1024.0 * 1024.0)) << " MB]"
            << " [" << (end_time - start_time) << "s]" << std::endl;

  CheckpointUtils::LogCheckpointOperation("load", end_time - start_time, file_size);

  return CHECKPOINT_OK;
}

// Optimized hash table save function - only saves non-empty buckets
bool OptimizedCheckpoint::SaveHashTableOptimized(FILE *f, HashTable &hashTable)
{
  // Count non-empty buckets first
  uint32_t non_empty_buckets = 0;
  uint64_t total_entries = 0;

  for (uint32_t h = 0; h < HASH_SIZE; h++)
  {
    if (hashTable.E[h].nbItem > 0)
    {
      non_empty_buckets++;
      total_entries += hashTable.E[h].nbItem;
    }
  }

  // Prepare hash table header
  HashTableHeader ht_header;
  memset(&ht_header, 0, sizeof(ht_header));

  ht_header.magic = HASHTABLE_MAGIC;
  ht_header.non_empty_buckets = non_empty_buckets;
  ht_header.total_entries = total_entries;
  ht_header.compression_type = 0; // No compression for now

  // Write hash table header
  if (!WriteWithErrorCheck(f, &ht_header, sizeof(ht_header)))
  {
    return false;
  }

  // Write non-empty buckets data
  uint64_t entries_written = 0;
  for (uint32_t h = 0; h < HASH_SIZE; h++)
  {
    if (hashTable.E[h].nbItem > 0)
    {
      // Write bucket index
      if (!WriteWithErrorCheck(f, &h, sizeof(uint32_t)))
      {
        return false;
      }

      // Write bucket metadata
      if (!WriteWithErrorCheck(f, &hashTable.E[h].nbItem, sizeof(uint32_t)))
      {
        return false;
      }
      if (!WriteWithErrorCheck(f, &hashTable.E[h].maxItem, sizeof(uint32_t)))
      {
        return false;
      }

      // Write bucket entries
      for (uint32_t i = 0; i < hashTable.E[h].nbItem; i++)
      {
        if (!WriteWithErrorCheck(f, &(hashTable.E[h].items[i]->x), 16))
        {
          return false;
        }
        if (!WriteWithErrorCheck(f, &(hashTable.E[h].items[i]->d), 16))
        {
          return false;
        }
        entries_written++;
      }
    }
  }

  // Verify we wrote the expected number of entries
  if (entries_written != total_entries)
  {
    std::cerr << "ERROR: Hash table entry count mismatch. Expected: "
              << total_entries << ", Written: " << entries_written << std::endl;
    return false;
  }

  std::cout << "Hash table saved: " << non_empty_buckets << " non-empty buckets, "
            << total_entries << " total entries" << std::endl;

  return true;
}

// Optimized hash table load function - loads only non-empty buckets
bool OptimizedCheckpoint::LoadHashTableOptimized(FILE *f, HashTable &hashTable)
{
  // Read hash table header
  HashTableHeader ht_header;
  if (!ReadWithErrorCheck(f, &ht_header, sizeof(ht_header)))
  {
    return false;
  }

  // Validate hash table header
  if (ht_header.magic != HASHTABLE_MAGIC)
  {
    std::cerr << "ERROR: Invalid hash table magic number: 0x"
              << std::hex << ht_header.magic << std::dec << std::endl;
    return false;
  }

  // Reset hash table
  hashTable.Reset();

  // Read non-empty buckets
  uint64_t entries_read = 0;
  for (uint32_t bucket_idx = 0; bucket_idx < ht_header.non_empty_buckets; bucket_idx++)
  {
    // Read bucket index
    uint32_t h;
    if (!ReadWithErrorCheck(f, &h, sizeof(uint32_t)))
    {
      return false;
    }

    // Validate bucket index
    if (h >= HASH_SIZE)
    {
      std::cerr << "ERROR: Invalid bucket index: " << h
                << " (max: " << (HASH_SIZE - 1) << ")" << std::endl;
      return false;
    }

    // Read bucket metadata
    if (!ReadWithErrorCheck(f, &hashTable.E[h].nbItem, sizeof(uint32_t)))
    {
      return false;
    }
    if (!ReadWithErrorCheck(f, &hashTable.E[h].maxItem, sizeof(uint32_t)))
    {
      return false;
    }

    // Validate bucket metadata
    if (hashTable.E[h].nbItem > hashTable.E[h].maxItem)
    {
      std::cerr << "ERROR: Invalid bucket metadata. nbItem: "
                << hashTable.E[h].nbItem << ", maxItem: "
                << hashTable.E[h].maxItem << std::endl;
      return false;
    }

    // Allocate bucket items array
    if (hashTable.E[h].maxItem > 0)
    {
      hashTable.E[h].items = (ENTRY **)malloc(sizeof(ENTRY *) * hashTable.E[h].maxItem);
      if (!hashTable.E[h].items)
      {
        std::cerr << "ERROR: Failed to allocate memory for bucket " << h << std::endl;
        return false;
      }
    }

    // Read bucket entries
    for (uint32_t i = 0; i < hashTable.E[h].nbItem; i++)
    {
      ENTRY *e = (ENTRY *)malloc(sizeof(ENTRY));
      if (!e)
      {
        std::cerr << "ERROR: Failed to allocate memory for entry" << std::endl;
        return false;
      }

      if (!ReadWithErrorCheck(f, &(e->x), 16))
      {
        free(e);
        return false;
      }
      if (!ReadWithErrorCheck(f, &(e->d), 16))
      {
        free(e);
        return false;
      }

      hashTable.E[h].items[i] = e;
      entries_read++;
    }
  }

  // Verify we read the expected number of entries
  if (entries_read != ht_header.total_entries)
  {
    std::cerr << "ERROR: Hash table entry count mismatch. Expected: "
              << ht_header.total_entries << ", Read: " << entries_read << std::endl;
    return false;
  }

  std::cout << "Hash table loaded: " << ht_header.non_empty_buckets
            << " non-empty buckets, " << ht_header.total_entries
            << " total entries" << std::endl;

  return true;
}

// Enhanced compressed hash table save function
bool OptimizedCheckpoint::SaveHashTableCompressed(FILE *f, HashTable &hashTable)
{
  std::cout << "Saving hash table with enhanced compression..." << std::endl;

  // Count non-empty buckets and calculate total size
  uint32_t non_empty_buckets = 0;
  uint64_t total_entries = 0;
  size_t estimated_size = 0;

  for (uint32_t h = 0; h < HASH_SIZE; h++)
  {
    if (hashTable.E[h].nbItem > 0)
    {
      non_empty_buckets++;
      total_entries += hashTable.E[h].nbItem;
      estimated_size += sizeof(uint32_t) * 3;       // bucket index + metadata
      estimated_size += hashTable.E[h].nbItem * 32; // entries (16+16 bytes each)
    }
  }

  // Prepare enhanced hash table header
  HashTableHeader ht_header;
  memset(&ht_header, 0, sizeof(ht_header));

  ht_header.magic = HASHTABLE_MAGIC;
  ht_header.non_empty_buckets = non_empty_buckets;
  ht_header.total_entries = total_entries;
  ht_header.original_size = static_cast<uint32_t>(estimated_size);
  ht_header.compression_type = 1; // Sparse compression

  // Create temporary buffer for compression
  std::vector<uint8_t> temp_buffer;
  temp_buffer.reserve(estimated_size);

  // Serialize hash table data to buffer
  for (uint32_t h = 0; h < HASH_SIZE; h++)
  {
    if (hashTable.E[h].nbItem > 0)
    {
      // Add bucket index
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&h),
                         reinterpret_cast<const uint8_t *>(&h) + sizeof(uint32_t));

      // Add bucket metadata
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&hashTable.E[h].nbItem),
                         reinterpret_cast<const uint8_t *>(&hashTable.E[h].nbItem) + sizeof(uint32_t));
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&hashTable.E[h].maxItem),
                         reinterpret_cast<const uint8_t *>(&hashTable.E[h].maxItem) + sizeof(uint32_t));

      // Add bucket entries
      for (uint32_t i = 0; i < hashTable.E[h].nbItem; i++)
      {
        temp_buffer.insert(temp_buffer.end(),
                           reinterpret_cast<const uint8_t *>(&hashTable.E[h].items[i]->x),
                           reinterpret_cast<const uint8_t *>(&hashTable.E[h].items[i]->x) + 16);
        temp_buffer.insert(temp_buffer.end(),
                           reinterpret_cast<const uint8_t *>(&hashTable.E[h].items[i]->d),
                           reinterpret_cast<const uint8_t *>(&hashTable.E[h].items[i]->d) + 16);
      }
    }
  }

  // Apply simple compression (remove redundant zeros and use delta encoding)
  std::vector<uint8_t> compressed_buffer;
  compressed_buffer.reserve(temp_buffer.size());

  // Simple run-length encoding for zero bytes
  for (size_t i = 0; i < temp_buffer.size();)
  {
    if (temp_buffer[i] == 0)
    {
      // Count consecutive zeros
      size_t zero_count = 0;
      while (i + zero_count < temp_buffer.size() &&
             temp_buffer[i + zero_count] == 0 &&
             zero_count < 255)
      {
        zero_count++;
      }

      if (zero_count >= 3) // Only compress if we have 3+ zeros
      {
        compressed_buffer.push_back(0xFF); // Escape byte
        compressed_buffer.push_back(static_cast<uint8_t>(zero_count));
        i += zero_count;
      }
      else
      {
        compressed_buffer.push_back(temp_buffer[i]);
        i++;
      }
    }
    else
    {
      compressed_buffer.push_back(temp_buffer[i]);
      i++;
    }
  }

  ht_header.compressed_size = compressed_buffer.size();

  // Validate compression ratio
  if (!ValidateCompressionRatio(temp_buffer.size(), compressed_buffer.size()))
  {
    std::cout << "Compression ratio not beneficial, using uncompressed format" << std::endl;
    return SaveHashTableOptimized(f, hashTable);
  }

  // Calculate checksum
  if (validate_checksums)
  {
    ht_header.data_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
  }

  // Write header
  if (!WriteWithErrorCheck(f, &ht_header, sizeof(ht_header)))
  {
    return false;
  }

  // Write compressed data
  if (!WriteWithErrorCheck(f, compressed_buffer.data(), compressed_buffer.size()))
  {
    return false;
  }

  double compression_ratio = static_cast<double>(compressed_buffer.size()) / temp_buffer.size();
  std::cout << "Compressed hash table saved: " << non_empty_buckets << " buckets, "
            << total_entries << " entries, compression ratio: "
            << std::fixed << std::setprecision(3) << compression_ratio << std::endl;

  return true;
}

// Enhanced compressed hash table load function
bool OptimizedCheckpoint::LoadHashTableCompressed(FILE *f, HashTable &hashTable)
{
  std::cout << "Loading compressed hash table..." << std::endl;

  // Read hash table header
  HashTableHeader ht_header;
  if (!ReadWithErrorCheck(f, &ht_header, sizeof(ht_header)))
  {
    return false;
  }

  // Validate hash table header
  if (ht_header.magic != HASHTABLE_MAGIC)
  {
    std::cerr << "ERROR: Invalid hash table magic number: 0x"
              << std::hex << ht_header.magic << std::dec << std::endl;
    return false;
  }

  // Reset hash table
  hashTable.Reset();

  // Read compressed data
  std::vector<uint8_t> compressed_buffer(ht_header.compressed_size);
  if (!ReadWithErrorCheck(f, compressed_buffer.data(), ht_header.compressed_size))
  {
    return false;
  }

  // Validate checksum if enabled
  if (validate_checksums && ht_header.data_checksum != 0)
  {
    uint64_t calculated_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
    if (calculated_checksum != ht_header.data_checksum)
    {
      std::cerr << "ERROR: Hash table data checksum mismatch" << std::endl;
      return false;
    }
  }

  // Decompress data
  std::vector<uint8_t> decompressed_buffer;
  decompressed_buffer.reserve(ht_header.original_size);

  // Simple run-length decoding for zero bytes
  for (size_t i = 0; i < compressed_buffer.size(); i++)
  {
    if (compressed_buffer[i] == 0xFF && i + 1 < compressed_buffer.size())
    {
      // Escape sequence for zeros
      uint8_t zero_count = compressed_buffer[i + 1];
      for (uint8_t j = 0; j < zero_count; j++)
      {
        decompressed_buffer.push_back(0);
      }
      i++; // Skip the count byte
    }
    else
    {
      decompressed_buffer.push_back(compressed_buffer[i]);
    }
  }

  // Parse decompressed data
  size_t offset = 0;
  uint64_t entries_read = 0;

  for (uint32_t bucket_idx = 0; bucket_idx < ht_header.non_empty_buckets; bucket_idx++)
  {
    if (offset + sizeof(uint32_t) * 3 > decompressed_buffer.size())
    {
      std::cerr << "ERROR: Insufficient data for bucket metadata" << std::endl;
      return false;
    }

    // Read bucket index
    uint32_t h = *reinterpret_cast<const uint32_t *>(decompressed_buffer.data() + offset);
    offset += sizeof(uint32_t);

    // Validate bucket index
    if (h >= HASH_SIZE)
    {
      std::cerr << "ERROR: Invalid bucket index: " << h << std::endl;
      return false;
    }

    // Read bucket metadata
    hashTable.E[h].nbItem = *reinterpret_cast<const uint32_t *>(decompressed_buffer.data() + offset);
    offset += sizeof(uint32_t);
    hashTable.E[h].maxItem = *reinterpret_cast<const uint32_t *>(decompressed_buffer.data() + offset);
    offset += sizeof(uint32_t);

    // Validate bucket metadata
    if (hashTable.E[h].nbItem > hashTable.E[h].maxItem)
    {
      std::cerr << "ERROR: Invalid bucket metadata" << std::endl;
      return false;
    }

    // Allocate bucket items array
    if (hashTable.E[h].maxItem > 0)
    {
      hashTable.E[h].items = (ENTRY **)malloc(sizeof(ENTRY *) * hashTable.E[h].maxItem);
      if (!hashTable.E[h].items)
      {
        std::cerr << "ERROR: Failed to allocate memory for bucket " << h << std::endl;
        return false;
      }
    }

    // Read bucket entries
    for (uint32_t i = 0; i < hashTable.E[h].nbItem; i++)
    {
      if (offset + 32 > decompressed_buffer.size())
      {
        std::cerr << "ERROR: Insufficient data for entry " << i << std::endl;
        return false;
      }

      hashTable.E[h].items[i] = (ENTRY *)malloc(sizeof(ENTRY));
      if (!hashTable.E[h].items[i])
      {
        std::cerr << "ERROR: Failed to allocate memory for entry" << std::endl;
        return false;
      }

      // Copy x and d values
      memcpy(&hashTable.E[h].items[i]->x, decompressed_buffer.data() + offset, 16);
      offset += 16;
      memcpy(&hashTable.E[h].items[i]->d, decompressed_buffer.data() + offset, 16);
      offset += 16;

      entries_read++;
    }
  }

  // Verify we read the expected number of entries
  if (entries_read != ht_header.total_entries)
  {
    std::cerr << "ERROR: Hash table entry count mismatch. Expected: "
              << ht_header.total_entries << ", Read: " << entries_read << std::endl;
    return false;
  }

  double compression_ratio = static_cast<double>(ht_header.compressed_size) / ht_header.original_size;
  std::cout << "Compressed hash table loaded: " << ht_header.non_empty_buckets
            << " buckets, " << ht_header.total_entries << " entries, "
            << "compression ratio: " << std::fixed << std::setprecision(3)
            << compression_ratio << std::endl;

  return true;
}

// Utility compression functions
size_t OptimizedCheckpoint::CompressData(const void *input, size_t input_size, void **output)
{
  if (!input || input_size == 0 || !output)
  {
    return 0;
  }

  const uint8_t *input_bytes = static_cast<const uint8_t *>(input);
  std::vector<uint8_t> compressed;
  compressed.reserve(input_size);

  // Simple run-length encoding
  for (size_t i = 0; i < input_size;)
  {
    if (input_bytes[i] == 0)
    {
      // Count consecutive zeros
      size_t zero_count = 0;
      while (i + zero_count < input_size &&
             input_bytes[i + zero_count] == 0 &&
             zero_count < 255)
      {
        zero_count++;
      }

      if (zero_count >= 3)
      {
        compressed.push_back(0xFF); // Escape byte
        compressed.push_back(static_cast<uint8_t>(zero_count));
        i += zero_count;
      }
      else
      {
        compressed.push_back(input_bytes[i]);
        i++;
      }
    }
    else
    {
      compressed.push_back(input_bytes[i]);
      i++;
    }
  }

  // Allocate output buffer
  *output = malloc(compressed.size());
  if (!*output)
  {
    return 0;
  }

  memcpy(*output, compressed.data(), compressed.size());
  return compressed.size();
}

size_t OptimizedCheckpoint::DecompressData(const void *input, size_t input_size, void **output, size_t expected_size)
{
  if (!input || input_size == 0 || !output)
  {
    return 0;
  }

  const uint8_t *input_bytes = static_cast<const uint8_t *>(input);
  std::vector<uint8_t> decompressed;
  decompressed.reserve(expected_size);

  // Simple run-length decoding
  for (size_t i = 0; i < input_size; i++)
  {
    if (input_bytes[i] == 0xFF && i + 1 < input_size)
    {
      // Escape sequence for zeros
      uint8_t zero_count = input_bytes[i + 1];
      for (uint8_t j = 0; j < zero_count; j++)
      {
        decompressed.push_back(0);
      }
      i++; // Skip the count byte
    }
    else
    {
      decompressed.push_back(input_bytes[i]);
    }
  }

  // Allocate output buffer
  *output = malloc(decompressed.size());
  if (!*output)
  {
    return 0;
  }

  memcpy(*output, decompressed.data(), decompressed.size());
  return decompressed.size();
}

bool OptimizedCheckpoint::ValidateCompressionRatio(size_t original_size, size_t compressed_size)
{
  if (original_size == 0)
  {
    return false;
  }

  double ratio = static_cast<double>(compressed_size) / original_size;

  // Only use compression if we achieve at least 5% reduction
  return ratio < MAX_COMPRESSION_RATIO;
}

// Enhanced compressed kangaroo save function
bool OptimizedCheckpoint::SaveKangaroosCompressed(FILE *f, TH_PARAM *threads, int nbThread)
{
  std::cout << "Saving kangaroo states with enhanced compression..." << std::endl;

  // Count total kangaroos
  uint64_t total_kangaroos = 0;
  for (int i = 0; i < nbThread; i++)
  {
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
  original_size += total_kangaroos * (32 + 32 + 32);  // px, py, distance (96 bytes each)

  // Create temporary buffer for compression
  std::vector<uint8_t> temp_buffer;
  temp_buffer.reserve(original_size);

  // Serialize kangaroo data to buffer
  for (int i = 0; i < nbThread; i++)
  {
    // Add thread kangaroo count
    temp_buffer.insert(temp_buffer.end(),
                       reinterpret_cast<const uint8_t *>(&threads[i].nbKangaroo),
                       reinterpret_cast<const uint8_t *>(&threads[i].nbKangaroo) + sizeof(uint64_t));

    // Add kangaroo states for this thread
    for (uint64_t n = 0; n < threads[i].nbKangaroo; n++)
    {
      // Add position (px, py) and distance
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&threads[i].px[n].bits64),
                         reinterpret_cast<const uint8_t *>(&threads[i].px[n].bits64) + 32);
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&threads[i].py[n].bits64),
                         reinterpret_cast<const uint8_t *>(&threads[i].py[n].bits64) + 32);
      temp_buffer.insert(temp_buffer.end(),
                         reinterpret_cast<const uint8_t *>(&threads[i].distance[n].bits64),
                         reinterpret_cast<const uint8_t *>(&threads[i].distance[n].bits64) + 32);
    }
  }

  // Apply compression using delta encoding and run-length encoding
  std::vector<uint8_t> compressed_buffer;
  compressed_buffer.reserve(temp_buffer.size());

  // Simple compression: delta encoding for similar values + RLE for zeros
  uint8_t prev_byte = 0;
  for (size_t i = 0; i < temp_buffer.size();)
  {
    if (temp_buffer[i] == 0)
    {
      // Count consecutive zeros
      size_t zero_count = 0;
      while (i + zero_count < temp_buffer.size() &&
             temp_buffer[i + zero_count] == 0 &&
             zero_count < 255)
      {
        zero_count++;
      }

      if (zero_count >= 4)
      {                                    // Only compress if we have 4+ zeros
        compressed_buffer.push_back(0xFE); // Escape byte for zeros
        compressed_buffer.push_back(static_cast<uint8_t>(zero_count));
        i += zero_count;
      }
      else
      {
        compressed_buffer.push_back(temp_buffer[i]);
        i++;
      }
    }
    else
    {
      // Delta encoding for non-zero bytes
      int16_t delta = static_cast<int16_t>(temp_buffer[i]) - static_cast<int16_t>(prev_byte);
      if (delta >= -127 && delta <= 127 && delta != -2)
      {                                                                 // Avoid conflict with escape
        compressed_buffer.push_back(0xFF);                              // Escape byte for delta
        compressed_buffer.push_back(static_cast<uint8_t>(delta + 128)); // Offset by 128
      }
      else
      {
        compressed_buffer.push_back(temp_buffer[i]);
      }
      prev_byte = temp_buffer[i];
      i++;
    }
  }

  // Validate compression ratio
  if (!ValidateCompressionRatio(temp_buffer.size(), compressed_buffer.size()))
  {
    std::cout << "Compression ratio not beneficial, using uncompressed format" << std::endl;
    return SaveKangaroosOptimized(f, threads, nbThread);
  }

  // Calculate checksum
  uint64_t data_checksum = 0;
  if (validate_checksums)
  {
    data_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
  }
  k_header.data_checksum = data_checksum;

  // Write kangaroo header
  if (!WriteWithErrorCheck(f, &k_header, sizeof(k_header)))
  {
    return false;
  }

  // Write original size
  uint32_t orig_size = static_cast<uint32_t>(temp_buffer.size());
  if (!WriteWithErrorCheck(f, &orig_size, sizeof(orig_size)))
  {
    return false;
  }

  // Write compressed size
  uint32_t comp_size = static_cast<uint32_t>(compressed_buffer.size());
  if (!WriteWithErrorCheck(f, &comp_size, sizeof(comp_size)))
  {
    return false;
  }

  // Write compressed data
  if (!WriteWithErrorCheck(f, compressed_buffer.data(), compressed_buffer.size()))
  {
    return false;
  }

  double compression_ratio = static_cast<double>(compressed_buffer.size()) / temp_buffer.size();
  std::cout << "Compressed kangaroo states saved: " << nbThread << " threads, "
            << total_kangaroos << " kangaroos, compression ratio: "
            << std::fixed << std::setprecision(3) << compression_ratio << std::endl;

  return true;
}

// Enhanced compressed kangaroo load function
bool OptimizedCheckpoint::LoadKangaroosCompressed(FILE *f, TH_PARAM *threads, int &nbThread)
{
  std::cout << "Loading compressed kangaroo states..." << std::endl;

  // Read kangaroo header
  KangarooHeader k_header;
  if (!ReadWithErrorCheck(f, &k_header, sizeof(k_header)))
  {
    return false;
  }

  // Validate kangaroo header
  if (k_header.magic != KANGAROO_MAGIC)
  {
    std::cerr << "ERROR: Invalid kangaroo magic number: 0x"
              << std::hex << k_header.magic << std::dec << std::endl;
    return false;
  }

  nbThread = static_cast<int>(k_header.thread_count);

  // Read size information
  uint32_t original_size, compressed_size;
  if (!ReadWithErrorCheck(f, &original_size, sizeof(original_size)))
  {
    return false;
  }
  if (!ReadWithErrorCheck(f, &compressed_size, sizeof(compressed_size)))
  {
    return false;
  }

  // Read compressed data
  std::vector<uint8_t> compressed_buffer(compressed_size);
  if (!ReadWithErrorCheck(f, compressed_buffer.data(), compressed_size))
  {
    return false;
  }

  // Validate checksum if enabled
  if (validate_checksums && k_header.data_checksum != 0)
  {
    uint64_t calculated_checksum = CalculateChecksum(compressed_buffer.data(), compressed_buffer.size());
    if (calculated_checksum != k_header.data_checksum)
    {
      std::cerr << "ERROR: Kangaroo data checksum mismatch" << std::endl;
      return false;
    }
  }

  // Decompress data
  std::vector<uint8_t> decompressed_buffer;
  decompressed_buffer.reserve(original_size);

  // Decompress using reverse delta encoding and RLE
  uint8_t prev_byte = 0;
  for (size_t i = 0; i < compressed_buffer.size(); i++)
  {
    if (compressed_buffer[i] == 0xFE && i + 1 < compressed_buffer.size())
    {
      // Zero run-length decoding
      uint8_t zero_count = compressed_buffer[i + 1];
      for (uint8_t j = 0; j < zero_count; j++)
      {
        decompressed_buffer.push_back(0);
      }
      i++; // Skip the count byte
    }
    else if (compressed_buffer[i] == 0xFF && i + 1 < compressed_buffer.size())
    {
      // Delta decoding
      int16_t delta = static_cast<int16_t>(compressed_buffer[i + 1]) - 128;
      uint8_t decoded_byte = static_cast<uint8_t>(static_cast<int16_t>(prev_byte) + delta);
      decompressed_buffer.push_back(decoded_byte);
      prev_byte = decoded_byte;
      i++; // Skip the delta byte
    }
    else
    {
      decompressed_buffer.push_back(compressed_buffer[i]);
      prev_byte = compressed_buffer[i];
    }
  }

  // Validate decompressed size
  if (decompressed_buffer.size() != original_size)
  {
    std::cerr << "ERROR: Decompressed size mismatch. Expected: " << original_size
              << ", Got: " << decompressed_buffer.size() << std::endl;
    return false;
  }

  // Parse decompressed kangaroo data
  size_t offset = 0;
  uint64_t kangaroos_read = 0;

  for (int i = 0; i < nbThread; i++)
  {
    if (offset + sizeof(uint64_t) > decompressed_buffer.size())
    {
      std::cerr << "ERROR: Insufficient data for thread " << i << " kangaroo count" << std::endl;
      return false;
    }

    // Read thread kangaroo count
    threads[i].nbKangaroo = *reinterpret_cast<const uint64_t *>(decompressed_buffer.data() + offset);
    offset += sizeof(uint64_t);

    // Read kangaroo states for this thread
    for (uint64_t n = 0; n < threads[i].nbKangaroo; n++)
    {
      if (offset + 96 > decompressed_buffer.size())
      {
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
  if (kangaroos_read != k_header.total_kangaroos)
  {
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
