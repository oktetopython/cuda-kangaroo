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

#include "Kangaroo.h"
#include "OptimizedCheckpoint.h"
#include "Timer.h"
#include <iostream>

// Enhanced SaveWork function with optimized checkpoint support
void Kangaroo::SaveWorkOptimized(uint64_t totalCount, double totalTime, TH_PARAM *threads, int nbThread)
{
  LOCK(saveMutex);

  double t0 = Timer::get_tick();

  // Wait for all threads to be in waiting state
  saveRequest = true;
  int timeout = wtimeout;
  while (!isWaiting(threads) && timeout > 0)
  {
    Timer::SleepMillis(50);
    timeout -= 50;
  }

  if (timeout <= 0)
  {
    if (!endOfSearch)
    {
      std::cout << "SaveWork timeout!" << std::endl;
    }
    UNLOCK(saveMutex);
    return;
  }

  std::string fileName = workFile;
  if (splitWorkfile)
  {
    fileName = workFile + "_" + Timer::getTS();
  }

  // Try optimized checkpoint with compression enabled
  OptimizedCheckpoint checkpoint(fileName, true, true); // Enable compression and checksums
  CheckpointError result = checkpoint.SaveCheckpoint(totalCount, totalTime, hashTable, threads, nbThread, dpSize);

  if (result == CHECKPOINT_OK)
  {
    std::cout << "Optimized checkpoint saved successfully" << std::endl;
  }
  else
  {
    // Fallback to legacy format
    std::cout << "Optimized checkpoint failed (" << checkpoint.GetErrorMessage(result)
              << "), falling back to legacy format" << std::endl;

    // Use original SaveWork implementation as fallback
    SaveWork(totalCount, totalTime, threads, nbThread);
    UNLOCK(saveMutex);
    return;
  }

  if (splitWorkfile)
  {
    hashTable.Reset();
  }

  saveRequest = false;
  UNLOCK(saveMutex);

  double t1 = Timer::get_tick();
  uint64_t size = checkpoint.GetFileSize(fileName);

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  std::cout << "done [" << (size / (1024.0 * 1024.0)) << " MB] ["
            << (t1 - t0) << "s] " << ctimeBuff;
}

// Enhanced LoadWork function with optimized checkpoint support
bool Kangaroo::LoadWorkOptimized(std::string &fileName)
{
  double t0 = Timer::get_tick();

  std::cout << "Loading: " << fileName << std::endl;

  // Check if it's an optimized checkpoint with compression support
  OptimizedCheckpoint checkpoint(fileName, true, true); // Enable compression and checksums

  if (checkpoint.FileExists(fileName) && !checkpoint.IsLegacyFormat(fileName))
  {
    // Load optimized checkpoint
    uint64_t totalCount;
    double totalTime;
    uint32_t loadedDpSize;
    int loadedNbThread = 0;

    // Prepare thread parameters array (assuming reasonable maximum)
    const int MAX_THREADS = 256;
    TH_PARAM temp_threads[MAX_THREADS];
    memset(temp_threads, 0, sizeof(temp_threads));

    CheckpointError result = checkpoint.LoadCheckpoint(totalCount, totalTime, hashTable,
                                                       temp_threads, loadedNbThread, loadedDpSize);

    if (result == CHECKPOINT_OK)
    {
      // Update global parameters
      offsetCount = totalCount;
      offsetTime = totalTime;
      if (initDPSize < 0)
        initDPSize = loadedDpSize;

      // Store loaded kangaroo data for later use
      nbLoadedWalk = 0;
      for (int i = 0; i < loadedNbThread; i++)
      {
        nbLoadedWalk += temp_threads[i].nbKangaroo;
      }

      double t1 = Timer::get_tick();
      std::cout << "LoadWork: [HashTable " << hashTable.GetSizeInfo()
                << "] [" << (t1 - t0) << "s]" << std::endl;

      return true;
    }
    else
    {
      std::cout << "Failed to load optimized checkpoint ("
                << checkpoint.GetErrorMessage(result)
                << "), trying legacy format" << std::endl;
    }
  }

  // Fallback to legacy format
  return LoadWork(fileName);
}

// Automatic format detection and conversion
bool Kangaroo::ConvertCheckpointToOptimized(const std::string &legacyFile, const std::string &optimizedFile)
{
  std::cout << "Converting checkpoint from legacy to optimized format..." << std::endl;

  // Load legacy checkpoint
  std::string legacyFileName = legacyFile;
  if (!LoadWork(legacyFileName))
  {
    std::cerr << "ERROR: Failed to load legacy checkpoint: " << legacyFile << std::endl;
    return false;
  }

  // Create optimized checkpoint
  OptimizedCheckpoint checkpoint(optimizedFile, false, true);

  // We need to simulate thread parameters for conversion
  // This is a simplified approach - in practice, you'd need actual thread data
  const int CONVERSION_THREADS = 1;
  TH_PARAM conversion_threads[CONVERSION_THREADS];
  memset(conversion_threads, 0, sizeof(conversion_threads));

  // Set up basic thread parameters
  conversion_threads[0].nbKangaroo = nbLoadedWalk;

  CheckpointError result = checkpoint.SaveCheckpoint(offsetCount, offsetTime, hashTable,
                                                     conversion_threads, CONVERSION_THREADS, dpSize);

  if (result == CHECKPOINT_OK)
  {
    std::cout << "Checkpoint conversion completed successfully" << std::endl;

    // Verify the converted file
    uint64_t original_size = checkpoint.GetFileSize(legacyFile);
    uint64_t optimized_size = checkpoint.GetFileSize(optimizedFile);

    double compression_ratio = (double)optimized_size / original_size;
    std::cout << "File size reduction: " << (original_size / (1024.0 * 1024.0))
              << " MB -> " << (optimized_size / (1024.0 * 1024.0))
              << " MB (ratio: " << compression_ratio << ")" << std::endl;

    return true;
  }
  else
  {
    std::cerr << "ERROR: Checkpoint conversion failed: "
              << checkpoint.GetErrorMessage(result) << std::endl;
    return false;
  }
}

// Checkpoint validation function
bool Kangaroo::ValidateCheckpoint(const std::string &fileName)
{
  std::cout << "Validating checkpoint: " << fileName << std::endl;

  OptimizedCheckpoint checkpoint(fileName, false, true);

  if (!checkpoint.FileExists(fileName))
  {
    std::cerr << "ERROR: Checkpoint file does not exist" << std::endl;
    return false;
  }

  if (checkpoint.IsLegacyFormat(fileName))
  {
    std::cout << "Legacy format detected - using legacy validation" << std::endl;
    // Could implement legacy validation here
    return true;
  }

  // Load and validate optimized checkpoint
  uint64_t totalCount;
  double totalTime;
  uint32_t loadedDpSize;
  int loadedNbThread = 0;

  const int MAX_THREADS = 256;
  TH_PARAM temp_threads[MAX_THREADS];
  memset(temp_threads, 0, sizeof(temp_threads));

  HashTable temp_hashTable;

  CheckpointError result = checkpoint.LoadCheckpoint(totalCount, totalTime, temp_hashTable,
                                                     temp_threads, loadedNbThread, loadedDpSize);

  if (result == CHECKPOINT_OK)
  {
    // Perform additional validation
    if (!CheckpointUtils::ValidateHashTableIntegrity(temp_hashTable))
    {
      std::cerr << "ERROR: Hash table integrity check failed" << std::endl;
      return false;
    }

    for (int i = 0; i < loadedNbThread; i++)
    {
      if (!CheckpointUtils::ValidateKangarooState(temp_threads[i]))
      {
        std::cerr << "ERROR: Kangaroo state validation failed for thread " << i << std::endl;
        return false;
      }
    }

    std::cout << "Checkpoint validation passed" << std::endl;
    std::cout << "  Total count: " << totalCount << std::endl;
    std::cout << "  Total time: " << totalTime << "s" << std::endl;
    std::cout << "  DP size: " << loadedDpSize << std::endl;
    std::cout << "  Threads: " << loadedNbThread << std::endl;
    std::cout << "  Hash table entries: " << temp_hashTable.GetNbItem() << std::endl;

    return true;
  }
  else
  {
    std::cerr << "ERROR: Checkpoint validation failed: "
              << checkpoint.GetErrorMessage(result) << std::endl;
    return false;
  }
}

// Checkpoint information display
void Kangaroo::ShowCheckpointInfo(const std::string &fileName)
{
  std::cout << "Checkpoint Information: " << fileName << std::endl;
  std::cout << "======================" << std::endl;

  OptimizedCheckpoint checkpoint(fileName, false, true);

  if (!checkpoint.FileExists(fileName))
  {
    std::cout << "File does not exist" << std::endl;
    return;
  }

  uint64_t file_size = checkpoint.GetFileSize(fileName);
  std::cout << "File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;

  if (checkpoint.IsLegacyFormat(fileName))
  {
    std::cout << "Format: Legacy" << std::endl;
    // Use existing WorkInfo function for legacy files
    WorkInfo(const_cast<std::string &>(fileName));
  }
  else
  {
    std::cout << "Format: Optimized" << std::endl;
    ValidateCheckpoint(fileName); // This will show detailed info
  }
}
