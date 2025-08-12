/*
 * This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include <memory>
#include "../Constants.h"
#include "../SECPK1/SECP256k1.h"
#include "CudaMemoryManager.h"

#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif

#define ITEM_SIZE 56
#define ITEM_SIZE32 (ITEM_SIZE / 4)

typedef struct
{
  Int x;
  Int d;
  uint64_t kIdx;
} ITEM;

// RAII GPU Memory Management Class
class CudaMemoryGuard
{
public:
  CudaMemoryGuard() = default;
  ~CudaMemoryGuard(); // Implementation in .cu file to avoid CUDA header dependency

  // Non-copyable, non-movable
  CudaMemoryGuard(const CudaMemoryGuard &) = delete;
  CudaMemoryGuard &operator=(const CudaMemoryGuard &) = delete;
  CudaMemoryGuard(CudaMemoryGuard &&) = delete;
  CudaMemoryGuard &operator=(CudaMemoryGuard &&) = delete;
};

class GPUEngine
{

public:
  GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound);
  ~GPUEngine();
  void SetParams(uint64_t dpMask, Int *distance, Int *px, Int *py);
  void SetKangaroos(Int *px, Int *py, Int *d);
  void GetKangaroos(Int *px, Int *py, Int *d);
  void SetKangaroo(uint64_t kIdx, Int *px, Int *py, Int *d);
  bool Launch(std::vector<ITEM> &hashFound, bool spinWait = false);
  void GenerateCode(Secp256K1 *secp);
  void SetWildOffset(Int *offset);
  int GetNbThread();
  int GetGroupSize();
  int GetMemory();
  bool callKernelAndWait();
  bool callKernel();

  // Force GPU memory cleanup (emergency cleanup)
  static void ForceGPUCleanup();

  std::string deviceName;

  static void *AllocatePinnedMemory(size_t size);
  static void FreePinnedMemory(void *buff);
  static void PrintCudaInfo();
  static bool GetGridSize(int gpuId, int *x, int *y);

private:
  // Helper functions for safe GPU data layout management
  inline size_t CalculateKangarooOffset(int group, int thread, int coord_word) const;
  inline void SetCoordinateData(int group, int thread, int coord_base, const uint64_t *data, int word_count);
  inline void GetCoordinateData(int group, int thread, int coord_base, uint64_t *data, int word_count) const;
  inline bool SetSingleCoordinateToGPU(uint64_t kIdx, int coord_word, uint64_t value);

  Int wildOffset;
  int nbThread;
  int nbThreadPerGroup;
  bool initialised;
  bool lostWarning;
  uint32_t maxFound;
  uint64_t dpMask;

  // Memory sizes
  size_t outputSize;
  size_t kangarooSize;
  size_t kangarooSizePinned;
  size_t jumpSize;

  // RAII Memory Management
  std::unique_ptr<KangarooMemoryLayout> memory_layout_;

  // Legacy compatibility - will be removed
  uint64_t *inputKangaroo;
  uint64_t *inputKangarooPinned;
  uint32_t *outputItem;
  uint32_t *outputItemPinned;
  uint64_t *jumpPinned;
};

#endif // GPUENGINEH
