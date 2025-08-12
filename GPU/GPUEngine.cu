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

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <cassert> // For assert function
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUCompute.h"
#include "../CudaErrorHandler.h" // Use the comprehensive error handler from root directory

// Use modern CUDA error handling system
// Legacy CheckCudaError function replaced by CudaErrorChecker class

// Compatibility wrapper for legacy CheckCudaError function
bool CheckCudaError(cudaError_t err, const char *message, bool exitOnError = false, const char *context = nullptr, int index = -1)
{
  if (err != cudaSuccess)
  {
    std::string full_message = std::string(message);
    if (context && index >= 0)
    {
      full_message += " (" + std::string(context) + " " + std::to_string(index) + ")";
    }

    printf("CUDA Error: %s - %s\n", full_message.c_str(), cudaGetErrorString(err));

    if (exitOnError)
    {
      exit(1);
    }
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------

__global__ void comp_kangaroos(uint64_t *kangaroos, uint32_t maxFound, uint32_t *found, uint64_t dpMask)
{

  int xPtr = (blockIdx.x * blockDim.x * GPU_GRP_SIZE) * KSIZE; // x[4] , y[4] , d[2], lastJump
  ComputeKangaroos(kangaroos + xPtr, maxFound, found, dpMask);
}

// ---------------------------------------------------------------------------------------
// #define GPU_CHECK
#ifdef GPU_CHECK
__global__ void check_gpu()
{

  // Check ModInv
  uint64_t N[5] = {0x0BE3D7593BE1147CULL, 0x4952AAF512875655ULL, 0x08884CCAACCB9B53ULL, 0x9EAE2E2225044292ULL, 0ULL};
  uint64_t I[5];
  uint64_t R[5];
  bool ok = true;

  /*
  for(uint64_t i=0;i<10000 && ok;i++) {

    Load(R,N);
    _ModInv(R);
    Load(I,R);
    _ModMult(R,N);
    SubP(R);
    if(!_IsOne(R)) {
      ok = false;
      printf("ModInv wrong %d\n",(int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n",N[4],N[3],N[2],N[1],N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n",I[4],I[3],I[2],I[1],I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n",R[4],R[3],R[2],R[1],R[0]);
    }

    N[0]++;

  }
  */
  I[4] = 0;
  R[4] = 0;
  for (uint64_t i = 0; i < 100000 && ok; i++)
  {

    _ModSqr(I, N);
    _ModMult(R, N, N);
    if (!_IsEqual(I, R))
    {
      ok = false;
      printf("_ModSqr wrong %d\n", (int)i);
      printf("N = %016llx %016llx %016llx %016llx %016llx\n", N[4], N[3], N[2], N[1], N[0]);
      printf("I = %016llx %016llx %016llx %016llx %016llx\n", I[4], I[3], I[2], I[1], I[0]);
      printf("R = %016llx %016llx %016llx %016llx %016llx\n", R[4], R[3], R[2], R[1], R[0]);
    }

    N[0]++;
  }
}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct
  {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
            // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60, 64},
      {0x61, 128},
      {0x62, 128},
      {0x70, 64},
      {0x72, 64},
      {0x75, 64},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1)
  {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
    {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;
}

void GPUEngine::SetWildOffset(Int *offset)
{
  wildOffset.Set(offset);
}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound)
{

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaDeviceProp deviceProp;

  err = cudaGetDeviceCount(&deviceCount);
  if (!CheckCudaError(err, "cudaGetDeviceCount"))
  {
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (!CheckCudaError(err, "cudaSetDevice"))
  {
    return;
  }

  err = cudaGetDeviceProperties(&deviceProp, gpuId);
  if (!CheckCudaError(err, "cudaGetDeviceProperties"))
  {
    return;
  }

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  outputSize = (maxFound * ITEM_SIZE + 4);

  char tmp[512];
  snprintf(tmp, sizeof(tmp), "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
           gpuId, deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           nbThread / nbThreadPerGroup,
           nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (!CheckCudaError(err, "cudaDeviceSetCacheConfig"))
  {
    return;
  }

  // Allocate memory
  inputKangaroo = NULL;
  inputKangarooPinned = NULL;
  outputItem = NULL;
  outputItemPinned = NULL;
  jumpPinned = NULL;

  // Memory allocation with error handling
  // Input kangaroos
  kangarooSize = nbThread * GPU_GRP_SIZE * KSIZE * 8;
  err = cudaMalloc((void **)&inputKangaroo, kangarooSize);
  if (!CheckCudaError(err, "cudaMalloc(inputKangaroo)"))
  {
    return;
  }

  kangarooSizePinned = nbThreadPerGroup * GPU_GRP_SIZE * KSIZE * 8;
  err = cudaHostAlloc(&inputKangarooPinned, kangarooSizePinned,
                      cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (!CheckCudaError(err, "cudaHostAlloc(inputKangarooPinned)"))
  {
    return;
  }

  // OutputHash
  err = cudaMalloc((void **)&outputItem, outputSize);
  if (!CheckCudaError(err, "cudaMalloc(outputItem)"))
  {
    return;
  }

  err = cudaHostAlloc(&outputItemPinned, outputSize, cudaHostAllocMapped);
  if (!CheckCudaError(err, "cudaHostAlloc(outputItemPinned)"))
  {
    return;
  }

  // Jump array
  jumpSize = NB_JUMP * 8 * 4;
  err = cudaHostAlloc(&jumpPinned, jumpSize, cudaHostAllocMapped);
  if (!CheckCudaError(err, "cudaHostAlloc(jumpPinned)"))
  {
    return;
  }

  lostWarning = false;
  initialised = true;
  wildOffset.SetInt32(0);

#ifdef GPU_CHECK

  double minT = 1e9;
  for (int i = 0; i < 5; i++)
  {
    double t0 = Timer::get_tick();
    CUDA_LAUNCH_KERNEL(check_gpu, 1, 1, 0, 0);
    CudaErrorChecker::deviceSynchronize();
    double t1 = Timer::get_tick();
    if ((t1 - t0) < minT)
      minT = (t1 - t0);
  }
  printf("Cuda: %.3f ms\n", minT * 1000.0);
  exit(0);

#endif
}

GPUEngine::~GPUEngine()
{

  // Safe memory deallocation with modern error handling
  auto safeFree = [](auto &ptr, auto freeFunc, const char *name)
  {
    if (ptr)
    {
      cudaError_t err = freeFunc(ptr);
      if (err != cudaSuccess)
      {
        // Don't throw in destructor - just log warning
        printf("Warning - Failed to free %s: %s\n", name, cudaGetErrorString(err));
      }
      ptr = nullptr;
    }
  };

  // Free GPU device memory
  safeFree(inputKangaroo, cudaFree, "inputKangaroo");
  safeFree(outputItem, cudaFree, "outputItem");

  // Free host pinned memory
  safeFree(inputKangarooPinned, cudaFreeHost, "inputKangarooPinned");
  safeFree(outputItemPinned, cudaFreeHost, "outputItemPinned");
  safeFree(jumpPinned, cudaFreeHost, "jumpPinned");

  // Force GPU context cleanup to prevent memory leaks
  // This ensures all GPU memory is properly released
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess)
  {
    printf("Warning - GPU device reset failed: %s\n", cudaGetErrorString(err));
  }
}

int GPUEngine::GetMemory()
{
  return static_cast<int>(kangarooSize + outputSize + jumpSize);
}

int GPUEngine::GetGroupSize()
{
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId, int *x, int *y)
{

  if (*x <= 0 || *y <= 0)
  {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
      printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
      printf("GPUEngine: There are no available device(s) that support CUDA\n");
      return false;
    }

    if (gpuId >= deviceCount)
    {
      printf("GPUEngine::GetGridSize() Invalid gpuId\n");
      return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    if (*x <= 0)
      *x = 2 * deviceProp.multiProcessorCount;
    if (*y <= 0)
      *y = 2 * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    if (*y <= 0)
      *y = 128;
  }

  return true;
}

void *GPUEngine::AllocatePinnedMemory(size_t size)
{

  void *buff;

  cudaError_t err = cudaHostAlloc(&buff, size, cudaHostAllocPortable);
  if (!CheckCudaError(err, "AllocatePinnedMemory"))
  {
    return NULL;
  }

  return buff;
}

void GPUEngine::FreePinnedMemory(void *buff)
{
  if (buff)
  {
    cudaError_t err = cudaFreeHost(buff);
    CheckCudaError(err, "Warning - Failed to free pinned memory");
  }
}

void GPUEngine::ForceGPUCleanup()
{
  // Emergency GPU memory cleanup function
  // This can be called from signal handlers or exception handlers
  cudaError_t err = cudaDeviceReset();
  if (!CheckCudaError(err, "ForceGPUCleanup: GPU reset failed"))
  {
    // Error already logged
  }
  else
  {
    printf("ForceGPUCleanup: GPU memory successfully released\n");
  }
}

// CudaMemoryGuard destructor implementation
CudaMemoryGuard::~CudaMemoryGuard()
{
  // Force cleanup of all GPU resources on destruction
  cudaError_t err = cudaDeviceReset();
  CheckCudaError(err, "CudaMemoryGuard: GPU cleanup warning");
}

void GPUEngine::PrintCudaInfo()
{

  cudaError_t err;

  const char *sComputeMode[] =
      {
          "Multiple host threads",
          "Only one host thread",
          "No host thread",
          "Multiple process threads",
          "Unknown",
          NULL};

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess)
  {
    CudaErrorHandler::HandleGPUEngineError(error_id, "CudaGetDeviceCount");
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for (int i = 0; i < deviceCount; i++)
  {

    err = cudaSetDevice(i);
    if (!CheckCudaError(err, ("cudaSetDevice(" + std::to_string(i) + ")").c_str()))
    {
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
           i, deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           deviceProp.major, deviceProp.minor, (double)deviceProp.totalGlobalMem / 1048576.0,
           sComputeMode[deviceProp.computeMode]);
  }
}

int GPUEngine::GetNbThread()
{
  return nbThread;
}

// Helper function to calculate safe memory offset with bounds checking
inline size_t GPUEngine::CalculateKangarooOffset(int group, int thread, int coord_word) const
{
  // Validate inputs to prevent buffer overflows
  assert(group >= 0 && group < GPU_GRP_SIZE);
  assert(thread >= 0 && thread < nbThreadPerGroup);
  assert(coord_word >= 0 && coord_word < KSIZE);

  const int strideSize = nbThreadPerGroup * KSIZE;
  size_t offset = group * strideSize + thread + coord_word * nbThreadPerGroup;

  // Additional bounds check for the calculated offset
#ifdef DEBUG
  size_t maxOffset = kangarooSizePinned / sizeof(uint64_t);
  assert(offset < maxOffset);
#else
  // In release builds, we trust the input validation above
  // This avoids the unused variable warning while maintaining safety
  assert(offset < kangarooSizePinned / sizeof(uint64_t));
#endif

  return offset;
}

// Helper function to safely set coordinate data
inline void GPUEngine::SetCoordinateData(int group, int thread, int coord_base, const uint64_t *data, int word_count)
{
  for (int i = 0; i < word_count; i++)
  {
    size_t offset = CalculateKangarooOffset(group, thread, coord_base + i);
    inputKangarooPinned[offset] = data[i];
  }
}

// Helper function to safely get coordinate data
inline void GPUEngine::GetCoordinateData(int group, int thread, int coord_base, uint64_t *data, int word_count) const
{
  for (int i = 0; i < word_count; i++)
  {
    size_t offset = CalculateKangarooOffset(group, thread, coord_base + i);
    data[i] = inputKangarooPinned[offset];
  }
}

// Helper function to safely set single kangaroo coordinate to GPU memory
inline bool GPUEngine::SetSingleCoordinateToGPU(uint64_t kIdx, int coord_word, uint64_t value)
{
  // Calculate block, group, thread from kangaroo index
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int blockSize = nbThreadPerGroup * gSize;

  uint64_t t = kIdx % nbThreadPerGroup;
  uint64_t g = (kIdx / nbThreadPerGroup) % GPU_GRP_SIZE;
  uint64_t b = kIdx / (nbThreadPerGroup * GPU_GRP_SIZE);

  // Validate indices
  if (b >= (uint64_t)(nbThread / nbThreadPerGroup) || g >= GPU_GRP_SIZE || t >= (uint64_t)nbThreadPerGroup)
  {
    printf("GPUEngine: SetSingleCoordinateToGPU: Invalid kangaroo index %llu\n", kIdx);
    return false;
  }

  // Calculate safe offset
  size_t offset = b * blockSize + g * strideSize + t + coord_word * nbThreadPerGroup;

  // Copy single value to GPU
  inputKangarooPinned[0] = value;
  cudaError_t err = cudaMemcpy(inputKangaroo + offset, inputKangarooPinned, sizeof(uint64_t), cudaMemcpyHostToDevice);

  return CheckCudaError(err, "SetSingleCoordinateToGPU", false, "kangaroo", static_cast<int>(kIdx));
}

void GPUEngine::SetKangaroos(Int *px, Int *py, Int *d)
{

  // Sets the kangaroos of each thread
  // Using safer approach with helper functions to eliminate manual index calculations
  int gSize = KSIZE * GPU_GRP_SIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for (int b = 0; b < nbBlock; b++)
  {
    for (int g = 0; g < GPU_GRP_SIZE; g++)
    {
      for (int t = 0; t < nbThreadPerGroup; t++)
      {

        // X coordinate (4 words: 0-3)
        SetCoordinateData(g, t, 0, px[idx].bits64, 4);

        // Y coordinate (4 words: 4-7)
        SetCoordinateData(g, t, 4, py[idx].bits64, 4);

        // Distance (2 words: 8-9)
        Int dOff;
        dOff.Set(&d[idx]);
        if (idx % 2 == WILD)
          dOff.ModAddK1order(&wildOffset);
        SetCoordinateData(g, t, 8, dOff.bits64, 2);

#ifdef USE_SYMMETRY
        // Last jump (1 word: 10)
        uint64_t jumpData = (uint64_t)NB_JUMP;
        SetCoordinateData(g, t, 10, &jumpData, 1);
#endif

        idx++;
      }
    }

    uint32_t offset = b * blockSize;
    cudaMemcpy(inputKangaroo + offset, inputKangarooPinned, kangarooSizePinned, cudaMemcpyHostToDevice);
  }

  cudaError_t err = cudaGetLastError();
  CheckCudaError(err, "SetKangaroos");
}

void GPUEngine::GetKangaroos(Int *px, Int *py, Int *d)
{

  if (inputKangarooPinned == NULL)
  {
    printf("GPUEngine: GetKangaroos: Cannot retreive kangaroos, mem has been freed\n");
    return;
  }

  // Gets the kangaroos of each thread
  // Using safer approach with helper functions to eliminate manual index calculations
  int gSize = KSIZE * GPU_GRP_SIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for (int b = 0; b < nbBlock; b++)
  {

    uint32_t offset = b * blockSize;
    cudaMemcpy(inputKangarooPinned, inputKangaroo + offset, kangarooSizePinned, cudaMemcpyDeviceToHost);

    for (int g = 0; g < GPU_GRP_SIZE; g++)
    {

      for (int t = 0; t < nbThreadPerGroup; t++)
      {

        // X coordinate (4 words: 0-3)
        GetCoordinateData(g, t, 0, px[idx].bits64, 4);
        px[idx].bits64[4] = 0; // Clear the 5th word

        // Y coordinate (4 words: 4-7)
        GetCoordinateData(g, t, 4, py[idx].bits64, 4);
        py[idx].bits64[4] = 0; // Clear the 5th word

        // Distance (2 words: 8-9)
        Int dOff;
        dOff.SetInt32(0);
        GetCoordinateData(g, t, 8, dOff.bits64, 2);
        if (idx % 2 == WILD)
          dOff.ModSubK1order(&wildOffset);
        d[idx].Set(&dOff);

        idx++;
      }
    }
  }

  cudaError_t err = cudaGetLastError();
  CheckCudaError(err, "GetKangaroos");
}

void GPUEngine::SetKangaroo(uint64_t kIdx, Int *px, Int *py, Int *d)
{
  // Using safer approach with helper function to eliminate manual index calculations

  // X coordinate - copy all 4 64-bit words (coordinates 0-3)
  for (int i = 0; i < 4; i++)
  {
    if (!SetSingleCoordinateToGPU(kIdx, i, px->bits64[i]))
      return;
  }

  // Y coordinate - copy all 4 64-bit words (coordinates 4-7)
  for (int i = 0; i < 4; i++)
  {
    if (!SetSingleCoordinateToGPU(kIdx, 4 + i, py->bits64[i]))
      return;
  }

  // Distance - apply wild offset and copy 2 64-bit words (coordinates 8-9)
  Int dOff;
  dOff.Set(d);
  if (kIdx % 2 == WILD)
    dOff.ModAddK1order(&wildOffset);

  for (int i = 0; i < 2; i++)
  {
    if (!SetSingleCoordinateToGPU(kIdx, 8 + i, dOff.bits64[i]))
      return;
  }

#ifdef USE_SYMMETRY
  // Last jump count (coordinate 10)
  if (!SetSingleCoordinateToGPU(kIdx, 10, (uint64_t)NB_JUMP))
    return;
#endif
}

bool GPUEngine::callKernel()
{

  // Reset nbFound
  cudaMemset(outputItem, 0, 4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  comp_kangaroos<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(inputKangaroo, maxFound, outputItem, dpMask);

  cudaError_t err = cudaGetLastError();
  return CheckCudaError(err, "Kernel", true);

  return true;
}

void GPUEngine::SetParams(uint64_t dpMask, Int *distance, Int *px, Int *py)
{

  this->dpMask = dpMask;

  for (int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 2 * i, distance[i].bits64, 16);
  cudaMemcpyToSymbol(jD, jumpPinned, jumpSize / 2);
  cudaError_t err = cudaGetLastError();
  if (!CheckCudaError(err, "SetParams: Failed to copy jD to constant memory"))
  {
    return;
  }

  for (int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i, px[i].bits64, 32);
  cudaMemcpyToSymbol(jPx, jumpPinned, jumpSize);
  err = cudaGetLastError();
  if (!CheckCudaError(err, "SetParams: Failed to copy jPx to constant memory"))
  {
    return;
  }

  for (int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i, py[i].bits64, 32);
  cudaMemcpyToSymbol(jPy, jumpPinned, jumpSize);
  err = cudaGetLastError();
  if (!CheckCudaError(err, "SetParams: Failed to copy jPy to constant memory"))
  {
    return;
  }
}

bool GPUEngine::callKernelAndWait()
{

  // Debug function
  callKernel();
  cudaMemcpy(outputItemPinned, outputItem, outputSize, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  return CheckCudaError(err, "callKernelAndWait", true);

  return true;
}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound, bool spinWait)
{

  hashFound.clear();

  // Get the result

  if (spinWait)
  {

    cudaMemcpy(outputItemPinned, outputItem, outputSize, cudaMemcpyDeviceToHost);
  }
  else
  {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputItemPinned, outputItem, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady)
    {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);
  }

  cudaError_t err = cudaGetLastError();
  if (!CheckCudaError(err, "Launch"))
  {
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputItemPinned[0];
  if (nbFound > maxFound)
  {
    // prefix has been lost
    uint32_t itemsLost = nbFound - maxFound;
    if (!lostWarning)
    {
      printf("\n[!] WARNING: %u distinguished points lost!\n", itemsLost);
      printf("[*] Solutions:\n");
      printf("   1. Reduce GPU threads: -g <smaller_x>,<smaller_y>\n");
      printf("   2. Increase DP size: -d <larger_value>\n");
      printf("   3. Current buffer size: %u items\n", maxFound);
      printf("   4. Detected overflow: %u items\n", nbFound);
      lostWarning = true;
    }

    // Log periodic warnings for severe overflow
    static uint32_t warningCounter = 0;
    warningCounter++;
    if (warningCounter % 100 == 0)
    {
      printf("[!] Overflow continues: %u items lost (warning #%u)\n", itemsLost, warningCounter);
    }

    nbFound = maxFound;
  }

  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy(outputItemPinned, outputItem, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost);

  for (uint32_t i = 0; i < nbFound; i++)
  {
    uint32_t *itemPtr = outputItemPinned + (i * ITEM_SIZE32 + 1);
    ITEM it;

    it.kIdx = *((uint64_t *)(itemPtr + 12));

    uint64_t *x = (uint64_t *)itemPtr;
    it.x.bits64[0] = x[0];
    it.x.bits64[1] = x[1];
    it.x.bits64[2] = x[2];
    it.x.bits64[3] = x[3];
    it.x.bits64[4] = 0;

    uint64_t *d = (uint64_t *)(itemPtr + 8);
    it.d.bits64[0] = d[0];
    it.d.bits64[1] = d[1];
    it.d.bits64[2] = 0;
    it.d.bits64[3] = 0;
    it.d.bits64[4] = 0;
    if (it.kIdx % 2 == WILD)
      it.d.ModSubK1order(&wildOffset);

    hashFound.push_back(it);
  }

  return callKernel();
}
