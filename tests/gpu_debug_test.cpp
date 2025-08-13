/**
 * @file gpu_debug_test.cpp
 * @brief GPU Debug Test to Identify Kangaroo Creation Issues
 *
 * This test isolates the GPU kangaroo creation process to identify
 * the root cause of the silent termination bug.
 */

#include <iostream>
#include <chrono>
#include <cstdio>

#include "../UTF8Console.h"
#include "../GPU/GPUEngine.h"
#include "../SECPK1/Int.h"
#include "../SECPK1/Point.h"
#include "../SECPK1/SECP256k1.h"

/**
 * @brief Test GPU Engine initialization and basic operations
 */
bool TestGPUEngineBasic()
{
  printf("🔍 Testing GPU Engine Basic Operations...\n");

  try
  {
    // Test 1: GPU Engine Creation
    printf("  Step 1: Creating GPU Engine...\n");
    GPUEngine *gpu = new GPUEngine(136, 128, 0, 65536 * 2);
    printf("  ✅ GPU Engine created successfully\n");
    printf("  GPU: %s\n", gpu->deviceName.c_str());
    printf("  Memory: %.1f MB\n", gpu->GetMemory() / 1048576.0);

    // Test 2: Basic GPU Operations
    printf("  Step 2: Testing basic GPU operations...\n");
    int nbThread = gpu->GetNbThread();
    printf("  GPU Threads: %d\n", nbThread);

    // Test 3: Memory Allocation Test
    printf("  Step 3: Testing memory allocation...\n");
    const int TEST_SIZE = 1000;
    Int *testPx = new Int[TEST_SIZE];
    Int *testPy = new Int[TEST_SIZE];
    Int *testD = new Int[TEST_SIZE];

    // Initialize test data
    for (int i = 0; i < TEST_SIZE; i++)
    {
      testPx[i].SetInt32(i + 1);
      testPy[i].SetInt32(i + 2);
      testD[i].SetInt32(i + 3);
    }

    printf("  ✅ Test data initialized\n");

    // Test 4: SetKangaroos (this might be where it fails)
    printf("  Step 4: Testing SetKangaroos...\n");
    gpu->SetKangaroos(testPx, testPy, testD);
    printf("  ✅ SetKangaroos completed\n");

    // Test 5: Basic kernel call (this is likely where it crashes)
    printf("  Step 5: Testing basic kernel call...\n");
    bool kernelResult = gpu->callKernel();
    if (kernelResult)
    {
      printf("  ✅ Kernel call successful\n");
    }
    else
    {
      printf("  ❌ Kernel call failed\n");
      delete[] testPx;
      delete[] testPy;
      delete[] testD;
      delete gpu;
      return false;
    }

    // Cleanup
    delete[] testPx;
    delete[] testPy;
    delete[] testD;
    delete gpu;

    printf("  ✅ GPU Engine basic test completed successfully\n");
    return true;
  }
  catch (const std::exception &e)
  {
    printf("  ❌ GPU Engine test failed with exception: %s\n", e.what());
    return false;
  }
  catch (...)
  {
    printf("  ❌ GPU Engine test failed with unknown exception\n");
    return false;
  }
}

/**
 * @brief Test GPU Engine with smaller configurations
 */
bool TestGPUEngineSmall()
{
  printf("🔍 Testing GPU Engine with Small Configuration...\n");

  try
  {
    // Use much smaller configuration
    printf("  Creating small GPU Engine (8x8)...\n");
    GPUEngine *gpu = new GPUEngine(8, 8, 0, 1024);
    printf("  ✅ Small GPU Engine created\n");

    // Test with minimal data
    const int SMALL_SIZE = 64;
    Int *testPx = new Int[SMALL_SIZE];
    Int *testPy = new Int[SMALL_SIZE];
    Int *testD = new Int[SMALL_SIZE];

    for (int i = 0; i < SMALL_SIZE; i++)
    {
      testPx[i].SetInt32(1);
      testPy[i].SetInt32(1);
      testD[i].SetInt32(1);
    }

    printf("  Setting small kangaroo data...\n");
    gpu->SetKangaroos(testPx, testPy, testD);
    printf("  ✅ Small kangaroo data set\n");

    printf("  Testing small kernel call...\n");
    bool result = gpu->callKernel();

    delete[] testPx;
    delete[] testPy;
    delete[] testD;
    delete gpu;

    if (result)
    {
      printf("  ✅ Small GPU test successful\n");
      return true;
    }
    else
    {
      printf("  ❌ Small GPU test failed\n");
      return false;
    }
  }
  catch (const std::exception &e)
  {
    printf("  ❌ Small GPU test failed with exception: %s\n", e.what());
    return false;
  }
  catch (...)
  {
    printf("  ❌ Small GPU test failed with unknown exception\n");
    return false;
  }
}

/**
 * @brief Test CUDA basic functionality
 */
bool TestCUDABasic()
{
  printf("🔍 Testing CUDA Basic Functionality...\n");

  try
  {
    // Test CUDA device count
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
    {
      printf("  ❌ CUDA GetDeviceCount failed: %s\n", cudaGetErrorString(error));
      return false;
    }
    printf("  CUDA Devices: %d\n", deviceCount);

    if (deviceCount == 0)
    {
      printf("  ❌ No CUDA devices available\n");
      return false;
    }

    // Test device properties
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess)
    {
      printf("  ❌ CUDA GetDeviceProperties failed: %s\n", cudaGetErrorString(error));
      return false;
    }

    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.1f MB\n", prop.totalGlobalMem / 1048576.0);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);

    // Test basic memory allocation with comprehensive error checking
    void *testPtr = nullptr;
    error = cudaMalloc(&testPtr, 1024);
    if (error != cudaSuccess)
    {
      printf("  ❌ CUDA memory allocation failed: %s\n", cudaGetErrorString(error));
      return false;
    }

    // Safe memory deallocation with error checking
    error = cudaFree(testPtr);
    if (error != cudaSuccess)
    {
      printf("  ⚠️ Warning: CUDA memory deallocation failed: %s\n", cudaGetErrorString(error));
      // Continue execution - this is not critical for the test
    }
    printf("  ✅ CUDA basic functionality test passed\n");
    return true;
  }
  catch (const std::exception &e)
  {
    printf("  ❌ CUDA basic test failed with exception: %s\n", e.what());
    return false;
  }
  catch (...)
  {
    printf("  ❌ CUDA basic test failed with unknown exception\n");
    return false;
  }
}

/**
 * @brief Main test function
 */
int main()
{
  // Initialize UTF-8 console
  INIT_UTF8_CONSOLE();

  printf("🚀 GPU Debug Test Suite\n");
  printf("🔧 Diagnosing Kangaroo GPU Creation Issues\n");
  printf("============================================================================\n");

  int passed = 0, total = 3;

  // Test 1: CUDA Basic
  if (TestCUDABasic())
  {
    passed++;
  }
  printf("\n");

  // Test 2: GPU Engine Small
  if (TestGPUEngineSmall())
  {
    passed++;
  }
  printf("\n");

  // Test 3: GPU Engine Basic
  if (TestGPUEngineBasic())
  {
    passed++;
  }
  printf("\n");

  // Results
  printf("============================================================================\n");
  printf("GPU Debug Test Results: %d/%d tests passed\n", passed, total);

  if (passed == total)
  {
    printf("✅ All GPU tests passed - issue may be in kangaroo creation logic\n");
  }
  else
  {
    printf("❌ GPU tests failed - root cause identified\n");
  }

  printf("============================================================================\n");

  return (passed == total) ? 0 : 1;
}
