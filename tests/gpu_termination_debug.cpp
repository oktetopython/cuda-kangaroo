/**
 * @file gpu_termination_debug.cpp
 * @brief GPU Termination Debug Test
 * 
 * This test specifically diagnoses the GPU termination issue
 * by running a controlled GPU search with detailed logging.
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <cstdio>

#include "../UTF8Console.h"
#include "../GPU/GPUEngine.h"
#include "../SECPK1/Int.h"
#include "../SECPK1/Point.h"
#include "../SECPK1/SECP256k1.h"

/**
 * @brief Test GPU search behavior with detailed logging
 */
bool TestGPUSearchBehavior() {
    printf("🔍 Testing GPU Search Behavior...\n");
    
    try {
        // Initialize SECP256K1
        Secp256K1* secp = new Secp256K1();
        secp->Init();
        printf("✅ SECP256K1 initialized\n");
        
        // Create GPU Engine
        printf("Creating GPU Engine...\n");
        GPUEngine* gpu = new GPUEngine(136, 128, 0, 65536 * 2);
        printf("✅ GPU Engine created: %s\n", gpu->deviceName.c_str());
        
        // Create test kangaroos
        const int NUM_KANGAROOS = 2048;
        Int* px = new Int[NUM_KANGAROOS];
        Int* py = new Int[NUM_KANGAROOS];
        Int* distance = new Int[NUM_KANGAROOS];
        
        printf("Initializing %d test kangaroos...\n", NUM_KANGAROOS);
        for(int i = 0; i < NUM_KANGAROOS; i++) {
            // Create simple test kangaroos
            distance[i].SetInt32(i + 1);
            Point p = secp->ComputePublicKey(&distance[i]);
            px[i].Set(&p.x);
            py[i].Set(&p.y);
        }
        printf("✅ Test kangaroos initialized\n");
        
        // Set up GPU parameters
        printf("Setting up GPU parameters...\n");
        Int jumpDistance[32];
        Int jumpPointx[32];
        Int jumpPointy[32];
        
        for(int i = 0; i < 32; i++) {
            jumpDistance[i].SetInt32((i + 1) * 1000);
            Point jp = secp->ComputePublicKey(&jumpDistance[i]);
            jumpPointx[i].Set(&jp.x);
            jumpPointy[i].Set(&jp.y);
        }
        
        uint64_t dpMask = 0x0; // No DP for this test
        gpu->SetParams(dpMask, jumpDistance, jumpPointx, jumpPointy);
        printf("✅ GPU parameters set\n");
        
        // Set kangaroos
        printf("Setting kangaroos on GPU...\n");
        gpu->SetKangaroos(px, py, distance);
        printf("✅ Kangaroos set on GPU\n");
        
        // Test initial kernel call
        printf("Testing initial kernel call...\n");
        bool kernelResult = gpu->callKernel();
        if(!kernelResult) {
            printf("❌ Initial kernel call failed!\n");
            delete[] px;
            delete[] py;
            delete[] distance;
            delete gpu;
            delete secp;
            return false;
        }
        printf("✅ Initial kernel call successful\n");
        
        // Test search loop behavior
        printf("Testing search loop behavior (10 iterations)...\n");
        std::vector<ITEM> found;
        bool searchSuccess = true;
        
        for(int iteration = 0; iteration < 10; iteration++) {
            printf("  Iteration %d: Launching GPU...\n", iteration + 1);
            
            auto start = std::chrono::high_resolution_clock::now();
            bool launchResult = gpu->Launch(found, false);
            auto end = std::chrono::high_resolution_clock::now();
            
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            if(!launchResult) {
                printf("  ❌ GPU Launch failed at iteration %d\n", iteration + 1);
                searchSuccess = false;
                break;
            }
            
            printf("  ✅ Iteration %d completed in %.3f seconds, found %zu items\n", 
                   iteration + 1, elapsed, found.size());
            
            if(found.size() > 0) {
                printf("  🎯 Found items in iteration %d!\n", iteration + 1);
                for(size_t i = 0; i < found.size(); i++) {
                    printf("    Item %zu: kIdx=%llu\n", i, (unsigned long long)found[i].kIdx);
                }
            }
            
            // Brief pause between iterations
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        if(searchSuccess) {
            printf("✅ Search loop completed successfully\n");
        } else {
            printf("❌ Search loop failed\n");
        }
        
        // Cleanup
        delete[] px;
        delete[] py;
        delete[] distance;
        delete gpu;
        delete secp;
        
        return searchSuccess;
        
    } catch(const std::exception& e) {
        printf("❌ GPU search test failed with exception: %s\n", e.what());
        return false;
    } catch(...) {
        printf("❌ GPU search test failed with unknown exception\n");
        return false;
    }
}

/**
 * @brief Test GPU behavior with different configurations
 */
bool TestGPUConfigurations() {
    printf("🔍 Testing GPU Configurations...\n");
    
    struct TestConfig {
        int gridX, gridY;
        const char* name;
    };
    
    TestConfig configs[] = {
        {8, 8, "Small (8x8)"},
        {32, 32, "Medium (32x32)"},
        {68, 64, "Large (68x64)"},
        {136, 128, "Full (136x128)"}
    };
    
    for(int i = 0; i < 4; i++) {
        printf("  Testing %s configuration...\n", configs[i].name);
        
        try {
            GPUEngine* gpu = new GPUEngine(configs[i].gridX, configs[i].gridY, 0, 1024);
            printf("    ✅ GPU Engine created for %s\n", configs[i].name);
            
            bool kernelResult = gpu->callKernel();
            if(kernelResult) {
                printf("    ✅ Kernel call successful for %s\n", configs[i].name);
            } else {
                printf("    ❌ Kernel call failed for %s\n", configs[i].name);
            }
            
            delete gpu;
            
        } catch(const std::exception& e) {
            printf("    ❌ %s configuration failed: %s\n", configs[i].name, e.what());
            return false;
        }
    }
    
    printf("✅ All GPU configurations tested\n");
    return true;
}

/**
 * @brief Main diagnostic function
 */
int main() {
    // Initialize UTF-8 console
    INIT_UTF8_CONSOLE();
    
    printf("🚀 GPU Termination Debug Test\n");
    printf("🔧 Diagnosing GPU Search Termination Issues\n");
    printf("============================================================================\n");
    
    int passed = 0, total = 2;
    
    // Test 1: GPU Configurations
    if(TestGPUConfigurations()) {
        passed++;
    }
    printf("\n");
    
    // Test 2: GPU Search Behavior
    if(TestGPUSearchBehavior()) {
        passed++;
    }
    printf("\n");
    
    // Results
    printf("============================================================================\n");
    printf("GPU Termination Debug Results: %d/%d tests passed\n", passed, total);
    
    if(passed == total) {
        printf("✅ GPU is working correctly - termination issue may be elsewhere\n");
    } else {
        printf("❌ GPU issues detected - this is the root cause\n");
    }
    
    printf("============================================================================\n");
    
    return (passed == total) ? 0 : 1;
}
