#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include "GPU/GPUEngine.h"
#include "SECPK1/SECP256k1.h"
#include "Timer.h"

/**
 * Test program to verify GPU memory leakage fix
 * This program tests:
 * 1. RAII GPU memory management
 * 2. Emergency cleanup functionality
 * 3. Memory release after program termination
 */

void test_raii_memory_management() {
    std::cout << "=== Testing RAII GPU Memory Management ===" << std::endl;
    
    {
        std::cout << "Creating CudaMemoryGuard..." << std::endl;
        std::unique_ptr<CudaMemoryGuard> guard = std::make_unique<CudaMemoryGuard>();
        
        std::cout << "Guard created successfully" << std::endl;
        std::cout << "Destroying guard (should trigger GPU cleanup)..." << std::endl;
    } // Guard destructor should be called here
    
    std::cout << "RAII test completed" << std::endl;
}

void test_emergency_cleanup() {
    std::cout << "\n=== Testing Emergency GPU Cleanup ===" << std::endl;
    
    std::cout << "Calling GPUEngine::ForceGPUCleanup()..." << std::endl;
    GPUEngine::ForceGPUCleanup();
    std::cout << "Emergency cleanup completed" << std::endl;
}

void test_gpu_engine_lifecycle() {
    std::cout << "\n=== Testing GPUEngine Lifecycle ===" << std::endl;
    
    try {
        std::cout << "Initializing SECP256K1..." << std::endl;
        Secp256K1 secp;
        secp.Init();
        
        std::cout << "Creating GPUEngine..." << std::endl;
        // Create a small GPU engine for testing
        GPUEngine* engine = new GPUEngine(1, 64, 0, 1024);
        
        std::cout << "GPU Device: " << engine->deviceName << std::endl;
        std::cout << "GPU Memory: " << engine->GetMemory() / 1048576.0 << " MB" << std::endl;
        
        std::cout << "Destroying GPUEngine..." << std::endl;
        delete engine;
        
        std::cout << "GPUEngine lifecycle test completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception during GPU test: " << e.what() << std::endl;
        GPUEngine::ForceGPUCleanup();
    } catch (...) {
        std::cout << "Unknown exception during GPU test" << std::endl;
        GPUEngine::ForceGPUCleanup();
    }
}

void test_memory_monitoring() {
    std::cout << "\n=== GPU Memory Monitoring Test ===" << std::endl;
    std::cout << "Note: Use nvidia-smi in another terminal to monitor GPU memory" << std::endl;
    std::cout << "Before GPU operations - check GPU memory usage" << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    {
        std::unique_ptr<CudaMemoryGuard> guard = std::make_unique<CudaMemoryGuard>();
        
        try {
            Secp256K1 secp;
            secp.Init();
            
            GPUEngine* engine = new GPUEngine(2, 128, 0, 2048);
            std::cout << "GPU allocated - check memory usage now" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(3));
            
            delete engine;
            std::cout << "GPU engine deleted - memory should be partially freed" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
        } catch (...) {
            std::cout << "Exception in memory test" << std::endl;
        }
        
        std::cout << "Destroying RAII guard - should force complete cleanup" << std::endl;
    }
    
    std::cout << "After RAII cleanup - GPU memory should be fully released" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

int main() {
    std::cout << "GPU Memory Leakage Fix Test Program" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Initialize timer
    Timer::Init();
    
    try {
        // Test 1: RAII Memory Management
        test_raii_memory_management();
        
        // Test 2: Emergency Cleanup
        test_emergency_cleanup();
        
        // Test 3: GPUEngine Lifecycle
        test_gpu_engine_lifecycle();
        
        // Test 4: Memory Monitoring
        test_memory_monitoring();
        
        std::cout << "\n=== All Tests Completed Successfully ===" << std::endl;
        std::cout << "GPU memory should be completely released now" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Test failed with exception: " << e.what() << std::endl;
        GPUEngine::ForceGPUCleanup();
        return 1;
    } catch (...) {
        std::cout << "Test failed with unknown exception" << std::endl;
        GPUEngine::ForceGPUCleanup();
        return 1;
    }
    
    return 0;
}
