/*
 * test_gpu_safety.cpp - GPU安全测试
 * 验证GPU内存管理和同步机制的修复效果
 * 
 * Copyright (c) 2025 Kangaroo Project
 */

#include <catch2/catch.hpp>
#include "../GPU/GPUMemoryGuard.h"
#include "../GPU/GPUSyncManager.h"
#include "../GPU/GPUSafeKernel.h"
#include "../ExceptionHandler.h"

// 测试GPU内存RAII封装
TEST_CASE("GPU Memory RAII", "[gpu][memory][raii]") {
    
    SECTION("Basic allocation/destruction") {
        REQUIRE_NOTHROW({
            GPUMemoryGuard<int> guard(100);
            REQUIRE(guard.get() != nullptr);
            REQUIRE(guard.size() == 100);
        });
    }
    
    SECTION("Memory exception handling") {
        REQUIRE_THROWS_AS({
            GPUMemoryGuard<int> guard(1000000000); // 超大分配
        }, GPUMemoryException);
    }
    
    SECTION("Move semantics") {
        GPUMemoryGuard<int> guard1(50);
        int* ptr1 = guard1.get();
        
        GPUMemoryGuard<int> guard2(std::move(guard1));
        
        REQUIRE(guard1.get() == nullptr); // 原对象应被移动
        REQUIRE(guard2.get() == ptr1);    // 新对象应获得资源
    }
}

// 测试GPU同步管理器
TEST_CASE("GPU Sync Manager", "[gpu][sync]") {
    
    SECTION("Basic synchronization") {
        GPUSyncManager sync_mgr;
        
        REQUIRE(sync_mgr.get_state() == GPUSyncState::READY);
        
        sync_mgr.start_kernel("test_kernel");
        REQUIRE(sync_mgr.get_state() == GPUSyncState::RUNNING);
        
        REQUIRE(sync_mgr.synchronize());
        REQUIRE(sync_mgr.get_state() == GPUSyncState::COMPLETED);
    }
    
    SECTION("Timeout handling") {
        GPUSyncManager sync_mgr;
        
        REQUIRE(sync_mgr.synchronize(100)); // 100ms超时
        REQUIRE(sync_mgr.get_state() == GPUSyncState::COMPLETED);
    }
    
    SECTION("Error state handling") {
        GPUSyncManager sync_mgr;
        
        sync_mgr.reset();
        REQUIRE(sync_mgr.get_state() == GPUSyncState::READY);
    }
}

// 测试GPU安全内核
TEST_CASE("GPU Safe Kernel", "[gpu][safe]") {
    
    SECTION("Parameter validation") {
        REQUIRE_NOTHROW({
            GPUSafeKernel::validateLaunchParameters(dim3(1,1,1), dim3(1,1,1));
        });
        
        REQUIRE_FALSE(
            GPUSafeKernel::validateLaunchParameters(dim3(999999,1,1), dim3(1,1,1))
        );
    }
    
    SECTION("Memory info retrieval") {
        size_t free, total;
        REQUIRE_NOTHROW({
            GPUSafeKernel::getMemoryInfo(free, total);
        });
        
        REQUIRE(free > 0);
        REQUIRE(total > 0);
        REQUIRE(free <= total);
    }
    
    SECTION("Memory availability check") {
        REQUIRE(GPUSafeKernel::hasEnoughMemory(1024)); // 1KB应该足够
        
        // 检查极端情况
        size_t free, total;
        GPUSafeKernel::getMemoryInfo(free, total);
        REQUIRE_FALSE(GPUSafeKernel::hasEnoughMemory(free * 2)); // 超过可用内存
    }
}

// 测试异常处理
TEST_CASE("Exception Handler", "[exception][safety]") {
    
    SECTION("Basic exception handling") {
        ExceptionHandler& handler = ExceptionHandler::get_instance();
        handler.configure(false, false); // 禁用日志记录
        
        REQUIRE_NOTHROW({
            handler.handle_exception(ExceptionInfo(
                ExceptionType::MEMORY_ERROR, "Test exception", __FILE__, __LINE__, __func__
            ));
        });
    }
    
    SECTION("CUDA error handling") {
        ExceptionHandler& handler = ExceptionHandler::get_instance();
        handler.configure(false, false);
        
        REQUIRE_NOTHROW({
            handler.handle_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__, __func__);
        });
    }
    
    SECTION("Exception guard") {
        bool executed = false;
        
        bool result = ExceptionGuard::run_safely([&]() {
            executed = true;
            // 正常执行
        }, "test_context");
        
        REQUIRE(result);
        REQUIRE(executed);
    }
    
    SECTION("Exception guard with exception") {
        bool result = ExceptionGuard::run_safely([&]() {
            throw std::runtime_error("Test exception");
        }, "test_exception");
        
        REQUIRE_FALSE(result);
    }
    
    SECTION("Safe resource management") {
        bool cleanup_called = false;
        
        {
            SafeResource<int> resource(new int(42), [&](int* p) {
                delete p;
                cleanup_called = true;
            });
            
            REQUIRE(*resource.get() == 42);
        } // 离开作用域应自动清理
        
        REQUIRE(cleanup_called);
    }
}

// 测试内存池性能
TEST_CASE("Memory Pool Performance", "[performance][memory]") {
    
    SECTION("Allocation speed comparison") {
        const int iterations = 10000;
        const size_t block_size = 128;
        
        // 测试内存池分配
        auto start = std::chrono::high_resolution_clock::now();
        {
            MemoryPool pool(block_size, iterations, iterations * 2);
            std::vector<void*> ptrs;
            
            for(int i = 0; i < iterations; ++i) {
                void* ptr = pool.allocate(block_size);
                if(ptr) ptrs.push_back(ptr);
            }
            
            for(void* ptr : ptrs) {
                pool.deallocate(ptr);
            }
        }
        auto pool_time = std::chrono::high_resolution_clock::now() - start;
        
        // 测试系统malloc分配
        start = std::chrono::high_resolution_clock::now();
        {
            std::vector<void*> ptrs;
            for(int i = 0; i < iterations; ++i) {
                void* ptr = malloc(block_size);
                if(ptr) ptrs.push_back(ptr);
            }
            
            for(void* ptr : ptrs) {
                free(ptr);
            }
        }
        auto malloc_time = std::chrono::high_resolution_clock::now() - start;
        
        // 内存池应该更快
        REQUIRE(pool_time < malloc_time * 2); // 允许2倍以内差异
    }
}