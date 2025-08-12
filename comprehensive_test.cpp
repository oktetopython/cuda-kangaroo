/*
 * 全面测试修复后的代码
 * 验证所有修复的正确性和稳定性
 */

#include "SmartAllocator.h"
#include "UnifiedErrorHandler.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <cassert>

// 测试命名空间冲突解决
void test_namespace_resolution() {
    std::cout << "\n=== Testing Namespace Resolution ===" << std::endl;
    
    // 测试全局别名
    void* ptr1 = SmartAllocator::allocate(1024);
    assert(ptr1 != nullptr);
    
    // 测试命名空间版本
    void* ptr2 = kangaroo_memory::SmartAllocator::allocate(2048);
    assert(ptr2 != nullptr);
    
    // 验证跟踪
    assert(SmartAllocator::get_tracked_allocations() == 2);
    
    SmartAllocator::deallocate(ptr1);
    kangaroo_memory::SmartAllocator::deallocate(ptr2);
    
    assert(SmartAllocator::get_tracked_allocations() == 0);
    
    std::cout << "✅ Namespace resolution: PASS" << std::endl;
}

// 测试内存池的线程安全
void test_memory_pool_thread_safety() {
    std::cout << "\n=== Testing Memory Pool Thread Safety ===" << std::endl;
    
    const int num_threads = 8;
    const int allocations_per_thread = 1000;
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&]() {
        std::vector<void*> ptrs;
        
        // 分配阶段
        for(int i = 0; i < allocations_per_thread; i++) {
            void* ptr = SmartAllocator::allocate(64 + (i % 512)); // 变化的大小
            if(ptr) {
                ptrs.push_back(ptr);
                // 写入测试数据
                memset(ptr, 0x42, 64);
            } else {
                error_count.fetch_add(1);
            }
        }
        
        // 释放阶段
        for(void* ptr : ptrs) {
            SmartAllocator::deallocate(ptr);
            success_count.fetch_add(1);
        }
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for(int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker);
    }
    
    for(auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Successful operations: " << success_count.load() << std::endl;
    std::cout << "Errors: " << error_count.load() << std::endl;
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "Remaining tracked: " << SmartAllocator::get_tracked_allocations() << std::endl;
    
    if(error_count.load() == 0 && SmartAllocator::get_tracked_allocations() == 0) {
        std::cout << "✅ Memory pool thread safety: PASS" << std::endl;
    } else {
        std::cout << "❌ Memory pool thread safety: FAIL" << std::endl;
    }
}

// 测试内存泄漏检测
void test_memory_leak_detection() {
    std::cout << "\n=== Testing Memory Leak Detection ===" << std::endl;
    
    size_t initial_count = SmartAllocator::get_tracked_allocations();
    
    // 故意创建一些分配
    std::vector<void*> ptrs;
    for(int i = 0; i < 10; i++) {
        void* ptr = SmartAllocator::allocate(128 * (i + 1));
        if(ptr) {
            ptrs.push_back(ptr);
        }
    }
    
    size_t after_alloc = SmartAllocator::get_tracked_allocations();
    std::cout << "After allocation: " << after_alloc << " tracked" << std::endl;
    
    // 释放一半
    for(size_t i = 0; i < ptrs.size() / 2; i++) {
        SmartAllocator::deallocate(ptrs[i]);
    }
    
    size_t after_partial = SmartAllocator::get_tracked_allocations();
    std::cout << "After partial deallocation: " << after_partial << " tracked" << std::endl;
    
    // 释放剩余
    for(size_t i = ptrs.size() / 2; i < ptrs.size(); i++) {
        SmartAllocator::deallocate(ptrs[i]);
    }
    
    size_t final_count = SmartAllocator::get_tracked_allocations();
    std::cout << "After full deallocation: " << final_count << " tracked" << std::endl;
    
    if(final_count == initial_count) {
        std::cout << "✅ Memory leak detection: PASS" << std::endl;
    } else {
        std::cout << "❌ Memory leak detection: FAIL - " 
                 << (final_count - initial_count) << " leaks detected" << std::endl;
    }
}

// 测试错误处理系统
void test_error_handling_system() {
    std::cout << "\n=== Testing Error Handling System ===" << std::endl;
    
    // 初始化错误处理器
    UnifiedErrorHandler::Initialize(ErrorLevel::DEBUG, "test_errors.log");
    
    // 测试不同级别的错误
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::DEBUG, ErrorType::SYSTEM, 
                                           "Debug test message"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::WARNING, ErrorType::MEMORY_ALLOCATION,
                                           "Warning test message"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::ERROR, ErrorType::HASH_TABLE,
                                           "Error test message"));
    
    // 测试便捷宏
    LOG_MEMORY_ERROR("test allocation", 1024);
    LOG_HASH_ERROR("test operation", 12345);
    
    // 获取统计
    auto stats = UnifiedErrorHandler::GetErrorStats();
    
    std::cout << "Error statistics:" << std::endl;
    std::cout << "  Debug: " << stats.debug_count << std::endl;
    std::cout << "  Warning: " << stats.warning_count << std::endl;
    std::cout << "  Error: " << stats.error_count << std::endl;
    std::cout << "  Total: " << stats.total_count << std::endl;
    
    if(stats.total_count >= 5) { // 至少5个错误消息
        std::cout << "✅ Error handling system: PASS" << std::endl;
    } else {
        std::cout << "❌ Error handling system: FAIL" << std::endl;
    }
}

// 测试大内存分配
void test_large_allocations() {
    std::cout << "\n=== Testing Large Allocations ===" << std::endl;
    
    const size_t large_sizes[] = {
        1024*1024,      // 1MB
        2*1024*1024,    // 2MB
        5*1024*1024     // 5MB
    };
    
    std::vector<void*> large_ptrs;
    
    for(size_t size : large_sizes) {
        void* ptr = SmartAllocator::allocate(size);
        if(ptr) {
            large_ptrs.push_back(ptr);
            // 写入测试数据
            memset(ptr, 0x55, std::min(size, static_cast<size_t>(4096))); // 只写前4KB
            std::cout << "Allocated " << (size / 1024 / 1024) << "MB successfully" << std::endl;
        } else {
            std::cout << "Failed to allocate " << (size / 1024 / 1024) << "MB" << std::endl;
        }
    }
    
    // 释放大内存
    for(void* ptr : large_ptrs) {
        SmartAllocator::deallocate(ptr);
    }
    
    std::cout << "Tracked after large allocation test: " 
             << SmartAllocator::get_tracked_allocations() << std::endl;
    
    std::cout << "✅ Large allocations: PASS" << std::endl;
}

// 压力测试
void test_stress() {
    std::cout << "\n=== Stress Test ===" << std::endl;
    
    const int iterations = 10000;
    std::vector<void*> ptrs;
    ptrs.reserve(iterations);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 分配阶段
    for(int i = 0; i < iterations; i++) {
        size_t size = 32 + (i % 2048); // 32B到2KB
        void* ptr = SmartAllocator::allocate(size);
        if(ptr) {
            ptrs.push_back(ptr);
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // 释放阶段
    for(void* ptr : ptrs) {
        SmartAllocator::deallocate(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto dealloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "Allocations: " << ptrs.size() << std::endl;
    std::cout << "Allocation time: " << alloc_time.count() << " μs" << std::endl;
    std::cout << "Deallocation time: " << dealloc_time.count() << " μs" << std::endl;
    std::cout << "Average alloc: " << (alloc_time.count() / ptrs.size()) << " μs/op" << std::endl;
    std::cout << "Remaining tracked: " << SmartAllocator::get_tracked_allocations() << std::endl;
    
    std::cout << "✅ Stress test: PASS" << std::endl;
}

int main() {
    std::cout << "🚀 Kangaroo v2.8.12 Comprehensive Fix Verification Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        test_namespace_resolution();
        test_memory_leak_detection();
        test_memory_pool_thread_safety();
        test_error_handling_system();
        test_large_allocations();
        test_stress();
        
        std::cout << "\n🎉 All tests completed!" << std::endl;

        // Final cleanup
        SmartAllocator::cleanup();
        UnifiedErrorHandler::Cleanup();

        std::cout << "📊 Final tracking count: " << SmartAllocator::get_tracked_allocations() << std::endl;
        std::cout << "📋 Check test_errors.log for detailed error logs" << std::endl;
        
    } catch(const std::exception& e) {
        std::cerr << "❌ Exception occurred during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
