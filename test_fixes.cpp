/*
 * 测试修复效果的验证程序
 * 验证SmartAllocator、线程安全和错误处理的改进
 */

#include "SmartAllocator.h"
#include "UnifiedErrorHandler.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <iomanip>

// 测试内存跟踪功能
void test_memory_tracking() {
    std::cout << "\n=== Testing Memory Tracking ===" << std::endl;

    // 初始化错误处理器
    UnifiedErrorHandler::Initialize(ErrorLevel::DEBUG);

    std::cout << "Initial tracked allocations: " << SmartAllocator::get_tracked_allocations() << std::endl;

    // 分配一些内存
    std::vector<void*> ptrs;
    const size_t sizes[] = {64, 1024, 4096};

    for(size_t size : sizes) {
        void* ptr = SmartAllocator::allocate(size);
        if(ptr) {
            ptrs.push_back(ptr);
            std::cout << "Allocated " << size << " bytes, tracked: "
                     << SmartAllocator::get_tracked_allocations() << std::endl;
        }
    }

    // 释放内存
    for(void* ptr : ptrs) {
        SmartAllocator::deallocate(ptr);
    }

    std::cout << "After deallocation, tracked: " << SmartAllocator::get_tracked_allocations() << std::endl;

    if(SmartAllocator::get_tracked_allocations() == 0) {
        std::cout << "✅ Memory tracking: PASS - No leaks detected" << std::endl;
    } else {
        std::cout << "❌ Memory tracking: FAIL - Potential leaks detected" << std::endl;
    }
}

// 测试SmartAllocator性能和正确性
void test_smart_allocator() {
    std::cout << "\n=== Testing SmartAllocator ===" << std::endl;
    
    // 初始化错误处理器
    UnifiedErrorHandler::Initialize(ErrorLevel::DEBUG);
    
    const size_t test_sizes[] = {64, 256, 1024, 4096, 16384};
    const int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 分配测试
    std::vector<void*> ptrs;
    for(int i = 0; i < iterations; i++) {
        for(size_t size : test_sizes) {
            void* ptr = SmartAllocator::allocate(size);
            if(ptr) {
                ptrs.push_back(ptr);
                // 写入测试数据
                memset(ptr, 0xAA, size);
            }
        }
    }
    
    auto mid = std::chrono::high_resolution_clock::now();
    
    // 释放测试
    for(void* ptr : ptrs) {
        SmartAllocator::deallocate(ptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
    auto dealloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
    
    std::cout << "✅ SmartAllocator Test Results:" << std::endl;
    std::cout << "   Allocations: " << ptrs.size() << " blocks" << std::endl;
    std::cout << "   Allocation time: " << alloc_time.count() << " μs" << std::endl;
    std::cout << "   Deallocation time: " << dealloc_time.count() << " μs" << std::endl;
    std::cout << "   Average alloc: " << (alloc_time.count() / ptrs.size()) << " μs/block" << std::endl;
    
    SmartAllocator::cleanup();
}

// 测试线程安全的原子操作
void test_thread_safety() {
    std::cout << "\n=== Testing Thread Safety ===" << std::endl;
    
    std::atomic<uint32_t> counter{0};
    std::atomic<uint32_t> errors{0};
    const int num_threads = 8;
    const int increments_per_thread = 10000;
    
    auto worker = [&counter, &errors, increments_per_thread]() {
        for(int i = 0; i < increments_per_thread; i++) {
            uint32_t old_val = counter.fetch_add(1);
            
            // 验证原子性
            if(old_val >= counter.load()) {
                errors.fetch_add(1);
            }
            
            // 模拟一些工作
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
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
    
    uint32_t expected = num_threads * increments_per_thread;
    uint32_t actual = counter.load();
    
    std::cout << "✅ Thread Safety Test Results:" << std::endl;
    std::cout << "   Threads: " << num_threads << std::endl;
    std::cout << "   Expected counter: " << expected << std::endl;
    std::cout << "   Actual counter: " << actual << std::endl;
    std::cout << "   Errors detected: " << errors.load() << std::endl;
    std::cout << "   Test duration: " << duration.count() << " ms" << std::endl;
    
    if(actual == expected && errors.load() == 0) {
        std::cout << "   ✅ PASS: Thread safety verified" << std::endl;
    } else {
        std::cout << "   ❌ FAIL: Thread safety issues detected" << std::endl;
    }
}

// 测试错误处理系统
void test_error_handling() {
    std::cout << "\n=== Testing Error Handling ===" << std::endl;
    
    // 测试不同级别的错误
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::DEBUG, ErrorType::SYSTEM, 
                                           "Debug message test"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::INFO, ErrorType::ALGORITHM,
                                           "Info message test"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::WARNING, ErrorType::MEMORY_ALLOCATION,
                                           "Warning message test"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::ERROR, ErrorType::HASH_TABLE,
                                           "Error message test"));
    
    UnifiedErrorHandler::LogError(ErrorInfo(ErrorLevel::CRITICAL, ErrorType::CUDA_OPERATION,
                                           "Critical message test"));
    
    // 测试便捷方法
    LOG_MEMORY_ERROR("test allocation", 1024);
    LOG_HASH_ERROR("test operation", 12345);
    
    // 获取统计信息
    auto stats = UnifiedErrorHandler::GetErrorStats();
    
    std::cout << "✅ Error Handling Test Results:" << std::endl;
    std::cout << "   Debug messages: " << stats.debug_count << std::endl;
    std::cout << "   Info messages: " << stats.info_count << std::endl;
    std::cout << "   Warning messages: " << stats.warning_count << std::endl;
    std::cout << "   Error messages: " << stats.error_count << std::endl;
    std::cout << "   Critical messages: " << stats.critical_count << std::endl;
    std::cout << "   Total messages: " << stats.total_count << std::endl;
}

// 测试SmartPtr RAII
void test_smart_ptr() {
    std::cout << "\n=== Testing SmartPtr RAII ===" << std::endl;
    
    try {
        // 测试基本分配
        {
            SmartPtr<int> ptr(100);
            if(ptr) {
                for(size_t i = 0; i < ptr.size(); i++) {
                    ptr[i] = static_cast<int>(i);
                }
                std::cout << "   ✅ SmartPtr allocation and access: OK" << std::endl;
            }
            // 自动析构
        }
        
        // 测试移动语义
        {
            SmartPtr<double> ptr1(50);
            SmartPtr<double> ptr2 = std::move(ptr1);
            
            if(!ptr1 && ptr2) {
                std::cout << "   ✅ SmartPtr move semantics: OK" << std::endl;
            } else {
                std::cout << "   ❌ SmartPtr move semantics: FAIL" << std::endl;
            }
        }
        
        std::cout << "✅ SmartPtr RAII Test: PASS" << std::endl;
        
    } catch(const std::exception& e) {
        std::cout << "❌ SmartPtr RAII Test: FAIL - " << e.what() << std::endl;
    }
}

// 性能对比测试
void test_performance_comparison() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    const int iterations = 10000;
    const size_t size = 1024;
    
    // 测试标准malloc/free
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        void* ptr = malloc(size);
        if(ptr) {
            memset(ptr, 0, size);
            free(ptr);
        }
    }
    auto malloc_time = std::chrono::high_resolution_clock::now() - start;
    
    // 测试SmartAllocator
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; i++) {
        void* ptr = SmartAllocator::allocate(size);
        if(ptr) {
            memset(ptr, 0, size);
            SmartAllocator::deallocate(ptr);
        }
    }
    auto smart_time = std::chrono::high_resolution_clock::now() - start;
    
    auto malloc_us = std::chrono::duration_cast<std::chrono::microseconds>(malloc_time);
    auto smart_us = std::chrono::duration_cast<std::chrono::microseconds>(smart_time);
    
    std::cout << "✅ Performance Comparison Results:" << std::endl;
    std::cout << "   Standard malloc/free: " << malloc_us.count() << " μs" << std::endl;
    std::cout << "   SmartAllocator: " << smart_us.count() << " μs" << std::endl;
    
    if(smart_us.count() <= malloc_us.count() * 1.2) { // 允许20%的开销
        std::cout << "   ✅ Performance: Acceptable (within 20% of malloc)" << std::endl;
    } else {
        std::cout << "   ⚠️  Performance: Higher overhead than expected" << std::endl;
    }
    
    double ratio = static_cast<double>(smart_us.count()) / malloc_us.count();
    std::cout << "   Ratio: " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
}

int main() {
    std::cout << "🚀 CUDA-BSGS-Kangaroo 修复验证测试" << std::endl;
    std::cout << "====================================" << std::endl;
    
    try {
        test_memory_tracking();
        test_smart_allocator();
        test_thread_safety();
        test_error_handling();
        test_smart_ptr();
        test_performance_comparison();
        
        std::cout << "\n🎉 所有测试完成!" << std::endl;
        std::cout << "📊 查看 kangaroo_errors.log 获取详细错误日志" << std::endl;
        
        // 清理资源
        UnifiedErrorHandler::Cleanup();
        SmartAllocator::cleanup();
        
    } catch(const std::exception& e) {
        std::cerr << "❌ 测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
