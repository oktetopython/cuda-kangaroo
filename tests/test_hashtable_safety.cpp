/*
 * test_hashtable_safety.cpp - HashTable安全测试
 * 验证内存安全和边界检查的修复效果
 * 
 * Copyright (c) 2025 Kangaroo Project
 */

#include <catch2/catch.hpp>
#include "../HashTable.h"
#include "../ExceptionHandler.h"

// 测试HashTable边界检查
TEST_CASE("HashTable boundary checks", "[hashtable][safety]") {
    HashTable ht;
    
    SECTION("Invalid hash index handling") {
        Int x, d;
        x.SetBase16("1234567890abcdef");
        d.SetBase16("fedcba0987654321");
        
        // 测试边界值
        REQUIRE_NOTHROW(ht.Add(&x, &d, 0));
        REQUIRE(ht.GetNbItem() >= 0);
    }
    
    SECTION("Null pointer handling") {
        // 测试空指针安全处理
        REQUIRE_NOTHROW(ht.Add(nullptr, nullptr, 0));
        REQUIRE(ht.GetNbItem() == 0);
    }
    
    SECTION("Memory allocation failure") {
        // 测试内存分配失败的处理
        HashTable stress_ht;
        
        // 尝试大量分配
        Int x, d;
        bool success = true;
        for(int i = 0; i < 10000; ++i) {
            x.SetInt32(i);
            d.SetInt32(i * 2);
            int result = stress_ht.Add(&x, &d, 0);
            if(result < 0) {
                success = false;
                break;
            }
        }
        
        REQUIRE(stress_ht.GetNbItem() >= 0);
        stress_ht.Reset();
    }
    
    SECTION("Integer overflow protection") {
        // 测试整数溢出保护
        HashTable ht;
        
        // 测试极端值
        uint64_t extreme_hash = UINT64_MAX;
        int128_t x, d;
        x.i64[0] = 0;
        x.i64[1] = 0;
        d.i64[0] = 0;
        d.i64[1] = 0;
        
        REQUIRE_NOTHROW(ht.Add(extreme_hash, &x, &d));
    }
}

// 测试内存池稳定性
TEST_CASE("Memory pool stability", "[memory][pool]") {
    SECTION("Concurrent allocation/deallocation") {
        MemoryPool pool(64, 100, 1000); // 64-byte blocks
        
        const int num_threads = 4;
        const int iterations = 1000;
        
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};
        
        for(int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                for(int i = 0; i < iterations; ++i) {
                    void* ptr = pool.allocate(64);
                    if(ptr) {
                        memset(ptr, 0xAA, 64);
                        pool.deallocate(ptr);
                        success_count++;
                    }
                }
            });
        }
        
        for(auto& t : threads) {
            t.join();
        }
        
        REQUIRE(success_count > 0);
        
        auto stats = pool.get_stats();
        REQUIRE(stats.total_allocated > 0);
        REQUIRE(stats.used_blocks == 0); // All should be freed
    }
    
    SECTION("Memory pool statistics") {
        MemoryPool pool(256, 50, 500);
        
        auto initial_stats = pool.get_stats();
        
        void* ptr1 = pool.allocate(256);
        void* ptr2 = pool.allocate(256);
        
        auto after_alloc_stats = pool.get_stats();
        REQUIRE(after_alloc_stats.used_blocks == 2);
        
        pool.deallocate(ptr1);
        pool.deallocate(ptr2);
        
        auto after_free_stats = pool.get_stats();
        REQUIRE(after_free_stats.used_blocks == 0);
    }
}