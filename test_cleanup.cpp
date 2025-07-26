// ============================================================================
// 🧪 Kangaroo 代码清理验证测试
// ============================================================================

#include "KangarooCommon.h"
#include <iostream>
#include <cassert>

// 测试安全内存分配
void test_safe_allocation() {
    std::cout << "Testing safe allocation..." << std::endl;
    
    // 测试正常分配
    int* ptr = KangarooUtils::safe_alloc<int>(100, "test allocation");
    assert(ptr != nullptr);
    delete[] ptr;
    
    // 测试大内存分配（可能失败）
    int* large_ptr = KangarooUtils::safe_alloc<int>(SIZE_MAX / sizeof(int), "large allocation");
    // 这个可能返回nullptr，这是正常的
    if (large_ptr) {
        delete[] large_ptr;
    }
    
    std::cout << "✅ Safe allocation test passed" << std::endl;
}

// 测试安全内存复制
void test_safe_memcpy() {
    std::cout << "Testing safe memcpy..." << std::endl;
    
    char src[100] = "Hello, World!";
    char dest[100];
    
    // 正常复制
    bool result = KangarooUtils::safe_memcpy(dest, sizeof(dest), src, strlen(src) + 1, "test memcpy");
    assert(result == true);
    assert(strcmp(dest, src) == 0);
    
    // 测试缓冲区溢出检测
    char small_dest[5];
    result = KangarooUtils::safe_memcpy(small_dest, sizeof(small_dest), src, strlen(src) + 1, "overflow test");
    assert(result == false); // 应该检测到溢出
    
    std::cout << "✅ Safe memcpy test passed" << std::endl;
}

// 测试文件操作安全性
void test_file_operations() {
    std::cout << "Testing file operations..." << std::endl;
    
    // 创建测试文件
    const char* test_file = "test_cleanup_temp.dat";
    FILE* f = fopen(test_file, "wb");
    assert(f != nullptr);
    
    // 写入测试数据
    const char test_data[] = "Test data for cleanup verification";
    fwrite(test_data, 1, sizeof(test_data), f);
    fclose(f);
    
    // 测试安全读取
    f = fopen(test_file, "rb");
    assert(f != nullptr);
    
    // 验证文件大小
    bool size_ok = KangarooUtils::validate_file_size(f, sizeof(test_data), "test file");
    assert(size_ok == true);
    
    // 安全读取
    char read_buffer[100];
    bool read_ok = KangarooUtils::safe_fread(read_buffer, 1, sizeof(test_data), f, "test read");
    assert(read_ok == true);
    assert(memcmp(read_buffer, test_data, sizeof(test_data)) == 0);
    
    fclose(f);
    remove(test_file);
    
    std::cout << "✅ File operations test passed" << std::endl;
}

// 测试错误计数器
void test_error_counter() {
    std::cout << "Testing error counter..." << std::endl;
    
    KangarooUtils::ErrorCounter::reset_counters();
    assert(KangarooUtils::ErrorCounter::get_error_count() == 0);
    assert(KangarooUtils::ErrorCounter::get_warning_count() == 0);
    
    KangarooUtils::ErrorCounter::increment_error();
    KangarooUtils::ErrorCounter::increment_warning();
    KangarooUtils::ErrorCounter::increment_warning();
    
    assert(KangarooUtils::ErrorCounter::get_error_count() == 1);
    assert(KangarooUtils::ErrorCounter::get_warning_count() == 2);
    
    std::cout << "✅ Error counter test passed" << std::endl;
}

// 测试常量定义
void test_constants() {
    std::cout << "Testing unified constants..." << std::endl;
    
    // 验证袋鼠类型常量
    assert(TAME == 0);
    assert(WILD == 1);
    
    // 验证GPU常量
    assert(GPU_GRP_SIZE == 128);
    assert(NB_RUN == 64);
    assert(NB_JUMP == 32);
    
    // 验证哈希表常量
    assert(ADD_OK == 0);
    assert(ADD_DUPLICATE == 1);
    assert(ADD_COLLISION == 2);
    
    std::cout << "✅ Constants test passed" << std::endl;
}

int main() {
    std::cout << "🧪 Starting Kangaroo cleanup verification tests..." << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        test_safe_allocation();
        test_safe_memcpy();
        test_file_operations();
        test_error_counter();
        test_constants();
        
        std::cout << "=================================================" << std::endl;
        std::cout << "🎉 All cleanup verification tests passed!" << std::endl;
        std::cout << "✅ Code cleanup appears to be successful" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
