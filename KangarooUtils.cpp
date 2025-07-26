// ============================================================================
// 🛡️ Kangaroo 安全工具函数实现
// ============================================================================

#include "KangarooCommon.h"
#include <errno.h>

namespace KangarooUtils {

/**
 * @brief 安全的文件读取函数
 * @param buffer 目标缓冲区
 * @param size 每个元素大小
 * @param count 元素数量
 * @param file 文件指针
 * @param context 上下文信息（用于错误报告）
 * @return 成功返回true，失败返回false
 */
bool safe_fread(void* buffer, size_t size, size_t count, FILE* file, 
               const char* context) {
    if (!buffer || !file) {
        ::printf("[ERROR] safe_fread: null pointer in %s\n", context);
        return false;
    }
    
    if (size == 0 || count == 0) {
        ::printf("[ERROR] safe_fread: invalid size/count in %s\n", context);
        return false;
    }
    
    // 检查文件状态
    if (feof(file)) {
        ::printf("[ERROR] safe_fread: EOF reached in %s\n", context);
        return false;
    }
    
    if (ferror(file)) {
        ::printf("[ERROR] safe_fread: file error in %s\n", context);
        return false;
    }
    
    size_t read_count = fread(buffer, size, count, file);
    if (read_count != count) {
        if (feof(file)) {
            ::printf("[ERROR] safe_fread: unexpected EOF in %s (read %zu/%zu)\n", 
                    context, read_count, count);
        } else if (ferror(file)) {
            ::printf("[ERROR] safe_fread: file error in %s: %s\n", 
                    context, strerror(errno));
        } else {
            ::printf("[ERROR] safe_fread: partial read in %s (read %zu/%zu)\n", 
                    context, read_count, count);
        }
        return false;
    }
    
    return true;
}

/**
 * @brief 边界检查的内存复制函数
 * @param dest 目标缓冲区
 * @param dest_size 目标缓冲区大小
 * @param src 源缓冲区
 * @param copy_size 要复制的字节数
 * @param context 上下文信息
 * @return 成功返回true，失败返回false
 */
bool safe_memcpy(void* dest, size_t dest_size, const void* src, 
                size_t copy_size, const char* context) {
    if (!dest || !src) {
        ::printf("[ERROR] safe_memcpy: null pointer in %s\n", context);
        return false;
    }
    
    if (copy_size == 0) {
        return true; // 复制0字节是合法的
    }
    
    if (copy_size > dest_size) {
        ::printf("[ERROR] safe_memcpy: buffer overflow in %s "
                "(copy_size=%zu > dest_size=%zu)\n", 
                context, copy_size, dest_size);
        return false;
    }
    
    memcpy(dest, src, copy_size);
    return true;
}

/**
 * @brief 安全的内存重分配函数
 * @param ptr 原指针
 * @param old_size 原大小
 * @param new_size 新大小
 * @param context 上下文信息
 * @return 成功返回新指针，失败返回nullptr
 */
void* safe_realloc(void* ptr, size_t old_size, size_t new_size, 
                  const char* context) {
    if (new_size == 0) {
        if (ptr) {
            free(ptr);
        }
        return nullptr;
    }
    
    void* new_ptr = realloc(ptr, new_size);
    if (!new_ptr) {
        ::printf("[ERROR] safe_realloc: allocation failed in %s "
                "(old_size=%zu, new_size=%zu)\n", 
                context, old_size, new_size);
        return nullptr;
    }
    
    // 如果扩大了内存，清零新分配的部分
    if (new_size > old_size && ptr) {
        memset((char*)new_ptr + old_size, 0, new_size - old_size);
    }
    
    return new_ptr;
}

/**
 * @brief 验证文件大小
 * @param file 文件指针
 * @param expected_min_size 期望的最小文件大小
 * @param context 上下文信息
 * @return 文件大小足够返回true，否则返回false
 */
bool validate_file_size(FILE* file, size_t expected_min_size, 
                       const char* context) {
    if (!file) {
        ::printf("[ERROR] validate_file_size: null file in %s\n", context);
        return false;
    }
    
    long current_pos = ftell(file);
    if (current_pos < 0) {
        ::printf("[ERROR] validate_file_size: ftell failed in %s\n", context);
        return false;
    }
    
    if (fseek(file, 0, SEEK_END) != 0) {
        ::printf("[ERROR] validate_file_size: fseek to end failed in %s\n", context);
        return false;
    }
    
    long file_size = ftell(file);
    if (file_size < 0) {
        ::printf("[ERROR] validate_file_size: ftell at end failed in %s\n", context);
        fseek(file, current_pos, SEEK_SET); // 恢复位置
        return false;
    }
    
    // 恢复原始位置
    if (fseek(file, current_pos, SEEK_SET) != 0) {
        ::printf("[ERROR] validate_file_size: fseek restore failed in %s\n", context);
        return false;
    }
    
    if ((size_t)file_size < expected_min_size) {
        ::printf("[ERROR] validate_file_size: file too small in %s "
                "(size=%ld, expected_min=%zu)\n", 
                context, file_size, expected_min_size);
        return false;
    }
    
    return true;
}

/**
 * @brief 线程安全的错误计数器
 */
class ErrorCounter {
private:
    static std::atomic<uint64_t> error_count_;
    static std::atomic<uint64_t> warning_count_;
    
public:
    static void increment_error() { error_count_++; }
    static void increment_warning() { warning_count_++; }
    static uint64_t get_error_count() { return error_count_.load(); }
    static uint64_t get_warning_count() { return warning_count_.load(); }
    static void reset_counters() { 
        error_count_ = 0; 
        warning_count_ = 0; 
    }
};

// 静态成员定义
std::atomic<uint64_t> ErrorCounter::error_count_{0};
std::atomic<uint64_t> ErrorCounter::warning_count_{0};

} // namespace KangarooUtils
