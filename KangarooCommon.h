#ifndef KANGAROO_COMMON_H
#define KANGAROO_COMMON_H

// ============================================================================
// 🎯 Kangaroo 统一头文件 - 消除重复包含
// ============================================================================

// 标准库头文件
#include <fstream>
#include <string.h>
#include <algorithm>
#include <vector>
#include <string>

// 数学相关
#define _USE_MATH_DEFINES
#include <math.h>

// 平台特定头文件
#ifndef WIN64
#include <pthread.h>
#define _strdup strdup
#else
#include <Windows.h>
#include "WindowsErrors.h"
#endif

// Kangaroo核心头文件
#include "Kangaroo.h"
#include "SECPK1/IntGroup.h"
#include "Timer.h"
#include "Constants.h"

// 使用标准命名空间
using namespace std;

// ============================================================================
// 🔧 统一的宏定义
// ============================================================================

// 安全删除宏
#define safe_delete_array(x) if(x) {delete[] x;x=NULL;}
#define safe_free(x) if(x) {free(x);x=NULL;}

// 错误处理宏
#define CHECK_ALLOC(ptr, msg) \
    if(!(ptr)) { \
        ::printf("Memory allocation failed: %s\n", msg); \
        return false; \
    }

#define CHECK_FILE_OP(op, msg) \
    if(!(op)) { \
        ::printf("File operation failed: %s\n", msg); \
        return false; \
    }

// 线程安全宏 (已存在但在此统一)
#ifdef WIN64
#define LOCK(mutex) WaitForSingleObject(mutex,INFINITE);
#define UNLOCK(mutex) ReleaseMutex(mutex);
#else
#define LOCK(mutex)  pthread_mutex_lock(&(mutex));
#define UNLOCK(mutex) pthread_mutex_unlock(&(mutex));
#endif

// ============================================================================
// 🎯 统一的常量定义 (避免重复)
// ============================================================================

// Kangaroo类型 (统一定义)
#ifndef KANGAROO_TYPES_DEFINED
#define KANGAROO_TYPES_DEFINED
#define TAME 0  // 驯服袋鼠
#define WILD 1  // 野生袋鼠
#endif

// GPU相关常量 (统一定义)
#ifndef GPU_CONSTANTS_DEFINED
#define GPU_CONSTANTS_DEFINED
#define GPU_GRP_SIZE 128
#define NB_RUN 64
#define NB_JUMP 32
#endif

// 哈希表常量 (统一定义)
#ifndef HASH_CONSTANTS_DEFINED
#define HASH_CONSTANTS_DEFINED
#define ADD_OK        0
#define ADD_DUPLICATE 1
#define ADD_COLLISION 2
#endif

// ============================================================================
// 🛡️ 内存安全辅助函数声明
// ============================================================================

namespace KangarooUtils {
    // 安全的内存分配
    template<typename T>
    T* safe_alloc(size_t count, const char* context = "unknown") {
        T* ptr = new(std::nothrow) T[count];
        if (!ptr) {
            ::printf("[ERROR] Memory allocation failed in %s: %zu items\n", 
                    context, count);
        }
        return ptr;
    }
    
    // 安全的文件读取
    bool safe_fread(void* buffer, size_t size, size_t count, FILE* file, 
                   const char* context = "unknown");
    
    // 边界检查的内存复制
    bool safe_memcpy(void* dest, size_t dest_size, const void* src, 
                    size_t copy_size, const char* context = "unknown");
}

#endif // KANGAROO_COMMON_H
