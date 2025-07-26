/**
 * @file HashTable512.h
 * @brief 512-bit哈希表 - 真正突破125-bit限制
 * 
 * 替代原始的128-bit哈希表，支持真正的大范围搜索
 * 距离字段从125-bit扩展到509-bit (512-bit - 3个标志位)
 */

#ifndef HASHTABLE512_H
#define HASHTABLE512_H

#include <string>
#include <vector>
#include "SECPK1/Point.h"
#include "HashTable.h"  // 包含int128_t定义

#ifdef WIN64
#include <Windows.h>
#endif

// 512-bit哈希表配置
#define HASH512_SIZE_BIT 20  // 增大哈希表 (1M entries vs 256K)
#define HASH512_SIZE (1<<HASH512_SIZE_BIT)
#define HASH512_MASK (HASH512_SIZE-1)

#define ADD512_OK        0
#define ADD512_DUPLICATE 1
#define ADD512_COLLISION 2

#define safe_free512(x) if(x) {free(x);x=NULL;}

/**
 * @brief 512-bit整数联合体 - 支持多种访问方式
 */
union int512_s {
    uint8_t  i8[64];   // 64个字节
    uint16_t i16[32];  // 32个16位
    uint32_t i32[16];  // 16个32位
    uint64_t i64[8];   // 8个64位
};

typedef union int512_s int512_t;

/**
 * @brief 512-bit哈希表条目 - 突破125-bit限制
 * 
 * 距离字段编码：
 * b511=sign b510=kangaroo type, b509..b0 distance (509位距离！)
 */
typedef struct {
    int512_t  x;    // 袋鼠位置 (512-bit)
    int512_t  d;    // 行走距离 (512-bit: 509位距离 + 3位标志)
} ENTRY512;

/**
 * @brief 哈希桶结构 - 支持动态扩展
 */
typedef struct {
    uint32_t   nbItem;     // 当前条目数
    uint32_t   maxItem;    // 最大条目数
    ENTRY512 **items;     // 条目指针数组
} HASH_ENTRY512;

/**
 * @brief 512-bit哈希表类 - 真正突破125-bit限制
 */
class HashTable512 {

public:
    /**
     * @brief 构造函数
     */
    HashTable512();
    
    /**
     * @brief 析构函数
     */
    ~HashTable512();

    /**
     * @brief 添加条目到哈希表
     * @param h 哈希值
     * @param x 袋鼠位置 (512-bit)
     * @param d 行走距离 (512-bit)
     * @return ADD512_OK, ADD512_DUPLICATE, 或 ADD512_COLLISION
     */
    int Add(uint64_t h, int512_t *x, int512_t *d);
    int Add(uint64_t h, ENTRY512* e);

    /**
     * @brief 计算距离和类型 - 支持509-bit距离
     * @param d 512-bit距离字段
     * @param kDist 输出：真实距离 (最大509位)
     * @param kType 输出：袋鼠类型 (0=驯服, 1=野生)
     */
    void CalcDistAndType512(int512_t d, Int* kDist, uint32_t* kType);

    /**
     * @brief 查找碰撞
     * @param h 哈希值
     * @param x 袋鼠位置
     * @param d 行走距离
     * @param kType 袋鼠类型
     * @return 找到的条目，或nullptr
     */
    ENTRY512* FindCollision(uint64_t h, int512_t *x, int512_t *d, uint32_t kType);

    /**
     * @brief 获取统计信息
     */
    void GetStats(uint64_t* totalItems, uint64_t* totalMemory, double* loadFactor);

    /**
     * @brief 重置哈希表
     */
    void Reset();

    /**
     * @brief 保存到文件
     */
    bool SaveToFile(const std::string& filename);

    /**
     * @brief 从文件加载
     */
    bool LoadFromFile(const std::string& filename);

    /**
     * @brief 计算512-bit哈希值
     */
    static uint64_t Hash512(int512_t *x);

    /**
     * @brief 比较512-bit整数
     */
    static bool IsEqual512(int512_t *a, int512_t *b);

    /**
     * @brief 复制512-bit整数
     */
    static void Copy512(int512_t *dest, int512_t *src);

    /**
     * @brief 创建512-bit条目
     */
    ENTRY512* CreateEntry512(int512_t *x, int512_t *d);

    /**
     * @brief 验证125-bit限制突破
     * @return true 如果支持超过125-bit的距离
     */
    bool VerifyLimitBreakthrough();

    /**
     * @brief 测试大距离值
     * @param bitLength 测试的位长度 (126-509)
     * @return true 如果成功处理大距离值
     */
    bool TestLargeDistance(int bitLength);

private:
    HASH_ENTRY512 *E;      // 哈希表数组
    uint64_t totalItems;   // 总条目数
    uint64_t totalMemory;  // 总内存使用
    uint64_t collisionCount; // 碰撞计数

    /**
     * @brief 重新分配哈希桶
     */
    void ReAllocate512(uint64_t h, uint32_t add);

    /**
     * @brief 释放条目
     */
    void FreeEntry512(ENTRY512* e);

    /**
     * @brief 计算负载因子
     */
    double CalculateLoadFactor();
};

/**
 * @brief 512-bit到256-bit的兼容转换
 */
class HashTableAdapter {
public:
    /**
     * @brief 将512-bit条目转换为256-bit格式 (用于兼容性)
     */
    static void Convert512To256(ENTRY512* src, int128_t* x_out, int128_t* d_out);

    /**
     * @brief 将256-bit条目转换为512-bit格式
     */
    static void Convert256To512(int128_t* x_in, int128_t* d_in, ENTRY512* dest);

    /**
     * @brief 检查是否需要512-bit支持
     */
    static bool Requires512Bit(Int* rangeWidth);
};

#endif // HASHTABLE512_H
