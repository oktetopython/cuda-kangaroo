/**
 * @file HashTable512.h
 * @brief 512-bit hash table - True breakthrough of 125-bit limit
 *
 * Replaces original 128-bit hash table, supports true large-range search
 * Distance field extended from 125-bit to 509-bit (512-bit - 3 flag bits)
 */

#ifndef HASHTABLE512_H
#define HASHTABLE512_H

#include <string>
#include <vector>
#include "SECPK1/Point.h"
#include "HashTable.h"  // For int128_t definition

#ifdef WIN64
#include <Windows.h>
#endif

// 512-bit hash table configuration
#define HASH512_SIZE_BIT 20  // Enlarged hash table (1M entries vs 256K)
#define HASH512_SIZE (1<<HASH512_SIZE_BIT)
#define HASH512_MASK (HASH512_SIZE-1)

#define ADD512_OK        0
#define ADD512_DUPLICATE 1
#define ADD512_COLLISION 2

#define safe_free512(x) if(x) {free(x);x=NULL;}

/**
 * @brief 512-bit integer union - Supports multiple access methods
 */
union int512_s {
    uint8_t  i8[64];   // 64 bytes
    uint16_t i16[32];  // 32 16-bit words
    uint32_t i32[16];  // 16 32-bit words
    uint64_t i64[8];   // 8 64-bit words
};

typedef union int512_s int512_t;

/**
 * @brief 512-bit hash table entry - Breakthrough of 125-bit limit
 *
 * Distance field encoding:
 * b511=sign b510=kangaroo type, b509..b0 distance (509-bit distance!)
 */
typedef struct {
    int512_t  x;    // Kangaroo position (512-bit)
    int512_t  d;    // Travel distance (512-bit: 509-bit distance + 3 flag bits)
} ENTRY512;

/**
 * @brief Hash bucket structure - Supports dynamic expansion
 */
typedef struct {
    uint32_t   nbItem;     // Current number of entries
    uint32_t   maxItem;    // Maximum number of entries
    ENTRY512 **items;     // Entry pointer array
} HASH_ENTRY512;

/**
 * @brief 512-bit hash table class - True breakthrough of 125-bit limit
 */
class HashTable512 {

public:
    /**
     * @brief Constructor
     */
    HashTable512();

    /**
     * @brief Destructor
     */
    ~HashTable512();

    /**
     * @brief Add entry to hash table
     * @param h Hash value
     * @param x Kangaroo position (512-bit)
     * @param d Travel distance (512-bit)
     * @return ADD512_OK, ADD512_DUPLICATE, or ADD512_COLLISION
     */
    int Add(uint64_t h, int512_t *x, int512_t *d);
    int Add(uint64_t h, ENTRY512* e);

    /**
     * @brief Calculate distance and type - Supports 509-bit distance
     * @param d 512-bit distance field
     * @param kDist Output: Real distance (max 509 bits)
     * @param kType Output: Kangaroo type (0=tame, 1=wild)
     */
    void CalcDistAndType512(int512_t d, Int* kDist, uint32_t* kType);

    /**
     * @brief Find collision
     * @param h Hash value
     * @param x Kangaroo position
     * @param d Travel distance
     * @param kType Kangaroo type
     * @return Found entry, or nullptr
     */
    ENTRY512* FindCollision(uint64_t h, int512_t *x, int512_t *d, uint32_t kType);

    /**
     * @brief Get statistics
     */
    void GetStats(uint64_t* totalItems, uint64_t* totalMemory, double* loadFactor);

    /**
     * @brief Reset hash table
     */
    void Reset();

    /**
     * @brief Save to file
     */
    bool SaveToFile(const std::string& filename);

    /**
     * @brief Load from file
     */
    bool LoadFromFile(const std::string& filename);

    /**
     * @brief Calculate 512-bit hash value
     */
    static uint64_t Hash512(int512_t *x);

    /**
     * @brief Compare 512-bit integers
     */
    static bool IsEqual512(int512_t *a, int512_t *b);

    /**
     * @brief Copy 512-bit integer
     */
    static void Copy512(int512_t *dest, int512_t *src);

    /**
     * @brief Create 512-bit entry
     */
    ENTRY512* CreateEntry512(int512_t *x, int512_t *d);

    /**
     * @brief Verify 125-bit limit breakthrough
     * @return true if supports distance beyond 125-bit
     */
    bool VerifyLimitBreakthrough();

    /**
     * @brief Test large distance values
     * @param bitLength Test bit length (126-509)
     * @return true if successfully handles large distance values
     */
    bool TestLargeDistance(int bitLength);

private:
    HASH_ENTRY512 *E;      // hash table[Chinese comment removed]
    uint64_t totalItems;   // [Chinese comment removed]
    uint64_t totalMemory;  // [Chinese comment removed]memory使用
    uint64_t collisionCount; // collisioncount

    /**
     * @brief Reallocate hash bucket
     */
    void ReAllocate512(uint64_t h, uint32_t add);

    /**
     * @brief Free entry
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
