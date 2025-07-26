#ifndef BERNSTEIN_TABLE_H
#define BERNSTEIN_TABLE_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <array>
#include "SECPK1/SECP256k1.h"
#include "SECPK1/Point.h"
#include "SECPK1/Int.h"

// 预计算表条目结构
struct PrecomputedTableEntry {
    // 存储32位哈希 (根据论文lg(T/γ)确定大小)
    uint32_t hash;
    uint32_t padding; // 8字节对齐
    
    // 离散对数 (256位，用8个32位整数表示)
    uint32_t log_value[8];
    
    // 可选：有用性度量
    uint64_t usefulness_metric;
    
    PrecomputedTableEntry() : hash(0), padding(0), usefulness_metric(1) {
        memset(log_value, 0, sizeof(log_value));
    }
};

// 确保结构体大小一致
static_assert(sizeof(PrecomputedTableEntry) == 48, "PrecomputedTableEntry size mismatch");

// 存储的对数值结构
struct StoredLog {
    uint32_t data[8];
    uint64_t usefulness;
    
    StoredLog() : usefulness(1) {
        memset(data, 0, sizeof(data));
    }
};

// 常量定义 - 避免重复的魔法数字
class BernsteinTable {
private:
    // 24号谜题的区间常量
    static constexpr uint64_t PUZZLE24_INTERVAL_START = 0x800000;
    static constexpr uint64_t PUZZLE24_INTERVAL_END = 0xffffff;
    static constexpr uint64_t PUZZLE24_INTERVAL_LENGTH = PUZZLE24_INTERVAL_END - PUZZLE24_INTERVAL_START;
    std::vector<PrecomputedTableEntry> table_entries;
    std::unordered_map<uint32_t, StoredLog> lookup_map;
    int t_bits;
    int w_bits;
    int l_bits;  // 添加l_bits成员变量
    uint64_t table_size;
    Secp256K1* secp;
    
public:
    BernsteinTable(Secp256K1* secp_ctx);
    ~BernsteinTable();
    
    // 生成预计算表
    bool GenerateTable(int t_bits, int w_bits, int l_bits, const std::string& filename);
    
    // 加载预计算表
    bool LoadTable(const std::string& filename);
    
    // 构建查找映射
    void BuildLookupMap();
    
    // 保存表到文件
    bool SaveTable(const std::string& filename) const;
    
    // 计算点的distinguished hash
    uint32_t ComputeDistinguishedHash(const Point& point) const;
    
    // 检查是否为distinguished point
    bool IsDistinguishedPoint(const Point& point) const;
    
    // 在表中查找
    bool LookupPoint(const Point& point, Int& result_log) const;
    
    // 获取表信息
    uint64_t GetTableSize() const { return table_size; }
    int GetTBits() const { return t_bits; }
    int GetWBits() const { return w_bits; }
    
    // 验证表完整性
    bool VerifyTable(int max_entries = 100) const;
    
private:
    // 内部辅助函数
    void IntToUint32Array(const Int& value, uint32_t array[8]) const;
    void Uint32ArrayToInt(const uint32_t array[8], Int& value) const;
    uint32_t ComputePointHash(const Point& point) const;
    bool PerformRandomWalk(const Int& start_scalar, Point& result_point, Int& result_log, const Int* offset = nullptr, int max_steps = 100000) const;

    // r-adding walks的全局步长（确保生成和查找阶段一致）
    static std::vector<Int> global_step_scalars;
    static std::vector<Point> global_step_points;
    static bool global_steps_initialized;
    static const int R_STEPS = 128;

    // 初始化全局步长
    void InitializeGlobalSteps(int l_bits, int w_bits);

    // 选择步长的一致性函数
    int SelectStepIndex(const Point& current_point, int step_counter) const;
};

#endif // BERNSTEIN_TABLE_H
