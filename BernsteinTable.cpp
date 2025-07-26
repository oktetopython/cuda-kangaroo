#include "BernsteinTable.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>
#include <atomic>
#include "Timer.h"
#include "hash/sha256.h"

// 全局步长的静态成员定义
std::vector<Int> BernsteinTable::global_step_scalars;
std::vector<Point> BernsteinTable::global_step_points;
bool BernsteinTable::global_steps_initialized = false;

BernsteinTable::BernsteinTable(Secp256K1* secp_ctx)
    : secp(secp_ctx), t_bits(0), w_bits(0), l_bits(0), table_size(0) {
}

BernsteinTable::~BernsteinTable() {
}

void BernsteinTable::IntToUint32Array(const Int& value, uint32_t array[8]) const {
    // 将Int转换为uint32_t数组 (小端序) - 更清晰的实现
    for(int i = 0; i < 4; i++) {
        array[i*2] = (uint32_t)(value.bits64[i] & 0xFFFFFFFFULL);        // 低32位
        array[i*2+1] = (uint32_t)(value.bits64[i] >> 32);                // 高32位
    }
}

void BernsteinTable::Uint32ArrayToInt(const uint32_t array[8], Int& value) const {
    // 将uint32_t数组转换为Int (小端序)
    for(int i = 0; i < 4; i++) {
        value.bits64[i] = ((uint64_t)array[i*2+1] << 32) | array[i*2];
    }
}

uint32_t BernsteinTable::ComputePointHash(const Point& point) const {
    // 计算点的压缩表示的SHA256哈希
    unsigned char compressed[33];
    
    // 生成压缩点格式
    Int px = point.x;
    Int py = point.y;
    px.Get32Bytes(compressed + 1);
    compressed[0] = py.IsEven() ? 0x02 : 0x03;
    
    // 计算SHA256哈希
    unsigned char hash[32];
    sha256(compressed, 33, hash);
    
    // 返回前32位作为哈希值 (使用小端序确保跨平台一致性)
    uint32_t result = ((uint32_t)hash[0]) |
                     ((uint32_t)hash[1] << 8) |
                     ((uint32_t)hash[2] << 16) |
                     ((uint32_t)hash[3] << 24);
    return result;
}

uint32_t BernsteinTable::ComputeDistinguishedHash(const Point& point) const {
    return ComputePointHash(point);
}

bool BernsteinTable::IsDistinguishedPoint(const Point& point) const {
    if(w_bits <= 0 || w_bits > 32) return false;  // 防止无效的w_bits值

    uint32_t hash = ComputePointHash(point);

    // 检查低w_bits位是否为0 (更常见的DP定义)
    // 防止w_bits=32时的溢出问题
    uint32_t mask = (w_bits == 32) ? 0xFFFFFFFFU : ((1U << w_bits) - 1);
    bool is_dp = (hash & mask) == 0;

    // 详细调试输出（前几个检查或所有DP）- 使用线程安全的方式
    static std::atomic<int> dp_check_count(0);
    int current_count = dp_check_count.fetch_add(1);
    if(current_count < 20) {  // 增加调试输出，便于观察DP密度
        Point point_copy = point;
        std::string point_str = point_copy.x.GetBase16().substr(0,16);
        // 更明确地显示检查的位数
        printf("DP检查[%d]: 点x=%s, 哈希=%08x, 检查低%d位(掩码=%08x), 值=%08x, 是DP=%s\n",
               current_count, point_str.c_str(), hash, w_bits, mask, hash & mask, is_dp ? "是" : "否");
    }

    return is_dp;
}

bool BernsteinTable::PerformRandomWalk(const Int& start_scalar, Point& result_point, Int& result_log, const Int* offset, int max_steps) const {
    // 确保全局步长已初始化
    if(!global_steps_initialized) {
        printf("错误：全局步长未初始化！\n");
        return false;
    }

    Int start_copy = start_scalar;
    Point current = secp->ComputePublicKey(&start_copy);
    Int current_log;

    // 始终从0开始计数游走步数，不包含偏移量
    // 偏移量只用于确定起始点，不影响步数计算
    current_log.SetInt32(0);  // 游走步数从0开始

    // 详细调试：打印起始信息 - 使用线程安全的方式
    static std::atomic<int> walk_count(0);
    int current_walk = walk_count.fetch_add(1);
    bool is_debug_walk = (current_walk < 3);  // 只调试前3次游走

    if(is_debug_walk) {
        std::string start_str = current.x.GetBase16().substr(0,16);
        std::string log_str = current_log.GetBase16().substr(0,16);
        printf("\n=== 游走[%d]开始 ===\n", current_walk);
        printf("起始标量: %s\n", log_str.c_str());
        printf("起始点x: %s\n", start_str.c_str());
    }

    for(int step = 0; step < max_steps; step++) {
        // 检查是否为distinguished point
        if(IsDistinguishedPoint(current)) {
            if(is_debug_walk) {
                printf("游走[%d]在第%d步找到DP!\n", current_walk, step);
            }
            result_point = current;
            // 修正的存储逻辑：
            // 预计算阶段：我们从 g^(range_end - offset) 开始，走了 current_log 步到达DP
            // 从右端点到DP的总步数是：offset + current_log
            // 查找阶段：我们从目标点开始，走了 current_log 步到达DP
            if(offset != nullptr) {
                // 最终修正: 标准Bernstein算法存储逻辑
                // 存储: y + d，其中y是起始点的离散对数，d是游走步数
                Int offset_copy = *offset;

                // 计算起始点的离散对数: y = range_end - offset
                Int interval_end;
                interval_end.SetInt32(0);
                interval_end.bits64[0] = PUZZLE24_INTERVAL_END;
                Int y = interval_end;
                y.ModSubK1order(&offset_copy);  // y = range_end - offset

                // 存储: y + d
                result_log = y;
                result_log.ModAddK1order(&current_log);  // y + d

                if(is_debug_walk) {
                    printf("  偏移量: 0x%016llx\n", offset_copy.bits64[0]);
                    printf("  起始点离散对数 y: 0x%016llx\n", y.bits64[0]);
                    printf("  游走步数 d: 0x%016llx\n", current_log.bits64[0]);
                    printf("  存储值 (y + d): 0x%016llx\n", result_log.bits64[0]);
                }
            } else {
                // 查找阶段：直接使用current_log
                result_log = current_log;
                if(is_debug_walk) {
                    printf("  查找阶段步数: 0x%016llx\n", result_log.bits64[0]);
                }
            }
            return true;
        }

        // 使用全局一致的步长选择
        int step_index = SelectStepIndex(current, step);

        // 应用预计算的步长
        current = secp->AddDirect(current, global_step_points[step_index]);

        // 更新累积步长
        current_log.ModAddK1order(&global_step_scalars[step_index]);

        // 每10000步检查一次进度
        if(step % 10000 == 0 && step > 0) {
            printf("随机游走进度: %d步\n", step);
        }
    }

    if(is_debug_walk) {
        printf("游走[%d]达到最大步数%d，未找到DP\n", current_walk, max_steps);
    }
    return false;  // 未找到DP
}

void BernsteinTable::InitializeGlobalSteps(int l_bits, int w_bits_param) {
    if(global_steps_initialized) return;

    printf("初始化全局r-adding步长...\n");

    global_step_scalars.resize(R_STEPS);
    global_step_points.resize(R_STEPS);

    // 修正：对于区间DLP，步长应该相对于区间长度
    uint64_t interval_length = PUZZLE24_INTERVAL_LENGTH;
    uint64_t w = 1ULL << w_bits_param;
    // 使用更合理的步长，平衡速度和准确性
    uint64_t avg_step = interval_length / (w * 4);  // 适度增加步长以加快预计算

    printf("区间长度=0x%llx, W=2^%d (%llu), 平均步长=2^%.1f (%llu)\n",
           interval_length, w_bits_param, w, log2((double)avg_step), avg_step);

    // 生成R_STEPS个步长，指数在[avg_step/4, avg_step*4]范围内
    std::mt19937_64 rng(12345);  // 固定种子确保一致性

    for(int i = 0; i < R_STEPS; i++) {
        // 生成指数k_i在合理范围内
        uint64_t min_exp = avg_step / 4;
        uint64_t max_exp = avg_step * 4;
        // 确保max_exp不小于min_exp，且不溢出
        if(max_exp <= min_exp) max_exp = min_exp + 1;
        uint64_t range = max_exp - min_exp + 1;
        uint64_t k_i = min_exp + (rng() % range);

        // 创建步长标量 - 使用计算出的合理k_i值
        if(k_i <= UINT32_MAX) {
            global_step_scalars[i].SetInt32((uint32_t)k_i);  // 使用计算出的合理步长
        } else {
            // 对于大于32位的值，使用SetBase16
            char hex_str[32];
            sprintf(hex_str, "%llx", k_i);
            global_step_scalars[i].SetBase16(hex_str);
        }

        // 计算对应的点 s_i = g^{k_i}
        global_step_points[i] = secp->ComputePublicKey(&global_step_scalars[i]);

        if(i < 5) {  // 打印前5个步长用于调试
            // 计算k_i的bit长度
            int k_i_bits = 0;
            uint64_t temp_k = k_i;
            while(temp_k > 0) { temp_k >>= 1; k_i_bits++; }

            std::string point_str = global_step_points[i].x.GetBase16().substr(0,16);
            printf("步长[%d]: k_i=%llu (约2^%.1f), 点x=%s\n",
                   i, k_i, (k_i > 0) ? log2((double)k_i) : 0.0, point_str.c_str());
        }
    }

    global_steps_initialized = true;
    printf("全局步长初始化完成: %d个步长\n", R_STEPS);
}

int BernsteinTable::SelectStepIndex(const Point& current_point, int step_counter) const {
    // 使用点的x坐标哈希来选择步长（确保确定性和一致性）
    uint32_t hash = ComputePointHash(current_point);
    return hash % R_STEPS;
}

bool BernsteinTable::GenerateTable(int t_bits_param, int w_bits_param, int l_bits_param, const std::string& filename) {
    t_bits = t_bits_param;
    w_bits = w_bits_param;  // 确保w_bits成员变量被正确设置
    l_bits = l_bits_param;  // 保存l_bits参数
    table_size = 1ULL << t_bits;

    printf("=== 生成Bernstein预计算表 ===\n");
    printf("参数: T=2^%d, W=2^%d\n", t_bits, w_bits);  // 确认打印的是正确的w_bits
    printf("目标表大小: %llu 条目\n", table_size);

    // 初始化全局步长（关键：确保一致性）
    InitializeGlobalSteps(l_bits_param, w_bits_param);  // 使用正确的l_bits和w_bits参数
    
    table_entries.clear();
    table_entries.reserve(table_size);
    
    double t0 = Timer::get_tick();
    uint64_t walks_completed = 0;
    uint64_t points_found = 0;
    uint64_t generation_target = table_size;
    
    printf("开始生成distinguished points...\n");
    
    // 使用固定的区间右端点作为起始点（Bernstein算法标准做法）
    Int range_end;
    range_end.SetBase16("ffffff");  // 对于范围[0x800000, 0xffffff]的右端点
    printf("使用固定的区间右端点作为起始点: %016llx\n", range_end.bits64[0]);

    while(points_found < generation_target && walks_completed < generation_target * 20) {

        // 生成小的随机偏移量，确保多样化的起始点
        // 偏移量应该远小于区间长度 (0xffffff - 0x800000 = 0x7fffff)
        Int offset;
        offset.SetInt32(rand() % 65536);  // 生成0到65535的小偏移量
        if(walks_completed < 5) {  // 只显示前几次的调试信息
            printf("游走[%d]: 随机偏移=%016llx\n", walks_completed, offset.bits64[0]);
        }

        // 计算实际起始点: actual_start = range_end - offset
        Int actual_start = range_end;
        actual_start.ModSubK1order(&offset);

        if(walks_completed < 5) {  // 调试信息
            printf("  实际起始点: %016llx\n", actual_start.bits64[0]);
        }

        // 执行随机游走
        Point dp;
        Int dp_log;

        if(PerformRandomWalk(actual_start, dp, dp_log, &offset)) {
            // 找到DP，添加到表中
            PrecomputedTableEntry entry;
            entry.hash = ComputeDistinguishedHash(dp);
            IntToUint32Array(dp_log, entry.log_value);
            entry.usefulness_metric = 1;
            
            table_entries.push_back(entry);
            points_found++;
        }
        
        walks_completed++;
        
        // 进度报告
        if(walks_completed % 10000 == 0) {
            double elapsed = Timer::get_tick() - t0;
            printf("进度: %llu 次游走, %llu 个DP (%.1fs)\n", 
                   walks_completed, points_found, elapsed);
        }
    }
    
    double total_time = Timer::get_tick() - t0;
    printf("预计算表生成完成: %llu 条目, 用时 %.1f 秒\n", points_found, total_time);
    
    // 保存到文件
    if(SaveTable(filename)) {
        printf("表已保存到: %s\n", filename.c_str());
        return true;
    } else {
        printf("保存表失败!\n");
        return false;
    }
}

bool BernsteinTable::SaveTable(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if(!ofs) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }
    
    // 写入表头信息
    uint64_t actual_size = table_entries.size();
    ofs.write(reinterpret_cast<const char*>(&actual_size), sizeof(actual_size));
    ofs.write(reinterpret_cast<const char*>(&t_bits), sizeof(t_bits));
    ofs.write(reinterpret_cast<const char*>(&w_bits), sizeof(w_bits));
    ofs.write(reinterpret_cast<const char*>(&l_bits), sizeof(l_bits));
    
    // 写入表项 - 添加错误检查
    for(const auto& entry : table_entries) {
        ofs.write(reinterpret_cast<const char*>(&entry), sizeof(PrecomputedTableEntry));
        if(!ofs.good()) {
            std::cerr << "Error writing table entry to " << filename << std::endl;
            ofs.close();
            return false;
        }
    }
    
    ofs.close();
    if(!ofs.good()) {
        std::cerr << "Error occurred while writing to file " << filename << std::endl;
        return false;
    }
    
    return true;
}

bool BernsteinTable::LoadTable(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if(!ifs) {
        std::cerr << "Error: Could not open file " << filename << " for reading." << std::endl;
        return false;
    }
    
    // 读取表头信息
    uint64_t actual_size = 0;
    ifs.read(reinterpret_cast<char*>(&actual_size), sizeof(actual_size));
    if(!ifs.good()) {
        std::cerr << "Error reading actual_size from " << filename << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&t_bits), sizeof(t_bits));
    if(!ifs.good()) {
        std::cerr << "Error reading t_bits from " << filename << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&w_bits), sizeof(w_bits));
    if(!ifs.good()) {
        std::cerr << "Error reading w_bits from " << filename << std::endl;
        return false;
    }

    ifs.read(reinterpret_cast<char*>(&l_bits), sizeof(l_bits));
    if(!ifs.good()) {
        std::cerr << "Error reading l_bits from " << filename << std::endl;
        return false;
    }

    table_size = actual_size;
    
    // 读取表项
    table_entries.resize(table_size);
    ifs.read(reinterpret_cast<char*>(table_entries.data()), table_size * sizeof(PrecomputedTableEntry));
    
    if(ifs.gcount() != static_cast<std::streamsize>(table_size * sizeof(PrecomputedTableEntry))) {
        std::cerr << "Error reading table data from " << filename << std::endl;
        return false;
    }
    
    ifs.close();

    printf("预计算表加载成功: %llu 条目\n", table_size);
    printf("表参数: T=2^%d, W=2^%d, L=2^%d\n", t_bits, w_bits, l_bits);

    // 关键修改：在加载表后，根据表的参数初始化全局步长
    // 这确保了查找阶段使用的步长与生成阶段完全一致
    if(!global_steps_initialized) {
        printf("加载表后初始化全局步长 (L=2^%d, W=2^%d)...\n", l_bits, w_bits);
        InitializeGlobalSteps(l_bits, w_bits);  // 使用从文件加载的正确参数
    } else {
        // 可选：检查已初始化的步长参数是否与表的参数匹配
        printf("警告：全局步长已在加载前初始化，参数可能不匹配。\n");
    }

    // 构建查找映射
    BuildLookupMap();

    return true;
}

void BernsteinTable::BuildLookupMap() {
    lookup_map.clear();
    
    for(const auto& entry : table_entries) {
        StoredLog log_val;
        memcpy(log_val.data, entry.log_value, sizeof(log_val.data));
        log_val.usefulness = entry.usefulness_metric;
        lookup_map[entry.hash] = log_val;
    }
    
    printf("查找映射构建完成: %llu 条目\n", lookup_map.size());
    printf("验证: table_entries.size()=%llu, lookup_map.size()=%llu\n", table_entries.size(), lookup_map.size());

    if(lookup_map.size() != table_entries.size()) {
        printf("警告: 查找映射大小与表条目数不匹配！可能存在哈希冲突。\n");
    }
}

bool BernsteinTable::LookupPoint(const Point& point, Int& result_log) const {
    // 确保全局步长已初始化
    if(!global_steps_initialized) {
        printf("错误：查找时全局步长未初始化！\n");
        return false;
    }

    printf("\n=== 开始完整的Bernstein查找算法 ===\n");
    Point point_copy = point;  // 创建非const副本
    std::string target_x = point_copy.x.GetBase16().substr(0,16);
    printf("目标点x: %s\n", target_x.c_str());

    // 添加小的随机扰动z (按照标准Bernstein算法)
    // 使用现代C++随机数生成器，确保真正的随机性
    uint64_t max_z = PUZZLE24_INTERVAL_LENGTH / 256;  // z的范围约为区间长度/256
    if (max_z == 0) max_z = 1000;  // 防止范围为空，最小1000

    // 使用高精度时钟作为种子，确保每次运行都不同
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<uint64_t> dis(1, max_z);  // 从1开始，避免z=0
    uint64_t z_val = dis(gen);

    Int z;
    z.SetInt32(static_cast<uint32_t>(z_val));
    printf("随机扰动 z: 0x%016llx (范围: [1, 0x%llx])\n", z.bits64[0], max_z);

    // 计算起始点: h * g^z
    Point g_z = secp->ComputePublicKey(&z);
    Point current = secp->AddDirect(point_copy, g_z);  // 使用已存在的point_copy

    printf("从 h * g^z 开始的野兔游走\n");

    const int max_lookup_steps = 5000000;  // 增加最大步数  // 最大查找步数
    int false_alarms = 0;  // 错误警报计数器

    for(int step = 0; step < max_lookup_steps; step++) {
        // 检查当前点是否为DP
        if(IsDistinguishedPoint(current)) {
            uint32_t hash = ComputeDistinguishedHash(current);
            printf("🔍 查找阶段找到DP (步 %d): 哈希=%08x\n", step, hash);

            // 在预计算表中查找这个DP
            auto it = lookup_map.find(hash);
            if(it != lookup_map.end()) {
                const StoredLog& stored_log = it->second;
                Int table_log;
                Uint32ArrayToInt(stored_log.data, table_log);

                printf("🎯 找到碰撞! 查找步数: %d, 错误警报: %d\n", step, false_alarms);

                // 正确公式: k = b - (table_steps + search_steps) mod (b-a)
                Int interval_start, interval_end;
                interval_start.SetInt32(0);
                interval_start.bits64[0] = PUZZLE24_INTERVAL_START;
                interval_end.SetInt32(0);
                interval_end.bits64[0] = PUZZLE24_INTERVAL_END;

                printf("区间: [0x%016llx, 0x%016llx]\n",
                       interval_start.bits64[0], interval_end.bits64[0]);
                printf("表中数值: 0x%016llx\n", table_log.bits64[0]);
                printf("查找步数: %d, 随机扰动z: 0x%016llx\n", step, z.bits64[0]);

                // 使用预定义的区间长度常量
                Int interval_length;
                interval_length.SetInt32(0);
                interval_length.bits64[0] = PUZZLE24_INTERVAL_LENGTH;
                printf("区间长度: 0x%016llx\n", PUZZLE24_INTERVAL_LENGTH);

                // 核心计算：使用有符号整数正确处理负数模运算
                // 这是Bernstein算法的关键数学计算部分
                printf("=== Bernstein算法核心计算 ===\n");

                // 转换为有符号64位整数进行计算
                int64_t table_log_val = static_cast<int64_t>(table_log.bits64[0]);
                int64_t z_val = static_cast<int64_t>(z.bits64[0]);
                int64_t search_steps_val = static_cast<int64_t>(step);
                int64_t interval_length_val = static_cast<int64_t>(interval_length.bits64[0]);

                printf("table_log (有符号): %lld\n", table_log_val);
                printf("z + search_steps: %lld + %lld = %lld\n", z_val, search_steps_val, z_val + search_steps_val);

                // 计算 k_raw (可能为负)
                int64_t k_raw_signed = table_log_val - (z_val + search_steps_val);
                printf("k_raw (有符号): %lld\n", k_raw_signed);

                // 正确处理负数的模运算
                int64_t k_in_range;
                if (k_raw_signed >= 0) {
                    k_in_range = k_raw_signed % interval_length_val;
                } else {
                    // 处理负数情况，确保结果在 [0, interval_length-1] 范围内
                    k_in_range = k_raw_signed % interval_length_val;
                    if (k_in_range < 0) {
                        k_in_range += interval_length_val;
                    }
                }
                printf("k_in_range (正确模运算): %lld\n", k_in_range);

                // 平移到实际区间
                uint64_t final_k = static_cast<uint64_t>(k_in_range) + PUZZLE24_INTERVAL_START;
                printf("最终结果 k = k_in_range + interval_start: %lld + %llu = %llu\n",
                       k_in_range, PUZZLE24_INTERVAL_START, final_k);

                result_log.SetInt32(0);
                result_log.bits64[0] = final_k;

                // 检查结果是否在区间内
                uint64_t k_candidate = result_log.bits64[0];
                if (k_candidate >= PUZZLE24_INTERVAL_START && k_candidate <= PUZZLE24_INTERVAL_END) {
                    printf("✅ 计算出的私钥在目标区间内\n");
                } else {
                    printf("⚠️ 警告：计算出的私钥 0x%016llx 不在目标区间 [0x%08llx, 0x%08llx] 内\n",
                           k_candidate, PUZZLE24_INTERVAL_START, PUZZLE24_INTERVAL_END);
                }

                printf("单次运行结果: 0x%016llx\n", result_log.bits64[0]);

                // 验证期望私钥是否真的能生成目标公钥 (24号谜题)
                Int expected_key;
                expected_key.SetBase16("dc2a04");  // 24号谜题的正确私钥
                Point expected_point = secp->ComputePublicKey(&expected_key);
                Point point_copy = point;  // 创建非const副本
                bool expected_correct = expected_point.x.IsEqual(&point_copy.x);

                printf("期望私钥验证 (24号谜题):\n");
                printf("  期望私钥: 0xdc2a04\n");
                printf("  期望私钥生成的公钥x: %s\n", expected_point.x.GetBase16().substr(0,16).c_str());
                printf("  目标公钥x: %s\n", point_copy.x.GetBase16().substr(0,16).c_str());
                printf("  期望私钥是否正确: %s\n", expected_correct ? "是" : "否");

                uint64_t expected = 0xdc2a04;

                // 直接比较 bits64[0] 以避免 GetBase16() 可能的问题
                bool is_match = (result_log.bits64[0] == expected);
                printf("是否匹配 (bits64[0] == expected): %s\n", is_match ? "是" : "否");

                // 验证计算出的私钥是否能生成目标公钥
                Point computed_point = secp->ComputePublicKey(&result_log);
                bool key_correct = computed_point.x.IsEqual(&point_copy.x);
                printf("计算私钥生成的公钥验证: %s\n", key_correct ? "正确" : "错误");

                // 如果 GetBase16() 修复了，也可以用它来打印最终结果
                std::string result_str = result_log.GetBase16().substr(0,16);
                printf("计算得到私钥 (GetBase16): %s\n", result_str.c_str());

                if (is_match && key_correct) {  // 使用 bits64 比较和公钥验证作为成功判断
                    printf("🎉 成功找到私钥!\n");
                    return true;
                } else {
                    printf("❌ 私钥不匹配! 计算过程可能仍有误。\n");
                    printf("   bits64匹配: %s, 公钥验证: %s\n", is_match ? "是" : "否", key_correct ? "是" : "否");
                    return false;  // 继续查找
                }
            } else {
                false_alarms++;  // 增加错误警报计数
                printf("❌ DP哈希 %08x 不在表中（错误警报）\n", hash);
                std::string dp_x = current.x.GetBase16().substr(0,16);
                printf("    详细: 点x=%s, 步数=%d\n", dp_x.c_str(), step);
            }
        }

        // 继续随机游走（使用与生成表时完全相同的步长选择）
        int step_index = SelectStepIndex(current, step);

        // 调试步长选择（每50000步）
        if(step % 50000 == 0) {
            printf("查找步 %d: 使用步长索引 %d\n", step, step_index);
        }

        current = secp->AddDirect(current, global_step_points[step_index]);
        // 修正：不再累积标量值，步数就是step变量本身

        // 调试输出（前几步）
        if(step < 3) {
            std::string current_str = current.x.GetBase16().substr(0,16);
            std::string step_str = global_step_scalars[step_index].GetBase16().substr(0,16);
            printf("查找步骤[%d]: 步长索引[%d], 步长值=%s, 当前点x=%s\n",
                   step, step_index, step_str.c_str(), current_str.c_str());
        }

        // 进度报告
        if(step % 25000 == 0 && step > 0) {
            printf("查找进度: %d步 (目标: %d步)\n", step, max_lookup_steps);
        }
    }

    printf("查找失败：达到最大步数 %d\n", max_lookup_steps);
    return false;
}

bool BernsteinTable::VerifyTable(int max_entries) const {
    printf("=== 验证预计算表完整性 ===\n");
    
    int verified_count = 0;
    int error_count = 0;
    int entries_to_check = (max_entries < (int)table_entries.size()) ? max_entries : (int)table_entries.size();
    
    for(int i = 0; i < entries_to_check; i++) {
        const auto& entry = table_entries[i];
        
        // 重新计算对应的公钥
        Int stored_log;
        Uint32ArrayToInt(entry.log_value, stored_log);
        Point computed_point = secp->ComputePublicKey(&stored_log);
        
        // 验证哈希是否匹配
        uint32_t computed_hash = ComputeDistinguishedHash(computed_point);
        
        if(computed_hash == entry.hash) {
            verified_count++;
            if(i < 5) {  // 只显示前5个验证结果
                printf("✅ 条目 %d: 验证正确\n", i+1);
            }
        } else {
            error_count++;
            printf("❌ 条目 %d: 验证失败!\n", i+1);
        }
    }
    
    printf("\n=== 验证结果 ===\n");
    printf("验证条目数: %d\n", verified_count + error_count);
    printf("正确条目数: %d\n", verified_count);
    printf("错误条目数: %d\n", error_count);
    printf("正确率: %.2f%%\n", (double)verified_count / (verified_count + error_count) * 100);
    
    if(error_count == 0) {
        printf("🎉 表完整性验证通过!\n");
        return true;
    } else {
        printf("❌ 表完整性验证失败!\n");
        return false;
    }
}
